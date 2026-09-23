# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import datetime
import json
import logging
from typing import Any
import uuid

import google.auth
import yaml
from fastapi import Depends
from google.api_core.exceptions import NotFound, InvalidArgument
from google.auth.transport.requests import AuthorizedSession
from google.cloud import workflows_v1
from google.cloud.workflows import executions_v1
from pydantic import BaseModel, ValidationError

from src.common.dto.pagination_response_dto import PaginationResponseDto
from src.config.config_service import config_service
from src.images.imagen_service import ImagenService
from src.source_assets.source_asset_service import SourceAssetService
from src.users.user_model import UserModel
from src.workflows.dto.batch_execution_dto import (
    BatchExecutionRequestDto,
    BatchExecutionResponseDto,
    BatchItemResultDto,
)
from src.workflows.dto.workflow_search_dto import WorkflowSearchDto
from src.workflows.repository.workflow_repository import WorkflowRepository
from src.workflows.repository.workflow_run_repository import (
    WorkflowRunRepository,
)
from src.workflows.repository.workflow_template_repository import (
    WorkflowTemplateRepository,
)
from src.workflows.schema.workflow_model import (
    NodeTypes,
    StepOutputReference,
    WorkflowBase,
    WorkflowCreateDto,
    WorkflowModel,
)
from src.workflows.schema.workflow_run_model import (
    WorkflowRunModel,
    WorkflowRunStatusEnum,
)
from src.workflows.schema.workflow_template_model import (
    WorkflowTemplateCreateDto,
    WorkflowTemplateModel,
)
from src.workflows import workflow_constants
from src.workflows.workflow_constants import IMAGE_MODE_ALLOWED_INPUTS
from src.workflows.workflow_utils import (
    build_iteration_step_name,
    interpolate_prompt_variables,
    parse_iteration_step_name,
)

logger = logging.getLogger(__name__)
PROJECT_ID = config_service.PROJECT_ID
LOCATION = config_service.WORKFLOWS_LOCATION
BACKEND_EXECUTOR_URL = config_service.WORKFLOWS_EXECUTOR_URL
# Safety cap when following `nextPageToken` on the stepEntries REST call.
# GCP returns at most 1000 entries per page, so 20 pages = 20k entries.
MAX_STEP_ENTRY_PAGES = 20


class WorkflowService:
    """Orchestrates multi-step generative AI workflows."""

    def __init__(
        self,
        workflow_repository: WorkflowRepository = Depends(),
        workflow_run_repository: WorkflowRunRepository = Depends(),
        source_asset_service: SourceAssetService = Depends(),
        workflow_template_repository: WorkflowTemplateRepository = Depends(),
    ):
        self.imagen_service = ImagenService()
        self.workflow_repository = workflow_repository
        self.workflow_run_repository = workflow_run_repository
        self.source_asset_service = source_asset_service
        self.workflow_template_repository = workflow_template_repository

    def _generate_workflow_yaml(
        self,
        workflow: WorkflowModel,
    ):
        """This function contains the business logic for generating the workflow."""
        user_id = workflow.user_id
        logger.info("Received workflow generation request for user %s", user_id)
        # A very basic transformation to a GCP-like workflow structure
        step_outputs = {}
        gcp_steps = []
        # We init with this default param that is going to propagate user auth header
        workflow_params = ["user_auth_header"]
        user_input_step_id = None
        # Build dependency graph for topological sorting
        steps_by_id = {s.step_id: s for s in workflow.steps}
        adj = {s.step_id: [] for s in workflow.steps}
        in_degree = {s.step_id: 0 for s in workflow.steps}

        for step in workflow.steps:
            if step.inputs:
                inputs_dump = step.inputs.model_dump()

                def extract_refs(val):
                    if isinstance(val, dict):
                        if "step" in val:
                            ref = val["step"]
                            if ref in adj:
                                adj[ref].append(step.step_id)
                                in_degree[step.step_id] += 1
                        for v in val.values():
                            extract_refs(v)
                    elif isinstance(val, list):
                        for item in val:
                            extract_refs(item)

                for input_value in inputs_dump.values():
                    extract_refs(input_value)

        queue = [s_id for s_id, deg in in_degree.items() if deg == 0]
        sorted_steps = []
        while queue:
            curr = queue.pop(0)
            sorted_steps.append(steps_by_id[curr])
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_steps) != len(workflow.steps):
            raise ValueError("Cycle detected in workflow graph")

        for step in sorted_steps:
            if step.type.value == NodeTypes.USER_INPUT:
                print("USER INPUT FOUND")
                # This is a user input step, so we should treat it as a workflow parameter
                user_input_step_id = step.step_id
                for output_name, output_value in step.outputs.items():
                    workflow_params.append(output_name)
                continue

            step_type = step.type.value.lower()
            step_name = step.step_id
            config = step.settings if step.settings else {}
            config = (
                config.model_dump() if isinstance(config, BaseModel) else config
            )

            # Resolve inputs
            resolved_inputs = {}

            def resolve_value(value):
                # If it's a StepOutputReference (dict with step and output)
                if (
                    isinstance(value, dict)
                    and "step" in value
                    and "output" in value
                ):
                    ref_step_id = value["step"]
                    ref_output_name = value["output"]

                    if ref_step_id == user_input_step_id:
                        return f"${{args.{ref_output_name}}}"
                    return f"${{{ref_step_id}_result.body.{ref_output_name}}}"
                # If it's a list, resolve each item
                if isinstance(value, list):
                    return [resolve_value(item) for item in value]
                # Otherwise, return as is
                return value

            for input_name, input_value in step.inputs.model_dump().items():
                resolved_inputs[input_name] = resolve_value(input_value)

            body = {
                "workspace_id": "${args.workspace_id}",  # Dynamically injected from workspaceId passed at execution
                "inputs": resolved_inputs,
                "config": config,
            }

            # MVP ONLY: emit the same step N times back to back so we can
            # observe how GCP records a step that executes more than once.
            # Iterations 0..N-2 are renamed `{step_id}__iter_{k}` (Cloud
            # Workflows rejects duplicate step names); the LAST iteration keeps
            # the original `{step_id}` / `{step_id}_result` names so downstream
            # `${step_id_result...}` references keep resolving to the last
            # iteration. With MVP_STEP_REPEAT_COUNT == 1 the output is
            # identical to the pre-MVP single-execution YAML.
            repeat_count = max(1, workflow_constants.MVP_STEP_REPEAT_COUNT)
            for iteration in range(repeat_count):
                emitted_name = build_iteration_step_name(
                    step_name,
                    iteration,
                    repeat_count,
                )
                gcp_step = {
                    emitted_name: {
                        "call": "http.post",
                        "args": {
                            "url": f"{BACKEND_EXECUTOR_URL}/{step_type}",
                            "headers": {
                                "Authorization": "${args.user_auth_header}"
                            },
                            # Deep copy so PyYAML does not emit anchors/aliases
                            # for the repeated body object.
                            "body": copy.deepcopy(body),
                        },
                        # Each iteration gets its own result variable so its
                        # output stays independently visible in GCP.
                        "result": f"{emitted_name}_result",
                    },
                }
                gcp_steps.append(gcp_step)

            # Store mock outputs for subsequent steps
            step_outputs[step_name] = {
                output_name: f"{step_name}_result.{output_name}"
                for output_name in step.outputs
            }

        gcp_workflow = {"main": {"params": ["args"], "steps": gcp_steps}}

        yaml_output = yaml.dump(gcp_workflow, indent=2)

        return yaml_output

    def validate_workflow(
        self,
        workflow_dto: WorkflowBase,
        user: UserModel,
    ) -> dict[str, Any]:
        """Validates workflow definition and steps structure without persisting to database or GCP."""
        transient_workflow = WorkflowModel(
            id="validation-temp",
            user_id=user.id,
            name=workflow_dto.name,
            description=workflow_dto.description,
            steps=workflow_dto.steps,
        )
        try:
            self._generate_workflow_yaml(transient_workflow)
        except Exception as e:
            logger.error("Workflow validation failed: %s", e)
            raise ValueError(f"Invalid workflow structure: {str(e)}")
        return {"valid": True, "message": "Workflow structure is valid."}

    def _create_gcp_workflow(self, source_contents: str, workflow_id: str):
        client = workflows_v1.WorkflowsClient()

        # Initialize request argument(s)
        workflow = workflows_v1.Workflow()
        workflow.source_contents = source_contents
        workflow.execution_history_level = (
            workflows_v1.ExecutionHistoryLevel.EXECUTION_HISTORY_DETAILED
        )
        if config_service.BACKEND_SERVICE_ACCOUNT_EMAIL:
            workflow.service_account = (
                config_service.BACKEND_SERVICE_ACCOUNT_EMAIL
            )

        request = workflows_v1.CreateWorkflowRequest(
            parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
            workflow=workflow,
            workflow_id=workflow_id,
        )

        try:
            operation = client.create_workflow(request=request)
            response = operation.result()
            return response
        except InvalidArgument as e:
            raise ValueError(str(e))

    def _update_gcp_workflow(self, source_contents: str, workflow_id: str):
        client = workflows_v1.WorkflowsClient()

        # Initialize request argument(s)
        workflow = workflows_v1.Workflow(
            name=f"projects/{PROJECT_ID}/locations/{LOCATION}/workflows/{workflow_id}",
        )
        workflow.source_contents = source_contents
        workflow.execution_history_level = (
            workflows_v1.ExecutionHistoryLevel.EXECUTION_HISTORY_DETAILED
        )
        if config_service.BACKEND_SERVICE_ACCOUNT_EMAIL:
            workflow.service_account = (
                config_service.BACKEND_SERVICE_ACCOUNT_EMAIL
            )

        request = workflows_v1.UpdateWorkflowRequest(
            workflow=workflow,
        )

        try:
            operation = client.update_workflow(request=request)
            response = operation.result()
            return response
        except InvalidArgument as e:
            raise ValueError(str(e))

    def _delete_gcp_workflow(self, workflow_id: str):
        client = workflows_v1.WorkflowsClient()

        # Construct the fully qualified location path.
        parent = client.workflow_path(
            config_service.PROJECT_ID,
            config_service.WORKFLOWS_LOCATION,
            workflow_id,
        )

        request = workflows_v1.DeleteWorkflowRequest(
            name=parent,
        )

        try:
            operation = client.delete_workflow(request=request)
            response = operation.result()
            logger.info(
                f"Deleted GCP workflow for id '{workflow_id}' with response '{response}'",
            )
            return response
        except NotFound:
            logger.warning(
                f"Workflow '{workflow_id}' not found in GCP. Proceeding with local deletion.",
            )
            return None

    async def create_workflow(
        self,
        workflow_dto: WorkflowCreateDto,
        user: UserModel,
    ) -> WorkflowModel:
        """Creates a new workflow definition."""
        try:
            # 1. Generate the ID manually
            workflow_id = f"id-{uuid.uuid4()}"

            # 2. Create the workflow in the database
            workflow_model = WorkflowModel(
                id=workflow_id,
                user_id=user.id,
                name=workflow_dto.name,
                description=workflow_dto.description,
                steps=workflow_dto.steps,
            )
            created_workflow = await self.workflow_repository.create(
                workflow_model
            )

            # 3. Generate GCP Workflow YAML (using the same ID)
            yaml_output = self._generate_workflow_yaml(created_workflow)
            logger.info("Generated YAML:")
            logger.info(yaml_output)

            # 4. Create GCP Workflow
            try:
                self._create_gcp_workflow(yaml_output, workflow_id)
            except Exception as e:
                # Rollback DB creation if GCP creation fails
                logger.error(
                    "Failed to create GCP workflow: %s. Rolling back DB.", e
                )
                await self.workflow_repository.delete(created_workflow.id)
                raise e

            return created_workflow
        except ValidationError as e:
            raise ValueError(str(e))
        except Exception as e:
            # TODO: Improve error handling here
            logging.exception(e)
            raise e

    async def get_workflow(self, user_id: int, workflow_id: str):
        #  Add logic here if needed before fetching from repository
        workflow = await self.workflow_repository.get_by_id(workflow_id)
        if workflow and workflow.user_id == user_id:
            return workflow
        return None

    async def get_by_id(self, workflow_id: str) -> WorkflowModel | None:
        """Retrieves a workflow by its ID without any authorization checks."""
        return await self.workflow_repository.get_by_id(workflow_id)

    async def query_workflows(
        self,
        user_id: int,
        search_dto: WorkflowSearchDto,
    ) -> PaginationResponseDto[WorkflowModel]:
        return await self.workflow_repository.query(user_id, search_dto)

    async def update_workflow(
        self,
        workflow_id: str,
        workflow_dto: WorkflowCreateDto,
        user: UserModel,
    ) -> WorkflowModel | None:
        """Validates and updates a workflow."""
        try:
            # Create the full model from the DTO, preserving the existing ID and user.
            updated_model = WorkflowModel(
                id=workflow_id,
                user_id=user.id,
                name=workflow_dto.name,
                description=workflow_dto.description,
                steps=workflow_dto.steps,
            )

            yaml_output = self._generate_workflow_yaml(updated_model)
            logger.info("Generated YAML for update:")
            logger.info(yaml_output)

            # The GCP workflow ID matches the DB ID (which is already in the format id-UUID)
            self._update_gcp_workflow(yaml_output, workflow_id)

            return await self.workflow_repository.update(
                workflow_id, updated_model
            )
        except ValidationError as e:
            raise ValueError(str(e))

    async def delete_by_id(self, workflow_id: str) -> bool:
        """Deletes a workflow from the system."""
        # The GCP workflow ID matches the DB ID
        self._delete_gcp_workflow(workflow_id)
        response = await self.workflow_repository.delete(workflow_id)
        return response

    async def execute_workflow(
        self,
        workflow_id: str,
        args: dict,
        user: UserModel,
    ) -> str:
        """Executes a workflow with snapshotting."""
        # 1. Fetch current workflow state (Snapshot source)
        workflow_model = await self.get_by_id(workflow_id)
        if not workflow_model:
            raise ValueError(f"Workflow {workflow_id} not found")

        # 2. Trigger GCP Execution
        # Initialize API clients.
        execution_client = executions_v1.ExecutionsAsyncClient()

        # Construct the fully qualified location path.
        # We use the static method from WorkflowsClient to avoid partial initialization of a sync client
        parent = workflows_v1.WorkflowsClient.workflow_path(
            config_service.PROJECT_ID,
            config_service.WORKFLOWS_LOCATION,
            workflow_id,
        )

        execution = executions_v1.Execution(argument=json.dumps(args))

        # Execute the workflow.
        response = await execution_client.create_execution(
            parent=parent,
            execution=execution,
        )

        execution_id = response.name.split("/")[-1]

        # 3. Save Snapshot
        workspace_id = args.get("workspace_id")
        # Ensure workspace_id is int if present
        if workspace_id:
            try:
                workspace_id = int(workspace_id)
            except:
                workspace_id = None

        await self._create_execution_snapshot(
            execution_id,
            workflow_id,
            workflow_model,
            user.id,
            workspace_id,
        )

        return execution_id

    async def _create_execution_snapshot(
        self,
        execution_id: str,
        workflow_id: str,
        snapshot: WorkflowModel,
        user_id: int,
        workspace_id: int | None = None,
    ):
        """Creates a DB record for the execution with a snapshot of the workflow."""
        try:
            # workflow_snapshot field is JSON type.
            # Use mode='json' to ensure all types (Enums, etc.) are serialized to primitives
            # We must pass a DICT to the Pydantic model now that the field is Dict[str, Any]
            # We MUST include 'id' and 'user_id' so that WorkflowModel.model_validate works during rehydration.
            # We still exclude created_at/updated_at to save space/noise, as they will be re-generated (or nullable) upon validation if defaults exist.
            snapshot_data = snapshot.model_dump(
                mode="json",
                exclude={"created_at", "updated_at"},
            )

            workflow_run = WorkflowRunModel(
                id=execution_id,
                workflow_id=workflow_id,
                user_id=user_id,
                workspace_id=workspace_id,
                status=WorkflowRunStatusEnum.RUNNING,
                started_at=datetime.datetime.now(datetime.UTC),
                workflow_snapshot=snapshot_data,
            )
            await self.workflow_run_repository.create(workflow_run)
            logger.info("Created snapshot for execution %s", execution_id)
        except Exception as e:
            logger.exception(
                f"Failed to create execution snapshot for {execution_id}: {e}",
            )

    async def batch_execute_workflow(
        self,
        workflow_id: str,
        batch_dto: BatchExecutionRequestDto,
        user: UserModel,
    ) -> BatchExecutionResponseDto:
        """Executes a workflow for each item in the batch request.
        Handles GCS URI ingestion for image arguments.
        """
        results: list[BatchItemResultDto] = []

        async def process_row(item) -> BatchItemResultDto:
            try:
                # 1. Process Arguments (Ingest GCS URIs)
                processed_args = {}
                workspace_id = item.args.get("workspace_id")

                for key, value in item.args.items():
                    is_gcs_string = isinstance(value, str) and value.startswith(
                        "gs://"
                    )
                    is_gcs_list = (
                        isinstance(value, list)
                        and len(value) > 0
                        and isinstance(value[0], str)
                        and value[0].startswith("gs://")
                    )

                    if is_gcs_string or is_gcs_list:
                        try:
                            if not workspace_id:
                                raise ValueError(
                                    "No workspace_id provided for GCS ingestion.",
                                )

                            w_id = int(workspace_id)
                            uris = [value] if is_gcs_string else value

                            assets = await asyncio.gather(
                                *[
                                    self.source_asset_service.create_from_gcs_uri(
                                        user=user,
                                        workspace_id=w_id,
                                        gcs_uri=uri,
                                    )
                                    for uri in uris
                                ],
                            )

                            ingested_results = [
                                {"sourceAssetId": asset.id, "previewUrl": uri}
                                for asset, uri in zip(assets, uris)
                            ]

                            processed_args[key] = (
                                ingested_results[0]
                                if is_gcs_string
                                else ingested_results
                            )

                        except Exception as e:
                            logger.exception(
                                f"Failed to ingest GCS URI in '{key}': {e!s} from row {item.row_index}",
                            )
                            return BatchItemResultDto(
                                row_index=item.row_index,
                                status="FAILED",
                                error=f"Invalid GCS URI in '{key}': {e!s}",
                            )
                    else:
                        processed_args[key] = value

                # 2. Execute Workflow
                execution_id = await self.execute_workflow(
                    workflow_id=workflow_id,
                    args=processed_args,
                    user=user,
                )

                return BatchItemResultDto(
                    row_index=item.row_index,
                    execution_id=execution_id,
                    status="SUCCESS",
                )

            except Exception as e:
                return BatchItemResultDto(
                    row_index=item.row_index,
                    status="FAILED",
                    error=str(e),
                )

        tasks = [process_row(item) for item in batch_dto.items]
        results = await asyncio.gather(*tasks)

        return BatchExecutionResponseDto(results=results)

    @staticmethod
    def _interpolate_prompt_variables(
        prompt: str,
        step_inputs: dict[str, Any],
    ) -> str:
        """Interpolates <var_name> placeholders in prompt using step inputs."""
        return interpolate_prompt_variables(
            prompt=prompt,
            variables=step_inputs,
            keep_unresolved=True,
        )

    async def get_execution_details(
        self,
        workflow_id: str,
        execution_id: str,
    ) -> dict | None:
        """Retrieves the details of a workflow execution."""
        client = executions_v1.ExecutionsClient()

        if not execution_id.startswith("projects/"):
            parent = client.workflow_path(
                config_service.PROJECT_ID,
                config_service.WORKFLOWS_LOCATION,
                workflow_id,
            )
            execution_name = f"{parent}/executions/{execution_id}"
        else:
            execution_name = execution_id

        try:
            execution = client.get_execution(name=execution_name)
        except NotFound:
            return None

        result = None
        user_inputs = (
            json.loads(execution.argument) if execution.argument else {}
        )
        if execution.state == executions_v1.Execution.State.SUCCEEDED:
            result = execution.result

        # Fetch step entries using REST API (paginated: a step that runs more
        # than once easily exceeds the 1000 entries returned per page).
        step_entries: list[dict[str, Any]] = []
        try:
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            authed_session = AuthorizedSession(credentials)
            url = f"https://workflowexecutions.googleapis.com/v1/{execution_name}/stepEntries"
            page_token: str | None = None
            for _ in range(MAX_STEP_ENTRY_PAGES):
                params = {"pageToken": page_token} if page_token else {}
                response = authed_session.get(url, params=params)
                if response.status_code != 200:
                    logger.warning(
                        "Failed to fetch step entries: %s", response.text
                    )
                    break
                payload = response.json()
                step_entries.extend(payload.get("stepEntries", []) or [])
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
            else:
                logger.warning(
                    "Stopped fetching step entries after %s pages; "
                    "results may be truncated.",
                    MAX_STEP_ENTRY_PAGES,
                )
        except Exception as e:
            logger.error("Error fetching step entries: %s", e)
            step_entries = []

        # Calculate duration
        duration = 0.0
        if execution.start_time:
            start_timestamp = execution.start_time.timestamp()  # type: ignore
            if execution.end_time:
                end_timestamp = execution.end_time.timestamp()  # type: ignore
                duration = end_timestamp - start_timestamp
            else:
                import time

                duration = time.time() - start_timestamp

        # Try to fetch snapshot from DB
        logger.info(
            "Attempting to fetch snapshot for execution_id: %s", execution_id
        )

        # Ensure we check the short ID if a long ID is passed
        lookup_id = execution_id
        if execution_id.startswith("projects/") or execution_id.startswith(
            "//"
        ):
            lookup_id = execution_id.rsplit("/", maxsplit=1)[-1]

        snapshot_run = await self.workflow_run_repository.get_by_id(lookup_id)

        workflow_model = None
        if snapshot_run and snapshot_run.workflow_snapshot:
            logger.info("Snapshot FOUND for execution_id: %s", execution_id)
            # Rehydrate WorkflowModel from snapshot
            try:
                # snapshot_run.workflow_snapshot is a dict
                workflow_model = WorkflowModel.model_validate(
                    snapshot_run.workflow_snapshot,
                )
            except Exception as e:
                logger.error("Failed to rehydrate snapshot: %s", e)
                workflow_model = None
        else:
            logger.warning(
                f"Snapshot NOT FOUND for execution_id: {execution_id}. Falling back to current workflow definition.",
            )
            # Fallback to current definition
            workflow_model = await self.get_by_id(workflow_id)

        if not workflow_model:
            # If workflow definition is missing, we might still return basic execution details
            logger.warning(
                f"Workflow definition {workflow_id} not found for execution {execution_id}",
            )
            return {
                "id": execution.name,
                "state": execution.state.name,
                "result": result,
                "duration": round(duration, 2),
                "error": execution.error.context if execution.error else None,
                "step_entries": [],  # Cannot map steps without definition
            }

        # --- Lazy Status Update Start ---
        # If we have a snapshot and its status is RUNNING but GCP says it's done, let's update the DB.
        # This acts as a lazy sync so we don't need a background poller.
        if (
            snapshot_run
            and snapshot_run.status == WorkflowRunStatusEnum.RUNNING.value
        ):
            final_status = None
            if execution.state == executions_v1.Execution.State.SUCCEEDED:
                final_status = WorkflowRunStatusEnum.COMPLETED
            elif execution.state == executions_v1.Execution.State.FAILED:
                final_status = WorkflowRunStatusEnum.FAILED
            elif execution.state == executions_v1.Execution.State.CANCELLED:
                final_status = WorkflowRunStatusEnum.CANCELED

            if final_status:
                try:
                    update_data = {
                        "status": final_status.value,
                        "completed_at": (
                            execution.end_time
                            if execution.end_time
                            else datetime.datetime.now(datetime.UTC)
                        ),
                    }
                    # We fire and forget this update essentially (await it but don't block return on failure)
                    await self.workflow_run_repository.update(
                        snapshot_run.id,
                        update_data,
                    )
                    logger.info(
                        f"Lazily updated execution {execution_id} status to {final_status.value}",
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to lazily update execution status: %s", e
                    )
        # --- Lazy Status Update End ---

        user_input_step = next(
            (
                step
                for step in workflow_model.steps
                if step.type == NodeTypes.USER_INPUT
            ),
            None,
        )
        user_input_step_id = (
            user_input_step.step_id
            if user_input_step
            else (
                workflow_model.steps[0].step_id
                if workflow_model.steps
                else "user_input"
            )
        )

        # MVP: a logical step can now produce several GCP step entries (one per
        # emitted iteration), so `previous_outputs` tracks a LIST of outputs
        # per step id instead of a single dict that each pass overwrote.
        previous_outputs: dict[str, list[Any]] = {}
        formatted_step_entries = []

        # 1. Add User Input Step Entry (Virtual)
        # This ensures the User Input step appears in the history and its outputs are available for resolution
        previous_outputs[user_input_step_id] = [user_inputs]
        if "user_input" not in previous_outputs:
            previous_outputs["user_input"] = [user_inputs]
        user_input_time = (
            execution.start_time.isoformat()  # type: ignore
            if execution.start_time
            else None
        )
        formatted_step_entries.append(
            {
                "step_id": user_input_step_id,
                "state": "STATE_SUCCEEDED",  # User input is always considered succeeded if execution started
                # NOTE: the flat `step_inputs` / `step_outputs` mirror the LAST
                # iteration and are kept only for backwards compatibility with
                # the current frontend. They will be dropped once the UI reads
                # `history[]`.
                "step_inputs": {},
                "step_outputs": user_inputs,
                "start_time": user_input_time,
                "end_time": user_input_time,  # Instant
                # `history` is always present, even for a single execution.
                "history": [
                    {
                        "iteration": 0,
                        "state": "STATE_SUCCEEDED",
                        "start_time": user_input_time,
                        "end_time": user_input_time,
                        "step_inputs": {},
                        "step_outputs": user_inputs,
                        "error": None,
                    },
                ],
            },
        )

        def outputs_for_iteration(step_id: str, iteration: int) -> Any:
            """Returns an upstream step's outputs for the given iteration.

            Falls back to that step's last available iteration when it ran
            fewer times than the step currently being resolved.
            """
            iterations = previous_outputs.get(step_id)
            if not iterations:
                return {}
            if iteration < len(iterations):
                return iterations[iteration] or {}
            return iterations[-1] or {}

        def resolve_value(value, iteration: int = 0):
            if isinstance(value, StepOutputReference):
                return outputs_for_iteration(value.step, iteration).get(
                    value.output
                )
            if (
                isinstance(value, dict)
                and "step" in value
                and "output" in value
            ):
                return outputs_for_iteration(value["step"], iteration).get(
                    value["output"]
                )
            if isinstance(value, list):
                return [resolve_value(item, iteration) for item in value]
            return value

        def resolve_step_inputs(current_step, iteration: int) -> dict:
            """Resolves one iteration's inputs from the step definition."""
            # Extract inputs from step
            raw_inputs = (
                current_step.inputs.model_dump()
                if isinstance(current_step.inputs, BaseModel)
                else (
                    current_step.inputs
                    if isinstance(current_step.inputs, dict)
                    else {}
                )
            )
            step_inputs: dict[str, Any] = {}

            if current_step.type == NodeTypes.IMAGE:
                settings_mode = (
                    getattr(current_step.settings, "mode", "generate_image")
                    if isinstance(current_step.settings, BaseModel)
                    else (
                        current_step.settings.get("mode", "generate_image")
                        if isinstance(current_step.settings, dict)
                        else "generate_image"
                    )
                )
                allowed_inputs = IMAGE_MODE_ALLOWED_INPUTS.get(
                    settings_mode, ["prompt"]
                )
                for inp_name, inp_value in raw_inputs.items():
                    if inp_name in allowed_inputs and inp_value is not None:
                        step_inputs[inp_name] = resolve_value(
                            inp_value, iteration
                        )
            else:
                for inp_name, inp_value in raw_inputs.items():
                    if inp_value is not None:
                        step_inputs[inp_name] = resolve_value(
                            inp_value, iteration
                        )

                if current_step.type == NodeTypes.GENERATE_TEXT:
                    prompt_val = step_inputs.get("prompt")
                    if isinstance(prompt_val, str):
                        step_inputs["prompt"] = (
                            self._interpolate_prompt_variables(
                                prompt_val, step_inputs
                            )
                        )
            return step_inputs

        def extract_step_outputs(current_step, entry, base_step_id) -> Any:
            """Reads one entry's outputs from its own variable snapshot."""
            emitted_name = entry.get("step") or base_step_id
            variable_data = entry.get("variableData", {}) or {}
            variables = variable_data.get("variables", {}) or {}
            # Each emitted iteration has its own result variable; fall back to
            # the base name for entries produced before this MVP existed.
            step_results = variables.get(f"{emitted_name}_result")
            if step_results is None:
                step_results = variables.get(f"{base_step_id}_result", {})
            raw_outputs = (step_results or {}).get("body", {})

            if current_step.type == NodeTypes.IMAGE and isinstance(
                raw_outputs, dict
            ):
                img_val = (
                    raw_outputs.get("generated_image")
                    or raw_outputs.get("edited_image")
                    or raw_outputs.get("upscaled_image")
                    or raw_outputs.get("image_output")
                )
                return (
                    {"generated_image": img_val} if img_val is not None else {}
                )
            return raw_outputs

        # `entryId` is monotonic per execution, so it is the natural sort key.
        # If any entry lacks it, keep the order returned by the API instead.
        def entry_sort_key(entry) -> int | None:
            try:
                return int(entry.get("entryId"))
            except (TypeError, ValueError):
                return None

        if step_entries and all(
            entry_sort_key(entry) is not None for entry in step_entries
        ):
            ordered_entries = sorted(step_entries, key=entry_sort_key)  # type: ignore[arg-type,return-value]
        else:
            ordered_entries = list(step_entries)

        # Group the entries of a logical step together: `{step}__iter_k` and
        # `{step}` all belong to the same base step id.
        steps_by_id = {step.step_id: step for step in workflow_model.steps}
        grouped_entries: dict[str, list[dict[str, Any]]] = {}
        group_order: list[str] = []
        for entry in ordered_entries:
            emitted_name = entry.get("step")
            if not emitted_name or emitted_name == "end":
                continue
            base_step_id, _iteration = parse_iteration_step_name(emitted_name)
            if base_step_id not in steps_by_id:
                continue
            if base_step_id not in grouped_entries:
                grouped_entries[base_step_id] = []
                group_order.append(base_step_id)
            grouped_entries[base_step_id].append(entry)

        for base_step_id in group_order:
            current_step = steps_by_id[base_step_id]
            history: list[dict[str, Any]] = []

            for iteration, entry in enumerate(grouped_entries[base_step_id]):
                step_inputs = resolve_step_inputs(current_step, iteration)
                step_outputs = extract_step_outputs(
                    current_step, entry, base_step_id
                )

                # Store this iteration's outputs for subsequent steps.
                previous_outputs.setdefault(base_step_id, []).append(
                    step_outputs
                )

                history.append(
                    {
                        "iteration": iteration,
                        "state": entry.get("state"),
                        "start_time": entry.get("createTime"),
                        "end_time": entry.get("updateTime"),
                        "step_inputs": step_inputs,
                        "step_outputs": step_outputs,
                        # A failed iteration returns an empty variableData, so
                        # without this it would render as an empty row.
                        "error": entry.get("exception"),
                    },
                )

            last_iteration = history[-1]
            formatted_step_entries.append(
                {
                    "step_id": base_step_id,
                    "state": last_iteration["state"],
                    # NOTE: kept for backwards compatibility with the current
                    # frontend (mirrors the LAST iteration). Drop them once the
                    # UI consumes `history[]`.
                    "step_inputs": last_iteration["step_inputs"],
                    "step_outputs": last_iteration["step_outputs"],
                    "start_time": history[0]["start_time"],
                    "end_time": last_iteration["end_time"],
                    "history": history,
                },
            )

        return {
            "id": execution.name,
            "state": execution.state.name,
            "result": result,
            "duration": round(duration, 2),
            "error": execution.error.context if execution.error else None,
            "step_entries": formatted_step_entries,
            "workflow_definition": (
                workflow_model.model_dump(by_alias=True)
                if workflow_model
                else None
            ),
        }

    def list_executions(
        self,
        workflow_id: str,
        limit: int = 10,
        page_token: str | None = None,
        filter_str: str | None = None,
    ):
        """Lists executions for a given workflow."""
        client = executions_v1.ExecutionsClient()
        parent = client.workflow_path(PROJECT_ID, LOCATION, workflow_id)

        request = executions_v1.ListExecutionsRequest(
            parent=parent,
            page_size=limit,
            page_token=page_token,
            filter=filter_str,
        )

        response = client.list_executions(request=request)
        pages_iterator = response.pages

        try:
            current_page = next(pages_iterator)
        except StopIteration:
            print("No executions found.")
            return None

        executions = []
        for execution in current_page.executions:
            # Calculate duration
            duration = 0.0
            if execution.start_time:
                start_timestamp = execution.start_time.timestamp()  # type: ignore
                if execution.end_time:
                    end_timestamp = execution.end_time.timestamp()  # type: ignore
                    duration = end_timestamp - start_timestamp
                else:
                    import time

                    duration = time.time() - start_timestamp

            executions.append(
                {
                    "id": execution.name.split("/")[-1],
                    "state": execution.state.name,
                    "start_time": execution.start_time,
                    "end_time": execution.end_time,
                    "duration": round(duration, 2),
                    "error": (
                        execution.error.context if execution.error else None
                    ),
                },
            )

        return {
            "executions": executions,
            "next_page_token": current_page.next_page_token,
        }

    async def create_template(
        self,
        template_dto: WorkflowTemplateCreateDto,
        user: UserModel,
    ) -> WorkflowTemplateModel:
        """Creates a new workflow template, ensuring the name is unique per user."""
        existing = await self.workflow_template_repository.get_by_user_and_name(
            user.id, template_dto.name
        )
        if existing:
            raise ValueError(
                f"A template named '{template_dto.name}' already exists. Please choose a unique name."
            )

        template_id = f"tmpl-{uuid.uuid4()}"
        template_model = WorkflowTemplateModel(
            id=template_id,
            user_id=user.id,
            name=template_dto.name.strip(),
            description=template_dto.description,
            steps=template_dto.steps,
        )

        # Validate workflow steps structure by generating GCP workflow YAML representation.
        # This guarantees that the template is a correct working version before saving.
        self.validate_workflow(template_dto, user)

        return await self.workflow_template_repository.create(template_model)

    async def list_templates(
        self,
        user_id: int,
    ) -> list[WorkflowTemplateModel]:
        """Retrieves all templates created by the user."""
        return await self.workflow_template_repository.list_by_user(user_id)

    async def get_template(
        self,
        template_id: str,
        user_id: int,
    ) -> WorkflowTemplateModel | None:
        """Retrieves a single template if owned by the user."""
        template = await self.workflow_template_repository.get_by_id(
            template_id
        )
        if template and template.user_id == user_id:
            return template
        return None

    async def delete_template(
        self,
        template_id: str,
        user_id: int,
    ) -> bool:
        """Deletes a template if owned by the user."""
        return await self.workflow_template_repository.delete_by_id_and_user(
            template_id, user_id
        )
