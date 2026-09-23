# Copyright 2026 Google LLC
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
"""Tests for the MVP static step repetition and the `history[]` mapper.

MVP ONLY: these cover the temporary loop simulation
(`MVP_STEP_REPEAT_COUNT`) and the grouped execution-details response. They
should be revisited when real loop nodes replace the static repetition.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from google.cloud.workflows import executions_v1 as exec_v1

from src.workflows.schema.workflow_model import (
    GenerateTextInputs,
    GenerateTextSettings,
    GenerateTextStep,
    NodeTypes,
    StepOutputReference,
    UserInputStep,
    WorkflowModel,
)
from src.workflows.schema.workflow_run_model import WorkflowRunStatusEnum
from src.workflows.workflow_service import (
    MAX_STEP_ENTRY_PAGES,
    BACKEND_EXECUTOR_URL,
    WorkflowService,
)
from src.workflows.workflow_utils import (
    build_iteration_step_name,
    parse_iteration_step_name,
)

REPEAT_CONSTANT = "src.workflows.workflow_constants.MVP_STEP_REPEAT_COUNT"


@pytest.fixture(name="mock_run_repo")
def fixture_mock_run_repo():
    repo = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.update = AsyncMock()
    return repo


@pytest.fixture(name="workflow_service")
def fixture_workflow_service(mock_run_repo):
    return WorkflowService(
        workflow_repository=AsyncMock(),
        workflow_run_repository=mock_run_repo,
        source_asset_service=MagicMock(),
        workflow_template_repository=AsyncMock(),
    )


def _single_step_workflow() -> WorkflowModel:
    return WorkflowModel(
        id="id-1234",
        user_id=1,
        name="Test Workflow",
        description="A test workflow",
        steps=[
            GenerateTextStep(
                step_id="step_1",
                type=NodeTypes.GENERATE_TEXT,
                inputs=GenerateTextInputs(prompt="Hello World"),
                settings=GenerateTextSettings(
                    model="gemini-1.5", temperature=0.7
                ),
            ),
        ],
    )


def _chained_workflow() -> WorkflowModel:
    """user_input -> step_a -> step_b."""
    return WorkflowModel(
        id="id-chain",
        user_id=1,
        name="Chained Workflow",
        description="user_input -> step_a -> step_b",
        steps=[
            UserInputStep(
                step_id="user_input",
                type=NodeTypes.USER_INPUT,
                outputs={"City": "string"},
            ),
            GenerateTextStep(
                step_id="step_a",
                type=NodeTypes.GENERATE_TEXT,
                inputs=GenerateTextInputs(
                    prompt="Weather in <City>",
                    City=StepOutputReference(step="user_input", output="City"),
                ),
                settings=GenerateTextSettings(
                    model="gemini-2.5-flash", temperature=0.7
                ),
            ),
            GenerateTextStep(
                step_id="step_b",
                type=NodeTypes.GENERATE_TEXT,
                inputs=GenerateTextInputs(
                    prompt="Advice for <topic>",
                    topic=StepOutputReference(
                        step="step_a", output="generated_text"
                    ),
                ),
                settings=GenerateTextSettings(
                    model="gemini-2.5-flash", temperature=0.7
                ),
            ),
        ],
    )


class TestIterationStepNames:
    """Tests for the `__iter_k` naming helpers."""

    def test_build_last_iteration_keeps_base_name(self):
        assert build_iteration_step_name("foo", 1, 2) == "foo"
        assert build_iteration_step_name("foo", 0, 1) == "foo"

    def test_build_non_last_iteration_is_suffixed(self):
        assert build_iteration_step_name("foo", 0, 2) == "foo__iter_0"
        assert build_iteration_step_name("foo", 1, 3) == "foo__iter_1"

    def test_parse_round_trip(self):
        assert parse_iteration_step_name("foo__iter_0") == ("foo", 0)
        assert parse_iteration_step_name("foo__iter_12") == ("foo", 12)

    def test_parse_plain_name(self):
        assert parse_iteration_step_name("foo") == ("foo", None)
        assert parse_iteration_step_name("") == ("", None)
        assert parse_iteration_step_name("foo__iter_x") == (
            "foo__iter_x",
            None,
        )


class TestMvpYamlRepetition:
    """Tests for `_generate_workflow_yaml` with the MVP repeat count."""

    @patch(REPEAT_CONSTANT, 1)
    def test_repeat_one_is_byte_identical_to_legacy_output(
        self, workflow_service
    ):
        """With the repeat disabled the YAML must match the pre-MVP output."""
        yaml_output = workflow_service._generate_workflow_yaml(
            _single_step_workflow()
        )

        expected = (
            "main:\n"
            "  params:\n"
            "  - args\n"
            "  steps:\n"
            "  - step_1:\n"
            "      args:\n"
            "        body:\n"
            "          config:\n"
            "            model: gemini-1.5\n"
            "            temperature: 0.7\n"
            "          inputs:\n"
            "            input_images: null\n"
            "            input_videos: null\n"
            "            prompt: Hello World\n"
            "          workspace_id: ${args.workspace_id}\n"
            "        headers:\n"
            "          Authorization: ${args.user_auth_header}\n"
            f"        url: {BACKEND_EXECUTOR_URL}/generate_text\n"
            "      call: http.post\n"
            "      result: step_1_result\n"
        )
        assert yaml_output == expected

    @patch(REPEAT_CONSTANT, 2)
    def test_repeat_two_emits_iter_zero_then_base_name(self, workflow_service):
        yaml_output = workflow_service._generate_workflow_yaml(
            _single_step_workflow()
        )
        # No YAML anchors/aliases: each copy carries its own body.
        assert "&id001" not in yaml_output
        assert "*id001" not in yaml_output

        steps = yaml.safe_load(yaml_output)["main"]["steps"]
        assert [list(step.keys())[0] for step in steps] == [
            "step_1__iter_0",
            "step_1",
        ]
        assert steps[0]["step_1__iter_0"]["result"] == "step_1__iter_0_result"
        assert steps[1]["step_1"]["result"] == "step_1_result"
        # Both copies are otherwise identical.
        assert steps[0]["step_1__iter_0"]["args"] == steps[1]["step_1"]["args"]

    @patch(REPEAT_CONSTANT, 3)
    def test_repeat_three_keeps_only_last_on_base_name(self, workflow_service):
        yaml_output = workflow_service._generate_workflow_yaml(
            _single_step_workflow()
        )
        steps = yaml.safe_load(yaml_output)["main"]["steps"]
        assert [list(step.keys())[0] for step in steps] == [
            "step_1__iter_0",
            "step_1__iter_1",
            "step_1",
        ]

    @patch(REPEAT_CONSTANT, 2)
    def test_downstream_references_are_unchanged(self, workflow_service):
        """Downstream steps keep pointing at `${step_a_result...}`."""
        yaml_output = workflow_service._generate_workflow_yaml(
            _chained_workflow()
        )
        steps = yaml.safe_load(yaml_output)["main"]["steps"]

        names = [list(step.keys())[0] for step in steps]
        # user_input is a workflow param, never a GCP step, never repeated.
        assert names == [
            "step_a__iter_0",
            "step_a",
            "step_b__iter_0",
            "step_b",
        ]
        assert not any("user_input" in name for name in names)

        for wrapper in steps[2:]:
            step = list(wrapper.values())[0]
            assert (
                step["args"]["body"]["inputs"]["topic"]
                == "${step_a_result.body.generated_text}"
            )
        # The user input reference is still a workflow argument.
        assert (
            steps[0]["step_a__iter_0"]["args"]["body"]["inputs"]["City"]
            == "${args.City}"
        )

    @patch(REPEAT_CONSTANT, 0)
    def test_repeat_below_one_falls_back_to_single_copy(self, workflow_service):
        yaml_output = workflow_service._generate_workflow_yaml(
            _single_step_workflow()
        )
        steps = yaml.safe_load(yaml_output)["main"]["steps"]
        assert [list(step.keys())[0] for step in steps] == ["step_1"]


def _mock_execution() -> MagicMock:
    execution = MagicMock()
    execution.name = "projects/p/locations/l/workflows/w/executions/e-1"
    execution.state = exec_v1.Execution.State.SUCCEEDED
    execution.argument = json.dumps({"City": "Taipei"})
    execution.result = "{}"
    execution.start_time = MagicMock()
    execution.start_time.isoformat.return_value = "2026-09-17T15:26:37Z"
    execution.end_time = MagicMock()
    execution.error = None
    return execution


def _variables(step_name: str, text: str) -> dict:
    return {
        "variables": {
            f"{step_name}_result": {"body": {"generated_text": text}},
        },
    }


async def _run_details(
    workflow_service,
    mock_run_repo,
    workflow_model,
    pages,
    mock_auth_session_class,
    mock_exec_client_class,
):
    """Drives `get_execution_details` with canned stepEntries pages."""
    mock_client = MagicMock()
    mock_exec_client_class.return_value = mock_client
    mock_client.get_execution.return_value = _mock_execution()

    responses = []
    for page in pages:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = page
        responses.append(response)
    mock_session = MagicMock()
    mock_session.get.side_effect = responses
    mock_auth_session_class.return_value = mock_session

    mock_run = MagicMock()
    mock_run.id = "e-1"
    mock_run.workflow_snapshot = workflow_model.model_dump(mode="json")
    mock_run.status = WorkflowRunStatusEnum.COMPLETED.value
    mock_run_repo.get_by_id.return_value = mock_run

    details = await workflow_service.get_execution_details(
        workflow_id="id-chain",
        execution_id="e-1",
    )
    return details, mock_session


# `google.auth.default` is already stubbed globally in tests/conftest.py.
@patch("src.workflows.workflow_service.executions_v1.ExecutionsClient")
@patch("src.workflows.workflow_service.AuthorizedSession")
class TestExecutionHistoryMapper:
    """Tests for the grouped `history[]` execution-details response."""

    @pytest.mark.anyio
    async def test_two_entries_of_same_step_become_one_history_of_two(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a__iter_0",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                    "createTime": "2026-09-17T15:26:38Z",
                    "updateTime": "2026-09-17T15:26:39Z",
                    "variableData": _variables("step_a__iter_0", "A0"),
                },
                {
                    "step": "step_a",
                    "entryId": "2",
                    "state": "STATE_SUCCEEDED",
                    "createTime": "2026-09-17T15:26:40Z",
                    "updateTime": "2026-09-17T15:26:41Z",
                    "variableData": _variables("step_a", "A1"),
                },
                {
                    "step": "step_b__iter_0",
                    "entryId": "3",
                    "state": "STATE_SUCCEEDED",
                    "createTime": "2026-09-17T15:26:42Z",
                    "updateTime": "2026-09-17T15:26:43Z",
                    "variableData": _variables("step_b__iter_0", "B0"),
                },
                {
                    "step": "step_b",
                    "entryId": "4",
                    "state": "STATE_SUCCEEDED",
                    "createTime": "2026-09-17T15:26:44Z",
                    "updateTime": "2026-09-17T15:26:45Z",
                    "variableData": _variables("step_b", "B1"),
                },
                {"step": "end", "entryId": "5", "state": "STATE_SUCCEEDED"},
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )

        entries = {e["step_id"]: e for e in details["step_entries"]}
        # user_input + step_a + step_b, NOT one entry per GCP entry.
        assert set(entries) == {"user_input", "step_a", "step_b"}

        step_a = entries["step_a"]
        assert [h["iteration"] for h in step_a["history"]] == [0, 1]
        assert step_a["history"][0]["step_outputs"] == {"generated_text": "A0"}
        assert step_a["history"][1]["step_outputs"] == {"generated_text": "A1"}
        # Aggregated window spans the first and last iteration.
        assert step_a["start_time"] == "2026-09-17T15:26:38Z"
        assert step_a["end_time"] == "2026-09-17T15:26:41Z"
        # Legacy flat fields mirror the LAST iteration.
        assert step_a["step_outputs"] == {"generated_text": "A1"}
        assert step_a["state"] == "STATE_SUCCEEDED"

    @pytest.mark.anyio
    async def test_inputs_resolve_against_matching_upstream_iteration(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a__iter_0",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a__iter_0", "A0"),
                },
                {
                    "step": "step_a",
                    "entryId": "2",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a", "A1"),
                },
                {
                    "step": "step_b__iter_0",
                    "entryId": "3",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_b__iter_0", "B0"),
                },
                {
                    "step": "step_b",
                    "entryId": "4",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_b", "B1"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}

        step_b_history = entries["step_b"]["history"]
        assert step_b_history[0]["step_inputs"]["topic"] == "A0"
        assert step_b_history[1]["step_inputs"]["topic"] == "A1"
        assert step_b_history[0]["step_inputs"]["prompt"] == "Advice for A0"
        assert step_b_history[1]["step_inputs"]["prompt"] == "Advice for A1"

    @pytest.mark.anyio
    async def test_fewer_upstream_iterations_fall_back_to_last(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a", "ONLY"),
                },
                {
                    "step": "step_b__iter_0",
                    "entryId": "2",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_b__iter_0", "B0"),
                },
                {
                    "step": "step_b",
                    "entryId": "3",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_b", "B1"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}

        assert len(entries["step_a"]["history"]) == 1
        step_b_history = entries["step_b"]["history"]
        assert step_b_history[0]["step_inputs"]["topic"] == "ONLY"
        assert step_b_history[1]["step_inputs"]["topic"] == "ONLY"

    @pytest.mark.anyio
    async def test_single_entry_still_yields_history_of_one(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                    "createTime": "2026-09-17T15:26:38Z",
                    "updateTime": "2026-09-17T15:26:39Z",
                    "variableData": _variables("step_a", "A"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}

        assert len(entries["step_a"]["history"]) == 1
        assert entries["step_a"]["history"][0]["iteration"] == 0
        assert entries["step_a"]["history"][0]["error"] is None
        # The virtual user input entry has a uniform single-item history.
        assert len(entries["user_input"]["history"]) == 1
        assert entries["user_input"]["history"][0]["step_outputs"] == {
            "City": "Taipei",
        }

    @pytest.mark.anyio
    async def test_exception_surfaces_as_error(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a__iter_0",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a__iter_0", "A0"),
                },
                {
                    "step": "step_a",
                    "entryId": "2",
                    "state": "STATE_FAILED",
                    "exception": {"payload": "boom"},
                    "variableData": {},
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}
        history = entries["step_a"]["history"]

        assert history[0]["error"] is None
        assert history[1]["error"] == {"payload": "boom"}
        assert history[1]["state"] == "STATE_FAILED"
        assert history[1]["step_outputs"] == {}
        assert entries["step_a"]["state"] == "STATE_FAILED"

    @pytest.mark.anyio
    async def test_entries_are_ordered_by_entry_id(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a",
                    "entryId": "9",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a", "LAST"),
                },
                {
                    "step": "step_a__iter_0",
                    "entryId": "2",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a__iter_0", "FIRST"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}
        outputs = [
            h["step_outputs"]["generated_text"]
            for h in entries["step_a"]["history"]
        ]
        assert outputs == ["FIRST", "LAST"]

    @pytest.mark.anyio
    async def test_missing_entry_id_keeps_api_order(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "step_a__iter_0",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a__iter_0", "FIRST"),
                },
                {
                    "step": "step_a",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a", "LAST"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}
        outputs = [
            h["step_outputs"]["generated_text"]
            for h in entries["step_a"]["history"]
        ]
        assert outputs == ["FIRST", "LAST"]

    @pytest.mark.anyio
    async def test_outputs_fall_back_to_base_result_variable(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        """An `__iter_k` entry whose snapshot only holds the base result."""
        page = {
            "stepEntries": [
                {
                    "step": "step_a__iter_0",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a", "FALLBACK"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        entries = {e["step_id"]: e for e in details["step_entries"]}
        assert entries["step_a"]["history"][0]["step_outputs"] == {
            "generated_text": "FALLBACK",
        }

    @pytest.mark.anyio
    async def test_unknown_steps_are_skipped(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        page = {
            "stepEntries": [
                {
                    "step": "not_in_definition",
                    "entryId": "1",
                    "state": "STATE_SUCCEEDED",
                },
                {"entryId": "2", "state": "STATE_SUCCEEDED"},
                {
                    "step": "step_a",
                    "entryId": "3",
                    "state": "STATE_SUCCEEDED",
                    "variableData": _variables("step_a", "A"),
                },
            ],
        }
        details, _ = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            [page],
            mock_auth_session_class,
            mock_exec_client_class,
        )
        step_ids = [e["step_id"] for e in details["step_entries"]]
        assert step_ids == ["user_input", "step_a"]

    @pytest.mark.anyio
    async def test_pagination_follows_next_page_token(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        pages = [
            {
                "stepEntries": [
                    {
                        "step": "step_a__iter_0",
                        "entryId": "1",
                        "state": "STATE_SUCCEEDED",
                        "variableData": _variables("step_a__iter_0", "A0"),
                    },
                ],
                "nextPageToken": "token-2",
            },
            {
                "stepEntries": [
                    {
                        "step": "step_a",
                        "entryId": "2",
                        "state": "STATE_SUCCEEDED",
                        "variableData": _variables("step_a", "A1"),
                    },
                ],
            },
        ]
        details, mock_session = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            pages,
            mock_auth_session_class,
            mock_exec_client_class,
        )

        assert mock_session.get.call_count == 2
        assert mock_session.get.call_args_list[0].kwargs["params"] == {}
        assert mock_session.get.call_args_list[1].kwargs["params"] == {
            "pageToken": "token-2",
        }
        entries = {e["step_id"]: e for e in details["step_entries"]}
        assert len(entries["step_a"]["history"]) == 2

    @pytest.mark.anyio
    async def test_pagination_stops_at_page_cap(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        """A never-ending `nextPageToken` must not loop forever."""
        pages = [
            {
                "stepEntries": [
                    {
                        "step": "step_a",
                        "state": "STATE_SUCCEEDED",
                        "variableData": _variables("step_a", f"A{i}"),
                    },
                ],
                "nextPageToken": f"token-{i}",
            }
            for i in range(MAX_STEP_ENTRY_PAGES + 5)
        ]
        _details, mock_session = await _run_details(
            workflow_service,
            mock_run_repo,
            _chained_workflow(),
            pages,
            mock_auth_session_class,
            mock_exec_client_class,
        )
        assert mock_session.get.call_count == MAX_STEP_ENTRY_PAGES

    @pytest.mark.anyio
    async def test_non_200_response_stops_pagination(
        self,
        mock_auth_session_class,
        mock_exec_client_class,
        workflow_service,
        mock_run_repo,
    ):
        mock_client = MagicMock()
        mock_exec_client_class.return_value = mock_client
        mock_client.get_execution.return_value = _mock_execution()

        response = MagicMock()
        response.status_code = 403
        response.text = "denied"
        mock_session = MagicMock()
        mock_session.get.return_value = response
        mock_auth_session_class.return_value = mock_session

        mock_run = MagicMock()
        mock_run.id = "e-1"
        mock_run.workflow_snapshot = _chained_workflow().model_dump(mode="json")
        mock_run.status = WorkflowRunStatusEnum.COMPLETED.value
        mock_run_repo.get_by_id.return_value = mock_run

        details = await workflow_service.get_execution_details(
            workflow_id="id-chain",
            execution_id="e-1",
        )

        assert mock_session.get.call_count == 1
        # Only the virtual user input entry survives.
        assert [e["step_id"] for e in details["step_entries"]] == [
            "user_input",
        ]
