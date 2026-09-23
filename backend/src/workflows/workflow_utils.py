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
"""Utility functions for workflows."""

import re
from typing import Any

from src.workflows.workflow_constants import MVP_ITERATION_SUFFIX


def build_iteration_step_name(
    step_id: str,
    iteration: int,
    total: int,
) -> str:
    """Builds the Cloud Workflows step name for one MVP iteration.

    MVP ONLY (temporary loop simulation). Cloud Workflows rejects duplicate
    step names, so every repeated copy but the last one gets a
    ``{step_id}__iter_{k}`` name. The LAST iteration keeps the original
    ``{step_id}`` name so downstream ``${step_id_result...}`` references keep
    working unchanged ("last iteration wins").

    Args:
        step_id: The logical step id from the workflow definition.
        iteration: Zero-based index of this copy.
        total: Total number of copies emitted for this step.

    Returns:
        The step name to emit in the generated YAML.
    """
    if total <= 1 or iteration >= total - 1:
        return step_id
    return f"{step_id}{MVP_ITERATION_SUFFIX}{iteration}"


def parse_iteration_step_name(emitted_name: str) -> tuple[str, int | None]:
    """Splits an emitted step name back into its base id and iteration.

    MVP ONLY (temporary loop simulation).

    Args:
        emitted_name: The step name as reported by Cloud Workflows.

    Returns:
        A ``(base_step_id, iteration)`` tuple. ``iteration`` is ``None`` when
        the name carries no ``__iter_k`` suffix (i.e. the final iteration, or
        a step emitted before this MVP existed).
    """
    if not emitted_name:
        return emitted_name, None
    base, separator, suffix = emitted_name.rpartition(MVP_ITERATION_SUFFIX)
    if separator and base and suffix.isdigit():
        return base, int(suffix)
    return emitted_name, None


def interpolate_prompt_variables(
    prompt: str,
    variables: dict[str, Any] | None = None,
    keep_unresolved: bool = False,
) -> str:
    """Interpolates <var_name> placeholders in prompt using provided variables.

    Args:
        prompt: The prompt template containing <var_name> placeholders.
        variables: Dictionary mapping variable names to their values.
        keep_unresolved: If True, placeholders with missing or None values
            remain unchanged (e.g. '<var_name>'). If False, placeholders with
            missing or None values are replaced with an empty string.

    Returns:
        The interpolated prompt string.
    """
    if not prompt:
        return prompt or ""

    vars_dict = variables or {}
    vars_lower = {str(k).lower(): v for k, v in vars_dict.items()}

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        val = vars_lower.get(var_name.lower())
        if val is None:
            return match.group(0) if keep_unresolved else ""
        if isinstance(val, dict):
            return str(val.get("generated_text") or val.get("text") or "")
        return str(val)

    return re.sub(r"<([a-zA-Z0-9_]+)>", replace_var, prompt)
