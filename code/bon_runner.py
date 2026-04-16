"""Best-of-N runner for SWT-Bench v1 (Option E).

SDK classes / functions used (mirroring run_infer.py pattern):
  - openhands.sdk.Agent                      (run_infer.py:253)
  - openhands.sdk.Conversation               (run_infer.py:272)
  - openhands.sdk.LLM                        (constructed from jina_test LLMClient config)
  - openhands.tools.preset.default.get_default_tools  (run_infer.py:254)
  - benchmarks.utils.litellm_proxy.build_eval_llm     (run_infer.py:231)
  - benchmarks.utils.fake_user_response.run_conversation_with_fake_user_response (run_infer.py:288)
  - benchmarks.utils.image_utils.create_docker_workspace (run_infer.py:178)

Temperature wiring:
  build_eval_llm returns an openhands LLM (with optional virtual key injected).
  We then call llm.model_copy(deep=True, update={"temperature": temperature}) so
  the base config is never mutated and each candidate gets its own isolated LLM.

Interface contract (consumed by runner.py _stage1_generate_tests):
  await bon_runner.run_single_candidate(
      instance=instance,       # dict with instance_id, repo, base_commit, problem_statement, ...
      prompt_path=prompt_path, # Path to .j2 template
      temperature=temperature, # float sampling temperature
      llm=llm,                 # jina_test.llm.LLMClient — model/api_key extracted from here
      agent_cfg=agent_cfg,     # jina_test.config.AgentConfig — max_iterations extracted
  ) -> dict

Return dict keys:
  instance_id, path_name, temperature, patch, test_files, is_viable, error
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from pydantic import SecretStr

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools

# benchmarks package imports
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.image_utils import create_docker_workspace
from benchmarks.utils.litellm_proxy import build_eval_llm
from openhands.sdk import get_logger

if TYPE_CHECKING:
    # jina_test types only needed for type hints; imported at runtime inside function
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Instruction rendering
# ---------------------------------------------------------------------------

def _render_instruction(
    instance: dict[str, Any],
    prompt_path: Path,
    workspace_path: str,
) -> str:
    """Render the Jinja2 prompt template for a given instance."""
    prompts_dir = str(prompt_path.parent)
    template_name = prompt_path.name
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    if "repo" in instance:
        workspace_dir_name = instance["repo"].split("/")[-1]
    elif "instance_id" in instance:
        workspace_dir_name = instance["instance_id"].replace("/", "_")
    else:
        workspace_dir_name = "workspace"

    test_instructions = ""
    if instance.get("test_cmd"):
        test_instructions = (
            f"\nThe test command to verify your implementation is:\n"
            f"```bash\n{instance['test_cmd']}\n```\n"
            "Make sure your implementation passes this test.\n"
        )

    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": None,
        "test_instructions": test_instructions,
    }
    return template.render(context)


# ---------------------------------------------------------------------------
# Extract test files from git diff
# ---------------------------------------------------------------------------

_TEST_FILE_RE = re.compile(
    r"^(?:\+\+\+ b/|--- a/)(.*(?:test[_s]|_test).*\.py|tests?/.*\.py)",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_test_files(patch: str) -> list[str]:
    """Return list of test file paths mentioned in the git diff."""
    if not patch:
        return []
    files: list[str] = []
    seen: set[str] = set()
    for m in _TEST_FILE_RE.finditer(patch):
        path = m.group(1)
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# Build openhands LLM from jina_test LLMClient
# ---------------------------------------------------------------------------

def _build_sdk_llm(jina_llm: Any) -> LLM:
    """Construct an openhands SDK LLM from a jina_test LLMClient.

    Uses the generator model and api_key from the jina_test client.
    Respects LLM_BASE_URL env var for proxy routing (same as benchmarks venv).
    """
    model = jina_llm.config.generator_model
    api_key = jina_llm.api_key
    base_url = os.getenv("LLM_BASE_URL") or None

    sdk_llm = LLM(
        model=model,
        api_key=SecretStr(api_key) if api_key else None,
        base_url=base_url,
    )
    return sdk_llm


# ---------------------------------------------------------------------------
# Core single-candidate runner (sync, runs in thread)
# ---------------------------------------------------------------------------

def _run_single_candidate_sync(
    instance: dict[str, Any],
    prompt_path: Path,
    temperature: float,
    sdk_llm_with_key: LLM,
    max_iterations: int,
) -> dict[str, Any]:
    """Synchronous implementation of run_single_candidate.

    Separated so it can be run via asyncio.to_thread without blocking the loop.
    """
    instance_id = instance["instance_id"]
    path_name = prompt_path.stem  # e.g. "direct", "snippet", "assertflip"

    result: dict[str, Any] = {
        "instance_id": instance_id,
        "path_name": path_name,
        "temperature": temperature,
        "patch": "",
        "test_files": [],
        "is_viable": False,
        "error": None,
    }

    workspace = None
    conversation = None
    try:
        # --- Build LLM with temperature ----------------------------------------
        llm = sdk_llm_with_key.model_copy(deep=True, update={"temperature": temperature})

        # --- Build tools -------------------------------------------------------
        tools = get_default_tools(enable_browser=False)

        # --- Create agent ------------------------------------------------------
        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        # --- Create Docker workspace -------------------------------------------
        # Derive Docker image from SWE-Bench official naming convention
        instance_id_clean = instance_id
        if "__" in instance_id_clean:
            repo_part, name_part = instance_id_clean.split("__", 1)
        else:
            repo_part = instance_id_clean
            name_part = instance_id_clean

        official_docker_image = (
            f"docker.io/swebench/sweb.eval.x86_64.{repo_part}_1776_{name_part}:latest".lower()
        )

        # Use EVAL_AGENT_SERVER_IMAGE env var if set, otherwise use a default
        from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
        build_target = "source-minimal"
        name_tag = official_docker_image.split("/")[-1]
        custom_tag = name_tag.split(":")[0]

        # SDK tag prefix: matches the acd5adc SDK SHA the baseline used.
        # benchmarks.utils.version.get_phased_image_tag_prefix() runs git
        # submodule status at import time, which fails unless CWD is the
        # benchmarks repo. Compute directly with the benchmarks repo as CWD.
        import os as _os
        import subprocess as _subprocess
        _sdk_prefix = _os.environ.get("SWT_BENCH_SDK_TAG_PREFIX")
        if not _sdk_prefix:
            try:
                _benchmarks_dir = _os.path.expanduser("~/benchmarks")
                _out = _subprocess.check_output(
                    ["git", "submodule", "status", "vendor/software-agent-sdk"],
                    cwd=_benchmarks_dir,
                    text=True,
                )
                _sdk_prefix = _out.strip().split()[0].lstrip("+-")[:7]
            except Exception:
                _sdk_prefix = "acd5adc"  # fallback to known baseline prefix
        agent_server_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:{_sdk_prefix}-{custom_tag}-{build_target}"
        )

        workspace = create_docker_workspace(
            agent_server_image=agent_server_image,
            base_image=official_docker_image,
            build_target=build_target,
        )

        # --- Set up repo in workspace ------------------------------------------
        repo_name = instance.get("repo", "").split("/")[-1] or "workspace"
        repo_path = f"/workspace/{repo_name}/"
        instance["repo_path"] = repo_path

        cp = workspace.execute_command(
            f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"
        )
        if cp.exit_code != 0:
            raise RuntimeError(f"cp /testbed failed: {cp.stderr}")

        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        if git_reset.exit_code != 0:
            raise RuntimeError(f"git reset failed: {git_reset.stderr}")

        # --- Create conversation -----------------------------------------------
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=max_iterations,
            delete_on_close=True,
        )

        # --- Render and run ----------------------------------------------------
        instruction = _render_instruction(
            instance=instance,
            prompt_path=prompt_path,
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        # --- Collect git patch -------------------------------------------------
        workspace.execute_command(f"cd {repo_path} ; git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'bon@openhands.dev' && "
            "git config --global user.name 'OpenHands BoN' && "
            "git commit --no-verify -m 'bon-patch'"
        )

        base_commit = instance.get("base_commit", "HEAD~1")
        diff_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
        )
        if diff_result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {diff_result.stderr}")

        patch = diff_result.stdout or ""
        test_files = _extract_test_files(patch)

        result["patch"] = patch
        result["test_files"] = test_files
        result["is_viable"] = bool(patch.strip())

    except Exception as exc:
        error_msg = str(exc)[:500]
        logger.warning(
            "[BoN] instance=%s path=%s temp=%.2f error: %s",
            instance_id,
            path_name,
            temperature,
            error_msg,
        )
        result["error"] = error_msg
    finally:
        # CRITICAL: tear down Docker workspace to prevent container leak.
        # Canonical pattern from benchmarks/utils/evaluation.py::_cleanup_workspace
        # (line 729+) which calls workspace.__exit__(None, None, None).
        # Without this, every BoN candidate leaks one agent-server container and
        # the VM OOMs within ~1 hour under concurrency.
        if workspace is not None:
            try:
                workspace.__exit__(None, None, None)
            except Exception as cleanup_exc:
                logger.warning(
                    "[BoN] workspace cleanup failed for instance=%s path=%s: %s",
                    instance_id,
                    path_name,
                    str(cleanup_exc)[:200],
                )

    return result


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BASE_DELAY = 30  # seconds


async def run_single_candidate(
    instance: dict[str, Any],
    prompt_path: Path,
    temperature: float,
    llm: Any,  # jina_test.llm.LLMClient
    agent_cfg: Any,  # jina_test.config.AgentConfig
) -> dict[str, Any]:
    """Run OpenHands SDK Agent once on a single instance with a specific prompt and temperature.

    This is the primary BoN entry point called by runner.py _stage1_generate_tests.
    Each call spins up a fresh Agent + Conversation + Docker workspace.

    Includes retry with exponential backoff for rate limit errors.

    Args:
        instance: SWT-Bench instance dict (instance_id, repo, base_commit, problem_statement, ...).
        prompt_path: Path to the .j2 Jinja2 template file.
        temperature: Sampling temperature for this candidate.
        llm: jina_test LLMClient — model name and api_key are extracted from it.
        agent_cfg: jina_test AgentConfig — max_iterations is extracted from it.

    Returns:
        dict with keys:
          - instance_id: str
          - path_name: str (stem of prompt_path, e.g. "direct")
          - temperature: float
          - patch: str (git diff; empty string if agent produced no changes)
          - test_files: list[str] (test file paths extracted from the patch)
          - is_viable: bool (True if patch is non-empty and no error)
          - error: str | None
    """
    import random

    # Build openhands SDK LLM from jina_test client config
    sdk_llm = _build_sdk_llm(llm)
    # Apply virtual key injection via build_eval_llm
    sdk_llm_with_key = build_eval_llm(sdk_llm)

    max_iterations = getattr(agent_cfg, "max_iterations", 80)

    instance_id = instance.get("instance_id", "?")
    path_name = prompt_path.stem

    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        result = await asyncio.to_thread(
            _run_single_candidate_sync,
            instance,
            prompt_path,
            temperature,
            sdk_llm_with_key,
            max_iterations,
        )

        # Check if failure was a rate limit error — retry with backoff
        error = result.get("error", "") or ""
        is_rate_limit = "RateLimitError" in error or "rate_limit" in error.lower()

        if is_rate_limit and attempt < _RATE_LIMIT_MAX_RETRIES:
            delay = _RATE_LIMIT_BASE_DELAY * (2 ** attempt) + random.uniform(0, 10)
            logger.warning(
                "[BoN] rate limit for instance=%s path=%s temp=%.2f, "
                "retrying in %.0fs (attempt %d/%d)",
                instance_id, path_name, temperature,
                delay, attempt + 1, _RATE_LIMIT_MAX_RETRIES,
            )
            await asyncio.sleep(delay)
            continue

        return result

    return result  # return last attempt's result even if it failed


# ---------------------------------------------------------------------------
# Legacy synchronous multi-path API (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def run_bon_path(
    instance: Any,
    workspace: RemoteWorkspace,
    prompt_path: str,
    temperature: float,
    metadata: Any,
) -> dict[str, Any]:
    """Synchronous single-path runner using a pre-created workspace.

    This legacy API accepts an EvalInstance + pre-prepared RemoteWorkspace
    (used by the Evaluation orchestrator pattern from run_infer.py).

    Args:
        instance: EvalInstance with .id and .data attributes.
        workspace: An already-prepared RemoteWorkspace.
        prompt_path: Absolute path string to a .j2 template.
        temperature: Sampling temperature.
        metadata: EvalMetadata with .llm, .max_iterations, .enable_delegation,
                  .enable_condenser, etc.

    Returns:
        dict with keys: prompt_path, temperature, git_patch, error, history, metrics
    """
    from benchmarks.utils.models import EvalInstance, EvalMetadata

    result: dict[str, Any] = {
        "prompt_path": prompt_path,
        "temperature": temperature,
        "git_patch": "",
        "error": None,
        "history": [],
        "metrics": {},
    }

    try:
        from openhands.sdk import Tool
        from openhands.sdk.context.condenser import LLMSummarizingCondenser

        base_llm = build_eval_llm(metadata.llm)
        llm = base_llm.model_copy(deep=True, update={"temperature": temperature})

        tools = get_default_tools(enable_browser=False)
        if getattr(metadata, "enable_delegation", False):
            from openhands.tools.delegate import DelegateTool
            tools.append(Tool(name=DelegateTool.name))

        condenser = None
        if getattr(metadata, "enable_condenser", False):
            condenser = LLMSummarizingCondenser(
                llm=build_eval_llm(metadata.llm, usage_id="condenser"),
                max_size=metadata.condenser_max_size,
                keep_first=metadata.condenser_keep_first,
            )

        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            condenser=condenser,
        )

        inst_data = instance.data if hasattr(instance, "data") else instance
        repo_path = f"/workspace/{inst_data['repo'].split('/')[-1]}/"
        inst_data["repo_path"] = repo_path

        cp = workspace.execute_command(
            f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"
        )
        if cp.exit_code != 0:
            raise RuntimeError(f"cp /testbed failed: {cp.stderr}")

        git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        if git_reset.exit_code != 0:
            raise RuntimeError(f"git reset failed: {git_reset.stderr}")

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=metadata.max_iterations,
            delete_on_close=False,
        )

        inst_id = instance.id if hasattr(instance, "id") else inst_data.get("instance_id", "?")
        instruction = _render_instruction(
            instance=inst_data,
            prompt_path=Path(prompt_path),
            workspace_path=workspace.working_dir,
        )
        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        workspace.execute_command(f"cd {repo_path} ; git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            "git config --global user.email 'bon@openhands.dev' && "
            "git config --global user.name 'OpenHands BoN' && "
            "git commit --no-verify -m 'bon-patch'"
        )

        base_commit = inst_data.get("base_commit", "HEAD~1")
        diff_result = workspace.execute_command(
            f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
        )
        if diff_result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {diff_result.stderr}")

        result["git_patch"] = diff_result.stdout or ""
        result["history"] = list(conversation.state.events)
        result["metrics"] = conversation.conversation_stats.get_combined_metrics()

    except Exception as exc:
        error_msg = str(exc)[:500]
        logger.warning(
            "run_bon_path failed for instance=%s prompt=%s temp=%.2f: %s",
            getattr(instance, "id", "?"),
            os.path.basename(prompt_path),
            temperature,
            error_msg,
        )
        result["error"] = error_msg

    return result


def run_bon(
    instance: Any,
    workspace: RemoteWorkspace,
    prompt_paths: list[str],
    temperatures: list[float],
    metadata: Any,
) -> list[dict[str, Any]]:
    """Run Agent+Conversation for every (prompt_path, temperature) combination.

    Legacy synchronous multi-path API using a pre-created workspace.

    Args:
        instance: EvalInstance with .id and .data attributes.
        workspace: An already-prepared RemoteWorkspace.
        prompt_paths: List of absolute .j2 template path strings.
        temperatures: List of sampling temperatures.
        metadata: EvalMetadata.

    Returns:
        List of result dicts (one per combination).
    """
    results: list[dict[str, Any]] = []
    total = len(prompt_paths) * len(temperatures)
    idx = 0
    for prompt_path in prompt_paths:
        for temperature in temperatures:
            idx += 1
            logger.info(
                "[BoN %d/%d] instance=%s prompt=%s temp=%.2f",
                idx,
                total,
                getattr(instance, "id", "?"),
                os.path.basename(prompt_path),
                temperature,
            )
            res = run_bon_path(
                instance=instance,
                workspace=workspace,
                prompt_path=prompt_path,
                temperature=temperature,
                metadata=metadata,
            )
            results.append(res)
    return results
