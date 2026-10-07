"""End-to-end Meta-Harness search for ALFWorld.

The proposer is a coding agent with filesystem access. Evaluation stays
outside the proposer, and each accepted candidate is archived with source,
proposal trace, rollout traces, score, and lifecycle state.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
import tomllib
import urllib.request
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROJECT = REPO.parent
AGENTBENCH = PROJECT / "Life-Harness" / "AgentBench"
BASE = ROOT / "base_harness.py"
SKILL = ROOT / ".claude" / "skills" / "meta-harness-alfworld" / "SKILL.md"
AUDIT_TAG = "[ALFWORLD_HARNESS_AUDIT_V1]\n"
MAX_PROPOSER_ATTEMPTS = 3
CORE_PROMPT_GUIDANCE = (
    "Before every action, silently re-read the original task. Keep the exact target "
    "object type, required count and state, destination, searched locations, inventory, "
    "and completed placements. Operate only on target objects. Follow find, take, any "
    "required clean, heat, cool, or examine operation, then go to the destination and put. "
    "For multiple objects, complete one before finding another instance. Never undo a "
    "completed placement. Choose exactly one action verbatim from the latest AVAILABLE "
    "ACTIONS. Return exactly one take_action tool call and no prose."
)


def prompt_harness_source(guidance: str) -> str:
    """Render a fixed harness whose only intervention is model-facing guidance."""
    return f'''"""Trajectory-derived prompt harness for ALFWorld."""
from copy import deepcopy
import json


class AgentHarness:
    def __init__(self, session, tools, max_steps):
        self.session = session
        self.max_steps = max_steps
        self.turn = 0
        self.trace = []
        self.guidance = {guidance!r}
        self.task_goal = ""
        self.session.set_tools(deepcopy(tools))
        self.session.set_full_history(True)

    def inject(self, item):
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str) and "Your task is to:" in content:
                self.task_goal = content.split("Your task is to:", 1)[1].split(
                    "AVAILABLE ACTIONS:", 1
                )[0].strip()
        self.session.inject(item)

    def prepare_history(self, history, remaining):
        return deepcopy(history)

    def process_output(self, output, history, remaining):
        return output

    @staticmethod
    def get_action(message):
        calls = message.get("tool_calls") or []
        if not calls:
            return None
        arguments = calls[0].get("function", {{}}).get("arguments", {{}})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return arguments.get("action") if isinstance(arguments, dict) else None

    @staticmethod
    def set_action(message, action):
        updated = deepcopy(message)
        call = (updated.get("tool_calls") or [])[0]
        arguments = call.get("function", {{}}).get("arguments", {{}})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
            arguments["action"] = action
            call["function"]["arguments"] = json.dumps(arguments)
        else:
            arguments["action"] = action
        return updated

    def sync_action(self):
        self.turn += 1
        remaining = max(0, self.max_steps - self.turn + 1)
        if self.turn == 1 or (self.turn - 1) % 5 == 0:
            reminder = self.guidance
            if self.task_goal:
                reminder += " Original task goal: " + self.task_goal
            self.session.inject({{"role": "system", "content": reminder}})
            self.trace.append({{"turn": self.turn, "remaining": remaining}})
        return self.session.sync_action()
'''


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def completed_evaluation(output: Path, candidate: Path, expected: int) -> dict | None:
    lifecycle = output / "lifecycle.json"
    if not lifecycle.exists():
        return None
    try:
        state = json.loads(lifecycle.read_text())
        verified = verify_evaluation(output, candidate, expected)
        if state.get("status") != "complete":
            verified["process_warning"] = state.get("error", "non-zero evaluator exit")
        return verified
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return None


def archive_partial(path: Path, runroot: Path) -> Path | None:
    if not path.exists():
        return None
    relative = "__".join(path.relative_to(runroot).parts)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = runroot / "interrupted" / f"{relative}-{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = 2
    while destination.exists():
        destination = runroot / "interrupted" / f"{relative}-{stamp}-{suffix}"
        suffix += 1
    shutil.move(str(path), str(destination))
    return destination


def trajectory_entities(runroot: Path) -> set[str]:
    """Extract concrete numbered environment entities from archived rollouts."""
    import re
    entities = set()
    pattern = re.compile(r"\b([a-z][a-z_]*(?:\s+[a-z][a-z_]*)?)\s+\d+\b")
    stop = {"a", "an", "the", "to", "on", "in", "at", "with", "and", "from",
            "go", "take", "put", "move", "open", "close", "check", "checked",
            "clean", "cool", "heat", "examine"}
    for runs_file in runroot.rglob("runs.jsonl"):
        for line in runs_file.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = (((row.get("output") or {}).get("result") or {})
                        .get("openai_messages") or [])
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    for match in pattern.finditer(content.lower()):
                        words = match.group(1).split()
                        entities.update(word for word in words if word not in stop)
    return entities


def validate_candidate(path: Path, forbidden_entities: set[str] | None = None) -> tuple[bool, str]:
    """Import and exercise the standalone AgentHarness interface cheaply."""
    try:
        import ast
        source = path.read_text()
        tree = ast.parse(source)
        if any(isinstance(node, ast.ImportFrom) and node.module == "alfworld"
               for node in ast.walk(tree)):
            return False, "do not import AgentHarness from alfworld; define it directly"
        if not any(isinstance(node, ast.ClassDef) and node.name == "AgentHarness"
                   for node in tree.body):
            return False, "candidate must define a top-level class named AgentHarness"
        candidate_class = next(node for node in tree.body
                               if isinstance(node, ast.ClassDef) and node.name == "AgentHarness")
        base_tree = ast.parse(BASE.read_text())
        base_class = next(node for node in base_tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "AgentHarness")
        candidate_methods = {node.name: ast.dump(node, include_attributes=False)
                             for node in candidate_class.body
                             if isinstance(node, ast.FunctionDef)}
        base_methods = {node.name: ast.dump(node, include_attributes=False)
                        for node in base_class.body
                        if isinstance(node, ast.FunctionDef)}
        if candidate_methods == base_methods:
            return False, "candidate is semantically identical to the no-op seed"
        import re
        numbered_entity = re.compile(r"\b[a-z][a-z_]*(?:\s+[a-z][a-z_]*)?\s+\d+\b")
        leaked = sorted({match.group(0) for node in ast.walk(tree)
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)
                         for match in numbered_entity.finditer(node.value.lower())})
        if leaked:
            return False, f"task-specific numbered entities are forbidden: {leaked}"
        if forbidden_entities:
            copied = sorted(entity for entity in forbidden_entities
                            if len(entity) >= 3 and
                            re.search(rf"\b{re.escape(entity)}\b", source.lower()))
            if copied:
                return False, f"trajectory-specific entities are forbidden: {copied}"
        spec = importlib.util.spec_from_file_location(
            f"alfworld_candidate_{time.time_ns()}", path
        )
        if spec is None or spec.loader is None:
            return False, "cannot create import spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        harness_type = getattr(module, "AgentHarness", None)
        if not isinstance(harness_type, type):
            return False, "missing AgentHarness class"
        from types import SimpleNamespace
        class FakeSession:
            def __init__(self):
                self.history = [
                    {"role": "system", "content": "Solve the household task."},
                    {"role": "user", "content": (
                        "Place the requested objects. AVAILABLE ACTIONS: look\\ninventory"
                    )},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "prior", "type": "function", "function": {
                            "name": "take_action", "arguments": '{"action":"look"}'
                        }
                    }]},
                    {"role": "tool", "content": (
                        "You inspect the room. AVAILABLE ACTIONS: look\\ninventory"
                    )},
                ]
                self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))
                self.covered_histories = []
            def set_tools(self, tools): self.tools = tools
            def set_full_history(self, enabled): self.full_history = enabled
            def inject(self, item):
                if isinstance(item, dict):
                    if "role" not in item:
                        raise TypeError("injected messages must include role")
                elif not hasattr(item, "reward"):
                    raise TypeError("injected items must be messages or reward records")
                self.history.append(deepcopy(item))
            def cover(self, history):
                if not isinstance(history, list) or any(
                    not isinstance(item, dict) or "role" not in item
                    for item in history
                ):
                    raise TypeError("history must contain OpenAI message dictionaries")
                self.covered_histories.append(deepcopy(history))
                self.history = deepcopy(history)
            def sync_action(self):
                return SimpleNamespace(messages=[{"role": "assistant", "content": None,
                    "tool_calls": [{"id": "test", "type": "function",
                                    "function": {"name": "take_action",
                                                 "arguments": '{"action":"inventory"}'}}]}])
        tools = [{"type": "function", "function": {
            "name": "take_action", "description": "Take an action.",
            "parameters": {"type": "object"},
        }}]
        harness = harness_type(FakeSession(), deepcopy(tools), 10)
        initial_system_messages = sum(
            item.get("role") == "system" for item in harness.session.history
            if isinstance(item, dict)
        )
        for method in ("inject", "sync_action"):
            if not callable(getattr(harness, method, None)):
                return False, f"AgentHarness must define {method}"
        harness.inject({"role": "user", "content": "Observation."})
        # Exercise stateful behavior, not just the first turn. Loop breakers and
        # history transforms commonly activate only after repeated actions.
        synthetic_actions = []
        for _ in range(7):
            output = harness.sync_action()
            if not hasattr(output, "messages"):
                return False, "sync_action must return an object with messages"
            if len(output.messages) != 1:
                return False, (
                    "candidate must transform the existing assistant output, not append "
                    "additional assistant messages or tool calls"
                )
            for message in output.messages:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    return False, "model outputs must be assistant message dictionaries"
                calls = message.get("tool_calls") or []
                if len(calls) != 1:
                    return False, "each model turn must preserve exactly one take_action call"
                for call in calls:
                    if not call.get("id") or call.get("type") != "function":
                        return False, "tool calls must preserve id and type='function'"
                    function = call.get("function") or {}
                    if function.get("name") != "take_action":
                        return False, "candidate may only emit the configured take_action tool"
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        return False, "tool-call arguments must be a JSON string"
                    parsed = json.loads(arguments)
                    action = parsed.get("action")
                    if not isinstance(action, str):
                        return False, "take_action requires a string action"
                    if action.strip().lower() == "stop":
                        return False, "'stop' is not an ALFWorld admissible action"
                    if action not in {
                        "look", "inventory",
                    }:
                        return False, (
                            "synthetic action rewrites may only preserve the supplied action "
                            "or use universally admissible look/inventory; never invent "
                            "placeholder object actions"
                        )
                    synthetic_actions.append(action)
        if not synthetic_actions or synthetic_actions[0] != "inventory":
            return False, (
                "candidate overrides a valid first action unconditionally; recovery "
                "must trigger only from explicit failure or repetition evidence"
            )
        for item in harness.session.history:
            if not isinstance(item, dict):
                if hasattr(item, "reward"):
                    continue
                return False, "session history contains a non-message item"
            role = item.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                return False, f"session history contains invalid role: {role!r}"
            if role == "assistant" and not (
                isinstance(item.get("content"), str) or item.get("tool_calls")
            ):
                return False, "assistant messages require content or tool_calls"
            for call in item.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") != "take_action":
                    return False, (
                        "candidate may only emit the configured take_action tool"
                    )
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    return False, "tool-call arguments must be a JSON string"
                parsed = json.loads(arguments)
                if not isinstance(parsed.get("action"), str):
                    return False, "take_action requires a string action"
        trace = getattr(harness, "trace", [])
        json.dumps(trace)
        if not trace:
            return False, (
                "candidate never records an intervention under the interface "
                "contract test; repair the prompt/history mechanism or self.trace accounting"
            )
        action_intervention = any(
            action != "inventory" for action in synthetic_actions[1:]
        )
        history_intervention = any(
            harness.session.covered_histories[index]
            != harness.session.covered_histories[index + 1]
            for index in range(0, len(harness.session.covered_histories) - 1, 2)
        )
        persistent_history_intervention = sum(
            item.get("role") == "system" for item in harness.session.history
            if isinstance(item, dict)
        ) > initial_system_messages
        if not action_intervention and not history_intervention and not persistent_history_intervention:
            return False, (
                "candidate records self.trace without changing the model-facing history "
                "or emitted action"
            )
        return True, "ok"
    except AttributeError as exc:
        if "'str' object has no attribute 'get'" in str(exc):
            return False, (
                "OpenAI tool-call function.arguments is a JSON string; call "
                "json.loads(arguments) before reading action, or use the seed's get_action helper"
            )
        return False, f"AttributeError: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def profile_for(phase: str) -> str:
    return {
        "candidate-smoke": "alfworld-candidate-smoke",
        "candidate-evolution": "alfworld-candidate-evolution",
        "heldout": "alfworld-heldout",
    }[phase]


def parse_audit(row: dict) -> dict | None:
    result = ((row.get("output") or {}).get("result") or {})
    for message in reversed(result.get("openai_messages") or []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.startswith(AUDIT_TAG):
            return json.loads(content[len(AUDIT_TAG):])
    return None


def verify_evaluation(output: Path, candidate: Path, expected: int) -> dict:
    overall_files = list(output.rglob("overall.json"))
    runs_files = list(output.rglob("runs.jsonl"))
    pairs = []
    for overall_path in overall_files:
        runs_path = overall_path.parent / "runs.jsonl"
        if runs_path.is_file():
            try:
                payload = json.loads(overall_path.read_text())
                total = int(payload.get("custom", {}).get("overall", {}).get("total", -1))
                rows = [line for line in runs_path.read_text().splitlines() if line.strip()]
                if total == expected and len(rows) == expected:
                    pairs.append((max(overall_path.stat().st_mtime, runs_path.stat().st_mtime), overall_path, runs_path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    if pairs:
        _, selected_overall, selected_runs = max(pairs, key=lambda item: item[0])
        overall_files = [selected_overall]
        runs_files = [selected_runs]
    if len(overall_files) != 1 or len(runs_files) != 1:
        raise RuntimeError(
            f"expected one overall.json and runs.jsonl, found "
            f"{len(overall_files)} and {len(runs_files)}"
        )
    overall = json.loads(overall_files[0].read_text())
    total = int(overall.get("custom", {}).get("overall", {}).get("total", -1))
    if total != expected:
        raise RuntimeError(f"partial evaluation: expected {expected}, got {total}")
    rows = [json.loads(line) for line in runs_files[0].read_text().splitlines() if line]
    if len(rows) != expected:
        raise RuntimeError(f"partial runs.jsonl: expected {expected}, got {len(rows)}")
    expected_source = str(candidate.resolve())
    active_harness_trials = 0
    intervention_events = 0
    status_counts = {}
    for index, row in enumerate(rows):
        if row.get("error") or not row.get("output"):
            raise RuntimeError(f"trial {index} has an execution error")
        audit = parse_audit(row)
        if audit is None:
            raise RuntimeError(f"trial {index} has no candidate audit marker")
        info = audit.get("harness_info") or {}
        if info.get("mode") != "candidate" or info.get("source") != expected_source:
            raise RuntimeError(
                f"trial {index} ran {info.get('mode')} from {info.get('source')}, "
                f"not candidate {expected_source}"
            )
        trace = audit.get("harness_trace") or []
        status = (row.get("output") or {}).get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        if isinstance(trace, dict):
            events = sum(len(value or []) for value in trace.values())
        elif isinstance(trace, list):
            events = len(trace)
        else:
            raise RuntimeError(f"trial {index} has invalid harness trace type")
        intervention_events += events
        active_harness_trials += int(events > 0)
    custom = overall["custom"]["overall"]
    return {
        "total": total,
        "pass": int(custom["pass"]),
        "wrong": int(custom["wrong"]),
        "pass_rate": float(custom["success_rate"]),
        "active_harness_trials": active_harness_trials,
        "intervention_events": intervention_events,
        "status_counts": status_counts,
        "overall": str(overall_files[0]),
        "runs": str(runs_files[0]),
    }


def evaluate(candidate: Path, phase: str, output: Path, expected: int, nonce: int) -> dict:
    """Run candidate eval, resuming a partial runs.jsonl by missing index."""
    output.mkdir(parents=True, exist_ok=True)
    complete_pairs = []
    for overall_path in output.rglob("overall.json"):
        runs_path = overall_path.with_name("runs.jsonl")
        try:
            payload = json.loads(overall_path.read_text())
            total = int(payload.get("custom", {}).get("overall", {}).get("total", -1))
            row_count = sum(1 for line in runs_path.read_text().splitlines() if line.strip())
            if total == expected and row_count == expected:
                complete_pairs.append(overall_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if complete_pairs:
        return verify_evaluation(output, candidate, expected)
    partial_rows = []
    partial_files = list(output.rglob("runs.jsonl"))
    if partial_files:
        partial_files.sort(key=lambda path: len(path.read_text(errors="replace").splitlines()), reverse=True)
        partial_rows = [json.loads(line) for line in partial_files[0].read_text().splitlines() if line.strip()]
        partial_rows = {int(row["index"]): row for row in partial_rows}.values()
        partial_rows = list(partial_rows)
    missing = expected - len(partial_rows)
    run_output = output
    if partial_rows and missing > 0:
        archive = output.parent / (output.name + ".partial")
        if archive.exists():
            archive = output.parent / (output.name + f".partial-{int(time.time())}")
        shutil.move(str(output), str(archive))
        output.mkdir(parents=True, exist_ok=True)
        run_output = output / "resume"
        run_output.mkdir(parents=True, exist_ok=True)
        agent_name = os.environ.get("QWEN_AGENT_NAME", "qwen3-8b")
        task_name = "alfworld-heldout" if phase == "heldout" else "alfworld-candidate-evolution"
        seed_dir = run_output / agent_name / task_name
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "runs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sorted(partial_rows, key=lambda row: int(row["index"])))
        )
    controller_port = 18000 + nonce % 1000
    worker_port = controller_port + 1
    tmpdir = Path(f"/tmp/alfworld-meta-{os.getpid()}-{nonce}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    missing_indices = None
    if partial_rows:
        missing_indices = sorted(set(range(expected)) - {int(row["index"]) for row in partial_rows})
        source = AGENTBENCH / "data/alfworld/alfworld_evolution.json" if phase != "heldout" else AGENTBENCH / "data/alfworld/alfworld_heldout.json"
        split = json.loads(source.read_text())
        files = [item for group in split.values() for item in group]
        subset = tmpdir / "missing_gamefiles.json"
        subset.write_text(json.dumps([files[i] for i in missing_indices]))
    else:
        subset = None
    env.update({
        "AGENTRL_CONTROLLER_PORT": str(controller_port),
        "AGENTRL_WORKER_PORT": str(worker_port),
        "ALFWORLD_HARNESS_FILE": str(candidate.resolve()),
        "ALFWORLD_TASK_LIMIT": str(missing if partial_rows else expected),
        "TMPDIR": str(tmpdir), "TEMP": str(tmpdir), "TMP": str(tmpdir),
    })
    if subset is not None:
        env["ALFWORLD_GAMEFILES_JSON"] = str(subset)
    controller_log = (run_output / "controller.log").open("w")
    worker_log = (run_output / "worker.log").open("w")
    controller = subprocess.Popen(
        [str(AGENTBENCH / ".native/controller/agentrl"), "controller",
         "--host", "127.0.0.1", "--port", str(controller_port),
         "--dashboard=false", "--long-timeout"],
        cwd=AGENTBENCH, env=env, stdout=controller_log, stderr=subprocess.STDOUT,
    )
    worker = subprocess.Popen(
        ["bash", "scripts/native/worker.sh", "alfworld", profile_for(phase)],
        cwd=AGENTBENCH, env=env, stdout=worker_log, stderr=subprocess.STDOUT,
    )
    command = [str(AGENTBENCH / "scripts/native/alfworld_evaluate_candidate.sh"),
               phase, str(run_output.resolve()), str(missing if partial_rows else expected)]
    lifecycle = {"candidate": str(candidate.resolve()), "phase": phase,
                 "expected_tasks": expected, "command": command,
                 "started_at": time.time()}
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if controller.poll() is not None or worker.poll() is not None:
                raise RuntimeError("controller or worker exited during startup")
            probe = subprocess.run(
                ["curl", "-sf", f"http://127.0.0.1:{worker_port}/api/get_sessions"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError("worker readiness timeout")
        proc = subprocess.run(command, cwd=AGENTBENCH, env=env, check=False)
        if partial_rows:
            new_runs = list(run_output.glob("*/*/runs.jsonl"))
            new_overall = list(run_output.glob("*/*/overall.json"))
            if len(new_runs) != 1 or len(new_overall) != 1:
                raise RuntimeError("resume did not produce exactly one partial runs.jsonl and overall.json")
            rows = {int(row["index"]): row for row in partial_rows}
            for line in new_runs[0].read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    source_index = int(row["index"])
                    if source_index in missing_indices:
                        target = source_index
                    elif 0 <= source_index < len(missing_indices):
                        target = missing_indices[source_index]
                    else:
                        continue
                    row["index"] = target
                    rows[target] = row
            if set(rows) != set(range(expected)):
                raise RuntimeError(f"resume index set incomplete: {sorted(set(range(expected)) - set(rows))[:10]}")
            new_runs[0].write_text("".join(json.dumps(rows[i]) + "\n" for i in range(expected)))
            overall = json.loads(new_overall[0].read_text())
            passed = sum(int((row.get("output") or {}).get("result", {}).get("reward", 0) > 0) for row in rows.values())
            overall["total"] = expected
            overall.setdefault("custom", {}).setdefault("overall", {}).update(
                {"total": expected, "pass": passed, "wrong": expected - passed, "success_rate": passed / expected}
            )
            new_overall[0].write_text(json.dumps(overall, indent=2) + "\n")
        lifecycle["result"] = verify_evaluation(output, candidate, expected)
        if proc.returncode:
            lifecycle["result"]["process_warning"] = (
                f"assigner exited {proc.returncode} after writing complete verified artifacts"
            )
        lifecycle["status"] = "complete"
        return lifecycle["result"]
    except Exception as exc:
        lifecycle["status"] = "failed"
        lifecycle["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        lifecycle["finished_at"] = time.time()
        atomic_json(output / "lifecycle.json", lifecycle)
        for process in (worker, controller):
            process.terminate()
        for process in (worker, controller):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        controller_log.close()
        worker_log.close()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path is outside proposer archive: {path}")
    return resolved


def _proposer_tools() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "list_files",
            "description": "List files in the Meta-Harness run archive.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Archive-relative directory"}
            }},
        }},
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a line range from a text file in the run archive.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 200},
            }, "required": ["path"]},
        }},
        {"type": "function", "function": {
            "name": "search_files",
            "description": "Regex-search text files in the run archive.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Archive-relative directory"},
            }, "required": ["pattern"]},
        }},
        {"type": "function", "function": {
            "name": "read_trajectory",
            "description": "Read one raw runs.jsonl trajectory with its task result and recent model/environment messages.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "line_number": {"type": "integer", "minimum": 1},
            }, "required": ["path", "line_number"]},
        }},
        {"type": "function", "function": {
            "name": "submit_candidate",
            "description": "Submit trajectory-derived guidance; the orchestrator renders a fixed prompt-only AgentHarness.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"},
                "guidance": {"type": "string", "description": "General model-facing ALFWorld policy inferred from successful and failed trajectories."},
                "hypothesis": {"type": "string"},
                "changes": {"type": "string"},
                "expected_effect": {"type": "string"},
                "failure_analysis": {"type": "string", "description": "Why prior candidates failed and how this mechanism differs."},
                "trigger_and_effect": {"type": "string", "description": "Exact runtime trigger, message field read, and observable intervention."},
                "self_check": {"type": "string", "description": "Why every emitted message/tool call is schema-valid and generally admissible."},
            }, "required": ["name", "guidance", "hypothesis", "changes", "expected_effect",
                              "failure_analysis", "trigger_and_effect", "self_check"]},
        }},
    ]


def _run_proposer_tool(name: str, arguments: dict, runroot: Path,
                       workspace: Path) -> str:
    relative = arguments.get("path") or "."
    target = _inside(runroot / relative, runroot)
    if name == "list_files":
        if not target.is_dir():
            return "ERROR: directory not found"
        files = [str(path.relative_to(runroot)) for path in target.rglob("*")
                 if path.is_file()]
        return "\n".join(sorted(files)[:2000])[:6000]
    if name == "read_file":
        if not target.is_file():
            return "ERROR: file not found"
        start = max(1, int(arguments.get("start_line", 1)))
        count = min(200, max(1, int(arguments.get("max_lines", 120))))
        lines = target.read_text(errors="replace").splitlines()
        selected = lines[start - 1:start - 1 + count]
        return "\n".join(f"{start + i}: {line}" for i, line in enumerate(selected))[:6000]
    if name == "search_files":
        import re
        pattern = re.compile(str(arguments["pattern"]))
        base = target if target.is_dir() else target.parent
        hits = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".json", ".jsonl", ".toml", ".txt", ".md"}:
                continue
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path.relative_to(runroot)}:{number}:{line[:500]}")
                    if len(hits) >= 100:
                        return "\n".join(hits)[:6000]
        return "\n".join(hits) or "NO_MATCHES"
    if name == "read_trajectory":
        if not target.is_file() or target.name != "runs.jsonl":
            return "ERROR: path must name a runs.jsonl file"
        number = int(arguments["line_number"])
        lines = target.read_text(errors="replace").splitlines()
        if number < 1 or number > len(lines):
            return f"ERROR: line_number must be between 1 and {len(lines)}"
        row = json.loads(lines[number - 1])
        result = ((row.get("output") or {}).get("result") or {})
        task = None
        steps = []
        pending_action = None
        for item in result.get("openai_messages") or []:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.startswith(AUDIT_TAG):
                continue
            role = item.get("role")
            if role == "system":
                continue
            if role == "user" and task is None and isinstance(content, str):
                task = content.partition("AVAILABLE ACTIONS:")[0].strip()
                continue
            if role == "assistant":
                calls = item.get("tool_calls") or []
                action = None
                if calls:
                    raw = (calls[0].get("function") or {}).get("arguments")
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        action = parsed.get("action") if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        action = None
                if action is None and isinstance(content, str) and content.strip():
                    steps.append(f"assistant text without tool -> {content.strip()[:120]}")
                else:
                    pending_action = action
                continue
            if isinstance(content, str) and role == "tool":
                observation = content.partition("AVAILABLE ACTIONS:")[0].strip()[:120]
                steps.append(f"{pending_action} -> {observation}")
                pending_action = None
                continue
        compressed_steps = []
        index = 0
        while index < len(steps):
            best_length = 0
            best_repeats = 1
            for length in range(1, min(12, (len(steps) - index) // 2) + 1):
                repeats = 1
                block = steps[index:index + length]
                while (
                    index + (repeats + 1) * length <= len(steps)
                    and steps[index + repeats * length:index + (repeats + 1) * length]
                    == block
                ):
                    repeats += 1
                if repeats > 1 and length * repeats > best_length * best_repeats:
                    best_length, best_repeats = length, repeats
            if best_repeats > 1:
                compressed_steps.append({
                    "repeat": best_repeats,
                    "steps": steps[index:index + best_length],
                })
                index += best_length * best_repeats
            else:
                compressed_steps.append(steps[index])
                index += 1
        compact = {
            "index": row.get("index"), "error": row.get("error"),
            "status": (row.get("output") or {}).get("status"),
            "reward": result.get("reward"), "metrics": result.get("metrics"),
            "task": task,
            "steps": compressed_steps,
        }
        return json.dumps(compact, ensure_ascii=False)
    if name == "submit_candidate":
        required = {"name", "guidance", "hypothesis", "changes", "expected_effect",
                    "failure_analysis", "trigger_and_effect", "self_check"}
        missing = sorted(required - set(arguments))
        if missing:
            return f"ERROR: submit_candidate is missing required fields: {missing}"
        candidate = workspace / f"candidate-{workspace.name.rsplit('-', 1)[-1]}.py"
        proposed_guidance = str(arguments["guidance"]).strip()
        if not proposed_guidance:
            return "ERROR: guidance must not be empty"
        if len(proposed_guidance.split()) > 80:
            return "ERROR: proposed guidance exceeds the 80-word limit"
        prohibited = [
            phrase for phrase in ("describe your", "list all", "explain your")
            if phrase in proposed_guidance.lower()
        ]
        if prohibited:
            return f"ERROR: guidance invites prose output: {prohibited}"
        guidance = CORE_PROMPT_GUIDANCE + " " + proposed_guidance
        arguments = {**arguments, "guidance": guidance}
        if len(guidance.split()) > 160:
            return "ERROR: combined guidance exceeds the 160-word limit"
        candidate.write_text(prompt_harness_source(guidance))
        valid, reason = validate_candidate(candidate, trajectory_entities(runroot))
        if not valid:
            return f"ERROR: candidate validation failed: {reason}; revise and submit again"
        import ast
        candidate_tree = ast.parse(candidate.read_text())
        candidate_class = next(
            node for node in candidate_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentHarness"
        )
        signature = ast.dump(candidate_class, include_attributes=False)
        for prior in runroot.rglob("candidate-*.py"):
            if prior.resolve() == candidate.resolve():
                continue
            try:
                prior_tree = ast.parse(prior.read_text())
                prior_class = next(
                    node for node in prior_tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "AgentHarness"
                )
            except (OSError, SyntaxError, StopIteration):
                continue
            if ast.dump(prior_class, include_attributes=False) == signature:
                return (
                    "ERROR: candidate semantically duplicates prior candidate "
                    f"{prior.relative_to(runroot)}; implement a different mechanism"
                )
        metadata = {key: arguments[key] for key in (
            "name", "guidance", "hypothesis", "changes", "expected_effect",
            "failure_analysis", "trigger_and_effect", "self_check",
        )}
        metadata["path"] = str(candidate.relative_to(runroot))
        atomic_json(workspace / "pending_eval.json", {"candidates": [metadata]})
        return f"SUBMITTED {candidate.relative_to(runroot)}"
    return f"ERROR: unknown tool {name}"


def _chat_completion(api_base: str, model: str, messages: list[dict],
                     max_tokens: int, timeout: int,
                     temperature: float, forced_tool: str | None = None) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "tools": _proposer_tools(),
        "tool_choice": ({"type": "function", "function": {"name": forced_tool}}
                        if forced_tool else "auto"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": os.environ.get("SELF_HARNESS_ENABLE_THINKING", "true").lower() == "true"
        },
    }).encode()
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]


def _compact_proposer_messages(messages: list[dict]) -> None:
    """Bound API context while retaining the full transcript on disk."""
    call_names = {
        call.get("id"): (call.get("function") or {}).get("name")
        for item in messages if item.get("role") == "assistant"
        for call in item.get("tool_calls") or []
    }
    assistant_indices = [i for i, item in enumerate(messages)
                         if item.get("role") == "assistant"]
    newest = assistant_indices[-1] if assistant_indices else -1
    for index, item in enumerate(messages):
        if (item.get("role") == "tool" and index < newest
                and call_names.get(item.get("tool_call_id")) != "read_trajectory"):
            item["content"] = "[Earlier tool output omitted; re-read the file if needed.]"
        if item.get("role") == "assistant":
            item.pop("reasoning_content", None)
            if isinstance(item.get("content"), str):
                item["content"] = item["content"][:500]
            if index < newest:
                for call in item.get("tool_calls") or []:
                    function = call.get("function") or {}
                    if function.get("name") != "read_trajectory":
                        function["arguments"] = "{}"


def _proposer_failure_context(runroot: Path, records: list[dict]) -> list[dict]:
    """Embed recent failure causes and source so repairs do not depend on rediscovery."""
    context = []
    for record in records[-MAX_PROPOSER_ATTEMPTS:]:
        item = {key: record.get(key) for key in (
            "iteration", "attempt", "outcome", "error", "hypothesis", "changes",
            "archived_workspace",
        ) if record.get(key) is not None}
        smoke = record.get("smoke") or {}
        if smoke:
            item["smoke"] = {key: smoke.get(key) for key in (
                "total", "pass", "active_harness_trials", "intervention_events",
            )}
        archive = record.get("archived_workspace")
        if archive:
            archive_path = _inside(runroot / archive, runroot)
            candidates = sorted(archive_path.glob("candidate-*.py"))
            if candidates:
                item["failed_candidate_path"] = str(
                    candidates[0].relative_to(runroot.resolve())
                )
        context.append(item)
    return context


def propose(runroot: Path, iteration: int, attempt: int, failure_feedback: list[dict],
            model: str, timeout: int,
            max_tokens: int, max_tool_turns: int, min_trajectories: int,
            temperature: float) -> list[dict]:
    workspace = runroot / f"iteration-{iteration:02d}"
    workspace.mkdir(parents=True, exist_ok=False)
    pending = workspace / "pending_eval.json"
    prompt = (
        f"Run ALFWorld Meta-Harness iteration {iteration}, repair attempt {attempt} of "
        f"{MAX_PROPOSER_ATTEMPTS}. The complete search archive is "
        f"{runroot}. The only seed inputs are under {runroot / 'inputs'}. Before writing code, "
        f"list the complete archive and use read_trajectory on successful and failed baseline rows "
        f"in runs.jsonl. Each trajectory contains its complete compact action-observation chain; "
        f"compare where successful runs preserve the task goal against where failed runs abandon "
        f"or forget it. Then read prior candidate source, frontier, and scores before choosing one "
        f"falsifiable prompt policy that helps the model track the goal, "
        f"searched locations, carried object, completed subgoals, and the exact currently available "
        f"actions. Successful plan-first traces use the family-appropriate sequence find target, "
        f"take it, perform any required clean/heat/cool operation, go to the destination, and put it; "
        f"two-object tasks repeat that sequence for a different instance. Guidance must prevent "
        f"substituting non-target objects or undoing completed placements and must require exactly "
        f"one take_action call chosen verbatim from the latest AVAILABLE ACTIONS. "
        f"Submit exactly one concise diagnostic guidance prompt of at most 80 words. The outer loop "
        f"will combine it with the fixed plan-first policy, including exactly one take_action tool call "
        f"and no prose, then render the fixed "
        f"prompt-only AgentHarness at {workspace / f'candidate-{iteration}.py'} and write the required "
        f"pending_eval.json to {pending}. Do not write Python and do not propose action rewriting. "
        f"The generated harness appends your guidance to the model's system prompt without changing "
        f"the emitted action. The fixed seed interface is:\n"
        f"{(runroot / 'inputs' / 'base_harness.py').read_text()}\n"
        f"Do not run benchmark episodes yourself."
    )
    if failure_feedback:
        prompt += (
            "\nRecent attempts, including earlier iterations, failed. The exact failed source is "
            "included below. Diagnose the behavior-level cause and improve the guidance rather than "
            "repeating an earlier policy under a new name. Do not include task-specific entities, "
            "example object names, numbered examples, or invented action strings in the guidance.\n"
            + json.dumps(_proposer_failure_context(runroot, failure_feedback), indent=2)
        )
    system = SKILL.read_text() + "\n\nYou are an autonomous coding agent. Use the filesystem tools repeatedly; do not answer until the candidate and pending_eval.json have been written."
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    transcript = workspace / "proposer_transcript.jsonl"
    api_base = os.environ.get("PROPOSER_API_BASE")
    if not api_base:
        raise RuntimeError("PROPOSER_API_BASE is not set")
    atomic_json(workspace / "proposer_config.json", {
        "iteration": iteration,
        "attempt": attempt,
        "model": model,
        "api_base": api_base,
        "max_tokens": max_tokens,
        "max_tool_turns": max_tool_turns,
        "temperature": temperature,
    })
    available_trajectory_outcomes: set[bool] = set()
    evidence_targets = []
    baseline_runs = sorted((runroot / "baseline").rglob("runs.jsonl"))
    other_runs = sorted(
        path for path in runroot.rglob("runs.jsonl") if path not in baseline_runs
    )
    for runs_file in baseline_runs + other_runs:
        for line_number, line in enumerate(
            runs_file.read_text(errors="replace").splitlines(), 1
        ):
            try:
                row = json.loads(line)
                reward = (((row.get("output") or {}).get("result") or {}).get("reward") or 0)
                openai_messages = (((row.get("output") or {}).get("result") or {})
                                   .get("openai_messages") or [])
                task_text = next(
                    (message.get("content", "") for message in openai_messages
                     if isinstance(message, dict) and message.get("role") == "user"),
                    "",
                ).lower()
                goal = task_text.partition("your task is to:")[2].partition(
                    "available actions:"
                )[0]
                family = "pick_place"
                for markers, name in (
                    ((" two ",), "pick_two"),
                    (("clean",), "clean"),
                    (("heat", " hot "), "heat"),
                    (("cool",), "cool"),
                    (("examine", "look at"), "examine"),
                ):
                    if any(marker in f" {goal} " for marker in markers):
                        family = name
                        break
                available_trajectory_outcomes.add(float(reward) > 0)
                evidence_targets.append({
                    "path": str(runs_file.relative_to(runroot)),
                    "line_number": line_number,
                    "success": float(reward) > 0,
                    "baseline": runs_file in baseline_runs,
                    "family": family,
                })
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    seen_trajectory_outcomes: set[bool] = set()
    trajectory_reads: set[tuple[str, int]] = set()
    required_reads = min(min_trajectories, len(evidence_targets))
    submission_attempts = 0
    submitted = False
    selected_targets = []
    for baseline_only in (True, False):
        for outcome in (True, False):
            matches = [
                item for item in evidence_targets
                if item["success"] is outcome
                and (item["baseline"] or not baseline_only)
                and item not in selected_targets
            ]
            families = []
            for item in matches:
                if item["family"] not in families:
                    selected_targets.append(item)
                    families.append(item["family"])
                if len(families) == 4:
                    break
        if len(selected_targets) >= max(required_reads, 8):
            break
    selected_targets.extend(
        item for item in evidence_targets if item not in selected_targets
    )
    selected_targets = selected_targets[:max(required_reads, 8)]
    messages[1]["content"] += (
        "\nUse these trajectory index entries with read_trajectory: "
        + json.dumps(selected_targets)
    )
    for turn in range(1, max_tool_turns + 1):
        evidence_ok = (
            len(trajectory_reads) >= required_reads
            and available_trajectory_outcomes.issubset(seen_trajectory_outcomes)
        )
        forced_tool = None
        forced_read_target = None
        if not evidence_ok:
            forced_tool = "read_trajectory"
            forced_read_target = next(
                item for item in selected_targets
                if (item["path"], item["line_number"]) not in trajectory_reads
            )
            messages.append({"role": "user", "content": (
                "Read this exact previously unseen trajectory now: "
                f"path={forced_read_target['path']!r}, "
                f"line_number={forced_read_target['line_number']}."
            )})
        message = _chat_completion(
            api_base, model, messages, max_tokens, timeout, temperature,
            forced_tool=forced_tool,
        )
        append_jsonl(transcript, {"turn": turn, "message": message})
        messages.append(message)
        calls = message.get("tool_calls") or []
        if forced_read_target is not None:
            for call in calls:
                function = call.get("function") or {}
                if function.get("name") == "read_trajectory":
                    function["arguments"] = json.dumps({
                        "path": forced_read_target["path"],
                        "line_number": forced_read_target["line_number"],
                    })
                    break
        if not calls:
            evidence_ok = (
                len(trajectory_reads) >= required_reads
                and available_trajectory_outcomes.issubset(seen_trajectory_outcomes)
            )
            candidate_ok = False
            pending_error = "pending_eval.json has not been written"
            if pending.exists():
                try:
                    payload = json.loads(pending.read_text())
                    entry = (payload.get("candidates") or [])[0]
                    candidate_path = Path(entry["path"])
                    if not candidate_path.is_absolute():
                        from_root = runroot / candidate_path
                        candidate_path = from_root if from_root.exists() else workspace / candidate_path
                    candidate_ok = validate_candidate(
                        candidate_path, trajectory_entities(runroot)
                    )[0]
                    if not candidate_ok:
                        pending_error = "the referenced candidate fails interface validation"
                    else:
                        pending_error = "ok"
                except Exception as exc:
                    candidate_ok = False
                    pending_error = f"pending_eval.json is invalid: {exc}"
            if pending.exists() and evidence_ok and candidate_ok:
                break
            messages.append({"role": "user", "content":
                f"The iteration is incomplete. Inspect at least {required_reads} distinct raw trajectories and every available success/failure outcome with read_trajectory, then write or revise a candidate that passes the AgentHarness interface. "
                f"Current artifact status: {pending_error}. Call submit_candidate with one concise guidance prompt and accurate metadata; do not write Python."})
            _compact_proposer_messages(messages)
            continue
        for call in calls:
            function = call.get("function") or {}
            arguments = {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                output = _run_proposer_tool(function.get("name", ""), arguments,
                                            runroot, workspace)
                if function.get("name") == "read_trajectory" and not output.startswith("ERROR:"):
                    trajectory = json.loads(output)
                    seen_trajectory_outcomes.add(float(trajectory.get("reward") or 0) > 0)
                    trajectory_reads.add((str(arguments.get("path")), int(arguments.get("line_number"))))
                if function.get("name") == "submit_candidate":
                    submission_attempts += 1
            except Exception as exc:
                output = f"ERROR: {type(exc).__name__}: {exc}"
            append_jsonl(transcript, {"turn": turn, "tool": function.get("name"),
                                      "arguments": arguments, "output": output[:12000]})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": output})
            if function.get("name") == "submit_candidate" and output.startswith("ERROR:"):
                if submission_attempts >= MAX_PROPOSER_ATTEMPTS:
                    raise RuntimeError(
                        "proposer exhausted its prompt submission attempts"
                    )
                messages.append({"role": "user", "content":
                    "The rendered prompt harness was rejected. Address the exact validation error by revising the guidance, then call submit_candidate again; do not write Python."})
            elif function.get("name") == "submit_candidate" and output.startswith("SUBMITTED "):
                submitted = True
        _compact_proposer_messages(messages)
        if submitted:
            break
    else:
        raise RuntimeError("proposer exhausted its filesystem-tool turn budget")
    if not pending.exists():
        raise RuntimeError("proposer did not write pending_eval.json")
    if len(trajectory_reads) < required_reads:
        raise RuntimeError(
            f"proposer inspected {len(trajectory_reads)}/{required_reads} required trajectories"
        )
    if not available_trajectory_outcomes.issubset(seen_trajectory_outcomes):
        raise RuntimeError(
            "proposer did not inspect every available success/failure outcome"
        )
    candidates = (json.loads(pending.read_text()).get("candidates") or [])
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one candidate, got {len(candidates)}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "evolution.toml")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--smoke-tasks", type=int)
    parser.add_argument("--evolution-tasks", type=int)
    parser.add_argument("--heldout-tasks", type=int)
    parser.add_argument("--proposer-model")
    parser.add_argument("--proposer-timeout", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    args.iterations = args.iterations or int(config["iterations"])
    args.smoke_tasks = args.smoke_tasks or int(config["smoke_tasks"])
    args.evolution_tasks = args.evolution_tasks or int(config["evolution_tasks"])
    args.heldout_tasks = args.heldout_tasks or int(config["heldout_tasks"])
    args.proposer_model = args.proposer_model or str(config["proposer_model"])
    args.proposer_timeout = (
        args.proposer_timeout or int(config["proposer_timeout_seconds"])
    )
    proposer_max_tokens = int(config["proposer_max_tokens"])
    proposer_max_tool_turns = int(config["proposer_max_tool_turns"])
    proposer_min_trajectories = int(config.get("proposer_min_trajectories", 8))
    proposer_temperature = float(config["proposer_temperature"])
    child_max_tokens = int(config["child_max_tokens"])
    child_max_model_len = int(config["child_max_model_len"])
    child_model = str(config["child_model"])
    if args.proposer_model != child_model:
        raise SystemExit("trajectory executor and proposer must use the same model")
    os.environ["CHILD_MAX_TOKENS"] = str(child_max_tokens)

    runroot = ROOT / "runs" / args.run
    if runroot.exists() and not args.resume:
        raise SystemExit(f"run root already exists: {runroot}; use a new name or --resume")
    runroot.mkdir(parents=True, exist_ok=True)
    inputs = runroot / "inputs"
    inputs.mkdir(exist_ok=True)
    seed = inputs / "base_harness.py"
    run_config = inputs / "evolution.toml"
    for source, target in ((BASE, seed), (args.config.resolve(), run_config)):
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise SystemExit(f"immutable run input differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
    protocol_path = runroot / "protocol.json"
    protocol = {"schema": 1, "iterations": args.iterations,
                "smoke_tasks": args.smoke_tasks,
                "evolution_tasks": args.evolution_tasks,
                "heldout_tasks": args.heldout_tasks,
                "base_model": os.environ.get("QWEN_MODEL"),
                "proposer_model": args.proposer_model,
                "child_max_tokens": child_max_tokens,
                "child_max_model_len": child_max_model_len,
                "proposer_max_tokens": proposer_max_tokens,
                "proposer_max_tool_turns": proposer_max_tool_turns,
                "proposer_temperature": proposer_temperature,
                "base_harness": str(seed), "config": str(run_config)}
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise SystemExit("resume protocol differs from protocol.json")
    atomic_json(protocol_path, protocol)

    frontier_path = runroot / "frontier_val.json"
    summary_path = runroot / "evolution_summary.jsonl"
    attempts_path = runroot / "proposer_attempts.jsonl"
    baseline_out = runroot / "baseline" / "evolution"
    baseline = completed_evaluation(baseline_out, seed, args.evolution_tasks)
    if baseline is None:
        archive_partial(baseline_out, runroot)
        baseline = evaluate(seed, "candidate-evolution", baseline_out,
                            args.evolution_tasks, nonce=1)
    if not frontier_path.exists():
        atomic_json(frontier_path, {
            "_best": {"agent": "baseline", "pass_rate": baseline["pass_rate"],
                      "source": str(seed.resolve())},
            "candidates": {"baseline": baseline},
        })

    completed = 0
    if summary_path.exists():
        latest_by_iteration = {}
        for line in summary_path.read_text().splitlines():
            if line:
                record = json.loads(line)
                latest_by_iteration[int(record["iteration"])] = record
        while (
            completed + 1 in latest_by_iteration
            and latest_by_iteration[completed + 1].get("outcome") != "evaluation_failed"
        ):
            completed += 1
    forced_resume = os.environ.get("ALFWORLD_RESUME_FROM_ITERATION")
    if forced_resume is not None:
        forced_completed = int(forced_resume)
        if not 0 <= forced_completed <= args.iterations:
            raise SystemExit("ALFWORLD_RESUME_FROM_ITERATION must be between 0 and --iterations")
        completed = forced_completed
    for iteration in range(completed + 1, args.iterations + 1):
        workspace = runroot / f"iteration-{iteration:02d}"
        # A scheduler timeout may interrupt only the full candidate evaluation.
        # Preserve an already submitted, validated candidate and its complete
        # smoke instead of asking the proposer to invent a different harness.
        pending = workspace / "pending_eval.json"
        if pending.exists():
            try:
                candidate_meta = (json.loads(pending.read_text()).get("candidates") or [])[0]
                candidate = Path(candidate_meta["path"])
                if not candidate.is_absolute():
                    candidate = runroot / candidate
                ok, reason = validate_candidate(candidate, trajectory_entities(runroot))
                smoke = completed_evaluation(
                    workspace / "smoke", candidate, args.smoke_tasks
                )
                if ok and smoke:
                    archive_partial(workspace / "evaluation", runroot)
                    evaluation = evaluate(
                        candidate, "candidate-evolution", workspace / "evaluation",
                        args.evolution_tasks, nonce=iteration * 100 + 91,
                    )
                    record = {
                        "iteration": iteration,
                        "attempt": json.loads(
                            (workspace / "proposer_config.json").read_text()
                        ).get("attempt"),
                        **candidate_meta,
                        "path": str(candidate.resolve()),
                        "validation": reason,
                        "smoke": smoke,
                        "evaluation": evaluation,
                        "outcome": "evaluated",
                        "resumed_stage": "evaluation",
                    }
                    frontier = json.loads(frontier_path.read_text())
                    frontier.setdefault("candidates", {})[candidate_meta["name"]] = {
                        **evaluation, "source": str(candidate.resolve())
                    }
                    if evaluation["pass_rate"] > frontier["_best"]["pass_rate"]:
                        frontier["_best"] = {
                            "agent": candidate_meta["name"],
                            "pass_rate": evaluation["pass_rate"],
                            "source": str(candidate.resolve()),
                        }
                    atomic_json(frontier_path, frontier)
                    append_jsonl(summary_path, record)
                    continue
            except (IndexError, KeyError, OSError, ValueError, json.JSONDecodeError):
                pass
        prior_archive = archive_partial(workspace, runroot)
        failure_feedback = []
        if summary_path.exists():
            failure_feedback.extend(
                record for record in map(json.loads, summary_path.read_text().splitlines())
                if int(record.get("iteration", -1)) == iteration
                and record.get("outcome") == "evaluation_failed"
            )
        prior_attempts = []
        if attempts_path.exists():
            all_attempts = list(map(json.loads, attempts_path.read_text().splitlines()))
            prior_attempts = [
                record for record in all_attempts
                if int(record.get("iteration", -1)) == iteration
                and "No space left on device" not in str(record.get("error", ""))
            ]
            failure_feedback.extend(all_attempts[-MAX_PROPOSER_ATTEMPTS:])
        if prior_archive is not None:
            failure_feedback.append({
                "outcome": "interrupted",
                "archived_workspace": str(prior_archive.relative_to(runroot)),
            })

        iteration_record = None
        start_attempt = len(prior_attempts) + 1
        for attempt in range(start_attempt, MAX_PROPOSER_ATTEMPTS + 1):
            attempt_record = {"iteration": iteration, "attempt": attempt}
            try:
                candidates = propose(
                    runroot, iteration, attempt, failure_feedback,
                    args.proposer_model, args.proposer_timeout,
                    proposer_max_tokens, proposer_max_tool_turns,
                    proposer_min_trajectories, proposer_temperature,
                )
                candidate_meta = candidates[0]
                candidate = Path(candidate_meta["path"])
                if not candidate.is_absolute():
                    from_root = runroot / candidate
                    candidate = from_root if from_root.exists() else workspace / candidate
                ok, reason = validate_candidate(candidate, trajectory_entities(runroot))
                attempt_record.update(candidate_meta)
                attempt_record.update({"path": str(candidate.resolve()), "validation": reason})
                if not ok:
                    raise RuntimeError(f"candidate validation failed: {reason}")
                attempt_record["smoke"] = evaluate(
                    candidate, "candidate-smoke", workspace / "smoke",
                    args.smoke_tasks, nonce=iteration * 100 + attempt * 10,
                )
                fatal_smoke = sum(
                    attempt_record["smoke"]["status_counts"].get(status, 0)
                    for status in ("agent invalid action", "task error")
                )
                if fatal_smoke == args.smoke_tasks:
                    raise RuntimeError(
                        "all smoke trials ended in invalid-action/task-error states"
                    )
                if attempt_record["smoke"]["active_harness_trials"] == 0:
                    attempt_record["smoke_warning"] = (
                        "conditional intervention did not trigger in this smoke slice; "
                        "synthetic multi-turn validation did trigger it"
                    )
                attempt_record["evaluation"] = evaluate(
                    candidate, "candidate-evolution", workspace / "evaluation",
                    args.evolution_tasks, nonce=iteration * 100 + attempt * 10 + 1,
                )
                attempt_record["outcome"] = "evaluated"
                iteration_record = attempt_record
                frontier = json.loads(frontier_path.read_text())
                frontier.setdefault("candidates", {})[candidate_meta["name"]] = {
                    **attempt_record["evaluation"], "source": str(candidate.resolve())}
                if attempt_record["evaluation"]["pass_rate"] > frontier["_best"]["pass_rate"]:
                    frontier["_best"] = {
                        "agent": candidate_meta["name"],
                        "pass_rate": attempt_record["evaluation"]["pass_rate"],
                        "source": str(candidate.resolve()),
                    }
                atomic_json(frontier_path, frontier)
                break
            except Exception as exc:
                attempt_record["outcome"] = "attempt_failed"
                attempt_record["error"] = f"{type(exc).__name__}: {exc}"
                archived = archive_partial(workspace, runroot)
                if archived is not None:
                    attempt_record["archived_workspace"] = str(
                        archived.relative_to(runroot)
                    )
                append_jsonl(attempts_path, attempt_record)
                failure_feedback.append(attempt_record)

        if iteration_record is None:
            iteration_record = {
                "iteration": iteration,
                "outcome": "attempts_exhausted",
                "attempts": MAX_PROPOSER_ATTEMPTS,
                "error": failure_feedback[-1].get("error", "all proposer attempts failed"),
            }
        append_jsonl(summary_path, iteration_record)

    frontier = json.loads(frontier_path.read_text())
    winner = Path(frontier["_best"]["source"])
    final_out = runroot / "final" / "heldout"
    final = completed_evaluation(final_out, winner, args.heldout_tasks)
    if final is None:
        archive_partial(final_out, runroot)
        final = evaluate(winner, "heldout", final_out, args.heldout_tasks, nonce=999)
    atomic_json(runroot / "final" / "summary.json",
                {"winner": frontier["_best"], "heldout": final})
    print((runroot / "final" / "summary.json").read_text())


if __name__ == "__main__":
    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
