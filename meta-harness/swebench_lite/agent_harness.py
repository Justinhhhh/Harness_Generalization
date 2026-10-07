"""Runtime harness for bounded-context SWE-bench agents."""
from __future__ import annotations

import json

from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent


class BudgetedSWEAgent(ProgressTrackingAgent):
    """Keep the fixed agent usable through long shell trajectories.

    This is deliberately task-agnostic: it never edits files or writes a patch.
    It only bounds retained conversation state. The benchmark's own task
    instructions remain responsible for workflow and submission decisions.
    """

    max_history_chars = 72_000
    repeated_call_threshold = 3
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harness_events: list[dict] = []
        self._last_call_signature = None
        self._repeat_count = 0
        self._loop_nudge_used = False

    def _compact_history(self) -> None:
        if sum(len(str(message.get("content", ""))) for message in self.messages) <= self.max_history_chars:
            return
        # Keep the immutable system/task prefix and a suffix beginning at an
        # assistant action.  This avoids orphaning a tool result from its call.
        prefix = self.messages[:2]
        suffix: list[dict] = []
        size = 0
        have_assistant = False
        for message in reversed(self.messages[2:]):
            size += len(str(message.get("content", "")))
            suffix.append(message)
            have_assistant = have_assistant or message.get("role") == "assistant"
            if size >= self.max_history_chars // 2 and have_assistant:
                break
        suffix.reverse()
        while suffix and suffix[0].get("role") not in {"assistant", "user"}:
            suffix.pop(0)
        note = {
            "role": "user",
            "content": (
                "Earlier shell interaction was compacted to stay within the context window. "
                "Re-read the original task, inspect the current repository state and git diff, "
                "then continue from the latest observations."
            ),
        }
        self.messages = prefix + [note] + suffix
        self.harness_events.append({"event": "compact_history", "turn": self.n_calls})

    def query(self):
        self._compact_history()
        self._break_repeated_tool_call()
        return super().query()

    def _break_repeated_tool_call(self) -> None:
        """Nudge only on an observed loop; never impose a fixed turn schedule."""
        if not self.messages:
            return
        # ``query`` is called after the tool result has been appended, so the
        # latest message is normally a ``tool`` message rather than the
        # assistant tool-call message.  Looking only at ``self.messages[-1]``
        # therefore never sees the loop we want to interrupt.
        assistant_calls = []
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                assistant_calls.append(message["tool_calls"])
            if len(assistant_calls) >= 6:
                break
        if len(assistant_calls) < self.repeated_call_threshold:
            return
        try:
            signatures = [
                json.dumps([
                    {
                        "name": call.get("function", {}).get("name"),
                        "arguments": call.get("function", {}).get("arguments"),
                    }
                    for call in calls
                ], sort_keys=True)
                for calls in assistant_calls
            ]
        except (TypeError, ValueError):
            return
        # The newest entries are reversed.  Catch both a strict repeat and a
        # short two-command oscillation (e.g. grep A / grep B / grep A ...).
        counts = {item: signatures.count(item) for item in set(signatures)}
        signature = signatures[0]
        loop_detected = counts.get(signature, 0) >= self.repeated_call_threshold
        loop_detected = loop_detected or (
            len(counts) <= 2 and max(counts.values()) >= self.repeated_call_threshold
        )
        # Use the multiset rather than the exact sliding-window order so a
        # single loop does not generate a new reminder on every turn.
        fingerprint = json.dumps(sorted(signatures), sort_keys=True)
        # A loop reminder is a recovery hint, not a turn-by-turn controller.
        # One nudge per task leaves the agent room to react without filling its
        # context with repeated copies of the same instruction.
        if loop_detected and not self._loop_nudge_used:
            self.add_messages({"role": "user", "content": (
                "The same tool call has been repeated without new information. Pause and use the "
                "latest observation: check the command's actual output, change the command or inspect "
                "a more targeted file, and continue solving the task."
            )})
            self.harness_events.append({"event": "repeated_tool_call_break", "turn": self.n_calls})
            self._loop_nudge_used = True

    def serialize(self, *extra_dicts):
        return super().serialize(*extra_dicts, {"info": {"harness_events": self.harness_events}})
