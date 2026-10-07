"""Minimal Self-Harness candidate surface for AgentBench ALFWorld.

The baseline is deliberately a no-op session wrapper. Only the two virtual
hook functions are exposed to the proposer; the runtime wrapper is protected.
"""
from copy import deepcopy
import json

from openai.types.chat import ChatCompletionSystemMessageParam


def build_system_prompt() -> str:
    return ""


def build_failure_recovery_instruction() -> str:
    return ""


class AgentHarness:
    def __init__(self, session, tools, max_steps):
        self.session = session
        self.max_steps = max_steps
        self.turn = 0
        self.trace = []
        self.session.set_tools(deepcopy(tools))
        self.session.set_full_history(True)

    def inject(self, item):
        self.session.inject(item)

    def sync_action(self):
        self.turn += 1
        guidance = build_system_prompt().strip()
        if guidance:
            self.session.inject(ChatCompletionSystemMessageParam(role="system", content=guidance))
            self.trace.append({"turn": self.turn, "hook": "build_system_prompt"})
        original = deepcopy(self.session.history)
        remaining = max(0, self.max_steps - self.turn + 1)
        self.session.cover(deepcopy(original))
        try:
            output = self.session.sync_action()
        finally:
            self.session.cover(original)
        for message in output.messages:
            self.session.inject(deepcopy(message))
        self.session.controller.env_output.history = deepcopy(self.session.history)
        return output
