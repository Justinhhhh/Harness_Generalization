"""Transparent standalone AgentHarness seed for ALFWorld Meta-Harness."""
from copy import deepcopy
import json


class AgentHarness:
    """Own the model-session boundary without changing baseline behavior."""

    def __init__(self, session, tools, max_steps):
        self.session = session
        self.max_steps = max_steps
        self.turn = 0
        self.trace = []
        self.session.set_tools(deepcopy(tools))
        self.session.set_full_history(True)

    def inject(self, item):
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
        arguments = calls[0].get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return arguments.get("action") if isinstance(arguments, dict) else None

    @staticmethod
    def set_action(message, action):
        updated = deepcopy(message)
        call = (updated.get("tool_calls") or [])[0]
        arguments = call.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
            arguments["action"] = action
            call["function"]["arguments"] = json.dumps(arguments)
        else:
            arguments["action"] = action
        return updated

    def sync_action(self):
        self.turn += 1
        original = deepcopy(self.session.history)
        remaining = max(0, self.max_steps - self.turn + 1)
        prepared = self.prepare_history(deepcopy(original), remaining)
        if not isinstance(prepared, list):
            raise TypeError("prepare_history must return a message list")
        self.session.cover(prepared)
        try:
            output = self.session.sync_action()
        finally:
            self.session.cover(original)
        raw_messages = deepcopy(output.messages)
        processed = self.process_output(output, deepcopy(original), remaining)
        if processed is None or not hasattr(processed, "messages"):
            raise TypeError("process_output must return an action result")
        for message in processed.messages:
            self.session.inject(deepcopy(message))
        self.session.controller.env_output.history = deepcopy(self.session.history)
        if prepared != original or processed.messages != raw_messages:
            self.trace.append({"turn": self.turn, "remaining": remaining})
        return processed
