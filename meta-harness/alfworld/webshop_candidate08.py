"""Minimal WebShop adapter for the winning ALFWorld candidate-08 guidance."""
from copy import deepcopy
import re


class WebShopCandidate08Harness:
    def __init__(self, session, tools, max_steps):
        self.session = session
        self.max_steps = max_steps
        self.turn = 0
        self.trace = []
        self.task_goal = ""
        self.guidance = (
            "Before every action, silently re-read the shopping instruction. "
            "Track the exact target product, brand, required attributes, searched "
            "queries, current page, selected attributes, and whether checkout is "
            "complete. Search using all important product terms, choose only a "
            "currently available action, select every required attribute, and click "
            "buy now once all attributes are selected. Do not switch to a different "
            "product or undo a completed selection. Always call exactly one tool."
        )
        self.session.set_tools(deepcopy(tools))
        self.session.set_full_history(True)

    def inject(self, item):
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                match = re.search(
                    r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]",
                    content,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if match:
                    self.task_goal = match.group(1).strip()
        self.session.inject(item)

    def sync_action(self):
        self.turn += 1
        if self.turn == 1 or (self.turn - 1) % 5 == 0:
            reminder = self.guidance
            if self.task_goal:
                reminder += " Shopping instruction: " + self.task_goal
            self.session.inject({"role": "system", "content": reminder})
            self.trace.append({"turn": self.turn})
        original = deepcopy(self.session.history)
        self.session.cover(deepcopy(original))
        try:
            output = self.session.sync_action()
        finally:
            self.session.cover(original)
        raw = deepcopy(output.messages)
        for message in output.messages:
            self.session.inject(deepcopy(message))
        self.session.controller.env_output.history = deepcopy(self.session.history)
        return output, raw, max(0, self.max_steps - self.turn + 1)

    def record_and_rewrite(self, output, _raw_messages, _clickables):
        return output
