"""Minimal WebShop transfer of the ALFWorld iteration-7 repeat-action harness.

WebShop has different tools, so the ALFWorld-specific ``look`` replacement is
not transferred.  This preserves the domain-independent part: identify a
consecutive duplicate tool invocation and ask the agent to reconsider it.
"""
from copy import deepcopy
import json


class WebShopTransferHarness:
    def __init__(self, _config=None):
        self.previous = None

    def h2(self, message, _history):
        return message

    def h3(self, tools):
        return deepcopy(tools)

    def h5(self, _history):
        return []

    def h4(self, history, _remaining):
        assistant = [item for item in history if item.get("role") == "assistant"]
        if not assistant:
            return []
        calls = assistant[-1].get("tool_calls") or []
        if not calls:
            return []
        call = calls[0].get("function", {})
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = args
        current = (call.get("name"), json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args))
        repeated = current == self.previous
        self.previous = current
        if repeated:
            return ["The previous tool call was identical. Re-check the current page and choose a different valid action if it will not make progress."]
        return []
