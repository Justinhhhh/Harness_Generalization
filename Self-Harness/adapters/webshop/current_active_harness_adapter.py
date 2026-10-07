"""WebShop bridge for the final ALFWorld Self-Harness active baseline surface.

The ALFWorld evolution selected its baseline surface after rejecting every
candidate.  WebShop therefore keeps only the shared Self-Harness session
boundary and does not carry over WebShop-specific Iter7 recovery behavior.
"""
from copy import deepcopy


class WebShopCandidate08Harness:
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
        original = deepcopy(self.session.history)
        self.session.cover(deepcopy(original))
        try:
            output = self.session.sync_action()
        finally:
            self.session.cover(original)
        raw_messages = deepcopy(output.messages)
        for message in output.messages:
            self.session.inject(deepcopy(message))
        self.session.controller.env_output.history = deepcopy(self.session.history)
        return output, raw_messages, max(0, self.max_steps - self.turn + 1)

    def record_and_rewrite(self, output, _raw_messages, _clickables):
        return output
