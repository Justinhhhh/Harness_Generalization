"""Protected AgentBench runtime for an original Self-Harness virtual surface."""
from __future__ import annotations
import json
from copy import deepcopy
from openai.types.chat import ChatCompletionSystemMessageParam

_surface = None
def configure_candidate(surface):
    global _surface
    _surface = surface
def _call(name, default):
    fn = getattr(_surface, name, None) if _surface else None
    return fn() if callable(fn) else default
def _guidance():
    text = [str(_call(n, "")).strip() for n in ("build_system_prompt", "build_bootstrap_instruction", "build_execution_instruction", "build_verification_instruction", "build_failure_recovery_instruction", "build_multimodal_instruction")]
    for n, default in (("build_memory_sources", []), ("build_subagents", []), ("build_skills", []), ("build_permissions", []), ("build_interrupt_on", None), ("build_tools", []), ("build_runtime_control_policy", {})):
        value = _call(n, default)
        if value: text.append(f"{n}: {json.dumps(value, ensure_ascii=False)}")
    return "\n\n".join(x for x in text if x)

class AgentHarness:
    def __init__(self, session, tools, max_steps):
        self.session, self.trace = session, []
        self.session.set_tools(deepcopy(tools)); self.session.set_full_history(True)
    def inject(self, item): self.session.inject(item)
    def sync_action(self):
        guidance = _guidance()
        if guidance:
            self.session.inject(ChatCompletionSystemMessageParam(role="system", content=guidance))
            self.trace.append({"hooks": True})
        return self.session.sync_action()
