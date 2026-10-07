"""Original Self-Harness virtual surface for the ALFWorld environment adapter."""
from alfworld_runtime import AgentHarness

def build_system_prompt() -> str:
    return ""
def build_memory_sources() -> list[str]:
    return []
def build_subagents() -> list[dict]:
    return []
def build_skills() -> list[str]:
    return []
def build_permissions() -> list:
    return []
def build_interrupt_on() -> dict | None:
    return None
def build_bootstrap_instruction() -> str:
    return ""
def build_execution_instruction() -> str:
    return ""
def build_verification_instruction() -> str:
    return ""
def build_failure_recovery_instruction() -> str:
    return ""
def build_multimodal_instruction() -> str:
    return ""
def build_tools() -> list:
    return []
def build_runtime_control_policy() -> dict:
    return {}
