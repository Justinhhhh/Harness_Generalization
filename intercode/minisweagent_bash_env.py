"""mini-SWE-agent environment adapter for the offline InterCode Bash image."""
import base64
from pathlib import Path
from typing import Any

from intercode.envs import BashEnv
from minisweagent.exceptions import Submitted


class IntercodeBashEnvironment:
    def __init__(self, *, image_name: str = "intercode-bash", setup_script: str = "", **kwargs: Any):
        self.env = BashEnv(image_name, network_mode="none", verbose=False, **kwargs)
        self.env.reset()
        self._setup_filesystem(setup_script)

    def _setup_filesystem(self, setup_script: str) -> None:
        if not setup_script:
            return
        setup = Path(setup_script).read_text()
        encoded = base64.b64encode(setup.encode()).decode()
        for container in (self.env.container, self.env.container_eval):
            result = container.exec_run(["/bin/sh", "-c", f"echo {encoded} | base64 -d | /bin/sh"])
            if result.exit_code != 0:
                raise RuntimeError(f"filesystem setup failed: {result.output!r}")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        self.env.exec_action(command)
        output = self.env.observation
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            raise Submitted({
                "role": "exit",
                "content": output,
                "extra": {"exit_status": "Submitted", "submission": output},
            })
        return {
            "output": output,
            "returncode": 0 if self.env.info.get("action_executed") else 1,
            "exception_info": "",
        }

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def serialize(self) -> dict:
        return {"info": {"environment_type": f"{__name__}.IntercodeBashEnvironment"}}

    def close(self) -> None:
        # These containers are task-scoped. Force-remove them instead of
        # waiting for Docker's graceful stop timeout on every completed task.
        for container in (self.env.container, self.env.container_eval):
            try:
                container.remove(force=True)
            except Exception:
                pass
