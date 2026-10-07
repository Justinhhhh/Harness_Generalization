import base64
import math
import shlex
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple

from intercode.envs.ic_env import (
    IntercodeEnv,
    AGENT_OBS, EVAL_OBS, CORRUPT_GOLD, ACTION_EXEC, REWARD
)
from intercode.utils import get_container, timeout
from intercode.utils.utils import TIMEOUT_DURATION

GIT_RESET_SCRIPT = "git reset --hard; git clean -fd;"
GIT_STATUS_SCRIPT = "git status --short;"

IMAGE_TO_SETTINGS = {
    "intercode-bash": "/bin/sh",
    "intercode-bash-qwen3-4b": "/bin/sh",
    "intercode-bash-qwen3-8b": "/bin/sh",
    "intercode-bash-replay-qwen3-4b": "/bin/sh",
    "intercode-bash-replay-qwen3-8b": "/bin/sh",
    "intercode-nl2bash": "/bin/bash",
    "intercode-ctf": "/bin/bash",
    "intercode-swe": "/bin/bash",
}

class BashEnv(IntercodeEnv):
    """Gym environment for bash shell"""
    name = "ic_bash"

    def __init__(self, image_name: str, **kwargs):
        super(BashEnv, self).__init__(image_name, **kwargs)

        # Establish connection with evaluation container
        container_suffix = self.kwargs.get("container_suffix", "")
        suffix = f"_{container_suffix}" if container_suffix else ""
        self.ctr_name_eval = f"{self.image_name}_ic_ctr_eval{suffix}"
        self.container_eval = get_container(self.ctr_name_eval, self.image_name)
    
    def reset_container(self) -> None:
        self.workdir = "/"
        exit_code, output = self.container.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset `{self.container_name}` container successfully: {output}")
    
    def exec_action(self, action: str) -> None:
        """Executes action in bash shell"""
        is_cd_flag = action.startswith("cd")
        if is_cd_flag:
            cd_arg = action[action.index("cd ")+3:].strip()
            new_path = self.simplify_path(self.workdir, cd_arg)
            action = f"cd {new_path}"
        
        try:
            with timeout():
                entrypoint = IMAGE_TO_SETTINGS[self.image_name]
                exit_code, output = self.container.exec_run(
                    ["timeout", f"{TIMEOUT_DURATION}s", entrypoint, "-c", action.strip()],
                    workdir="/" if is_cd_flag else self.workdir)
                self.observation = output.decode("utf-8")
                if exit_code == 124:
                    self.observation = "Command timed out"
                    self.info[ACTION_EXEC] = False
                else:
                    self.info[ACTION_EXEC] = exit_code == 0
        except TimeoutError:
            self.observation = f"Command timed out"
            self.info[ACTION_EXEC] = False
        except Exception as exc:
            self.observation = "Malformed command"
            self.info[ACTION_EXEC] = False
            self.info["action_error"] = f"{type(exc).__name__}: {exc}"
        
        if is_cd_flag and self.info[ACTION_EXEC]:
            self.workdir = new_path
            
    def get_reward(self) -> Tuple[float, Dict]:
        """
        The reward currently is calculated as a weighted sum of the following:
        - 0.33: (File System Diff) Difference in file system states between agent, gold command
        - 0.33: (File Content) Verify each file was correctly changed by agent using hashing
        - 0.33: (Observation) Verify that correct output was generated
        """
        # Reset evaluation container state
        if not self.kwargs.get("preserve_eval_state", False):
            exit_code, output = self.container_eval.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
            if exit_code != 0:
                raise RuntimeError(f"Failed to reset `{self.ctr_name_eval}` container successfully: {output}")
            setup_script = self.kwargs.get("eval_setup_script")
            if setup_script:
                encoded = base64.b64encode(Path(setup_script).read_bytes()).decode()
                setup_result = self.container_eval.exec_run(
                    ["/bin/sh", "-c", f"echo {encoded} | base64 -d | /bin/sh"]
                )
                if setup_result.exit_code != 0:
                    raise RuntimeError(f"evaluator filesystem setup failed: {setup_result.output!r}")
        
        # Run gold command(s) in evaluation container
        self.observation_eval = None
        try:
            if isinstance(self.gold, str):
                gold_result = self.exec_timed(self.container_eval, self.gold)
                self.observation_eval = gold_result.output.decode("utf-8")
                if gold_result.exit_code != 0:
                    self.info[CORRUPT_GOLD] = True
                    return 0.0, {REWARD: {"corrupt_gold": True}, CORRUPT_GOLD: True}
            elif isinstance(self.gold, List):
                gold_result = self.exec_timed(self.container_eval, ";".join(self.gold))
                self.observation_eval = gold_result.output.decode("utf-8")
                if gold_result.exit_code != 0:
                    self.info[CORRUPT_GOLD] = True
                    return 0.0, {REWARD: {"corrupt_gold": True}, CORRUPT_GOLD: True}
            self.info[CORRUPT_GOLD] = False
        except Exception as e:
            self.info[CORRUPT_GOLD] = True

        # Calculate Rewards
        reward, info = 0.01, {}
        info[REWARD] = {}

        # PART 1: Compare file system states
        diff_agent = self.parse_status(self.exec_timed(self.container, GIT_STATUS_SCRIPT).output.decode("utf-8"))
        diff_eval = self.parse_status(self.exec_timed(self.container_eval, GIT_STATUS_SCRIPT).output.decode("utf-8"))
        info["diff_miss"] = list(set(diff_eval) - set(diff_agent))
        info["diff_extra"] = list(set(diff_agent) - set(diff_eval))
        p1_score = round(0.33 * (1 - math.erf(len(info["diff_miss"]) + len(info["diff_extra"]))), 2)
        info[REWARD]["file_diff"] = p1_score
        reward += p1_score

        # PART 2: Check if files changed by both agent, gold commands were modified correctly
        p2_score = 0.33
        # Only check corrects of common changes that were added or modified
        filter_changes = lambda x: (x[1] in ["A", "??", "C"])
        diff_same = [x for x in list(set(diff_agent) & set(diff_eval)) if filter_changes(x)]
        
        if len(diff_same) > 0:
            same_changes = 0
            # Compute hashes for files and folders differently using md5 checksums
            get_hash_cmd = lambda x: f"md5sum {x}" if "." in x else f"md5deep -r {x}"

            for path in diff_same:
                hash_cmd = get_hash_cmd(path[0])
                agent_hash = self.exec_timed(self.container, hash_cmd).output.decode("utf-8")
                gold_hash = self.exec_timed(self.container_eval, hash_cmd).output.decode("utf-8")
                same_changes += 1 if agent_hash == gold_hash else 0
            
            info["diff_same"] = {"files": diff_same, "correct": same_changes, "total": len(diff_same)}
            p2_score = round(0.33 * (same_changes / len(diff_same)), 2)
        info[REWARD]["file_changes"] = p2_score
        reward += p2_score
        
        # PART 3: Compare agent, query answers
        info[AGENT_OBS] = self.observation
        info[EVAL_OBS] = self.observation_eval
        try:
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform([info[AGENT_OBS], info[EVAL_OBS]])
            answer_similarity = tfidf * tfidf.T
            info["answer_similarity"] = answer_similarity.toarray()[0][1]
        except:
            info["answer_similarity"] = 1 if info[AGENT_OBS] == info[EVAL_OBS] else 0
        p3_score = round(0.33 * info["answer_similarity"], 2)
        info[REWARD]["answer_similarity"] = p3_score
        reward += p3_score

        self.reward = reward 
        self.info.update(info)

        self.logger.info(f"Info: {self.info}")
        self.logger.info(f"Reward: {self.reward}")
        return reward, info

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.container_eval.stop()
        self.logger.info("Agent, evaluation containers stopped")
    
    ############################
    ### MARK: Helper methods ###
    ############################

    def clean_cmd(self, action: str) -> str:
        """Cleans action string"""
        entrypoint = IMAGE_TO_SETTINGS[self.image_name]
        return f"{entrypoint} -c {shlex.quote(action.strip())}"

    def exec_timed(self, container, action: str):
        entrypoint = IMAGE_TO_SETTINGS[self.image_name]
        return container.exec_run(["timeout", f"{TIMEOUT_DURATION}s", entrypoint, "-c", action.strip()])

    def parse_status(self, status: str) -> List:
        """Parses git status output into list of changes"""
        status_lst = status.split()
        changes = []
        for i in range(0, len(status_lst), 2):
            changes.append((status_lst[i+1], status_lst[i]))
        return changes

    def simplify_path(self, current: str, changed: str) -> str:
        """Resolves path from current working directory path and the argument of the `cd` command"""
        if not changed:
            return current
        if changed[0] == "/":
            current = ""

        path = []
        
        for segment in (current + "/" + changed).split("/"):
            if segment == "..":
                if path:
                    path.pop()
            elif segment and segment != ".":
                path.append(segment)

        return "/" + "/".join(path)
