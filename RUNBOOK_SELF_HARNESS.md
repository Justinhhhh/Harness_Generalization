# Self-Harness / Meta-Harness 运行记录

本文记录 ALFWorld 与 WebShop 的实际运行入口、模型服务配置、并发设置和 GPU 操作。

## 当前结论

- ALFWorld Qwen3-4B 10 轮 evolution + heldout 已完成。
- ALFWorld heldout：38/134，28.36%。
- WebShop Qwen3-4B 100 条已完成：success@1=27%，average reward=0.615833。
- Qwen3.5-9B 权重已下载到本地，但 9B ALFWorld debug 服务和主任务已停止，尚未产生正式 9B 结果。
- ALFWorld 最终选中的 harness 是 baseline；第 10 轮 proposer 的 candidate 全部被拒绝。

## 共享设置

- agent：`thinking=false`，`max_tokens=2048`
- proposer：`max_tokens=8192`，`thinking=false`
- vLLM：`max-model-len=32768`
- tool calling：`--enable-auto-tool-choice --tool-call-parser hermes`
- ALFWorld candidate evolution：`SELF_HARNESS_WORKERS=4`
- ALFWorld heldout：沿用原 Meta-Harness 逻辑，task concurrency 为 1

## Qwen3-4B ALFWorld Meta-Harness

入口脚本：

```bash
/home/mohanz/Harness_Generalization/Self-Harness/adapters/alfworld/run_full_alfworld_self_4b.sbatch
```

该脚本使用：

```text
evolution tasks = 140
heldout tasks = 134
iterations = 10
workers = 4
QWEN_MODEL = qwen3-4b
```

提交方式：

```bash
sbatch --export=ALL,QWEN_API_BASE=http://<node>:<port>/v1,PROPOSER_API_BASE=http://<node>:<port>/v1 \
  /home/mohanz/Harness_Generalization/Self-Harness/adapters/alfworld/run_full_alfworld_self_4b.sbatch
```

结果目录：

```text
/home/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-4b-10iter/
```

## Qwen3-4B vLLM 服务

服务脚本：

```bash
/home/mohanz/Harness_Generalization/Self-Harness/jobs/self-harness-qwen3-4b-ada.sbatch
```

关键参数：

```text
served model = qwen3-4b
max model len = 32768
tool parser = hermes
```

启动后先检查：

```bash
curl http://<node>:30018/v1/models
```

确认返回 `qwen3-4b` 后，才能启动 ALFWorld 主任务。

## WebShop：迁移后的 ALFWorld baseline harness + Qwen3-4B

适配文件：

```text
/home/mohanz/Harness_Generalization/Self-Harness/adapters/webshop/current_active_harness_adapter.py
```

运行脚本：

```bash
/home/mohanz/Harness_Generalization/Self-Harness/adapters/webshop/run_alfworld_best_4b.sh
```

配置文件：

```text
/home/mohanz/Harness_Generalization/Self-Harness/adapters/webshop/assign-alfworld-best-4b.yaml
```

该配置使用 100 条 `webshop-test`，task/agent concurrency 为 2。提交 CPU 控制器和 worker：

```bash
sbatch --partition=all --qos=scavenger --cpus-per-task=4 --mem=16G \
  --time=1-00:00:00 --job-name=webshop-alfworld-4b \
  --output=/home/mohanz/Harness_Generalization/Self-Harness/runs/webshop-alfworld-best-4b/slurm-%j.out \
  /home/mohanz/Harness_Generalization/Self-Harness/adapters/webshop/run_alfworld_best_4b.sh
```

结果文件：

```text
/home/mohanz/Harness_Generalization/Self-Harness/runs/webshop-alfworld-best-4b/qwen3-4b/webshop-test/overall.json
```

## Qwen3.5-9B ALFWorld Meta-Harness

模型快照：

```text
/home/mohanz/.cache/huggingface/models--Qwen--Qwen3.5-9B/
```

下载方式：

```bash
HF_HOME=/home/mohanz/.cache/huggingface \
/home/mohanz/Harness_Generalization/Life-Harness/AgentBench/.venv-vllm-cu128/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id='Qwen/Qwen3.5-9B',
    cache_dir='/home/mohanz/.cache/huggingface',
))
PY
```

9B vLLM 服务脚本：

```bash
/home/mohanz/Harness_Generalization/Self-Harness/jobs/serve_qwen3_5_9b_blackwell.sbatch
```

该服务必须使用 CUDA 12.8/vLLM 0.30 环境和 Blackwell。服务健康检查：

```bash
curl http://unites8.ib:31615/v1/models
```

ALFWorld Meta-Harness 主脚本：

```bash
/home/mohanz/Harness_Generalization/Self-Harness/adapters/alfworld/run_full_alfworld_self_qwen35_9b.sbatch
```

该脚本设置：

```text
QWEN_MODEL = qwen3.5-9b
agent max tokens = 2048
proposer max tokens = 8192
thinking = false
SELF_HARNESS_WORKERS = 4
evolution = 140 tasks x 10 iterations
heldout = 134 tasks
```

正式启动顺序必须是：

```bash
service=$(sbatch /home/mohanz/Harness_Generalization/Self-Harness/jobs/serve_qwen3_5_9b_blackwell.sbatch | awk '{print $NF}')
# 等待 curl health check 返回 qwen3.5-9b 后再提交：
main=$(sbatch /home/mohanz/Harness_Generalization/Self-Harness/adapters/alfworld/run_full_alfworld_self_qwen35_9b.sbatch | awk '{print $NF}')
```

如果服务未健康，不能先启动主任务，否则会产生 connection refused 的失败轨迹。

## GPU 检查和释放

```bash
squeue -u mohanz -o '%.18i %.32j %.10T %.12M %.12l %.20R'
scontrol show job <job_id>
scancel <job_id>
```

释放服务前，先确认结果文件完整，例如：

```bash
wc -l <result>/runs.jsonl
cat <result>/overall.json
```

