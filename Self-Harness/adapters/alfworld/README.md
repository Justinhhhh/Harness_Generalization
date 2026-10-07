# Self-Harness ALFWorld adapter

Self-Harness remains the proposer, diagnosis, and acceptance loop. AgentBench
remains the ALFWorld episode executor. This adapter runs both ALFWorld splits
and converts verified `runs.jsonl` results into Self-Harness `result.json`.

The adapter loads candidates through `ALFWORLD_HARNESS_FILE`; AgentBench's
legacy h2/h3/h4/h5 harness is not enabled by this path.
