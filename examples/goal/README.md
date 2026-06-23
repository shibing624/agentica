# Goal — `run_goal()` Loop Examples

`run_goal()` drives the agent in a loop until the task is **complete**,
**paused**, or a **budget** (turn / token / wall-clock) is exhausted.

Each file is self-contained — runs with one API key (`DEEPSEEK_API_KEY`).

| File | Pattern | What you'll see |
|---|---|---|
| [`task_runner.py`](./task_runner.py) | **Fan-out**: read `tasks.jsonl` → many independent `run_goal()` concurrently → `task_results.md` | Each task is its own goal with its own judge loop and its own budget; all run side-by-side |
| [`task_dag_runner.py`](./task_dag_runner.py) | **Single DAG**: one multi-phase `run_goal()` loop; the agent decides parallel vs serial itself | Same-turn tool calls run in parallel; cross-turn wait for deps; loop ends on judge/budget |

The two demos show the two axes of concurrency:

- **task_runner.py** — concurrency *across* goals (N independent goals at once).
- **task_dag_runner.py** — concurrency *within* one goal (a DAG of tool calls).

`tasks.jsonl` is one task object per line, so parsing is a single
`json.loads` — no regex. Each line may carry its own budget, because
independent goals are independently configurable:

```jsonl
{"task": "...", "turn_budget": 3}
{"task": "..."}
```

## Run

```bash
export DEEPSEEK_API_KEY="sk-..."

python examples/goal/task_runner.py
python examples/goal/task_dag_runner.py
```

## Key concepts

- **Same turn = parallel.** The Runner runs all tool calls from one LLM
  response via `asyncio.gather`. No user-side orchestration needed.
- **Cross turn = serial.** Natural dependency — you must receive turn N
  results before issuing turn N+1 calls.
- **run_goal() isolates each call.** Internally calls `agent.clone()`, so
  concurrent invocations don't share mutable state.