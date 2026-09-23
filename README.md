# agent-lab

`agent-lab` is a cumulative 16-week LLM/Agent engineering project. Each phase adds reusable
infrastructure to the same codebase; exercises are not throwaway demos. The end state is a
**Personal Engineering Research Agent** that can retrieve evidence, call tools, trace every
observable action, run regression evals, and safely write results back to engineering systems.

## Phase 1 scope: 00 → 04

| Phase | Topic | Deliverable |
|---|---|---|
| 00 | LLM API quickstart | Responses API adapter + local trace |
| 01 | Transformer fundamentals | NumPy attention + causal-mask experiment |
| 02 | Tokenizer / sampling / inference | token-count utility + sampling experiment |
| 03 | Context engineering | deterministic context packer + regression cases |
| 04 | Structured tools | strict tool registry + manual tool loop + failure handling |

**Deliberately out of scope for Phase 1:** RAG, vector databases, MCP servers, LangGraph,
multi-agent orchestration and write-capable production tools. Those build on this foundation.

## Architecture

```text
experiments/phaseXX       # learning entry points; thin by design
        |
        v
+------------------- src/agent_lab --------------------+
| config | context | trace | evals                     |
|          |                                             |
|          +--> llm/OpenAI Responses adapter             |
|          +--> tools/registry + strict contracts        |
+--------------------------------------------------------+
        |
        v
Week 5+: RAG -> MCP -> Agent runtime -> eval/security -> production
```

## Setup

```bash
git clone <your-agent-lab-url>
cd agent-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,tokens]'
cp .env.example .env
```

Set `OPENAI_API_KEY` and `AGENT_LAB_MODEL` only when running online experiments. Offline
experiments and unit tests require no API key.

## Run

```bash
# Offline, deterministic
agent-lab phase01
agent-lab phase02 --text "FOC current loop and EtherCAT CSP"
agent-lab phase03

# Online
agent-lab phase00 "Explain KV cache in 5 bullet points"
agent-lab phase04 "Calculate (17.5 * 8) / 7, then count the characters in the result"

# Quality gate
make check
```

Local traces are written under `runs/` as JSONL and are intentionally gitignored.

## Design rules

1. **No hidden reasoning dependency.** Persist observable inputs, outputs, tool calls, tool
   results, latencies and termination reasons—not private chain-of-thought.
2. **Tools are contracts.** JSON schema is strict, arguments are validated, execution errors
   are data, and side effects are explicit.
3. **Offline tests first.** Model calls are behind an adapter so core behavior is testable
   without network access.
4. **Evals before cleverness.** Changes to prompts/context/tool schemas must be regression-tested.
5. **Least privilege by construction.** Phase 1 ships only read-only/local deterministic tools.

See [`docs/roadmap.md`](docs/roadmap.md) and [`docs/architecture.md`](docs/architecture.md).
