# Architecture

## Long-lived boundaries

```text
CLI / phase experiments
        |
        +---- ContextPolicy --------- deterministic context selection
        |
        +---- ResponsesProvider ----- model boundary
        |         |
        |         +---- OpenAI Responses API
        |
        +---- ToolRegistry ---------- schema + validation + execution
        |
        +---- TraceWriter ----------- observable JSONL events
        |
        +---- Eval utilities -------- JSONL regression datasets
```

The first five learning topics intentionally establish interfaces that remain useful after RAG,
MCP and a durable agent runtime are added.

## Why the manual loop comes before an Agent SDK

Phase 04 owns the Responses API loop directly so the learner can see the protocol boundary:
model output -> function call -> validated local execution -> function_call_output -> next model
turn. Later phases can replace orchestration with an SDK while keeping tools, traces and evals.

## Trace policy

Record only observable engineering events:

- request metadata and model name
- response id / output text
- tool name, arguments and result/error
- elapsed time
- context item ids and truncation
- termination reason

Do not require, request or store hidden chain-of-thought.

## Future extensions

- Week 5-6: `retrieval/` and citation-bearing RAG
- Week 7-10: `agent/`, MCP client/server, durable state
- Week 11-13: checkpointing, eval harness, security policy
- Week 14-16: local inference, research workflows and production deployment
