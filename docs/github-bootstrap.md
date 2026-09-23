# GitHub bootstrap

The current ChatGPT GitHub connection can edit existing repositories but cannot create a new
repository. Once an empty repository is created, this tree is ready to push without restructuring.

## Option A — GitHub CLI

```bash
cd agent-lab
git init -b main
git add .
git commit -m "feat: bootstrap cumulative LLM and agent engineering lab"
gh repo create agent-lab --public --source=. --remote=origin --push
```

For a private lab, replace `--public` with `--private`.

## Recommended first development branch

```bash
git switch -c feat/phase-00-04-foundation
```

The first PR should keep scope limited to Phase 00-04. RAG/MCP/Agent SDK work belongs in later PRs.
