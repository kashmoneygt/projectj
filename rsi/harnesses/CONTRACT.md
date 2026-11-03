# Harness contract: what a variant may change

A variant is the unit of improvement: a named package under `variants/<name>`, made from a parent node and run as
that node's successor. The baseline is the variant with no changes, the harness's own package. This contract fixes the
**vocabulary of what a teacher may propose**, so proposals and their diffs compare across harnesses. Each harness maps
the vocabulary onto its own mechanisms in `harnesses/<name>/`, and its `HARNESS.md` says how (the teacher reads it as
`harness.md`). Variants for different harnesses share what they may propose, not the code of their changes.

## The three axes

1. **What the model sees.** Every request holds a system prompt, tool definitions and the messages so far. The first
   two are the harness's: the system prompt (persona, procedure, guardrails, durable memory) and each tool's
   declaration and guidance.
2. **What the model can do.** Tool selection, new tools, edited tools (declaration, call behavior, results,
   implementation), scripts, external tool servers.
3. **What happens between calls.** Enforcement hooks (block, rewrite, forced continuation), compaction, delegation
   (subagents, skills), model routing.

## What a variant may propose

| # | Thing | pi (verified on 1.4.0) | deepseek-harness (verified in source) |
|---|---|---|---|
| 1 | System prompt / persona | `SYSTEM.md` replaces pi's default system prompt, `APPEND_SYSTEM.md` adds to it; `before_agent_start` returning `systemPrompt` | `dsh-system-prompt` sections (agent-scoped override of same-named globals); persona rows patchable |
| 2 | Guardrails | rules in the system prompt; hard enforcement via `tool_call` block | `guard/` family (repeat-tool reminder, per-call timeout — tunable or removable by composition); `tools/pre-execute` |
| 3 | Durable memory | text in the system prompt; an extension that adds messages (`before_agent_start`, `context`) | AGENTS.md/CLAUDE.md chain (`dsh-agent-instructions`, byte-budgeted, refreshed on fs activity); prompt sections |
| 4 | Tool guidance / declarations | a tool's description and prompt snippet (`registerTool`); `prepareLoadout` description rewrite | tool-schema sources in `dsh-system-prompt` (descriptions, ordering) |
| 5 | Tool selection | `settings.json` `defaultTools` (add/remove built-ins; `codemode`, `tool_search`) | tool-registry rows; composition patches |
| 6 | New tool | `registerTool` in a plugin (active on registration, `direct` exposure) | a plugin contributes to the `core/tools` registry |
| 7 | Edit an existing tool | declaration: `prepareLoadout`; call: `tool_call` rewrite/block; result: `tool_result`; implementation: same-name registration shadows the built-in | `tools/pre-execute` / `tools/post-execute`; the tool registry itself is a replaceable plugin |
| 8 | Subagents | `agents/<name>.md` + vendored subagent extension (own context window) | `subagent/` family (in-process and out-of-process children: ACP, Codex, Claude Code) |
| 9 | Skills | `skills/<n>/SKILL.md`, on demand via the `skill` tool | `skill/` family (filesystem providers, `skill` tool, `/name`) |
| 10 | Scripts | `scripts/<n>.py` run via `bash` | `bash`/shell |
| 11 | Enforcement / flow control | `tool_call` block; `turn_end` / `agent_before_settle` forced continuation | `agent/*` events (observe or intercept work in flight); guard timeout policy |
| 12 | Compaction | `settings.json` + custom compaction extension | `compaction/` family (automatic, `/compact`, tool-output trimming) — mount explicitly to tune |
| 13 | External tools (MCP) | `mcp.json` in the agent dir; `registerMcpServer` | `mcp/` family (client + resources, per-scope config, server instructions) |
| 14 | Model routing | fixed while the model is the experiment's constant: `check-variant` rejects `registerProvider` / `registerVirtualModel` | `llm/` adapter seam (replaceable from configuration) |
| 15 | The agent loop itself | not exposed by the extension API — a human-level axis | `core/agent-loop` is a replaceable plugin |

Rows 1–14 are the portable vocabulary, which every supported harness has. Row 15 records the difference between them:
a variant for a more open harness may do what one for a closed harness can't. Each harness's `HARNESS.md` lists its
own gaps.

## Variants must not target the workload

A variant makes the harness better at the agent's kind of work in general, not at the workload its parent happened to
run; the workload is a probe, not the goal. Three kinds of change, in order of preference:

1. **Invariant**: fixes a defect that recurs on any workload, in turn structure, tool use, context management,
   verification or recovery.
2. **Domain**: knowledge or procedure for the task family, kept in an on-demand skill so it never weighs on every call.
3. **Protocol**: instructions that only make sense under one evaluation protocol (how many tasks the model gets, what
   token it must reply with, which files it must not touch). The protocol carries its own rules, so putting them in the
   harness is invalid.

The runner catches the obvious cases: `rsi variants generate` rejects candidates whose agent-visible files name the
workload or its protocol. The rest is up to the teacher and to the people reviewing.

## Out of a variant's reach

- **Model parameters**, everything `models.toml` pins (sampling, the context window, the values the chat template
  reads such as thinking and its effort, the output cap). They belong to the model axis and stay fixed so generations
  compare. How the model thinks within them is the harness's to change: what it's told, what it sees, and how long it
  may think per request (a thinking budget, where the harness and server support one).
- **The environment** (container, repo state, network): the benchmark protocol's.
- **The evaluation contract**: the instrumentation (metrics that only read each request), the provider config and the
  per-request output cap, the task rules, the per-task wall clock and the one-shot execution mode. The baseline is the
  harness as it ships, with no turn limit and no settings of the runner's own. A run still going at the wall clock is
  stopped and scored on whatever the repo holds.

Each harness pins the contract as far as its extension model allows:

- **pi**: on every start the runner installs the metrics extension and `models.json` into the task's agent dir,
  outside the read-only variant package, and mounts both read-only (pi writes its own files next to them). It enforces
  the wall clock whatever the variant loads. pi's extension API can still reach the request
  (`before_provider_request`), the model and thinking level (`setModel`, `setThinkingLevel`, providers) and the
  instrumentation's environment, all in-process, so `check-variant` rejects packages that use those surfaces, and
  `settings.json` only takes the keys about how the agent works and the display (no model, provider or thinking
  level, no proxy, no code or files from outside the package). Every request is audited where it leaves the
  container: the model proxy on the sandbox network logs each one's parameters (not its messages) and the usage the
  server reports, and each task keeps its container's share in `requests.jsonl`. So nested calls
  (`ctx.modelRegistry`) and second `pi` processes are checked against the pinned spec like the agent's own requests
  (`contract_violations` on each task, which also flags keys pi doesn't send) and count toward the task's requests and
  tokens. A thinking budget goes through pi's own `thinkingBudgets` setting: the runner names the server's budget
  field in `models.json` only for a variant whose `settings.json` sets `thinkingBudgets.medium`, the level runs think
  at, so the baseline sends none, as plain pi does.
- **deepseek-harness** (design): the runner mounts the instrumentation in a composition layer the variant's patches
  can't override (layers apply in a fixed order and the top one wins), and the composition is verified at boot.

## Harnesses

| Harness | A variant is | Execution mode | Status |
|---|---|---|---|
| [pi](https://github.com/earendil-works/pi) 1.4.0 | a directory of files (markdown, config, TS extensions); the runner stages the runtime subset (allowlist, `harnesses/pi/check_variant.py`) read-only at `/opt/harness`, so the variant's card (VARIANT.md) never reaches the model; see [pi/HARNESS.md](pi/HARNESS.md) | `pi --print` (same agent and session mechanism as the TUI) | implemented |
| [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) | an out-of-tree plugin plus a `cordis.yml` patch, installed into a profile (`dsh plugin add` / `--patch`) | `dsh --profile headless` (one task per invocation, `--json` event stream, `--session-id`) | designed, not implemented |

Markdown and config (rows 1–5, 9–10) carry over between harnesses nearly as they are; plugins (rows 6–8, 11–14) are
written again in each harness's extension language.
