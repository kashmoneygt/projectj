# How a variant runs in pi

Your variant is a pi package: a folder of files that pi 1.4.0 loads on top of its defaults. `parent/` is the
package the parent ran. The baseline package adds nothing, so the baseline is pi as it ships: its own system
prompt, its four tools (`read`, `bash`, `edit`, `write`) and its own defaults for everything else, like
compaction and retries.

## How a task runs

Each task runs `pi --print -e /opt/harness --mode json` in a container that starts in the task's repo, with your
package at `/opt/harness`. The agent gets one message, the task as a person would ask for it, and pi adds nothing to
it. Each transcript in `/work` starts with that message. Nobody answers the agent's questions. The run ends when the
agent stops calling tools, and there's no turn limit. Each task has a time limit the agent isn't told about, and a
run still going then is stopped. Whatever the agent left in the repo, new files included, is what gets judged.
`overview.md` gives that limit, the context window and the most tokens one response may have.

## What a package can hold

- `SYSTEM.md` - replaces pi's system prompt. pi still adds `APPEND_SYSTEM.md`, the skills list and the working
  directory after it
- `APPEND_SYSTEM.md` - added to the end of the system prompt
- `extensions/*.ts` - TypeScript extensions. They can add tools, change or replace the built-in ones, and hook into
  every stage of a run (see where a change can go, below)
- `skills/<name>/SKILL.md` - skills. pi lists each one's name and description in the system prompt, and the
  model reads the rest when it decides to
- `settings.json` - pi's settings about how the agent works, like which built-in tools are on, compaction and
  retries
- `mcp.json` - MCP servers. The container has no network, so a server has to run inside it
- `scripts/` - files the agent can run. The package can be installed anywhere, so an extension finds them from its
  own path (`import.meta.url`), and a skill keeps its files in its own folder and names them relative to `SKILL.md`
- `agents/*.md` - copied to `agents/` in pi's agent dir. pi doesn't read them itself, but an extension can,
  like pi's example subagent extension
- `package.json` - the manifest. Every `extensions/*.ts` and `skills/*/` loads through its globs. Leave it as it is,
  since your card in `VARIANT.md` says what changed

Nothing else reaches the agent. `VARIANT.md` is your card for the people who run the variants, and the agent never
sees it.

## Where a change can go

- what the model is told - `SYSTEM.md`, `APPEND_SYSTEM.md`, or `before_agent_start` (prompt sections, active
  tools). It's read on every request
- its tools - which ones it has, and what each is called, takes, does and returns: `defaultTools` in
  `settings.json` (the built-ins are `read`, `bash`, `edit`, `write`, `grep`, `find` and `ls`, plus `codemode` and
  `tool_search`), `registerTool` (a built-in's name replaces it) and `prepareLoadout`
- a tool call or what comes back - `tool_call` changes the input or blocks the call, `tool_result` adds to the
  result or replaces it
- a finished response - `message_end` can replace it, and its role stays
- what of its history the model sees - `context` and `context_with_system`, on every request, and the compaction
  settings
- when the run ends - `agent_before_settle` can add a message and ask for one more turn, and `turn_end` runs after
  each turn
- know-how for when it needs it - skills, which only help when the model reads them
- work in a fresh context - a second `pi` process that an extension starts

`packages/coding-agent/examples/extensions/` has working examples of most of these, like `tool-override.ts`,
`truncated-tool.ts`, `permission-gate.ts`, `custom-compaction.ts` and `subagent/`.

Every change has a cost. Text the model gets on every request is paid for on every request. Changing earlier
messages makes the model server work through them again, since it reuses the unchanged start of the last request.
There's one model server, so a second `pi` process waits its turn and adds time. A call to the model from an
extension or a second process goes through the same server with the same settings, and counts toward a run's cost
like the agent's own.

Whenever your change kicks in (it blocks or rewrites a tool call, adds a message, asks for one more turn), record it
with `pi.appendEntry("acted", { what: "<a few words>" })`. The model never sees it. Each run counts these, and
`tree.md` shows on how many runs each variant kicked in, so a variant that never did isn't mistaken for an idea that
failed.

## pi's docs and source

`/opt/pi-src` is pi's repo at 1.4.0. Read it instead of guessing, and once you know what you want to build,
read only what building it needs:

- `packages/coding-agent/docs/` - pi's docs, listed in `index.md`. `extensions.md` covers the run lifecycle,
  and its "Follow the extension contracts" says what each event's handler may change
- `packages/coding-agent/examples/extensions/` - working extensions to copy from
- `packages/coding-agent/src/core/extensions/types.ts` - the exact type of every event and result
- `packages/ai/src/types.ts` - the messages an extension sees. The transcripts flatten their parts into text, so
  take field names from here
- `packages/agent/src/agent-loop.ts` - the agent loop itself

## What's fixed

These stay as they are. Every request to the model is audited where it leaves the container, however it's made.
`check-variant` rejects the uses named here, and any other way of changing them is out of bounds too:

- the model and how it's served: the model, its sampling, what its chat template is told (thinking and its
  effort) and the most tokens one response may have, with no separate cap on thinking. Runs think at pi's default
  level, `medium`. So extensions can't use `before_provider_request`, `before_provider_headers`,
  `registerProvider`, `registerVirtualModel`, `setModel` or `setThinkingLevel`, and `settings.json` only takes
  settings about how the agent works, not its thinking level or a thinking budget (`check-variant` names any key it
  won't take)
- how runs are measured: a metrics extension that reads each request and never changes it, and its `RSI_*`
  environment variables
- the task message, the container and the repo it starts from

## Checking a package

`check-variant <dir>` makes sure the package has its card, stages it exactly the way a run does, and tells
you about files that never reach the agent, a changed manifest, extensions the manifest doesn't load, uses of the
fixed APIs, words in what the agent sees that name the workload or how it's judged, and whether pi starts with the
package (against a model that never answers). The same checks run when you hand a variant in, and a variant that
fails them is never run.

## Trying a package

`try-variant <dir> <repo> <task file>` runs a package the way a task run does, against the agent's own model: pi
starts in a copy of the repo with the package staged, and the agent gets the task file's text as its one message, so
write it the way a person would ask for the work. It prints each model response's tool calls, every message the
package added, how the run ended and what the run changed, and keeps the run's record under `/out/tries`. The tasks'
repos aren't in your container, so build a small repo in `/tmp` that recreates the moment your change is for (pytest
is installed). A run takes minutes, like a real task, and is stopped after 540 seconds unless you pass `--timeout`.
Run a few to see that your change fires when it should and stays quiet when it shouldn't.

`try-variant --question <dir>` asks the agent a question about a small JavaScript repo, one that needs no change.
When you hand a variant in, it's asked this question and one like it about a Python repo. It's rejected if, on two
tries in a row, the package keeps the agent going after it has answered, the run leaves the repo changed (scratch files
included), or pi's requests fail while the model answers a plain one. So the agent's answer has to be its last
message, in plain text, and a package may ask for another turn only once the run has changed something. Anything else
gets in the way of someone who just asks.
