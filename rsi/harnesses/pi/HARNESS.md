# How a variant runs in pi

Your variant is a pi package: a folder of files that pi 1.5.2 loads on top of its defaults. `parent/` is the
package the parent ran. The baseline package adds nothing, so the baseline is pi as it ships: its own system
prompt, its four tools (`read`, `bash`, `edit`, `write`) and its own defaults for everything else, like
compaction and retries.

## How the agent runs

Every run is `pi --print -e /opt/harness --mode json` in the task's own container, started in its working folder,
with your package at `/opt/harness`. Tasks come from all kinds of work: a project with an issue to resolve or a
question about its code, a machine with a service or build to fix, data to analyze. So the folder may or may not be a
git repo or have tests, the code can be in any language, and some tasks have the internet while others don't. pi runs
on its own copy of node, which is last on the task's `PATH`, so `node` there may be the task's own.

The agent gets one message, the task as a person would ask for it, and pi adds nothing to it. Nobody answers the
agent's questions. The run ends when the agent stops calling tools, or once it has made 100 model requests, like a
user who stops waiting (the model isn't told). A run that goes on too long is stopped too, along with everything it
started. Then the task is judged on what the agent left behind: its files, anything it left running, and its last
message, which is its answer when the task asks a question. When git is in the container, what the agent changed in
its working folder, new files included, is the run's diff.

## What a run leaves

Each task's folder under `/work/runs/` is named after the task's kind of work plus a short hash, so the same task has
the same folder name in every run. It holds two files from the run:

- `session.jsonl` - pi's own record of the session, one JSON object per line: the system prompt pi built, the task
  message, each response with its thinking and tool calls, each tool result, and each record a package made with
  `appendEntry`. pi's `packages/coding-agent/docs/session-format.md` describes it
- `metrics.jsonl` - what the metrics extension saw in the agent's own pi (not in a second pi an extension starts): the
  system prompt and tools as they were sent (`"type": "context"`), the size of each request, and each response's
  timing (`"type": "timing"`)

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
- `mcp.json` - MCP servers. Some tasks have no network, so a server has to run inside the container
- `scripts/` - files the agent can run. The package can be installed anywhere, so an extension finds them from its
  own path (`import.meta.url`), and a skill keeps its files in its own folder and names them relative to `SKILL.md`
- `agents/*.md` - copied to `agents/` in pi's agent dir. pi doesn't read them itself, but an extension can,
  like pi's example subagent extension
- `package.json` - the manifest. Every `extensions/*.ts` and `skills/*/` loads through its globs. Leave it as it is,
  since your card in `variant.json` says what changed

Nothing else reaches the agent. `variant.json` is your card for the people who run the variants, and the agent never
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
- work in a fresh context - a second `pi` process that an extension starts. Start it with `process.execPath` and
  `process.argv[1]`, since the `node` on the task's `PATH` may be the task's own

`packages/coding-agent/examples/extensions/` has working examples of most of these, like `tool-override.ts`,
`truncated-tool.ts`, `permission-gate.ts`, `custom-compaction.ts` and `subagent/`.

A tool you register under a built-in's name doesn't get the settings people gave pi for it. So build yours on pi's
own and pass them on, like `createBashToolDefinition(cwd, { shellPath, commandPrefix })` with `getShellPath()` and
`getShellCommandPrefix()` from `SettingsManager.create(cwd)`.

Every change has a cost. Text the model gets on every request is paid for on every request. Changing earlier
messages makes the model server work through them again, since it reuses the unchanged start of the last request.
There's one model server, so a second `pi` process waits its turn and adds time. A call to the model from an
extension or a second process goes through the same server with the same settings, and counts toward a run's cost
and its 100 requests like the agent's own.

Whenever your change kicks in (it blocks or rewrites a tool call, adds a message, asks for one more turn), record it
with `pi.appendEntry("acted", { what: "<what your change did, in a few words>" })`. The model never sees it. Text
that's in every request, like a prompt section, needs no record. Each record shows up in the run's `session.jsonl`,
and `tree.json` says how often each variant kicked in, so one that never did isn't mistaken for an idea that failed.

## pi's docs and source

`/opt/pi-src` is pi's repo at 1.5.2. Read it instead of guessing, and once you know what you want to build,
read only what building it needs:

- `packages/coding-agent/docs/` - pi's docs, listed in `index.md`. `extensions.md` covers the run lifecycle,
  and its "Handler Composition" says what each event's handler may change
- `packages/coding-agent/examples/extensions/` - working extensions to copy from
- `packages/coding-agent/src/core/extensions/types.ts` - the exact type of every event and result
- `packages/ai/src/types.ts` - the messages an extension sees, with the same fields as in `session.jsonl`
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
- how runs are measured and stopped: a metrics extension that reads each request and never changes it, an extension
  that ends the run at its request budget, and their `RSI_*` environment variables
- the task message, the container and the folder it starts in

## Checking a package

`check-variant <dir>` makes sure the package has its card and stages it exactly the way a run does. Then it
tells you about:

- files that never reach the agent
- a changed manifest, or extensions the manifest doesn't load
- uses of the fixed APIs
- words in what the agent sees that name the workload or how it's judged
- whether pi starts with the package (against a model that never answers)

The same checks run when you hand a variant in, and a variant that fails them is never run.

## Trying a package

`try-variant <dir> <repo> <task file>` runs a package much like a task run, against the agent's own model but without
the request budget. pi starts in a copy of the repo with the package staged, and the agent gets the task file's text
as its one message, so write it the way a person would ask for the work. It prints each model response's tool calls,
every message the package added and every `acted` record, how the run ended and what the run changed, and keeps the
run's record under `/out/tries`. Build a small repo yourself in `/tmp` that sets up the problem your change is for, or
the kind of work you want to watch (pytest and node are installed). A run takes minutes, like a real task, and is
stopped after 540 seconds unless you pass `--timeout`. Run a few to see that your change fires when it should and
stays quiet when it shouldn't. A background job dies when its tool call ends, so to run tries side by side, start them
in one call and `wait`, or use `setsid`. Stop any server a try leaves running.

`try-variant --question <dir>` asks the agent a question about a small JavaScript repo, one that needs no change.
When you hand a variant in, it's asked this question and one like it about a Python repo. It's rejected if, on two
tries in a row, the package keeps the agent going after it has answered, the run leaves the repo changed (scratch files
included), or pi's requests fail while the model answers a plain one. So the agent's answer has to be its last
message, in plain text, and a package may ask for another turn only once the run has changed something. Anything else
gets in the way of someone who just asks.
