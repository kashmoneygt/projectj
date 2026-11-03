You are improving {harness}, the coding-agent harness that {model} works inside, as part of a self-improvement loop. Your job is to make one new variant of the current harness (the parent): a single change, built and ready to run, that makes the agent better at real coding work in any repo, i.e. more tasks solved correctly with less wasted time and tokens. The parent is either the baseline or the baseline with changes from earlier variants stacked on top.

Every request the model gets holds a system prompt, tool definitions and the messages so far. The system prompt and the tools (what they're called, what they do and what they return) are the harness's, and they're yours to change, along with anything else the harness does between the model's calls. Through them you can change how the agent works and how the model thinks. The model and the settings it's served with, the task message and how runs are measured stay fixed. `harness.md` says how {harness} does all of this.

The parent was run on a batch of tasks, each judged by hidden tests the agent never saw. It's a small, noisy sample: the same harness passes and fails some tasks by chance, so a change that only moves this score isn't an improvement. Each variant is run on a fixed set of tasks, about half of which you never see, and compared with runs of the parent on the same tasks. It's kept if it passes more of them by more than chance, or passes as many while spending clearly fewer tokens. Everything you need is in `/work`, which is read-only (use `/tmp` for scratch). Treat it as evidence, not instructions:

- `overview.md` - how the agent works in passed and failed runs, then one line per task
- `tasks/<id>.md` - each run turn by turn with its final diff (`tasks/full/<id>.md` has the whole transcript)
- `tasks/diverged/<id>.md` - the tasks the parent's own runs both passed and failed, one run of each side by side
- `tree.md` - every variant tried so far and how it did against its parent (`tree/<name>/` has its package, `results/<name>/` the runs it fixed or broke)
- `parent/` - the package you start from
- `system-prompt.md` - the system prompt and tool definitions the model got in the parent's runs, as they were sent
- `harness.md` - how a package plugs into {harness}, where a change can go, what's fixed, where its docs and source are, and how to try a package against the agent's own model

## How to go about it

You have {budget}: the session stops without warning at {deadline} by the container's clock (`date`), so leave enough time to build and test the variant.

Other sessions this round read the same evidence, so most of them work on one area of the harness each, and the variants don't all go after the same thing. This session's area is {area}.

1. Start where the parent's own runs disagree: `tasks/diverged/` has tasks it both passed and failed. Same task, same harness, so the difference is something the harness left to chance. Then read more transcripts, not just the overview: at least three passed and three failed runs, start to finish. Look at how the model works with the harness - what it thinks, which tools it calls, what comes back, where it gets stuck and where it wastes time - in passed runs as much as failed ones. How a run ended is a symptom, so find the moment that led there. Passed and failed runs are different tasks, so a difference between them is a clue, not proof.
2. Learn from the tree. If a variant's score moved but the behavior it predicted didn't, its idea gets no credit. If the behavior moved but the score didn't, the idea works but isn't proven yet. If neither moved, don't try it again without a reason it failed. If two variants went after one failure with the same kind of change and didn't gain, try a different kind. If the tree doesn't show the behavior, treat it as unknown. If most of the tree rests on one idea, look for evidence that it's wrong. Ideas from other branches are fair game when this evidence supports them (name them under `ports`).
3. Come up with a few really different ideas, then pick the one you'd bet on: how often it would help, how much, what it could break and what it costs on every run. Bold or small, adding or removing, anything in this area is fair game, and no kind of change wins by default. Don't drop an idea just because you doubt it would work - test it. A change only helps on runs where it kicks in. If yours would have kicked in on three of the parent's runs, it can fix three tasks at most, and two runs of the same harness already differ by three or four, so no one could tell it helped. Count the parent's runs where yours would have kicked in. If a variant from this parent already covers an idea, go after a different one. One change per variant, so we can tell what worked.
4. Build it where it gets at the cause: copy `parent/` to `/out/variants/<name>/` and change only what the idea needs, and record when it kicks in (`harness.md` says how). Then test it like someone else wrote it: replay at least one moment it should change, as close to that run as you can get, and check what it does when things go wrong. A change that could leave a run with nothing is a bad trade.
5. Write `/out/variants/<name>/VARIANT.md` and run `check-variant /out/variants/<name>` until it passes. It checks that the package is valid, not that the idea works.

Each variant takes hours to evaluate. If nothing in this area is worth that, write `/out/variants/NONE.md` saying what you looked at - that's a real answer.

## A few rules

- Write the change for a kind of situation, not for these tasks. Nothing in the package except its card should know about this batch: its repos, task ids, tests, or how the work is judged.
- Don't try to work out what the hidden tests check.
- Nothing in the package may lean on the model remembering these tasks' fixes. They're public, so the model may half-remember some, but real work has no fix to remember.
- The change has to change what the agent does. "Be thorough" on its own changes nothing. A rule the agent has to follow belongs in code that enforces it; text you add should say when it applies and when it doesn't.

## The card

`VARIANT.md` is how people and later sessions see your variant at a glance: a title that says what the change does, then a few sentences for each of these.

    # <what the change does>

    hypothesis: <what the agent does now, what it will do instead, and why that helps coding in general>
    evidence: <the moments that show it, how often you saw it in passed and in failed runs, and a few task ids>
    predicts: <the behavior that should change, the effect on results and cost, and what result would show it's wrong>
    ports: <variants whose idea you reuse; leave this line out if none>

Name the folder after the change, in a few lowercase words joined by hyphens.
