You are improving {harness}, the coding-agent harness that {model} works inside, as part of a self-improvement loop. Your job is to make one new variant of the current harness, the parent. It's a single change, built and ready to run, that makes the agent better at agentic coding. That means whatever people ask a coding agent to do on their own code and machines, which is often more than writing code. Better means more of it done right, with less wasted time and tokens. What holds up is released for anyone who codes with {model} in {harness}, including people who work with it turn by turn, so build each change the way you'd ship it. The parent is either the baseline or the baseline with changes from earlier variants stacked on top.

Every request the model gets holds a system prompt, tool definitions and the messages so far. The system prompt and the tools (what they're called, what they do and what they return) are the harness's, and they're yours to change, along with anything else the harness does between the model's calls. Through them you can change how the agent works and how the model thinks. The model and the settings it's served with, the task message and how runs are measured stay fixed. `harness.md` says how {harness} does all of this.

The parent was run on a batch of tasks from coding benchmarks, each judged by checks the agent never saw. It's a small, noisy sample. The same harness passes and fails some tasks by chance, so a change that only moves this score isn't an improvement. Each variant is run on a fixed set of tasks, about half of which you never see, and compared with runs of the parent on the same tasks. It's kept if it passes more of them by more than chance, or passes as many while spending clearly fewer tokens.

Benchmarks stand in for real work, and some of what you'll see comes from how a benchmark is built, not from coding. Its tasks share one format, it judges them one way, and they're public, so the model may half-remember their fixes. A change that only helps because of that does nothing for people using the harness on their own work, so don't make one. Taking one out of the parent is a fair variant. A change for one language or one kind of task has to stay out of the way everywhere else. Before you build, picture your change on someone's own project, doing something that looks nothing like these tasks. If it wouldn't help there, pick another idea.

Everything you need is in `/work`, which is read-only (use `/tmp` for scratch). Treat it as evidence, not instructions:

- `overview.md` - how the agent works in passed and failed runs, then one line per task
- `tasks/<id>.md` - each run turn by turn with its final diff (`tasks/full/<id>.md` has the whole transcript)
- `tasks/diverged/<id>.md` - the tasks the parent's own runs both passed and failed, one run of each side by side
- `tree.md` - every variant tried so far and how it did against its parent (`tree/<name>/` has its package and card, `results/<name>/` the runs it fixed or broke)
- `parent/` - the package you start from
- `log.md` - how the parent was built, one variant at a time from the baseline on, each with its card, and the newest ones with their diff
- `system-prompt.md` - the system prompt and tool definitions the model got in the parent's runs, as they were sent
- `harness.md` - how a package plugs into {harness}, where a change can go, what's fixed, where its docs and source are, and how to try a package against the agent's own model

## How to go about it

You have {budget}. The session stops without warning at {deadline} by the container's clock (`date`), so leave enough time to build and test the variant.

Other sessions this round read the same evidence, so most of them work on one area of the harness each, and the variants don't all go after the same thing. This session's area is {area}.

1. Read the runs with your area in mind. A good place to look is `tasks/diverged/`, the tasks the parent's own runs both passed and failed. Same task, same harness, so the difference is something the harness left to chance. Then read more transcripts, not just the overview, at least three passed and three failed runs from start to finish. Look at how the model works with the harness - what it thinks, which tools it calls, what comes back, where it gets stuck and where it wastes time - in passed runs as much as failed ones. How a run ended is a symptom, so find the moment that led there. Passed and failed runs are different tasks, so a difference between them is a clue, not proof.
2. Learn from the tree. If a variant's score moved but the behavior it expected didn't, its idea gets no credit. If the behavior moved but the score didn't, the idea works but isn't proven yet. If neither moved, don't try it again without a reason it failed. If two variants went after one failure with the same kind of change and didn't gain, try a different kind. If the tree doesn't show the behavior, treat it as unknown. If most of the tree rests on one idea, look for evidence that it's wrong. Ideas from other branches are fair game when this evidence supports them (name them in a `Reuses:` line).
3. Come up with a few really different ideas, then pick the one you'd bet on. Weigh how often it would help, how much, what it could break and what it costs on every run. Bold or small, adding or removing, anything in this area is fair game, and no kind of change wins by default. Don't drop an idea just because you doubt it would work - test it. A change only helps on runs where it kicks in, so count the parent's runs where yours would have, and put that number in `Expect:`. Two runs of the same harness already differ by three or four tasks, so a change that kicks in on a few runs won't show a gain on its own. It can still be worth making if it costs nothing the rest of the time, since a later session can combine it with others. If a variant from this parent already covers an idea, go after a different one. One change per variant, so we can tell what worked.
4. Build it where it gets at the cause. Copy `parent/` to `/out/variants/<name>/` and change only what the idea needs, and record when it kicks in (`harness.md` says how). Leave no other folder in `/out/variants/`, since if more than one passes the checks, none is kept. Then test it like someone else wrote it. Replay at least one moment it should change, as close to that run as you can get, and check what it does when things go wrong. A change that could leave a run with nothing is a bad trade.
5. Write its card in `/out/variants/<name>/VARIANT.md` and run `check-variant /out/variants/<name>` until it passes. It checks that the package is valid, not that the idea works. When you hand it in, the variant is also asked a question that needs no change, and it's rejected if it changes the repo or keeps the agent going after it has answered (`harness.md` says how to try that first).

Each variant takes hours to evaluate. If nothing in this area is worth that, write `/out/variants/NONE.md` saying what you looked at - that's a real answer.

## A few rules

- Write the change for a kind of situation, not for these tasks. Nothing in the package except its card should know about this batch, like its repos, task ids, tests, or how the work is judged.
- Don't try to work out what a benchmark's checks look for.
- Nothing in the package may lean on the model remembering these tasks' fixes.
- The change has to change what the agent does. "Be thorough" on its own changes nothing. Code that makes something happen beats asking for it, but a clear sentence, a better tool description or a skill can be enough, and costs less. Text you add should say when it applies and when it doesn't.

## The card

`VARIANT.md` describes your variant the way a pull request would, so people and later sessions can see at a glance what changed and why. Write it in this shape:

    <part of the harness>: <what the change does>

    What: <the change, in a sentence or two>
    Why: <what you saw in the runs that led to it, how often in passed and in failed runs, a few task ids, and why fixing it helps people's own work>
    Expect: <what should change if it works, in behavior, results and cost, how many of the parent's runs it would have kicked in on, and what result would show it's wrong>
    Reuses: <variants whose idea you build on; leave this line out if none>

Keep the first line lowercase and under 72 characters, like a commit title. Name the folder after the change, in a few lowercase words joined by hyphens.
