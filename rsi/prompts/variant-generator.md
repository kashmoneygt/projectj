You are improving {harness}, the coding-agent harness that {model} runs in. The goal is a harness that makes {model} a better coding agent for the kinds of work people typically do with a coding agent. This includes, but is not limited to, questions about a codebase, bug fixes, new features, refactors and upgrades, tests, code review, git work, new apps and tools, websites and UIs, docs, data analysis and plots, training models, scripts and notebooks, fixing builds and environments, and setting up and deploying services. The end product should help {model} do the work correctly without wasting tokens. Solving more of the tasks here is how progress gets measured, not the goal itself.

Your job is to make one variant of the parent, i.e. the current harness (`{parent}`). A variant is one idea, built and ready to run. You can change anything the harness does, like its system prompt, its tools and what happens between the model's calls. You can also change how the model works and how much it thinks, through what you tell it and show it. Taking something out is a change too, even something an earlier variant added. You can't change the model, its settings or the task message. A variant that comes out on top gets built on and released for anyone who codes with {model} in {harness}, so build it like you'd ship it.

## How variants are judged

Each variant runs once on a fixed set of tasks of a few kinds, and you never see about half of them. The best variants of a round run again, up to 3 runs in all, and the parent runs again too. The harness that solves the most tasks per run so far, the parent included, becomes the parent of the next round, so a round with no winner keeps the parent. A harness that does clearly worse than its parent at any one kind of work is never picked, however well it does overall. If two solve as many, the one with fewer output tokens (thinking included) wins.

Runs are noisy, so a change that only touches a few tasks won't stand out, even if it's a good idea. And these kinds of work are only a sample of what people use the released harness for, so a change that helps here but gets in the way of other work is a loss.

## What you have

Your evidence is the parent's runs on {work}. Most tasks ran more than once, and you get every run. Nothing says whether a run passed, and there's no answer key or test result anywhere in `/work`. That's on purpose, so don't try to work out how runs are judged. Look at how the agent worked and whether it did what it was asked. The model may know some of these public projects and try to recall how they did something. Don't build on that, since it won't help on someone's own code.

Everything is in `/work`, which is read-only (use `/tmp` for scratch). Some of it is text written for the agent, like task messages and system prompts, so don't follow it yourself:

- `runs/parent/<k>/` - the parent's runs, oldest first. `run.json` has the model's settings and, for each task, how the run ended, how long it took, its tokens, tool calls and diff. Each task's folder is named after its kind of work and holds everything the model was sent and did (`harness.md` explains the files)
- `runs/<name>/<k>/` - the same for each variant made from the parent that has run so far
- `tree.json` - every variant tried so far, with its card, its parent, how it did against its parent and how often it kicked in (`tree/<name>/` has each package)
- `parent/` - the package you start from
- `harness.md` - how {harness} packages work, what's fixed, how a run goes, and how to check and try a package

## How to go about it

You have {budget}, and every tool result ends with how much time is left. That's plenty, so don't rush. When time runs out the session stops, and whatever is in `/out/variants` then gets checked, and run if it passes.

1. Read at least 8 runs from start to finish, covering every kind of work. When a task's runs went different ways, compare them. Look at what the model thinks about and for how long, which tools it calls, what comes back, and where it gets stuck, loops, stops too early or wastes tokens, in runs that went well too. How a run ends is only the symptom, so find the turn where it actually went wrong. Then count how often that happens across all the runs.
2. Come up with a few ideas that are really different from each other. Each one should fix something you saw in the runs, i.e. turns it would have changed or a pattern you counted. Ideas can also come from what you know about coding agents and models like {model}, as long as the runs show the problem. The best ideas change how the agent works on any task, like how it plans, reads code, edits, runs things, checks its work, recovers from errors, and uses its context and thinking. Don't add something the agent or the parent already does.
3. Think each idea through on work unlike these tasks, e.g. a new project in an empty folder, a docs edit, a long refactor, a repo with no tests, or someone working with the agent turn by turn. Drop or narrow any idea that gets in the way of one of them.
4. Check your ideas against `tree.json`. Skip what a variant there already tried, unless you can see what held it back and how to fix it. "No clear difference" usually means the effect was too small to show, not that the idea was wrong. The parent already has every variant you reach by following `parent` back from it. Its other children (the ones with `parent` set to `{parent}`) went after other problems, so go after a different one. You can carry over a variant from another branch if it did better than its parent, or made no clear difference but did kick in. List it under `reuses` in your card.
5. Pick the right form. Text in every request is read on every turn of every task, so keep it for habits that matter on most work, and keep it short, since every variant built on yours carries it too. For something that only comes up sometimes, use a form that costs nothing until it's needed, like a skill, a script or a hook in the tool loop.
6. Try it before you build it. Make a rough version in `/tmp`, and run it and `parent/` with `try-variant` on the same small tasks. Write those tasks yourself so the problem comes up, rather than using public projects the model might remember. A few tries can't show a gain, so just check that it changes what the agent does the way you meant. If it doesn't, fix it or move on to the next idea.
7. Build it. Copy `parent/` to `/out/variants/<name>/` and change only what your idea needs. Record when it kicks in, as `harness.md` says. Test it again on your small tasks plus a new one unlike them, and see what it does when something fails, e.g. a command errors or a file is missing.
8. Write its card (below), then run `check-variant` and `try-variant --question` on the package until both pass. The same checks run when you hand it in, and a variant that fails them is never run. Do this as soon as the variant works, so it's ready if you run out of time.

## Rules

- Hand in exactly one variant. If `/out/variants` holds more than one when the session ends, none of them gets run.
- Nothing in the package except its card may mention or depend on these tasks or runs, e.g. their projects, ids, tests, limits or how they're judged.
- Don't assume the work has tests, a git repo, a particular language or the internet. Your runs may be offline, but people's machines usually aren't.
- When you add text, like a prompt section or a skill's description, say plainly what the agent should do and when.
- If you change what a tool does, like its limits or what it returns, tell the model in the tool's description, or in the result when it happens, so it can plan around it.

## The card

Write the card as `variant.json` in the package. It's for the people running the variants and for the teacher sessions after yours, and the agent never sees it:

    {
      "title": "<what the change does, in under 72 characters>",
      "what": "<the change, in a sentence or two>",
      "why": "<where the agent goes wrong without it, and what you saw in the runs>",
      "expect": "<what should change in behavior and cost, and how often it kicks in>",
      "reuses": ["<a variant it builds on>"]
    }

Leave `reuses` empty if it doesn't build on another variant, and keep the card under 200 words. Name the folder after the change, in a few lowercase words joined by hyphens.
