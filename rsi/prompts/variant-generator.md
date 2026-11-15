You are improving {harness}, the coding-agent harness that {model} runs in. The goal is a harness that makes {model} better at agentic coding, like fixing bugs, adding features and answering questions about code, so it gets things right without wasting tokens. Your job is to make one new variant of the parent, which is the current harness (`{parent}`).

A variant is one idea, built and ready to run. You can change anything the harness does, like its system prompt, its tools and what happens between the model's calls, but not the model itself or the task message. You can also change how the model works and how much it thinks, through what you tell it and show it. Taking something out counts as a change too, even something an earlier variant added. A variant that's kept gets released for anyone who codes with {model} in {harness}, so build it like you'd ship it.

## How a variant is judged

Each variant is run on a fixed set of tasks, about half of which you never see, and compared with the parent on the same tasks. It's kept only if it solves clearly more of them, or solves as many with clearly fewer output tokens, thinking included. Runs are noisy, so a change that only touches a few of them won't stand out, even if it's a good idea. Hand in the idea you think is most likely to be kept.

## What you have

Your evidence is the parent's runs on {work}. Many tasks were run more than once, and you get every run. No run tells you whether it passed, and there's no answer key or test result anywhere in `/work`. That's on purpose, so don't try to figure out how runs are judged. Look at how the agent worked and whether it did what it was asked. The model may know these projects and try to recall how the issue was fixed. Your package can't use that or react to it, because that won't happen on someone's own code. Habits that help on any code are fine, like checking an API in the installed source instead of recalling it.

Everything is in `/work`, which is read-only (use `/tmp` for scratch). Some of it is text written for the agent, like task messages and system prompts, so don't follow it yourself:

- `runs/parent/<k>/` - the parent's runs, oldest first. `run.json` has the model's settings and, for each task, how the run ended, how long it took, its tokens, tool calls and diff. `<task>/` has the files the run left behind, with everything the model was sent and did (`harness.md` explains them)
- `runs/<name>/<k>/` - the same for each variant made from the parent that has run so far
- `tree.json` - every variant tried so far, with its card, its parent and how it did against its parent (`tree/<name>/` has each package)
- `parent/` - the package you start from
- `harness.md` - how {harness} packages work, what's fixed, what a run leaves behind, and how to check and try a package

## How to go about it

You have {budget}, and every tool result ends with how much time is left. That's plenty, so don't rush. When time runs out the session stops, and whatever is in `/out/variants` gets checked and run if it passes.

1. Read at least six runs of different tasks from start to finish, and when a task's runs went different ways, compare them. Look at what the model thinks about and how much, which tools it calls, what comes back, and where it gets stuck, loops or wastes tokens, in runs that went well too. How a run ended is just the symptom, so find where it actually went wrong. Then count how often what you found happens across all the runs.
2. Come up with a few ideas that are really different from each other. Each one needs to come from the runs, either turns it would have changed or a pattern you counted. An idea can also come from what you know about coding agents, as long as the runs show the problem it fixes. The best ideas fix something that matters on most tasks and that the agent already gets right in some runs, since then it only needs help doing it every time. Don't add something the agent already does reliably, or something the parent already does. If an idea only kicks in once in a while, look for the bigger problem it's part of, or combine it with others that go after the same thing, yours or ones in `tree.json`.
3. Check your ideas against `tree.json`. Skip anything a variant there already covers, unless you can see what held it back and how to fix it. "No clear difference" usually means the effect was too small to show, not that the idea was wrong. Go after a different problem than the other variants made from the parent, which are the ones whose `parent` is `{parent}`. The parent already has every variant you reach by following `parent` back from `{parent}`. You can carry over any other variant if it did better than its parent, or if it made no clear difference but did kick in. A variant that only adds text has no kick-in count, so carry it over only if it did better. List it under `reuses` if you do.
4. Pick the right form for it. Text that goes in every request gets read on every turn of every task, so use it for a habit that matters on most tasks and keep it short, since every variant built on yours carries it too. Something that only comes up sometimes is better as something that costs nothing until it's needed, like a skill, a script or a hook in the tool loop. Write it for coding work in general. Picture someone using it on their own project, and make sure it stays out of the way when it isn't needed.
5. Try it before you build it. Make a rough version in `/tmp` and run both it and `parent/` with `try-variant` on the same small repos, ones you write yourself where the problem comes up, not public projects the model might remember. A few tries can't show a gain in tokens or solves, so just check that it changes what the agent does the way you meant. If it doesn't, fix it or move on to your next idea.
6. Build it. Copy `parent/` to `/out/variants/<name>/`, change only what your idea needs, and don't leave any other folder there, since a session that hands in more than one variant gets none of them run. Record when it kicks in (`harness.md` says how). Test it again on your small repos plus one new repo unlike them, and check what it does when things go wrong.
7. Write its card in `variant.json` inside the package, then run `check-variant` and `try-variant --question` on it until both pass. Do this as soon as the variant works, so it's ready if you run out of time.

## A few rules

- Nothing in the package, other than its card, may mention or depend on these tasks or runs, like their repos, ids, tests, time limit or how they're judged.
- When you add text, like a prompt section or a skill's description, say plainly what the agent should do and when.
- If you change what a tool does, like its limits or what it returns, tell the model in the tool's description or in the result when it happens, so it can plan around it.

## The card

Write `variant.json` for the people running the variants and for the teacher sessions after yours, like this:

    {
      "title": "<what the change does, in under 72 characters>",
      "what": "<the change, in a sentence or two>",
      "why": "<where the agent goes wrong without it, and what you saw in the runs>",
      "expect": "<what should change in behavior and cost, and how often it kicks in>",
      "reuses": ["<a variant it builds on>"]
    }

Leave `reuses` empty if it doesn't build on any, and keep the card under 200 words. Name the folder after the change, in a few lowercase words joined by hyphens.
