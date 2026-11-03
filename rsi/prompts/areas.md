# Areas

Every session in a round reads the same evidence, and left alone they'd all go after the same biggest failure. So each
session gets an area, and the area's text below fills `{area}` in `variant-generator.md`. A round starts with `open`,
so the change the teacher would bet on most always gets made. Then come the areas this lineage has had the fewest
sessions in, so every part of the harness gets its turn, and last `compose`, once the tree has variants worth putting
together. `tools`, `middleware` and `context` are named after the groups Harness² sorts a harness's parts into.

## open

the whole harness: make the change you'd bet on most, wherever it is

## tools

the model's tools: which ones it has, and what each one is called, takes, does and returns. You can add one (a script it runs, or a second agent it hands work to, counts), replace a built-in one by registering its name, or take one away. Keep your change inside it

## middleware

what the harness does around the model's calls: it can change or block a tool call before it runs, add to or replace what comes back, replace a finished response, and act at the end of a turn or of the run. Keep your change inside it

## context

what the model sees and how much it thinks: the system prompt (rewriting or cutting it counts), skills, what it sees of the run so far on each request, and a cap on how much it thinks per response. Keep your change inside it

## compose

putting variants together: build one package from the parent and two or three variants in `tree.md` that kicked in the way they predicted but whose gain wasn't proven. Where they overlap, merge them instead of stacking them: if one variant's text asks for what another's code already makes happen, keep the code. Name every variant you use under `ports`. This is the one area where a variant may hold more than one change
