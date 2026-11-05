import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Two guards on how a run ends, sharing one settle hook so they never both fire on the same stop:
// 1. Runaway guard. settings.json caps each response's reasoning at thinkingBudgets.medium. If a response still
//    ends the run by accident (cut at the length limit, reasoning only, a tool call written as text), add one short
//    message and ask for another turn. Looping or length-cut blocks are trimmed so they don't ride along in every
//    later request.
// 2. Finish review. The first time the agent really tries to end the run, show it the task next to the change it
//    made (git diff), say whether a test command ran after its last edit, and ask for one requirement-by-requirement
//    check, including a probe of other inputs that reach the same cause as the task's example. Once per run; never
//    blocks anything.
// A capped response that does answer is a real attempt to finish, so the review stands in for the "capped" nudge.

// Keep in sync with thinkingBudgets.medium in settings.json.
const THINKING_BUDGET = 8192;
const MAX_NUDGES_PER_RUN = 4;
const MAX_NUDGES_IN_A_ROW = 2;
// How much of a looping or length-cut block stays in the context.
const KEEP_CHARS = 2000;

const TASK_MAX_CHARS = 6000;
const DIFF_MAX_CHARS = 8000;
// Skip the review when the context is already close to the window, so the review itself can't overflow it.
const MAX_CONTEXT_TOKENS = 100_000;

type Reason = "length" | "empty" | "markup" | "capped";

interface LastResponse {
	stopReason: string;
	hasToolCall: boolean;
	hasText: boolean;
	textHasCallMarkup: boolean;
	capped: boolean;
}

const NUDGES: Record<Reason, string> = {
	length:
		"Your last response was cut off at the response length limit before it made a tool call, so nothing was run and the task is not finished. Continue from where you are: make your next tool call now, with only brief reasoning before it.",
	empty:
		"Your last response contained only reasoning: no tool call and no answer, so nothing was run and the task is not finished. A tool call written inside your reasoning is not executed. Continue from where you are: make your next tool call now, with only brief reasoning before it.",
	markup:
		"Your last response wrote a tool call as plain text, so it was not executed. Make that call again as a real tool call.",
	capped:
		"Your reasoning in the last response hit its length limit and was cut off, and the response made no tool call. If the work is not finished, continue now with your next tool call, reasoning briefly. If it is finished, reply with a short summary and no tool call.",
};

const ACTED: Record<Reason, string> = {
	length: "a response cut at the length limit",
	empty: "a reasoning-only response",
	markup: "a tool call written as text",
	capped: "a capped response with no tool call",
};

// Added to the finish review when there is a change: probe other inputs that reach the same cause, so a fix that
// only covers the task's own example gets extended to the other ways the same problem shows up.
const SAME_CAUSE_STEP =
	"Other inputs with the same cause. The task's example is one way to hit the problem; a fix that covers only that way is incomplete. In one line, name the cause your change removes: the code that went wrong, not the symptom (for a new feature, the code path it changes). Then pick the 2-4 other inputs most likely to reach that same code by another route, ones that would have hit the same problem before your change, and try them in one short script. Look first at other places in the function or class you changed that use the same value the same way, then at the same state reached another way (grep the repo for other code that sets the same attribute or calls the same function: an alias or property, a keyword instead of a positional argument, another method or module that sets it), then at the same value in another type or form the code accepts (a tuple for a list, bytes for str), and at every function, method or option the task names. Use inputs that are easy to build here. Run the script once; if the script itself breaks, fix it once or drop that input. If an input still shows the problem the task reports, extend the fix to cover it; leave results that are a different problem alone. Stay on the code path the task is about: don't change other implementations it doesn't name (another backend, printer, format or subclass), and don't add behavior it doesn't ask for.";

const TEST_COMMAND =
	/(\bpytest\b|py\.test|runtests\.py|\btox\b|\bnox\b|-m\s+unittest|manage\.py\s+test|bin\/test\b|\btest\(\)|\b(npm|yarn|pnpm)\s+(run\s+)?test|\bgo\s+test\b|\bcargo\s+test\b|\bmvn\b|\bgradlew?\b|\bmake\s+(check|test)\b|\bctest\b|\bjest\b|\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b)/;
const UNTRACKED_NOISE = /(^|\/)(__pycache__|\.pytest_cache|\.mypy_cache|\.tox|node_modules|[^/]*\.egg-info)(\/|$)|\.(pyc|pyo|o|so|log)$/;
const BASH_EDIT = /(\bsed\s+-i|\bpatch\b|\bgit\s+(apply|checkout|restore|stash\s+pop|stash\s+apply)\b|\bperl\s+-p?i)/;

// A block that has fallen into a loop: a long run of comma-separated numbers, or a tail made mostly of
// the same few phrases over and over.
function looksDegenerate(s: string): boolean {
	if (s.length < 3000) return false;
	if (/(?:\d+\s*,\s*){80,}/.test(s.slice(-20000))) return true;
	const chunks = s
		.slice(-6000)
		.split(/[\n.;,]+/)
		.map((c) => c.trim())
		.filter((c) => c.length >= 8);
	if (chunks.length < 20) return false;
	const counts = new Map<string, number>();
	for (const c of chunks) counts.set(c, (counts.get(c) ?? 0) + 1);
	let repeated = 0;
	for (const n of counts.values()) if (n >= 4) repeated += n;
	return repeated / chunks.length >= 0.6;
}

function shorten(s: string, why: string): string {
	return `${s.slice(0, KEEP_CHARS)}\n\n[... ${s.length - KEEP_CHARS} more characters removed: ${why} ...]`;
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((b: any) => b && b.type === "text" && typeof b.text === "string")
		.map((b: any) => b.text)
		.join("\n");
}

function clip(text: string, max: number, what: string): string {
	if (text.length <= max) return text;
	return `${text.slice(0, max)}\n[... ${what} cut at ${max} characters ...]`;
}

function accidentalEnd(last: LastResponse | undefined): Reason | undefined {
	if (!last || last.hasToolCall) return undefined;
	if (last.stopReason === "length") return "length";
	if (!last.hasText) return "empty";
	if (last.textHasCallMarkup) return "markup";
	if (last.capped) return "capped";
	return undefined;
}

export default function (pi: ExtensionAPI) {
	let last: LastResponse | undefined;
	let nudges = 0;
	let inARow = 0;

	let reviewed = false;
	let diffAtReview: string | undefined;
	let changeAfterReviewSeen = false;
	let taskText = "";
	let step = 0;
	let lastEditStep = -1;
	let lastTestStep = -1;
	let lastTestCommand = "";
	let lastContextTokens = 0;

	pi.on("tool_call", async (event) => {
		step++;
		if (event.toolName === "edit" || event.toolName === "write") {
			lastEditStep = step;
		} else if (event.toolName === "bash") {
			const command = String((event.input as { command?: unknown }).command ?? "");
			if (TEST_COMMAND.test(command)) {
				lastTestStep = step;
				lastTestCommand = command;
			} else if (BASH_EDIT.test(command)) {
				lastEditStep = step;
			}
		}
	});

	pi.on("message_end", async (event) => {
		const msg = event.message as any;
		// Keep the task text: a compaction can drop it from the context before the review needs it.
		if (msg?.role === "user" && !taskText) taskText = textOf(msg.content).trim();
		if (!msg || msg.role !== "assistant") return;
		if (msg.usage) {
			const u = msg.usage;
			lastContextTokens = (u.input ?? 0) + (u.cacheRead ?? 0) + (u.cacheWrite ?? 0) + (u.output ?? 0);
		}
		if (!Array.isArray(msg.content)) return;
		const blocks = msg.content as any[];
		const hasToolCall = blocks.some((b) => b?.type === "toolCall");
		const text = blocks
			.filter((b) => b?.type === "text" && typeof b.text === "string")
			.map((b) => b.text as string)
			.join("\n");
		const thinking = blocks
			.filter((b) => b?.type === "thinking" && typeof b.thinking === "string")
			.map((b) => b.thinking as string)
			.join("\n");
		const output = Number(msg.usage?.output ?? 0);
		const reasoning = Number(msg.usage?.reasoning ?? 0);
		const capped =
			thinking.length > 0 && (reasoning > 0 ? reasoning >= THINKING_BUDGET - 64 : output >= THINKING_BUDGET);
		const stopReason = String(msg.stopReason ?? "");
		last = {
			stopReason,
			hasToolCall,
			hasText: text.trim().length > 0,
			textHasCallMarkup: /<tool_call>|<function=/.test(text),
			capped,
		};
		if (hasToolCall) inARow = 0;
		if (capped) pi.appendEntry("acted", { what: "reasoning hit the budget" });

		// Keep a looping or length-cut block out of every later request.
		const cutAtLength = stopReason === "length";
		let changed = false;
		const content = blocks.map((b) => {
			if (b?.type === "thinking" && typeof b.thinking === "string" && b.thinking.length > KEEP_CHARS) {
				if (cutAtLength || looksDegenerate(b.thinking)) {
					changed = true;
					return { ...b, thinking: shorten(b.thinking, cutAtLength ? "cut at the length limit" : "it was repeating itself") };
				}
			}
			if (b?.type === "text" && typeof b.text === "string" && b.text.length > KEEP_CHARS) {
				if (cutAtLength || looksDegenerate(b.text)) {
					changed = true;
					return { ...b, text: shorten(b.text, cutAtLength ? "cut at the length limit" : "it was repeating itself") };
				}
			}
			return b;
		});
		if (!changed) return;
		pi.appendEntry("acted", { what: "trimmed a looping or cut response" });
		return { message: { ...msg, content } };
	});

	async function buildReview(contextMessages: any[]): Promise<{ content: string; what: string } | undefined> {
		const firstUser = contextMessages.find((m: any) => m?.role === "user") as any;
		const task = taskText || textOf(firstUser?.content).trim();
		if (!task) return undefined;

		let stat = "";
		let diff = "";
		let untracked: string[] = [];
		let gitOk = false;
		try {
			const s = await pi.exec("git", ["diff", "--stat", "HEAD"], { timeout: 10_000 });
			const d = await pi.exec("git", ["diff", "HEAD"], { timeout: 10_000 });
			const u = await pi.exec("git", ["ls-files", "--others", "--exclude-standard"], { timeout: 10_000 });
			if (s.code === 0 && d.code === 0) {
				gitOk = true;
				stat = s.stdout.trim();
				diff = d.stdout.trim();
				diffAtReview = diff;
				untracked =
					u.code === 0
						? u.stdout
								.split("\n")
								.map((l) => l.trim())
								.filter((l) => l && !UNTRACKED_NOISE.test(l))
						: [];
			}
		} catch {
			gitOk = false;
		}

		const parts: string[] = [];
		parts.push(
			"Before this run ends, check your change against the task once more (this check happens once).",
			"",
			"The task you were given:",
			"<task>",
			clip(task, TASK_MAX_CHARS, "task"),
			"</task>",
			"",
		);
		if (gitOk) {
			if (diff) {
				parts.push(
					"Your change so far (`git diff HEAD`):",
					"```diff",
					clip(`${stat}\n\n${diff}`, DIFF_MAX_CHARS, "diff"),
					"```",
				);
			} else {
				parts.push(
					"`git diff HEAD` is empty: no tracked file has been changed. If the task needs a code change, it is not done yet.",
				);
			}
			if (untracked.length > 0) {
				parts.push(
					`Untracked files: ${untracked.slice(0, 20).join(", ")}${untracked.length > 20 ? ", ..." : ""}`,
					"Delete the ones that are your own throwaway scripts; keep any that are part of the change.",
				);
			}
			parts.push("");
		}
		const hasChange = gitOk ? diff !== "" || untracked.length > 0 : lastEditStep >= 0;
		const testedAfterEdit = lastTestStep >= 0 && lastTestStep > lastEditStep;
		if (hasChange) {
			parts.push(
				testedAfterEdit
					? `Last test command after your last edit: \`${clip(lastTestCommand.trim(), 300, "command")}\``
					: "No test command has run since your last edit.",
				"",
			);
		}
		const steps: string[] = [
			"List what the task asks for: its title and every behavior or expected result it states. For each one, point to the part of your change that delivers it. If something is missing or only partly done, finish it.",
			"If the task shows an example (code, command or input) with the result it expects, run that example on the current code and compare the actual output with the expected one exactly: names, formatting, types, messages. A near match is a mismatch.",
		];
		if (hasChange) steps.push(SAME_CAUSE_STEP);
		steps.push(
			!hasChange || testedAfterEdit
				? "If you change anything now, run the existing tests for the code you changed again afterwards."
				: "Run the existing tests for the code you changed (only those test files or modules, not the whole suite) and fix anything your change broke.",
			"Decide by the task text and this repo's code and tests, not by what you remember of how the project fixed this elsewhere.",
		);
		parts.push("Do this now:", ...steps.map((s, i) => `${i + 1}. ${s}`));
		parts.push(
			hasChange
				? "If every point already holds and you have already run what steps 2, 3 and 4 ask for after your last edit, don't call any tools: reply with a two-line summary and stop."
				: "If every point already holds and you have already run what steps 2 and 3 ask for after your last edit, don't call any tools: reply with a two-line summary and stop.",
		);
		return {
			content: parts.join("\n"),
			what: `finish review${hasChange ? " with same-cause probe" : " (no change)"}${hasChange && !testedAfterEdit ? " (untested change)" : ""}`,
		};
	}

	pi.on("agent_before_settle", async (event) => {
		// Observation only: record once if the tracked diff changed after the review was shown.
		if (diffAtReview !== undefined && !changeAfterReviewSeen) {
			try {
				const d = await pi.exec("git", ["diff", "HEAD"], { timeout: 10_000 });
				if (d.code === 0 && d.stdout.trim() !== diffAtReview) {
					changeAfterReviewSeen = true;
					pi.appendEntry("acted", { what: "changed the code after the review" });
				}
			} catch {}
		}
		if (event.outcome !== "completed") return;
		const reviewPossible = !reviewed && lastContextTokens <= MAX_CONTEXT_TOKENS;

		const reason = accidentalEnd(last);
		// A capped response that still answered is a real finish: the review covers it when it can run.
		if (reason && !(reason === "capped" && reviewPossible)) {
			if (nudges < MAX_NUDGES_PER_RUN && inARow < MAX_NUDGES_IN_A_ROW) {
				nudges++;
				inARow++;
				last = undefined;
				pi.appendEntry("acted", { what: `continued after ${ACTED[reason]}` });
				return {
					entries: [
						...event.entries,
						{ type: "custom_message", customType: "response-cut", content: NUDGES[reason], display: true },
					],
					continue: true,
				};
			}
		}

		if (!reviewPossible) return;
		reviewed = true;
		try {
			const review = await buildReview(event.context.contextMessages as any[]);
			if (!review) return;
			last = undefined;
			pi.appendEntry("acted", { what: review.what });
			return {
				entries: [
					...event.entries,
					{ type: "custom_message", customType: "finish-review", content: review.content, display: true },
				],
				continue: true,
			};
		} catch {
			return;
		}
	});
}
