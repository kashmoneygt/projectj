import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Two guards on how a run ends, sharing one settle hook so they never both fire on the same stop:
// 1. Runaway guard. settings.json caps each response's reasoning at thinkingBudgets.medium. If a response still
//    ends the run by accident (cut at the length limit, reasoning only, a tool call written as text), add one short
//    message and ask for another turn. Looping or length-cut blocks are trimmed so they don't ride along in every
//    later request.
// 2. Finish review, only when the change has an objective gap: no change at all, no test run since the last edit,
//    a last test run that reported failures, or untracked files the agent left in the repo. Then show the task next
//    to the change (git diff), say what the last test run after the last edit was, and ask for one
//    requirement-by-requirement check. Once per run. A non-empty change whose last test run after the last edit
//    passed ends the run without it: there the review rarely changed the diff, and it cost extra turns at the
//    largest context of the run.
//    A test command whose output shows that no test ran (no runner installed, a label or path that names nothing,
//    0 tests collected) does not count as testing the change. Each run is judged by its result as the model saw
//    it, so the verdict test-feedback.ts recovers from a filtered run counts too.
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

const TEST_COMMAND =
	/(\bpytest\b|py\.test|runtests\.py|\btox\b|\bnox\b|-m\s+unittest|manage\.py\s+test|bin\/test\b|\btest\(\)|\b(npm|yarn|pnpm)\s+(run\s+)?test|\bgo\s+test\b|\bcargo\s+test\b|\bmvn\b|\bgradlew?\b|\bmake\s+(check|test)\b|\bctest\b|\bjest\b|\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b)/;
const UNTRACKED_NOISE = /(^|\/)(__pycache__|\.pytest_cache|\.mypy_cache|\.tox|node_modules|[^/]*\.egg-info)(\/|$)|\.(pyc|pyo|o|so|log)$/;
// Output of a test command that ran no test at all.
const NO_TESTS =
	/\bRan 0 tests?\b|\bno tests ran\b|\bcollected 0 items\b|No module named '?pytest'?|ERROR: file or directory not found|ERROR: not found:|\berror: unrecognized arguments\b|\btests finished: 0 passed, in\b/;
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

// True only on positive evidence that no test ran: a runner reports nothing ran, or every test it ran is a
// label that names no module. Empty or filtered-away output is not evidence either way.
// Whether a test run's output reports failing or erroring tests. Errors from labels that name no module
// (`ERROR: <label> (unittest.loader._FailedTest)`) come from the command, not the code, and don't count.
const LABEL_ERROR = /^ERROR: \S+ \(unittest\.loader\._FailedTest.*$/gm;
export function reportedFailures(output: string): boolean {
	const text = output.replace(ANSI, "");
	const labelErrors = (text.match(LABEL_ERROR) ?? []).length;
	const rest = text.replace(LABEL_ERROR, "");
	for (const m of rest.matchAll(/FAILED\s+\(([^)]*)\)/g)) {
		const failures = Number(/failures=(\d+)/.exec(m[1])?.[1] ?? 0);
		const errors = Number(/errors=(\d+)/.exec(m[1])?.[1] ?? 0);
		if (failures > 0 || errors > labelErrors) return true;
	}
	return /\b[1-9]\d*\s+(failed|errors?|exceptions?)\b|^FAILED\s(?!\()|^ERROR:\s|^FAIL:\s/m.test(rest);
}

export function ranNoTests(output: string): boolean {
	const text = output.replace(ANSI, "");
	const labelErrors = (text.match(/^ERROR: \S+ \(unittest\.loader\._FailedTest/gm) ?? []).length;
	let ranCounts = 0;
	for (const m of text.matchAll(/\bRan (\d+) tests?\b/g)) {
		ranCounts++;
		if (Number(m[1]) > labelErrors) return false;
	}
	if (/\b[1-9]\d* (passed|failed|xfailed|xpassed)\b/.test(text)) return false;
	if (NO_TESTS.test(text)) return true;
	return ranCounts > 0 && labelErrors > 0;
}

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

interface Facts {
	gitOk: boolean;
	stat: string;
	diff: string;
	untracked: string[];
	hasChange: boolean;
	testedAfterEdit: boolean;
	// The last test command after the last edit that ran tests, and whether it reported failures.
	lastTestCommand: string;
	lastTestFailed: boolean;
	// Set when no test ran after the last edit although a test command did: the last such command.
	emptyTestCommand: string;
}

type Gap = "no change" | "untested change" | "failing tests" | "untracked files";

// What is objectively unfinished about the change, if anything.
function gapOf(f: Facts): Gap | undefined {
	if (!f.hasChange) return "no change";
	if (!f.testedAfterEdit) return "untested change";
	if (f.lastTestFailed) return "failing tests";
	if (f.untracked.length > 0) return "untracked files";
	return undefined;
}

export default function (pi: ExtensionAPI) {
	let last: LastResponse | undefined;
	let nudges = 0;
	let inARow = 0;

	let reviewed = false;
	let skipRecorded = false;
	let taskText = "";
	let step = 0;
	let lastEditStep = -1;
	let lastContextTokens = 0;
	// Every test command, with the output it returned (a fallback for when the result is no longer in the context).
	const testCalls: { step: number; id: string; command: string; output?: string }[] = [];

	// Untracked files that were there before the agent's first tool call are not its leftovers.
	let untrackedAtStart: Set<string> | undefined;
	async function listUntracked(): Promise<string[] | undefined> {
		const u = await pi.exec("git", ["ls-files", "--others", "--exclude-standard"], { timeout: 10_000 });
		if (u.code !== 0) return undefined;
		return u.stdout
			.split("\n")
			.map((l) => l.trim())
			.filter((l) => l && !UNTRACKED_NOISE.test(l));
	}

	pi.on("tool_call", async (event) => {
		if (!untrackedAtStart) {
			untrackedAtStart = new Set();
			try {
				for (const f of (await listUntracked()) ?? []) untrackedAtStart.add(f);
			} catch {
				// no git: nothing to subtract
			}
		}
		step++;
		if (event.toolName === "edit" || event.toolName === "write") {
			lastEditStep = step;
		} else if (event.toolName === "bash") {
			const command = String((event.input as { command?: unknown }).command ?? "");
			if (TEST_COMMAND.test(command)) {
				testCalls.push({ step, id: event.toolCallId, command });
			} else if (BASH_EDIT.test(command)) {
				lastEditStep = step;
			}
		}
	});

	pi.on("tool_result", async (event) => {
		const call = testCalls.find((c) => c.id === event.toolCallId);
		if (call) call.output = textOf(event.content);
	});

	// The test runs after the last edit, judged by each result as the model saw it.
	function testsAfterEdit(contextMessages: any[]): Pick<Facts, "testedAfterEdit" | "lastTestCommand" | "lastTestFailed" | "emptyTestCommand"> {
		const seen = new Map<string, string>();
		for (const m of contextMessages) {
			if (m?.role === "toolResult" && typeof m.toolCallId === "string") seen.set(m.toolCallId, textOf(m.content));
		}
		let lastRan: { command: string; output: string } | undefined;
		let lastEmpty = "";
		for (const c of testCalls) {
			if (c.step <= lastEditStep) continue;
			const output = seen.get(c.id) ?? c.output ?? "";
			if (ranNoTests(output)) lastEmpty = c.command;
			else lastRan = { command: c.command, output };
		}
		return {
			testedAfterEdit: lastRan !== undefined,
			lastTestCommand: lastRan?.command ?? "",
			lastTestFailed: lastRan !== undefined && reportedFailures(lastRan.output),
			emptyTestCommand: lastRan ? "" : lastEmpty,
		};
	}

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

	async function gatherFacts(contextMessages: any[]): Promise<Facts> {
		let stat = "";
		let diff = "";
		let untracked: string[] = [];
		let gitOk = false;
		try {
			const s = await pi.exec("git", ["diff", "--stat", "HEAD"], { timeout: 10_000 });
			const d = await pi.exec("git", ["diff", "HEAD"], { timeout: 10_000 });
			const u = await listUntracked();
			if (s.code === 0 && d.code === 0) {
				gitOk = true;
				stat = s.stdout.trim();
				diff = d.stdout.trim();
				untracked = (u ?? []).filter((f) => !untrackedAtStart?.has(f));
			}
		} catch {
			gitOk = false;
		}
		const hasChange = gitOk ? diff !== "" || untracked.length > 0 : lastEditStep >= 0;
		return { gitOk, stat, diff, untracked, hasChange, ...testsAfterEdit(contextMessages) };
	}

	function buildReview(
		contextMessages: any[],
		facts: Facts,
		gap: Gap,
	): { content: string; what: string } | undefined {
		const firstUser = contextMessages.find((m: any) => m?.role === "user") as any;
		const task = taskText || textOf(firstUser?.content).trim();
		if (!task) return undefined;
		const { gitOk, stat, diff, untracked, hasChange, testedAfterEdit, lastTestCommand, lastTestFailed, emptyTestCommand } =
			facts;

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
		if (hasChange) {
			parts.push(
				testedAfterEdit
					? `Last test command after your last edit: \`${clip(lastTestCommand.trim(), 300, "command")}\`${
							lastTestFailed
								? ". It reported failing tests: settle each one by your instructions on existing tests that fail after your change before you finish."
								: ""
						}`
					: emptyTestCommand
						? `Your last test command after your last edit ran no test (no runner, or no test found for what it named): \`${clip(emptyTestCommand.trim(), 300, "command")}\`. No test of your change has run since your last edit.`
						: "No test command has run since your last edit.",
				"",
			);
		}
		parts.push(
			"Do this now:",
			"1. List what the task asks for: its title and every behavior or expected result it states. For each one, point to the part of your change that delivers it. If something is missing or only partly done, finish it.",
			"2. If the task shows an example (code, command or input) with the result it expects, run that example on the current code and compare the actual output with the expected one exactly: names, formatting, types, messages. A near match is a mismatch.",
			!hasChange || testedAfterEdit
				? "3. If you change anything now, run the existing tests for the code you changed again afterwards."
				: "3. Run the existing tests for the code you changed (only those test files or modules, not the whole suite) and fix anything your change broke.",
			"4. Decide by the task text and this repo's code and tests, not by what you remember of how the project fixed this elsewhere.",
			"If every point already holds and you have already run what step 2 and 3 ask for after your last edit, don't call any tools: reply with a two-line summary and stop.",
		);
		const emptyRun = gap === "untested change" && emptyTestCommand !== "";
		return { content: parts.join("\n"), what: `finish review (${gap}${emptyRun ? ": last test run ran no tests" : ""})` };
	}

	pi.on("agent_before_settle", async (event) => {
		if (event.outcome !== "completed") return;
		const reviewPossible = !reviewed && lastContextTokens <= MAX_CONTEXT_TOKENS;
		const reason = accidentalEnd(last);

		// The review runs only when the change has an objective gap. A non-empty change that a test command
		// exercised after the last edit ends the run here, with no extra turn.
		let facts: Facts | undefined;
		let gap: Gap | undefined;
		if (reviewPossible) {
			try {
				facts = await gatherFacts(event.context.contextMessages as any[]);
				gap = gapOf(facts);
			} catch {
				facts = undefined;
				gap = undefined;
			}
		}
		const reviewWanted = reviewPossible && gap !== undefined;

		// A capped response that still answered is a real finish: the review covers it when it runs.
		if (reason && !(reason === "capped" && reviewWanted)) {
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

		if (!reviewWanted || !facts || !gap) {
			if (facts && !gap && !skipRecorded) {
				skipRecorded = true;
				pi.appendEntry("acted", { what: "skipped finish review: change tested and passing after last edit" });
			}
			return;
		}
		reviewed = true;
		try {
			const review = buildReview(event.context.contextMessages as any[], facts, gap);
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
