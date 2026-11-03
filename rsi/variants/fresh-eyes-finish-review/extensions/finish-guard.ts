import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Two guards on how a run ends, sharing one settle hook so they never both fire on the same stop:
// 1. Runaway guard. settings.json caps each response's reasoning at thinkingBudgets.medium. If a response still
//    ends the run by accident (cut at the length limit, reasoning only, a tool call written as text), add one short
//    message and ask for another turn. Looping or length-cut blocks are trimmed so they don't ride along in every
//    later request.
// 2. Fresh-eyes finish review. The first time the agent really tries to end the run, show it the change it made
//    (git diff), the test commands it ran and the last test output, and ask for one review against the task. From
//    then on every request carries only the task and the review onward: the agent's earlier reasoning and tool
//    calls are left out, so the review isn't anchored on the rationale that produced the change. Once per run;
//    never blocks anything.
// A capped response that does answer is a real attempt to finish, so the review stands in for the "capped" nudge.

// Keep in sync with thinkingBudgets.medium in settings.json.
const THINKING_BUDGET = 8192;
const MAX_NUDGES_PER_RUN = 4;
const MAX_NUDGES_IN_A_ROW = 2;
// How much of a looping or length-cut block stays in the context.
const KEEP_CHARS = 2000;

const TASK_MAX_CHARS = 6000;
// The in-context fallback review is skipped when the context is already close to the window, so it can't overflow it.
const MAX_CONTEXT_TOKENS = 100_000;
const DIFF_MAX_CHARS = 16000;
const TEST_OUTPUT_MAX_CHARS = 2000;
const MAX_TEST_COMMANDS = 4;
const REVIEW_TYPE = "finish-review";

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
	// Set once the review message is in the session: from then on requests leave out what came before it.
	let fresh = false;
	let outcomeNoted = false;
	let reviewDiff: string | undefined;
	let taskText = "";
	let taskMessage: any;
	let step = 0;
	let lastEditStep = -1;
	let lastTestStep = -1;
	let lastTestCommand = "";
	let lastTestCallId = "";
	let lastTestOutput = "";
	let testCommands: string[] = [];
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
				lastTestCallId = event.toolCallId;
				lastTestOutput = "";
				const short = clip(command.trim(), 300, "command");
				testCommands = [...testCommands.filter((c) => c !== short), short].slice(-MAX_TEST_COMMANDS);
			} else if (BASH_EDIT.test(command)) {
				lastEditStep = step;
			}
		}
	});

	pi.on("tool_result", async (event) => {
		if (event.toolName !== "bash" || !lastTestCallId || event.toolCallId !== lastTestCallId) return;
		const out = textOf(event.content).trim();
		lastTestOutput = out.length > TEST_OUTPUT_MAX_CHARS ? `[...]\n${out.slice(-TEST_OUTPUT_MAX_CHARS)}` : out;
	});

	pi.on("message_end", async (event) => {
		const msg = event.message as any;
		// Keep the task: the fresh-context review starts from it, and a compaction can drop it from the context.
		if (msg?.role === "user" && !taskText) {
			taskText = textOf(msg.content).trim();
			taskMessage = msg;
		}
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

	// After the review starts, each request holds the task, then the review message and what followed it.
	pi.on("context", async (event) => {
		const msgs = event.messages as any[];
		if (!taskText) {
			const first = msgs.find((m) => m?.role === "user");
			const text = first ? textOf(first.content).trim() : "";
			if (text) {
				taskText = text;
				taskMessage = first;
			}
		}
		if (!fresh) return;
		let at = -1;
		for (let i = msgs.length - 1; i >= 0; i--) {
			if (msgs[i]?.role === "custom" && msgs[i]?.customType === REVIEW_TYPE) {
				at = i;
				break;
			}
		}
		if (at < 0) return;
		const task =
			taskMessage ??
			msgs.find((m) => m?.role === "user") ?? {
				role: "user",
				content: [{ type: "text", text: clip(taskText, TASK_MAX_CHARS, "task") }],
				timestamp: Date.now(),
			};
		const tail = msgs.slice(at);
		if (msgs[0] === task && at === 1) return;
		return { messages: [task, ...tail] };
	});

	async function gitState(): Promise<{ ok: boolean; stat: string; diff: string; untracked: string[] }> {
		try {
			const s = await pi.exec("git", ["diff", "--stat", "HEAD"], { timeout: 10_000 });
			const d = await pi.exec("git", ["diff", "HEAD"], { timeout: 10_000 });
			const u = await pi.exec("git", ["ls-files", "--others", "--exclude-standard"], { timeout: 10_000 });
			if (s.code !== 0 || d.code !== 0) return { ok: false, stat: "", diff: "", untracked: [] };
			const untracked =
				u.code === 0
					? u.stdout
							.split("\n")
							.map((l) => l.trim())
							.filter((l) => l && !UNTRACKED_NOISE.test(l))
					: [];
			return { ok: true, stat: s.stdout.trim(), diff: d.stdout.trim(), untracked };
		} catch {
			return { ok: false, stat: "", diff: "", untracked: [] };
		}
	}

	async function buildReview(): Promise<{ content: string; what: string; diff: string | undefined; fresh: boolean } | undefined> {
		if (!taskText) return undefined;
		const git = await gitState();
		// Without a diff there is nothing for an outside reviewer to judge: review in place instead.
		const freshReview = git.ok && (git.diff !== "" || git.untracked.length > 0);
		if (!freshReview && lastContextTokens > MAX_CONTEXT_TOKENS) return undefined;

		const parts: string[] = [];
		if (freshReview) {
			parts.push(
				"Before this run ends, your change gets one independent review (this happens once).",
				"",
				"Your earlier reasoning and tool calls in this session are left out from here on, on purpose: review the change the way a careful reviewer who did not write it would, judging it only by the task above, this repository's code and tests, and what you run now. Your memory of how the project fixed this elsewhere is not evidence.",
				"",
			);
		} else {
			parts.push(
				"Before this run ends, check your change against the task once more (this happens once).",
				"",
				"The task you were given:",
				"<task>",
				clip(taskText, TASK_MAX_CHARS, "task"),
				"</task>",
				"",
				"Judge the change only by the task, this repository's code and tests, and what you run now. Your memory of how the project fixed this elsewhere is not evidence.",
				"",
			);
		}
		if (git.ok) {
			if (git.diff) {
				parts.push("The change (`git diff HEAD`):", "```diff", clip(`${git.stat}\n\n${git.diff}`, DIFF_MAX_CHARS, "diff"), "```");
			} else {
				parts.push(
					git.untracked.length > 0
						? "`git diff HEAD` is empty: no tracked file has been changed. Read the untracked files below; if none of them is part of the change and the task needs one, it is not done yet."
						: "`git diff HEAD` is empty: no tracked file has been changed. If the task needs a code change, it is not done yet.",
				);
			}
			if (git.untracked.length > 0) {
				parts.push(
					`Untracked files: ${git.untracked.slice(0, 20).join(", ")}${git.untracked.length > 20 ? ", ..." : ""}`,
					"Delete the ones that are throwaway scripts; keep any that are part of the change.",
				);
			}
			parts.push("");
		}
		const hasChange = git.ok ? git.diff !== "" || git.untracked.length > 0 : lastEditStep >= 0;
		const testedAfterEdit = lastTestStep >= 0 && lastTestStep > lastEditStep;
		if (testCommands.length > 0) {
			parts.push("Test commands run in this session (latest last):", ...testCommands.map((c) => `- \`${c}\``), "");
			parts.push(
				testedAfterEdit
					? "The last one ran after the last edit. Its output:"
					: "The last one ran before the last edit, so it says nothing about the current code. Its output then:",
				"```",
				lastTestOutput || "(no output captured)",
				"```",
				"",
			);
		} else if (hasChange) {
			parts.push("No test command has run in this session.", "");
		}
		parts.push(
			"Review it now:",
			"1. Requirements: list what the task asks for: its title and every behavior or expected result it states. For each one, point to the part of the change that delivers it. If something is missing or only partly done, finish it.",
			"2. Scope: for each part of the change, name the requirement that needs it. A part no requirement needs changes behavior nobody asked about (another class, another code path, a shared helper): undo it, unless a needed part can't work without it.",
			"3. Expected output: if the task shows an example (code, command or input) with the result it expects, run it on the current code. Then copy the expected result word for word from the task and the actual output word for word, one above the other, and compare them exactly: names, qualification, formatting, types, messages. A near match is a mismatch.",
			"4. Existing tests: run the existing tests for the code the change touches (only those test files or modules, not the whole suite). For each failure, read the test. If it asserts exactly the behavior the task says is wrong, that failure is expected. Otherwise the change broke something it shouldn't: fix the change, not the test.",
			"5. Change the code only for a concrete reason found in 1-4, not because you would have written it differently. After any change, rerun the example and those tests.",
			"If everything holds and nothing needs to change, reply with a two-line summary and no tool call.",
		);
		return {
			content: parts.join("\n"),
			what: `${freshReview ? "fresh-context" : "in-context"} finish review${hasChange ? "" : " (no change)"}${hasChange && !testedAfterEdit ? " (untested change)" : ""}`,
			diff: git.ok ? `${git.diff}\n${git.untracked.join("\n")}` : undefined,
			fresh: freshReview,
		};
	}

	// Once, when the run really ends after the review: record whether the review changed the diff.
	async function noteOutcome() {
		if (outcomeNoted || !reviewed || reviewDiff === undefined) return;
		outcomeNoted = true;
		const git = await gitState();
		if (!git.ok) return;
		if (`${git.diff}\n${git.untracked.join("\n")}` !== reviewDiff) {
			pi.appendEntry("acted", { what: "review changed the diff" });
		}
	}

	pi.on("agent_before_settle", async (event) => {
		if (event.outcome !== "completed") return;
		const reviewPossible = !reviewed && taskText !== "";

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

		if (!reviewPossible) {
			try {
				await noteOutcome();
			} catch {}
			return;
		}
		reviewed = true;
		try {
			const review = await buildReview();
			if (!review) return;
			last = undefined;
			fresh = review.fresh;
			reviewDiff = review.diff;
			pi.appendEntry("acted", { what: review.what });
			return {
				entries: [
					...event.entries,
					{ type: "custom_message", customType: REVIEW_TYPE, content: review.content, display: true },
				],
				continue: true,
			};
		} catch {
			fresh = false;
			return;
		}
	});
}
