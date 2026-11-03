import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// The first time the agent tries to end a run, show it the task again next to the change it actually made
// (git diff), say whether any test command ran after its last edit, and ask for one requirement-by-requirement
// check. Fires at most once per run; never blocks anything.

const TASK_MAX_CHARS = 6000;
const DIFF_MAX_CHARS = 8000;
// Skip when the context is already close to the window, so the review itself can't overflow it.
const MAX_CONTEXT_TOKENS = 100_000;

const TEST_COMMAND =
	/(\bpytest\b|py\.test|runtests\.py|\btox\b|\bnox\b|-m\s+unittest|manage\.py\s+test|bin\/test\b|\btest\(\)|\b(npm|yarn|pnpm)\s+(run\s+)?test|\bgo\s+test\b|\bcargo\s+test\b|\bmvn\b|\bgradlew?\b|\bmake\s+(check|test)\b|\bctest\b|\bjest\b|\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b)/;
const UNTRACKED_NOISE = /(^|\/)(__pycache__|\.pytest_cache|\.mypy_cache|\.tox|node_modules|[^/]*\.egg-info)(\/|$)|\.(pyc|pyo|o|so|log)$/;
const BASH_EDIT = /(\bsed\s+-i|\bpatch\b|\bgit\s+(apply|checkout|restore|stash\s+pop|stash\s+apply)\b|\bperl\s+-p?i)/;

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

export default function (pi: ExtensionAPI) {
	let reviewed = false;
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
		const message = event.message as any;
		if (message?.role !== "assistant" || !message.usage) return;
		const u = message.usage;
		lastContextTokens = (u.input ?? 0) + (u.cacheRead ?? 0) + (u.cacheWrite ?? 0) + (u.output ?? 0);
	});

	pi.on("agent_before_settle", async (event) => {
		if (reviewed || event.outcome !== "completed") return;
		if (lastContextTokens > MAX_CONTEXT_TOKENS) return;
		reviewed = true;
		try {
			const firstUser = event.context.contextMessages.find((m: any) => m?.role === "user") as any;
			const task = textOf(firstUser?.content).trim();
			if (!task) return;

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

			pi.appendEntry("acted", {
				what: `finish review${hasChange ? "" : " (no change)"}${hasChange && !testedAfterEdit ? " (untested change)" : ""}`,
			});
			return {
				entries: [
					...event.entries,
					{ type: "custom_message", customType: "finish-review", content: parts.join("\n"), display: true },
				],
				continue: true,
			};
		} catch {
			return;
		}
	});
}
