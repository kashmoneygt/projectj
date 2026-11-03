import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Steers the agent off reconstructing a project's upstream fix from memory (ticket or PR numbers, upstream
// commits, "the actual fix", a later version's code) and back to the code in the repo. APPEND_SYSTEM.md states
// the rule; this enforces it where the behavior shows up:
// - after a response whose reasoning does it, a short note goes at the end of that response's first tool result
//   (at most MAX_NUDGES per run, at least MIN_RESPONSES_BETWEEN responses apart);
// - if the response that would end the run leans on such a recollection, the agent gets one more turn to check
//   the change against the task and the repo instead (once per run).

const RECALL: RegExp[] = [
	/\bupstream\b[^.\n]{0,40}\b(fix|fixes|fixed|patch|commit|change|changes|PR|pull request|implementation|version|code|solution)\b/i,
	/\b(actual|real|official)\s+(upstream\s+)?(fix|patch|commit)\b/i,
	/\b(recall|remember)\b[^.\n]{0,60}\b(fix|fixed|upstream|commit|patch|ticket|PR|pull request|release notes?|changelog)\b/i,
	/\b(ticket|PR|pull request)\s*#\s*\d{3,}/i,
	/\bfrom (my )?memory\b/i,
];

const MAX_NUDGES = 3;
const MIN_RESPONSES_BETWEEN = 5;
const SCAN_CHARS = 200_000;
const UNTRACKED_NOISE = /(^|\/)(__pycache__|\.pytest_cache|\.mypy_cache|\.tox|node_modules|[^/]*\.egg-info)(\/|$)|\.(pyc|pyo|o|so|log)$/;

const NUDGE =
	"[harness note] Your last reasoning tried to reconstruct how this was fixed upstream (a ticket, PR, commit or a later version's code). That can't be checked here and is often wrong for this checkout, so stop reconstructing it. Work from this repo: trace the code path, reproduce, and match the behavior the task states. Treat any remembered detail as a guess to verify in the repo.";

const FINAL_CHECK =
	"Before you finish: your last response relies on what you remember of an upstream fix, and that isn't evidence here. Check the change against the task and this repo instead:\n1. Where the task states an expected output or behavior, run its example on the current code and compare exactly (names, formatting, types, messages).\n2. Revisit any concern you set aside because the change \"matches upstream\", and settle it by reading the code or running a case.\n3. Run the existing tests for the code you changed.\nFix whatever doesn't hold. If everything holds, reply with a short summary and no tool call.";

const NO_CHANGE_CHECK =
	"Your last response ended without a tool call while trying to recall an upstream fix, and no file in the repo has been changed, so the task isn't done. Stop recalling: decide the fix from the code you've read in this repo and make the edit now, briefly reasoning first. Then run the task's example and the existing tests for the code you changed.";

function hasRecall(s: string): boolean {
	if (!s) return false;
	const t = s.length > SCAN_CHARS ? s.slice(0, SCAN_CHARS) : s;
	return RECALL.some((re) => re.test(t));
}

export default function (pi: ExtensionAPI) {
	let responses = 0;
	let nudges = 0;
	let lastNudgeAt = -1_000;
	let pending = false;
	let finalChecked = false;
	let last: { hasToolCall: boolean; recall: boolean } | undefined;

	pi.on("message_end", async (event) => {
		const msg = event.message as any;
		if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) return;
		responses++;
		const blocks = msg.content as any[];
		const hasToolCall = blocks.some((b) => b?.type === "toolCall");
		const thinking = blocks
			.filter((b) => b?.type === "thinking" && typeof b.thinking === "string")
			.map((b) => b.thinking as string)
			.join("\n");
		const text = blocks
			.filter((b) => b?.type === "text" && typeof b.text === "string")
			.map((b) => b.text as string)
			.join("\n");
		const recall = hasRecall(thinking) || hasRecall(text);
		last = { hasToolCall, recall };
		pending =
			recall && hasToolCall && nudges < MAX_NUDGES && responses - lastNudgeAt >= MIN_RESPONSES_BETWEEN;
	});

	pi.on("tool_result", async (event) => {
		if (!pending || event.parentToolCallId) return;
		pending = false;
		nudges++;
		lastNudgeAt = responses;
		pi.appendEntry("acted", { what: "recall nudge" });
		return {
			content: [...(event.content ?? []), { type: "text" as const, text: `\n\n${NUDGE}` }],
			details: (event as any).details,
		};
	});

	pi.on("agent_before_settle", async (event) => {
		if (finalChecked || event.outcome !== "completed" || !last || last.hasToolCall || !last.recall) return;
		finalChecked = true;
		let changed: boolean | undefined;
		try {
			const r = await pi.exec("git", ["status", "--porcelain", "--untracked-files=all"], { timeout: 10_000 });
			if (r.code === 0) {
				changed = r.stdout
					.split("\n")
					.map((l) => l.slice(3).trim())
					.some((p) => p && !UNTRACKED_NOISE.test(p));
			}
		} catch {
			changed = undefined;
		}
		const noChange = changed === false;
		pi.appendEntry("acted", { what: noChange ? "recall check at finish (no change)" : "recall check at finish" });
		return {
			entries: [
				...event.entries,
				{
					type: "custom_message",
					customType: "recall-check",
					content: noChange ? NO_CHANGE_CHECK : FINAL_CHECK,
					display: true,
				},
			],
			continue: true,
		};
	});
}
