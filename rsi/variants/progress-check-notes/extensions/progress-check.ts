import { isAbsolute, relative, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Calls before the first periodic progress check, and the spacing after it.
const FIRST_CHECK = 25;
const CHECK_EVERY = 15;
// From this many calls on, the periodic check also asks to wrap up.
const LONG_RUN = 55;
// Most notes after a passing test run, per run.
const MAX_PASS_NOTES = 2;
const MAX_FILES_SHOWN = 12;

// A test run counts as passing only if its output has a passing summary and no failure summary.
const PASS_SUMMARY =
	/\b[1-9]\d*\s+passed\b|^Ran\s+[1-9]\d*\s+tests?\b[\s\S]*^OK\b|\b[1-9]\d*\s+passing\b|^Tests:\s+[1-9]\d*\s+passed|^(#|ℹ)\s*pass\s+[1-9]|test result: ok\.|^ok\s+\S+\s+[\d.]+s\b|\b[1-9]\d*\s+examples?,\s+0\s+failures\b/m;
const FAILURE_COUNT = /\b[1-9]\d*\s+(failed|failing|errors?|exceptions?)\b/i;
const FAILURE_MARK =
	/\bFAILED\b|^FAIL\b|^ERROR\b|^(#|ℹ)\s*fail\s+[1-9]|^not ok\b|Traceback \(most recent call last\)|\bno tests ran\b|test result: FAILED/m;
const TEST_PATH =
	/(^|\/)(tests?|testing|__tests__|spec)\/|(^|\/)test_[^/]*$|_test\.[a-z]+$|\.(test|spec)\.[a-z]+$|(^|\/)conftest\.py$/i;

type Content = { type: string; text?: string }[];

// A file inside the working directory that isn't a test file.
function isSourceFile(cwd: string, path: string): boolean {
	const rel = relative(cwd, resolve(cwd, path));
	if (!rel || rel.startsWith("..") || isAbsolute(rel)) return false;
	return !TEST_PATH.test(rel);
}

function textOf(content: Content): string {
	return content
		.filter((c) => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text)
		.join("\n");
}

export default function (pi: ExtensionAPI) {
	let calls = 0;
	let nextCheck = FIRST_CHECK;
	let passNotes = 0;
	let sourceChangedSinceNote = false;
	let baseline: Set<string> | undefined;
	let baselineTaken = false;
	const editedFiles = new Set<string>();

	async function gitStatus(cwd: string, signal?: AbortSignal): Promise<string[] | undefined> {
		try {
			const r = await pi.exec("git", ["status", "--porcelain=v1", "--untracked-files=all"], {
				cwd,
				timeout: 5000,
				signal,
			});
			if (r.code !== 0) return undefined;
			return r.stdout.split("\n").filter((l) => l.trim().length > 0);
		} catch {
			return undefined;
		}
	}

	async function changeSummary(cwd: string, signal?: AbortSignal): Promise<string> {
		const status = await gitStatus(cwd, signal);
		if (status === undefined) {
			if (editedFiles.size === 0) return "Files you changed since you started: none with edit/write yet.";
			const files = [...editedFiles];
			const more = files.length > MAX_FILES_SHOWN ? ` (+${files.length - MAX_FILES_SHOWN} more)` : "";
			return `Files you changed since you started with edit/write: ${files.slice(0, MAX_FILES_SHOWN).join(", ")}${more}.`;
		}
		const mine = status.filter((l) => !baseline?.has(l));
		const before = status.length - mine.length;
		const lines = mine.slice(0, MAX_FILES_SHOWN).map((l) => `  ${l}`);
		if (mine.length > MAX_FILES_SHOWN) lines.push(`  ... ${mine.length - MAX_FILES_SHOWN} more`);
		let text = mine.length
			? `Files changed since you started (git status; these are your own changes):\n${lines.join("\n")}`
			: "Files changed since you started (git status): none.";
		if (before > 0) text += `\n(${before} other changed file(s) were already like that before you started.)`;
		return text;
	}

	// Each prompt is a new piece of work; the git baseline is taken once, before the agent changes anything.
	pi.on("before_agent_start", async (_event, ctx) => {
		calls = 0;
		nextCheck = FIRST_CHECK;
		passNotes = 0;
		sourceChangedSinceNote = false;
		if (baselineTaken) return;
		baselineTaken = true;
		const status = await gitStatus(ctx.cwd, ctx.signal);
		baseline = new Set(status ?? []);
	});

	pi.on("tool_result", async (event, ctx) => {
		if (event.parentToolCallId) return;
		calls++;
		const input = (event.input ?? {}) as { path?: unknown; command?: unknown };

		if ((event.toolName === "edit" || event.toolName === "write") && !event.isError) {
			const path = typeof input.path === "string" ? input.path : "";
			if (path) {
				editedFiles.add(path);
				if (isSourceFile(ctx.cwd, path)) sourceChangedSinceNote = true;
			}
		}

		let passNote = false;
		if (event.toolName === "bash" && !event.isError && sourceChangedSinceNote && passNotes < MAX_PASS_NOTES) {
			const command = typeof input.command === "string" ? input.command : "";
			const output = textOf(event.content as Content);
			passNote =
				!/git\s+stash/.test(command) &&
				PASS_SUMMARY.test(output) &&
				!FAILURE_COUNT.test(output) &&
				!FAILURE_MARK.test(output);
		}
		const countNote = calls >= nextCheck;
		if (!passNote && !countNote) return;
		// Update state before awaiting: results of parallel calls are handled concurrently.
		const callNo = calls;
		if (countNote) nextCheck = calls + CHECK_EVERY;
		if (passNote) {
			passNotes++;
			sourceChangedSinceNote = false;
		}

		const summary = await changeSummary(ctx.cwd, ctx.signal);
		let note: string;
		if (passNote) {
			note = [
				"[progress check] The output above looks like a passing test run after your latest source change.",
				summary,
				"If your change now does what was asked: run the existing tests closest to the code you changed if you haven't yet, delete scratch files you added to the repo, then stop and write your final answer. Don't keep re-verifying or polishing what already works.",
				"If part of the request is still unhandled, name it in one sentence and go straight to it.",
			].join("\n");
			pi.appendEntry("acted", { what: `progress check after passing tests (call ${callNo})` });
		} else {
			note = [
				`[progress check: ${callNo} tool calls so far]`,
				summary,
				"Take stock briefly. If what was asked is done and you have seen it work, stop and write your final answer now. Otherwise name in one sentence what is still missing and take the most direct step toward it, reusing what you already found instead of re-reading or re-searching it.",
			].join("\n");
			if (callNo >= LONG_RUN) {
				note +=
					"\nThis run is getting long: unless something specific is still broken, wrap up and mention anything you could not verify in your final answer.";
			}
			pi.appendEntry("acted", { what: `periodic progress check (call ${callNo})` });
		}

		return { content: [...(event.content as Content), { type: "text", text: `\n\n${note}` }] } as never;
	});
}
