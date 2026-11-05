import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Failing test runs get a baseline. When a test command the agent runs after editing reports failures, the harness
// runs the same command once more with the agent's uncommitted changes to tracked files stashed, restores them right
// away, and appends to that result which failures were already there without the change and which come with it.
// This stands in for the agent's own `git stash` / rerun / `git stash pop` round trips, which cost turns, get
// compared by eye, and can leave the change stashed.
// Safety: only when the response made a single tool call (pi runs a response's calls in parallel), only for commands
// that don't write into the repo or touch git state, only while a tracked change exists, never late in a run, and
// the restored diff is compared with the one before the stash.

const TEST_COMMAND =
	/(\bpytest\b|py\.test|runtests\.py|\btox\b|\bnox\b|-m\s+unittest|manage\.py\s+test|bin\/test\b|bin\/doctest\b|\btest\(\)|\.test\(|\b(npm|yarn|pnpm)\s+(run\s+)?test|\bgo\s+test\b|\bcargo\s+test\b|\bjest\b|\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b)/;
const BASH_EDIT = /(\bsed\s+-i|\bpatch\b|\bgit\s+(apply|checkout|restore|stash\s+pop|stash\s+apply)\b|\bperl\s+-p?i)/;
const FAILURES =
	/\b[1-9]\d*\s+(failed|errors?|exceptions?)\b|FAILED\s+\((failures|errors)=|^FAILED\b|^ERROR:\s|^FAIL:\s|DO \*NOT\* COMMIT/m;
// Commands that change files, git state or the environment, run in the background, or depend on pytest's cache.
const UNSAFE =
	/\bgit\s+(stash|checkout|reset|restore|apply|am|commit|clean|switch|merge|rebase|cherry-pick|pull|push|worktree|add|rm|mv)\b|\bsed\s+(-[a-zA-Z]*i|--in-place)|\bperl\s+-[a-zA-Z]*i|\bpatch\b|\brm\s|\bmv\s|\bcp\s|\btee\b|\btouch\b|\bmkdir\b|\bpip\d?\s+(install|uninstall)\b|setup\.py\s+(develop|install|build)|\bnohup\b|\bkill\b|&\s*$|--lf\b|--last-failed|--ff\b|--failed-first|--sw\b|--stepwise|\bwrite_text\(|open\([^)]*['"][wa]/;
const ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;

const STASH_MSG = "pi-harness-baseline";
const MAX_BASELINES = 6;
const MAX_ORIGINAL_MS = 240_000;
const MAX_BASELINE_S = 300;
// A run is stopped from outside at its time limit; never hold the change in a stash near it.
const LATE_MS = 40 * 60_000;
const MAX_IDS_SHOWN = 10;

interface Baseline {
	output: string;
	timedOut: boolean;
	seconds: number;
}

function textOf(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.filter((b: any) => b && b.type === "text" && typeof b.text === "string")
		.map((b: any) => b.text)
		.join("\n");
}

function shellQuote(s: string): string {
	return `'${s.replace(/'/g, `'\\''`)}'`;
}

// Redirections are fine only into /tmp, /dev/null or another stream.
function writesIntoRepo(command: string): boolean {
	for (const m of command.matchAll(/\d*>>?\s*([^\s|;&)<>]+)/g)) {
		const target = m[1].replace(/^['"]|['"]$/g, "");
		if (!/^(\/tmp\/|\/dev\/null$|\/dev\/std(out|err)$)/.test(target)) return true;
	}
	return false;
}

// Failing test ids, by the most specific format the output has.
function failingIds(out: string): Set<string> {
	const pytest = new Set<string>();
	const unit = new Set<string>();
	const headers = new Set<string>();
	for (const raw of out.replace(ANSI, "").split("\n")) {
		const line = raw.trim();
		let m = /^(?:FAILED|ERROR)\s+(\S+::\S+)/.exec(line) ?? /^(\S+::\S+)\s+(?:FAILED|ERROR)\b/.exec(line);
		if (m) {
			pytest.add(m[1]);
			continue;
		}
		m = /^(?:FAIL|ERROR):\s+(\S+)\s+\(([^)]+)\)/.exec(line);
		if (m) {
			unit.add(`${m[1]} (${m[2]})`);
			continue;
		}
		m = /^_{3,}\s+(\S(?:.*\S)?)\s+_{3,}$/.exec(line);
		if (m && !/^(summary|FAILURES|ERRORS)$/i.test(m[1])) headers.add(m[1]);
	}
	if (pytest.size > 0) return pytest;
	if (unit.size > 0) return unit;
	return headers;
}

function verdicts(out: string): string[] {
	const lines = out
		.replace(ANSI, "")
		.split("\n")
		.map((l) => l.replace(/^=+\s*|\s*=+$/g, "").trim())
		.filter(Boolean);
	return lines
		.filter(
			(l) =>
				/^(OK|FAILED)(\s+\(.*\))?$/.test(l) ||
				/^Ran \d+ tests? in\b/.test(l) ||
				/\b\d+ (passed|failed|errors?|error)\b.*\bin [\d.]+ ?s/.test(l) ||
				/^tests finished:/.test(l) ||
				/^no tests ran\b/.test(l),
		)
		.slice(-3);
}

function normalizeVerdict(v: string[]): string {
	return v
		.map((l) => l.replace(/\bin [\d.]+ ?s(econds)?\b.*$/, "").trim())
		.join(" | ");
}

// Failures and errors a summary line counts, 0 when it shows a summary without any, undefined without a summary.
function failCount(out: string): number | undefined {
	const v = verdicts(out).join(" | ");
	if (!v) return undefined;
	let total = 0;
	for (const m of v.matchAll(/\b(\d+) (failed|errors?|exceptions?)\b/g)) total += Number(m[1]);
	for (const m of v.matchAll(/\b(failures|errors)=(\d+)/g)) total += Number(m[2]);
	return total;
}

function list(ids: string[]): string {
	const shown = ids.slice(0, MAX_IDS_SHOWN).join(", ");
	return ids.length > MAX_IDS_SHOWN ? `${shown}, ... (${ids.length - MAX_IDS_SHOWN} more)` : shown;
}

function compare(withChange: string, base: Baseline, cached: boolean): string {
	const head = [
		`[harness] Baseline for this failing run: the same command was run again with your uncommitted changes to tracked files stashed${cached ? " (earlier in this run; the code without your change has not changed since)" : ""}, and your changes were restored right after, so your files are as you left them.`,
	];
	if (base.timedOut) {
		head.push(`Without your change the command did not finish within ${base.seconds} s, so there is no comparison.`);
		return head.join("\n");
	}
	const now = verdicts(withChange);
	const before = verdicts(base.output);
	head.push(`Without your change: ${before.length > 0 ? before.join(" | ") : "(no test summary line in its output)"}`);

	const a = failingIds(withChange);
	const b = failingIds(base.output);
	const baseFails = FAILURES.test(base.output);
	if (a.size > 0 || b.size > 0) {
		const onlyNow = [...a].filter((id) => !b.has(id));
		const both = [...a].filter((id) => b.has(id));
		const onlyBefore = [...b].filter((id) => !a.has(id));
		if (onlyNow.length > 0) head.push(`Failing only with your change (${onlyNow.length}): ${list(onlyNow)}`);
		if (both.length > 0)
			head.push(`Failing without your change too, so not caused by it (${both.length}): ${list(both)}`);
		if (onlyBefore.length > 0)
			head.push(`Failing only without your change (${onlyBefore.length}): ${list(onlyBefore)}`);
		if (a.size === 0 && onlyBefore.length > 0)
			head.push("The output above names no failing test, so compare its summary with the one without your change.");
		return head.join("\n");
	}
	const nNow = failCount(withChange);
	const nBefore = failCount(base.output);
	if (!baseFails) {
		head.push("Without your change this command reports no failures, so the failures above come with your change.");
	} else if (now.length > 0 && normalizeVerdict(now) === normalizeVerdict(before)) {
		head.push("Same result without your change: these failures were already there before it.");
	} else if (nNow !== undefined && nBefore !== undefined) {
		head.push(
			`Failures and errors in the summaries: ${nNow} with your change, ${nBefore} without it. The output names no failing test, so only the counts can be compared.`,
		);
	} else {
		head.push("Without your change the command also reports failures; compare the two summaries.");
	}
	return head.join("\n");
}

export default function (pi: ExtensionAPI) {
	const started = Date.now();
	let edited = false;
	let toolCallsInResponse = 0;
	let baselinesRun = 0;
	const callStarts = new Map<string, number>();
	const cache = new Map<string, Baseline>();

	async function git(args: string[]) {
		return pi.exec("git", args, { timeout: 20_000 });
	}

	pi.on("message_end", async (event) => {
		const msg = event.message as any;
		if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) return;
		toolCallsInResponse = msg.content.filter((b: any) => b?.type === "toolCall").length;
	});

	pi.on("tool_call", async (event) => {
		callStarts.set(event.toolCallId, Date.now());
		if (event.toolName === "edit" || event.toolName === "write") {
			edited = true;
		} else if (event.toolName === "bash") {
			const command = String((event.input as { command?: unknown }).command ?? "");
			if (BASH_EDIT.test(command)) edited = true;
		}
	});

	async function runBaseline(command: string, originalMs: number): Promise<Baseline | "restore-failed" | undefined> {
		const before = await git(["diff", "HEAD", "--binary"]);
		if (before.code !== 0 || before.stdout.trim() === "") return undefined;
		const names = await git(["diff", "--name-only", "-z", "HEAD"]);
		const changedPaths = names.code === 0 ? names.stdout.split("\0").filter(Boolean) : [];
		const stashRef = async () => (await git(["rev-parse", "-q", "--verify", "refs/stash"])).stdout.trim();
		const stashBefore = await stashRef();
		const pushed = await git(["stash", "push", "--quiet", "-m", STASH_MSG]);
		if (pushed.code !== 0) return undefined;
		// Only pop an entry this push created.
		const stashAfter = await stashRef();
		if (stashAfter === "" || stashAfter === stashBefore) return undefined;

		const seconds = Math.min(MAX_BASELINE_S, Math.max(60, Math.ceil((2 * originalMs) / 1000) + 30));
		let output = "";
		let timedOut = false;
		try {
			const wrapped = `PYTEST_ADDOPTS="\${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }-p no:cacheprovider" timeout -k 5 ${seconds} bash -c ${shellQuote(command)}`;
			const r = await pi.exec("bash", ["-c", wrapped], { timeout: (seconds + 20) * 1000 });
			output = `${r.stdout}\n${r.stderr}`;
			timedOut = r.killed || r.code === 124 || r.code === 137;
		} catch {
			timedOut = true;
		} finally {
			// Anything the baseline run changed in tracked files belongs to the clean tree; drop it so the pop applies.
			const dirty = await git(["diff", "--quiet", "HEAD"]);
			if (dirty.code === 1) await git(["checkout", "-q", "--", ":/"]);
			// A file the run created at a path the stash restores (e.g. a new file of the change) would block the pop.
			if (changedPaths.length > 0) await git(["clean", "-fq", "--", ...changedPaths.map((p) => `:(top,literal)${p}`)]);
			const popped = await git(["stash", "pop", "--index", "--quiet"]);
			if (popped.code !== 0) await git(["stash", "pop", "--quiet"]);
		}
		const after = await git(["diff", "HEAD", "--binary"]);
		if (after.code !== 0 || after.stdout !== before.stdout) return "restore-failed";
		return { output, timedOut, seconds };
	}

	pi.on("tool_result", async (event) => {
		if (event.toolName !== "bash") return;
		const startedAt = callStarts.get(event.toolCallId);
		callStarts.delete(event.toolCallId);
		try {
			const command = String((event.input as { command?: unknown }).command ?? "");
			if (!edited || !TEST_COMMAND.test(command) || UNSAFE.test(command) || writesIntoRepo(command)) return;
			if (toolCallsInResponse !== 1 || event.parentToolCallId) return;
			const output = textOf(event.content);
			if (!FAILURES.test(output)) return;
			if (Date.now() - started > LATE_MS) return;
			const originalMs = startedAt ? Date.now() - startedAt : 0;
			if (originalMs > MAX_ORIGINAL_MS) return;

			// The code without the change is HEAD's, so a baseline holds for as long as HEAD does.
			const head = await git(["rev-parse", "HEAD"]);
			if (head.code !== 0) return;
			const key = `${head.stdout.trim()}\n${command}`;
			let base = cache.get(key);
			const cached = base !== undefined;
			if (!base) {
				if (baselinesRun >= MAX_BASELINES) return;
				baselinesRun++;
				const result = await runBaseline(command, originalMs);
				if (result === undefined) return;
				if (result === "restore-failed") {
					pi.appendEntry("acted", { what: "baseline could not restore the change" });
					return {
						content: [
							...event.content,
							{
								type: "text" as const,
								text: `[harness] To compare this failing run with the code without your change, the harness stashed your uncommitted changes (stash message "${STASH_MSG}") and could not restore them exactly. Check \`git stash list\` and \`git diff\` now, and run \`git stash pop\` if your change is still stashed.`,
							},
						],
					};
				}
				base = result;
				cache.set(key, base);
			}
			pi.appendEntry("acted", { what: cached ? "showed the cached baseline for failing tests" : "reran failing tests without the change" });
			return {
				content: [...event.content, { type: "text" as const, text: compare(output, base, cached) }],
			};
		} catch {
			return;
		}
	});
}
