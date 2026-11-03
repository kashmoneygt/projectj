import { closeSync, existsSync, mkdirSync, openSync, readSync, statSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// When a bash command pipes a test runner's output through a filter (`| tail -5`, `| grep ...`), the
// runner's summary and the names of the failing tests are often cut away, and the next call reruns the
// suite just to see them. This keeps a copy of the runner's unfiltered output (a `tee` inserted before
// the filter) and, when the filtered output lacks the summary or the failing test names, adds them to the
// bash result together with the path of the saved output.

// Same notion of a test command as finish-guard.ts.
const TEST_COMMAND =
	/(\bpytest\b|py\.test|runtests\.py|\btox\b|\bnox\b|-m\s+unittest|manage\.py\s+test|bin\/test\b|\btest\(\)|\b(npm|yarn|pnpm)\s+(run\s+)?test|\bgo\s+test\b|\bcargo\s+test\b|\bmvn\b|\bgradlew?\b|\bmake\s+(check|test)\b|\bctest\b|\bjest\b|\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b)/;
// Filters that read the whole stream before ending. `head` is left out: it closes the pipe early and the
// runner would stop part way, with or without a copy.
const FILTER = /^\s*(tail|grep|egrep|fgrep|sed|awk|sort|uniq|cut|tr|cat|less|more|wc)\b/;
// A left side that only mentions a runner (searching or printing files) is not a test run.
const NOT_A_RUN = /^\s*(grep|egrep|fgrep|rg|cat|find|ls|echo|printf|git|sed|awk|head|tail|less|wc|diff)\b/;
const LOG_DIR = "/tmp/pi-test-output";
const MAX_LOG_BYTES = 8 * 1024 * 1024;
const MAX_FAILING_SHOWN = 25;
const MAX_LINE = 300;
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

interface Insertion {
	at: number;
	log: string;
}

// Finds the pipes to tap: at the top level of the command (outside quotes, $(...), (...), {...}), the
// first `|` or `|&` of a pipeline whose left side runs tests and whose right side starts with a filter.
// Returns nothing for commands it can't read safely (heredocs, loops, background jobs, unbalanced quotes).
export function planTaps(command: string, logFor: (k: number) => string): Insertion[] {
	// A loop would run the tap many times over one file.
	if (command.includes("<<") || /\b(for|while|until)\b[\s\S]*\bdo\b/.test(command)) return [];
	const out: Insertion[] = [];
	let depth = 0;
	let quote: string | null = null;
	let segStart = 0;
	let pipeSeen = false;
	const n = command.length;
	for (let i = 0; i < n; i++) {
		const c = command[i];
		if (quote) {
			if (c === "\\" && quote === '"') {
				i++;
				continue;
			}
			if (c === quote) quote = null;
			continue;
		}
		if (c === "\\") {
			i++;
			continue;
		}
		if (c === "'" || c === '"' || c === "`") {
			quote = c;
			continue;
		}
		if (c === "#" && (i === 0 || /\s/.test(command[i - 1]))) {
			// A comment runs to the end of the line.
			const nl = command.indexOf("\n", i);
			if (nl < 0) break;
			i = nl - 1;
			continue;
		}
		if (c === "(" || c === "{") {
			depth++;
			continue;
		}
		if (c === ")" || c === "}") {
			depth = Math.max(0, depth - 1);
			continue;
		}
		if (depth > 0) continue;
		const next = command[i + 1];
		const prev = i > 0 ? command[i - 1] : "";
		if (c === "&") {
			if (next === "&") {
				i++;
				segStart = i + 1;
				pipeSeen = false;
				continue;
			}
			if (next === ">" || prev === ">" || prev === "<" || prev === "|") continue;
			// A background job: leave the command alone.
			return [];
		}
		if (c === ";" || c === "\n") {
			segStart = i + 1;
			pipeSeen = false;
			continue;
		}
		if (c === "|") {
			if (next === "|") {
				i++;
				segStart = i + 1;
				pipeSeen = false;
				continue;
			}
			const after = next === "&" ? i + 2 : i + 1;
			if (!pipeSeen) {
				pipeSeen = true;
				const left = command.slice(segStart, i);
				const right = command.slice(after);
				if (TEST_COMMAND.test(left) && !NOT_A_RUN.test(left) && FILTER.test(right) && out.length < 3) {
					out.push({ at: after, log: logFor(out.length + 1) });
				}
			}
			if (next === "&") i++;
		}
	}
	if (quote || depth !== 0) return [];
	return out;
}

export function applyTaps(command: string, taps: Insertion[]): string {
	let result = command;
	for (const tap of [...taps].sort((a, b) => b.at - a.at)) {
		result = `${result.slice(0, tap.at)} tee '${tap.log}' |${result.slice(tap.at)}`;
	}
	return result;
}

function clip(line: string): string {
	const t = line.trim();
	return t.length > MAX_LINE ? `${t.slice(0, MAX_LINE)}…` : t;
}

function readTail(path: string): string | undefined {
	try {
		const size = statSync(path).size;
		const len = Math.min(size, MAX_LOG_BYTES);
		const buf = Buffer.alloc(len);
		const fd = openSync(path, "r");
		try {
			readSync(fd, buf, 0, len, size - len);
		} finally {
			closeSync(fd);
		}
		return buf.toString("utf8");
	} catch {
		return undefined;
	}
}

export interface Digest {
	summary: string[];
	failing: string[];
}

// The runner's closing summary and the tests it reports as failing, for the output formats of unittest
// (and runners built on it), pytest, runners that end with "tests finished: ...", and a few common others.
export function digest(raw: string): Digest {
	const lines = raw.replace(ANSI, "").split(/\r?\n/);
	const summary: string[] = [];
	const failing: string[] = [];
	const seen = new Set<string>();
	const addFailing = (id: string) => {
		const t = clip(id);
		if (t && !seen.has(t)) {
			seen.add(t);
			failing.push(t);
		}
	};
	let ran = -1;
	let pytestLine = -1;
	let finishedLine = -1;
	const other: number[] = [];
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		let m: RegExpExecArray | null;
		if (/^Ran \d+ tests? in /.test(line)) ran = i;
		else if (/^=+ .*\b(passed|failed|errors?|skipped|deselected|xfailed|xpassed|no tests ran)\b.* in [\d.]+s.* =+$/.test(line))
			pytestLine = i;
		else if (/^\d+ (passed|failed|errors?|skipped|deselected)\b.* in [\d.]+s/.test(line)) pytestLine = i;
		else if (/^=*\s*no tests ran in [\d.]+s/.test(line)) pytestLine = i;
		else if (/tests finished: .* in [\d.]+ seconds/.test(line)) finishedLine = i;
		else if (/^(test result: |Tests?:\s+\d+|Test Suites:\s+\d+)/.test(line.trim())) other.push(i);
		if ((m = /^(FAIL|ERROR): (\S+ \(.*\))/.exec(line))) addFailing(`${m[1]}: ${m[2]}`);
		else if ((m = /^(FAILED|ERROR) (\S+::\S+|\S+\.py)/.exec(line))) addFailing(`${m[1]} ${m[2]}`);
		else if ((m = /^_{3,} (\S+\.py:\S+) _{3,}$/.exec(line))) addFailing(m[1]);
		else if ((m = /^--- FAIL: (\S+)/.exec(line))) addFailing(`FAIL: ${m[1]}`);
	}
	if (ran >= 0) {
		summary.push(clip(lines[ran]));
		for (let j = ran + 1; j < Math.min(lines.length, ran + 4); j++) {
			if (/^(OK|FAILED)\b/.test(lines[j])) {
				summary.push(clip(lines[j]));
				break;
			}
		}
	}
	if (pytestLine >= 0) {
		// When nothing ran, the reason is usually a usage error printed as `ERROR: ...`.
		if (/no tests ran/.test(lines[pytestLine]))
			summary.push(...lines.filter((l) => /^ERROR: (?!\S+ \()/.test(l)).slice(-3).map(clip));
		summary.push(clip(lines[pytestLine].replace(/^=+\s*|\s*=+$/g, "")));
	}
	if (finishedLine >= 0) summary.push(clip(lines[finishedLine].replace(/^=+\s*|\s*=+$/g, "")));
	for (const i of other.slice(-3)) summary.push(clip(lines[i]));
	return { summary, failing };
}

function lastLines(raw: string, count: number): string[] {
	const lines = raw.replace(ANSI, "").split(/\r?\n/).filter((l) => l.trim() !== "");
	return lines.slice(-count).map(clip);
}

// What to add to the filtered output, or undefined when it already shows everything that matters.
export function note(visibleRaw: string, logRaw: string, log: string): string | undefined {
	const visible = visibleRaw.replace(ANSI, "");
	const visibleLines = visible.split(/\r?\n/).map((l) => l.trim());
	const shows = (text: string) =>
		text.length < 12 ? visibleLines.includes(text) : visibleLines.some((l) => l.includes(text));
	const d = digest(logRaw);
	// The last summary line carries the verdict (unittest's OK/FAILED, pytest's or other runners' totals).
	const summaryShown = d.summary.length === 0 || shows(d.summary[d.summary.length - 1]);
	const missingFailing = d.failing.filter((f) => !visible.includes(f.replace(/^(FAIL|ERROR|FAILED):?\s+/, "").split(" ")[0]));
	if (d.summary.length === 0 && d.failing.length === 0) {
		// Nothing recognizable, e.g. the runner failed to start. Only speak up when the filter left
		// nothing at all to look at.
		const shown = visible
			.split(/\r?\n/)
			.filter((l) => l.trim() !== "" && !/^\(no output\)$/.test(l.trim()) && !/^Command exited with code/.test(l.trim()));
		const tail = lastLines(logRaw, 12);
		if (shown.length > 0 || tail.length === 0) return undefined;
		return [
			`[Your filter left no output. The last lines of the unfiltered test output (saved in full at ${log}):`,
			...tail,
			"]",
		].join("\n");
	}
	if (summaryShown && missingFailing.length === 0) return undefined;
	const parts = [`[Test results your filter cut, from the unfiltered output (saved in full at ${log}; grep it instead of rerunning):`];
	if (d.summary.length === 0) parts.push("(no summary line found; the run may have stopped early)");
	else if (!summaryShown) parts.push(...d.summary);
	if (missingFailing.length > 0) {
		const shown = d.failing.slice(0, MAX_FAILING_SHOWN);
		parts.push(`Failing (${d.failing.length}):`, ...shown);
		if (d.failing.length > shown.length) parts.push(`… ${d.failing.length - shown.length} more in the saved output`);
	}
	parts.push("]");
	return parts.join("\n");
}

function haveTee(): boolean {
	for (const dir of (process.env.PATH ?? "/usr/bin:/bin").split(":")) {
		if (dir && existsSync(`${dir}/tee`)) return true;
	}
	return false;
}

export default function (pi: ExtensionAPI) {
	const pending = new Map<string, string[]>();
	let calls = 0;
	// Without tee the inserted stage would break the pipeline, so leave commands alone.
	const enabled = haveTee();

	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash" || !enabled) return;
		try {
			const input = event.input as { command?: unknown };
			const command = typeof input.command === "string" ? input.command : "";
			if (!command || !TEST_COMMAND.test(command) || !command.includes("|")) return;
			calls++;
			const id = `${process.pid}-${calls}`;
			const taps = planTaps(command, (k) => `${LOG_DIR}/run-${id}-${k}.log`);
			if (taps.length === 0) return;
			mkdirSync(LOG_DIR, { recursive: true });
			input.command = applyTaps(command, taps);
			pending.set(event.toolCallId, taps.map((t) => t.log));
		} catch {
			// Leave the command as the model wrote it.
		}
	});

	pi.on("tool_result", async (event) => {
		const logs = pending.get(event.toolCallId);
		if (!logs) return;
		pending.delete(event.toolCallId);
		try {
			const visible = event.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");
			const notes: string[] = [];
			for (const log of logs) {
				const raw = readTail(log);
				if (raw === undefined) continue;
				const n = note(visible, raw, log);
				if (n) notes.push(n);
			}
			if (notes.length === 0) return;
			pi.appendEntry("acted", { what: "added the test results a filter cut" });
			return { content: [...event.content, { type: "text", text: `\n${notes.join("\n")}` }] };
		} catch {
			return;
		}
	});
}
