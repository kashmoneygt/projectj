import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// What a bash test run tells the agent, in one tool_result note:
// 1. Results a filter cut. When a test run is piped through a filter (`| tail -5`, `| grep ...`), a `tee` before
//    the filter keeps the runner's unfiltered output; if the filtered result lacks the verdict or the failing test
//    names, they are added, with the path of the saved output.
// 2. Labels that name no module. Runners that take dotted labels (a `runtests.py` over a directory of test
//    packages, `python -m unittest`) report a missing label as an ordinary-looking error. Each label is checked
//    against the files on disk, and a missing one is named, with similar labels. Its `_FailedTest` error is then
//    left out of the failing tests in part 1, since this note already explains it.

// ---- 2. Labels that name no module ----

const MAX_WALK_FILES = 4000;
const MAX_FILE_BYTES = 2 * 1024 * 1024;

// runtests.py / unittest options whose value is the next word.
const VALUE_OPTS = new Set([
	"--verbosity", "--durations", "--settings", "-k", "--tag", "--exclude-tag", "--start-after", "--start-at",
	"--selenium", "--selenium-hub", "--external-host", "--bisect", "--pair", "--pythonpath", "--testrunner",
]);
// Options whose value is optional: skip the next word only when it looks like a value.
const OPTIONAL_VALUE_OPTS = new Set(["-v", "--parallel", "--shuffle"]);

const LABEL_RE = /^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$/;

type Tok = { word: string } | { op: string };

function tokenize(cmd: string): Tok[] {
	// Heredoc bodies are not commands; drop everything after the line that opens one.
	const hd = cmd.indexOf("<<");
	if (hd >= 0) {
		const nl = cmd.indexOf("\n", hd);
		if (nl >= 0) cmd = cmd.slice(0, nl);
	}
	const out: Tok[] = [];
	let i = 0;
	let cur = "";
	let inWord = false;
	let skipNext = false;
	const push = () => {
		if (inWord) {
			if (skipNext) skipNext = false;
			else out.push({ word: cur });
		}
		cur = "";
		inWord = false;
	};
	while (i < cmd.length) {
		const c = cmd[i];
		if (c === "'") {
			const j = cmd.indexOf("'", i + 1);
			const end = j < 0 ? cmd.length : j;
			cur += cmd.slice(i + 1, end);
			inWord = true;
			i = end + 1;
			continue;
		}
		if (c === '"') {
			let j = i + 1;
			while (j < cmd.length && cmd[j] !== '"') {
				if (cmd[j] === "\\" && j + 1 < cmd.length) {
					cur += cmd[j + 1];
					j += 2;
					continue;
				}
				cur += cmd[j];
				j++;
			}
			inWord = true;
			i = j + 1;
			continue;
		}
		if (c === "\\" && i + 1 < cmd.length) {
			if (cmd[i + 1] !== "\n") {
				cur += cmd[i + 1];
				inWord = true;
			}
			i += 2;
			continue;
		}
		if (c === " " || c === "\t") {
			push();
			i++;
			continue;
		}
		if (c === "<" || c === ">") {
			// A redirection: `2>&1`, `>file`, `> file`, `<in`. Its target is not an argument.
			if (inWord && /^\d+$/.test(cur)) {
				cur = "";
				inWord = false;
			} else push();
			let j = i;
			while (j < cmd.length && (cmd[j] === "<" || cmd[j] === ">")) j++;
			if (cmd[j] === "&") {
				j++;
				while (j < cmd.length && /[\d-]/.test(cmd[j])) j++;
			} else {
				while (j < cmd.length && (cmd[j] === " " || cmd[j] === "\t")) j++;
				skipNext = true;
			}
			i = j;
			continue;
		}
		if (c === "&" || c === "|" || c === ";" || c === "\n" || c === "(" || c === ")") {
			push();
			skipNext = false;
			const two = cmd.slice(i, i + 2);
			if (two === "&&" || two === "||") {
				out.push({ op: two });
				i += 2;
			} else {
				out.push({ op: c });
				i++;
			}
			continue;
		}
		cur += c;
		inWord = true;
		i++;
	}
	push();
	return out;
}

function simpleCommands(cmd: string): string[][] {
	const cmds: string[][] = [];
	let cur: string[] = [];
	for (const t of tokenize(cmd)) {
		if ("op" in t) {
			if (cur.length) cmds.push(cur);
			cur = [];
		} else cur.push(t.word);
	}
	if (cur.length) cmds.push(cur);
	return cmds;
}

function isDir(p: string): boolean {
	try {
		return statSync(p).isDirectory();
	} catch {
		return false;
	}
}

function isFile(p: string): boolean {
	try {
		return statSync(p).isFile();
	} catch {
		return false;
	}
}

function readText(p: string): string {
	try {
		if (statSync(p).size > MAX_FILE_BYTES) return "";
		return readFileSync(p, "utf8");
	} catch {
		return "";
	}
}

function hasWord(text: string, name: string): boolean {
	return new RegExp(`\\b${name}\\b`).test(text);
}

function levenshtein(a: string, b: string): number {
	const m = a.length;
	const n = b.length;
	let prev = Array.from({ length: n + 1 }, (_, j) => j);
	for (let i = 1; i <= m; i++) {
		const cur = [i];
		for (let j = 1; j <= n; j++) {
			cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
		}
		prev = cur;
	}
	return prev[n];
}

function similarity(a: string, b: string): number {
	a = a.toLowerCase();
	b = b.toLowerCase();
	if (a === b) return 1;
	if (a.length >= 3 && b.length >= 3 && (a.includes(b) || b.includes(a))) {
		return 0.9 - Math.abs(a.length - b.length) * 0.005;
	}
	return 1 - levenshtein(a, b) / Math.max(a.length, b.length);
}

// Module names in a directory: packages and .py files.
function moduleNames(dir: string): string[] {
	let entries: string[];
	try {
		entries = readdirSync(dir);
	} catch {
		return [];
	}
	const names: string[] = [];
	for (const e of entries) {
		if (e.startsWith(".") || e.startsWith("__")) continue;
		if (e.endsWith(".py")) names.push(e.slice(0, -3));
		else if (!e.includes(".") && isDir(join(dir, e))) names.push(e);
	}
	return names;
}

// The test modules of a package, as labels: "pkg.tests, pkg.test_x" (at most `limit`).
function testModules(dir: string, label: string, limit = 4): string {
	const mods = moduleNames(dir).filter((n) => /^tests?$|^tests?_|_tests?$/.test(n)).sort();
	if (!mods.length) return "";
	const shown = mods.slice(0, limit).map((m) => `${label}.${m}`);
	return mods.length > limit ? `${shown.join(", ")}, …` : shown.join(", ");
}

// Similar names with, for the closest one that is a package, the test modules in it.
function describeSimilar(names: string[], dir: string, prefix: string): string {
	if (!names.length) return "";
	const label = (n: string) => (prefix ? `${prefix}.${n}` : n);
	const first = names[0];
	const inner = isDir(join(dir, first)) ? testModules(join(dir, first), label(first)) : "";
	const parts = names.map((n, i) => (i === 0 && inner ? `${label(n)} (test modules: ${inner})` : label(n)));
	return ` Similar: ${parts.join(", ")}`;
}

function similarNames(name: string, dir: string, limit = 3): string[] {
	return moduleNames(dir)
		.map((n) => ({ n, s: similarity(name, n) }))
		.filter((x) => x.s >= 0.6 && x.n !== name)
		.sort((x, y) => y.s - x.s || x.n.localeCompare(y.n))
		.slice(0, limit)
		.map((x) => x.n);
}

// Labels (relative to root) of the modules under dir that define `class <name>`.
function findClass(name: string, dir: string, root: string, limit = 3): string[] {
	const found: string[] = [];
	const re = new RegExp(`^\\s*class\\s+${name}\\b`, "m");
	let seen = 0;
	const walk = (d: string, depth: number) => {
		if (found.length >= limit || seen > MAX_WALK_FILES || depth > 4) return;
		let entries: string[];
		try {
			entries = readdirSync(d).sort();
		} catch {
			return;
		}
		for (const e of entries) {
			if (found.length >= limit || seen > MAX_WALK_FILES) return;
			if (e.startsWith(".") || e === "__pycache__" || e === "node_modules") continue;
			const p = join(d, e);
			if (e.endsWith(".py")) {
				seen++;
				if (re.test(readText(p))) {
					found.push(relative(root, p).slice(0, -3).split("/").join("."));
				}
			} else if (!e.includes(".") && isDir(p)) walk(p, depth + 1);
		}
	};
	walk(dir, 0);
	return found;
}

// Paths in the note are shown relative to the directory the agent started in.
let displayBase = process.cwd();
function show(p: string): string {
	const r = relative(displayBase, p);
	if (r === "") return ".";
	return r.startsWith("..") ? p : r;
}

// Returns a description of what's wrong with the label, or null when it resolves (or can't be judged).
// strictTop: every label must start under root (a runtests.py over test packages). Otherwise (unittest) a top-level
// name not found under root may be an installed package, and isn't judged.
function checkLabel(label: string, root: string, cwd: string, strictTop: boolean): { msg: string; part: string } | null {
	// File paths (`test_x.py`) are a different kind of label and aren't judged.
	if (!LABEL_RE.test(label) || label.endsWith(".py")) return null;
	const parts = label.split(".");
	if (!isDir(join(root, parts[0])) && !isFile(join(root, `${parts[0]}.py`))) {
		if (!strictTop) return null;
		// The project's own package (or anything else next to the test root) is importable another way.
		if (isDir(join(dirname(root), parts[0])) || isFile(join(dirname(root), `${parts[0]}.py`))) return null;
		if (isDir(join(cwd, parts[0])) || isFile(join(cwd, `${parts[0]}.py`))) return null;
		const sim = similarNames(parts[0], root);
		const where = show(root);
		const cls = /^[A-Z]/.test(parts[0]) ? findClass(parts[0], root, root) : [];
		let msg = `\`${label}\`: ${where}/ has no ${parts[0]}/ or ${parts[0]}.py.`;
		if (cls.length) msg += ` class ${parts[0]} is in: ${cls.map((c) => `${c}.${parts.join(".")}`).join(", ")}`;
		else msg += describeSimilar(sim, root, "");
		return { msg, part: parts[0] };
	}
	let cur = root;
	for (let k = 0; k < parts.length; k++) {
		const part = parts[k];
		if (isDir(cur)) {
			if (isDir(join(cur, part))) {
				cur = join(cur, part);
				continue;
			}
			if (isFile(join(cur, `${part}.py`))) {
				cur = join(cur, `${part}.py`);
				continue;
			}
			// Something the package's __init__ defines or imports.
			const init = readText(join(cur, "__init__.py"));
			if (hasWord(init, part) || /\bimport\s+\*/.test(init)) return null;
			const prefix = parts.slice(0, k).join(".");
			const where = show(cur);
			if (/^[A-Z]/.test(part)) {
				const cls = findClass(part, cur, root);
				const rest = parts.slice(k).join(".");
				let msg = `\`${label}\`: ${where}/ has no module ${part}.`;
				if (cls.length) msg += ` class ${part} is defined in ${cls.join(", ")} (label: ${cls.map((c) => `${c}.${rest}`).join(", ")})`;
				return { msg, part };
			}
			let msg = `\`${label}\`: ${where}/ has no ${part}/ or ${part}.py.`;
			const sim = similarNames(part, cur);
			const mods = testModules(cur, prefix, 6);
			if (sim.length) msg += describeSimilar(sim, cur, prefix);
			else if (mods) msg += ` Its test modules: ${mods}`;
			else msg += describeSimilar(similarNames(part, root), root, "");
			return { msg, part };
		}
		// cur is a module file; part is an attribute of it (a test class). Methods are not checked.
		const text = readText(cur);
		if (!text || hasWord(text, part) || /\bimport\s+\*/.test(text)) return null;
		const modLabel = parts.slice(0, k).join(".");
		const topDir = join(root, parts[0]);
		const cls = /^[A-Z]/.test(part) && isDir(topDir) ? findClass(part, topDir, root) : [];
		const rest = parts.slice(k).join(".");
		let msg = `\`${label}\`: ${show(cur)} has no ${part}.`;
		if (cls.length) msg += ` class ${part} is defined in ${cls.join(", ")} (label: ${cls.map((c) => `${c}.${rest}`).join(", ")})`;
		else {
			const classes = [...text.matchAll(/^class\s+([A-Za-z_]\w*)/gm)].map((m) => m[1]).slice(0, 8);
			if (classes.length) msg += ` Classes in ${modLabel}: ${classes.join(", ")}`;
		}
		return { msg, part };
	}
	return null;
}

// A directory holding at least three packages with test modules (tests.py, test_*.py or a tests/ package).
function isTestPackageRoot(dir: string): boolean {
	let entries: string[];
	try {
		entries = readdirSync(dir);
	} catch {
		return false;
	}
	let n = 0;
	for (const e of entries) {
		if (e.startsWith(".") || e.startsWith("__") || e.includes(".")) continue;
		const d = join(dir, e);
		let inner: string[];
		try {
			inner = readdirSync(d);
		} catch {
			continue;
		}
		if (inner.some((f) => f === "tests.py" || f === "tests" || /^test_.*\.py$/.test(f))) {
			if (++n >= 3) return true;
		}
	}
	return false;
}

interface Problem {
	root: string;
	// The label component the runner names in its `_FailedTest` error.
	part: string;
	msg: string;
}

function resolveDir(cwd: string, arg: string): string | null {
	if (!arg || arg.includes("$") || arg.includes("`") || arg === "-") return null;
	if (arg.startsWith("~")) {
		const home = process.env.HOME;
		if (!home) return null;
		arg = home + arg.slice(1);
	}
	return resolve(cwd, arg);
}

function labelsAfter(words: string[]): string[] {
	const labels: string[] = [];
	for (let i = 0; i < words.length; i++) {
		const w = words[i];
		if (w.startsWith("-")) {
			if (!w.includes("=") && VALUE_OPTS.has(w)) i++;
			else if (!w.includes("=") && OPTIONAL_VALUE_OPTS.has(w) && i + 1 < words.length && /^(\d+|auto)$/.test(words[i + 1])) i++;
			continue;
		}
		labels.push(w);
	}
	return labels;
}

export function checkCommand(command: string, startCwd: string): Problem[] {
	const problems: Problem[] = [];
	displayBase = startCwd;
	let cwd: string | null = startCwd;
	for (const words of simpleCommands(command)) {
		if (words[0] === "cd") {
			cwd = cwd === null ? null : resolveDir(cwd, words[1] ?? process.env.HOME ?? "");
			continue;
		}
		if (cwd === null) continue;
		let root: string | null = null;
		let rest: string[] = [];
		let strictTop = false;
		const si = words.findIndex((w) => /(^|\/)runtests\.py$/.test(w));
		if (si >= 0) {
			const script = resolve(cwd, words[si]);
			// Only a runner that takes labels and sits in a directory of test packages, whose labels are
			// module paths under that directory.
			if (!isFile(script) || !/label/i.test(readText(script)) || !isTestPackageRoot(dirname(script))) continue;
			root = dirname(script);
			rest = words.slice(si + 1);
			strictTop = true;
		} else {
			const mi = words.findIndex((w, j) => (w === "-m" && words[j + 1] === "unittest") || w === "-munittest");
			if (mi < 0) continue;
			root = cwd;
			rest = words.slice(words[mi] === "-m" ? mi + 2 : mi + 1);
			if (rest[0] === "discover") continue;
		}
		for (const label of labelsAfter(rest)) {
			const found = checkLabel(label, root, cwd, strictTop);
			if (found && !problems.some((p) => p.msg === found.msg)) problems.push({ root, part: found.part, msg: found.msg });
		}
	}
	return problems;
}

export function formatNote(problems: Problem[]): string {
	const roots = [...new Set(problems.map((p) => `${show(p.root)}/`))];
	const one = problems.length === 1;
	return [
		`[test labels] ${one ? "This label names" : "These labels name"} no module on disk, so the runner can't load ${one ? "it" : "them"}. It reports each as an error of its own (\`ERROR: <label> (unittest.loader._FailedTest)\`, ImportError or AttributeError), counted in \`FAILED (errors=N)\`. That error comes from the label, not from the code, and the tests meant by ${one ? "it" : "them"} did not run:`,
		...problems.map((p) => `- ${p.msg}`),
		`Labels are dotted paths under ${roots.join(", ")}: package.module.Class.method.`,
	].join("\n");
}

// ---- 1. Results a filter cut ----

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
// `covered` marks failing ids another note already explains (a label that names no module).
export function note(
	visibleRaw: string,
	logRaw: string,
	log: string,
	covered: (id: string) => boolean = () => false,
): string | undefined {
	const visible = visibleRaw.replace(ANSI, "");
	const visibleLines = visible.split(/\r?\n/).map((l) => l.trim());
	const shows = (text: string) =>
		text.length < 12 ? visibleLines.includes(text) : visibleLines.some((l) => l.includes(text));
	const all = digest(logRaw);
	const d: Digest = { summary: all.summary, failing: all.failing.filter((f) => !covered(f)) };
	// The last summary line carries the verdict (unittest's OK/FAILED, pytest's or other runners' totals).
	const summaryShown = d.summary.length === 0 || shows(d.summary[d.summary.length - 1]);
	// A failing test counts as shown only when its whole id is in view: a bare name like `tests` also matches
	// "Ran 2 tests", and would hide a real import error.
	const missingFailing = d.failing.filter((f) => !visible.includes(f.replace(/^(FAIL|ERROR|FAILED):?\s+/, "")));
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

// `ERROR: <name> (unittest.loader._FailedTest)` where <name> is the part of a label found missing.
function labelError(id: string, names: Set<string>): boolean {
	const m = /^ERROR: (\S+) \(unittest\.loader\._FailedTest(?:\.\S+)?\)/.exec(id);
	return !!m && names.has(m[1]);
}

const LABEL_RUNNER = /runtests\.py|-m\s*unittest/;

export default function (pi: ExtensionAPI) {
	const pending = new Map<string, { command: string; logs: string[] }>();
	let calls = 0;
	// Without tee the inserted stage would break the pipeline, so leave commands alone.
	const canTap = haveTee();

	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash") return;
		try {
			const input = event.input as { command?: unknown };
			const command = typeof input.command === "string" ? input.command : "";
			if (!command || !(TEST_COMMAND.test(command) || LABEL_RUNNER.test(command))) return;
			let logs: string[] = [];
			if (canTap && TEST_COMMAND.test(command) && command.includes("|")) {
				calls++;
				const id = `${process.pid}-${calls}`;
				const taps = planTaps(command, (k) => `${LOG_DIR}/run-${id}-${k}.log`);
				if (taps.length > 0) {
					mkdirSync(LOG_DIR, { recursive: true });
					input.command = applyTaps(command, taps);
					logs = taps.map((t) => t.log);
				}
			}
			pending.set(event.toolCallId, { command, logs });
		} catch {
			// Leave the command as the model wrote it.
		}
	});

	pi.on("tool_result", async (event, ctx) => {
		if (event.toolName !== "bash") return;
		const call = pending.get(event.toolCallId);
		if (!call) return;
		pending.delete(event.toolCallId);
		try {
			let problems: Problem[] = [];
			if (LABEL_RUNNER.test(call.command)) {
				try {
					problems = checkCommand(call.command, ctx.cwd || process.cwd());
				} catch {
					problems = [];
				}
			}
			const missing = new Set(problems.map((p) => p.part));
			const notes: string[] = [];
			if (call.logs.length > 0) {
				const visible = event.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");
				for (const log of call.logs) {
					const raw = readTail(log);
					if (raw === undefined) continue;
					const n = note(visible, raw, log, (id) => labelError(id, missing));
					if (n) notes.push(n);
				}
				if (notes.length > 0) pi.appendEntry("acted", { what: "added the test results a filter cut" });
			}
			if (problems.length > 0) {
				notes.push(formatNote(problems));
				pi.appendEntry("acted", { what: "explained missing test labels" });
			}
			if (notes.length === 0) return;
			return { content: [...event.content, { type: "text", text: `\n${notes.join("\n")}` }] };
		} catch {
			return;
		}
	});
}
