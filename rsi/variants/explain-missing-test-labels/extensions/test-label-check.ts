import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Test runners that take dotted module labels (a `runtests.py` over a directory of test packages,
// `python -m unittest`) report a label that doesn't exist as an ordinary-looking error
// ("ERROR: x (unittest.loader._FailedTest)", counted in "FAILED (errors=N)"), which a filtered view
// (| tail, | grep) often hides. After such a bash call, check each label against the files on disk and,
// when one doesn't exist, say so next to the output, with similar labels.

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
function checkLabel(label: string, root: string, cwd: string, strictTop: boolean): string | null {
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
		return msg;
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
				return msg;
			}
			let msg = `\`${label}\`: ${where}/ has no ${part}/ or ${part}.py.`;
			const sim = similarNames(part, cur);
			const mods = testModules(cur, prefix, 6);
			if (sim.length) msg += describeSimilar(sim, cur, prefix);
			else if (mods) msg += ` Its test modules: ${mods}`;
			else msg += describeSimilar(similarNames(part, root), root, "");
			return msg;
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
		return msg;
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
			const msg = checkLabel(label, root, cwd, strictTop);
			if (msg && !problems.some((p) => p.msg === msg)) problems.push({ root, msg });
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

export default function (pi: ExtensionAPI) {
	pi.on("tool_result", async (event, ctx) => {
		if (event.toolName !== "bash") return;
		const command = (event.input as { command?: unknown })?.command;
		if (typeof command !== "string" || !/runtests\.py|-m\s*unittest/.test(command)) return;
		const cwd = ctx.cwd || process.cwd();
		let problems: Problem[];
		try {
			problems = checkCommand(command, cwd);
		} catch {
			return;
		}
		if (!problems.length) return;
		pi.appendEntry("acted", { what: "explained missing test labels" });
		return { content: [...event.content, { type: "text", text: formatNote(problems) }] };
	});
}
