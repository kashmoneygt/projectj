import { readFileSync, realpathSync, statSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// After a successful edit/write of a Python source file, find the function and class the changed lines
// sit in and append the existing test files (and test functions) that reference them, once per name.

const GIT_TIMEOUT_MS = 10_000;
const MAX_FILES_SHOWN = 4;
const MAX_TESTS_PER_FILE = 6;
// A name referenced from more test files than this is too common to point at anything.
const MAX_FILES_FOR_NAME = 15;
const MAX_NAMES_PER_EDIT = 2;
const MAX_SEARCHES_PER_EDIT = 4;
const MAX_FILE_BYTES = 3 * 1024 * 1024;

const GENERIC = new Set([
	"setup", "setUp", "tearDown", "main", "run", "get", "set", "save", "call", "update", "apply", "check",
	"handle", "process", "execute", "render", "clean", "validate", "parse", "format", "copy", "close", "open",
	"read", "write", "load", "dump", "init", "build", "test", "wrapper", "inner", "decorator", "helper",
	"func", "function", "value", "values", "items", "keys", "name", "data", "args", "kwargs", "self", "cls",
	"Meta", "Options", "Base", "Mixin", "Error", "Test", "Config", "Settings",
]);

const DEF_RE = /^(\s*)(?:async\s+def|def|class)\s+([A-Za-z_]\w*)/;

// Test code: anything under a `tests/` directory, a top-level `test/` or `testing/` directory, or a file
// named like a test module. (A nested `<pkg>/test/` or `<pkg>/testing/` package is usually library code.)
function isTestPath(p: string): boolean {
	return (
		/(^|\/)tests\//.test(p) ||
		/^(test|testing)\//.test(p) ||
		/(^|\/)(test_[^/]*\.py|[^/]*_tests?\.py|tests?\.py|conftest\.py)$/.test(p)
	);
}

function indentOf(line: string): number {
	const m = /^(\s*)/.exec(line);
	return m ? m[1].replace(/\t/g, "        ").length : 0;
}

interface Scope {
	kind: "def" | "class";
	name: string;
}

// For each line, whether it continues an earlier one: it starts inside a triple-quoted string or an open
// bracket, or follows a backslash. Such lines say nothing about block nesting.
function continuationLines(lines: string[]): boolean[] {
	const out: boolean[] = [];
	let delim: string | null = null;
	let depth = 0;
	let backslash = false;
	for (const line of lines) {
		// A top-level def/class/decorator resets the tracking, so one misread line can't hide the rest.
		if (delim === null && /^(?:async\s+def|def|class|@)/.test(line)) {
			depth = 0;
			backslash = false;
		}
		out.push(delim !== null || depth > 0 || backslash);
		let code = "";
		let pos = 0;
		for (;;) {
			if (delim !== null) {
				const end = line.indexOf(delim, pos);
				if (end < 0) break;
				delim = null;
				pos = end + 3;
				continue;
			}
			const hash = line.indexOf("#", pos);
			const d = line.indexOf('"""', pos);
			const q = line.indexOf("'''", pos);
			const idx = d < 0 ? q : q < 0 ? d : Math.min(d, q);
			if (idx < 0 || (hash >= 0 && hash < idx)) {
				code += line.slice(pos);
				break;
			}
			code += line.slice(pos, idx);
			delim = line.slice(idx, idx + 3);
			pos = idx + 3;
		}
		code = code.replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"/g, "").replace(/#.*/, "");
		for (const ch of code) {
			if (ch === "(" || ch === "[" || ch === "{") depth++;
			else if ((ch === ")" || ch === "]" || ch === "}") && depth > 0) depth--;
		}
		backslash = delim === null && /\\\s*$/.test(code);
	}
	return out;
}

// Innermost-first list of the def/class blocks that contain 1-based line `lineNo`.
function enclosingScopes(lines: string[], inStr: boolean[], lineNo: number): Scope[] {
	const scopes: Scope[] = [];
	let start = Math.min(Math.max(lineNo, 1), lines.length) - 1;
	while (start > 0 && inStr[start]) start--;
	if (start < 0 || start >= lines.length) return scopes;
	let cur = indentOf(lines[start]);
	for (let i = start; i >= 0; i--) {
		const line = lines[i];
		if (line.trim() === "" || (inStr[i] && i !== start)) continue;
		const m = DEF_RE.exec(line);
		const ind = indentOf(line);
		const t = line.trimStart();
		if (m && (i === start || ind < cur)) {
			scopes.push({ kind: t.startsWith("class") ? "class" : "def", name: m[2] });
			cur = ind;
			if (cur === 0) break;
		} else if (!m && ind < cur && !t.startsWith("#") && !t.startsWith("@") && !/^[)\]}]/.test(t)) {
			// A dedented statement ends the blocks deeper than it.
			cur = ind;
			if (cur === 0) break;
		}
	}
	return scopes;
}

function isGeneric(name: string): boolean {
	return (name.startsWith("__") && name.endsWith("__")) || name.length < 4 || GENERIC.has(name);
}

// New-file line ranges [first, last] where `git diff -U0 HEAD -- file` has hunks. A pure deletion is
// reported after line `start`, which sits in the block the lines were removed from.
function changedRanges(diff: string): [number, number][] {
	const out: [number, number][] = [];
	for (const m of diff.matchAll(/^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/gm)) {
		const start = Number(m[1]);
		const count = m[2] === undefined ? 1 : Number(m[2]);
		out.push(count === 0 ? [Math.max(start, 1), Math.max(start, 1)] : [start, start + count - 1]);
	}
	return out;
}

function isCode(line: string | undefined): boolean {
	if (line === undefined) return false;
	const t = line.trim();
	return t !== "" && !t.startsWith("#");
}

// The first and last code lines of a changed range; falls back to the nearest code line above it.
function probeLines(lines: string[], [a, b]: [number, number]): number[] {
	const inside: number[] = [];
	for (let i = a; i <= b && i <= lines.length; i++) if (isCode(lines[i - 1])) inside.push(i);
	if (inside.length > 0) return inside.length === 1 ? inside : [inside[0], inside[inside.length - 1]];
	let i = Math.min(a, lines.length);
	while (i > 1 && !isCode(lines[i - 1])) i--;
	return [i];
}

interface FileHits {
	path: string;
	lines: number[];
}

export default function (pi: ExtensionAPI) {
	// Names already searched (listed or not) and names whose tests were listed, across the whole run.
	const reported = new Set<string>();
	const shown = new Set<string>();
	const roots = new Map<string, string | null>();

	async function git(args: string[], cwd: string): Promise<{ stdout: string; code: number }> {
		try {
			const r = await pi.exec("git", args, { cwd, timeout: GIT_TIMEOUT_MS });
			return { stdout: r.stdout, code: r.killed ? -1 : r.code };
		} catch {
			return { stdout: "", code: -1 };
		}
	}

	async function root(dir: string): Promise<string | null> {
		const cached = roots.get(dir);
		if (cached !== undefined) return cached;
		const r = await git(["rev-parse", "--show-toplevel"], dir);
		const top = r.code === 0 && r.stdout.trim() ? r.stdout.trim() : null;
		roots.set(dir, top);
		return top;
	}

	async function testHits(name: string, top: string): Promise<FileHits[] | null> {
		const r = await git(["grep", "-n", "-w", "-I", "-F", "-e", name, "--", "*.py"], top);
		// git grep exits 1 when nothing matches.
		if (r.code !== 0 && r.code !== 1) return null;
		const byFile = new Map<string, number[]>();
		for (const line of r.stdout.split("\n")) {
			const m = /^(.*?):(\d+):/.exec(line);
			if (!m || !isTestPath(m[1])) continue;
			const arr = byFile.get(m[1]) ?? [];
			arr.push(Number(m[2]));
			byFile.set(m[1], arr);
		}
		return [...byFile.entries()].map(([path, lines]) => ({ path, lines })).sort((a, b) => b.lines.length - a.lines.length);
	}

	// Names of the test functions (as Class.test_x when inside a class) that contain the matched lines.
	function testNames(top: string, hit: FileHits): string[] {
		let lines: string[];
		try {
			const abs = resolve(top, hit.path);
			if (statSync(abs).size > MAX_FILE_BYTES) return [];
			lines = readFileSync(abs, "utf8").split("\n");
		} catch {
			return [];
		}
		const names: string[] = [];
		const inStr = continuationLines(lines);
		for (const ln of hit.lines) {
			const scopes = enclosingScopes(lines, inStr, ln);
			const i = scopes.findIndex((s) => s.kind === "def" && s.name.startsWith("test"));
			if (i < 0) continue;
			const cls = scopes.slice(i + 1).find((s) => s.kind === "class");
			const label = cls ? `${cls.name}.${scopes[i].name}` : scopes[i].name;
			if (!names.includes(label)) names.push(label);
		}
		return names;
	}

	pi.on("tool_result", async (event, ctx) => {
		if ((event.toolName !== "edit" && event.toolName !== "write") || event.isError) return;
		const rawPath = typeof event.input?.path === "string" ? event.input.path.trim().replace(/^@/, "") : "";
		if (!rawPath.endsWith(".py")) return;
		try {
			const cwd = ctx.cwd;
			const given = isAbsolute(rawPath) ? rawPath : resolve(cwd, rawPath);
			// git reports the repo root with symlinks resolved, so compare against the file's real path.
			const abs = realpathSync(given);
			const top = await root(dirname(abs));
			if (!top) return;
			const rel = relative(top, abs);
			if (rel.startsWith("..") || isTestPath(rel)) return;

			const diff = await git(["diff", "-U0", "--no-color", "--no-ext-diff", "HEAD", "--", rel], top);
			if (diff.code !== 0 || !diff.stdout.trim()) return;
			const lines = readFileSync(abs, "utf8").split("\n");
			const inStr = continuationLines(lines);

			// For each changed spot: its innermost specific functions, then its class as a fallback.
			const spots: { name: string; where: string }[][] = [];
			const seen = new Set<string>();
			const probes = changedRanges(diff.stdout).flatMap((r) => probeLines(lines, r));
			for (const ln of probes) {
				const scopes = enclosingScopes(lines, inStr, ln);
				const fns = scopes.filter((s) => s.kind === "def" && !isGeneric(s.name)).slice(0, 2);
				const cls = scopes.find((s) => s.kind === "class" && !isGeneric(s.name));
				const key = [...fns.map((f) => f.name), cls?.name ?? ""].join("|");
				if (seen.has(key)) continue;
				seen.add(key);
				const opts: { name: string; where: string }[] = [];
				for (const fn of fns) {
					const owner = scopes.slice(scopes.indexOf(fn) + 1).find((s) => s.kind === "class");
					opts.push({ name: fn.name, where: `${owner ? `${owner.name}.` : ""}${fn.name}()` });
				}
				if (cls) opts.push({ name: cls.name, where: `class ${cls.name}` });
				if (opts.length > 0) spots.push(opts);
			}

			const blocks: string[] = [];
			let searches = 0;
			for (const opts of spots) {
				if (blocks.length >= MAX_NAMES_PER_EDIT || searches >= MAX_SEARCHES_PER_EDIT) break;
				for (const c of opts) {
					// A name already listed covers this spot; one that found nothing falls through to the class.
					if (reported.has(c.name)) {
						if (shown.has(c.name)) break;
						continue;
					}
					if (searches >= MAX_SEARCHES_PER_EDIT) break;
					searches++;
					reported.add(c.name);
					const hits = await testHits(c.name, top);
					if (!hits || hits.length === 0 || hits.length > MAX_FILES_FOR_NAME) continue;
					const rows = hits.slice(0, MAX_FILES_SHOWN).map((h) => {
						const names = testNames(top, h);
						const list = names.slice(0, MAX_TESTS_PER_FILE).join(", ");
						const more = names.length > MAX_TESTS_PER_FILE ? `, +${names.length - MAX_TESTS_PER_FILE} more` : "";
						const n = h.lines.length;
						return `  ${h.path} (${n} ref${n === 1 ? "" : "s"})${list ? `: ${list}${more}` : ""}`;
					});
					if (hits.length > MAX_FILES_SHOWN) rows.push(`  … and ${hits.length - MAX_FILES_SHOWN} more test files`);
					blocks.push(`Existing tests that reference ${c.where} (changed in ${rel}):\n${rows.join("\n")}`);
					shown.add(c.name);
					break;
				}
			}
			if (blocks.length === 0) return;

			const note =
				`\n\n${blocks.join("\n")}\n` +
				"Run these files' tests once your change is in place (with this repo's own test runner). A failure in them that the task doesn't call for means the change breaks behavior these tests expect; a failure that is also there without your change is not yours.";
			pi.appendEntry("acted", { what: "listed tests for changed code" });
			return { content: [...event.content, { type: "text", text: note }] };
		} catch {
			return;
		}
	});
}
