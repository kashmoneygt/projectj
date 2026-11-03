import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const LOG_DIR = "/tmp/pi-run-tests";
const DEFAULT_TIMEOUT_S = 600;
const MAX_TIMEOUT_S = 1800;
const MAX_CAPTURE = 30 * 1024 * 1024;
const MAX_LISTED = 25;
const MAX_LINE = 300;

interface RunResult {
	code: number | null;
	output: string;
	timedOut: boolean;
	aborted: boolean;
	seconds: number;
}

interface Failure {
	id: string;
	msg: string;
}

interface Report {
	summary: string[];
	failures: Failure[];
	loadErrors: Failure[];
}

function clip(s: string, n = MAX_LINE): string {
	const t = s.trim();
	return t.length > n ? `${t.slice(0, n)}…` : t;
}

function runShell(
	command: string,
	cwd: string,
	env: NodeJS.ProcessEnv,
	timeoutS: number,
	signal: AbortSignal | undefined,
): Promise<RunResult> {
	return new Promise((resolve) => {
		const start = Date.now();
		const chunks: Buffer[] = [];
		let size = 0;
		let timedOut = false;
		let aborted = false;
		let settled = false;
		const child = spawn("bash", ["-c", command], { cwd, env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
		const kill = () => {
			try {
				if (child.pid) process.kill(-child.pid, "SIGKILL");
			} catch {
				try {
					child.kill("SIGKILL");
				} catch {}
			}
		};
		const onData = (b: Buffer) => {
			if (size < MAX_CAPTURE) {
				chunks.push(b);
				size += b.length;
			}
		};
		child.stdout?.on("data", onData);
		child.stderr?.on("data", onData);
		const timer = setTimeout(() => {
			timedOut = true;
			kill();
		}, timeoutS * 1000);
		const onAbort = () => {
			aborted = true;
			kill();
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		const finish = (code: number | null) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
			resolve({
				code,
				output: Buffer.concat(chunks).toString("utf8"),
				timedOut,
				aborted,
				seconds: (Date.now() - start) / 1000,
			});
		};
		child.on("error", (e) => {
			chunks.push(Buffer.from(`\n[could not start bash: ${e.message}]\n`));
			finish(127);
		});
		child.on("close", (code) => finish(code));
	});
}

const SUMMARY_RES: RegExp[] = [
	/^Ran \d+ tests? in /,
	/^OK( \(.*\))?$/,
	/^FAILED \(.*\)$/,
	/^=+ .*\b(passed|failed|errors?|skipped|deselected|xfailed|xpassed|no tests ran)\b.*=+$/,
	/^\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b.*\bin [\d.]+ ?s/,
	/\btests finished: /,
	/^no tests ran\b/,
];

const SEPARATOR = /^(={10,}|-{10,})$/;

function parseOutput(out: string): Report {
	const lines = out.split(/\r?\n/);
	const summary: string[] = [];
	const shortSummary = new Map<string, string>();
	const unittest = new Map<string, string>();
	const sections = new Map<string, string>();
	const loadErrors = new Map<string, string>();

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i].replace(/\s+$/, "");
		if (SUMMARY_RES.some((re) => re.test(line))) summary.push(line);

		let m = /^(FAIL|ERROR): (\S+) \(([^)]*)\)/.exec(line);
		if (m) {
			// unittest-style block: the error is the last line before the next separator after the traceback
			let j = i + 1;
			if (j < lines.length && SEPARATOR.test(lines[j].trim())) j++;
			let last = "";
			let importMsg = "";
			for (; j < lines.length && !SEPARATOR.test(lines[j].trim()); j++) {
				const t = lines[j].trim();
				if (t) last = t;
				if (/^(ImportError|ModuleNotFoundError):/.test(t) || /Failed to import test module/.test(t)) importMsg = t;
			}
			if (/unittest\.loader\.(_FailedTest|ModuleImportFailure)/.test(m[3])) {
				loadErrors.set(m[2], importMsg || last);
			} else {
				const id = `${m[1]}: ${m[2]} (${m[3]})`;
				if (!unittest.has(id)) unittest.set(id, last);
			}
			continue;
		}

		m = /^(FAILED|ERROR) (\S+)(?: - (.*))?$/.exec(line);
		if (m && !/^\(/.test(m[2])) {
			if (!shortSummary.has(m[2])) shortSummary.set(m[2], m[3] ?? "");
			continue;
		}

		m = /^_{3,} (.+?) _{3,}$/.exec(line);
		if (m) {
			let eLine = "";
			let last = "";
			for (let j = i + 1; j < lines.length; j++) {
				const t = lines[j].replace(/\s+$/, "");
				if (/^_{3,} .+ _{3,}$/.test(t) || /^={5,}/.test(t) || /^_{20,}$/.test(t)) break;
				if (!eLine && /^E\s+\S/.test(t)) eLine = t.replace(/^E\s+/, "");
				if (t.trim()) last = t.trim();
			}
			if (!sections.has(m[1])) sections.set(m[1], eLine || last);
		}
	}

	// pytest cuts the short-summary message; the matching section's `E` line is complete
	for (const [id, msg] of shortSummary) {
		const key = id.split("::").slice(1).join(".");
		const full = key ? sections.get(key) : undefined;
		if (full && (!msg || msg.endsWith("..."))) shortSummary.set(id, full);
	}
	const chosen = shortSummary.size > 0 ? shortSummary : unittest.size > 0 ? unittest : sections;
	const failures: Failure[] = [];
	for (const [id, msg] of chosen) failures.push({ id, msg });
	if (chosen !== unittest) for (const [id, msg] of unittest) failures.push({ id, msg });
	return {
		summary: summary.slice(-4),
		failures,
		loadErrors: [...loadErrors].map(([id, msg]) => ({ id, msg })),
	};
}

function gitRoot(cwd: string): string | undefined {
	const r = spawnSync("git", ["rev-parse", "--show-toplevel"], { cwd, encoding: "utf8" });
	if (r.status !== 0) return undefined;
	const root = r.stdout.trim();
	return root || undefined;
}

let pythonBin: string | undefined;
function python(): string {
	if (!pythonBin) {
		const r = spawnSync("python", ["-c", "pass"], { encoding: "utf8" });
		pythonBin = r.error || r.status !== 0 ? "python3" : "python";
	}
	return pythonBin;
}

function findRunners(root: string): string[] {
	const found: string[] = [];
	const py = spawnSync(python(), ["-c", "import pytest"], { cwd: root, encoding: "utf8" });
	found.push(py.status === 0 ? `\`${python()} -m pytest\` (pytest is installed)` : "pytest is NOT installed in this environment");
	const files = [
		"runtests.py",
		"tests/runtests.py",
		"test/runtests.py",
		"run_tests.py",
		"tests/run_tests.py",
		"bin/test",
		"bin/runtests",
		"manage.py",
		"pytest.ini",
		"tox.ini",
		"setup.cfg",
		"Makefile",
	];
	for (const f of files) {
		const p = join(root, f);
		if (!existsSync(p)) continue;
		if (f === "setup.cfg" || f === "tox.ini") {
			let txt = "";
			try {
				txt = readFileSync(p, "utf8");
			} catch {}
			const hits = txt.split("\n").filter((l) => /^\s*\[(tool:pytest|pytest|test|testenv)\]/.test(l) || /^\s*commands\s*=/.test(l));
			if (hits.length) found.push(`${f}: ${hits.slice(0, 3).map((h) => h.trim()).join(" ")}`);
			continue;
		}
		if (f === "Makefile") {
			let txt = "";
			try {
				txt = readFileSync(p, "utf8");
			} catch {}
			if (/^test[\w-]*:/m.test(txt)) found.push("Makefile has a test target");
			continue;
		}
		found.push(`${f} (try \`python ${f} --help\` for its options)`);
	}
	return found;
}

let pristine: { root: string; dir: string } | undefined;

// A copy of the repository as of HEAD with no local changes, built without touching the working tree.
// Ignored files (compiled extensions, build output) are kept so the code imports the same way.
function ensurePristine(root: string): { dir?: string; error?: string } {
	if (pristine && pristine.root === root && existsSync(pristine.dir)) return { dir: pristine.dir };
	try {
		mkdirSync(LOG_DIR, { recursive: true });
		const dir = mkdtempSync("/tmp/pi-original-");
		const q = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
		const script = [
			"set -e",
			`tar -C ${q(root)} --exclude=./.git -cf - . | tar -C ${q(dir)} -xf -`,
			`git -C ${q(root)} archive HEAD | tar -C ${q(dir)} -xf -`,
			`cd ${q(root)} && git ls-files --others --exclude-standard -z | (cd ${q(dir)} && xargs -0 -r rm -f --)`,
		].join("\n");
		const r = spawnSync("bash", ["-c", script], { encoding: "utf8", timeout: 600_000 });
		if (r.status !== 0) {
			rmSync(dir, { recursive: true, force: true });
			return { error: clip(`${r.stderr || r.stdout || "copy failed"}`, 400) };
		}
		pristine = { root, dir };
		return { dir };
	} catch (e: any) {
		return { error: clip(String(e?.message ?? e), 400) };
	}
}

const IMPORT_ROOTS_PY = `
import sys, os, importlib.util
root = sys.argv[1]
out = set(p for p in sys.path if p and (p == root or p.startswith(root + os.sep)))
for base in (root, os.path.join(root, "src"), os.path.join(root, "lib")):
    if not os.path.isdir(base):
        continue
    for n in os.listdir(base):
        if n.isidentifier() and os.path.isfile(os.path.join(base, n, "__init__.py")):
            try:
                s = importlib.util.find_spec(n)
            except Exception:
                continue
            if s and s.origin and s.origin.startswith(root + os.sep):
                out.add(os.path.dirname(os.path.dirname(s.origin)))
print("\\n".join(sorted(out)))
`;

// Where the installed environment imports the repository's code from, mapped into the copy, so the
// copy's code is imported the same way the working tree's code is.
function pythonPathFor(root: string, dir: string): string | undefined {
	const r = spawnSync(python(), ["-c", IMPORT_ROOTS_PY, root], {
		cwd: "/",
		encoding: "utf8",
		env: { ...process.env, PYTHONPATH: "" },
		timeout: 60_000,
	});
	const entries: string[] = [];
	if (r.status === 0) {
		for (const p of r.stdout.split("\n").map((x) => x.trim())) {
			if (p && (p === root || p.startsWith(`${root}/`))) entries.push(dir + p.slice(root.length));
		}
	}
	const prev = process.env.PYTHONPATH;
	const all = [...entries, ...(prev ? [prev.split(root).join(dir)] : [])];
	return all.length ? all.join(":") : undefined;
}

// A trailing `| tail ...` or `| head ...` only hides the summary this tool extracts anyway.
function stripTrailingFilter(cmd: string): string {
	return cmd.replace(/\s*\|\s*(tail|head)(\s+-n)?(\s+-?\+?\d+)?\s*$/, "").trim();
}

function lastLines(out: string, n: number): string {
	const lines = out.replace(/\s+$/, "").split(/\r?\n/);
	const tail = lines.slice(-n).map((l) => (l.length > 400 ? `${l.slice(0, 400)}…` : l));
	return tail.join("\n");
}

function statusLine(r: RunResult): string {
	if (r.aborted) return `aborted after ${r.seconds.toFixed(1)} s`;
	if (r.timedOut) return `stopped after ${r.seconds.toFixed(0)} s (timeout)`;
	return `exit code ${r.code} · ${r.seconds.toFixed(1)} s`;
}

function listFailures(title: string, items: Failure[], withMsg = true): string[] {
	if (!items.length) return [];
	const out = [`${title} (${items.length}):`];
	for (const f of items.slice(0, MAX_LISTED)) {
		out.push(`  - ${clip(f.id, 200)}`);
		if (withMsg && f.msg) out.push(`      ${clip(f.msg)}`);
	}
	if (items.length > MAX_LISTED) out.push(`  … and ${items.length - MAX_LISTED} more (see the log)`);
	return out;
}

let runCount = 0;

export default function (pi: ExtensionAPI) {
	const cleanup = () => {
		if (!pristine) return;
		try {
			rmSync(pristine.dir, { recursive: true, force: true });
		} catch {}
		pristine = undefined;
	};
	pi.on("session_shutdown", cleanup);
	process.once("exit", cleanup);

	pi.registerTool(
		defineTool({
			name: "run_tests",
			label: "Run tests",
			description:
				"Run a test command and get a compact report instead of raw output: the runner's result summary, every failing or erroring test with its error message, test labels/modules that could not be loaded, and the last lines of output. The full output is saved to a log file you can read. " +
				"Pass the plain command you would type in bash, for example `python -m pytest tests/test_x.py -q` or the repository's own test runner script with its arguments; do not pipe it through tail or grep. " +
				"If the command cannot start a test runner (for example pytest is not installed), the report lists the test runners found in the repository. " +
				"Set compare_with_original to also run the same command on an untouched copy of the repository at its original commit (your changes left out; your working tree is not modified) and see which failures your changes introduced, which were already failing, and which your changes fixed.",
			promptSnippet:
				"run_tests: Run a test command; get its summary and each failing test with its error, optionally compared against the original code",
			promptGuidelines: [
				"Run test suites with run_tests rather than bash, passing the plain test command. Keep using bash for other commands, such as reproduction scripts.",
				"When tests fail after your change and you need to know whether your change caused it, call run_tests with compare_with_original instead of stashing your changes.",
			],
			parameters: Type.Object({
				command: Type.String({
					description: "Shell command that runs the tests, run from the current working directory",
				}),
				compare_with_original: Type.Optional(
					Type.Boolean({
						description:
							"Also run the command on a copy of the repository without your changes and report which failures are new, pre-existing, or fixed (default false)",
					}),
				),
				timeout: Type.Optional(
					Type.Number({ description: `Timeout in seconds for each run (default ${DEFAULT_TIMEOUT_S}, max ${MAX_TIMEOUT_S})` }),
				),
			}),

			async execute(_toolCallId, params, signal, _onUpdate, ctx) {
				const cwd = ctx.cwd;
				const command = stripTrailingFilter(String(params.command ?? ""));
				if (!command) throw new Error("command is empty");
				const timeoutS = Math.max(10, Math.min(MAX_TIMEOUT_S, Number(params.timeout) || DEFAULT_TIMEOUT_S));
				const compare = params.compare_with_original === true || String(params.compare_with_original).toLowerCase() === "true";
				runCount++;
				pi.appendEntry("acted", { what: compare ? "run_tests with comparison" : "run_tests" });

				mkdirSync(LOG_DIR, { recursive: true });
				const logPath = join(LOG_DIR, `run-${runCount}.log`);
				const res = await runShell(command, cwd, process.env, timeoutS, signal);
				try {
					writeFileSync(logPath, `$ ${command}\n${res.output}`);
				} catch {}
				const rep = parseOutput(res.output);
				const nLines = res.output ? res.output.split("\n").length : 0;

				const out: string[] = [`$ ${command}`, `${statusLine(res)} · full output (${nLines} lines): ${logPath}`];
				out.push(rep.summary.length ? `summary: ${rep.summary.map((s) => clip(s, 200)).join(" | ")}` : "summary: no test-runner summary recognized in the output");
				out.push(...listFailures("failing tests", rep.failures));
				if (rep.loadErrors.length) {
					out.push(...listFailures("could not load (a wrong test label/path, or an import error)", rep.loadErrors));
				}

				const noRunner =
					rep.summary.length === 0 &&
					rep.failures.length === 0 &&
					/No module named '?pytest'?|pytest: command not found|No module named '?nose'?|: command not found|can't open file|No such file or directory/.test(
						res.output,
					);
				if (noRunner) {
					const root = gitRoot(cwd) ?? cwd;
					out.push(`The command did not start a test runner. Found in ${root}:`);
					for (const r of findRunners(root)) out.push(`  - ${r}`);
				}

				const tailN = rep.summary.length && (rep.failures.length || res.code === 0) ? 12 : 40;
				out.push(`--- last ${tailN} lines of output ---`, lastLines(res.output, tailN) || "(no output)");

				if (compare && !res.aborted) {
					out.push("", "=== same command on the original code (your changes left out) ===");
					const root = gitRoot(cwd);
					const dirty = root
						? spawnSync("git", ["status", "--porcelain", "--untracked-files=normal"], { cwd: root, encoding: "utf8" }).stdout.trim()
						: "";
					if (!root) {
						out.push("comparison unavailable: not inside a git repository");
					} else if (!dirty) {
						out.push("comparison skipped: the working tree has no changes, so the original code is what you just ran");
					} else {
						const p = ensurePristine(root);
						if (!p.dir) {
							out.push(`comparison unavailable: could not copy the repository (${p.error})`);
						} else {
							const dir = p.dir;
							const relCwd = relative(root, cwd);
							const cwd2 = relCwd.startsWith("..") ? cwd : join(dir, relCwd);
							const cmd2 = command.split(root).join(dir);
							const pyPath = pythonPathFor(root, dir);
							const env2: NodeJS.ProcessEnv = { ...process.env };
							if (pyPath) env2.PYTHONPATH = pyPath;
							const res2 = await runShell(cmd2, cwd2, env2, timeoutS, signal);
							const out2 = res2.output.split(dir).join(root);
							const log2 = join(LOG_DIR, `run-${runCount}-original.log`);
							try {
								writeFileSync(log2, `$ ${cmd2}\n(in ${cwd2})\n${out2}`);
							} catch {}
							const rep2 = parseOutput(out2);
							out.push(`${statusLine(res2)} · full output: ${log2}`);
							out.push(rep2.summary.length ? `summary: ${rep2.summary.map((s) => clip(s, 200)).join(" | ")}` : "summary: no test-runner summary recognized in the output");
							const before = new Map(rep2.failures.map((f) => [f.id, f]));
							const after = new Map(rep.failures.map((f) => [f.id, f]));
							const introduced = rep.failures.filter((f) => !before.has(f.id));
							const persisting = rep.failures.filter((f) => before.has(f.id));
							const fixed = rep2.failures.filter((f) => !after.has(f.id));
							if (!rep.failures.length && !rep2.failures.length) {
								out.push(
									res.code === res2.code
										? "no failing tests recognized in either run; exit codes match"
										: `no failing tests recognized in either run, but exit codes differ (with your changes ${res.code}, original ${res2.code}); compare the logs`,
								);
							} else {
								out.push(...listFailures("FAILING ONLY WITH YOUR CHANGES", introduced));
								out.push(...listFailures("failing before and after your changes (pre-existing)", persisting, false));
								out.push(...listFailures("failing on the original code, passing with your changes", fixed, false));
								if (!introduced.length) out.push("none of the failures are new with your changes");
							}
							if (rep2.loadErrors.length) out.push(...listFailures("could not load on the original code", rep2.loadErrors));
							if (!rep2.summary.length && !rep2.failures.length) {
								out.push(`--- last 15 lines (original code) ---`, lastLines(out2, 15) || "(no output)");
							}
						}
					}
				}

				return {
					content: [{ type: "text", text: out.join("\n") }],
					details: { command, exitCode: res.code, failures: rep.failures.length, compare },
				};
			},
		}),
	);
}
