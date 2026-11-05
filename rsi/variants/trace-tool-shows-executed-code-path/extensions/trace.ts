import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "typebox";

// `trace` tool: run a Python snippet the way `python -` would from the working directory, and return its exit
// code and output plus the tree of functions from this checkout that it executed (who called whom, how often, and
// the simple return values seen). It shows which code path a reproduction actually takes instead of guessing it
// from grep results or by tracing the code by hand.

const TRACER = join(dirname(fileURLToPath(import.meta.url)), "..", "scripts", "trace_calls.py");
const DEFAULT_MAX_LINES = 120;
// Once files were edited, a trace is usually a check of the fix: keep the default tree short.
const DEFAULT_MAX_LINES_AFTER_EDIT = 40;
const DEFAULT_TIMEOUT_S = 60;
const OUTPUT_TAIL_CHARS = 3000;

const TraceParams = Type.Object({
	code: Type.String({
		description:
			"Python source to run, as you would pipe it to `python -` from the working directory (imports, any setup the project needs, then the calls that reproduce the behavior).",
	}),
	focus: Type.Optional(
		Type.String({
			description:
				"Regex matched against 'function file:line'. Lists only matching calls, the calls that led to them and up to 2 levels below them.",
		}),
	),
	max_lines: Type.Optional(Type.Number({ description: `Most call-tree lines to return (default ${DEFAULT_MAX_LINES}).` })),
	timeout: Type.Optional(Type.Number({ description: `Seconds before the run is stopped (default ${DEFAULT_TIMEOUT_S}).` })),
});

function tail(text: string, max: number): string {
	if (text.length <= max) return text;
	return `[... ${text.length - max} chars cut ...]\n${text.slice(text.length - max)}`;
}

function gitRoot(cwd: string): Promise<string> {
	return new Promise((resolve) => {
		const child = spawn("git", ["rev-parse", "--show-toplevel"], { cwd, stdio: ["ignore", "pipe", "ignore"] });
		let out = "";
		child.stdout.on("data", (d) => (out += d.toString()));
		child.on("error", () => resolve(cwd));
		child.on("close", (code) => resolve(code === 0 && out.trim() ? out.trim() : cwd));
	});
}

export default function (pi: ExtensionAPI) {
	let edited = false;
	pi.on("tool_call", async (event) => {
		if (event.toolName === "edit" || event.toolName === "write") edited = true;
	});

	pi.registerTool({
		name: "trace",
		label: "trace",
		description:
			"Run a Python snippet and see which functions of this repository it actually executed. Returns the exit code, the tail of stdout/stderr, and a call tree of every function defined in files of this checkout that ran (indent = called by the line above, xN = call count, -> = return values seen for str/number/bool/None/simple containers; other objects show as <TypeName>). Code run at import time is left out. " +
			"Use it when you need to know where behavior comes from: which function produces a wrong value or message, which branch or override a reproduction goes through, whether there is a second code path to the same result, or whether your fix is on the path the issue's example takes. Start without focus to see the overall path, then pass focus (a function or file regex) to see the calls around one function. " +
			"To only see a snippet's output, use bash; trace adds the call tree (after your first edit its default length drops to " + DEFAULT_MAX_LINES_AFTER_EDIT + " lines). Not for running test suites (use bash) and not for non-Python code. The snippet runs for real: anything it writes or deletes happens.",
		parameters: TraceParams,

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const cwd = ctx.cwd;
			const root = await gitRoot(cwd);
			const defaultLines = edited ? DEFAULT_MAX_LINES_AFTER_EDIT : DEFAULT_MAX_LINES;
			const maxLines = Math.max(20, Math.min(400, Math.floor(params.max_lines ?? defaultLines)));
			const timeoutS = Math.max(5, Math.min(600, params.timeout ?? DEFAULT_TIMEOUT_S));
			const dir = mkdtempSync(join(tmpdir(), "pi-trace-"));
			const snippet = join(dir, "snippet.py");
			const report = join(dir, "report.txt");
			writeFileSync(snippet, params.code, "utf8");
			pi.appendEntry("acted", {
				what: `trace tool run ${edited ? "after" : "before"} the first edit${params.focus ? ", with focus" : ""}`,
			});

			const q = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
			const args = [TRACER, root, snippet, report, String(maxLines), String(timeoutS), params.focus ?? ""].map(q).join(" ");
			const command = `if command -v python >/dev/null 2>&1; then exec python ${args}; else exec python3 ${args}; fi`;

			const result = await new Promise<{ code: number | null; stdout: string; stderr: string; killed: boolean }>(
				(resolve) => {
					const child = spawn("bash", ["-c", command], { cwd, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
					let stdout = "";
					let stderr = "";
					let killed = false;
					const limit = 2_000_000;
					child.stdout.on("data", (d) => {
						if (stdout.length < limit) stdout += d.toString();
					});
					child.stderr.on("data", (d) => {
						if (stderr.length < limit) stderr += d.toString();
					});
					const kill = () => {
						killed = true;
						child.kill("SIGKILL");
					};
					const timer = setTimeout(kill, (timeoutS + 20) * 1000);
					signal?.addEventListener("abort", kill, { once: true });
					child.on("error", (err) => {
						clearTimeout(timer);
						resolve({ code: null, stdout, stderr: stderr + String(err), killed });
					});
					child.on("close", (code) => {
						clearTimeout(timer);
						resolve({ code, stdout, stderr, killed });
					});
				},
			);

			let tree = "";
			try {
				tree = readFileSync(report, "utf8");
			} catch {
				tree = result.killed
					? `The run was killed after ${timeoutS + 20} s before a call tree could be written.`
					: `No call tree was written (exit code ${result.code}); see stderr above.`;
			}
			try {
				rmSync(dir, { recursive: true, force: true });
			} catch {}

			const parts: string[] = [];
			if (result.stdout.trim()) parts.push(`stdout:\n${tail(result.stdout.trimEnd(), OUTPUT_TAIL_CHARS)}`);
			if (result.stderr.trim()) parts.push(`stderr:\n${tail(result.stderr.trimEnd(), OUTPUT_TAIL_CHARS)}`);
			parts.push(tree.trimEnd());
			return {
				content: [{ type: "text", text: parts.join("\n\n") }],
				details: { exitCode: result.code, focus: params.focus ?? null },
			};
		},
	});
}
