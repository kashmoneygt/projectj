import { existsSync, readdirSync, realpathSync } from "node:fs";
import { lookup } from "node:dns/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Scripts some projects use instead of a bare pytest run, relative to the repo root.
const RUNNER_SCRIPTS = ["tests/runtests.py", "runtests.py", "run_tests.py", "tests/run_tests.py", "bin/test", "bin/runtests"];

// Files that mark a Python project; other projects get no Python facts.
const PY_MARKERS = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "tox.ini", "pytest.ini", "Pipfile"];

// Runs with a neutral working directory, so imports resolve the way they do for a script outside the repo.
const PROBE = String.raw`
import json, os, re, sys
root = os.path.realpath(sys.argv[1])
sys.path = [x for x in sys.path if x not in ("", ".")]
try:
    import importlib.util as ilu
    def find(n):
        s = ilu.find_spec(n)
        if s is None:
            return None, False
        o = s.origin if s.origin and s.origin not in ("namespace", "built-in") else None
        if o is None and s.submodule_search_locations:
            o = list(s.submodule_search_locations)[0]
        return o, True
except ImportError:
    import imp
    def find(n):
        try:
            f, p, _ = imp.find_module(n)
        except ImportError:
            return None, False
        if f:
            f.close()
        return p, True
out = {"exe": sys.executable, "ver": "%d.%d.%d" % tuple(sys.version_info[:3]), "pkgs": []}
try:
    from importlib.metadata import version
    out["pytest"] = version("pytest")
except Exception:
    try:
        out["pytest"] = "yes" if find("pytest")[1] else None
    except Exception:
        out["pytest"] = None
SKIP = set(["test", "tests", "testing", "doc", "docs", "example", "examples",
            "script", "scripts", "tools", "build", "dist", "ci", "bin", "venv", "env", "node_modules"])
cands = []
for base in ("", "src", "lib"):
    d = os.path.join(root, base)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        continue
    for n in names:
        p = os.path.join(d, n)
        if n.startswith((".", "_")) or n.lower() in SKIP or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n):
            continue
        if os.path.isfile(os.path.join(p, "__init__.py")):
            count = 0
            for _, _, files in os.walk(p):
                count += sum(f.endswith(".py") for f in files)
                if count > 2000:
                    break
            same = n.lower() in os.path.basename(root).lower()
            cands.append((not same, -count, n, d))
cands.sort()
for _, _, n, d in cands[:1]:
    try:
        origin = find(n)[0]
    except Exception:
        origin = None
    if origin and os.path.basename(origin) in ("__init__.py", "__init__.pyc"):
        origin = os.path.dirname(origin)
    real = os.path.realpath(origin) if origin else None
    out["pkgs"].append({"name": n, "parent": d, "origin": origin,
                        "inRepo": bool(real) and (real + os.sep).startswith(os.path.join(root, ""))})
print(json.dumps(out))
`;

// undefined when the lookup didn't finish in time.
async function online(): Promise<boolean | undefined> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const timeout = new Promise<undefined>((res) => {
		timer = setTimeout(() => res(undefined), 2000);
	});
	const result = await Promise.race([
		lookup("pypi.org").then(
			() => true,
			() => false,
		),
		timeout,
	]);
	clearTimeout(timer);
	return result;
}

function looksLikePython(root: string): boolean {
	if (PY_MARKERS.some((p) => existsSync(join(root, p)))) return true;
	for (const base of ["", "src", "lib"]) {
		let names: string[] = [];
		try {
			names = readdirSync(join(root, base));
		} catch {
			continue;
		}
		if (base === "" && names.some((n) => n.endsWith(".py"))) return true;
		if (names.some((n) => !n.startsWith(".") && existsSync(join(root, base, n, "__init__.py")))) return true;
	}
	return false;
}

type ProbeResult = {
	exe: string;
	ver: string;
	pytest: string | null;
	pkgs: { name: string; parent: string; origin: string | null; inRepo: boolean }[];
};

export default function (pi: ExtensionAPI) {
	let facts: { text: string; kinds: string[] } | undefined;

	async function has(cmd: string, cwd: string): Promise<string | undefined> {
		try {
			const r = await pi.exec("sh", ["-c", `command -v ${cmd}`], { cwd, timeout: 3000 });
			const path = r.stdout.trim();
			return r.code === 0 && path ? path : undefined;
		} catch {
			return undefined;
		}
	}

	// Whether PYTHONPATH=<parent> makes `import <name>` load the repo's copy.
	async function resolvesInto(python: string, name: string, parent: string, root: string): Promise<boolean> {
		try {
			const code =
				"import os, sys\ntry:\n    import importlib.util as u; s = u.find_spec(sys.argv[1]); o = s.origin if s else ''\nexcept ImportError:\n    import imp; o = imp.find_module(sys.argv[1])[1]\nprint(os.path.realpath(o) if o else '')";
			const r = await pi.exec("env", [`PYTHONPATH=${parent}`, python, "-c", code, name], { cwd: tmpdir(), timeout: 5000 });
			const origin = r.stdout.trim();
			return r.code === 0 && origin.startsWith(join(realpathSync(root), "/"));
		} catch {
			return false;
		}
	}

	async function collect(cwd: string): Promise<{ text: string; kinds: string[] }> {
		const lines: string[] = [];
		const kinds: string[] = [];
		let root = cwd;
		try {
			const r = await pi.exec("git", ["rev-parse", "--show-toplevel"], { cwd, timeout: 3000 });
			if (r.code === 0 && r.stdout.trim()) root = r.stdout.trim();
		} catch {}

		const shellWorks = !!(await has("sh", cwd));
		const pythonProject = shellWorks && looksLikePython(root);
		const python = !pythonProject
			? undefined
			: (await has("python", cwd))
				? "python"
				: (await has("python3", cwd))
					? "python3"
					: undefined;
		const runners = RUNNER_SCRIPTS.filter((p) => existsSync(join(root, p)));
		let probe: ProbeResult | undefined;
		if (python) {
			try {
				const r = await pi.exec(python, ["-c", PROBE, root], { cwd: tmpdir(), timeout: 8000 });
				if (r.code === 0) probe = JSON.parse(r.stdout.trim().split("\n").pop() ?? "");
			} catch {}
		}

		if (probe) {
			let line = `- \`${python}\` is ${probe.exe} (Python ${probe.ver})`;
			if (probe.pytest) {
				line += probe.pytest === "yes" ? " with pytest." : ` with pytest ${probe.pytest}.`;
			} else {
				line += `. pytest is NOT installed for it, so \`${python} -m pytest\` fails; don't try to install it.`;
				kinds.push("no-pytest");
			}
			if (python === "python3") {
				line += " There is no `python` command; use `python3`.";
				kinds.push("python3-only");
			}
			lines.push(line);
			kinds.push("python");
		}
		if (runners.length > 0) {
			const list = runners.map((r) => `\`${r}\``).join(", ");
			lines.push(
				`- This project has its own test runner: ${list}. Run the relevant tests with it (see its --help for how to select tests)${probe && !probe.pytest ? "" : " when plain pytest doesn't work"}.`,
			);
			kinds.push("runner");
		}
		const pkg = probe?.pkgs[0];
		if (pkg && python) {
			const parent = pkg.parent.replace(/\/$/, "");
			if (pkg.inRepo) {
				lines.push(
					`- \`import ${pkg.name}\` loads this checkout (${pkg.origin}) from any directory, so scripts anywhere exercise your edits.`,
				);
				kinds.push("checkout-import");
			} else if (pkg.origin) {
				kinds.push("installed-copy");
				if (await resolvesInto(python, pkg.name, parent, root)) {
					lines.push(
						`- \`import ${pkg.name}\` from a script outside ${parent} loads an installed copy at ${pkg.origin}, NOT this checkout. Prefix python commands with \`PYTHONPATH=${parent}\` so they import the code you edit; never edit the installed copy.`,
					);
				} else {
					lines.push(
						`- \`import ${pkg.name}\` loads an installed copy at ${pkg.origin}, NOT this checkout, and that copy comes before PYTHONPATH. Only \`python -c\`, \`python -m\` or \`python -\` started in ${parent}, or scripts that start with \`import sys; sys.path.insert(0, "${parent}")\`, see your edits; anything else (a script file run by path, a test runner outside ${parent}) tests the installed copy. Never edit the installed copy.`,
					);
				}
			} else {
				lines.push(
					`- \`import ${pkg.name}\` only works with ${parent} on the path (it isn't installed): run python from there or with \`PYTHONPATH=${parent}\`.`,
				);
				kinds.push("not-installed");
			}
		}
		if ((await online()) === false) {
			lines.push(
				"- There is no network access: package installs (pip, npm) and URL fetches fail. Work with what is installed and what is in the repo.",
			);
			kinds.push("no-network");
		}
		if (shellWorks && !(await has("rg", cwd))) {
			lines.push("- `rg` is not installed; search with `grep -rn` (filter with --include) and `find`.");
			kinds.push("no-rg");
		}
		if (lines.length === 0) return { text: "", kinds };
		return {
			text: [
				"Facts about this machine, checked when the session started. Rely on them instead of re-checking:",
				...lines,
			].join("\n"),
			kinds,
		};
	}

	pi.on("before_agent_start", async (event, ctx) => {
		if (!facts) {
			try {
				facts = await collect(ctx.cwd);
			} catch {
				facts = { text: "", kinds: [] };
			}
			if (facts.kinds.length > 0) pi.appendEntry("acted", { what: `environment facts: ${facts.kinds.join(", ")}` });
		}
		if (facts.text) event.systemPromptOptions.sections.environment = facts.text;
		// pi's default rules suggest rg; point them at grep when rg isn't there.
		if (facts.kinds.includes("no-rg")) {
			const rules = /<rules>\n([\s\S]*?)\n<\/rules>/.exec(event.systemPrompt)?.[1];
			if (rules?.includes("ls, rg, find")) {
				event.systemPromptOptions.sections.rules = rules.replace("ls, rg, find", "ls, grep, find");
			}
		}
	});
}
