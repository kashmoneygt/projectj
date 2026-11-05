import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Observation only: SYSTEM.md's evidence rules matter most once a test run after an edit reports failures.
// Record that moment (once per run) so the runs where the rules could steer a decision can be counted.
// Nothing here changes a tool call, a result or the context.

const TEST_COMMAND =
	/(\bpytest\b|py\.test|runtests\.py|\btox\b|\bnox\b|-m\s+unittest|manage\.py\s+test|bin\/test\b|\btest\(\)|\.test\(|\b(npm|yarn|pnpm)\s+(run\s+)?test|\bgo\s+test\b|\bcargo\s+test\b|\bmvn\b|\bgradlew?\b|\bmake\s+(check|test)\b|\bctest\b|\bjest\b|\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b)/;
const BASH_EDIT = /(\bsed\s+-i|\bpatch\b|\bgit\s+(apply|checkout|restore|stash\s+pop|stash\s+apply)\b|\bperl\s+-p?i)/;
const FAILURES = /\b[1-9]\d*\s+(failed|errors?|exceptions?)\b|FAILED\s+\((failures|errors)=|^FAILED\b|^ERROR:\s/m;

function textOf(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.filter((b: any) => b && b.type === "text" && typeof b.text === "string")
		.map((b: any) => b.text)
		.join("\n");
}

export default function (pi: ExtensionAPI) {
	let edited = false;
	let recorded = false;

	pi.on("tool_call", async (event) => {
		if (event.toolName === "edit" || event.toolName === "write") {
			edited = true;
		} else if (event.toolName === "bash") {
			const command = String((event.input as { command?: unknown }).command ?? "");
			if (!TEST_COMMAND.test(command) && BASH_EDIT.test(command)) edited = true;
		}
	});

	pi.on("tool_result", async (event) => {
		if (recorded || !edited || event.toolName !== "bash") return;
		try {
			const command = String((event.input as { command?: unknown }).command ?? "");
			if (!TEST_COMMAND.test(command)) return;
			if (!FAILURES.test(textOf(event.content))) return;
			recorded = true;
			pi.appendEntry("acted", { what: "tests failed after an edit" });
		} catch {
			// observation only
		}
	});
}
