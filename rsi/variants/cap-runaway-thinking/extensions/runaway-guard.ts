import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// settings.json caps each response's reasoning at thinkingBudgets.medium; keep this in sync with it.
const THINKING_BUDGET = 8192;
const MAX_NUDGES_PER_RUN = 4;
const MAX_NUDGES_IN_A_ROW = 2;
// How much of a looping or length-cut block stays in the context.
const KEEP_CHARS = 2000;

type Reason = "length" | "empty" | "markup" | "capped";

interface LastResponse {
	stopReason: string;
	hasToolCall: boolean;
	hasText: boolean;
	textHasCallMarkup: boolean;
	capped: boolean;
}

const NUDGES: Record<Reason, string> = {
	length:
		"Your last response was cut off at the response length limit before it made a tool call, so nothing was run and the task is not finished. Continue from where you are: make your next tool call now, with only brief reasoning before it.",
	empty:
		"Your last response contained only reasoning: no tool call and no answer, so nothing was run and the task is not finished. A tool call written inside your reasoning is not executed. Continue from where you are: make your next tool call now, with only brief reasoning before it.",
	markup:
		"Your last response wrote a tool call as plain text, so it was not executed. Make that call again as a real tool call.",
	capped:
		"Your reasoning in the last response hit its length limit and was cut off, and the response made no tool call. If the work is not finished, continue now with your next tool call, reasoning briefly. If it is finished, reply with a short summary and no tool call.",
};

const ACTED: Record<Reason, string> = {
	length: "a response cut at the length limit",
	empty: "a reasoning-only response",
	markup: "a tool call written as text",
	capped: "a capped response with no tool call",
};

// A block that has fallen into a loop: a long run of comma-separated numbers, or a tail made mostly of
// the same few phrases over and over.
function looksDegenerate(s: string): boolean {
	if (s.length < 3000) return false;
	if (/(?:\d+\s*,\s*){80,}/.test(s.slice(-20000))) return true;
	const chunks = s
		.slice(-6000)
		.split(/[\n.;,]+/)
		.map((c) => c.trim())
		.filter((c) => c.length >= 8);
	if (chunks.length < 20) return false;
	const counts = new Map<string, number>();
	for (const c of chunks) counts.set(c, (counts.get(c) ?? 0) + 1);
	let repeated = 0;
	for (const n of counts.values()) if (n >= 4) repeated += n;
	return repeated / chunks.length >= 0.6;
}

function shorten(s: string, why: string): string {
	return `${s.slice(0, KEEP_CHARS)}\n\n[... ${s.length - KEEP_CHARS} more characters removed: ${why} ...]`;
}

export default function (pi: ExtensionAPI) {
	let last: LastResponse | undefined;
	let nudges = 0;
	let inARow = 0;

	pi.on("message_end", async (event) => {
		const msg = event.message as any;
		if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) return;
		const blocks = msg.content as any[];
		const hasToolCall = blocks.some((b) => b?.type === "toolCall");
		const text = blocks
			.filter((b) => b?.type === "text" && typeof b.text === "string")
			.map((b) => b.text as string)
			.join("\n");
		const thinking = blocks
			.filter((b) => b?.type === "thinking" && typeof b.thinking === "string")
			.map((b) => b.thinking as string)
			.join("\n");
		const output = Number(msg.usage?.output ?? 0);
		const reasoning = Number(msg.usage?.reasoning ?? 0);
		const capped =
			thinking.length > 0 && (reasoning > 0 ? reasoning >= THINKING_BUDGET - 64 : output >= THINKING_BUDGET);
		const stopReason = String(msg.stopReason ?? "");
		last = {
			stopReason,
			hasToolCall,
			hasText: text.trim().length > 0,
			textHasCallMarkup: /<tool_call>|<function=/.test(text),
			capped,
		};
		if (hasToolCall) inARow = 0;
		if (capped) pi.appendEntry("acted", { what: "reasoning hit the budget" });

		// Keep a looping or length-cut block out of every later request.
		const cutAtLength = stopReason === "length";
		let changed = false;
		const content = blocks.map((b) => {
			if (b?.type === "thinking" && typeof b.thinking === "string" && b.thinking.length > KEEP_CHARS) {
				if (cutAtLength || looksDegenerate(b.thinking)) {
					changed = true;
					return { ...b, thinking: shorten(b.thinking, cutAtLength ? "cut at the length limit" : "it was repeating itself") };
				}
			}
			if (b?.type === "text" && typeof b.text === "string" && b.text.length > KEEP_CHARS) {
				if (cutAtLength || looksDegenerate(b.text)) {
					changed = true;
					return { ...b, text: shorten(b.text, cutAtLength ? "cut at the length limit" : "it was repeating itself") };
				}
			}
			return b;
		});
		if (!changed) return;
		pi.appendEntry("acted", { what: "trimmed a looping or cut response" });
		return { message: { ...msg, content } };
	});

	pi.on("agent_before_settle", async (event) => {
		if (event.outcome !== "completed" || !last || last.hasToolCall) return;
		let reason: Reason | undefined;
		if (last.stopReason === "length") reason = "length";
		else if (!last.hasText) reason = "empty";
		else if (last.textHasCallMarkup) reason = "markup";
		else if (last.capped) reason = "capped";
		if (!reason || nudges >= MAX_NUDGES_PER_RUN || inARow >= MAX_NUDGES_IN_A_ROW) return;
		nudges++;
		inARow++;
		last = undefined;
		pi.appendEntry("acted", { what: `continued after ${ACTED[reason]}` });
		return {
			entries: [
				...event.entries,
				{ type: "custom_message", customType: "response-cut", content: NUDGES[reason], display: true },
			],
			continue: true,
		};
	});
}
