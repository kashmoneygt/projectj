import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "node:fs";

// Measures model calls for the rsi runner, writing one JSON line per event to RSI_METRICS_PATH.
// For each request we write how its context splits up (counted with /tokenize) and the parameters as
// sent (model, sampling, output cap), and, whenever they change, the system prompt and tool definitions.
// This extension loads after the variant's, so it sees the request after every variant hook ran. For each
// assistant message we write ttft (start to first token), gen, and thinking (first token to first content token).
// A failure in here should never break the run.
const root = process.env.RSI_MODEL_ROOT ?? "";
const out = process.env.RSI_METRICS_PATH ?? "";

async function tokenize(text: string): Promise<number> {
	if (!text) return 0;
	const resp = await fetch(`${root}/tokenize`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ content: text }),
	});
	if (!resp.ok) throw new Error(`tokenize failed: ${resp.status}`);
	const data = (await resp.json()) as { tokens: number[] };
	return data.tokens.length;
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	return JSON.stringify(content ?? "");
}

// sync on purpose, because an async append can get lost if pi exits right after the last message_end
function writeLine(line: object) {
	try {
		appendFileSync(out, `${JSON.stringify(line)}\n`);
	} catch (err) {
		console.error("rsi metrics: append failed:", err);
	}
}

export default function (pi: ExtensionAPI) {
	if (!root || !out) return;
	let msgStartTs = 0;
	let firstTokenTs = 0;
	let firstContentTs = 0;
	let lastContext = "";
	try {
		pi.on("before_provider_request", async (event) => {
			const startTs = Date.now();
			let params: Record<string, unknown> = {};
			try {
				const payload = event.payload as {
					messages?: { role?: string; content?: unknown }[];
					tools?: unknown;
					system?: string;
					[key: string]: unknown;
				};
				const { messages: _messages, tools: _tools, system: _system, ...sent } = payload;
				params = sent;
				const messages = Array.isArray(payload.messages) ? payload.messages : [];
				// pi sends the system prompt as role "developer" when the model has reasoning enabled
				const isSystem = (m: { role?: string }) => m.role === "system" || m.role === "developer";
				const system = [
					payload.system ?? "",
					...messages.filter(isSystem).map((m) => textOf(m.content)),
				].join("\n");
				const rest = messages.filter((m) => !isSystem(m));
				const context = JSON.stringify({ system, tools: payload.tools ?? null });
				if (context !== lastContext) {
					writeLine({ type: "context", ts: startTs, system, tools: payload.tools ?? null });
					lastContext = context;
				}
				const [systemTokens, toolTokens, messageTokens] = await Promise.all([
					tokenize(system),
					tokenize(JSON.stringify(payload.tools ?? null)),
					tokenize(JSON.stringify(rest)),
				]);
				writeLine({
					ts: startTs,
					system_tokens: systemTokens,
					tool_def_tokens: toolTokens,
					message_tokens: messageTokens,
					params,
				});
			} catch (err) {
				console.error("rsi metrics: no context breakdown for this request:", err);
				writeLine({ ts: startTs, params });
			}
		});
		pi.on("message_start", (event) => {
			if (event.message.role === "assistant") {
				msgStartTs = Date.now();
				firstTokenTs = 0;
				firstContentTs = 0;
			}
		});
		pi.on("message_update", (event) => {
			if (!msgStartTs) return;
			const kind = event.assistantMessageEvent.type;
			const isToken =
				kind === "thinking_start" || kind === "text_start" || kind === "toolcall_start" || kind.endsWith("_delta");
			if (isToken && !firstTokenTs) firstTokenTs = Date.now();
			if ((kind === "text_start" || kind === "toolcall_start") && !firstContentTs) firstContentTs = Date.now();
		});
		pi.on("message_end", (event) => {
			const message = event.message;
			if (message.role !== "assistant" || !msgStartTs) return;
			const endTs = Date.now();
			// think_ms: from the first token (which may be thinking) to the first content token
			writeLine({
				type: "timing",
				ts: endTs,
				ttft_ms: (firstTokenTs || endTs) - msgStartTs,
				gen_ms: endTs - msgStartTs,
				think_ms: firstTokenTs ? Math.max(0, (firstContentTs || endTs) - firstTokenTs) : 0,
				stop_reason: message.stopReason,
			});
			msgStartTs = 0;
			firstTokenTs = 0;
			firstContentTs = 0;
		});
	} catch (err) {
		console.error("rsi metrics: failed to register handlers:", err);
	}
}
