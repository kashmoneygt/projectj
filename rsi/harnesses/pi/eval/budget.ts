import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync, statSync } from "node:fs";

// Stop the run once pi tries more than RSI_MAX_CALLS model requests, like the step limit in benchmark harnesses.
// The model isn't told. All pi processes in the container append to the same counter file, so a second pi counts too.
// Compaction calls the model without this hook, so the runner's audit allows a few requests over the budget.
const max = Number(process.env.RSI_MAX_CALLS ?? 0);
const counter = process.env.RSI_CALLS_PATH ?? "";
const metrics = process.env.RSI_METRICS_PATH ?? "";

export default function (pi: ExtensionAPI) {
	if (!max || !counter) return;
	pi.on("before_provider_request", (_event, ctx) => {
		appendFileSync(counter, ".");
		const calls = statSync(counter).size;
		if (calls <= max) return;
		if (metrics) appendFileSync(metrics, `${JSON.stringify({ type: "budget", ts: Date.now(), calls: max })}\n`);
		ctx.abort();
	});
}
