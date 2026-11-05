// Renders every page of viewer.html in a stub DOM to catch runtime errors node --check misses, against a running
// viewer or a file saved with `rsi view --save`.
// Usage: node tests/smoke_viewer_js.mjs [http://127.0.0.1:8765 | saved.html] [tasks per run, default 3]
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = process.argv[2] || "http://127.0.0.1:8765";
const perRun = Number(process.argv[3] || 3);
const saved = source.endsWith(".html");
const base = source.replace(/\/$/, "");
const page = saved ? await readFile(source, "utf8") : await (await fetch(base + "/")).text();

class El {
    innerHTML = "";
    textContent = "";
    dataset = {};
    style = {};
    classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
    addEventListener() {}
    querySelector() {
        return null;
    }
    querySelectorAll() {
        return [];
    }
    insertAdjacentHTML(_where, html) {
        this.innerHTML += html;
    }
    scrollIntoView() {}
}
const els = {};
const data = page.match(/<script type="application\/json" id="saved">([\s\S]*?)<\/script>/)?.[1];
const sandbox = {
    console,
    atob,
    Blob,
    Response,
    DecompressionStream,
    fetch: (path, opts) => fetch(base + path, opts),
    setInterval: () => 0,
    CSS: { escape: (s) => s },
    location: { hash: "" },
    history: { replaceState() {}, pushState() {}, back() {} },
    scrollY: 0,
    scrollTo() {},
    addEventListener() {},
    document: {
        hidden: false,
        body: new El(),
        querySelector: (sel) => (sel.startsWith("#") ? (els[sel] ??= new El()) : null),
        querySelectorAll: () => [],
        getElementById: (id) => (id === "saved" && data ? { textContent: data } : null),
        addEventListener() {},
    },
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1], sandbox, { filename: "viewer.html" });
if (saved && !data) throw new Error(`${source} has no saved answers`);
// the page's own api(), live or from the saved file
const get = (path) => vm.runInContext(`api(${JSON.stringify(path)})`, sandbox);
// a file saved with --node keeps only those runs' trajectories
const trajectory = (path) =>
    get(path).catch((error) => {
        if (saved && error.message === "not saved in this file") return null;
        throw error;
    });

let checked = 0;
const render = (what, data, code, marker) => {
    sandbox.__data = data;
    const html = vm.runInContext(code, sandbox) ?? els["#main"].innerHTML;
    if (!html.includes(marker)) throw new Error(`${what}: no ${marker} in what it rendered`);
    checked += 1;
};
const tree = await get("/api/tree");
render("the tree as a graph", tree, "HOME = 'graph'; DATA = __data; renderTree()", 'class="sg"');
render("the tree as cards", tree, "HOME = 'cards'; DATA = __data; renderTree()", 'class="node');
render("tasks across runs", await get("/api/flips"), "DATA = __data; renderFlips()", "<");
const cards = [];
const walk = (card) => {
    cards.push(card);
    card.children.forEach(walk);
};
tree.lineages.forEach((lineage) => walk(lineage.tree));
// each file the graph lists for a variant has its own part of the diff shown on hover
for (const card of cards.filter((c) => c.changed)) {
    sandbox.__name = card.name;
    sandbox.__changed = card.changed;
    const listed = vm.runInContext("changes(__changed).filter((f) => f.path).map((f) => f.path)", sandbox);
    const found = await vm.runInContext("Promise.all(changes(__changed).filter((f) => f.path).map((f) => "
        + "fileDiff(__name, f.path).then((lines) => lines.length)))", sandbox);
    const missing = listed.filter((_, k) => !found[k]);
    if (!listed.length || missing.length) throw new Error(`${card.name}: no diff for ${missing.join(", ") || "any file"}`);
    checked += 1;
}
for (const card of cards) {
    if (!card.runs.length) {
        const variant = await get(`/api/variant?name=${encodeURIComponent(card.name)}`);
        render(card.name, variant, "DATA = __data; renderVariant()", "panel");
        continue;
    }
    for (const run of card.runs) {
        const name = encodeURIComponent(run.name);
        const node = await get(`/api/node?name=${name}`);
        sandbox.__name = run.name;
        render(run.name, node, "VIEW = { page: 'node', name: __name, open: null }; DATA = __data; renderNode(true)", 'id="tasks"');
        for (const task of node.tasks.filter((t) => t.state !== "pending").slice(0, perRun)) {
            const data = await trajectory(`/api/task?node=${name}&iid=${encodeURIComponent(task.iid)}`);
            if (data === null) continue;
            render(`${run.name} ${task.iid}`, data, "trajHtml(__data, 'task')", 'class="chat"');
        }
        for (const round of node.rounds) {
            for (const session of round.sessions) {
                const data = await trajectory(`/api/session?node=${name}&round=${round.id}&session=${session.n}`);
                if (data === null) continue;
                render(`${run.name} ${round.id}/${session.n}`, data, "trajHtml(__data, 'gen')", 'class="chat"');
            }
        }
    }
}
console.log(`SMOKE PASS: ${checked} pages from ${source}`);
