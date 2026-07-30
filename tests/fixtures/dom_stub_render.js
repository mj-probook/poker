// Execute the SHIPPED app.js render() against real server payloads with a
// stub DOM. The stub's one hard rule: document.getElementById() may only be
// asked for ids that exist in the shipped index.html (passed in argv) — a
// renderer reaching for a slot the page does not have is exactly the class
// of regression the text-contract tests cannot see.
//
// usage: node dom_stub_render.js <app.js> <spots.json> <ids.json>
"use strict";
const fs = require("fs");
const [appJsPath, spotsPath, idsPath] = process.argv.slice(2);
const validIds = new Set(JSON.parse(fs.readFileSync(idsPath, "utf8")));
const spots = JSON.parse(fs.readFileSync(spotsPath, "utf8"));

class El {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.className = "";
    this.textContent = "";
    this._innerHTML = "";
  }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener() {}
  setAttribute(k, v) { this[k] = v; }
  set innerHTML(v) { this._innerHTML = v; if (v === "") this.children = []; }
  get innerHTML() { return this._innerHTML; }
  // subtree text/markup for assertions
  dump() {
    return [this.textContent, this._innerHTML,
            ...this.children.map((c) => c.dump())].join(" ");
  }
}

const byId = new Map();
const document = {
  getElementById(id) {
    if (!validIds.has(id)) {
      console.error(`FAIL unknown element id: ${id}`);
      process.exit(1);
    }
    if (!byId.has(id)) byId.set(id, new El("div"));
    return byId.get(id);
  },
  createElement(tag) { return new El(tag); },
};
const window = { addEventListener() {} };
const localStorage = { getItem: () => null, setItem() {} };
const fetch = () => new Promise(() => {}); // render() under test never awaits

const src = fs.readFileSync(appJsPath, "utf8");
const load = new Function("document", "window", "localStorage", "fetch",
                          src + "\n;return { render: render };");
const { render } = load(document, window, localStorage, fetch);

const out = {};
for (const [name, spot] of Object.entries(spots)) {
  render(spot);
  out[name] = {
    board_cards: byId.get("board").children.length,
    hole_cards: byId.get("holecards").children.length,
    hole_dump: byId.get("holecards").dump(),
    provenance: byId.get("provenance").textContent,
    action_line: byId.get("actionline").textContent,
    buttons: byId.get("actions").children.map((b) => b.textContent),
    seats_dump: byId.get("seats").dump(),
    meta_dump: byId.get("meta").dump(),
  };
}
console.log(JSON.stringify(out));
