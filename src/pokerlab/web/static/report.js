// Session report (wave-3 [P5']): render what the training loop produced.
// No framework, no build — same house style as app.js.
//
// The one rule this file must never break (plan §9): the exact-tier EV-loss
// ranking and the tier-3 frequency deviations are rendered as SEPARATE
// sections, and the tier-3 section carries the label the server sent. The
// label is never hard-coded here — it travels with the data, so the page
// cannot claim more precision than the grader did.

const API_REPORT = "/api/report";

async function errorText(res) {
  try {
    const body = await res.json();
    if (body && body.detail) return body.detail;
  } catch (e) {
    /* not JSON — fall back to the status line */
  }
  return `${res.status} ${res.statusText}`;
}

function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}

function table(head, rows, render) {
  if (!rows.length) return el("p", "no data yet", "empty");
  const t = el("table");
  const hr = el("tr");
  head.forEach((h) => hr.appendChild(el("th", h)));
  t.appendChild(hr);
  rows.forEach((r) => {
    const tr = el("tr");
    render(r).forEach(([text, cls]) => tr.appendChild(el("td", text, cls)));
    t.appendChild(tr);
  });
  return t;
}

function num(x, digits) {
  return x === null || x === undefined ? "—" : Number(x).toFixed(digits);
}

function renderSummary(s) {
  const box = document.getElementById("summary");
  box.innerHTML = "";
  box.appendChild(
    el("p", `${s.hands} hands imported · ${s.graded} decisions graded`)
  );
  // A partial session is a claim about COMPLETENESS, so it gets a banner and
  // not a footnote: numbers below it are computed over an incomplete sample.
  // This now only fires for work that is genuinely still coming — terminal
  // rows are reported below instead, because promising that a drain will move
  // numbers it can never move is the same over-claim the tier labels exist to
  // prevent (wave-3 [P5'] follow-on).
  if (s.partial) {
    box.appendChild(el(
      "div",
      `⚠️ PARTIAL — ${s.queued} of this session's decisions are still queued ` +
      `for solving. Run pokerlab-batch; the leak numbers below will change.`,
      "banner partial"
    ));
  }
  // The DB-wide backlog, shown only when it exceeds this session's own share —
  // otherwise it is the same number twice and reads as a second problem. Its
  // scope is the server's sentence, never restated here: this page cannot know
  // which imports those rows came from, so it must not describe them
  // (round-4 [9]).
  if (s.in_flight_total > s.queued) {
    const line = el("div", "", "banner blocked");
    line.appendChild(el("strong", `${s.in_flight_total} queued in total`));
    line.appendChild(el("span", ` — ${s.in_flight_label}`));
    box.appendChild(line);
  }
  // Each terminal cause carries the action that actually clears IT. Rendering
  // them as one number told the user to retry rows a retry cannot fix — and for
  // 'mismatched' a retry re-derives the same wrong spot and fails identically.
  (s.blocked || []).forEach((b) => {
    const line = el("div", "", "banner blocked");
    line.appendChild(el("strong", `${b.count} ${b.status}`));
    line.appendChild(el("span", ` — ${b.action}`));
    box.appendChild(line);
  });

  // Failed hands: PLAN §8's M4 exit promises these are never silently dropped.
  // They are NOT part of the partial banner — draining cannot change them — so
  // they get their own block, and it has to be unmissable, because `partial`
  // reading false while hands went ungraded is only honest if this is visible.
  if (s.failed) {
    const box2 = el("div", "", "banner failed");
    box2.appendChild(el("strong", `${s.failed} hands could not be graded`));
    box2.appendChild(el("span", ` — ${s.failed_action}`));
    const list = el("ul");
    (s.failed_hands || []).forEach((f) => {
      // `label` is the server's rendering of a possibly-absent hand number:
      // an unknown id is stated, never blanked and never invented.
      list.appendChild(el("li", `${f.site} ${f.label} — ${f.reason}`));
    });
    box2.appendChild(list);
    box.appendChild(box2);
  }
}

function renderLeaks(leaks) {
  const box = document.getElementById("leaks");
  box.innerHTML = "";
  const rows = leaks.rows;
  box.appendChild(table(
    ["category", "decisions", "EV-loss/100 (bb)", "reference"], rows,
    (r) => [
      [r.leak_key], [String(r.decisions), "num"],
      [num(r.ev_loss_per_100, 2), "num"],
      // Per-row, and a COUNT — a leak_key mixes tiers, so some of these
      // decisions were graded against an exact chart reference and some
      // against an approximated solve. "exact" is the honest word for a row
      // with nothing to disclose: a NULL provenance means there is nothing to
      // disclose, not that the reference is unknown.
      r.approx_decisions
        ? [`${r.approx_decisions} of ${r.decisions} approximated`, "approx-cell"]
        : ["exact"],
    ]
  ));
  // Explained once, beneath the table, and only when something in it is
  // actually approximated. The sentence is the server's — this page must not
  // author its own account of what the solver assumed, or it becomes a second
  // home for a claim the grader owns.
  if (rows.some((r) => r.approx_decisions)) {
    const note = el("p", "", "approx");
    note.appendChild(el("span", leaks.approx_caveat));
    // The assumption string itself, as recorded, for rows that carry one.
    const seen = [...new Set(rows.map((r) => r.provenance).filter(Boolean))];
    if (seen.length) note.appendChild(el("code", ` ${seen.join("; ")}`));
    box.appendChild(note);
  }
}

function renderTier3(tier3) {
  const box = document.getElementById("tier3");
  box.innerHTML = "";
  // The server's own words for what this tier can and cannot claim.
  box.appendChild(el("p", tier3.label, "tier-label"));
  box.appendChild(table(
    ["category", "your action", "times", "population freq", "flags"],
    tier3.rows,
    (r) => [
      [r.leak_key], [r.chosen], [String(r.decisions), "num"],
      // A missing baseline is "unknown", never "0% of the population" —
      // rendering it as 0 would invent a deviation that was never measured.
      [r.population_frequency === null
        ? "no baseline"
        : `${(r.population_frequency * 100).toFixed(0)}%`, "num"],
      [r.flags.length ? r.flags.join(", ") : "—"],
    ]
  ));
}

function renderGates(gates) {
  const box = document.getElementById("gates");
  box.innerHTML = "";

  box.appendChild(el("h3", "EV-loss/100 over time (exact tiers)"));
  box.appendChild(table(
    ["bucket", "category", "decisions", "EV-loss/100 (bb)"],
    gates.ev_loss_trend,
    (r) => [[r.bucket], [r.leak_key], [String(r.decisions), "num"],
            [num(r.ev_loss_per_100, 2), "num"]]
  ));

  box.appendChild(el("h3", "Drill accuracy by kind (last 30 days)"));
  box.appendChild(table(
    ["kind", "attempts", "accuracy"], gates.accuracy_by_kind,
    (r) => [[r.kind], [String(r.attempts), "num"],
            [`${(r.accuracy * 100).toFixed(0)}%`, "num"]]
  ));
}

async function load() {
  const res = await fetch(API_REPORT);
  if (!res.ok) {
    document.getElementById("summary").textContent =
      `⚠️ could not load the report: ${await errorText(res)}`;
    return;
  }
  const body = await res.json();
  renderSummary(body.session);
  renderLeaks(body.leaks);
  renderTier3(body.tier3);
  renderGates(body.gates);
}

window.addEventListener("DOMContentLoaded", load);
