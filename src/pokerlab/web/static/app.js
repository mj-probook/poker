// Drill loop + table-scene renderer: fetch next spot -> draw the room ->
// POST the chosen action -> show the Score -> repeat. No framework, no build.
//
// Rendering honesty rules (same discipline as the payload, plan §9):
//   - The page renders SERVER words for every claim (tier_label, ev_unit,
//     explanation) and never authors its own.
//   - Facts the payload can't back are not drawn: villain seats in multiway
//     spots carry no invented stack numbers (seat order of `stacks_all` is
//     unspecified — the stacks strip shows them unattributed), and the dealer
//     button is drawn only where it is a rules fact (heads-up: SB has it).
//   - Suits on the hero's cards are display-only: the 169-class label is what
//     is graded ("AKs" = any suited combo), so suits are assigned fixed.

const API_NEXT = "/api/drill/next";
const API_ANSWER = "/api/drill/answer";

let current = null;
let lastAnswer = null; // the full answer payload; renderRta reads its .rta

// Poker order for the button row, not payload order: fold cheapest ->
// all-in. Off-tree distractors interleave with the priced actions and are
// styled identically — a visual tell would give the answer away. Unknown
// labels sort last rather than first.
const ACTION_ORDER = ["fold", "check", "limp", "call", "bet", "bet 33%",
                      "bet 75%", "raise 2.2bb", "raise 3bb", "jam"];
const orderKey = (a) => {
  const i = ACTION_ORDER.indexOf(a);
  return i === -1 ? ACTION_ORDER.length : i;
};

// RTA toggle: a display preference persisted per-browser. The panel data
// rides on every answer regardless — the toggle only decides visibility.
let rtaOn = localStorage.getItem("rta") === "on";
const rtaBtn = document.getElementById("rta-toggle");
rtaBtn.setAttribute("aria-pressed", String(rtaOn));
rtaBtn.addEventListener("click", () => {
  rtaOn = !rtaOn;
  localStorage.setItem("rta", rtaOn ? "on" : "off");
  rtaBtn.setAttribute("aria-pressed", String(rtaOn));
  renderRta(); // reflect immediately on the answer already on screen
});

// The solver readout, from SERVER numbers only: per-action EV + frequency
// from the Solution, the decision-ε actually used, and a server-authored
// reasoning sentence. The page adds structural labels, never claims.
function renderRta() {
  const panel = document.getElementById("rta");
  if (!rtaOn || !lastAnswer || !lastAnswer.rta) {
    panel.hidden = true;
    return;
  }
  const r = lastAnswer.rta;
  const unit = lastAnswer.ev_unit;
  const rows = r.actions
    .slice()
    .sort((a, b) => b.ev - a.ev)
    .map(
      (a) =>
        `<tr class="${a.action === r.best_action ? "best-row" : ""}">` +
        `<td>${a.action}</td><td class="num">${a.ev.toFixed(2)}</td>` +
        `<td class="num">${(a.frequency * 100).toFixed(0)}%</td></tr>`
    )
    .join("");
  panel.innerHTML =
    `<h2>RTA · solver readout</h2>` +
    `<p class="best">best: <strong>${r.best_action}</strong></p>` +
    `<table><tr><th>action</th><th class="num">EV (${unit})</th>` +
    `<th class="num">freq</th></tr>${rows}</table>` +
    `<p class="eps">ε = ${r.epsilon.toFixed(2)} ${unit}</p>` +
    `<p class="why">${r.reasoning}</p>`;
  panel.hidden = false;
}

// A failed request used to fall straight through into the happy path, so the
// UI rendered "undefined" instead of saying what went wrong. Surface the
// server's error detail instead (FastAPI puts it in `detail`).
async function errorText(res) {
  try {
    const body = await res.json();
    if (body && body.detail) return body.detail;
  } catch (e) {
    /* not JSON — fall back to the status line */
  }
  return `${res.status} ${res.statusText}`;
}

function showError(message) {
  const fb = document.getElementById("feedback");
  fb.textContent = `⚠️ ${message}`;
  fb.className = "incorrect";
}

// ---- practice filter: choose WHAT to drill, on demand ---------------------
// The option lists are UI affordances only — the server owns the vocabulary
// and answers an impossible combination with a loud 400 (shown via
// showError), never a silent fallback.
const MODE_POS = {
  "all": [],
  "hu": ["SB", "BB"],
  "icm": ["SB", "BB"],
  "ring-jam": ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB"],
  "ring-defend": ["UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
  "open": ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB"],
  "resteal": ["UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
  "river": ["BB"],
  "multiway": ["BB"],
};
const modeSel = document.getElementById("mode");
const posSel = document.getElementById("pos");

function syncPosOptions() {
  const opts = MODE_POS[modeSel.value] || [];
  posSel.innerHTML = "";
  const any = document.createElement("option");
  any.value = "all";
  any.textContent = "any";
  posSel.appendChild(any);
  opts.forEach((p) => {
    const o = document.createElement("option");
    o.value = p;
    o.textContent = p;
    posSel.appendChild(o);
  });
  posSel.disabled = opts.length === 0;
}

modeSel.value = localStorage.getItem("mode") || "all";
if (!MODE_POS[modeSel.value]) modeSel.value = "all";
syncPosOptions();
const savedPos = localStorage.getItem("pos") || "all";
if (MODE_POS[modeSel.value].indexOf(savedPos) !== -1) posSel.value = savedPos;

modeSel.addEventListener("change", () => {
  localStorage.setItem("mode", modeSel.value);
  syncPosOptions();
  localStorage.setItem("pos", "all");
  loadNext();
});
posSel.addEventListener("change", () => {
  localStorage.setItem("pos", posSel.value);
  loadNext();
});

// players filter: how many are dealt in. The server owns which counts exist
// per mode and answers an impossible combination with a loud 400 (shown via
// showError) — the option list here is an affordance, not the vocabulary.
const playersSel = document.getElementById("players");
playersSel.value = localStorage.getItem("players") || "all";
playersSel.addEventListener("change", () => {
  localStorage.setItem("players", playersSel.value);
  loadNext();
});

async function loadNext() {
  const pos = posSel.disabled ? "all" : posSel.value || "all";
  const res = await fetch(
    `${API_NEXT}?mode=${modeSel.value}&pos=${pos}&players=${playersSel.value}`);
  if (!res.ok) {
    showError(`could not load the next drill: ${await errorText(res)}`);
    return;
  }
  const spot = await res.json();
  current = spot;
  render(spot);
}

// 1st/2nd/3rd/4th... 11-13 are the irregular cases ("11th", not "11st").
function ordinal(n) {
  const rem100 = n % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
  return `${n}${{ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"}`;
}

// ---- hero cards from the 169-class label ("AKs" / "T9o" / "77") ----------
function heroCards(hand) {
  const r1 = hand[0], r2 = hand[1];
  const suited = hand.length > 2 && hand[2] === "s";
  // fixed display suits: suited pairs of ranks share spades; otherwise ♠ + ♥
  return suited ? [[r1, "♠"], [r2, "♠"]] : [[r1, "♠"], [r2, "♥"]];
}

// concrete card string from the server ("As", "9d") -> [rank, suit glyph].
// Postflop payloads carry these because on a real board the suits ARE the
// strategy — the synthetic display suits above would misstate the spot.
const SUIT_GLYPH = { c: "♣", d: "♦", h: "♥", s: "♠" };
function concreteCard(cs) {
  return [cs[0], SUIT_GLYPH[cs[1]]];
}

function cardEl([rank, suit]) {
  const div = document.createElement("div");
  div.className = "card" + (suit === "♥" || suit === "♦" ? " red" : "");
  const rankLabel = rank === "T" ? "10" : rank;
  div.innerHTML =
    `<span class="corner">${rankLabel}<span class="s">${suit}</span></span>` +
    `<span class="pip">${suit}</span>`;
  return div;
}

// ---- seats around the felt ellipse ---------------------------------------
// Hero sits bottom-center (angle 90°); the rest spread evenly. Percent coords.
function seatXY(i, n) {
  const angle = (Math.PI / 2) + (i * 2 * Math.PI) / n;
  return [50 + 43 * Math.cos(angle), 50 + 38 * Math.sin(angle)];
}

function seatEl(x, y, { hero, pos, stack, cards, allin, folded, bet }) {
  const seat = document.createElement("div");
  seat.className = "seat" + (hero ? " hero" : "") + (folded ? " folded" : "");
  seat.style.left = `${x}%`;
  seat.style.top = `${y}%`;
  let html = `<div class="chairback"></div><div class="plate">`;
  if (pos) html += `<span class="pos">${pos}</span> `;
  if (stack != null) html += `<span class="stack">${stack}bb</span>`;
  // all-in replaces the stack number: those chips are in the middle now, and
  // a plate still reading "10bb" would state a stack the player no longer has
  if (allin) html += `<span class="allin">ALL-IN</span>`;
  // a raise shows the amount in front; `stack` already carries the server's
  // "behind" number for these seats
  if (bet) html += `<span class="rbet">${bet}</span>`;
  if (folded) html += `<span class="foldtag">fold</span>`;
  html += `</div>`;
  if (cards) html += `<div class="cardsback"><div class="miniback"></div><div class="miniback"></div></div>`;
  seat.innerHTML = html;
  return seat;
}

// chips pushed from a seat toward the pot, on the seat->center line.
// Three chips for a whole stack (all-in), one for a raise.
function jamChipsEl(x, y, count) {
  const jc = document.createElement("div");
  jc.className = "jamchips";
  const chips = ['<div class="chip c1"></div>', '<div class="chip c2"></div>',
                 '<div class="chip c3"></div>'];
  jc.innerHTML = chips.slice(0, count || 3).join("");
  jc.style.left = `${x + (50 - x) * 0.38}%`;
  jc.style.top = `${y + (44 - y) * 0.38}%`;
  return jc;
}

let heroSeatEl = null; // to-act pulse lives on the hero seat until answered

function render(spot) {
  document.getElementById("description").textContent = spot.description;

  // spot chips: server-authored labels only
  const meta = document.getElementById("meta");
  meta.innerHTML = "";
  const chips = [spot.tier_label, spot.kind];
  if (spot.ev_unit) chips.splice(1, 0, `EV in ${spot.ev_unit}`);
  chips.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "infochip";
    chip.textContent = label;
    meta.appendChild(chip);
  });

  const tc = spot.tournament;

  // The payout ladder is part of the QUESTION, not decoration (wave-3 [P8']).
  // An ICM spot is unanswerable without it: identical stacks play completely
  // differently on a flat ladder versus a top-heavy one, because what ICM
  // prices is the $ shape of the prizes. Showing "4 left" and the stacks while
  // withholding the payouts asked the user to solve for information the drill
  // was holding back — and the answer key uses it.
  const ladder = document.getElementById("ladder");
  const stacksline = document.getElementById("stacksline");
  ladder.innerHTML = "";
  stacksline.textContent = "";
  if (tc) {
    const left = document.createElement("span");
    left.className = "plaque left";
    left.textContent = `${tc.players_remaining} left`;
    ladder.appendChild(left);
    tc.payouts.forEach((p, i) => {
      const pl = document.createElement("span");
      pl.className = "plaque";
      pl.textContent = `${ordinal(i + 1)} · ${p}`;
      ladder.appendChild(pl);
    });
    // unattributed on purpose: the payload does not say which stack sits in
    // which seat, so the strip states them without pretending to know.
    stacksline.textContent = `stacks (chips): ${tc.stacks_all.join(" / ")}`;
  }

  // pot + ante
  const pot = document.getElementById("pot");
  const anteLine = tc && tc.ante ? `<div class="ante">ante ${tc.ante}</div>` : "";
  pot.innerHTML =
    `<div class="chips"><div class="chip c1"></div><div class="chip c2"></div>` +
    `<div class="chip c3"></div></div>` +
    `<div class="amt">pot ${spot.pot_bb}bb</div>` + anteLine;

  // the prior action, drawn on the felt in the server's words — the scene
  // never derives its own account of what happened before hero's decision
  document.getElementById("actionline").textContent = spot.action_line || "";

  // seats. Two payload shapes, deliberately (plan §9 — draw only facts):
  //   * spot.seats — per-seat states in action order, present where seat
  //     attribution IS a rules fact (HU + 9-max ring). Hero rotates to the
  //     bottom; folded seats muck; the jammer's chips push toward the pot.
  //   * null (ICM multiway) — the payload does not attribute stacks to
  //     seats, so villains stay anonymous card-backs, exactly as before.
  const seatsBox = document.getElementById("seats");
  seatsBox.innerHTML = "";
  if (spot.seats) {
    const n = spot.seats.length;
    const heroIdx = spot.seats.findIndex((s) => s.state === "hero");
    // dealer button seat: BTN when someone holds that label; heads-up the
    // SB has it (rules fact)
    const btnPos = spot.seats.some((s) => s.pos === "BTN") ? "BTN" : "SB";
    spot.seats.forEach((s, i) => {
      const [x, y] = seatXY((i - heroIdx + n) % n, n);
      const el = seatEl(x, y, {
        hero: s.state === "hero",
        pos: s.pos,
        // symmetric stacks are part of the solved model, so live seats may
        // state the shared depth; a raiser's plate carries the server's
        // "behind" number; a folded or all-in seat shows no stack
        stack: s.state === "hero" || s.state === "live" ? spot.depth_bb
             : s.state === "raise" ? s.behind : null,
        cards: s.state === "live" || s.state === "all-in" || s.state === "raise",
        allin: s.state === "all-in",
        folded: s.state === "folded",
        bet: s.state === "raise" ? s.bet : null,
      });
      if (s.state === "hero") {
        heroSeatEl = el;
        el.className += " toact"; // pulses until the answer lands
      }
      seatsBox.appendChild(el);
      if (s.state === "all-in") seatsBox.appendChild(jamChipsEl(x, y, 3));
      if (s.state === "raise") seatsBox.appendChild(jamChipsEl(x, y, 1));
      if (s.pos === btnPos) {
        const d = document.createElement("div");
        d.className = "dbtn";
        d.textContent = "D";
        d.style.left = `${x + (50 - x) * 0.22}%`;
        d.style.top = `${y + (44 - y) * 0.22}%`;
        seatsBox.appendChild(d);
      }
    });
  } else {
    const n = tc ? tc.players_remaining : 2;
    for (let i = 0; i < n; i++) {
      const [x, y] = seatXY(i, n);
      if (i === 0) {
        heroSeatEl = seatEl(x, y, {
          hero: true, pos: spot.position, stack: spot.depth_bb, cards: false,
        });
        heroSeatEl.className += " toact";
        seatsBox.appendChild(heroSeatEl);
      } else {
        seatsBox.appendChild(seatEl(x, y, { hero: false, cards: true }));
      }
    }
  }

  // community cards — server facts, drawn only when the payload states them
  const boardBox = document.getElementById("board");
  boardBox.innerHTML = "";
  (spot.board || []).forEach((cs) => boardBox.appendChild(cardEl(concreteCard(cs))));

  // hero hole cards: concrete server cards when stated (postflop), else the
  // 169-class label with synthetic display suits (preflop)
  const hole = document.getElementById("holecards");
  hole.innerHTML = "";
  const heroC = spot.hero_cards ? spot.hero_cards.map(concreteCard)
                                : heroCards(spot.hand);
  heroC.forEach((c) => hole.appendChild(cardEl(c)));

  // tier-2 provenance, server words only (null for chart drills hides it)
  document.getElementById("provenance").textContent = spot.provenance || "";

  // actions: priced + off-tree distractor buttons, one indistinguishable row
  // in poker order. Distractors are the server's (spot.off_tree_actions) —
  // the page never invents a button, so BB-facing-a-jam stays call/fold.
  const box = document.getElementById("actions");
  box.innerHTML = "";
  const all = spot.legal_actions.concat(spot.off_tree_actions || []);
  all.sort((a, b) => orderKey(a) - orderKey(b));
  all.forEach((action) => {
    const btn = document.createElement("button");
    btn.textContent = action;
    btn.dataset.action = action;
    btn.addEventListener("click", () => submitAnswer(action));
    box.appendChild(btn);
  });
  // why the action set is complete at two, in the server's words ("" hides it)
  document.getElementById("action-note").textContent = spot.action_note || "";
  document.getElementById("feedback").textContent = "";
  document.getElementById("feedback").className = "";
  lastAnswer = null;
  rangeBtn.disabled = true;      // grid holds this hand's row — answer first
  rangePanel.hidden = true;
  renderAdvisory(null);
  renderRta();
}

// ---- range grid: the key's own 13×13 range, revealed after answering ----
// Colors are per-ACTION so the same legend reads across kinds; frequencies
// stack left-to-right in each cell (the classic range-viewer encoding).
const ACTION_COLORS = {
  "jam": "#b03a30", "call": "#2e7d52", "check": "#33567f",
  "raise 2.2bb": "#d8b25c", "raise 3bb": "#b8862d",
  "bet 33%": "#d8b25c", "bet 75%": "#b8862d", "fold": "#54504a",
};
const rangeBtn = document.getElementById("range-btn");
const rangePanel = document.getElementById("range-panel");

async function showRange() {
  if (!lastAnswer) return;   // revealed AFTER the answer, never before —
                             // the grid contains this hand's row
  const res = await fetch(
    `/api/drill/range?drill_id=${encodeURIComponent(current.drill_id)}`);
  if (!res.ok) {
    showError(`could not load the range: ${await errorText(res)}`);
    return;
  }
  const body = await res.json();
  const legend = document.getElementById("range-legend");
  legend.innerHTML = "";
  body.actions.forEach((a) => {
    const item = document.createElement("span");
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = ACTION_COLORS[a] || "#888";
    item.appendChild(sw);
    item.appendChild(document.createTextNode(a));
    legend.appendChild(item);
  });
  const gridEl = document.getElementById("range-grid");
  gridEl.innerHTML = "";
  body.grid.forEach((row) => row.forEach((cell) => {
    const div = document.createElement("div");
    div.className = "rcell" + (cell.freq === null ? " absent" : "");
    div.textContent = cell.label;
    if (cell.freq !== null) {
      // stacked gradient in the SERVER's action order; title states the
      // exact numbers so the color is never the only carrier
      const stops = [];
      let at = 0;
      body.actions.forEach((a) => {
        const f = cell.freq[a] || 0;
        const color = ACTION_COLORS[a] || "#888";
        stops.push(`${color} ${at * 100}%`, `${color} ${(at + f) * 100}%`);
        at += f;
      });
      div.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
      div.title = body.actions
        .map((a) => `${a}: ${Math.round((cell.freq[a] || 0) * 100)}%`)
        .join("  ");
    } else {
      div.title = "not in this key's range here";
    }
    gridEl.appendChild(div);
  }));
  document.getElementById("range-ctx").textContent =
    `${body.tier_label} — ${body.range_ctx}`;
  rangePanel.hidden = false;
}
rangeBtn.addEventListener("click", showRange);

async function submitAnswer(action) {
  const res = await fetch(API_ANSWER, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drill_id: current.drill_id, action: action }),
  });
  if (!res.ok) {
    showError(`could not submit the answer: ${await errorText(res)}`);
    return;
  }
  const score = await res.json();
  const fb = document.getElementById("feedback");
  // three states, not two: tier-3 answers carry correct === null — NO
  // right/wrong claim exists there, so neither mark may render (a ❌ would
  // be a wrongness claim the tier bans; caught on a live screenshot)
  const mark = score.correct === null ? "" : score.correct ? "✅ " : "❌ ";
  fb.textContent = mark + score.explanation;
  fb.className = score.correct === null ? "neutral"
               : score.correct ? "correct" : "incorrect";
  if (heroSeatEl) heroSeatEl.className = "seat hero"; // action complete
  lastAnswer = score;
  rangeBtn.disabled = false;
  renderAdvisory(score.advisory);
  renderRta();
}

// Tier-3 advisory block: study numbers rendered UNDER their server-authored
// caveat, never without it — the caveat is the claim, the numbers are data.
function renderAdvisory(adv) {
  const box = document.getElementById("advisory");
  if (!adv) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  // Advisory fields render only when actually present and numeric — a
  // malformed payload must show LESS, never "NaN%"/"undefined" (which would
  // read as data in a block whose whole job is honest numbers).
  let html = adv.caveat ? `<p class="caveat">${adv.caveat}</p>` : "";
  if (adv.solve) {
    const rows = Object.entries(adv.solve)
      .sort((a, b) => b[1] - a[1])
      .map(([a, ev]) => `<tr><td>${a}</td><td>${ev.toFixed(2)}bb</td></tr>`)
      .join("");
    html += `<table>${rows}</table>`;
    html += `<p class="eqnote">HU-collapsed solve, measured gap ${adv.gap}bb</p>`;
  }
  if (Number.isFinite(adv.equity_vs_field)) {
    html += `<p>equity vs the field: ` +
            `${Math.round(adv.equity_vs_field * 100)}%</p>`;
  }
  if (adv.equity_note) html += `<p class="eqnote">${adv.equity_note}</p>`;
  box.innerHTML = html;
  box.hidden = false;
}

document.getElementById("next").addEventListener("click", loadNext);
window.addEventListener("DOMContentLoaded", loadNext);
