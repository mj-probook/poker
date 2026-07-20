// Minimal drill loop (Slice E): fetch next spot -> render action buttons ->
// POST the chosen action -> show the Score -> repeat. No framework, no build.

const API_NEXT = "/api/drill/next";
const API_ANSWER = "/api/drill/answer";

let current = null;

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

async function loadNext() {
  const res = await fetch(API_NEXT);
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

function render(spot) {
  document.getElementById("description").textContent = spot.description;
  document.getElementById("meta").textContent =
    `${spot.position} · ${spot.depth_bb}bb · pot ${spot.pot_bb}bb`;
  const tc = spot.tournament;
  // The payout ladder is part of the QUESTION, not decoration (wave-3 [P8']).
  // An ICM spot is unanswerable without it: identical stacks play completely
  // differently on a flat ladder versus a top-heavy one, because what ICM
  // prices is the $ shape of the prizes. Showing "4 left" and the stacks while
  // withholding the payouts asked the user to solve for information the drill
  // was holding back — and the answer key uses it.
  document.getElementById("tournament").textContent = tc
    ? `ICM · ${tc.players_remaining} left · stacks ${tc.stacks_all.join("/")}` +
      ` · pays ${tc.payouts.map((p, i) => `${ordinal(i + 1)} ${p}`).join(" / ")}`
    : "";
  const box = document.getElementById("actions");
  box.innerHTML = "";
  spot.legal_actions.forEach((action) => {
    const btn = document.createElement("button");
    btn.textContent = action;
    btn.addEventListener("click", () => submitAnswer(action));
    box.appendChild(btn);
  });
  document.getElementById("feedback").textContent = "";
  document.getElementById("feedback").className = "";
}

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
  fb.textContent = (score.correct ? "✅ " : "❌ ") + score.explanation;
  fb.className = score.correct ? "correct" : "incorrect";
}

document.getElementById("next").addEventListener("click", loadNext);
window.addEventListener("DOMContentLoaded", loadNext);
