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

function render(spot) {
  document.getElementById("description").textContent = spot.description;
  document.getElementById("meta").textContent =
    `${spot.position} · ${spot.depth_bb}bb · pot ${spot.pot_bb}bb`;
  const tc = spot.tournament;
  document.getElementById("tournament").textContent = tc
    ? `ICM · ${tc.players_remaining} left · stacks ${tc.stacks_all.join("/")}`
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
