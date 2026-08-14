// Drive the REAL served app through the five acceptance criteria, headless.
// Evidence: screenshots + a JSON verdict per criterion on stdout.
//
//   POKERLAB_PORT=8321 uv run pokerlab-web &        (temp DB)
//   node scripts/playwright_verify.mjs http://127.0.0.1:8321 /tmp/shots
//
// This is an ACCEPTANCE script, not a unit test: it clicks what the user
// clicks and reads what the user reads. Anything it asserts is a sentence
// the page actually rendered.
import { chromium } from "playwright";

const base = process.argv[2] || "http://127.0.0.1:8321";
const outdir = process.argv[3] || "/tmp/pokerlab-shots";
const results = {};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
page.setDefaultTimeout(15000);

async function shot(name) {
  await page.screenshot({ path: `${outdir}/${name}.png`, fullPage: true });
}

async function pickMode(mode, players = "all") {
  // ALWAYS set both selects — a sticky players filter from a previous step
  // otherwise leaks into every later criterion (measured the hard way)
  await page.selectOption("#mode", mode);
  await page.waitForTimeout(300);
  await page.selectOption("#players", players);
  await page.waitForTimeout(400);
  await page.waitForSelector("#actions button");
}

// Deal until the served spot satisfies `want`. Each miss is ANSWERED before
// dealing again — the due-first scheduler re-serves an unanswered category
// forever, so clicking "next" alone never advances the vocabulary.
async function dealUntil(want, tries = 40) {
  for (let i = 0; i < tries; i++) {
    const labels = await page.locator("#actions button").allTextContents();
    if (want(labels)) return labels;
    await answerFirst();
    await page.click("#next");
    await page.waitForTimeout(250);
  }
  return page.locator("#actions button").allTextContents();
}

async function answerFirst() {
  await page.click("#actions button >> nth=0");
  await page.waitForSelector("#feedback:not(:empty)");
}

try {
  await page.goto(base);
  await page.waitForSelector("#actions button");

  // ---- 1. switch between drills (practice modes) --------------------------
  const seen = [];
  for (const mode of ["hu", "icm", "ring-jam", "open", "river", "multiway"]) {
    await pickMode(mode);
    const desc = await page.textContent("#description");
    const kindChip = await page.textContent("#meta");
    seen.push({ mode, desc: desc.slice(0, 60), chips: kindChip });
  }
  results["1_switch_drills"] = { pass: seen.length === 6, seen };
  await shot("1-mode-switching");

  // ---- 2. select table sizes up to 8 (and beyond: 2–9) --------------------
  const sizes = {};
  for (const n of ["3", "5", "8"]) {
    await pickMode("ring-jam", n);
    sizes[n] = await page.locator("#seats .seat").count();
  }
  await shot("2-eight-handed");
  results["2_players_selector"] = {
    pass: sizes["3"] === 3 && sizes["5"] === 5 && sizes["8"] === 8,
    seats_rendered: sizes,
  };

  // ---- 3. view range grid -------------------------------------------------
  await pickMode("open", "all");
  await dealUntil((ls) => ls.some((l) => l.startsWith("raise")));
  await answerFirst();
  await page.click("#range-btn");
  await page.waitForSelector("#range-grid .rcell");
  const cells = await page.locator("#range-grid .rcell").count();
  const legend = await page.textContent("#range-legend");
  const ctx = await page.textContent("#range-ctx");
  await shot("3-range-grid");
  results["3_range_grid"] = {
    pass: cells === 169 && legend.includes("raise") && ctx.length > 10,
    cells, legend: legend.slice(0, 80), ctx: ctx.slice(0, 90),
  };

  // ---- 4. options beyond jam/fold ----------------------------------------
  await pickMode("open", "all");
  const labels = await dealUntil((ls) => ls.some((l) => l.startsWith("raise")));
  const beyond = labels.filter((l) => !["jam", "fold"].includes(l));
  await shot("4-open-buttons");
  results["4_beyond_jamfold"] = { pass: beyond.length >= 2, buttons: labels };

  // ---- 5. multiway postflop with advisory guidance ------------------------
  await pickMode("multiway", "all");
  const board = await page.locator("#board .card").count();
  const mwButtons = await page.locator("#actions button").allTextContents();
  await answerFirst();
  const feedback = await page.textContent("#feedback");
  const advisoryHidden = await page.locator("#advisory").isHidden();
  const advisory = advisoryHidden ? "" : await page.textContent("#advisory");
  await shot("5-multiway-advisory");
  results["5_multiway"] = {
    pass: board >= 3 && mwButtons.includes("check") && mwButtons.includes("bet")
      && feedback.toLowerCase().includes("tier 3")
      && !feedback.includes("✅") && !feedback.includes("❌")   // no verdict
      && advisory.includes("Advisory"),
    board_cards: board, buttons: mwButtons,
    feedback: feedback.slice(0, 140),
    advisory: advisory.slice(0, 160),
  };
} catch (err) {
  results["error"] = String(err);
}

await browser.close();
const allPass = Object.values(results).every((r) => r.pass === true);
console.log(JSON.stringify({ allPass, results }, null, 2));
process.exit(allPass ? 0 : 1);
