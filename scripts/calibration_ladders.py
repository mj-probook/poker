"""[R4-a] ICM calibration falsification probe across LADDER SHAPES.

Claim under test (src/pokerlab/drills/scoring.py:50-56):
  within-eps rate chip-EV 12.3% vs ICM 1.8% is NOT mis-scaling but a LOW-TAIL
  effect -- chip p10 0.78x (charts genuinely mix) vs ICM p10 1.83x (near-pure).
Falsifier: if the low-tail explanation is fixture-specific, ICM p10 should move
a lot across ladder shapes, or the 1.8% should not track solve purity.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pokerlab.drills import generator as gen
from pokerlab.drills.scoring import epsilon
from pokerlab.types import TournamentContext

POOL = 1000  # held CONSTANT so ladder SHAPE is the only variable
LADDERS = {
    "bubble (recorded, 5/3/2)": (500, 300, 200),
    "flat        (334/333/333)": (334, 333, 333),
    "top-heavy   (800/150/50)": (800, 150, 50),
    "winner-take-all (1000)":   (1000,),
}

def stats(drills):
    mult, within, n_nonbest, purity = [], 0, 0, []
    for d in drills:
        acts = d.solution.actions
        best = max(acts, key=lambda a: acts[a][0])
        eps = epsilon(d.pot_bb, bb_value=d.bb_value)
        purity.append(max(f for a, (_, f) in acts.items() if a != best))
        for a, (ev, _) in acts.items():
            if a == best:
                continue
            n_nonbest += 1
            loss = acts[best][0] - ev
            mult.append(loss / eps if eps > 0 else np.inf)
            within += (loss <= eps)
    m = np.array(mult)
    return (100.0*within/n_nonbest, np.percentile(m,10), np.percentile(m,50),
            np.percentile(m,90), max(purity), n_nonbest)

chip = gen.jamfold_drills()
w,p10,p50,p90,pur,n = stats(chip)
print(f"{'CHIP-EV (ladder-independent)':32s} within-eps {w:5.1f}%   p10 {p10:5.2f}x  p50 {p50:5.2f}x  p90 {p90:5.2f}x   max non-best freq {pur:.4f}  (n={n})")
print(f"{'  recorded claim':32s} within-eps  12.3%   p10  0.78x  p50  7.40x  p90 24.40x")
print()
for name, pay in LADDERS.items():
    tc = TournamentContext(payouts=pay, players_remaining=4,
                           stacks_all=(1000,1000,1000,1000), bb=100, ante=0)
    d = gen.icm_drills(tc)
    w,p10,p50,p90,pur,n = stats(d)
    bbv = sum(pay)/ (sum(tc.stacks_all)/tc.bb)
    print(f"ICM {name:28s} within-eps {w:5.1f}%   p10 {p10:5.2f}x  p50 {p50:5.2f}x  p90 {p90:5.2f}x   max non-best freq {pur:.4f}   bb_value {bbv:.1f}$")
print(f"{'  recorded (bubble)':32s} within-eps   1.8%   p10  1.83x  p50  4.30x  p90 29.80x   bb_value 25.0$")

# scale-freeness: same SHAPE, 100x pool -> within-eps must be identical
tc10 = TournamentContext(payouts=(50000,30000,20000), players_remaining=4,
                         stacks_all=(1000,1000,1000,1000), bb=100, ante=0)
w2,*_ = stats(gen.icm_drills(tc10))
w1,*_ = stats(gen.icm_drills(TournamentContext(payouts=(500,300,200), players_remaining=4,
                         stacks_all=(1000,1000,1000,1000), bb=100, ante=0)))
print(f"\nscale-freeness check: pool 1000 -> {w1:.2f}%   pool 100000 (same shape) -> {w2:.2f}%   {'IDENTICAL' if abs(w1-w2)<1e-9 else 'DIVERGES'}")
