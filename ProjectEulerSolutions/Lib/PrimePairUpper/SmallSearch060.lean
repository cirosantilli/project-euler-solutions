import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch059
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch060

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan587 : scan edge (search edge 4) 5 [] 0 tail587 best = best := by
  have cons : tail587 = 4297 :: tail588 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates587_checked]
  change scan edge (search edge 4) 5 [] 0 tail588
    (if 4 ≤ rootCandidates587.length then search edge 4 [4297] rootCandidates587 4297 best else best) = best
  rw [if_pos (by decide), rootBranch587]
  exact outerScan588

set_option maxRecDepth 4096 in
theorem outerScan586 : scan edge (search edge 4) 5 [] 0 tail586 best = best := by
  have cons : tail586 = 4289 :: tail587 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates586_checked]
  change scan edge (search edge 4) 5 [] 0 tail587
    (if 4 ≤ rootCandidates586.length then search edge 4 [4289] rootCandidates586 4289 best else best) = best
  rw [if_pos (by decide), rootBranch586]
  exact outerScan587

set_option maxRecDepth 4096 in
theorem outerScan585 : scan edge (search edge 4) 5 [] 0 tail585 best = best := by
  have cons : tail585 = 4283 :: tail586 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates585_checked]
  change scan edge (search edge 4) 5 [] 0 tail586
    (if 4 ≤ rootCandidates585.length then search edge 4 [4283] rootCandidates585 4283 best else best) = best
  rw [if_pos (by decide), rootBranch585]
  exact outerScan586

set_option maxRecDepth 4096 in
theorem outerScan584 : scan edge (search edge 4) 5 [] 0 tail584 best = best := by
  have cons : tail584 = 4273 :: tail585 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates584_checked]
  change scan edge (search edge 4) 5 [] 0 tail585
    (if 4 ≤ rootCandidates584.length then search edge 4 [4273] rootCandidates584 4273 best else best) = best
  rw [if_pos (by decide), rootBranch584]
  exact outerScan585

set_option maxRecDepth 4096 in
theorem outerScan583 : scan edge (search edge 4) 5 [] 0 tail583 best = best := by
  have cons : tail583 = 4271 :: tail584 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates583_checked]
  change scan edge (search edge 4) 5 [] 0 tail584
    (if 4 ≤ rootCandidates583.length then search edge 4 [4271] rootCandidates583 4271 best else best) = best
  rw [if_pos (by decide), rootBranch583]
  exact outerScan584

set_option maxRecDepth 4096 in
theorem outerScan582 : scan edge (search edge 4) 5 [] 0 tail582 best = best := by
  have cons : tail582 = 4261 :: tail583 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates582_checked]
  change scan edge (search edge 4) 5 [] 0 tail583
    (if 4 ≤ rootCandidates582.length then search edge 4 [4261] rootCandidates582 4261 best else best) = best
  rw [if_pos (by decide), rootBranch582]
  exact outerScan583

set_option maxRecDepth 4096 in
theorem outerScan581 : scan edge (search edge 4) 5 [] 0 tail581 best = best := by
  have cons : tail581 = 4259 :: tail582 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates581_checked]
  change scan edge (search edge 4) 5 [] 0 tail582
    (if 4 ≤ rootCandidates581.length then search edge 4 [4259] rootCandidates581 4259 best else best) = best
  rw [if_pos (by decide), rootBranch581]
  exact outerScan582

end ProjectEulerSolutions.PrimePairUpper.Search
