import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch110
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch111

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan353 : scan edge (search edge 4) 5 [] 0 tail353 best = best := by
  have cons : tail353 = 2393 :: tail354 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates353_checked]
  change scan edge (search edge 4) 5 [] 0 tail354
    (if 4 ≤ rootCandidates353.length then search edge 4 [2393] rootCandidates353 2393 best else best) = best
  rw [if_pos (by decide), rootBranch353]
  exact outerScan354

set_option maxRecDepth 4096 in
theorem outerScan352 : scan edge (search edge 4) 5 [] 0 tail352 best = best := by
  have cons : tail352 = 2389 :: tail353 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates352_checked]
  change scan edge (search edge 4) 5 [] 0 tail353
    (if 4 ≤ rootCandidates352.length then search edge 4 [2389] rootCandidates352 2389 best else best) = best
  rw [if_pos (by decide), rootBranch352]
  exact outerScan353

set_option maxRecDepth 4096 in
theorem outerScan351 : scan edge (search edge 4) 5 [] 0 tail351 best = best := by
  have cons : tail351 = 2383 :: tail352 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates351_checked]
  change scan edge (search edge 4) 5 [] 0 tail352
    (if 4 ≤ rootCandidates351.length then search edge 4 [2383] rootCandidates351 2383 best else best) = best
  rw [if_pos (by decide), rootBranch351]
  exact outerScan352

set_option maxRecDepth 4096 in
theorem outerScan350 : scan edge (search edge 4) 5 [] 0 tail350 best = best := by
  have cons : tail350 = 2381 :: tail351 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates350_checked]
  change scan edge (search edge 4) 5 [] 0 tail351
    (if 4 ≤ rootCandidates350.length then search edge 4 [2381] rootCandidates350 2381 best else best) = best
  rw [if_pos (by decide), rootBranch350]
  exact outerScan351

end ProjectEulerSolutions.PrimePairUpper.Search
