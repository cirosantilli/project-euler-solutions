import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch105
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch106

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan371 : scan edge (search edge 4) 5 [] 0 tail371 best = best := by
  have cons : tail371 = 2551 :: tail372 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates371_checked]
  change scan edge (search edge 4) 5 [] 0 tail372
    (if 4 ≤ rootCandidates371.length then search edge 4 [2551] rootCandidates371 2551 best else best) = best
  rw [if_pos (by decide), rootBranch371]
  exact outerScan372

set_option maxRecDepth 4096 in
theorem outerScan370 : scan edge (search edge 4) 5 [] 0 tail370 best = best := by
  have cons : tail370 = 2549 :: tail371 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates370_checked]
  change scan edge (search edge 4) 5 [] 0 tail371
    (if 4 ≤ rootCandidates370.length then search edge 4 [2549] rootCandidates370 2549 best else best) = best
  rw [if_pos (by decide), rootBranch370]
  exact outerScan371

set_option maxRecDepth 4096 in
theorem outerScan369 : scan edge (search edge 4) 5 [] 0 tail369 best = best := by
  have cons : tail369 = 2543 :: tail370 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates369_checked]
  change scan edge (search edge 4) 5 [] 0 tail370
    (if 4 ≤ rootCandidates369.length then search edge 4 [2543] rootCandidates369 2543 best else best) = best
  rw [if_pos (by decide), rootBranch369]
  exact outerScan370

end ProjectEulerSolutions.PrimePairUpper.Search
