import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch213
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch214

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan74 : scan edge (search edge 4) 5 [] 0 tail74 best = best := by
  have cons : tail74 = 389 :: tail75 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates74_checked]
  change scan edge (search edge 4) 5 [] 0 tail75
    (if 4 ≤ rootCandidates74.length then search edge 4 [389] rootCandidates74 389 best else best) = best
  rw [if_pos (by decide), rootBranch74]
  exact outerScan75

set_option maxRecDepth 4096 in
theorem outerScan73 : scan edge (search edge 4) 5 [] 0 tail73 best = best := by
  have cons : tail73 = 383 :: tail74 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates73_checked]
  change scan edge (search edge 4) 5 [] 0 tail74
    (if 4 ≤ rootCandidates73.length then search edge 4 [383] rootCandidates73 383 best else best) = best
  rw [if_pos (by decide), rootBranch73]
  exact outerScan74

end ProjectEulerSolutions.PrimePairUpper.Search
