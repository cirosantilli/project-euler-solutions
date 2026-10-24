import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch170
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch171

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan169 : scan edge (search edge 4) 5 [] 0 tail169 best = best := by
  have cons : tail169 = 1021 :: tail170 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates169_checked]
  change scan edge (search edge 4) 5 [] 0 tail170
    (if 4 ≤ rootCandidates169.length then search edge 4 [1021] rootCandidates169 1021 best else best) = best
  rw [if_pos (by decide), rootBranch169]
  exact outerScan170

set_option maxRecDepth 4096 in
theorem outerScan168 : scan edge (search edge 4) 5 [] 0 tail168 best = best := by
  have cons : tail168 = 1019 :: tail169 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates168_checked]
  change scan edge (search edge 4) 5 [] 0 tail169
    (if 4 ≤ rootCandidates168.length then search edge 4 [1019] rootCandidates168 1019 best else best) = best
  rw [if_pos (by decide), rootBranch168]
  exact outerScan169

end ProjectEulerSolutions.PrimePairUpper.Search
