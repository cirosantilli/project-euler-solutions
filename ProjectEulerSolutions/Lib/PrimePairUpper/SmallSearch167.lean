import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch166
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch167

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan179 : scan edge (search edge 4) 5 [] 0 tail179 best = best := by
  have cons : tail179 = 1091 :: tail180 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates179_checked]
  change scan edge (search edge 4) 5 [] 0 tail180
    (if 4 ≤ rootCandidates179.length then search edge 4 [1091] rootCandidates179 1091 best else best) = best
  rw [if_pos (by decide), rootBranch179]
  exact outerScan180

set_option maxRecDepth 4096 in
theorem outerScan178 : scan edge (search edge 4) 5 [] 0 tail178 best = best := by
  have cons : tail178 = 1087 :: tail179 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates178_checked]
  change scan edge (search edge 4) 5 [] 0 tail179
    (if 4 ≤ rootCandidates178.length then search edge 4 [1087] rootCandidates178 1087 best else best) = best
  rw [if_pos (by decide), rootBranch178]
  exact outerScan179

set_option maxRecDepth 4096 in
theorem outerScan177 : scan edge (search edge 4) 5 [] 0 tail177 best = best := by
  have cons : tail177 = 1069 :: tail178 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates177_checked]
  change scan edge (search edge 4) 5 [] 0 tail178
    (if 4 ≤ rootCandidates177.length then search edge 4 [1069] rootCandidates177 1069 best else best) = best
  rw [if_pos (by decide), rootBranch177]
  exact outerScan178

end ProjectEulerSolutions.PrimePairUpper.Search
