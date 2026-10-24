import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch160
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch161

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan194 : scan edge (search edge 4) 5 [] 0 tail194 best = best := by
  have cons : tail194 = 1201 :: tail195 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates194_checked]
  change scan edge (search edge 4) 5 [] 0 tail195
    (if 4 ≤ rootCandidates194.length then search edge 4 [1201] rootCandidates194 1201 best else best) = best
  rw [if_pos (by decide), rootBranch194]
  exact outerScan195

set_option maxRecDepth 4096 in
theorem outerScan193 : scan edge (search edge 4) 5 [] 0 tail193 best = best := by
  have cons : tail193 = 1193 :: tail194 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates193_checked]
  change scan edge (search edge 4) 5 [] 0 tail194
    (if 4 ≤ rootCandidates193.length then search edge 4 [1193] rootCandidates193 1193 best else best) = best
  rw [if_pos (by decide), rootBranch193]
  exact outerScan194

set_option maxRecDepth 4096 in
theorem outerScan192 : scan edge (search edge 4) 5 [] 0 tail192 best = best := by
  have cons : tail192 = 1187 :: tail193 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates192_checked]
  change scan edge (search edge 4) 5 [] 0 tail193
    (if 4 ≤ rootCandidates192.length then search edge 4 [1187] rootCandidates192 1187 best else best) = best
  rw [if_pos (by decide), rootBranch192]
  exact outerScan193

end ProjectEulerSolutions.PrimePairUpper.Search
