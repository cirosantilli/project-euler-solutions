import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch173
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch174

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan161 : scan edge (search edge 4) 5 [] 0 tail161 best = best := by
  have cons : tail161 = 971 :: tail162 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates161_checked]
  change scan edge (search edge 4) 5 [] 0 tail162
    (if 4 ≤ rootCandidates161.length then search edge 4 [971] rootCandidates161 971 best else best) = best
  rw [if_pos (by decide), rootBranch161]
  exact outerScan162

set_option maxRecDepth 4096 in
theorem outerScan160 : scan edge (search edge 4) 5 [] 0 tail160 best = best := by
  have cons : tail160 = 967 :: tail161 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates160_checked]
  change scan edge (search edge 4) 5 [] 0 tail161
    (if 4 ≤ rootCandidates160.length then search edge 4 [967] rootCandidates160 967 best else best) = best
  rw [if_pos (by decide), rootBranch160]
  exact outerScan161

end ProjectEulerSolutions.PrimePairUpper.Search
