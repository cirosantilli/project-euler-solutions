import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch147
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch148

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan232 : scan edge (search edge 4) 5 [] 0 tail232 best = best := by
  have cons : tail232 = 1483 :: tail233 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates232_checked]
  change scan edge (search edge 4) 5 [] 0 tail233
    (if 4 ≤ rootCandidates232.length then search edge 4 [1483] rootCandidates232 1483 best else best) = best
  rw [if_pos (by decide), rootBranch232]
  exact outerScan233

set_option maxRecDepth 4096 in
theorem outerScan231 : scan edge (search edge 4) 5 [] 0 tail231 best = best := by
  have cons : tail231 = 1481 :: tail232 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates231_checked]
  change scan edge (search edge 4) 5 [] 0 tail232
    (if 4 ≤ rootCandidates231.length then search edge 4 [1481] rootCandidates231 1481 best else best) = best
  rw [if_pos (by decide), rootBranch231]
  exact outerScan232

set_option maxRecDepth 4096 in
theorem outerScan230 : scan edge (search edge 4) 5 [] 0 tail230 best = best := by
  have cons : tail230 = 1471 :: tail231 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates230_checked]
  change scan edge (search edge 4) 5 [] 0 tail231
    (if 4 ≤ rootCandidates230.length then search edge 4 [1471] rootCandidates230 1471 best else best) = best
  rw [if_pos (by decide), rootBranch230]
  exact outerScan231

end ProjectEulerSolutions.PrimePairUpper.Search
