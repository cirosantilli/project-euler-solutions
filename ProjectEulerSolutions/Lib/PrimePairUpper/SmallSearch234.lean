import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch233
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch234

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan39 : scan edge (search edge 4) 5 [] 0 tail39 best = best := by
  have cons : tail39 = 181 :: tail40 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates39_checked]
  change scan edge (search edge 4) 5 [] 0 tail40
    (if 4 ≤ rootCandidates39.length then search edge 4 [181] rootCandidates39 181 best else best) = best
  rw [if_pos (by decide), rootBranch39]
  exact outerScan40

set_option maxRecDepth 4096 in
theorem outerScan38 : scan edge (search edge 4) 5 [] 0 tail38 best = best := by
  have cons : tail38 = 179 :: tail39 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates38_checked]
  change scan edge (search edge 4) 5 [] 0 tail39
    (if 4 ≤ rootCandidates38.length then search edge 4 [179] rootCandidates38 179 best else best) = best
  rw [if_pos (by decide), rootBranch38]
  exact outerScan39

end ProjectEulerSolutions.PrimePairUpper.Search
