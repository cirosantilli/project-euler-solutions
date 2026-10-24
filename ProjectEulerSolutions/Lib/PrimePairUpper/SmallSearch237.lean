import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch236
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch237

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan33 : scan edge (search edge 4) 5 [] 0 tail33 best = best := by
  have cons : tail33 = 151 :: tail34 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates33_checked]
  change scan edge (search edge 4) 5 [] 0 tail34
    (if 4 ≤ rootCandidates33.length then search edge 4 [151] rootCandidates33 151 best else best) = best
  rw [if_pos (by decide), rootBranch33]
  exact outerScan34

set_option maxRecDepth 4096 in
theorem outerScan32 : scan edge (search edge 4) 5 [] 0 tail32 best = best := by
  have cons : tail32 = 149 :: tail33 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates32_checked]
  change scan edge (search edge 4) 5 [] 0 tail33
    (if 4 ≤ rootCandidates32.length then search edge 4 [149] rootCandidates32 149 best else best) = best
  rw [if_pos (by decide), rootBranch32]
  exact outerScan33

end ProjectEulerSolutions.PrimePairUpper.Search
