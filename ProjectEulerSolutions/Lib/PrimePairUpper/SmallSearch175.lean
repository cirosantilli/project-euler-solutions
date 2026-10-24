import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch174
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch175

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan159 : scan edge (search edge 4) 5 [] 0 tail159 best = best := by
  have cons : tail159 = 953 :: tail160 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates159_checked]
  change scan edge (search edge 4) 5 [] 0 tail160
    (if 4 ≤ rootCandidates159.length then search edge 4 [953] rootCandidates159 953 best else best) = best
  rw [if_pos (by decide), rootBranch159]
  exact outerScan160

set_option maxRecDepth 4096 in
theorem outerScan158 : scan edge (search edge 4) 5 [] 0 tail158 best = best := by
  have cons : tail158 = 947 :: tail159 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates158_checked]
  change scan edge (search edge 4) 5 [] 0 tail159
    (if 4 ≤ rootCandidates158.length then search edge 4 [947] rootCandidates158 947 best else best) = best
  rw [if_pos (by decide), rootBranch158]
  exact outerScan159

set_option maxRecDepth 4096 in
theorem outerScan157 : scan edge (search edge 4) 5 [] 0 tail157 best = best := by
  have cons : tail157 = 941 :: tail158 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates157_checked]
  change scan edge (search edge 4) 5 [] 0 tail158
    (if 4 ≤ rootCandidates157.length then search edge 4 [941] rootCandidates157 941 best else best) = best
  rw [if_pos (by decide), rootBranch157]
  exact outerScan158

end ProjectEulerSolutions.PrimePairUpper.Search
