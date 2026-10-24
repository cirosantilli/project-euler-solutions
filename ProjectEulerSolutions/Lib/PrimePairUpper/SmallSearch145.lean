import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch144
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch145

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan241 : scan edge (search edge 4) 5 [] 0 tail241 best = best := by
  have cons : tail241 = 1549 :: tail242 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates241_checked]
  change scan edge (search edge 4) 5 [] 0 tail242
    (if 4 ≤ rootCandidates241.length then search edge 4 [1549] rootCandidates241 1549 best else best) = best
  rw [if_pos (by decide), rootBranch241]
  exact outerScan242

set_option maxRecDepth 4096 in
theorem outerScan240 : scan edge (search edge 4) 5 [] 0 tail240 best = best := by
  have cons : tail240 = 1543 :: tail241 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates240_checked]
  change scan edge (search edge 4) 5 [] 0 tail241
    (if 4 ≤ rootCandidates240.length then search edge 4 [1543] rootCandidates240 1543 best else best) = best
  rw [if_pos (by decide), rootBranch240]
  exact outerScan241

end ProjectEulerSolutions.PrimePairUpper.Search
