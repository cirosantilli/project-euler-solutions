import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch171
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch172

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan167 : scan edge (search edge 4) 5 [] 0 tail167 best = best := by
  have cons : tail167 = 1013 :: tail168 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates167_checked]
  change scan edge (search edge 4) 5 [] 0 tail168
    (if 4 ≤ rootCandidates167.length then search edge 4 [1013] rootCandidates167 1013 best else best) = best
  rw [if_pos (by decide), rootBranch167]
  exact outerScan168

set_option maxRecDepth 4096 in
theorem outerScan166 : scan edge (search edge 4) 5 [] 0 tail166 best = best := by
  have cons : tail166 = 1009 :: tail167 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates166_checked]
  change scan edge (search edge 4) 5 [] 0 tail167
    (if 4 ≤ rootCandidates166.length then search edge 4 [1009] rootCandidates166 1009 best else best) = best
  rw [if_pos (by decide), rootBranch166]
  exact outerScan167

set_option maxRecDepth 4096 in
theorem outerScan165 : scan edge (search edge 4) 5 [] 0 tail165 best = best := by
  have cons : tail165 = 997 :: tail166 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates165_checked]
  change scan edge (search edge 4) 5 [] 0 tail166
    (if 4 ≤ rootCandidates165.length then search edge 4 [997] rootCandidates165 997 best else best) = best
  rw [if_pos (by decide), rootBranch165]
  exact outerScan166

end ProjectEulerSolutions.PrimePairUpper.Search
