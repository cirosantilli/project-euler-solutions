import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch155
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch156

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan208 : scan edge (search edge 4) 5 [] 0 tail208 best = best := by
  have cons : tail208 = 1297 :: tail209 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates208_checked]
  change scan edge (search edge 4) 5 [] 0 tail209
    (if 4 ≤ rootCandidates208.length then search edge 4 [1297] rootCandidates208 1297 best else best) = best
  rw [if_pos (by decide), rootBranch208]
  exact outerScan209

set_option maxRecDepth 4096 in
theorem outerScan207 : scan edge (search edge 4) 5 [] 0 tail207 best = best := by
  have cons : tail207 = 1291 :: tail208 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates207_checked]
  change scan edge (search edge 4) 5 [] 0 tail208
    (if 4 ≤ rootCandidates207.length then search edge 4 [1291] rootCandidates207 1291 best else best) = best
  rw [if_pos (by decide), rootBranch207]
  exact outerScan208

set_option maxRecDepth 4096 in
theorem outerScan206 : scan edge (search edge 4) 5 [] 0 tail206 best = best := by
  have cons : tail206 = 1289 :: tail207 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates206_checked]
  change scan edge (search edge 4) 5 [] 0 tail207
    (if 4 ≤ rootCandidates206.length then search edge 4 [1289] rootCandidates206 1289 best else best) = best
  rw [if_pos (by decide), rootBranch206]
  exact outerScan207

set_option maxRecDepth 4096 in
theorem outerScan205 : scan edge (search edge 4) 5 [] 0 tail205 best = best := by
  have cons : tail205 = 1283 :: tail206 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates205_checked]
  change scan edge (search edge 4) 5 [] 0 tail206
    (if 4 ≤ rootCandidates205.length then search edge 4 [1283] rootCandidates205 1283 best else best) = best
  rw [if_pos (by decide), rootBranch205]
  exact outerScan206

end ProjectEulerSolutions.PrimePairUpper.Search
