import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch153
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch154

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan214 : scan edge (search edge 4) 5 [] 0 tail214 best = best := by
  have cons : tail214 = 1327 :: tail215 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates214_checked]
  change scan edge (search edge 4) 5 [] 0 tail215
    (if 4 ≤ rootCandidates214.length then search edge 4 [1327] rootCandidates214 1327 best else best) = best
  rw [if_pos (by decide), rootBranch214]
  exact outerScan215

set_option maxRecDepth 4096 in
theorem outerScan213 : scan edge (search edge 4) 5 [] 0 tail213 best = best := by
  have cons : tail213 = 1321 :: tail214 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates213_checked]
  change scan edge (search edge 4) 5 [] 0 tail214
    (if 4 ≤ rootCandidates213.length then search edge 4 [1321] rootCandidates213 1321 best else best) = best
  rw [if_pos (by decide), rootBranch213]
  exact outerScan214

set_option maxRecDepth 4096 in
theorem outerScan212 : scan edge (search edge 4) 5 [] 0 tail212 best = best := by
  have cons : tail212 = 1319 :: tail213 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates212_checked]
  change scan edge (search edge 4) 5 [] 0 tail213
    (if 4 ≤ rootCandidates212.length then search edge 4 [1319] rootCandidates212 1319 best else best) = best
  rw [if_pos (by decide), rootBranch212]
  exact outerScan213

end ProjectEulerSolutions.PrimePairUpper.Search
