import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch151
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch152

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan221 : scan edge (search edge 4) 5 [] 0 tail221 best = best := by
  have cons : tail221 = 1423 :: tail222 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates221_checked]
  change scan edge (search edge 4) 5 [] 0 tail222
    (if 4 ≤ rootCandidates221.length then search edge 4 [1423] rootCandidates221 1423 best else best) = best
  rw [if_pos (by decide), rootBranch221]
  exact outerScan222

set_option maxRecDepth 4096 in
theorem outerScan220 : scan edge (search edge 4) 5 [] 0 tail220 best = best := by
  have cons : tail220 = 1409 :: tail221 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates220_checked]
  change scan edge (search edge 4) 5 [] 0 tail221
    (if 4 ≤ rootCandidates220.length then search edge 4 [1409] rootCandidates220 1409 best else best) = best
  rw [if_pos (by decide), rootBranch220]
  exact outerScan221

set_option maxRecDepth 4096 in
theorem outerScan219 : scan edge (search edge 4) 5 [] 0 tail219 best = best := by
  have cons : tail219 = 1399 :: tail220 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates219_checked]
  change scan edge (search edge 4) 5 [] 0 tail220
    (if 4 ≤ rootCandidates219.length then search edge 4 [1399] rootCandidates219 1399 best else best) = best
  rw [if_pos (by decide), rootBranch219]
  exact outerScan220

set_option maxRecDepth 4096 in
theorem outerScan218 : scan edge (search edge 4) 5 [] 0 tail218 best = best := by
  have cons : tail218 = 1381 :: tail219 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates218_checked]
  change scan edge (search edge 4) 5 [] 0 tail219
    (if 4 ≤ rootCandidates218.length then search edge 4 [1381] rootCandidates218 1381 best else best) = best
  rw [if_pos (by decide), rootBranch218]
  exact outerScan219

end ProjectEulerSolutions.PrimePairUpper.Search
