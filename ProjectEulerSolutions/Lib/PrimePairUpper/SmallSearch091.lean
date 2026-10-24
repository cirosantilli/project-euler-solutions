import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch090
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch091

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan429 : scan edge (search edge 4) 5 [] 0 tail429 best = best := by
  have cons : tail429 = 3011 :: tail430 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates429_checked]
  change scan edge (search edge 4) 5 [] 0 tail430
    (if 4 ≤ rootCandidates429.length then search edge 4 [3011] rootCandidates429 3011 best else best) = best
  rw [if_pos (by decide), rootBranch429]
  exact outerScan430

set_option maxRecDepth 4096 in
theorem outerScan428 : scan edge (search edge 4) 5 [] 0 tail428 best = best := by
  have cons : tail428 = 3001 :: tail429 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates428_checked]
  change scan edge (search edge 4) 5 [] 0 tail429
    (if 4 ≤ rootCandidates428.length then search edge 4 [3001] rootCandidates428 3001 best else best) = best
  rw [if_pos (by decide), rootBranch428]
  exact outerScan429

set_option maxRecDepth 4096 in
theorem outerScan427 : scan edge (search edge 4) 5 [] 0 tail427 best = best := by
  have cons : tail427 = 2999 :: tail428 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates427_checked]
  change scan edge (search edge 4) 5 [] 0 tail428
    (if 4 ≤ rootCandidates427.length then search edge 4 [2999] rootCandidates427 2999 best else best) = best
  rw [if_pos (by decide), rootBranch427]
  exact outerScan428

set_option maxRecDepth 4096 in
theorem outerScan426 : scan edge (search edge 4) 5 [] 0 tail426 best = best := by
  have cons : tail426 = 2971 :: tail427 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates426_checked]
  change scan edge (search edge 4) 5 [] 0 tail427
    (if 4 ≤ rootCandidates426.length then search edge 4 [2971] rootCandidates426 2971 best else best) = best
  rw [if_pos (by decide), rootBranch426]
  exact outerScan427

end ProjectEulerSolutions.PrimePairUpper.Search
