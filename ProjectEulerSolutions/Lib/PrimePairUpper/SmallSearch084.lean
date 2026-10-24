import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch083
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch084

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan461 : scan edge (search edge 4) 5 [] 0 tail461 best = best := by
  have cons : tail461 = 3301 :: tail462 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates461_checked]
  change scan edge (search edge 4) 5 [] 0 tail462
    (if 4 ≤ rootCandidates461.length then search edge 4 [3301] rootCandidates461 3301 best else best) = best
  rw [if_pos (by decide), rootBranch461]
  exact outerScan462

set_option maxRecDepth 4096 in
theorem outerScan460 : scan edge (search edge 4) 5 [] 0 tail460 best = best := by
  have cons : tail460 = 3299 :: tail461 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates460_checked]
  change scan edge (search edge 4) 5 [] 0 tail461
    (if 4 ≤ rootCandidates460.length then search edge 4 [3299] rootCandidates460 3299 best else best) = best
  rw [if_pos (by decide), rootBranch460]
  exact outerScan461

set_option maxRecDepth 4096 in
theorem outerScan459 : scan edge (search edge 4) 5 [] 0 tail459 best = best := by
  have cons : tail459 = 3271 :: tail460 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates459_checked]
  change scan edge (search edge 4) 5 [] 0 tail460
    (if 4 ≤ rootCandidates459.length then search edge 4 [3271] rootCandidates459 3271 best else best) = best
  rw [if_pos (by decide), rootBranch459]
  exact outerScan460

set_option maxRecDepth 4096 in
theorem outerScan458 : scan edge (search edge 4) 5 [] 0 tail458 best = best := by
  have cons : tail458 = 3259 :: tail459 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates458_checked]
  change scan edge (search edge 4) 5 [] 0 tail459
    (if 4 ≤ rootCandidates458.length then search edge 4 [3259] rootCandidates458 3259 best else best) = best
  rw [if_pos (by decide), rootBranch458]
  exact outerScan459

set_option maxRecDepth 4096 in
theorem outerScan457 : scan edge (search edge 4) 5 [] 0 tail457 best = best := by
  have cons : tail457 = 3257 :: tail458 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates457_checked]
  change scan edge (search edge 4) 5 [] 0 tail458
    (if 4 ≤ rootCandidates457.length then search edge 4 [3257] rootCandidates457 3257 best else best) = best
  rw [if_pos (by decide), rootBranch457]
  exact outerScan458

end ProjectEulerSolutions.PrimePairUpper.Search
