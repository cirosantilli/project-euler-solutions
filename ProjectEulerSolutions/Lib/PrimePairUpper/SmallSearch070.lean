import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch069
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch070

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan530 : scan edge (search edge 4) 5 [] 0 tail530 best = best := by
  have cons : tail530 = 3847 :: tail531 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates530_checked]
  change scan edge (search edge 4) 5 [] 0 tail531
    (if 4 ≤ rootCandidates530.length then search edge 4 [3847] rootCandidates530 3847 best else best) = best
  rw [if_pos (by decide), rootBranch530]
  exact outerScan531

set_option maxRecDepth 4096 in
theorem outerScan529 : scan edge (search edge 4) 5 [] 0 tail529 best = best := by
  have cons : tail529 = 3833 :: tail530 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates529_checked]
  change scan edge (search edge 4) 5 [] 0 tail530
    (if 4 ≤ rootCandidates529.length then search edge 4 [3833] rootCandidates529 3833 best else best) = best
  rw [if_pos (by decide), rootBranch529]
  exact outerScan530

set_option maxRecDepth 4096 in
theorem outerScan528 : scan edge (search edge 4) 5 [] 0 tail528 best = best := by
  have cons : tail528 = 3823 :: tail529 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates528_checked]
  change scan edge (search edge 4) 5 [] 0 tail529
    (if 4 ≤ rootCandidates528.length then search edge 4 [3823] rootCandidates528 3823 best else best) = best
  rw [if_pos (by decide), rootBranch528]
  exact outerScan529

set_option maxRecDepth 4096 in
theorem outerScan527 : scan edge (search edge 4) 5 [] 0 tail527 best = best := by
  have cons : tail527 = 3821 :: tail528 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates527_checked]
  change scan edge (search edge 4) 5 [] 0 tail528
    (if 4 ≤ rootCandidates527.length then search edge 4 [3821] rootCandidates527 3821 best else best) = best
  rw [if_pos (by decide), rootBranch527]
  exact outerScan528

set_option maxRecDepth 4096 in
theorem outerScan526 : scan edge (search edge 4) 5 [] 0 tail526 best = best := by
  have cons : tail526 = 3803 :: tail527 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates526_checked]
  change scan edge (search edge 4) 5 [] 0 tail527
    (if 4 ≤ rootCandidates526.length then search edge 4 [3803] rootCandidates526 3803 best else best) = best
  rw [if_pos (by decide), rootBranch526]
  exact outerScan527

end ProjectEulerSolutions.PrimePairUpper.Search
