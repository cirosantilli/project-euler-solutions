import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch071
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch072

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan520 : scan edge (search edge 4) 5 [] 0 tail520 best = best := by
  have cons : tail520 = 3761 :: tail521 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates520_checked]
  change scan edge (search edge 4) 5 [] 0 tail521
    (if 4 ≤ rootCandidates520.length then search edge 4 [3761] rootCandidates520 3761 best else best) = best
  rw [if_pos (by decide), rootBranch520]
  exact outerScan521

set_option maxRecDepth 4096 in
theorem outerScan519 : scan edge (search edge 4) 5 [] 0 tail519 best = best := by
  have cons : tail519 = 3739 :: tail520 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates519_checked]
  change scan edge (search edge 4) 5 [] 0 tail520
    (if 4 ≤ rootCandidates519.length then search edge 4 [3739] rootCandidates519 3739 best else best) = best
  rw [if_pos (by decide), rootBranch519]
  exact outerScan520

set_option maxRecDepth 4096 in
theorem outerScan518 : scan edge (search edge 4) 5 [] 0 tail518 best = best := by
  have cons : tail518 = 3733 :: tail519 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates518_checked]
  change scan edge (search edge 4) 5 [] 0 tail519
    (if 4 ≤ rootCandidates518.length then search edge 4 [3733] rootCandidates518 3733 best else best) = best
  rw [if_pos (by decide), rootBranch518]
  exact outerScan519

set_option maxRecDepth 4096 in
theorem outerScan517 : scan edge (search edge 4) 5 [] 0 tail517 best = best := by
  have cons : tail517 = 3727 :: tail518 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates517_checked]
  change scan edge (search edge 4) 5 [] 0 tail518
    (if 4 ≤ rootCandidates517.length then search edge 4 [3727] rootCandidates517 3727 best else best) = best
  rw [if_pos (by decide), rootBranch517]
  exact outerScan518

set_option maxRecDepth 4096 in
theorem outerScan516 : scan edge (search edge 4) 5 [] 0 tail516 best = best := by
  have cons : tail516 = 3719 :: tail517 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates516_checked]
  change scan edge (search edge 4) 5 [] 0 tail517
    (if 4 ≤ rootCandidates516.length then search edge 4 [3719] rootCandidates516 3719 best else best) = best
  rw [if_pos (by decide), rootBranch516]
  exact outerScan517

set_option maxRecDepth 4096 in
theorem outerScan515 : scan edge (search edge 4) 5 [] 0 tail515 best = best := by
  have cons : tail515 = 3709 :: tail516 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates515_checked]
  change scan edge (search edge 4) 5 [] 0 tail516
    (if 4 ≤ rootCandidates515.length then search edge 4 [3709] rootCandidates515 3709 best else best) = best
  rw [if_pos (by decide), rootBranch515]
  exact outerScan516

set_option maxRecDepth 4096 in
theorem outerScan514 : scan edge (search edge 4) 5 [] 0 tail514 best = best := by
  have cons : tail514 = 3701 :: tail515 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates514_checked]
  change scan edge (search edge 4) 5 [] 0 tail515
    (if 4 ≤ rootCandidates514.length then search edge 4 [3701] rootCandidates514 3701 best else best) = best
  rw [if_pos (by decide), rootBranch514]
  exact outerScan515

end ProjectEulerSolutions.PrimePairUpper.Search
