import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch051
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch052

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan639 : scan edge (search edge 4) 5 [] 0 tail639 best = best := by
  have cons : tail639 = 4783 :: tail640 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates639_checked]
  change scan edge (search edge 4) 5 [] 0 tail640
    (if 4 ≤ rootCandidates639.length then search edge 4 [4783] rootCandidates639 4783 best else best) = best
  rw [if_pos (by decide), rootBranch639]
  exact outerScan640

set_option maxRecDepth 4096 in
theorem outerScan638 : scan edge (search edge 4) 5 [] 0 tail638 best = best := by
  have cons : tail638 = 4759 :: tail639 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates638_checked]
  change scan edge (search edge 4) 5 [] 0 tail639
    (if 4 ≤ rootCandidates638.length then search edge 4 [4759] rootCandidates638 4759 best else best) = best
  rw [if_pos (by decide), rootBranch638]
  exact outerScan639

set_option maxRecDepth 4096 in
theorem outerScan637 : scan edge (search edge 4) 5 [] 0 tail637 best = best := by
  have cons : tail637 = 4751 :: tail638 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates637_checked]
  change scan edge (search edge 4) 5 [] 0 tail638
    (if 4 ≤ rootCandidates637.length then search edge 4 [4751] rootCandidates637 4751 best else best) = best
  rw [if_pos (by decide), rootBranch637]
  exact outerScan638

set_option maxRecDepth 4096 in
theorem outerScan636 : scan edge (search edge 4) 5 [] 0 tail636 best = best := by
  have cons : tail636 = 4733 :: tail637 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates636_checked]
  change scan edge (search edge 4) 5 [] 0 tail637
    (if 4 ≤ rootCandidates636.length then search edge 4 [4733] rootCandidates636 4733 best else best) = best
  rw [if_pos (by decide), rootBranch636]
  exact outerScan637

set_option maxRecDepth 4096 in
theorem outerScan635 : scan edge (search edge 4) 5 [] 0 tail635 best = best := by
  have cons : tail635 = 4729 :: tail636 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates635_checked]
  change scan edge (search edge 4) 5 [] 0 tail636
    (if 4 ≤ rootCandidates635.length then search edge 4 [4729] rootCandidates635 4729 best else best) = best
  rw [if_pos (by decide), rootBranch635]
  exact outerScan636

set_option maxRecDepth 4096 in
theorem outerScan634 : scan edge (search edge 4) 5 [] 0 tail634 best = best := by
  have cons : tail634 = 4723 :: tail635 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates634_checked]
  change scan edge (search edge 4) 5 [] 0 tail635
    (if 4 ≤ rootCandidates634.length then search edge 4 [4723] rootCandidates634 4723 best else best) = best
  rw [if_pos (by decide), rootBranch634]
  exact outerScan635

set_option maxRecDepth 4096 in
theorem outerScan633 : scan edge (search edge 4) 5 [] 0 tail633 best = best := by
  have cons : tail633 = 4721 :: tail634 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates633_checked]
  change scan edge (search edge 4) 5 [] 0 tail634
    (if 4 ≤ rootCandidates633.length then search edge 4 [4721] rootCandidates633 4721 best else best) = best
  rw [if_pos (by decide), rootBranch633]
  exact outerScan634

end ProjectEulerSolutions.PrimePairUpper.Search
