import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch036
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch037

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan751 : scan edge (search edge 4) 5 [] 0 tail751 best = best := by
  have cons : tail751 = 5737 :: tail752 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates751_checked]
  change scan edge (search edge 4) 5 [] 0 tail752
    (if 4 ≤ rootCandidates751.length then search edge 4 [5737] rootCandidates751 5737 best else best) = best
  rw [if_pos (by decide), rootBranch751]
  exact outerScan752

set_option maxRecDepth 4096 in
theorem outerScan750 : scan edge (search edge 4) 5 [] 0 tail750 best = best := by
  have cons : tail750 = 5717 :: tail751 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates750_checked]
  change scan edge (search edge 4) 5 [] 0 tail751
    (if 4 ≤ rootCandidates750.length then search edge 4 [5717] rootCandidates750 5717 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan751

set_option maxRecDepth 4096 in
theorem outerScan749 : scan edge (search edge 4) 5 [] 0 tail749 best = best := by
  have cons : tail749 = 5711 :: tail750 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates749_checked]
  change scan edge (search edge 4) 5 [] 0 tail750
    (if 4 ≤ rootCandidates749.length then search edge 4 [5711] rootCandidates749 5711 best else best) = best
  rw [if_pos (by decide), rootBranch749]
  exact outerScan750

set_option maxRecDepth 4096 in
theorem outerScan748 : scan edge (search edge 4) 5 [] 0 tail748 best = best := by
  have cons : tail748 = 5701 :: tail749 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates748_checked]
  change scan edge (search edge 4) 5 [] 0 tail749
    (if 4 ≤ rootCandidates748.length then search edge 4 [5701] rootCandidates748 5701 best else best) = best
  rw [if_pos (by decide), rootBranch748]
  exact outerScan749

set_option maxRecDepth 4096 in
theorem outerScan747 : scan edge (search edge 4) 5 [] 0 tail747 best = best := by
  have cons : tail747 = 5693 :: tail748 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates747_checked]
  change scan edge (search edge 4) 5 [] 0 tail748
    (if 4 ≤ rootCandidates747.length then search edge 4 [5693] rootCandidates747 5693 best else best) = best
  rw [if_pos (by decide), rootBranch747]
  exact outerScan748

set_option maxRecDepth 4096 in
theorem outerScan746 : scan edge (search edge 4) 5 [] 0 tail746 best = best := by
  have cons : tail746 = 5689 :: tail747 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates746_checked]
  change scan edge (search edge 4) 5 [] 0 tail747
    (if 4 ≤ rootCandidates746.length then search edge 4 [5689] rootCandidates746 5689 best else best) = best
  rw [if_pos (by decide), rootBranch746]
  exact outerScan747

set_option maxRecDepth 4096 in
theorem outerScan745 : scan edge (search edge 4) 5 [] 0 tail745 best = best := by
  have cons : tail745 = 5683 :: tail746 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates745_checked]
  change scan edge (search edge 4) 5 [] 0 tail746
    (if 4 ≤ rootCandidates745.length then search edge 4 [5683] rootCandidates745 5683 best else best) = best
  rw [if_pos (by decide), rootBranch745]
  exact outerScan746

set_option maxRecDepth 4096 in
theorem outerScan744 : scan edge (search edge 4) 5 [] 0 tail744 best = best := by
  have cons : tail744 = 5669 :: tail745 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates744_checked]
  change scan edge (search edge 4) 5 [] 0 tail745
    (if 4 ≤ rootCandidates744.length then search edge 4 [5669] rootCandidates744 5669 best else best) = best
  rw [if_pos (by decide), rootBranch744]
  exact outerScan745

end ProjectEulerSolutions.PrimePairUpper.Search
