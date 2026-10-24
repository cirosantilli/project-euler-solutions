import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch033
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch034

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan775 : scan edge (search edge 4) 5 [] 0 tail775 best = best := by
  have cons : tail775 = 5923 :: tail776 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates775_checked]
  change scan edge (search edge 4) 5 [] 0 tail776
    (if 4 ≤ rootCandidates775.length then search edge 4 [5923] rootCandidates775 5923 best else best) = best
  rw [if_pos (by decide), rootBranch775]
  exact outerScan776

set_option maxRecDepth 4096 in
theorem outerScan774 : scan edge (search edge 4) 5 [] 0 tail774 best = best := by
  have cons : tail774 = 5903 :: tail775 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates774_checked]
  change scan edge (search edge 4) 5 [] 0 tail775
    (if 4 ≤ rootCandidates774.length then search edge 4 [5903] rootCandidates774 5903 best else best) = best
  rw [if_pos (by decide), rootBranch774]
  exact outerScan775

set_option maxRecDepth 4096 in
theorem outerScan773 : scan edge (search edge 4) 5 [] 0 tail773 best = best := by
  have cons : tail773 = 5897 :: tail774 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates773_checked]
  change scan edge (search edge 4) 5 [] 0 tail774
    (if 4 ≤ rootCandidates773.length then search edge 4 [5897] rootCandidates773 5897 best else best) = best
  rw [if_pos (by decide), rootBranch773]
  exact outerScan774

set_option maxRecDepth 4096 in
theorem outerScan772 : scan edge (search edge 4) 5 [] 0 tail772 best = best := by
  have cons : tail772 = 5881 :: tail773 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates772_checked]
  change scan edge (search edge 4) 5 [] 0 tail773
    (if 4 ≤ rootCandidates772.length then search edge 4 [5881] rootCandidates772 5881 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan773

set_option maxRecDepth 4096 in
theorem outerScan771 : scan edge (search edge 4) 5 [] 0 tail771 best = best := by
  have cons : tail771 = 5879 :: tail772 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates771_checked]
  change scan edge (search edge 4) 5 [] 0 tail772
    (if 4 ≤ rootCandidates771.length then search edge 4 [5879] rootCandidates771 5879 best else best) = best
  rw [if_pos (by decide), rootBranch771]
  exact outerScan772

set_option maxRecDepth 4096 in
theorem outerScan770 : scan edge (search edge 4) 5 [] 0 tail770 best = best := by
  have cons : tail770 = 5869 :: tail771 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates770_checked]
  change scan edge (search edge 4) 5 [] 0 tail771
    (if 4 ≤ rootCandidates770.length then search edge 4 [5869] rootCandidates770 5869 best else best) = best
  rw [if_pos (by decide), rootBranch770]
  exact outerScan771

set_option maxRecDepth 4096 in
theorem outerScan769 : scan edge (search edge 4) 5 [] 0 tail769 best = best := by
  have cons : tail769 = 5867 :: tail770 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates769_checked]
  change scan edge (search edge 4) 5 [] 0 tail770
    (if 4 ≤ rootCandidates769.length then search edge 4 [5867] rootCandidates769 5867 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan770

set_option maxRecDepth 4096 in
theorem outerScan768 : scan edge (search edge 4) 5 [] 0 tail768 best = best := by
  have cons : tail768 = 5861 :: tail769 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates768_checked]
  change scan edge (search edge 4) 5 [] 0 tail769
    (if 4 ≤ rootCandidates768.length then search edge 4 [5861] rootCandidates768 5861 best else best) = best
  rw [if_pos (by decide), rootBranch768]
  exact outerScan769

end ProjectEulerSolutions.PrimePairUpper.Search
