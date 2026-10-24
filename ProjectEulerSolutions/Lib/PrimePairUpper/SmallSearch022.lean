import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch021
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch022

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan871 : scan edge (search edge 4) 5 [] 0 tail871 best = best := by
  have cons : tail871 = 6791 :: tail872 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates871_checked]
  change scan edge (search edge 4) 5 [] 0 tail872
    (if 4 ≤ rootCandidates871.length then search edge 4 [6791] rootCandidates871 6791 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan872

set_option maxRecDepth 4096 in
theorem outerScan870 : scan edge (search edge 4) 5 [] 0 tail870 best = best := by
  have cons : tail870 = 6781 :: tail871 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates870_checked]
  change scan edge (search edge 4) 5 [] 0 tail871
    (if 4 ≤ rootCandidates870.length then search edge 4 [6781] rootCandidates870 6781 best else best) = best
  rw [if_pos (by decide), rootBranch870]
  exact outerScan871

set_option maxRecDepth 4096 in
theorem outerScan869 : scan edge (search edge 4) 5 [] 0 tail869 best = best := by
  have cons : tail869 = 6779 :: tail870 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates869_checked]
  change scan edge (search edge 4) 5 [] 0 tail870
    (if 4 ≤ rootCandidates869.length then search edge 4 [6779] rootCandidates869 6779 best else best) = best
  rw [if_pos (by decide), rootBranch869]
  exact outerScan870

set_option maxRecDepth 4096 in
theorem outerScan868 : scan edge (search edge 4) 5 [] 0 tail868 best = best := by
  have cons : tail868 = 6763 :: tail869 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates868_checked]
  change scan edge (search edge 4) 5 [] 0 tail869
    (if 4 ≤ rootCandidates868.length then search edge 4 [6763] rootCandidates868 6763 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan869

set_option maxRecDepth 4096 in
theorem outerScan867 : scan edge (search edge 4) 5 [] 0 tail867 best = best := by
  have cons : tail867 = 6761 :: tail868 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates867_checked]
  change scan edge (search edge 4) 5 [] 0 tail868
    (if 4 ≤ rootCandidates867.length then search edge 4 [6761] rootCandidates867 6761 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan868

set_option maxRecDepth 4096 in
theorem outerScan866 : scan edge (search edge 4) 5 [] 0 tail866 best = best := by
  have cons : tail866 = 6737 :: tail867 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates866_checked]
  change scan edge (search edge 4) 5 [] 0 tail867
    (if 4 ≤ rootCandidates866.length then search edge 4 [6737] rootCandidates866 6737 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan867

set_option maxRecDepth 4096 in
theorem outerScan865 : scan edge (search edge 4) 5 [] 0 tail865 best = best := by
  have cons : tail865 = 6733 :: tail866 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates865_checked]
  change scan edge (search edge 4) 5 [] 0 tail866
    (if 4 ≤ rootCandidates865.length then search edge 4 [6733] rootCandidates865 6733 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan866

set_option maxRecDepth 4096 in
theorem outerScan864 : scan edge (search edge 4) 5 [] 0 tail864 best = best := by
  have cons : tail864 = 6719 :: tail865 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates864_checked]
  change scan edge (search edge 4) 5 [] 0 tail865
    (if 4 ≤ rootCandidates864.length then search edge 4 [6719] rootCandidates864 6719 best else best) = best
  rw [if_pos (by decide), rootBranch864]
  exact outerScan865

end ProjectEulerSolutions.PrimePairUpper.Search
