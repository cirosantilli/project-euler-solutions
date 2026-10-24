import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch019
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch020

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan887 : scan edge (search edge 4) 5 [] 0 tail887 best = best := by
  have cons : tail887 = 6917 :: tail888 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates887_checked]
  change scan edge (search edge 4) 5 [] 0 tail888
    (if 4 ≤ rootCandidates887.length then search edge 4 [6917] rootCandidates887 6917 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan888

set_option maxRecDepth 4096 in
theorem outerScan886 : scan edge (search edge 4) 5 [] 0 tail886 best = best := by
  have cons : tail886 = 6911 :: tail887 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates886_checked]
  change scan edge (search edge 4) 5 [] 0 tail887
    (if 4 ≤ rootCandidates886.length then search edge 4 [6911] rootCandidates886 6911 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan887

set_option maxRecDepth 4096 in
theorem outerScan885 : scan edge (search edge 4) 5 [] 0 tail885 best = best := by
  have cons : tail885 = 6907 :: tail886 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates885_checked]
  change scan edge (search edge 4) 5 [] 0 tail886
    (if 4 ≤ rootCandidates885.length then search edge 4 [6907] rootCandidates885 6907 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan886

set_option maxRecDepth 4096 in
theorem outerScan884 : scan edge (search edge 4) 5 [] 0 tail884 best = best := by
  have cons : tail884 = 6899 :: tail885 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates884_checked]
  change scan edge (search edge 4) 5 [] 0 tail885
    (if 4 ≤ rootCandidates884.length then search edge 4 [6899] rootCandidates884 6899 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan885

set_option maxRecDepth 4096 in
theorem outerScan883 : scan edge (search edge 4) 5 [] 0 tail883 best = best := by
  have cons : tail883 = 6883 :: tail884 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates883_checked]
  change scan edge (search edge 4) 5 [] 0 tail884
    (if 4 ≤ rootCandidates883.length then search edge 4 [6883] rootCandidates883 6883 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan884

set_option maxRecDepth 4096 in
theorem outerScan882 : scan edge (search edge 4) 5 [] 0 tail882 best = best := by
  have cons : tail882 = 6871 :: tail883 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates882_checked]
  change scan edge (search edge 4) 5 [] 0 tail883
    (if 4 ≤ rootCandidates882.length then search edge 4 [6871] rootCandidates882 6871 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan883

set_option maxRecDepth 4096 in
theorem outerScan881 : scan edge (search edge 4) 5 [] 0 tail881 best = best := by
  have cons : tail881 = 6869 :: tail882 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates881_checked]
  change scan edge (search edge 4) 5 [] 0 tail882
    (if 4 ≤ rootCandidates881.length then search edge 4 [6869] rootCandidates881 6869 best else best) = best
  rw [if_pos (by decide), rootBranch881]
  exact outerScan882

set_option maxRecDepth 4096 in
theorem outerScan880 : scan edge (search edge 4) 5 [] 0 tail880 best = best := by
  have cons : tail880 = 6863 :: tail881 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates880_checked]
  change scan edge (search edge 4) 5 [] 0 tail881
    (if 4 ≤ rootCandidates880.length then search edge 4 [6863] rootCandidates880 6863 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan881

end ProjectEulerSolutions.PrimePairUpper.Search
