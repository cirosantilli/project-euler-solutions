import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch025
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch026

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan839 : scan edge (search edge 4) 5 [] 0 tail839 best = best := by
  have cons : tail839 = 6491 :: tail840 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates839_checked]
  change scan edge (search edge 4) 5 [] 0 tail840
    (if 4 ≤ rootCandidates839.length then search edge 4 [6491] rootCandidates839 6491 best else best) = best
  rw [if_pos (by decide), rootBranch839]
  exact outerScan840

set_option maxRecDepth 4096 in
theorem outerScan838 : scan edge (search edge 4) 5 [] 0 tail838 best = best := by
  have cons : tail838 = 6481 :: tail839 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates838_checked]
  change scan edge (search edge 4) 5 [] 0 tail839
    (if 4 ≤ rootCandidates838.length then search edge 4 [6481] rootCandidates838 6481 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan839

set_option maxRecDepth 4096 in
theorem outerScan837 : scan edge (search edge 4) 5 [] 0 tail837 best = best := by
  have cons : tail837 = 6473 :: tail838 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates837_checked]
  change scan edge (search edge 4) 5 [] 0 tail838
    (if 4 ≤ rootCandidates837.length then search edge 4 [6473] rootCandidates837 6473 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan838

set_option maxRecDepth 4096 in
theorem outerScan836 : scan edge (search edge 4) 5 [] 0 tail836 best = best := by
  have cons : tail836 = 6469 :: tail837 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates836_checked]
  change scan edge (search edge 4) 5 [] 0 tail837
    (if 4 ≤ rootCandidates836.length then search edge 4 [6469] rootCandidates836 6469 best else best) = best
  rw [if_pos (by decide), rootBranch836]
  exact outerScan837

set_option maxRecDepth 4096 in
theorem outerScan835 : scan edge (search edge 4) 5 [] 0 tail835 best = best := by
  have cons : tail835 = 6451 :: tail836 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates835_checked]
  change scan edge (search edge 4) 5 [] 0 tail836
    (if 4 ≤ rootCandidates835.length then search edge 4 [6451] rootCandidates835 6451 best else best) = best
  rw [if_pos (by decide), rootBranch835]
  exact outerScan836

set_option maxRecDepth 4096 in
theorem outerScan834 : scan edge (search edge 4) 5 [] 0 tail834 best = best := by
  have cons : tail834 = 6449 :: tail835 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates834_checked]
  change scan edge (search edge 4) 5 [] 0 tail835
    (if 4 ≤ rootCandidates834.length then search edge 4 [6449] rootCandidates834 6449 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan835

set_option maxRecDepth 4096 in
theorem outerScan833 : scan edge (search edge 4) 5 [] 0 tail833 best = best := by
  have cons : tail833 = 6427 :: tail834 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates833_checked]
  change scan edge (search edge 4) 5 [] 0 tail834
    (if 4 ≤ rootCandidates833.length then search edge 4 [6427] rootCandidates833 6427 best else best) = best
  rw [if_pos (by decide), rootBranch833]
  exact outerScan834

set_option maxRecDepth 4096 in
theorem outerScan832 : scan edge (search edge 4) 5 [] 0 tail832 best = best := by
  have cons : tail832 = 6421 :: tail833 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates832_checked]
  change scan edge (search edge 4) 5 [] 0 tail833
    (if 4 ≤ rootCandidates832.length then search edge 4 [6421] rootCandidates832 6421 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan833

end ProjectEulerSolutions.PrimePairUpper.Search
