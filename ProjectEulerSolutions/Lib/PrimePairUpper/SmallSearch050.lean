import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch049
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch050

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan652 : scan edge (search edge 4) 5 [] 0 tail652 best = best := by
  have cons : tail652 = 4903 :: tail653 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates652_checked]
  change scan edge (search edge 4) 5 [] 0 tail653
    (if 4 ≤ rootCandidates652.length then search edge 4 [4903] rootCandidates652 4903 best else best) = best
  rw [if_pos (by decide), rootBranch652]
  exact outerScan653

set_option maxRecDepth 4096 in
theorem outerScan651 : scan edge (search edge 4) 5 [] 0 tail651 best = best := by
  have cons : tail651 = 4889 :: tail652 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates651_checked]
  change scan edge (search edge 4) 5 [] 0 tail652
    (if 4 ≤ rootCandidates651.length then search edge 4 [4889] rootCandidates651 4889 best else best) = best
  rw [if_pos (by decide), rootBranch651]
  exact outerScan652

set_option maxRecDepth 4096 in
theorem outerScan650 : scan edge (search edge 4) 5 [] 0 tail650 best = best := by
  have cons : tail650 = 4877 :: tail651 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates650_checked]
  change scan edge (search edge 4) 5 [] 0 tail651
    (if 4 ≤ rootCandidates650.length then search edge 4 [4877] rootCandidates650 4877 best else best) = best
  rw [if_pos (by decide), rootBranch650]
  exact outerScan651

set_option maxRecDepth 4096 in
theorem outerScan649 : scan edge (search edge 4) 5 [] 0 tail649 best = best := by
  have cons : tail649 = 4871 :: tail650 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates649_checked]
  change scan edge (search edge 4) 5 [] 0 tail650
    (if 4 ≤ rootCandidates649.length then search edge 4 [4871] rootCandidates649 4871 best else best) = best
  rw [if_pos (by decide), rootBranch649]
  exact outerScan650

set_option maxRecDepth 4096 in
theorem outerScan648 : scan edge (search edge 4) 5 [] 0 tail648 best = best := by
  have cons : tail648 = 4861 :: tail649 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates648_checked]
  change scan edge (search edge 4) 5 [] 0 tail649
    (if 4 ≤ rootCandidates648.length then search edge 4 [4861] rootCandidates648 4861 best else best) = best
  rw [if_pos (by decide), rootBranch648]
  exact outerScan649

set_option maxRecDepth 4096 in
theorem outerScan647 : scan edge (search edge 4) 5 [] 0 tail647 best = best := by
  have cons : tail647 = 4831 :: tail648 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates647_checked]
  change scan edge (search edge 4) 5 [] 0 tail648
    (if 4 ≤ rootCandidates647.length then search edge 4 [4831] rootCandidates647 4831 best else best) = best
  rw [if_pos (by decide), rootBranch647]
  exact outerScan648

end ProjectEulerSolutions.PrimePairUpper.Search
