import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch047
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch048

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan667 : scan edge (search edge 4) 5 [] 0 tail667 best = best := by
  have cons : tail667 = 5003 :: tail668 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates667_checked]
  change scan edge (search edge 4) 5 [] 0 tail668
    (if 4 ≤ rootCandidates667.length then search edge 4 [5003] rootCandidates667 5003 best else best) = best
  rw [if_pos (by decide), rootBranch667]
  exact outerScan668

set_option maxRecDepth 4096 in
theorem outerScan666 : scan edge (search edge 4) 5 [] 0 tail666 best = best := by
  have cons : tail666 = 4999 :: tail667 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates666_checked]
  change scan edge (search edge 4) 5 [] 0 tail667
    (if 4 ≤ rootCandidates666.length then search edge 4 [4999] rootCandidates666 4999 best else best) = best
  rw [if_pos (by decide), rootBranch666]
  exact outerScan667

set_option maxRecDepth 4096 in
theorem outerScan665 : scan edge (search edge 4) 5 [] 0 tail665 best = best := by
  have cons : tail665 = 4993 :: tail666 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates665_checked]
  change scan edge (search edge 4) 5 [] 0 tail666
    (if 4 ≤ rootCandidates665.length then search edge 4 [4993] rootCandidates665 4993 best else best) = best
  rw [if_pos (by decide), rootBranch665]
  exact outerScan666

set_option maxRecDepth 4096 in
theorem outerScan664 : scan edge (search edge 4) 5 [] 0 tail664 best = best := by
  have cons : tail664 = 4987 :: tail665 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates664_checked]
  change scan edge (search edge 4) 5 [] 0 tail665
    (if 4 ≤ rootCandidates664.length then search edge 4 [4987] rootCandidates664 4987 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan665

set_option maxRecDepth 4096 in
theorem outerScan663 : scan edge (search edge 4) 5 [] 0 tail663 best = best := by
  have cons : tail663 = 4973 :: tail664 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates663_checked]
  change scan edge (search edge 4) 5 [] 0 tail664
    (if 4 ≤ rootCandidates663.length then search edge 4 [4973] rootCandidates663 4973 best else best) = best
  rw [if_pos (by decide), rootBranch663]
  exact outerScan664

set_option maxRecDepth 4096 in
theorem outerScan662 : scan edge (search edge 4) 5 [] 0 tail662 best = best := by
  have cons : tail662 = 4969 :: tail663 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates662_checked]
  change scan edge (search edge 4) 5 [] 0 tail663
    (if 4 ≤ rootCandidates662.length then search edge 4 [4969] rootCandidates662 4969 best else best) = best
  rw [if_pos (by decide), rootBranch662]
  exact outerScan663

set_option maxRecDepth 4096 in
theorem outerScan661 : scan edge (search edge 4) 5 [] 0 tail661 best = best := by
  have cons : tail661 = 4967 :: tail662 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates661_checked]
  change scan edge (search edge 4) 5 [] 0 tail662
    (if 4 ≤ rootCandidates661.length then search edge 4 [4967] rootCandidates661 4967 best else best) = best
  rw [if_pos (by decide), rootBranch661]
  exact outerScan662

set_option maxRecDepth 4096 in
theorem outerScan660 : scan edge (search edge 4) 5 [] 0 tail660 best = best := by
  have cons : tail660 = 4957 :: tail661 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates660_checked]
  change scan edge (search edge 4) 5 [] 0 tail661
    (if 4 ≤ rootCandidates660.length then search edge 4 [4957] rootCandidates660 4957 best else best) = best
  rw [if_pos (by decide), rootBranch660]
  exact outerScan661

end ProjectEulerSolutions.PrimePairUpper.Search
