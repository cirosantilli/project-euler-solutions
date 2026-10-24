import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch027
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch028

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan823 : scan edge (search edge 4) 5 [] 0 tail823 best = best := by
  have cons : tail823 = 6343 :: tail824 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates823_checked]
  change scan edge (search edge 4) 5 [] 0 tail824
    (if 4 ≤ rootCandidates823.length then search edge 4 [6343] rootCandidates823 6343 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan824

set_option maxRecDepth 4096 in
theorem outerScan822 : scan edge (search edge 4) 5 [] 0 tail822 best = best := by
  have cons : tail822 = 6337 :: tail823 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates822_checked]
  change scan edge (search edge 4) 5 [] 0 tail823
    (if 4 ≤ rootCandidates822.length then search edge 4 [6337] rootCandidates822 6337 best else best) = best
  rw [if_pos (by decide), rootBranch822]
  exact outerScan823

set_option maxRecDepth 4096 in
theorem outerScan821 : scan edge (search edge 4) 5 [] 0 tail821 best = best := by
  have cons : tail821 = 6329 :: tail822 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates821_checked]
  change scan edge (search edge 4) 5 [] 0 tail822
    (if 4 ≤ rootCandidates821.length then search edge 4 [6329] rootCandidates821 6329 best else best) = best
  rw [if_pos (by decide), rootBranch821]
  exact outerScan822

set_option maxRecDepth 4096 in
theorem outerScan820 : scan edge (search edge 4) 5 [] 0 tail820 best = best := by
  have cons : tail820 = 6323 :: tail821 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates820_checked]
  change scan edge (search edge 4) 5 [] 0 tail821
    (if 4 ≤ rootCandidates820.length then search edge 4 [6323] rootCandidates820 6323 best else best) = best
  rw [if_pos (by decide), rootBranch820]
  exact outerScan821

set_option maxRecDepth 4096 in
theorem outerScan819 : scan edge (search edge 4) 5 [] 0 tail819 best = best := by
  have cons : tail819 = 6317 :: tail820 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates819_checked]
  change scan edge (search edge 4) 5 [] 0 tail820
    (if 4 ≤ rootCandidates819.length then search edge 4 [6317] rootCandidates819 6317 best else best) = best
  rw [if_pos (by decide), rootBranch819]
  exact outerScan820

set_option maxRecDepth 4096 in
theorem outerScan818 : scan edge (search edge 4) 5 [] 0 tail818 best = best := by
  have cons : tail818 = 6311 :: tail819 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates818_checked]
  change scan edge (search edge 4) 5 [] 0 tail819
    (if 4 ≤ rootCandidates818.length then search edge 4 [6311] rootCandidates818 6311 best else best) = best
  rw [if_pos (by decide), rootBranch818]
  exact outerScan819

set_option maxRecDepth 4096 in
theorem outerScan817 : scan edge (search edge 4) 5 [] 0 tail817 best = best := by
  have cons : tail817 = 6301 :: tail818 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates817_checked]
  change scan edge (search edge 4) 5 [] 0 tail818
    (if 4 ≤ rootCandidates817.length then search edge 4 [6301] rootCandidates817 6301 best else best) = best
  rw [if_pos (by decide), rootBranch817]
  exact outerScan818

set_option maxRecDepth 4096 in
theorem outerScan816 : scan edge (search edge 4) 5 [] 0 tail816 best = best := by
  have cons : tail816 = 6299 :: tail817 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates816_checked]
  change scan edge (search edge 4) 5 [] 0 tail817
    (if 4 ≤ rootCandidates816.length then search edge 4 [6299] rootCandidates816 6299 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan817

end ProjectEulerSolutions.PrimePairUpper.Search
