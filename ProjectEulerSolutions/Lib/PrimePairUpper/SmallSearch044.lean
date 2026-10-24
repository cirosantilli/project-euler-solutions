import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch043
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch044

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan697 : scan edge (search edge 4) 5 [] 0 tail697 best = best := by
  have cons : tail697 = 5279 :: tail698 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates697_checked]
  change scan edge (search edge 4) 5 [] 0 tail698
    (if 4 ≤ rootCandidates697.length then search edge 4 [5279] rootCandidates697 5279 best else best) = best
  rw [if_pos (by decide), rootBranch697]
  exact outerScan698

set_option maxRecDepth 4096 in
theorem outerScan696 : scan edge (search edge 4) 5 [] 0 tail696 best = best := by
  have cons : tail696 = 5273 :: tail697 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates696_checked]
  change scan edge (search edge 4) 5 [] 0 tail697
    (if 4 ≤ rootCandidates696.length then search edge 4 [5273] rootCandidates696 5273 best else best) = best
  rw [if_pos (by decide), rootBranch696]
  exact outerScan697

set_option maxRecDepth 4096 in
theorem outerScan695 : scan edge (search edge 4) 5 [] 0 tail695 best = best := by
  have cons : tail695 = 5261 :: tail696 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates695_checked]
  change scan edge (search edge 4) 5 [] 0 tail696
    (if 4 ≤ rootCandidates695.length then search edge 4 [5261] rootCandidates695 5261 best else best) = best
  rw [if_pos (by decide), rootBranch695]
  exact outerScan696

set_option maxRecDepth 4096 in
theorem outerScan694 : scan edge (search edge 4) 5 [] 0 tail694 best = best := by
  have cons : tail694 = 5237 :: tail695 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates694_checked]
  change scan edge (search edge 4) 5 [] 0 tail695
    (if 4 ≤ rootCandidates694.length then search edge 4 [5237] rootCandidates694 5237 best else best) = best
  rw [if_pos (by decide), rootBranch694]
  exact outerScan695

set_option maxRecDepth 4096 in
theorem outerScan693 : scan edge (search edge 4) 5 [] 0 tail693 best = best := by
  have cons : tail693 = 5233 :: tail694 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates693_checked]
  change scan edge (search edge 4) 5 [] 0 tail694
    (if 4 ≤ rootCandidates693.length then search edge 4 [5233] rootCandidates693 5233 best else best) = best
  rw [if_pos (by decide), rootBranch693]
  exact outerScan694

set_option maxRecDepth 4096 in
theorem outerScan692 : scan edge (search edge 4) 5 [] 0 tail692 best = best := by
  have cons : tail692 = 5231 :: tail693 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates692_checked]
  change scan edge (search edge 4) 5 [] 0 tail693
    (if 4 ≤ rootCandidates692.length then search edge 4 [5231] rootCandidates692 5231 best else best) = best
  rw [if_pos (by decide), rootBranch692]
  exact outerScan693

set_option maxRecDepth 4096 in
theorem outerScan691 : scan edge (search edge 4) 5 [] 0 tail691 best = best := by
  have cons : tail691 = 5227 :: tail692 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates691_checked]
  change scan edge (search edge 4) 5 [] 0 tail692
    (if 4 ≤ rootCandidates691.length then search edge 4 [5227] rootCandidates691 5227 best else best) = best
  rw [if_pos (by decide), rootBranch691]
  exact outerScan692

set_option maxRecDepth 4096 in
theorem outerScan690 : scan edge (search edge 4) 5 [] 0 tail690 best = best := by
  have cons : tail690 = 5209 :: tail691 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates690_checked]
  change scan edge (search edge 4) 5 [] 0 tail691
    (if 4 ≤ rootCandidates690.length then search edge 4 [5209] rootCandidates690 5209 best else best) = best
  rw [if_pos (by decide), rootBranch690]
  exact outerScan691

end ProjectEulerSolutions.PrimePairUpper.Search
