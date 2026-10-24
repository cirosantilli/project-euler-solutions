import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch048
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch049

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan659 : scan edge (search edge 4) 5 [] 0 tail659 best = best := by
  have cons : tail659 = 4951 :: tail660 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates659_checked]
  change scan edge (search edge 4) 5 [] 0 tail660
    (if 4 ≤ rootCandidates659.length then search edge 4 [4951] rootCandidates659 4951 best else best) = best
  rw [if_pos (by decide), rootBranch659]
  exact outerScan660

set_option maxRecDepth 4096 in
theorem outerScan658 : scan edge (search edge 4) 5 [] 0 tail658 best = best := by
  have cons : tail658 = 4943 :: tail659 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates658_checked]
  change scan edge (search edge 4) 5 [] 0 tail659
    (if 4 ≤ rootCandidates658.length then search edge 4 [4943] rootCandidates658 4943 best else best) = best
  rw [if_pos (by decide), rootBranch658]
  exact outerScan659

set_option maxRecDepth 4096 in
theorem outerScan657 : scan edge (search edge 4) 5 [] 0 tail657 best = best := by
  have cons : tail657 = 4937 :: tail658 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates657_checked]
  change scan edge (search edge 4) 5 [] 0 tail658
    (if 4 ≤ rootCandidates657.length then search edge 4 [4937] rootCandidates657 4937 best else best) = best
  rw [if_pos (by decide), rootBranch657]
  exact outerScan658

set_option maxRecDepth 4096 in
theorem outerScan656 : scan edge (search edge 4) 5 [] 0 tail656 best = best := by
  have cons : tail656 = 4933 :: tail657 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates656_checked]
  change scan edge (search edge 4) 5 [] 0 tail657
    (if 4 ≤ rootCandidates656.length then search edge 4 [4933] rootCandidates656 4933 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan657

set_option maxRecDepth 4096 in
theorem outerScan655 : scan edge (search edge 4) 5 [] 0 tail655 best = best := by
  have cons : tail655 = 4931 :: tail656 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates655_checked]
  change scan edge (search edge 4) 5 [] 0 tail656
    (if 4 ≤ rootCandidates655.length then search edge 4 [4931] rootCandidates655 4931 best else best) = best
  rw [if_pos (by decide), rootBranch655]
  exact outerScan656

set_option maxRecDepth 4096 in
theorem outerScan654 : scan edge (search edge 4) 5 [] 0 tail654 best = best := by
  have cons : tail654 = 4919 :: tail655 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates654_checked]
  change scan edge (search edge 4) 5 [] 0 tail655
    (if 4 ≤ rootCandidates654.length then search edge 4 [4919] rootCandidates654 4919 best else best) = best
  rw [if_pos (by decide), rootBranch654]
  exact outerScan655

set_option maxRecDepth 4096 in
theorem outerScan653 : scan edge (search edge 4) 5 [] 0 tail653 best = best := by
  have cons : tail653 = 4909 :: tail654 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates653_checked]
  change scan edge (search edge 4) 5 [] 0 tail654
    (if 4 ≤ rootCandidates653.length then search edge 4 [4909] rootCandidates653 4909 best else best) = best
  rw [if_pos (by decide), rootBranch653]
  exact outerScan654

end ProjectEulerSolutions.PrimePairUpper.Search
