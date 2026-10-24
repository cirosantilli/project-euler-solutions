import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch054
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch055

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan618 : scan edge (search edge 4) 5 [] 0 tail618 best = best := by
  have cons : tail618 = 4591 :: tail619 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates618_checked]
  change scan edge (search edge 4) 5 [] 0 tail619
    (if 4 ≤ rootCandidates618.length then search edge 4 [4591] rootCandidates618 4591 best else best) = best
  rw [if_pos (by decide), rootBranch618]
  exact outerScan619

set_option maxRecDepth 4096 in
theorem outerScan617 : scan edge (search edge 4) 5 [] 0 tail617 best = best := by
  have cons : tail617 = 4583 :: tail618 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates617_checked]
  change scan edge (search edge 4) 5 [] 0 tail618
    (if 4 ≤ rootCandidates617.length then search edge 4 [4583] rootCandidates617 4583 best else best) = best
  rw [if_pos (by decide), rootBranch617]
  exact outerScan618

set_option maxRecDepth 4096 in
theorem outerScan616 : scan edge (search edge 4) 5 [] 0 tail616 best = best := by
  have cons : tail616 = 4567 :: tail617 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates616_checked]
  change scan edge (search edge 4) 5 [] 0 tail617
    (if 4 ≤ rootCandidates616.length then search edge 4 [4567] rootCandidates616 4567 best else best) = best
  rw [if_pos (by decide), rootBranch616]
  exact outerScan617

set_option maxRecDepth 4096 in
theorem outerScan615 : scan edge (search edge 4) 5 [] 0 tail615 best = best := by
  have cons : tail615 = 4561 :: tail616 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates615_checked]
  change scan edge (search edge 4) 5 [] 0 tail616
    (if 4 ≤ rootCandidates615.length then search edge 4 [4561] rootCandidates615 4561 best else best) = best
  rw [if_pos (by decide), rootBranch615]
  exact outerScan616

set_option maxRecDepth 4096 in
theorem outerScan614 : scan edge (search edge 4) 5 [] 0 tail614 best = best := by
  have cons : tail614 = 4549 :: tail615 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates614_checked]
  change scan edge (search edge 4) 5 [] 0 tail615
    (if 4 ≤ rootCandidates614.length then search edge 4 [4549] rootCandidates614 4549 best else best) = best
  rw [if_pos (by decide), rootBranch614]
  exact outerScan615

set_option maxRecDepth 4096 in
theorem outerScan613 : scan edge (search edge 4) 5 [] 0 tail613 best = best := by
  have cons : tail613 = 4547 :: tail614 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates613_checked]
  change scan edge (search edge 4) 5 [] 0 tail614
    (if 4 ≤ rootCandidates613.length then search edge 4 [4547] rootCandidates613 4547 best else best) = best
  rw [if_pos (by decide), rootBranch613]
  exact outerScan614

end ProjectEulerSolutions.PrimePairUpper.Search
