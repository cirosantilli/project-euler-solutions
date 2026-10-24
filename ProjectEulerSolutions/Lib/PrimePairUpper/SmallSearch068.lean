import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch067
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch068

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan540 : scan edge (search edge 4) 5 [] 0 tail540 best = best := by
  have cons : tail540 = 3919 :: tail541 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates540_checked]
  change scan edge (search edge 4) 5 [] 0 tail541
    (if 4 ≤ rootCandidates540.length then search edge 4 [3919] rootCandidates540 3919 best else best) = best
  rw [if_pos (by decide), rootBranch540]
  exact outerScan541

set_option maxRecDepth 4096 in
theorem outerScan539 : scan edge (search edge 4) 5 [] 0 tail539 best = best := by
  have cons : tail539 = 3917 :: tail540 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates539_checked]
  change scan edge (search edge 4) 5 [] 0 tail540
    (if 4 ≤ rootCandidates539.length then search edge 4 [3917] rootCandidates539 3917 best else best) = best
  rw [if_pos (by decide), rootBranch539]
  exact outerScan540

set_option maxRecDepth 4096 in
theorem outerScan538 : scan edge (search edge 4) 5 [] 0 tail538 best = best := by
  have cons : tail538 = 3911 :: tail539 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates538_checked]
  change scan edge (search edge 4) 5 [] 0 tail539
    (if 4 ≤ rootCandidates538.length then search edge 4 [3911] rootCandidates538 3911 best else best) = best
  rw [if_pos (by decide), rootBranch538]
  exact outerScan539

set_option maxRecDepth 4096 in
theorem outerScan537 : scan edge (search edge 4) 5 [] 0 tail537 best = best := by
  have cons : tail537 = 3907 :: tail538 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates537_checked]
  change scan edge (search edge 4) 5 [] 0 tail538
    (if 4 ≤ rootCandidates537.length then search edge 4 [3907] rootCandidates537 3907 best else best) = best
  rw [if_pos (by decide), rootBranch537]
  exact outerScan538

set_option maxRecDepth 4096 in
theorem outerScan536 : scan edge (search edge 4) 5 [] 0 tail536 best = best := by
  have cons : tail536 = 3889 :: tail537 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates536_checked]
  change scan edge (search edge 4) 5 [] 0 tail537
    (if 4 ≤ rootCandidates536.length then search edge 4 [3889] rootCandidates536 3889 best else best) = best
  rw [if_pos (by decide), rootBranch536]
  exact outerScan537

end ProjectEulerSolutions.PrimePairUpper.Search
