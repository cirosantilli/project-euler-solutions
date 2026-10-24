import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch062
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch063

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan567 : scan edge (search edge 4) 5 [] 0 tail567 best = best := by
  have cons : tail567 = 4139 :: tail568 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates567_checked]
  change scan edge (search edge 4) 5 [] 0 tail568
    (if 4 ≤ rootCandidates567.length then search edge 4 [4139] rootCandidates567 4139 best else best) = best
  rw [if_pos (by decide), rootBranch567]
  exact outerScan568

set_option maxRecDepth 4096 in
theorem outerScan566 : scan edge (search edge 4) 5 [] 0 tail566 best = best := by
  have cons : tail566 = 4133 :: tail567 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates566_checked]
  change scan edge (search edge 4) 5 [] 0 tail567
    (if 4 ≤ rootCandidates566.length then search edge 4 [4133] rootCandidates566 4133 best else best) = best
  rw [if_pos (by decide), rootBranch566]
  exact outerScan567

set_option maxRecDepth 4096 in
theorem outerScan565 : scan edge (search edge 4) 5 [] 0 tail565 best = best := by
  have cons : tail565 = 4129 :: tail566 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates565_checked]
  change scan edge (search edge 4) 5 [] 0 tail566
    (if 4 ≤ rootCandidates565.length then search edge 4 [4129] rootCandidates565 4129 best else best) = best
  rw [if_pos (by decide), rootBranch565]
  exact outerScan566

set_option maxRecDepth 4096 in
theorem outerScan564 : scan edge (search edge 4) 5 [] 0 tail564 best = best := by
  have cons : tail564 = 4127 :: tail565 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates564_checked]
  change scan edge (search edge 4) 5 [] 0 tail565
    (if 4 ≤ rootCandidates564.length then search edge 4 [4127] rootCandidates564 4127 best else best) = best
  rw [if_pos (by decide), rootBranch564]
  exact outerScan565

set_option maxRecDepth 4096 in
theorem outerScan563 : scan edge (search edge 4) 5 [] 0 tail563 best = best := by
  have cons : tail563 = 4111 :: tail564 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates563_checked]
  change scan edge (search edge 4) 5 [] 0 tail564
    (if 4 ≤ rootCandidates563.length then search edge 4 [4111] rootCandidates563 4111 best else best) = best
  rw [if_pos (by decide), rootBranch563]
  exact outerScan564

end ProjectEulerSolutions.PrimePairUpper.Search
