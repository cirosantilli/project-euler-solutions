import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch094
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch095

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan412 : scan edge (search edge 4) 5 [] 0 tail412 best = best := by
  have cons : tail412 = 2857 :: tail413 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates412_checked]
  change scan edge (search edge 4) 5 [] 0 tail413
    (if 4 ≤ rootCandidates412.length then search edge 4 [2857] rootCandidates412 2857 best else best) = best
  rw [if_pos (by decide), rootBranch412]
  exact outerScan413

set_option maxRecDepth 4096 in
theorem outerScan411 : scan edge (search edge 4) 5 [] 0 tail411 best = best := by
  have cons : tail411 = 2851 :: tail412 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates411_checked]
  change scan edge (search edge 4) 5 [] 0 tail412
    (if 4 ≤ rootCandidates411.length then search edge 4 [2851] rootCandidates411 2851 best else best) = best
  rw [if_pos (by decide), rootBranch411]
  exact outerScan412

set_option maxRecDepth 4096 in
theorem outerScan410 : scan edge (search edge 4) 5 [] 0 tail410 best = best := by
  have cons : tail410 = 2843 :: tail411 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates410_checked]
  change scan edge (search edge 4) 5 [] 0 tail411
    (if 4 ≤ rootCandidates410.length then search edge 4 [2843] rootCandidates410 2843 best else best) = best
  rw [if_pos (by decide), rootBranch410]
  exact outerScan411

set_option maxRecDepth 4096 in
theorem outerScan409 : scan edge (search edge 4) 5 [] 0 tail409 best = best := by
  have cons : tail409 = 2837 :: tail410 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates409_checked]
  change scan edge (search edge 4) 5 [] 0 tail410
    (if 4 ≤ rootCandidates409.length then search edge 4 [2837] rootCandidates409 2837 best else best) = best
  rw [if_pos (by decide), rootBranch409]
  exact outerScan410

end ProjectEulerSolutions.PrimePairUpper.Search
