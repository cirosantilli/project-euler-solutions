import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch084
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch085

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan456 : scan edge (search edge 4) 5 [] 0 tail456 best = best := by
  have cons : tail456 = 3253 :: tail457 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates456_checked]
  change scan edge (search edge 4) 5 [] 0 tail457
    (if 4 ≤ rootCandidates456.length then search edge 4 [3253] rootCandidates456 3253 best else best) = best
  rw [if_pos (by decide), rootBranch456]
  exact outerScan457

set_option maxRecDepth 4096 in
theorem outerScan455 : scan edge (search edge 4) 5 [] 0 tail455 best = best := by
  have cons : tail455 = 3251 :: tail456 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates455_checked]
  change scan edge (search edge 4) 5 [] 0 tail456
    (if 4 ≤ rootCandidates455.length then search edge 4 [3251] rootCandidates455 3251 best else best) = best
  rw [if_pos (by decide), rootBranch455]
  exact outerScan456

set_option maxRecDepth 4096 in
theorem outerScan454 : scan edge (search edge 4) 5 [] 0 tail454 best = best := by
  have cons : tail454 = 3229 :: tail455 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates454_checked]
  change scan edge (search edge 4) 5 [] 0 tail455
    (if 4 ≤ rootCandidates454.length then search edge 4 [3229] rootCandidates454 3229 best else best) = best
  rw [if_pos (by decide), rootBranch454]
  exact outerScan455

set_option maxRecDepth 4096 in
theorem outerScan453 : scan edge (search edge 4) 5 [] 0 tail453 best = best := by
  have cons : tail453 = 3221 :: tail454 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates453_checked]
  change scan edge (search edge 4) 5 [] 0 tail454
    (if 4 ≤ rootCandidates453.length then search edge 4 [3221] rootCandidates453 3221 best else best) = best
  rw [if_pos (by decide), rootBranch453]
  exact outerScan454

end ProjectEulerSolutions.PrimePairUpper.Search
