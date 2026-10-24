import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch060
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch061

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan580 : scan edge (search edge 4) 5 [] 0 tail580 best = best := by
  have cons : tail580 = 4253 :: tail581 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates580_checked]
  change scan edge (search edge 4) 5 [] 0 tail581
    (if 4 ≤ rootCandidates580.length then search edge 4 [4253] rootCandidates580 4253 best else best) = best
  rw [if_pos (by decide), rootBranch580]
  exact outerScan581

set_option maxRecDepth 4096 in
theorem outerScan579 : scan edge (search edge 4) 5 [] 0 tail579 best = best := by
  have cons : tail579 = 4243 :: tail580 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates579_checked]
  change scan edge (search edge 4) 5 [] 0 tail580
    (if 4 ≤ rootCandidates579.length then search edge 4 [4243] rootCandidates579 4243 best else best) = best
  rw [if_pos (by decide), rootBranch579]
  exact outerScan580

set_option maxRecDepth 4096 in
theorem outerScan578 : scan edge (search edge 4) 5 [] 0 tail578 best = best := by
  have cons : tail578 = 4241 :: tail579 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates578_checked]
  change scan edge (search edge 4) 5 [] 0 tail579
    (if 4 ≤ rootCandidates578.length then search edge 4 [4241] rootCandidates578 4241 best else best) = best
  rw [if_pos (by decide), rootBranch578]
  exact outerScan579

set_option maxRecDepth 4096 in
theorem outerScan577 : scan edge (search edge 4) 5 [] 0 tail577 best = best := by
  have cons : tail577 = 4231 :: tail578 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates577_checked]
  change scan edge (search edge 4) 5 [] 0 tail578
    (if 4 ≤ rootCandidates577.length then search edge 4 [4231] rootCandidates577 4231 best else best) = best
  rw [if_pos (by decide), rootBranch577]
  exact outerScan578

set_option maxRecDepth 4096 in
theorem outerScan576 : scan edge (search edge 4) 5 [] 0 tail576 best = best := by
  have cons : tail576 = 4229 :: tail577 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates576_checked]
  change scan edge (search edge 4) 5 [] 0 tail577
    (if 4 ≤ rootCandidates576.length then search edge 4 [4229] rootCandidates576 4229 best else best) = best
  rw [if_pos (by decide), rootBranch576]
  exact outerScan577

set_option maxRecDepth 4096 in
theorem outerScan575 : scan edge (search edge 4) 5 [] 0 tail575 best = best := by
  have cons : tail575 = 4219 :: tail576 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates575_checked]
  change scan edge (search edge 4) 5 [] 0 tail576
    (if 4 ≤ rootCandidates575.length then search edge 4 [4219] rootCandidates575 4219 best else best) = best
  rw [if_pos (by decide), rootBranch575]
  exact outerScan576

end ProjectEulerSolutions.PrimePairUpper.Search
