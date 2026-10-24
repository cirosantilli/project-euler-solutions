import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch072
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch073

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan513 : scan edge (search edge 4) 5 [] 0 tail513 best = best := by
  have cons : tail513 = 3697 :: tail514 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates513_checked]
  change scan edge (search edge 4) 5 [] 0 tail514
    (if 4 ≤ rootCandidates513.length then search edge 4 [3697] rootCandidates513 3697 best else best) = best
  rw [if_pos (by decide), rootBranch513]
  exact outerScan514

set_option maxRecDepth 4096 in
theorem outerScan512 : scan edge (search edge 4) 5 [] 0 tail512 best = best := by
  have cons : tail512 = 3691 :: tail513 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates512_checked]
  change scan edge (search edge 4) 5 [] 0 tail513
    (if 4 ≤ rootCandidates512.length then search edge 4 [3691] rootCandidates512 3691 best else best) = best
  rw [if_pos (by decide), rootBranch512]
  exact outerScan513

set_option maxRecDepth 4096 in
theorem outerScan511 : scan edge (search edge 4) 5 [] 0 tail511 best = best := by
  have cons : tail511 = 3677 :: tail512 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates511_checked]
  change scan edge (search edge 4) 5 [] 0 tail512
    (if 4 ≤ rootCandidates511.length then search edge 4 [3677] rootCandidates511 3677 best else best) = best
  rw [if_pos (by decide), rootBranch511]
  exact outerScan512

set_option maxRecDepth 4096 in
theorem outerScan510 : scan edge (search edge 4) 5 [] 0 tail510 best = best := by
  have cons : tail510 = 3673 :: tail511 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates510_checked]
  change scan edge (search edge 4) 5 [] 0 tail511
    (if 4 ≤ rootCandidates510.length then search edge 4 [3673] rootCandidates510 3673 best else best) = best
  rw [if_pos (by decide), rootBranch510]
  exact outerScan511

set_option maxRecDepth 4096 in
theorem outerScan509 : scan edge (search edge 4) 5 [] 0 tail509 best = best := by
  have cons : tail509 = 3671 :: tail510 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates509_checked]
  change scan edge (search edge 4) 5 [] 0 tail510
    (if 4 ≤ rootCandidates509.length then search edge 4 [3671] rootCandidates509 3671 best else best) = best
  rw [if_pos (by decide), rootBranch509]
  exact outerScan510

set_option maxRecDepth 4096 in
theorem outerScan508 : scan edge (search edge 4) 5 [] 0 tail508 best = best := by
  have cons : tail508 = 3659 :: tail509 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates508_checked]
  change scan edge (search edge 4) 5 [] 0 tail509
    (if 4 ≤ rootCandidates508.length then search edge 4 [3659] rootCandidates508 3659 best else best) = best
  rw [if_pos (by decide), rootBranch508]
  exact outerScan509

end ProjectEulerSolutions.PrimePairUpper.Search
