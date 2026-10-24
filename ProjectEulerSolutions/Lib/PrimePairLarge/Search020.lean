import ProjectEulerSolutions.Lib.PrimePairLarge.Search019
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot522
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot523
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot524
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot525
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot526
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot527
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot528
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot529
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan529 : scan edge (search edge 4) 5 [] 0 tail529 winner = winner := by
  rw [tail529, scan, if_neg (by decide), root529Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail530
    (if 4 ≤ root529Candidates.length then search edge 4 [3833] root529Candidates 3833 winner else winner) = winner
  rw [if_pos (by decide), root529Branch]
  exact outerScan530
set_option maxRecDepth 4096 in
theorem outerScan528 : scan edge (search edge 4) 5 [] 0 tail528 winner = winner := by
  rw [tail528, scan, if_neg (by decide), root528Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail529
    (if 4 ≤ root528Candidates.length then search edge 4 [3823] root528Candidates 3823 winner else winner) = winner
  rw [if_pos (by decide), root528Branch]
  exact outerScan529
set_option maxRecDepth 4096 in
theorem outerScan527 : scan edge (search edge 4) 5 [] 0 tail527 winner = winner := by
  rw [tail527, scan, if_neg (by decide), root527Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail528
    (if 4 ≤ root527Candidates.length then search edge 4 [3821] root527Candidates 3821 winner else winner) = winner
  rw [if_pos (by decide), root527Branch]
  exact outerScan528
set_option maxRecDepth 4096 in
theorem outerScan526 : scan edge (search edge 4) 5 [] 0 tail526 winner = winner := by
  rw [tail526, scan, if_neg (by decide), root526Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail527
    (if 4 ≤ root526Candidates.length then search edge 4 [3803] root526Candidates 3803 winner else winner) = winner
  rw [if_pos (by decide), root526Branch]
  exact outerScan527
set_option maxRecDepth 4096 in
theorem outerScan525 : scan edge (search edge 4) 5 [] 0 tail525 winner = winner := by
  rw [tail525, scan, if_neg (by decide), root525Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail526
    (if 4 ≤ root525Candidates.length then search edge 4 [3797] root525Candidates 3797 winner else winner) = winner
  rw [if_pos (by decide), root525Branch]
  exact outerScan526
set_option maxRecDepth 4096 in
theorem outerScan524 : scan edge (search edge 4) 5 [] 0 tail524 winner = winner := by
  rw [tail524, scan, if_neg (by decide), root524Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail525
    (if 4 ≤ root524Candidates.length then search edge 4 [3793] root524Candidates 3793 winner else winner) = winner
  rw [if_pos (by decide), root524Branch]
  exact outerScan525
set_option maxRecDepth 4096 in
theorem outerScan523 : scan edge (search edge 4) 5 [] 0 tail523 winner = winner := by
  rw [tail523, scan, if_neg (by decide), root523Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail524
    (if 4 ≤ root523Candidates.length then search edge 4 [3779] root523Candidates 3779 winner else winner) = winner
  rw [if_pos (by decide), root523Branch]
  exact outerScan524
set_option maxRecDepth 4096 in
theorem outerScan522 : scan edge (search edge 4) 5 [] 0 tail522 winner = winner := by
  rw [tail522, scan, if_neg (by decide), root522Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail523
    (if 4 ≤ root522Candidates.length then search edge 4 [3769] root522Candidates 3769 winner else winner) = winner
  rw [if_pos (by decide), root522Branch]
  exact outerScan523
end ProjectEulerSolutions.PrimePairLarge.Search
