import ProjectEulerSolutions.Lib.PrimePairLarge.Search022
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot498
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot499
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot500
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot501
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot502
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot503
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot504
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot505
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan505 : scan edge (search edge 4) 5 [] 0 tail505 winner = winner := by
  rw [tail505, scan, if_neg (by decide), root505Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail506
    (if 4 ≤ root505Candidates.length then search edge 4 [3631] root505Candidates 3631 winner else winner) = winner
  rw [if_pos (by decide), root505Branch]
  exact outerScan506
set_option maxRecDepth 4096 in
theorem outerScan504 : scan edge (search edge 4) 5 [] 0 tail504 winner = winner := by
  rw [tail504, scan, if_neg (by decide), root504Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail505
    (if 4 ≤ root504Candidates.length then search edge 4 [3623] root504Candidates 3623 winner else winner) = winner
  rw [if_pos (by decide), root504Branch]
  exact outerScan505
set_option maxRecDepth 4096 in
theorem outerScan503 : scan edge (search edge 4) 5 [] 0 tail503 winner = winner := by
  rw [tail503, scan, if_neg (by decide), root503Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail504
    (if 4 ≤ root503Candidates.length then search edge 4 [3617] root503Candidates 3617 winner else winner) = winner
  rw [if_pos (by decide), root503Branch]
  exact outerScan504
set_option maxRecDepth 4096 in
theorem outerScan502 : scan edge (search edge 4) 5 [] 0 tail502 winner = winner := by
  rw [tail502, scan, if_neg (by decide), root502Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail503
    (if 4 ≤ root502Candidates.length then search edge 4 [3613] root502Candidates 3613 winner else winner) = winner
  rw [if_pos (by decide), root502Branch]
  exact outerScan503
set_option maxRecDepth 4096 in
theorem outerScan501 : scan edge (search edge 4) 5 [] 0 tail501 winner = winner := by
  rw [tail501, scan, if_neg (by decide), root501Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail502
    (if 4 ≤ root501Candidates.length then search edge 4 [3607] root501Candidates 3607 winner else winner) = winner
  rw [if_pos (by decide), root501Branch]
  exact outerScan502
set_option maxRecDepth 4096 in
theorem outerScan500 : scan edge (search edge 4) 5 [] 0 tail500 winner = winner := by
  rw [tail500, scan, if_neg (by decide), root500Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail501
    (if 4 ≤ root500Candidates.length then search edge 4 [3593] root500Candidates 3593 winner else winner) = winner
  rw [if_pos (by decide), root500Branch]
  exact outerScan501
set_option maxRecDepth 4096 in
theorem outerScan499 : scan edge (search edge 4) 5 [] 0 tail499 winner = winner := by
  rw [tail499, scan, if_neg (by decide), root499Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail500
    (if 4 ≤ root499Candidates.length then search edge 4 [3583] root499Candidates 3583 winner else winner) = winner
  rw [if_pos (by decide), root499Branch]
  exact outerScan500
set_option maxRecDepth 4096 in
theorem outerScan498 : scan edge (search edge 4) 5 [] 0 tail498 winner = winner := by
  rw [tail498, scan, if_neg (by decide), root498Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail499
    (if 4 ≤ root498Candidates.length then search edge 4 [3581] root498Candidates 3581 winner else winner) = winner
  rw [if_pos (by decide), root498Branch]
  exact outerScan499
end ProjectEulerSolutions.PrimePairLarge.Search
