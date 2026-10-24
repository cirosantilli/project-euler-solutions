import ProjectEulerSolutions.Lib.PrimePairLarge.Search071
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot106
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot107
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot108
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot109
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot110
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot111
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot112
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot113
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan113 : scan edge (search edge 4) 5 [] 0 tail113 winner = winner := by
  rw [tail113, scan, if_neg (by decide), root113Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail114
    (if 4 ≤ root113Candidates.length then search edge 4 [641] root113Candidates 641 winner else winner) = winner
  rw [if_pos (by decide), root113Branch]
  exact outerScan114
set_option maxRecDepth 4096 in
theorem outerScan112 : scan edge (search edge 4) 5 [] 0 tail112 winner = winner := by
  rw [tail112, scan, if_neg (by decide), root112Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail113
    (if 4 ≤ root112Candidates.length then search edge 4 [631] root112Candidates 631 winner else winner) = winner
  rw [if_pos (by decide), root112Branch]
  exact outerScan113
set_option maxRecDepth 4096 in
theorem outerScan111 : scan edge (search edge 4) 5 [] 0 tail111 winner = winner := by
  rw [tail111, scan, if_neg (by decide), root111Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail112
    (if 4 ≤ root111Candidates.length then search edge 4 [619] root111Candidates 619 winner else winner) = winner
  rw [if_pos (by decide), root111Branch]
  exact outerScan112
set_option maxRecDepth 4096 in
theorem outerScan110 : scan edge (search edge 4) 5 [] 0 tail110 winner = winner := by
  rw [tail110, scan, if_neg (by decide), root110Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail111
    (if 4 ≤ root110Candidates.length then search edge 4 [617] root110Candidates 617 winner else winner) = winner
  rw [if_pos (by decide), root110Branch]
  exact outerScan111
set_option maxRecDepth 4096 in
theorem outerScan109 : scan edge (search edge 4) 5 [] 0 tail109 winner = winner := by
  rw [tail109, scan, if_neg (by decide), root109Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail110
    (if 4 ≤ root109Candidates.length then search edge 4 [613] root109Candidates 613 winner else winner) = winner
  rw [if_pos (by decide), root109Branch]
  exact outerScan110
set_option maxRecDepth 4096 in
theorem outerScan108 : scan edge (search edge 4) 5 [] 0 tail108 winner = winner := by
  rw [tail108, scan, if_neg (by decide), root108Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail109
    (if 4 ≤ root108Candidates.length then search edge 4 [607] root108Candidates 607 winner else winner) = winner
  rw [if_pos (by decide), root108Branch]
  exact outerScan109
set_option maxRecDepth 4096 in
theorem outerScan107 : scan edge (search edge 4) 5 [] 0 tail107 winner = winner := by
  rw [tail107, scan, if_neg (by decide), root107Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail108
    (if 4 ≤ root107Candidates.length then search edge 4 [601] root107Candidates 601 winner else winner) = winner
  rw [if_pos (by decide), root107Branch]
  exact outerScan108
set_option maxRecDepth 4096 in
theorem outerScan106 : scan edge (search edge 4) 5 [] 0 tail106 winner = winner := by
  rw [tail106, scan, if_neg (by decide), root106Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail107
    (if 4 ≤ root106Candidates.length then search edge 4 [599] root106Candidates 599 winner else winner) = winner
  rw [if_pos (by decide), root106Branch]
  exact outerScan107
end ProjectEulerSolutions.PrimePairLarge.Search
