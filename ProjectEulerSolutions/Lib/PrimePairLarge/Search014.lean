import ProjectEulerSolutions.Lib.PrimePairLarge.Search013
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot570
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot571
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot572
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot573
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot574
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot575
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot576
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot577
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan577 : scan edge (search edge 4) 5 [] 0 tail577 winner = winner := by
  rw [tail577, scan, if_neg (by decide), root577Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail578
    (if 4 ≤ root577Candidates.length then search edge 4 [4231] root577Candidates 4231 winner else winner) = winner
  rw [if_pos (by decide), root577Branch]
  exact outerScan578
set_option maxRecDepth 4096 in
theorem outerScan576 : scan edge (search edge 4) 5 [] 0 tail576 winner = winner := by
  rw [tail576, scan, if_neg (by decide), root576Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail577
    (if 4 ≤ root576Candidates.length then search edge 4 [4229] root576Candidates 4229 winner else winner) = winner
  rw [if_pos (by decide), root576Branch]
  exact outerScan577
set_option maxRecDepth 4096 in
theorem outerScan575 : scan edge (search edge 4) 5 [] 0 tail575 winner = winner := by
  rw [tail575, scan, if_neg (by decide), root575Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail576
    (if 4 ≤ root575Candidates.length then search edge 4 [4219] root575Candidates 4219 winner else winner) = winner
  rw [if_pos (by decide), root575Branch]
  exact outerScan576
set_option maxRecDepth 4096 in
theorem outerScan574 : scan edge (search edge 4) 5 [] 0 tail574 winner = winner := by
  rw [tail574, scan, if_neg (by decide), root574Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail575
    (if 4 ≤ root574Candidates.length then search edge 4 [4217] root574Candidates 4217 winner else winner) = winner
  rw [if_pos (by decide), root574Branch]
  exact outerScan575
set_option maxRecDepth 4096 in
theorem outerScan573 : scan edge (search edge 4) 5 [] 0 tail573 winner = winner := by
  rw [tail573, scan, if_neg (by decide), root573Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail574
    (if 4 ≤ root573Candidates.length then search edge 4 [4211] root573Candidates 4211 winner else winner) = winner
  rw [if_pos (by decide), root573Branch]
  exact outerScan574
set_option maxRecDepth 4096 in
theorem outerScan572 : scan edge (search edge 4) 5 [] 0 tail572 winner = winner := by
  rw [tail572, scan, if_neg (by decide), root572Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail573
    (if 4 ≤ root572Candidates.length then search edge 4 [4201] root572Candidates 4201 winner else winner) = winner
  rw [if_pos (by decide), root572Branch]
  exact outerScan573
set_option maxRecDepth 4096 in
theorem outerScan571 : scan edge (search edge 4) 5 [] 0 tail571 winner = winner := by
  rw [tail571, scan, if_neg (by decide), root571Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail572
    (if 4 ≤ root571Candidates.length then search edge 4 [4177] root571Candidates 4177 winner else winner) = winner
  rw [if_pos (by decide), root571Branch]
  exact outerScan572
set_option maxRecDepth 4096 in
theorem outerScan570 : scan edge (search edge 4) 5 [] 0 tail570 winner = winner := by
  rw [tail570, scan, if_neg (by decide), root570Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail571
    (if 4 ≤ root570Candidates.length then search edge 4 [4159] root570Candidates 4159 winner else winner) = winner
  rw [if_pos (by decide), root570Branch]
  exact outerScan571
end ProjectEulerSolutions.PrimePairLarge.Search
