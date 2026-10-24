import ProjectEulerSolutions.Lib.PrimePairLarge.Search009
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot602
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot603
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot604
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot605
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot606
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot607
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot608
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot609
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan609 : scan edge (search edge 4) 5 [] 0 tail609 winner = winner := by
  rw [tail609, scan, if_neg (by decide), root609Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail610
    (if 4 ≤ root609Candidates.length then search edge 4 [4513] root609Candidates 4513 winner else winner) = winner
  rw [if_pos (by decide), root609Branch]
  exact outerScan610
set_option maxRecDepth 4096 in
theorem outerScan608 : scan edge (search edge 4) 5 [] 0 tail608 winner = winner := by
  rw [tail608, scan, if_neg (by decide), root608Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail609
    (if 4 ≤ root608Candidates.length then search edge 4 [4507] root608Candidates 4507 winner else winner) = winner
  rw [if_pos (by decide), root608Branch]
  exact outerScan609
set_option maxRecDepth 4096 in
theorem outerScan607 : scan edge (search edge 4) 5 [] 0 tail607 winner = winner := by
  rw [tail607, scan, if_neg (by decide), root607Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail608
    (if 4 ≤ root607Candidates.length then search edge 4 [4493] root607Candidates 4493 winner else winner) = winner
  rw [if_pos (by decide), root607Branch]
  exact outerScan608
set_option maxRecDepth 4096 in
theorem outerScan606 : scan edge (search edge 4) 5 [] 0 tail606 winner = winner := by
  rw [tail606, scan, if_neg (by decide), root606Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail607
    (if 4 ≤ root606Candidates.length then search edge 4 [4483] root606Candidates 4483 winner else winner) = winner
  rw [if_pos (by decide), root606Branch]
  exact outerScan607
set_option maxRecDepth 4096 in
theorem outerScan605 : scan edge (search edge 4) 5 [] 0 tail605 winner = winner := by
  rw [tail605, scan, if_neg (by decide), root605Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail606
    (if 4 ≤ root605Candidates.length then search edge 4 [4481] root605Candidates 4481 winner else winner) = winner
  rw [if_pos (by decide), root605Branch]
  exact outerScan606
set_option maxRecDepth 4096 in
theorem outerScan604 : scan edge (search edge 4) 5 [] 0 tail604 winner = winner := by
  rw [tail604, scan, if_neg (by decide), root604Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail605
    (if 4 ≤ root604Candidates.length then search edge 4 [4463] root604Candidates 4463 winner else winner) = winner
  rw [if_pos (by decide), root604Branch]
  exact outerScan605
set_option maxRecDepth 4096 in
theorem outerScan603 : scan edge (search edge 4) 5 [] 0 tail603 winner = winner := by
  rw [tail603, scan, if_neg (by decide), root603Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail604
    (if 4 ≤ root603Candidates.length then search edge 4 [4457] root603Candidates 4457 winner else winner) = winner
  rw [if_pos (by decide), root603Branch]
  exact outerScan604
set_option maxRecDepth 4096 in
theorem outerScan602 : scan edge (search edge 4) 5 [] 0 tail602 winner = winner := by
  rw [tail602, scan, if_neg (by decide), root602Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail603
    (if 4 ≤ root602Candidates.length then search edge 4 [4451] root602Candidates 4451 winner else winner) = winner
  rw [if_pos (by decide), root602Branch]
  exact outerScan603
end ProjectEulerSolutions.PrimePairLarge.Search
