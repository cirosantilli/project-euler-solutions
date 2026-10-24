import ProjectEulerSolutions.Lib.PrimePairLarge.Search023
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot490
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot491
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot492
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot493
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot494
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot495
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot496
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot497
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan497 : scan edge (search edge 4) 5 [] 0 tail497 winner = winner := by
  rw [tail497, scan, if_neg (by decide), root497Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail498
    (if 4 ≤ root497Candidates.length then search edge 4 [3571] root497Candidates 3571 winner else winner) = winner
  rw [if_pos (by decide), root497Branch]
  exact outerScan498
set_option maxRecDepth 4096 in
theorem outerScan496 : scan edge (search edge 4) 5 [] 0 tail496 winner = winner := by
  rw [tail496, scan, if_neg (by decide), root496Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail497
    (if 4 ≤ root496Candidates.length then search edge 4 [3559] root496Candidates 3559 winner else winner) = winner
  rw [if_pos (by decide), root496Branch]
  exact outerScan497
set_option maxRecDepth 4096 in
theorem outerScan495 : scan edge (search edge 4) 5 [] 0 tail495 winner = winner := by
  rw [tail495, scan, if_neg (by decide), root495Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail496
    (if 4 ≤ root495Candidates.length then search edge 4 [3557] root495Candidates 3557 winner else winner) = winner
  rw [if_pos (by decide), root495Branch]
  exact outerScan496
set_option maxRecDepth 4096 in
theorem outerScan494 : scan edge (search edge 4) 5 [] 0 tail494 winner = winner := by
  rw [tail494, scan, if_neg (by decide), root494Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail495
    (if 4 ≤ root494Candidates.length then search edge 4 [3547] root494Candidates 3547 winner else winner) = winner
  rw [if_pos (by decide), root494Branch]
  exact outerScan495
set_option maxRecDepth 4096 in
theorem outerScan493 : scan edge (search edge 4) 5 [] 0 tail493 winner = winner := by
  rw [tail493, scan, if_neg (by decide), root493Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail494
    (if 4 ≤ root493Candidates.length then search edge 4 [3541] root493Candidates 3541 winner else winner) = winner
  rw [if_pos (by decide), root493Branch]
  exact outerScan494
set_option maxRecDepth 4096 in
theorem outerScan492 : scan edge (search edge 4) 5 [] 0 tail492 winner = winner := by
  rw [tail492, scan, if_neg (by decide), root492Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail493
    (if 4 ≤ root492Candidates.length then search edge 4 [3539] root492Candidates 3539 winner else winner) = winner
  rw [if_pos (by decide), root492Branch]
  exact outerScan493
set_option maxRecDepth 4096 in
theorem outerScan491 : scan edge (search edge 4) 5 [] 0 tail491 winner = winner := by
  rw [tail491, scan, if_neg (by decide), root491Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail492
    (if 4 ≤ root491Candidates.length then search edge 4 [3533] root491Candidates 3533 winner else winner) = winner
  rw [if_pos (by decide), root491Branch]
  exact outerScan492
set_option maxRecDepth 4096 in
theorem outerScan490 : scan edge (search edge 4) 5 [] 0 tail490 winner = winner := by
  rw [tail490, scan, if_neg (by decide), root490Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail491
    (if 4 ≤ root490Candidates.length then search edge 4 [3529] root490Candidates 3529 winner else winner) = winner
  rw [if_pos (by decide), root490Branch]
  exact outerScan491
end ProjectEulerSolutions.PrimePairLarge.Search
