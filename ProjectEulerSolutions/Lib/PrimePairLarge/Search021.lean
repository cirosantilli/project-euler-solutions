import ProjectEulerSolutions.Lib.PrimePairLarge.Search020
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot514
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot515
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot516
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot517
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot518
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot519
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot520
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot521
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan521 : scan edge (search edge 4) 5 [] 0 tail521 winner = winner := by
  rw [tail521, scan, if_neg (by decide), root521Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail522
    (if 4 ≤ root521Candidates.length then search edge 4 [3767] root521Candidates 3767 winner else winner) = winner
  rw [if_pos (by decide), root521Branch]
  exact outerScan522
set_option maxRecDepth 4096 in
theorem outerScan520 : scan edge (search edge 4) 5 [] 0 tail520 winner = winner := by
  rw [tail520, scan, if_neg (by decide), root520Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail521
    (if 4 ≤ root520Candidates.length then search edge 4 [3761] root520Candidates 3761 winner else winner) = winner
  rw [if_pos (by decide), root520Branch]
  exact outerScan521
set_option maxRecDepth 4096 in
theorem outerScan519 : scan edge (search edge 4) 5 [] 0 tail519 winner = winner := by
  rw [tail519, scan, if_neg (by decide), root519Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail520
    (if 4 ≤ root519Candidates.length then search edge 4 [3739] root519Candidates 3739 winner else winner) = winner
  rw [if_pos (by decide), root519Branch]
  exact outerScan520
set_option maxRecDepth 4096 in
theorem outerScan518 : scan edge (search edge 4) 5 [] 0 tail518 winner = winner := by
  rw [tail518, scan, if_neg (by decide), root518Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail519
    (if 4 ≤ root518Candidates.length then search edge 4 [3733] root518Candidates 3733 winner else winner) = winner
  rw [if_pos (by decide), root518Branch]
  exact outerScan519
set_option maxRecDepth 4096 in
theorem outerScan517 : scan edge (search edge 4) 5 [] 0 tail517 winner = winner := by
  rw [tail517, scan, if_neg (by decide), root517Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail518
    (if 4 ≤ root517Candidates.length then search edge 4 [3727] root517Candidates 3727 winner else winner) = winner
  rw [if_pos (by decide), root517Branch]
  exact outerScan518
set_option maxRecDepth 4096 in
theorem outerScan516 : scan edge (search edge 4) 5 [] 0 tail516 winner = winner := by
  rw [tail516, scan, if_neg (by decide), root516Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail517
    (if 4 ≤ root516Candidates.length then search edge 4 [3719] root516Candidates 3719 winner else winner) = winner
  rw [if_pos (by decide), root516Branch]
  exact outerScan517
set_option maxRecDepth 4096 in
theorem outerScan515 : scan edge (search edge 4) 5 [] 0 tail515 winner = winner := by
  rw [tail515, scan, if_neg (by decide), root515Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail516
    (if 4 ≤ root515Candidates.length then search edge 4 [3709] root515Candidates 3709 winner else winner) = winner
  rw [if_pos (by decide), root515Branch]
  exact outerScan516
set_option maxRecDepth 4096 in
theorem outerScan514 : scan edge (search edge 4) 5 [] 0 tail514 winner = winner := by
  rw [tail514, scan, if_neg (by decide), root514Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail515
    (if 4 ≤ root514Candidates.length then search edge 4 [3701] root514Candidates 3701 winner else winner) = winner
  rw [if_pos (by decide), root514Branch]
  exact outerScan515
end ProjectEulerSolutions.PrimePairLarge.Search
