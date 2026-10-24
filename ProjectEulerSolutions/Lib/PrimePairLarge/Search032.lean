import ProjectEulerSolutions.Lib.PrimePairLarge.Search031
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot426
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot427
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot428
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot429
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot430
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot431
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot432
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot433
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan433 : scan edge (search edge 4) 5 [] 0 tail433 winner = winner := by
  rw [tail433, scan, if_neg (by decide), root433Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail434
    (if 4 ≤ root433Candidates.length then search edge 4 [3041] root433Candidates 3041 winner else winner) = winner
  rw [if_pos (by decide), root433Branch]
  exact outerScan434
set_option maxRecDepth 4096 in
theorem outerScan432 : scan edge (search edge 4) 5 [] 0 tail432 winner = winner := by
  rw [tail432, scan, if_neg (by decide), root432Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail433
    (if 4 ≤ root432Candidates.length then search edge 4 [3037] root432Candidates 3037 winner else winner) = winner
  rw [if_pos (by decide), root432Branch]
  exact outerScan433
set_option maxRecDepth 4096 in
theorem outerScan431 : scan edge (search edge 4) 5 [] 0 tail431 winner = winner := by
  rw [tail431, scan, if_neg (by decide), root431Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail432
    (if 4 ≤ root431Candidates.length then search edge 4 [3023] root431Candidates 3023 winner else winner) = winner
  rw [if_pos (by decide), root431Branch]
  exact outerScan432
set_option maxRecDepth 4096 in
theorem outerScan430 : scan edge (search edge 4) 5 [] 0 tail430 winner = winner := by
  rw [tail430, scan, if_neg (by decide), root430Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail431
    (if 4 ≤ root430Candidates.length then search edge 4 [3019] root430Candidates 3019 winner else winner) = winner
  rw [if_pos (by decide), root430Branch]
  exact outerScan431
set_option maxRecDepth 4096 in
theorem outerScan429 : scan edge (search edge 4) 5 [] 0 tail429 winner = winner := by
  rw [tail429, scan, if_neg (by decide), root429Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail430
    (if 4 ≤ root429Candidates.length then search edge 4 [3011] root429Candidates 3011 winner else winner) = winner
  rw [if_pos (by decide), root429Branch]
  exact outerScan430
set_option maxRecDepth 4096 in
theorem outerScan428 : scan edge (search edge 4) 5 [] 0 tail428 winner = winner := by
  rw [tail428, scan, if_neg (by decide), root428Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail429
    (if 4 ≤ root428Candidates.length then search edge 4 [3001] root428Candidates 3001 winner else winner) = winner
  rw [if_pos (by decide), root428Branch]
  exact outerScan429
set_option maxRecDepth 4096 in
theorem outerScan427 : scan edge (search edge 4) 5 [] 0 tail427 winner = winner := by
  rw [tail427, scan, if_neg (by decide), root427Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail428
    (if 4 ≤ root427Candidates.length then search edge 4 [2999] root427Candidates 2999 winner else winner) = winner
  rw [if_pos (by decide), root427Branch]
  exact outerScan428
set_option maxRecDepth 4096 in
theorem outerScan426 : scan edge (search edge 4) 5 [] 0 tail426 winner = winner := by
  rw [tail426, scan, if_neg (by decide), root426Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail427
    (if 4 ≤ root426Candidates.length then search edge 4 [2971] root426Candidates 2971 winner else winner) = winner
  rw [if_pos (by decide), root426Branch]
  exact outerScan427
end ProjectEulerSolutions.PrimePairLarge.Search
