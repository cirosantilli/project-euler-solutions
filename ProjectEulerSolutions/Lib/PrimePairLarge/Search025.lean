import ProjectEulerSolutions.Lib.PrimePairLarge.Search024
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot482
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot483
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot484
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot485
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot486
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot487
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot488
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot489
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan489 : scan edge (search edge 4) 5 [] 0 tail489 winner = winner := by
  rw [tail489, scan, if_neg (by decide), root489Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail490
    (if 4 ≤ root489Candidates.length then search edge 4 [3527] root489Candidates 3527 winner else winner) = winner
  rw [if_pos (by decide), root489Branch]
  exact outerScan490
set_option maxRecDepth 4096 in
theorem outerScan488 : scan edge (search edge 4) 5 [] 0 tail488 winner = winner := by
  rw [tail488, scan, if_neg (by decide), root488Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail489
    (if 4 ≤ root488Candidates.length then search edge 4 [3517] root488Candidates 3517 winner else winner) = winner
  rw [if_pos (by decide), root488Branch]
  exact outerScan489
set_option maxRecDepth 4096 in
theorem outerScan487 : scan edge (search edge 4) 5 [] 0 tail487 winner = winner := by
  rw [tail487, scan, if_neg (by decide), root487Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail488
    (if 4 ≤ root487Candidates.length then search edge 4 [3511] root487Candidates 3511 winner else winner) = winner
  rw [if_pos (by decide), root487Branch]
  exact outerScan488
set_option maxRecDepth 4096 in
theorem outerScan486 : scan edge (search edge 4) 5 [] 0 tail486 winner = winner := by
  rw [tail486, scan, if_neg (by decide), root486Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail487
    (if 4 ≤ root486Candidates.length then search edge 4 [3499] root486Candidates 3499 winner else winner) = winner
  rw [if_pos (by decide), root486Branch]
  exact outerScan487
set_option maxRecDepth 4096 in
theorem outerScan485 : scan edge (search edge 4) 5 [] 0 tail485 winner = winner := by
  rw [tail485, scan, if_neg (by decide), root485Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail486
    (if 4 ≤ root485Candidates.length then search edge 4 [3491] root485Candidates 3491 winner else winner) = winner
  rw [if_pos (by decide), root485Branch]
  exact outerScan486
set_option maxRecDepth 4096 in
theorem outerScan484 : scan edge (search edge 4) 5 [] 0 tail484 winner = winner := by
  rw [tail484, scan, if_neg (by decide), root484Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail485
    (if 4 ≤ root484Candidates.length then search edge 4 [3469] root484Candidates 3469 winner else winner) = winner
  rw [if_pos (by decide), root484Branch]
  exact outerScan485
set_option maxRecDepth 4096 in
theorem outerScan483 : scan edge (search edge 4) 5 [] 0 tail483 winner = winner := by
  rw [tail483, scan, if_neg (by decide), root483Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail484
    (if 4 ≤ root483Candidates.length then search edge 4 [3467] root483Candidates 3467 winner else winner) = winner
  rw [if_pos (by decide), root483Branch]
  exact outerScan484
set_option maxRecDepth 4096 in
theorem outerScan482 : scan edge (search edge 4) 5 [] 0 tail482 winner = winner := by
  rw [tail482, scan, if_neg (by decide), root482Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail483
    (if 4 ≤ root482Candidates.length then search edge 4 [3463] root482Candidates 3463 winner else winner) = winner
  rw [if_pos (by decide), root482Branch]
  exact outerScan483
end ProjectEulerSolutions.PrimePairLarge.Search
