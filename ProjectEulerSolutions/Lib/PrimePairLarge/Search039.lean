import ProjectEulerSolutions.Lib.PrimePairLarge.Search038
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot370
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot371
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot372
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot373
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot374
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot375
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot376
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot377
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan377 : scan edge (search edge 4) 5 [] 0 tail377 winner = winner := by
  rw [tail377, scan, if_neg (by decide), root377Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail378
    (if 4 ≤ root377Candidates.length then search edge 4 [2617] root377Candidates 2617 winner else winner) = winner
  rw [if_pos (by decide), root377Branch]
  exact outerScan378
set_option maxRecDepth 4096 in
theorem outerScan376 : scan edge (search edge 4) 5 [] 0 tail376 winner = winner := by
  rw [tail376, scan, if_neg (by decide), root376Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail377
    (if 4 ≤ root376Candidates.length then search edge 4 [2609] root376Candidates 2609 winner else winner) = winner
  rw [if_pos (by decide), root376Branch]
  exact outerScan377
set_option maxRecDepth 4096 in
theorem outerScan375 : scan edge (search edge 4) 5 [] 0 tail375 winner = winner := by
  rw [tail375, scan, if_neg (by decide), root375Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail376
    (if 4 ≤ root375Candidates.length then search edge 4 [2593] root375Candidates 2593 winner else winner) = winner
  rw [if_pos (by decide), root375Branch]
  exact outerScan376
set_option maxRecDepth 4096 in
theorem outerScan374 : scan edge (search edge 4) 5 [] 0 tail374 winner = winner := by
  rw [tail374, scan, if_neg (by decide), root374Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail375
    (if 4 ≤ root374Candidates.length then search edge 4 [2591] root374Candidates 2591 winner else winner) = winner
  rw [if_pos (by decide), root374Branch]
  exact outerScan375
set_option maxRecDepth 4096 in
theorem outerScan373 : scan edge (search edge 4) 5 [] 0 tail373 winner = winner := by
  rw [tail373, scan, if_neg (by decide), root373Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail374
    (if 4 ≤ root373Candidates.length then search edge 4 [2579] root373Candidates 2579 winner else winner) = winner
  rw [if_pos (by decide), root373Branch]
  exact outerScan374
set_option maxRecDepth 4096 in
theorem outerScan372 : scan edge (search edge 4) 5 [] 0 tail372 winner = winner := by
  rw [tail372, scan, if_neg (by decide), root372Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail373
    (if 4 ≤ root372Candidates.length then search edge 4 [2557] root372Candidates 2557 winner else winner) = winner
  rw [if_pos (by decide), root372Branch]
  exact outerScan373
set_option maxRecDepth 4096 in
theorem outerScan371 : scan edge (search edge 4) 5 [] 0 tail371 winner = winner := by
  rw [tail371, scan, if_neg (by decide), root371Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail372
    (if 4 ≤ root371Candidates.length then search edge 4 [2551] root371Candidates 2551 winner else winner) = winner
  rw [if_pos (by decide), root371Branch]
  exact outerScan372
set_option maxRecDepth 4096 in
theorem outerScan370 : scan edge (search edge 4) 5 [] 0 tail370 winner = winner := by
  rw [tail370, scan, if_neg (by decide), root370Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail371
    (if 4 ≤ root370Candidates.length then search edge 4 [2549] root370Candidates 2549 winner else winner) = winner
  rw [if_pos (by decide), root370Branch]
  exact outerScan371
end ProjectEulerSolutions.PrimePairLarge.Search
