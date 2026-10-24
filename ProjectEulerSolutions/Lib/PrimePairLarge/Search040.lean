import ProjectEulerSolutions.Lib.PrimePairLarge.Search039
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot362
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot363
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot364
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot365
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot366
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot367
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot368
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot369
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan369 : scan edge (search edge 4) 5 [] 0 tail369 winner = winner := by
  rw [tail369, scan, if_neg (by decide), root369Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail370
    (if 4 ≤ root369Candidates.length then search edge 4 [2543] root369Candidates 2543 winner else winner) = winner
  rw [if_pos (by decide), root369Branch]
  exact outerScan370
set_option maxRecDepth 4096 in
theorem outerScan368 : scan edge (search edge 4) 5 [] 0 tail368 winner = winner := by
  rw [tail368, scan, if_neg (by decide), root368Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail369
    (if 4 ≤ root368Candidates.length then search edge 4 [2539] root368Candidates 2539 winner else winner) = winner
  rw [if_pos (by decide), root368Branch]
  exact outerScan369
set_option maxRecDepth 4096 in
theorem outerScan367 : scan edge (search edge 4) 5 [] 0 tail367 winner = winner := by
  rw [tail367, scan, if_neg (by decide), root367Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail368
    (if 4 ≤ root367Candidates.length then search edge 4 [2531] root367Candidates 2531 winner else winner) = winner
  rw [if_pos (by decide), root367Branch]
  exact outerScan368
set_option maxRecDepth 4096 in
theorem outerScan366 : scan edge (search edge 4) 5 [] 0 tail366 winner = winner := by
  rw [tail366, scan, if_neg (by decide), root366Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail367
    (if 4 ≤ root366Candidates.length then search edge 4 [2521] root366Candidates 2521 winner else winner) = winner
  rw [if_pos (by decide), root366Branch]
  exact outerScan367
set_option maxRecDepth 4096 in
theorem outerScan365 : scan edge (search edge 4) 5 [] 0 tail365 winner = winner := by
  rw [tail365, scan, if_neg (by decide), root365Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail366
    (if 4 ≤ root365Candidates.length then search edge 4 [2503] root365Candidates 2503 winner else winner) = winner
  rw [if_pos (by decide), root365Branch]
  exact outerScan366
set_option maxRecDepth 4096 in
theorem outerScan364 : scan edge (search edge 4) 5 [] 0 tail364 winner = winner := by
  rw [tail364, scan, if_neg (by decide), root364Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail365
    (if 4 ≤ root364Candidates.length then search edge 4 [2477] root364Candidates 2477 winner else winner) = winner
  rw [if_pos (by decide), root364Branch]
  exact outerScan365
set_option maxRecDepth 4096 in
theorem outerScan363 : scan edge (search edge 4) 5 [] 0 tail363 winner = winner := by
  rw [tail363, scan, if_neg (by decide), root363Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail364
    (if 4 ≤ root363Candidates.length then search edge 4 [2473] root363Candidates 2473 winner else winner) = winner
  rw [if_pos (by decide), root363Branch]
  exact outerScan364
set_option maxRecDepth 4096 in
theorem outerScan362 : scan edge (search edge 4) 5 [] 0 tail362 winner = winner := by
  rw [tail362, scan, if_neg (by decide), root362Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail363
    (if 4 ≤ root362Candidates.length then search edge 4 [2467] root362Candidates 2467 winner else winner) = winner
  rw [if_pos (by decide), root362Branch]
  exact outerScan363
end ProjectEulerSolutions.PrimePairLarge.Search
