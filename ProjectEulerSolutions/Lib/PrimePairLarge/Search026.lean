import ProjectEulerSolutions.Lib.PrimePairLarge.Search025
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot474
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot475
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot476
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot477
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot478
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot479
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot480
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot481
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan481 : scan edge (search edge 4) 5 [] 0 tail481 winner = winner := by
  rw [tail481, scan, if_neg (by decide), root481Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail482
    (if 4 ≤ root481Candidates.length then search edge 4 [3461] root481Candidates 3461 winner else winner) = winner
  rw [if_pos (by decide), root481Branch]
  exact outerScan482
set_option maxRecDepth 4096 in
theorem outerScan480 : scan edge (search edge 4) 5 [] 0 tail480 winner = winner := by
  rw [tail480, scan, if_neg (by decide), root480Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail481
    (if 4 ≤ root480Candidates.length then search edge 4 [3457] root480Candidates 3457 winner else winner) = winner
  rw [if_pos (by decide), root480Branch]
  exact outerScan481
set_option maxRecDepth 4096 in
theorem outerScan479 : scan edge (search edge 4) 5 [] 0 tail479 winner = winner := by
  rw [tail479, scan, if_neg (by decide), root479Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail480
    (if 4 ≤ root479Candidates.length then search edge 4 [3449] root479Candidates 3449 winner else winner) = winner
  rw [if_pos (by decide), root479Branch]
  exact outerScan480
set_option maxRecDepth 4096 in
theorem outerScan478 : scan edge (search edge 4) 5 [] 0 tail478 winner = winner := by
  rw [tail478, scan, if_neg (by decide), root478Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail479
    (if 4 ≤ root478Candidates.length then search edge 4 [3433] root478Candidates 3433 winner else winner) = winner
  rw [if_pos (by decide), root478Branch]
  exact outerScan479
set_option maxRecDepth 4096 in
theorem outerScan477 : scan edge (search edge 4) 5 [] 0 tail477 winner = winner := by
  rw [tail477, scan, if_neg (by decide), root477Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail478
    (if 4 ≤ root477Candidates.length then search edge 4 [3413] root477Candidates 3413 winner else winner) = winner
  rw [if_pos (by decide), root477Branch]
  exact outerScan478
set_option maxRecDepth 4096 in
theorem outerScan476 : scan edge (search edge 4) 5 [] 0 tail476 winner = winner := by
  rw [tail476, scan, if_neg (by decide), root476Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail477
    (if 4 ≤ root476Candidates.length then search edge 4 [3407] root476Candidates 3407 winner else winner) = winner
  rw [if_pos (by decide), root476Branch]
  exact outerScan477
set_option maxRecDepth 4096 in
theorem outerScan475 : scan edge (search edge 4) 5 [] 0 tail475 winner = winner := by
  rw [tail475, scan, if_neg (by decide), root475Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail476
    (if 4 ≤ root475Candidates.length then search edge 4 [3391] root475Candidates 3391 winner else winner) = winner
  rw [if_pos (by decide), root475Branch]
  exact outerScan476
set_option maxRecDepth 4096 in
theorem outerScan474 : scan edge (search edge 4) 5 [] 0 tail474 winner = winner := by
  rw [tail474, scan, if_neg (by decide), root474Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail475
    (if 4 ≤ root474Candidates.length then search edge 4 [3389] root474Candidates 3389 winner else winner) = winner
  rw [if_pos (by decide), root474Branch]
  exact outerScan475
end ProjectEulerSolutions.PrimePairLarge.Search
