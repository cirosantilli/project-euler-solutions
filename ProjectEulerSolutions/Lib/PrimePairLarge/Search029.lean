import ProjectEulerSolutions.Lib.PrimePairLarge.Search028
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot450
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot451
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot452
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot453
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot454
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot455
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot456
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot457
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan457 : scan edge (search edge 4) 5 [] 0 tail457 winner = winner := by
  rw [tail457, scan, if_neg (by decide), root457Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail458
    (if 4 ≤ root457Candidates.length then search edge 4 [3257] root457Candidates 3257 winner else winner) = winner
  rw [if_pos (by decide), root457Branch]
  exact outerScan458
set_option maxRecDepth 4096 in
theorem outerScan456 : scan edge (search edge 4) 5 [] 0 tail456 winner = winner := by
  rw [tail456, scan, if_neg (by decide), root456Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail457
    (if 4 ≤ root456Candidates.length then search edge 4 [3253] root456Candidates 3253 winner else winner) = winner
  rw [if_pos (by decide), root456Branch]
  exact outerScan457
set_option maxRecDepth 4096 in
theorem outerScan455 : scan edge (search edge 4) 5 [] 0 tail455 winner = winner := by
  rw [tail455, scan, if_neg (by decide), root455Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail456
    (if 4 ≤ root455Candidates.length then search edge 4 [3251] root455Candidates 3251 winner else winner) = winner
  rw [if_pos (by decide), root455Branch]
  exact outerScan456
set_option maxRecDepth 4096 in
theorem outerScan454 : scan edge (search edge 4) 5 [] 0 tail454 winner = winner := by
  rw [tail454, scan, if_neg (by decide), root454Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail455
    (if 4 ≤ root454Candidates.length then search edge 4 [3229] root454Candidates 3229 winner else winner) = winner
  rw [if_pos (by decide), root454Branch]
  exact outerScan455
set_option maxRecDepth 4096 in
theorem outerScan453 : scan edge (search edge 4) 5 [] 0 tail453 winner = winner := by
  rw [tail453, scan, if_neg (by decide), root453Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail454
    (if 4 ≤ root453Candidates.length then search edge 4 [3221] root453Candidates 3221 winner else winner) = winner
  rw [if_pos (by decide), root453Branch]
  exact outerScan454
set_option maxRecDepth 4096 in
theorem outerScan452 : scan edge (search edge 4) 5 [] 0 tail452 winner = winner := by
  rw [tail452, scan, if_neg (by decide), root452Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail453
    (if 4 ≤ root452Candidates.length then search edge 4 [3217] root452Candidates 3217 winner else winner) = winner
  rw [if_pos (by decide), root452Branch]
  exact outerScan453
set_option maxRecDepth 4096 in
theorem outerScan451 : scan edge (search edge 4) 5 [] 0 tail451 winner = winner := by
  rw [tail451, scan, if_neg (by decide), root451Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail452
    (if 4 ≤ root451Candidates.length then search edge 4 [3209] root451Candidates 3209 winner else winner) = winner
  rw [if_pos (by decide), root451Branch]
  exact outerScan452
set_option maxRecDepth 4096 in
theorem outerScan450 : scan edge (search edge 4) 5 [] 0 tail450 winner = winner := by
  rw [tail450, scan, if_neg (by decide), root450Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail451
    (if 4 ≤ root450Candidates.length then search edge 4 [3203] root450Candidates 3203 winner else winner) = winner
  rw [if_pos (by decide), root450Branch]
  exact outerScan451
end ProjectEulerSolutions.PrimePairLarge.Search
