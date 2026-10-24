import ProjectEulerSolutions.Lib.PrimePairLarge.Search012
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot578
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot579
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot580
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot581
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot582
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot583
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot584
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot585
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan585 : scan edge (search edge 4) 5 [] 0 tail585 winner = winner := by
  rw [tail585, scan, if_neg (by decide), root585Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail586
    (if 4 ≤ root585Candidates.length then search edge 4 [4283] root585Candidates 4283 winner else winner) = winner
  rw [if_pos (by decide), root585Branch]
  exact outerScan586
set_option maxRecDepth 4096 in
theorem outerScan584 : scan edge (search edge 4) 5 [] 0 tail584 winner = winner := by
  rw [tail584, scan, if_neg (by decide), root584Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail585
    (if 4 ≤ root584Candidates.length then search edge 4 [4273] root584Candidates 4273 winner else winner) = winner
  rw [if_pos (by decide), root584Branch]
  exact outerScan585
set_option maxRecDepth 4096 in
theorem outerScan583 : scan edge (search edge 4) 5 [] 0 tail583 winner = winner := by
  rw [tail583, scan, if_neg (by decide), root583Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail584
    (if 4 ≤ root583Candidates.length then search edge 4 [4271] root583Candidates 4271 winner else winner) = winner
  rw [if_pos (by decide), root583Branch]
  exact outerScan584
set_option maxRecDepth 4096 in
theorem outerScan582 : scan edge (search edge 4) 5 [] 0 tail582 winner = winner := by
  rw [tail582, scan, if_neg (by decide), root582Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail583
    (if 4 ≤ root582Candidates.length then search edge 4 [4261] root582Candidates 4261 winner else winner) = winner
  rw [if_pos (by decide), root582Branch]
  exact outerScan583
set_option maxRecDepth 4096 in
theorem outerScan581 : scan edge (search edge 4) 5 [] 0 tail581 winner = winner := by
  rw [tail581, scan, if_neg (by decide), root581Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail582
    (if 4 ≤ root581Candidates.length then search edge 4 [4259] root581Candidates 4259 winner else winner) = winner
  rw [if_pos (by decide), root581Branch]
  exact outerScan582
set_option maxRecDepth 4096 in
theorem outerScan580 : scan edge (search edge 4) 5 [] 0 tail580 winner = winner := by
  rw [tail580, scan, if_neg (by decide), root580Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail581
    (if 4 ≤ root580Candidates.length then search edge 4 [4253] root580Candidates 4253 winner else winner) = winner
  rw [if_pos (by decide), root580Branch]
  exact outerScan581
set_option maxRecDepth 4096 in
theorem outerScan579 : scan edge (search edge 4) 5 [] 0 tail579 winner = winner := by
  rw [tail579, scan, if_neg (by decide), root579Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail580
    (if 4 ≤ root579Candidates.length then search edge 4 [4243] root579Candidates 4243 winner else winner) = winner
  rw [if_pos (by decide), root579Branch]
  exact outerScan580
set_option maxRecDepth 4096 in
theorem outerScan578 : scan edge (search edge 4) 5 [] 0 tail578 winner = winner := by
  rw [tail578, scan, if_neg (by decide), root578Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail579
    (if 4 ≤ root578Candidates.length then search edge 4 [4241] root578Candidates 4241 winner else winner) = winner
  rw [if_pos (by decide), root578Branch]
  exact outerScan579
end ProjectEulerSolutions.PrimePairLarge.Search
