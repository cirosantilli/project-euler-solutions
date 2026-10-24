import ProjectEulerSolutions.Lib.PrimePairLarge.Search045
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot314
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot315
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot316
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot317
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot318
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot319
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot320
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot321
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan321 : scan edge (search edge 4) 5 [] 0 tail321 winner = winner := by
  rw [tail321, scan, if_neg (by decide), root321Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail322
    (if 4 ≤ root321Candidates.length then search edge 4 [2143] root321Candidates 2143 winner else winner) = winner
  rw [if_pos (by decide), root321Branch]
  exact outerScan322
set_option maxRecDepth 4096 in
theorem outerScan320 : scan edge (search edge 4) 5 [] 0 tail320 winner = winner := by
  rw [tail320, scan, if_neg (by decide), root320Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail321
    (if 4 ≤ root320Candidates.length then search edge 4 [2141] root320Candidates 2141 winner else winner) = winner
  rw [if_pos (by decide), root320Branch]
  exact outerScan321
set_option maxRecDepth 4096 in
theorem outerScan319 : scan edge (search edge 4) 5 [] 0 tail319 winner = winner := by
  rw [tail319, scan, if_neg (by decide), root319Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail320
    (if 4 ≤ root319Candidates.length then search edge 4 [2137] root319Candidates 2137 winner else winner) = winner
  rw [if_pos (by decide), root319Branch]
  exact outerScan320
set_option maxRecDepth 4096 in
theorem outerScan318 : scan edge (search edge 4) 5 [] 0 tail318 winner = winner := by
  rw [tail318, scan, if_neg (by decide), root318Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail319
    (if 4 ≤ root318Candidates.length then search edge 4 [2131] root318Candidates 2131 winner else winner) = winner
  rw [if_pos (by decide), root318Branch]
  exact outerScan319
set_option maxRecDepth 4096 in
theorem outerScan317 : scan edge (search edge 4) 5 [] 0 tail317 winner = winner := by
  rw [tail317, scan, if_neg (by decide), root317Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail318
    (if 4 ≤ root317Candidates.length then search edge 4 [2129] root317Candidates 2129 winner else winner) = winner
  rw [if_pos (by decide), root317Branch]
  exact outerScan318
set_option maxRecDepth 4096 in
theorem outerScan316 : scan edge (search edge 4) 5 [] 0 tail316 winner = winner := by
  rw [tail316, scan, if_neg (by decide), root316Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail317
    (if 4 ≤ root316Candidates.length then search edge 4 [2113] root316Candidates 2113 winner else winner) = winner
  rw [if_pos (by decide), root316Branch]
  exact outerScan317
set_option maxRecDepth 4096 in
theorem outerScan315 : scan edge (search edge 4) 5 [] 0 tail315 winner = winner := by
  rw [tail315, scan, if_neg (by decide), root315Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail316
    (if 4 ≤ root315Candidates.length then search edge 4 [2111] root315Candidates 2111 winner else winner) = winner
  rw [if_pos (by decide), root315Branch]
  exact outerScan316
set_option maxRecDepth 4096 in
theorem outerScan314 : scan edge (search edge 4) 5 [] 0 tail314 winner = winner := by
  rw [tail314, scan, if_neg (by decide), root314Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail315
    (if 4 ≤ root314Candidates.length then search edge 4 [2099] root314Candidates 2099 winner else winner) = winner
  rw [if_pos (by decide), root314Branch]
  exact outerScan315
end ProjectEulerSolutions.PrimePairLarge.Search
