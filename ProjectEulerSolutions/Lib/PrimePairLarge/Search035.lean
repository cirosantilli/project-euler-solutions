import ProjectEulerSolutions.Lib.PrimePairLarge.Search034
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot402
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot403
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot404
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot405
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot406
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot407
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot408
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot409
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan409 : scan edge (search edge 4) 5 [] 0 tail409 winner = winner := by
  rw [tail409, scan, if_neg (by decide), root409Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail410
    (if 4 ≤ root409Candidates.length then search edge 4 [2837] root409Candidates 2837 winner else winner) = winner
  rw [if_pos (by decide), root409Branch]
  exact outerScan410
set_option maxRecDepth 4096 in
theorem outerScan408 : scan edge (search edge 4) 5 [] 0 tail408 winner = winner := by
  rw [tail408, scan, if_neg (by decide), root408Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail409
    (if 4 ≤ root408Candidates.length then search edge 4 [2833] root408Candidates 2833 winner else winner) = winner
  rw [if_pos (by decide), root408Branch]
  exact outerScan409
set_option maxRecDepth 4096 in
theorem outerScan407 : scan edge (search edge 4) 5 [] 0 tail407 winner = winner := by
  rw [tail407, scan, if_neg (by decide), root407Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail408
    (if 4 ≤ root407Candidates.length then search edge 4 [2819] root407Candidates 2819 winner else winner) = winner
  rw [if_pos (by decide), root407Branch]
  exact outerScan408
set_option maxRecDepth 4096 in
theorem outerScan406 : scan edge (search edge 4) 5 [] 0 tail406 winner = winner := by
  rw [tail406, scan, if_neg (by decide), root406Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail407
    (if 4 ≤ root406Candidates.length then search edge 4 [2803] root406Candidates 2803 winner else winner) = winner
  rw [if_pos (by decide), root406Branch]
  exact outerScan407
set_option maxRecDepth 4096 in
theorem outerScan405 : scan edge (search edge 4) 5 [] 0 tail405 winner = winner := by
  rw [tail405, scan, if_neg (by decide), root405Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail406
    (if 4 ≤ root405Candidates.length then search edge 4 [2801] root405Candidates 2801 winner else winner) = winner
  rw [if_pos (by decide), root405Branch]
  exact outerScan406
set_option maxRecDepth 4096 in
theorem outerScan404 : scan edge (search edge 4) 5 [] 0 tail404 winner = winner := by
  rw [tail404, scan, if_neg (by decide), root404Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail405
    (if 4 ≤ root404Candidates.length then search edge 4 [2797] root404Candidates 2797 winner else winner) = winner
  rw [if_pos (by decide), root404Branch]
  exact outerScan405
set_option maxRecDepth 4096 in
theorem outerScan403 : scan edge (search edge 4) 5 [] 0 tail403 winner = winner := by
  rw [tail403, scan, if_neg (by decide), root403Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail404
    (if 4 ≤ root403Candidates.length then search edge 4 [2791] root403Candidates 2791 winner else winner) = winner
  rw [if_pos (by decide), root403Branch]
  exact outerScan404
set_option maxRecDepth 4096 in
theorem outerScan402 : scan edge (search edge 4) 5 [] 0 tail402 winner = winner := by
  rw [tail402, scan, if_neg (by decide), root402Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail403
    (if 4 ≤ root402Candidates.length then search edge 4 [2789] root402Candidates 2789 winner else winner) = winner
  rw [if_pos (by decide), root402Branch]
  exact outerScan403
end ProjectEulerSolutions.PrimePairLarge.Search
