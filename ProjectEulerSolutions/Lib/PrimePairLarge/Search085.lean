import ProjectEulerSolutions.Lib.PrimePairLarge.Search084
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot002
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot003
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot004
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot005
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot006
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot007
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot008
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot009
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan9 : scan edge (search edge 4) 5 [] 0 tail9 winner = winner := by
  rw [tail9, scan, if_neg (by decide), root9Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail10
    (if 4 ≤ root9Candidates.length then search edge 4 [37] root9Candidates 37 winner else winner) = winner
  rw [if_pos (by decide), root9Branch]
  exact outerScan10
set_option maxRecDepth 4096 in
theorem outerScan8 : scan edge (search edge 4) 5 [] 0 tail8 winner = winner := by
  rw [tail8, scan, if_neg (by decide), root8Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail9
    (if 4 ≤ root8Candidates.length then search edge 4 [31] root8Candidates 31 winner else winner) = winner
  rw [if_pos (by decide), root8Branch]
  exact outerScan9
set_option maxRecDepth 4096 in
theorem outerScan7 : scan edge (search edge 4) 5 [] 0 tail7 winner = winner := by
  rw [tail7, scan, if_neg (by decide), root7Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail8
    (if 4 ≤ root7Candidates.length then search edge 4 [29] root7Candidates 29 winner else winner) = winner
  rw [if_pos (by decide), root7Branch]
  exact outerScan8
set_option maxRecDepth 4096 in
theorem outerScan6 : scan edge (search edge 4) 5 [] 0 tail6 winner = winner := by
  rw [tail6, scan, if_neg (by decide), root6Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail7
    (if 4 ≤ root6Candidates.length then search edge 4 [23] root6Candidates 23 winner else winner) = winner
  rw [if_pos (by decide), root6Branch]
  exact outerScan7
set_option maxRecDepth 4096 in
theorem outerScan5 : scan edge (search edge 4) 5 [] 0 tail5 winner = winner := by
  rw [tail5, scan, if_neg (by decide), root5Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail6
    (if 4 ≤ root5Candidates.length then search edge 4 [19] root5Candidates 19 winner else winner) = winner
  rw [if_pos (by decide), root5Branch]
  exact outerScan6
set_option maxRecDepth 4096 in
theorem outerScan4 : scan edge (search edge 4) 5 [] 0 tail4 winner = winner := by
  rw [tail4, scan, if_neg (by decide), root4Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail5
    (if 4 ≤ root4Candidates.length then search edge 4 [17] root4Candidates 17 winner else winner) = winner
  rw [if_pos (by decide), root4Branch]
  exact outerScan5
set_option maxRecDepth 4096 in
theorem outerScan3 : scan edge (search edge 4) 5 [] 0 tail3 seed = winner := by
  rw [tail3, scan, if_neg (by decide), root3Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail4
    (if 4 ≤ root3Candidates.length then search edge 4 [13] root3Candidates 13 seed else seed) = winner
  rw [if_pos (by decide), root3Branch]
  exact outerScan4
set_option maxRecDepth 4096 in
theorem outerScan2 : scan edge (search edge 4) 5 [] 0 tail2 seed = winner := by
  rw [tail2, scan, if_neg (by decide), root2Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail3
    (if 4 ≤ root2Candidates.length then search edge 4 [11] root2Candidates 11 seed else seed) = winner
  rw [if_pos (by decide), root2Branch]
  exact outerScan3
end ProjectEulerSolutions.PrimePairLarge.Search
