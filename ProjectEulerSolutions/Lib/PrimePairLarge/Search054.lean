import ProjectEulerSolutions.Lib.PrimePairLarge.Search053
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot250
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot251
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot252
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot253
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot254
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot255
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot256
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot257
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan257 : scan edge (search edge 4) 5 [] 0 tail257 winner = winner := by
  rw [tail257, scan, if_neg (by decide), root257Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail258
    (if 4 ≤ root257Candidates.length then search edge 4 [1657] root257Candidates 1657 winner else winner) = winner
  rw [if_pos (by decide), root257Branch]
  exact outerScan258
set_option maxRecDepth 4096 in
theorem outerScan256 : scan edge (search edge 4) 5 [] 0 tail256 winner = winner := by
  rw [tail256, scan, if_neg (by decide), root256Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail257
    (if 4 ≤ root256Candidates.length then search edge 4 [1637] root256Candidates 1637 winner else winner) = winner
  rw [if_pos (by decide), root256Branch]
  exact outerScan257
set_option maxRecDepth 4096 in
theorem outerScan255 : scan edge (search edge 4) 5 [] 0 tail255 winner = winner := by
  rw [tail255, scan, if_neg (by decide), root255Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail256
    (if 4 ≤ root255Candidates.length then search edge 4 [1627] root255Candidates 1627 winner else winner) = winner
  rw [if_pos (by decide), root255Branch]
  exact outerScan256
set_option maxRecDepth 4096 in
theorem outerScan254 : scan edge (search edge 4) 5 [] 0 tail254 winner = winner := by
  rw [tail254, scan, if_neg (by decide), root254Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail255
    (if 4 ≤ root254Candidates.length then search edge 4 [1621] root254Candidates 1621 winner else winner) = winner
  rw [if_pos (by decide), root254Branch]
  exact outerScan255
set_option maxRecDepth 4096 in
theorem outerScan253 : scan edge (search edge 4) 5 [] 0 tail253 winner = winner := by
  rw [tail253, scan, if_neg (by decide), root253Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail254
    (if 4 ≤ root253Candidates.length then search edge 4 [1619] root253Candidates 1619 winner else winner) = winner
  rw [if_pos (by decide), root253Branch]
  exact outerScan254
set_option maxRecDepth 4096 in
theorem outerScan252 : scan edge (search edge 4) 5 [] 0 tail252 winner = winner := by
  rw [tail252, scan, if_neg (by decide), root252Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail253
    (if 4 ≤ root252Candidates.length then search edge 4 [1613] root252Candidates 1613 winner else winner) = winner
  rw [if_pos (by decide), root252Branch]
  exact outerScan253
set_option maxRecDepth 4096 in
theorem outerScan251 : scan edge (search edge 4) 5 [] 0 tail251 winner = winner := by
  rw [tail251, scan, if_neg (by decide), root251Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail252
    (if 4 ≤ root251Candidates.length then search edge 4 [1609] root251Candidates 1609 winner else winner) = winner
  rw [if_pos (by decide), root251Branch]
  exact outerScan252
set_option maxRecDepth 4096 in
theorem outerScan250 : scan edge (search edge 4) 5 [] 0 tail250 winner = winner := by
  rw [tail250, scan, if_neg (by decide), root250Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail251
    (if 4 ≤ root250Candidates.length then search edge 4 [1607] root250Candidates 1607 winner else winner) = winner
  rw [if_pos (by decide), root250Branch]
  exact outerScan251
end ProjectEulerSolutions.PrimePairLarge.Search
