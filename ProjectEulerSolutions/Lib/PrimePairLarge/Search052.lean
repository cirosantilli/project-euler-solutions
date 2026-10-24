import ProjectEulerSolutions.Lib.PrimePairLarge.Search051
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot266
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot267
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot268
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot269
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot270
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot271
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot272
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot273
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan273 : scan edge (search edge 4) 5 [] 0 tail273 winner = winner := by
  rw [tail273, scan, if_neg (by decide), root273Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail274
    (if 4 ≤ root273Candidates.length then search edge 4 [1783] root273Candidates 1783 winner else winner) = winner
  rw [if_pos (by decide), root273Branch]
  exact outerScan274
set_option maxRecDepth 4096 in
theorem outerScan272 : scan edge (search edge 4) 5 [] 0 tail272 winner = winner := by
  rw [tail272, scan, if_neg (by decide), root272Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail273
    (if 4 ≤ root272Candidates.length then search edge 4 [1777] root272Candidates 1777 winner else winner) = winner
  rw [if_pos (by decide), root272Branch]
  exact outerScan273
set_option maxRecDepth 4096 in
theorem outerScan271 : scan edge (search edge 4) 5 [] 0 tail271 winner = winner := by
  rw [tail271, scan, if_neg (by decide), root271Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail272
    (if 4 ≤ root271Candidates.length then search edge 4 [1759] root271Candidates 1759 winner else winner) = winner
  rw [if_pos (by decide), root271Branch]
  exact outerScan272
set_option maxRecDepth 4096 in
theorem outerScan270 : scan edge (search edge 4) 5 [] 0 tail270 winner = winner := by
  rw [tail270, scan, if_neg (by decide), root270Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail271
    (if 4 ≤ root270Candidates.length then search edge 4 [1753] root270Candidates 1753 winner else winner) = winner
  rw [if_pos (by decide), root270Branch]
  exact outerScan271
set_option maxRecDepth 4096 in
theorem outerScan269 : scan edge (search edge 4) 5 [] 0 tail269 winner = winner := by
  rw [tail269, scan, if_neg (by decide), root269Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail270
    (if 4 ≤ root269Candidates.length then search edge 4 [1747] root269Candidates 1747 winner else winner) = winner
  rw [if_pos (by decide), root269Branch]
  exact outerScan270
set_option maxRecDepth 4096 in
theorem outerScan268 : scan edge (search edge 4) 5 [] 0 tail268 winner = winner := by
  rw [tail268, scan, if_neg (by decide), root268Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail269
    (if 4 ≤ root268Candidates.length then search edge 4 [1741] root268Candidates 1741 winner else winner) = winner
  rw [if_pos (by decide), root268Branch]
  exact outerScan269
set_option maxRecDepth 4096 in
theorem outerScan267 : scan edge (search edge 4) 5 [] 0 tail267 winner = winner := by
  rw [tail267, scan, if_neg (by decide), root267Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail268
    (if 4 ≤ root267Candidates.length then search edge 4 [1733] root267Candidates 1733 winner else winner) = winner
  rw [if_pos (by decide), root267Branch]
  exact outerScan268
set_option maxRecDepth 4096 in
theorem outerScan266 : scan edge (search edge 4) 5 [] 0 tail266 winner = winner := by
  rw [tail266, scan, if_neg (by decide), root266Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail267
    (if 4 ≤ root266Candidates.length then search edge 4 [1723] root266Candidates 1723 winner else winner) = winner
  rw [if_pos (by decide), root266Branch]
  exact outerScan267
end ProjectEulerSolutions.PrimePairLarge.Search
