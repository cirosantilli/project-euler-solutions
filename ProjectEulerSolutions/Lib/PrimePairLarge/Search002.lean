import ProjectEulerSolutions.Lib.PrimePairLarge.Search001
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot666
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot667
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot668
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot669
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot670
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot671
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot672
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot673
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan673 : scan edge (search edge 4) 5 [] 0 tail673 winner = winner := by
  rw [tail673, scan, if_neg (by decide), root673Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail674
    (if 4 ≤ root673Candidates.length then search edge 4 [5051] root673Candidates 5051 winner else winner) = winner
  rw [if_pos (by decide), root673Branch]
  exact outerScan674
set_option maxRecDepth 4096 in
theorem outerScan672 : scan edge (search edge 4) 5 [] 0 tail672 winner = winner := by
  rw [tail672, scan, if_neg (by decide), root672Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail673
    (if 4 ≤ root672Candidates.length then search edge 4 [5039] root672Candidates 5039 winner else winner) = winner
  rw [if_pos (by decide), root672Branch]
  exact outerScan673
set_option maxRecDepth 4096 in
theorem outerScan671 : scan edge (search edge 4) 5 [] 0 tail671 winner = winner := by
  rw [tail671, scan, if_neg (by decide), root671Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail672
    (if 4 ≤ root671Candidates.length then search edge 4 [5023] root671Candidates 5023 winner else winner) = winner
  rw [if_pos (by decide), root671Branch]
  exact outerScan672
set_option maxRecDepth 4096 in
theorem outerScan670 : scan edge (search edge 4) 5 [] 0 tail670 winner = winner := by
  rw [tail670, scan, if_neg (by decide), root670Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail671
    (if 4 ≤ root670Candidates.length then search edge 4 [5021] root670Candidates 5021 winner else winner) = winner
  rw [if_pos (by decide), root670Branch]
  exact outerScan671
set_option maxRecDepth 4096 in
theorem outerScan669 : scan edge (search edge 4) 5 [] 0 tail669 winner = winner := by
  rw [tail669, scan, if_neg (by decide), root669Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail670
    (if 4 ≤ root669Candidates.length then search edge 4 [5011] root669Candidates 5011 winner else winner) = winner
  rw [if_pos (by decide), root669Branch]
  exact outerScan670
set_option maxRecDepth 4096 in
theorem outerScan668 : scan edge (search edge 4) 5 [] 0 tail668 winner = winner := by
  rw [tail668, scan, if_neg (by decide), root668Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail669
    (if 4 ≤ root668Candidates.length then search edge 4 [5009] root668Candidates 5009 winner else winner) = winner
  rw [if_pos (by decide), root668Branch]
  exact outerScan669
set_option maxRecDepth 4096 in
theorem outerScan667 : scan edge (search edge 4) 5 [] 0 tail667 winner = winner := by
  rw [tail667, scan, if_neg (by decide), root667Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail668
    (if 4 ≤ root667Candidates.length then search edge 4 [5003] root667Candidates 5003 winner else winner) = winner
  rw [if_pos (by decide), root667Branch]
  exact outerScan668
set_option maxRecDepth 4096 in
theorem outerScan666 : scan edge (search edge 4) 5 [] 0 tail666 winner = winner := by
  rw [tail666, scan, if_neg (by decide), root666Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail667
    (if 4 ≤ root666Candidates.length then search edge 4 [4999] root666Candidates 4999 winner else winner) = winner
  rw [if_pos (by decide), root666Branch]
  exact outerScan667
end ProjectEulerSolutions.PrimePairLarge.Search
