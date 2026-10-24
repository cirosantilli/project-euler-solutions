import ProjectEulerSolutions.Lib.PrimePairLarge.Search004
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot642
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot643
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot644
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot645
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot646
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot647
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot648
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot649
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan649 : scan edge (search edge 4) 5 [] 0 tail649 winner = winner := by
  rw [tail649, scan, if_neg (by decide), root649Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail650
    (if 4 ≤ root649Candidates.length then search edge 4 [4871] root649Candidates 4871 winner else winner) = winner
  rw [if_pos (by decide), root649Branch]
  exact outerScan650
set_option maxRecDepth 4096 in
theorem outerScan648 : scan edge (search edge 4) 5 [] 0 tail648 winner = winner := by
  rw [tail648, scan, if_neg (by decide), root648Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail649
    (if 4 ≤ root648Candidates.length then search edge 4 [4861] root648Candidates 4861 winner else winner) = winner
  rw [if_pos (by decide), root648Branch]
  exact outerScan649
set_option maxRecDepth 4096 in
theorem outerScan647 : scan edge (search edge 4) 5 [] 0 tail647 winner = winner := by
  rw [tail647, scan, if_neg (by decide), root647Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail648
    (if 4 ≤ root647Candidates.length then search edge 4 [4831] root647Candidates 4831 winner else winner) = winner
  rw [if_pos (by decide), root647Branch]
  exact outerScan648
set_option maxRecDepth 4096 in
theorem outerScan646 : scan edge (search edge 4) 5 [] 0 tail646 winner = winner := by
  rw [tail646, scan, if_neg (by decide), root646Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail647
    (if 4 ≤ root646Candidates.length then search edge 4 [4817] root646Candidates 4817 winner else winner) = winner
  rw [if_pos (by decide), root646Branch]
  exact outerScan647
set_option maxRecDepth 4096 in
theorem outerScan645 : scan edge (search edge 4) 5 [] 0 tail645 winner = winner := by
  rw [tail645, scan, if_neg (by decide), root645Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail646
    (if 4 ≤ root645Candidates.length then search edge 4 [4813] root645Candidates 4813 winner else winner) = winner
  rw [if_pos (by decide), root645Branch]
  exact outerScan646
set_option maxRecDepth 4096 in
theorem outerScan644 : scan edge (search edge 4) 5 [] 0 tail644 winner = winner := by
  rw [tail644, scan, if_neg (by decide), root644Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail645
    (if 4 ≤ root644Candidates.length then search edge 4 [4801] root644Candidates 4801 winner else winner) = winner
  rw [if_pos (by decide), root644Branch]
  exact outerScan645
set_option maxRecDepth 4096 in
theorem outerScan643 : scan edge (search edge 4) 5 [] 0 tail643 winner = winner := by
  rw [tail643, scan, if_neg (by decide), root643Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail644
    (if 4 ≤ root643Candidates.length then search edge 4 [4799] root643Candidates 4799 winner else winner) = winner
  rw [if_pos (by decide), root643Branch]
  exact outerScan644
set_option maxRecDepth 4096 in
theorem outerScan642 : scan edge (search edge 4) 5 [] 0 tail642 winner = winner := by
  rw [tail642, scan, if_neg (by decide), root642Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail643
    (if 4 ≤ root642Candidates.length then search edge 4 [4793] root642Candidates 4793 winner else winner) = winner
  rw [if_pos (by decide), root642Branch]
  exact outerScan643
end ProjectEulerSolutions.PrimePairLarge.Search
