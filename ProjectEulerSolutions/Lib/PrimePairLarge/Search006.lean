import ProjectEulerSolutions.Lib.PrimePairLarge.Search005
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot634
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot635
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot636
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot637
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot638
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot639
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot640
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot641
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan641 : scan edge (search edge 4) 5 [] 0 tail641 winner = winner := by
  rw [tail641, scan, if_neg (by decide), root641Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail642
    (if 4 ≤ root641Candidates.length then search edge 4 [4789] root641Candidates 4789 winner else winner) = winner
  rw [if_pos (by decide), root641Branch]
  exact outerScan642
set_option maxRecDepth 4096 in
theorem outerScan640 : scan edge (search edge 4) 5 [] 0 tail640 winner = winner := by
  rw [tail640, scan, if_neg (by decide), root640Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail641
    (if 4 ≤ root640Candidates.length then search edge 4 [4787] root640Candidates 4787 winner else winner) = winner
  rw [if_pos (by decide), root640Branch]
  exact outerScan641
set_option maxRecDepth 4096 in
theorem outerScan639 : scan edge (search edge 4) 5 [] 0 tail639 winner = winner := by
  rw [tail639, scan, if_neg (by decide), root639Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail640
    (if 4 ≤ root639Candidates.length then search edge 4 [4783] root639Candidates 4783 winner else winner) = winner
  rw [if_pos (by decide), root639Branch]
  exact outerScan640
set_option maxRecDepth 4096 in
theorem outerScan638 : scan edge (search edge 4) 5 [] 0 tail638 winner = winner := by
  rw [tail638, scan, if_neg (by decide), root638Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail639
    (if 4 ≤ root638Candidates.length then search edge 4 [4759] root638Candidates 4759 winner else winner) = winner
  rw [if_pos (by decide), root638Branch]
  exact outerScan639
set_option maxRecDepth 4096 in
theorem outerScan637 : scan edge (search edge 4) 5 [] 0 tail637 winner = winner := by
  rw [tail637, scan, if_neg (by decide), root637Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail638
    (if 4 ≤ root637Candidates.length then search edge 4 [4751] root637Candidates 4751 winner else winner) = winner
  rw [if_pos (by decide), root637Branch]
  exact outerScan638
set_option maxRecDepth 4096 in
theorem outerScan636 : scan edge (search edge 4) 5 [] 0 tail636 winner = winner := by
  rw [tail636, scan, if_neg (by decide), root636Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail637
    (if 4 ≤ root636Candidates.length then search edge 4 [4733] root636Candidates 4733 winner else winner) = winner
  rw [if_pos (by decide), root636Branch]
  exact outerScan637
set_option maxRecDepth 4096 in
theorem outerScan635 : scan edge (search edge 4) 5 [] 0 tail635 winner = winner := by
  rw [tail635, scan, if_neg (by decide), root635Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail636
    (if 4 ≤ root635Candidates.length then search edge 4 [4729] root635Candidates 4729 winner else winner) = winner
  rw [if_pos (by decide), root635Branch]
  exact outerScan636
set_option maxRecDepth 4096 in
theorem outerScan634 : scan edge (search edge 4) 5 [] 0 tail634 winner = winner := by
  rw [tail634, scan, if_neg (by decide), root634Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail635
    (if 4 ≤ root634Candidates.length then search edge 4 [4723] root634Candidates 4723 winner else winner) = winner
  rw [if_pos (by decide), root634Branch]
  exact outerScan635
end ProjectEulerSolutions.PrimePairLarge.Search
