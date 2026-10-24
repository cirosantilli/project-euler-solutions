import ProjectEulerSolutions.Lib.PrimePairLarge.SearchEnd
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot682
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot683
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot684
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot685
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot686
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot687
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot688
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot689
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan689 : scan edge (search edge 4) 5 [] 0 tail689 winner = winner := by
  rw [tail689, scan, if_neg (by decide), root689Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail690
    (if 4 ≤ root689Candidates.length then search edge 4 [5197] root689Candidates 5197 winner else winner) = winner
  rw [if_pos (by decide), root689Branch]
  exact outerScan690
set_option maxRecDepth 4096 in
theorem outerScan688 : scan edge (search edge 4) 5 [] 0 tail688 winner = winner := by
  rw [tail688, scan, if_neg (by decide), root688Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail689
    (if 4 ≤ root688Candidates.length then search edge 4 [5189] root688Candidates 5189 winner else winner) = winner
  rw [if_pos (by decide), root688Branch]
  exact outerScan689
set_option maxRecDepth 4096 in
theorem outerScan687 : scan edge (search edge 4) 5 [] 0 tail687 winner = winner := by
  rw [tail687, scan, if_neg (by decide), root687Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail688
    (if 4 ≤ root687Candidates.length then search edge 4 [5179] root687Candidates 5179 winner else winner) = winner
  rw [if_pos (by decide), root687Branch]
  exact outerScan688
set_option maxRecDepth 4096 in
theorem outerScan686 : scan edge (search edge 4) 5 [] 0 tail686 winner = winner := by
  rw [tail686, scan, if_neg (by decide), root686Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail687
    (if 4 ≤ root686Candidates.length then search edge 4 [5171] root686Candidates 5171 winner else winner) = winner
  rw [if_pos (by decide), root686Branch]
  exact outerScan687
set_option maxRecDepth 4096 in
theorem outerScan685 : scan edge (search edge 4) 5 [] 0 tail685 winner = winner := by
  rw [tail685, scan, if_neg (by decide), root685Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail686
    (if 4 ≤ root685Candidates.length then search edge 4 [5167] root685Candidates 5167 winner else winner) = winner
  rw [if_pos (by decide), root685Branch]
  exact outerScan686
set_option maxRecDepth 4096 in
theorem outerScan684 : scan edge (search edge 4) 5 [] 0 tail684 winner = winner := by
  rw [tail684, scan, if_neg (by decide), root684Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail685
    (if 4 ≤ root684Candidates.length then search edge 4 [5153] root684Candidates 5153 winner else winner) = winner
  rw [if_pos (by decide), root684Branch]
  exact outerScan685
set_option maxRecDepth 4096 in
theorem outerScan683 : scan edge (search edge 4) 5 [] 0 tail683 winner = winner := by
  rw [tail683, scan, if_neg (by decide), root683Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail684
    (if 4 ≤ root683Candidates.length then search edge 4 [5147] root683Candidates 5147 winner else winner) = winner
  rw [if_pos (by decide), root683Branch]
  exact outerScan684
set_option maxRecDepth 4096 in
theorem outerScan682 : scan edge (search edge 4) 5 [] 0 tail682 winner = winner := by
  rw [tail682, scan, if_neg (by decide), root682Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail683
    (if 4 ≤ root682Candidates.length then search edge 4 [5119] root682Candidates 5119 winner else winner) = winner
  rw [if_pos (by decide), root682Branch]
  exact outerScan683
end ProjectEulerSolutions.PrimePairLarge.Search
