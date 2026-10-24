import ProjectEulerSolutions.Lib.PrimePairLarge.Search081
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot026
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot027
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot028
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot029
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot030
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot031
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot032
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot033
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan33 : scan edge (search edge 4) 5 [] 0 tail33 winner = winner := by
  rw [tail33, scan, if_neg (by decide), root33Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail34
    (if 4 ≤ root33Candidates.length then search edge 4 [151] root33Candidates 151 winner else winner) = winner
  rw [if_pos (by decide), root33Branch]
  exact outerScan34
set_option maxRecDepth 4096 in
theorem outerScan32 : scan edge (search edge 4) 5 [] 0 tail32 winner = winner := by
  rw [tail32, scan, if_neg (by decide), root32Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail33
    (if 4 ≤ root32Candidates.length then search edge 4 [149] root32Candidates 149 winner else winner) = winner
  rw [if_pos (by decide), root32Branch]
  exact outerScan33
set_option maxRecDepth 4096 in
theorem outerScan31 : scan edge (search edge 4) 5 [] 0 tail31 winner = winner := by
  rw [tail31, scan, if_neg (by decide), root31Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail32
    (if 4 ≤ root31Candidates.length then search edge 4 [139] root31Candidates 139 winner else winner) = winner
  rw [if_pos (by decide), root31Branch]
  exact outerScan32
set_option maxRecDepth 4096 in
theorem outerScan30 : scan edge (search edge 4) 5 [] 0 tail30 winner = winner := by
  rw [tail30, scan, if_neg (by decide), root30Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail31
    (if 4 ≤ root30Candidates.length then search edge 4 [137] root30Candidates 137 winner else winner) = winner
  rw [if_pos (by decide), root30Branch]
  exact outerScan31
set_option maxRecDepth 4096 in
theorem outerScan29 : scan edge (search edge 4) 5 [] 0 tail29 winner = winner := by
  rw [tail29, scan, if_neg (by decide), root29Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail30
    (if 4 ≤ root29Candidates.length then search edge 4 [131] root29Candidates 131 winner else winner) = winner
  rw [if_pos (by decide), root29Branch]
  exact outerScan30
set_option maxRecDepth 4096 in
theorem outerScan28 : scan edge (search edge 4) 5 [] 0 tail28 winner = winner := by
  rw [tail28, scan, if_neg (by decide), root28Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail29
    (if 4 ≤ root28Candidates.length then search edge 4 [127] root28Candidates 127 winner else winner) = winner
  rw [if_pos (by decide), root28Branch]
  exact outerScan29
set_option maxRecDepth 4096 in
theorem outerScan27 : scan edge (search edge 4) 5 [] 0 tail27 winner = winner := by
  rw [tail27, scan, if_neg (by decide), root27Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail28
    (if 4 ≤ root27Candidates.length then search edge 4 [113] root27Candidates 113 winner else winner) = winner
  rw [if_pos (by decide), root27Branch]
  exact outerScan28
set_option maxRecDepth 4096 in
theorem outerScan26 : scan edge (search edge 4) 5 [] 0 tail26 winner = winner := by
  rw [tail26, scan, if_neg (by decide), root26Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail27
    (if 4 ≤ root26Candidates.length then search edge 4 [109] root26Candidates 109 winner else winner) = winner
  rw [if_pos (by decide), root26Branch]
  exact outerScan27
end ProjectEulerSolutions.PrimePairLarge.Search
