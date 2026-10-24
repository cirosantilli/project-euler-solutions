import ProjectEulerSolutions.Lib.PrimePairLarge.Search079
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot042
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot043
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot044
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot045
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot046
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot047
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot048
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot049
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan49 : scan edge (search edge 4) 5 [] 0 tail49 winner = winner := by
  rw [tail49, scan, if_neg (by decide), root49Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail50
    (if 4 ≤ root49Candidates.length then search edge 4 [239] root49Candidates 239 winner else winner) = winner
  rw [if_pos (by decide), root49Branch]
  exact outerScan50
set_option maxRecDepth 4096 in
theorem outerScan48 : scan edge (search edge 4) 5 [] 0 tail48 winner = winner := by
  rw [tail48, scan, if_neg (by decide), root48Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail49
    (if 4 ≤ root48Candidates.length then search edge 4 [233] root48Candidates 233 winner else winner) = winner
  rw [if_pos (by decide), root48Branch]
  exact outerScan49
set_option maxRecDepth 4096 in
theorem outerScan47 : scan edge (search edge 4) 5 [] 0 tail47 winner = winner := by
  rw [tail47, scan, if_neg (by decide), root47Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail48
    (if 4 ≤ root47Candidates.length then search edge 4 [229] root47Candidates 229 winner else winner) = winner
  rw [if_pos (by decide), root47Branch]
  exact outerScan48
set_option maxRecDepth 4096 in
theorem outerScan46 : scan edge (search edge 4) 5 [] 0 tail46 winner = winner := by
  rw [tail46, scan, if_neg (by decide), root46Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail47
    (if 4 ≤ root46Candidates.length then search edge 4 [227] root46Candidates 227 winner else winner) = winner
  rw [if_pos (by decide), root46Branch]
  exact outerScan47
set_option maxRecDepth 4096 in
theorem outerScan45 : scan edge (search edge 4) 5 [] 0 tail45 winner = winner := by
  rw [tail45, scan, if_neg (by decide), root45Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail46
    (if 4 ≤ root45Candidates.length then search edge 4 [223] root45Candidates 223 winner else winner) = winner
  rw [if_pos (by decide), root45Branch]
  exact outerScan46
set_option maxRecDepth 4096 in
theorem outerScan44 : scan edge (search edge 4) 5 [] 0 tail44 winner = winner := by
  rw [tail44, scan, if_neg (by decide), root44Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail45
    (if 4 ≤ root44Candidates.length then search edge 4 [211] root44Candidates 211 winner else winner) = winner
  rw [if_pos (by decide), root44Branch]
  exact outerScan45
set_option maxRecDepth 4096 in
theorem outerScan43 : scan edge (search edge 4) 5 [] 0 tail43 winner = winner := by
  rw [tail43, scan, if_neg (by decide), root43Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail44
    (if 4 ≤ root43Candidates.length then search edge 4 [199] root43Candidates 199 winner else winner) = winner
  rw [if_pos (by decide), root43Branch]
  exact outerScan44
set_option maxRecDepth 4096 in
theorem outerScan42 : scan edge (search edge 4) 5 [] 0 tail42 winner = winner := by
  rw [tail42, scan, if_neg (by decide), root42Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail43
    (if 4 ≤ root42Candidates.length then search edge 4 [197] root42Candidates 197 winner else winner) = winner
  rw [if_pos (by decide), root42Branch]
  exact outerScan43
end ProjectEulerSolutions.PrimePairLarge.Search
