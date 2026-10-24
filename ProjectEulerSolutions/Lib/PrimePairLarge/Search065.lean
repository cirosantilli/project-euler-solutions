import ProjectEulerSolutions.Lib.PrimePairLarge.Search064
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot162
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot163
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot164
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot165
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot166
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot167
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot168
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot169
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan169 : scan edge (search edge 4) 5 [] 0 tail169 winner = winner := by
  rw [tail169, scan, if_neg (by decide), root169Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail170
    (if 4 ≤ root169Candidates.length then search edge 4 [1021] root169Candidates 1021 winner else winner) = winner
  rw [if_pos (by decide), root169Branch]
  exact outerScan170
set_option maxRecDepth 4096 in
theorem outerScan168 : scan edge (search edge 4) 5 [] 0 tail168 winner = winner := by
  rw [tail168, scan, if_neg (by decide), root168Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail169
    (if 4 ≤ root168Candidates.length then search edge 4 [1019] root168Candidates 1019 winner else winner) = winner
  rw [if_pos (by decide), root168Branch]
  exact outerScan169
set_option maxRecDepth 4096 in
theorem outerScan167 : scan edge (search edge 4) 5 [] 0 tail167 winner = winner := by
  rw [tail167, scan, if_neg (by decide), root167Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail168
    (if 4 ≤ root167Candidates.length then search edge 4 [1013] root167Candidates 1013 winner else winner) = winner
  rw [if_pos (by decide), root167Branch]
  exact outerScan168
set_option maxRecDepth 4096 in
theorem outerScan166 : scan edge (search edge 4) 5 [] 0 tail166 winner = winner := by
  rw [tail166, scan, if_neg (by decide), root166Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail167
    (if 4 ≤ root166Candidates.length then search edge 4 [1009] root166Candidates 1009 winner else winner) = winner
  rw [if_pos (by decide), root166Branch]
  exact outerScan167
set_option maxRecDepth 4096 in
theorem outerScan165 : scan edge (search edge 4) 5 [] 0 tail165 winner = winner := by
  rw [tail165, scan, if_neg (by decide), root165Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail166
    (if 4 ≤ root165Candidates.length then search edge 4 [997] root165Candidates 997 winner else winner) = winner
  rw [if_pos (by decide), root165Branch]
  exact outerScan166
set_option maxRecDepth 4096 in
theorem outerScan164 : scan edge (search edge 4) 5 [] 0 tail164 winner = winner := by
  rw [tail164, scan, if_neg (by decide), root164Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail165
    (if 4 ≤ root164Candidates.length then search edge 4 [991] root164Candidates 991 winner else winner) = winner
  rw [if_pos (by decide), root164Branch]
  exact outerScan165
set_option maxRecDepth 4096 in
theorem outerScan163 : scan edge (search edge 4) 5 [] 0 tail163 winner = winner := by
  rw [tail163, scan, if_neg (by decide), root163Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail164
    (if 4 ≤ root163Candidates.length then search edge 4 [983] root163Candidates 983 winner else winner) = winner
  rw [if_pos (by decide), root163Branch]
  exact outerScan164
set_option maxRecDepth 4096 in
theorem outerScan162 : scan edge (search edge 4) 5 [] 0 tail162 winner = winner := by
  rw [tail162, scan, if_neg (by decide), root162Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail163
    (if 4 ≤ root162Candidates.length then search edge 4 [977] root162Candidates 977 winner else winner) = winner
  rw [if_pos (by decide), root162Branch]
  exact outerScan163
end ProjectEulerSolutions.PrimePairLarge.Search
