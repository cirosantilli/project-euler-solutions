import ProjectEulerSolutions.Lib.PrimePairLarge.Search063
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot170
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot171
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot172
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot173
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot174
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot175
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot176
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot177
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan177 : scan edge (search edge 4) 5 [] 0 tail177 winner = winner := by
  rw [tail177, scan, if_neg (by decide), root177Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail178
    (if 4 ≤ root177Candidates.length then search edge 4 [1069] root177Candidates 1069 winner else winner) = winner
  rw [if_pos (by decide), root177Branch]
  exact outerScan178
set_option maxRecDepth 4096 in
theorem outerScan176 : scan edge (search edge 4) 5 [] 0 tail176 winner = winner := by
  rw [tail176, scan, if_neg (by decide), root176Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail177
    (if 4 ≤ root176Candidates.length then search edge 4 [1063] root176Candidates 1063 winner else winner) = winner
  rw [if_pos (by decide), root176Branch]
  exact outerScan177
set_option maxRecDepth 4096 in
theorem outerScan175 : scan edge (search edge 4) 5 [] 0 tail175 winner = winner := by
  rw [tail175, scan, if_neg (by decide), root175Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail176
    (if 4 ≤ root175Candidates.length then search edge 4 [1061] root175Candidates 1061 winner else winner) = winner
  rw [if_pos (by decide), root175Branch]
  exact outerScan176
set_option maxRecDepth 4096 in
theorem outerScan174 : scan edge (search edge 4) 5 [] 0 tail174 winner = winner := by
  rw [tail174, scan, if_neg (by decide), root174Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail175
    (if 4 ≤ root174Candidates.length then search edge 4 [1051] root174Candidates 1051 winner else winner) = winner
  rw [if_pos (by decide), root174Branch]
  exact outerScan175
set_option maxRecDepth 4096 in
theorem outerScan173 : scan edge (search edge 4) 5 [] 0 tail173 winner = winner := by
  rw [tail173, scan, if_neg (by decide), root173Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail174
    (if 4 ≤ root173Candidates.length then search edge 4 [1049] root173Candidates 1049 winner else winner) = winner
  rw [if_pos (by decide), root173Branch]
  exact outerScan174
set_option maxRecDepth 4096 in
theorem outerScan172 : scan edge (search edge 4) 5 [] 0 tail172 winner = winner := by
  rw [tail172, scan, if_neg (by decide), root172Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail173
    (if 4 ≤ root172Candidates.length then search edge 4 [1039] root172Candidates 1039 winner else winner) = winner
  rw [if_pos (by decide), root172Branch]
  exact outerScan173
set_option maxRecDepth 4096 in
theorem outerScan171 : scan edge (search edge 4) 5 [] 0 tail171 winner = winner := by
  rw [tail171, scan, if_neg (by decide), root171Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail172
    (if 4 ≤ root171Candidates.length then search edge 4 [1033] root171Candidates 1033 winner else winner) = winner
  rw [if_pos (by decide), root171Branch]
  exact outerScan172
set_option maxRecDepth 4096 in
theorem outerScan170 : scan edge (search edge 4) 5 [] 0 tail170 winner = winner := by
  rw [tail170, scan, if_neg (by decide), root170Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail171
    (if 4 ≤ root170Candidates.length then search edge 4 [1031] root170Candidates 1031 winner else winner) = winner
  rw [if_pos (by decide), root170Branch]
  exact outerScan171
end ProjectEulerSolutions.PrimePairLarge.Search
