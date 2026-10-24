import ProjectEulerSolutions.Lib.PrimePairLarge.Search059
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot202
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot203
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot204
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot205
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot206
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot207
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot208
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot209
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan209 : scan edge (search edge 4) 5 [] 0 tail209 winner = winner := by
  rw [tail209, scan, if_neg (by decide), root209Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail210
    (if 4 ≤ root209Candidates.length then search edge 4 [1301] root209Candidates 1301 winner else winner) = winner
  rw [if_pos (by decide), root209Branch]
  exact outerScan210
set_option maxRecDepth 4096 in
theorem outerScan208 : scan edge (search edge 4) 5 [] 0 tail208 winner = winner := by
  rw [tail208, scan, if_neg (by decide), root208Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail209
    (if 4 ≤ root208Candidates.length then search edge 4 [1297] root208Candidates 1297 winner else winner) = winner
  rw [if_pos (by decide), root208Branch]
  exact outerScan209
set_option maxRecDepth 4096 in
theorem outerScan207 : scan edge (search edge 4) 5 [] 0 tail207 winner = winner := by
  rw [tail207, scan, if_neg (by decide), root207Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail208
    (if 4 ≤ root207Candidates.length then search edge 4 [1291] root207Candidates 1291 winner else winner) = winner
  rw [if_pos (by decide), root207Branch]
  exact outerScan208
set_option maxRecDepth 4096 in
theorem outerScan206 : scan edge (search edge 4) 5 [] 0 tail206 winner = winner := by
  rw [tail206, scan, if_neg (by decide), root206Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail207
    (if 4 ≤ root206Candidates.length then search edge 4 [1289] root206Candidates 1289 winner else winner) = winner
  rw [if_pos (by decide), root206Branch]
  exact outerScan207
set_option maxRecDepth 4096 in
theorem outerScan205 : scan edge (search edge 4) 5 [] 0 tail205 winner = winner := by
  rw [tail205, scan, if_neg (by decide), root205Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail206
    (if 4 ≤ root205Candidates.length then search edge 4 [1283] root205Candidates 1283 winner else winner) = winner
  rw [if_pos (by decide), root205Branch]
  exact outerScan206
set_option maxRecDepth 4096 in
theorem outerScan204 : scan edge (search edge 4) 5 [] 0 tail204 winner = winner := by
  rw [tail204, scan, if_neg (by decide), root204Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail205
    (if 4 ≤ root204Candidates.length then search edge 4 [1279] root204Candidates 1279 winner else winner) = winner
  rw [if_pos (by decide), root204Branch]
  exact outerScan205
set_option maxRecDepth 4096 in
theorem outerScan203 : scan edge (search edge 4) 5 [] 0 tail203 winner = winner := by
  rw [tail203, scan, if_neg (by decide), root203Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail204
    (if 4 ≤ root203Candidates.length then search edge 4 [1277] root203Candidates 1277 winner else winner) = winner
  rw [if_pos (by decide), root203Branch]
  exact outerScan204
set_option maxRecDepth 4096 in
theorem outerScan202 : scan edge (search edge 4) 5 [] 0 tail202 winner = winner := by
  rw [tail202, scan, if_neg (by decide), root202Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail203
    (if 4 ≤ root202Candidates.length then search edge 4 [1259] root202Candidates 1259 winner else winner) = winner
  rw [if_pos (by decide), root202Branch]
  exact outerScan203
end ProjectEulerSolutions.PrimePairLarge.Search
