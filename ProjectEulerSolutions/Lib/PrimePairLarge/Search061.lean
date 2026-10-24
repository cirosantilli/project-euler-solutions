import ProjectEulerSolutions.Lib.PrimePairLarge.Search060
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot194
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot195
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot196
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot197
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot198
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot199
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot200
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot201
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan201 : scan edge (search edge 4) 5 [] 0 tail201 winner = winner := by
  rw [tail201, scan, if_neg (by decide), root201Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail202
    (if 4 ≤ root201Candidates.length then search edge 4 [1249] root201Candidates 1249 winner else winner) = winner
  rw [if_pos (by decide), root201Branch]
  exact outerScan202
set_option maxRecDepth 4096 in
theorem outerScan200 : scan edge (search edge 4) 5 [] 0 tail200 winner = winner := by
  rw [tail200, scan, if_neg (by decide), root200Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail201
    (if 4 ≤ root200Candidates.length then search edge 4 [1237] root200Candidates 1237 winner else winner) = winner
  rw [if_pos (by decide), root200Branch]
  exact outerScan201
set_option maxRecDepth 4096 in
theorem outerScan199 : scan edge (search edge 4) 5 [] 0 tail199 winner = winner := by
  rw [tail199, scan, if_neg (by decide), root199Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail200
    (if 4 ≤ root199Candidates.length then search edge 4 [1231] root199Candidates 1231 winner else winner) = winner
  rw [if_pos (by decide), root199Branch]
  exact outerScan200
set_option maxRecDepth 4096 in
theorem outerScan198 : scan edge (search edge 4) 5 [] 0 tail198 winner = winner := by
  rw [tail198, scan, if_neg (by decide), root198Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail199
    (if 4 ≤ root198Candidates.length then search edge 4 [1229] root198Candidates 1229 winner else winner) = winner
  rw [if_pos (by decide), root198Branch]
  exact outerScan199
set_option maxRecDepth 4096 in
theorem outerScan197 : scan edge (search edge 4) 5 [] 0 tail197 winner = winner := by
  rw [tail197, scan, if_neg (by decide), root197Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail198
    (if 4 ≤ root197Candidates.length then search edge 4 [1223] root197Candidates 1223 winner else winner) = winner
  rw [if_pos (by decide), root197Branch]
  exact outerScan198
set_option maxRecDepth 4096 in
theorem outerScan196 : scan edge (search edge 4) 5 [] 0 tail196 winner = winner := by
  rw [tail196, scan, if_neg (by decide), root196Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail197
    (if 4 ≤ root196Candidates.length then search edge 4 [1217] root196Candidates 1217 winner else winner) = winner
  rw [if_pos (by decide), root196Branch]
  exact outerScan197
set_option maxRecDepth 4096 in
theorem outerScan195 : scan edge (search edge 4) 5 [] 0 tail195 winner = winner := by
  rw [tail195, scan, if_neg (by decide), root195Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail196
    (if 4 ≤ root195Candidates.length then search edge 4 [1213] root195Candidates 1213 winner else winner) = winner
  rw [if_pos (by decide), root195Branch]
  exact outerScan196
set_option maxRecDepth 4096 in
theorem outerScan194 : scan edge (search edge 4) 5 [] 0 tail194 winner = winner := by
  rw [tail194, scan, if_neg (by decide), root194Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail195
    (if 4 ≤ root194Candidates.length then search edge 4 [1201] root194Candidates 1201 winner else winner) = winner
  rw [if_pos (by decide), root194Branch]
  exact outerScan195
end ProjectEulerSolutions.PrimePairLarge.Search
