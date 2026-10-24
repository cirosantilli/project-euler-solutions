import ProjectEulerSolutions.Lib.PrimePairLarge.Search058
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot210
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot211
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot212
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot213
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot214
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot215
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot216
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot217
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan217 : scan edge (search edge 4) 5 [] 0 tail217 winner = winner := by
  rw [tail217, scan, if_neg (by decide), root217Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail218
    (if 4 ≤ root217Candidates.length then search edge 4 [1373] root217Candidates 1373 winner else winner) = winner
  rw [if_pos (by decide), root217Branch]
  exact outerScan218
set_option maxRecDepth 4096 in
theorem outerScan216 : scan edge (search edge 4) 5 [] 0 tail216 winner = winner := by
  rw [tail216, scan, if_neg (by decide), root216Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail217
    (if 4 ≤ root216Candidates.length then search edge 4 [1367] root216Candidates 1367 winner else winner) = winner
  rw [if_pos (by decide), root216Branch]
  exact outerScan217
set_option maxRecDepth 4096 in
theorem outerScan215 : scan edge (search edge 4) 5 [] 0 tail215 winner = winner := by
  rw [tail215, scan, if_neg (by decide), root215Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail216
    (if 4 ≤ root215Candidates.length then search edge 4 [1361] root215Candidates 1361 winner else winner) = winner
  rw [if_pos (by decide), root215Branch]
  exact outerScan216
set_option maxRecDepth 4096 in
theorem outerScan214 : scan edge (search edge 4) 5 [] 0 tail214 winner = winner := by
  rw [tail214, scan, if_neg (by decide), root214Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail215
    (if 4 ≤ root214Candidates.length then search edge 4 [1327] root214Candidates 1327 winner else winner) = winner
  rw [if_pos (by decide), root214Branch]
  exact outerScan215
set_option maxRecDepth 4096 in
theorem outerScan213 : scan edge (search edge 4) 5 [] 0 tail213 winner = winner := by
  rw [tail213, scan, if_neg (by decide), root213Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail214
    (if 4 ≤ root213Candidates.length then search edge 4 [1321] root213Candidates 1321 winner else winner) = winner
  rw [if_pos (by decide), root213Branch]
  exact outerScan214
set_option maxRecDepth 4096 in
theorem outerScan212 : scan edge (search edge 4) 5 [] 0 tail212 winner = winner := by
  rw [tail212, scan, if_neg (by decide), root212Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail213
    (if 4 ≤ root212Candidates.length then search edge 4 [1319] root212Candidates 1319 winner else winner) = winner
  rw [if_pos (by decide), root212Branch]
  exact outerScan213
set_option maxRecDepth 4096 in
theorem outerScan211 : scan edge (search edge 4) 5 [] 0 tail211 winner = winner := by
  rw [tail211, scan, if_neg (by decide), root211Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail212
    (if 4 ≤ root211Candidates.length then search edge 4 [1307] root211Candidates 1307 winner else winner) = winner
  rw [if_pos (by decide), root211Branch]
  exact outerScan212
set_option maxRecDepth 4096 in
theorem outerScan210 : scan edge (search edge 4) 5 [] 0 tail210 winner = winner := by
  rw [tail210, scan, if_neg (by decide), root210Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail211
    (if 4 ≤ root210Candidates.length then search edge 4 [1303] root210Candidates 1303 winner else winner) = winner
  rw [if_pos (by decide), root210Branch]
  exact outerScan211
end ProjectEulerSolutions.PrimePairLarge.Search
