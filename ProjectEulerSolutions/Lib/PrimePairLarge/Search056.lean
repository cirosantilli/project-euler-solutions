import ProjectEulerSolutions.Lib.PrimePairLarge.Search055
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot234
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot235
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot236
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot237
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot238
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot239
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot240
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot241
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan241 : scan edge (search edge 4) 5 [] 0 tail241 winner = winner := by
  rw [tail241, scan, if_neg (by decide), root241Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail242
    (if 4 ≤ root241Candidates.length then search edge 4 [1549] root241Candidates 1549 winner else winner) = winner
  rw [if_pos (by decide), root241Branch]
  exact outerScan242
set_option maxRecDepth 4096 in
theorem outerScan240 : scan edge (search edge 4) 5 [] 0 tail240 winner = winner := by
  rw [tail240, scan, if_neg (by decide), root240Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail241
    (if 4 ≤ root240Candidates.length then search edge 4 [1543] root240Candidates 1543 winner else winner) = winner
  rw [if_pos (by decide), root240Branch]
  exact outerScan241
set_option maxRecDepth 4096 in
theorem outerScan239 : scan edge (search edge 4) 5 [] 0 tail239 winner = winner := by
  rw [tail239, scan, if_neg (by decide), root239Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail240
    (if 4 ≤ root239Candidates.length then search edge 4 [1531] root239Candidates 1531 winner else winner) = winner
  rw [if_pos (by decide), root239Branch]
  exact outerScan240
set_option maxRecDepth 4096 in
theorem outerScan238 : scan edge (search edge 4) 5 [] 0 tail238 winner = winner := by
  rw [tail238, scan, if_neg (by decide), root238Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail239
    (if 4 ≤ root238Candidates.length then search edge 4 [1523] root238Candidates 1523 winner else winner) = winner
  rw [if_pos (by decide), root238Branch]
  exact outerScan239
set_option maxRecDepth 4096 in
theorem outerScan237 : scan edge (search edge 4) 5 [] 0 tail237 winner = winner := by
  rw [tail237, scan, if_neg (by decide), root237Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail238
    (if 4 ≤ root237Candidates.length then search edge 4 [1511] root237Candidates 1511 winner else winner) = winner
  rw [if_pos (by decide), root237Branch]
  exact outerScan238
set_option maxRecDepth 4096 in
theorem outerScan236 : scan edge (search edge 4) 5 [] 0 tail236 winner = winner := by
  rw [tail236, scan, if_neg (by decide), root236Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail237
    (if 4 ≤ root236Candidates.length then search edge 4 [1499] root236Candidates 1499 winner else winner) = winner
  rw [if_pos (by decide), root236Branch]
  exact outerScan237
set_option maxRecDepth 4096 in
theorem outerScan235 : scan edge (search edge 4) 5 [] 0 tail235 winner = winner := by
  rw [tail235, scan, if_neg (by decide), root235Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail236
    (if 4 ≤ root235Candidates.length then search edge 4 [1493] root235Candidates 1493 winner else winner) = winner
  rw [if_pos (by decide), root235Branch]
  exact outerScan236
set_option maxRecDepth 4096 in
theorem outerScan234 : scan edge (search edge 4) 5 [] 0 tail234 winner = winner := by
  rw [tail234, scan, if_neg (by decide), root234Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail235
    (if 4 ≤ root234Candidates.length then search edge 4 [1489] root234Candidates 1489 winner else winner) = winner
  rw [if_pos (by decide), root234Branch]
  exact outerScan235
end ProjectEulerSolutions.PrimePairLarge.Search
