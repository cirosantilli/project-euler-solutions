import ProjectEulerSolutions.Lib.PrimePairLarge.Search029
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot442
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot443
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot444
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot445
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot446
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot447
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot448
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot449
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan449 : scan edge (search edge 4) 5 [] 0 tail449 winner = winner := by
  rw [tail449, scan, if_neg (by decide), root449Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail450
    (if 4 ≤ root449Candidates.length then search edge 4 [3191] root449Candidates 3191 winner else winner) = winner
  rw [if_pos (by decide), root449Branch]
  exact outerScan450
set_option maxRecDepth 4096 in
theorem outerScan448 : scan edge (search edge 4) 5 [] 0 tail448 winner = winner := by
  rw [tail448, scan, if_neg (by decide), root448Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail449
    (if 4 ≤ root448Candidates.length then search edge 4 [3187] root448Candidates 3187 winner else winner) = winner
  rw [if_pos (by decide), root448Branch]
  exact outerScan449
set_option maxRecDepth 4096 in
theorem outerScan447 : scan edge (search edge 4) 5 [] 0 tail447 winner = winner := by
  rw [tail447, scan, if_neg (by decide), root447Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail448
    (if 4 ≤ root447Candidates.length then search edge 4 [3181] root447Candidates 3181 winner else winner) = winner
  rw [if_pos (by decide), root447Branch]
  exact outerScan448
set_option maxRecDepth 4096 in
theorem outerScan446 : scan edge (search edge 4) 5 [] 0 tail446 winner = winner := by
  rw [tail446, scan, if_neg (by decide), root446Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail447
    (if 4 ≤ root446Candidates.length then search edge 4 [3169] root446Candidates 3169 winner else winner) = winner
  rw [if_pos (by decide), root446Branch]
  exact outerScan447
set_option maxRecDepth 4096 in
theorem outerScan445 : scan edge (search edge 4) 5 [] 0 tail445 winner = winner := by
  rw [tail445, scan, if_neg (by decide), root445Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail446
    (if 4 ≤ root445Candidates.length then search edge 4 [3167] root445Candidates 3167 winner else winner) = winner
  rw [if_pos (by decide), root445Branch]
  exact outerScan446
set_option maxRecDepth 4096 in
theorem outerScan444 : scan edge (search edge 4) 5 [] 0 tail444 winner = winner := by
  rw [tail444, scan, if_neg (by decide), root444Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail445
    (if 4 ≤ root444Candidates.length then search edge 4 [3163] root444Candidates 3163 winner else winner) = winner
  rw [if_pos (by decide), root444Branch]
  exact outerScan445
set_option maxRecDepth 4096 in
theorem outerScan443 : scan edge (search edge 4) 5 [] 0 tail443 winner = winner := by
  rw [tail443, scan, if_neg (by decide), root443Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail444
    (if 4 ≤ root443Candidates.length then search edge 4 [3137] root443Candidates 3137 winner else winner) = winner
  rw [if_pos (by decide), root443Branch]
  exact outerScan444
set_option maxRecDepth 4096 in
theorem outerScan442 : scan edge (search edge 4) 5 [] 0 tail442 winner = winner := by
  rw [tail442, scan, if_neg (by decide), root442Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail443
    (if 4 ≤ root442Candidates.length then search edge 4 [3121] root442Candidates 3121 winner else winner) = winner
  rw [if_pos (by decide), root442Branch]
  exact outerScan443
end ProjectEulerSolutions.PrimePairLarge.Search
