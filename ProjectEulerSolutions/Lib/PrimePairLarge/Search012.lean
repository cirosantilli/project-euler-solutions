import ProjectEulerSolutions.Lib.PrimePairLarge.Search011
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot586
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot587
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot588
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot589
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot590
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot591
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot592
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot593
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan593 : scan edge (search edge 4) 5 [] 0 tail593 winner = winner := by
  rw [tail593, scan, if_neg (by decide), root593Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail594
    (if 4 ≤ root593Candidates.length then search edge 4 [4363] root593Candidates 4363 winner else winner) = winner
  rw [if_pos (by decide), root593Branch]
  exact outerScan594
set_option maxRecDepth 4096 in
theorem outerScan592 : scan edge (search edge 4) 5 [] 0 tail592 winner = winner := by
  rw [tail592, scan, if_neg (by decide), root592Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail593
    (if 4 ≤ root592Candidates.length then search edge 4 [4357] root592Candidates 4357 winner else winner) = winner
  rw [if_pos (by decide), root592Branch]
  exact outerScan593
set_option maxRecDepth 4096 in
theorem outerScan591 : scan edge (search edge 4) 5 [] 0 tail591 winner = winner := by
  rw [tail591, scan, if_neg (by decide), root591Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail592
    (if 4 ≤ root591Candidates.length then search edge 4 [4349] root591Candidates 4349 winner else winner) = winner
  rw [if_pos (by decide), root591Branch]
  exact outerScan592
set_option maxRecDepth 4096 in
theorem outerScan590 : scan edge (search edge 4) 5 [] 0 tail590 winner = winner := by
  rw [tail590, scan, if_neg (by decide), root590Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail591
    (if 4 ≤ root590Candidates.length then search edge 4 [4339] root590Candidates 4339 winner else winner) = winner
  rw [if_pos (by decide), root590Branch]
  exact outerScan591
set_option maxRecDepth 4096 in
theorem outerScan589 : scan edge (search edge 4) 5 [] 0 tail589 winner = winner := by
  rw [tail589, scan, if_neg (by decide), root589Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail590
    (if 4 ≤ root589Candidates.length then search edge 4 [4337] root589Candidates 4337 winner else winner) = winner
  rw [if_pos (by decide), root589Branch]
  exact outerScan590
set_option maxRecDepth 4096 in
theorem outerScan588 : scan edge (search edge 4) 5 [] 0 tail588 winner = winner := by
  rw [tail588, scan, if_neg (by decide), root588Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail589
    (if 4 ≤ root588Candidates.length then search edge 4 [4327] root588Candidates 4327 winner else winner) = winner
  rw [if_pos (by decide), root588Branch]
  exact outerScan589
set_option maxRecDepth 4096 in
theorem outerScan587 : scan edge (search edge 4) 5 [] 0 tail587 winner = winner := by
  rw [tail587, scan, if_neg (by decide), root587Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail588
    (if 4 ≤ root587Candidates.length then search edge 4 [4297] root587Candidates 4297 winner else winner) = winner
  rw [if_pos (by decide), root587Branch]
  exact outerScan588
set_option maxRecDepth 4096 in
theorem outerScan586 : scan edge (search edge 4) 5 [] 0 tail586 winner = winner := by
  rw [tail586, scan, if_neg (by decide), root586Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail587
    (if 4 ≤ root586Candidates.length then search edge 4 [4289] root586Candidates 4289 winner else winner) = winner
  rw [if_pos (by decide), root586Branch]
  exact outerScan587
end ProjectEulerSolutions.PrimePairLarge.Search
