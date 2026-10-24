import ProjectEulerSolutions.Lib.PrimePairLarge.Search008
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot610
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot611
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot612
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot613
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot614
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot615
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot616
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot617
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan617 : scan edge (search edge 4) 5 [] 0 tail617 winner = winner := by
  rw [tail617, scan, if_neg (by decide), root617Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail618
    (if 4 ≤ root617Candidates.length then search edge 4 [4583] root617Candidates 4583 winner else winner) = winner
  rw [if_pos (by decide), root617Branch]
  exact outerScan618
set_option maxRecDepth 4096 in
theorem outerScan616 : scan edge (search edge 4) 5 [] 0 tail616 winner = winner := by
  rw [tail616, scan, if_neg (by decide), root616Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail617
    (if 4 ≤ root616Candidates.length then search edge 4 [4567] root616Candidates 4567 winner else winner) = winner
  rw [if_pos (by decide), root616Branch]
  exact outerScan617
set_option maxRecDepth 4096 in
theorem outerScan615 : scan edge (search edge 4) 5 [] 0 tail615 winner = winner := by
  rw [tail615, scan, if_neg (by decide), root615Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail616
    (if 4 ≤ root615Candidates.length then search edge 4 [4561] root615Candidates 4561 winner else winner) = winner
  rw [if_pos (by decide), root615Branch]
  exact outerScan616
set_option maxRecDepth 4096 in
theorem outerScan614 : scan edge (search edge 4) 5 [] 0 tail614 winner = winner := by
  rw [tail614, scan, if_neg (by decide), root614Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail615
    (if 4 ≤ root614Candidates.length then search edge 4 [4549] root614Candidates 4549 winner else winner) = winner
  rw [if_pos (by decide), root614Branch]
  exact outerScan615
set_option maxRecDepth 4096 in
theorem outerScan613 : scan edge (search edge 4) 5 [] 0 tail613 winner = winner := by
  rw [tail613, scan, if_neg (by decide), root613Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail614
    (if 4 ≤ root613Candidates.length then search edge 4 [4547] root613Candidates 4547 winner else winner) = winner
  rw [if_pos (by decide), root613Branch]
  exact outerScan614
set_option maxRecDepth 4096 in
theorem outerScan612 : scan edge (search edge 4) 5 [] 0 tail612 winner = winner := by
  rw [tail612, scan, if_neg (by decide), root612Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail613
    (if 4 ≤ root612Candidates.length then search edge 4 [4523] root612Candidates 4523 winner else winner) = winner
  rw [if_pos (by decide), root612Branch]
  exact outerScan613
set_option maxRecDepth 4096 in
theorem outerScan611 : scan edge (search edge 4) 5 [] 0 tail611 winner = winner := by
  rw [tail611, scan, if_neg (by decide), root611Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail612
    (if 4 ≤ root611Candidates.length then search edge 4 [4519] root611Candidates 4519 winner else winner) = winner
  rw [if_pos (by decide), root611Branch]
  exact outerScan612
set_option maxRecDepth 4096 in
theorem outerScan610 : scan edge (search edge 4) 5 [] 0 tail610 winner = winner := by
  rw [tail610, scan, if_neg (by decide), root610Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail611
    (if 4 ≤ root610Candidates.length then search edge 4 [4517] root610Candidates 4517 winner else winner) = winner
  rw [if_pos (by decide), root610Branch]
  exact outerScan611
end ProjectEulerSolutions.PrimePairLarge.Search
