import ProjectEulerSolutions.Lib.PrimePairLarge.Search018
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot530
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot531
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot532
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot533
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot534
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot535
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot536
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot537
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan537 : scan edge (search edge 4) 5 [] 0 tail537 winner = winner := by
  rw [tail537, scan, if_neg (by decide), root537Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail538
    (if 4 ≤ root537Candidates.length then search edge 4 [3907] root537Candidates 3907 winner else winner) = winner
  rw [if_pos (by decide), root537Branch]
  exact outerScan538
set_option maxRecDepth 4096 in
theorem outerScan536 : scan edge (search edge 4) 5 [] 0 tail536 winner = winner := by
  rw [tail536, scan, if_neg (by decide), root536Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail537
    (if 4 ≤ root536Candidates.length then search edge 4 [3889] root536Candidates 3889 winner else winner) = winner
  rw [if_pos (by decide), root536Branch]
  exact outerScan537
set_option maxRecDepth 4096 in
theorem outerScan535 : scan edge (search edge 4) 5 [] 0 tail535 winner = winner := by
  rw [tail535, scan, if_neg (by decide), root535Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail536
    (if 4 ≤ root535Candidates.length then search edge 4 [3881] root535Candidates 3881 winner else winner) = winner
  rw [if_pos (by decide), root535Branch]
  exact outerScan536
set_option maxRecDepth 4096 in
theorem outerScan534 : scan edge (search edge 4) 5 [] 0 tail534 winner = winner := by
  rw [tail534, scan, if_neg (by decide), root534Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail535
    (if 4 ≤ root534Candidates.length then search edge 4 [3877] root534Candidates 3877 winner else winner) = winner
  rw [if_pos (by decide), root534Branch]
  exact outerScan535
set_option maxRecDepth 4096 in
theorem outerScan533 : scan edge (search edge 4) 5 [] 0 tail533 winner = winner := by
  rw [tail533, scan, if_neg (by decide), root533Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail534
    (if 4 ≤ root533Candidates.length then search edge 4 [3863] root533Candidates 3863 winner else winner) = winner
  rw [if_pos (by decide), root533Branch]
  exact outerScan534
set_option maxRecDepth 4096 in
theorem outerScan532 : scan edge (search edge 4) 5 [] 0 tail532 winner = winner := by
  rw [tail532, scan, if_neg (by decide), root532Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail533
    (if 4 ≤ root532Candidates.length then search edge 4 [3853] root532Candidates 3853 winner else winner) = winner
  rw [if_pos (by decide), root532Branch]
  exact outerScan533
set_option maxRecDepth 4096 in
theorem outerScan531 : scan edge (search edge 4) 5 [] 0 tail531 winner = winner := by
  rw [tail531, scan, if_neg (by decide), root531Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail532
    (if 4 ≤ root531Candidates.length then search edge 4 [3851] root531Candidates 3851 winner else winner) = winner
  rw [if_pos (by decide), root531Branch]
  exact outerScan532
set_option maxRecDepth 4096 in
theorem outerScan530 : scan edge (search edge 4) 5 [] 0 tail530 winner = winner := by
  rw [tail530, scan, if_neg (by decide), root530Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail531
    (if 4 ≤ root530Candidates.length then search edge 4 [3847] root530Candidates 3847 winner else winner) = winner
  rw [if_pos (by decide), root530Branch]
  exact outerScan531
end ProjectEulerSolutions.PrimePairLarge.Search
