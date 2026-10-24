import ProjectEulerSolutions.Lib.PrimePairLarge.Search073
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot090
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot091
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot092
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot093
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot094
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot095
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot096
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot097
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan97 : scan edge (search edge 4) 5 [] 0 tail97 winner = winner := by
  rw [tail97, scan, if_neg (by decide), root97Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail98
    (if 4 ≤ root97Candidates.length then search edge 4 [541] root97Candidates 541 winner else winner) = winner
  rw [if_pos (by decide), root97Branch]
  exact outerScan98
set_option maxRecDepth 4096 in
theorem outerScan96 : scan edge (search edge 4) 5 [] 0 tail96 winner = winner := by
  rw [tail96, scan, if_neg (by decide), root96Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail97
    (if 4 ≤ root96Candidates.length then search edge 4 [523] root96Candidates 523 winner else winner) = winner
  rw [if_pos (by decide), root96Branch]
  exact outerScan97
set_option maxRecDepth 4096 in
theorem outerScan95 : scan edge (search edge 4) 5 [] 0 tail95 winner = winner := by
  rw [tail95, scan, if_neg (by decide), root95Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail96
    (if 4 ≤ root95Candidates.length then search edge 4 [521] root95Candidates 521 winner else winner) = winner
  rw [if_pos (by decide), root95Branch]
  exact outerScan96
set_option maxRecDepth 4096 in
theorem outerScan94 : scan edge (search edge 4) 5 [] 0 tail94 winner = winner := by
  rw [tail94, scan, if_neg (by decide), root94Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail95
    (if 4 ≤ root94Candidates.length then search edge 4 [509] root94Candidates 509 winner else winner) = winner
  rw [if_pos (by decide), root94Branch]
  exact outerScan95
set_option maxRecDepth 4096 in
theorem outerScan93 : scan edge (search edge 4) 5 [] 0 tail93 winner = winner := by
  rw [tail93, scan, if_neg (by decide), root93Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail94
    (if 4 ≤ root93Candidates.length then search edge 4 [503] root93Candidates 503 winner else winner) = winner
  rw [if_pos (by decide), root93Branch]
  exact outerScan94
set_option maxRecDepth 4096 in
theorem outerScan92 : scan edge (search edge 4) 5 [] 0 tail92 winner = winner := by
  rw [tail92, scan, if_neg (by decide), root92Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail93
    (if 4 ≤ root92Candidates.length then search edge 4 [499] root92Candidates 499 winner else winner) = winner
  rw [if_pos (by decide), root92Branch]
  exact outerScan93
set_option maxRecDepth 4096 in
theorem outerScan91 : scan edge (search edge 4) 5 [] 0 tail91 winner = winner := by
  rw [tail91, scan, if_neg (by decide), root91Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail92
    (if 4 ≤ root91Candidates.length then search edge 4 [491] root91Candidates 491 winner else winner) = winner
  rw [if_pos (by decide), root91Branch]
  exact outerScan92
set_option maxRecDepth 4096 in
theorem outerScan90 : scan edge (search edge 4) 5 [] 0 tail90 winner = winner := by
  rw [tail90, scan, if_neg (by decide), root90Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail91
    (if 4 ≤ root90Candidates.length then search edge 4 [487] root90Candidates 487 winner else winner) = winner
  rw [if_pos (by decide), root90Branch]
  exact outerScan91
end ProjectEulerSolutions.PrimePairLarge.Search
