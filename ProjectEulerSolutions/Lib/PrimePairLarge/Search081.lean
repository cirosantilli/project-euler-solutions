import ProjectEulerSolutions.Lib.PrimePairLarge.Search080
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot034
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot035
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot036
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot037
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot038
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot039
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot040
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot041
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan41 : scan edge (search edge 4) 5 [] 0 tail41 winner = winner := by
  rw [tail41, scan, if_neg (by decide), root41Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail42
    (if 4 ≤ root41Candidates.length then search edge 4 [193] root41Candidates 193 winner else winner) = winner
  rw [if_pos (by decide), root41Branch]
  exact outerScan42
set_option maxRecDepth 4096 in
theorem outerScan40 : scan edge (search edge 4) 5 [] 0 tail40 winner = winner := by
  rw [tail40, scan, if_neg (by decide), root40Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail41
    (if 4 ≤ root40Candidates.length then search edge 4 [191] root40Candidates 191 winner else winner) = winner
  rw [if_pos (by decide), root40Branch]
  exact outerScan41
set_option maxRecDepth 4096 in
theorem outerScan39 : scan edge (search edge 4) 5 [] 0 tail39 winner = winner := by
  rw [tail39, scan, if_neg (by decide), root39Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail40
    (if 4 ≤ root39Candidates.length then search edge 4 [181] root39Candidates 181 winner else winner) = winner
  rw [if_pos (by decide), root39Branch]
  exact outerScan40
set_option maxRecDepth 4096 in
theorem outerScan38 : scan edge (search edge 4) 5 [] 0 tail38 winner = winner := by
  rw [tail38, scan, if_neg (by decide), root38Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail39
    (if 4 ≤ root38Candidates.length then search edge 4 [179] root38Candidates 179 winner else winner) = winner
  rw [if_pos (by decide), root38Branch]
  exact outerScan39
set_option maxRecDepth 4096 in
theorem outerScan37 : scan edge (search edge 4) 5 [] 0 tail37 winner = winner := by
  rw [tail37, scan, if_neg (by decide), root37Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail38
    (if 4 ≤ root37Candidates.length then search edge 4 [173] root37Candidates 173 winner else winner) = winner
  rw [if_pos (by decide), root37Branch]
  exact outerScan38
set_option maxRecDepth 4096 in
theorem outerScan36 : scan edge (search edge 4) 5 [] 0 tail36 winner = winner := by
  rw [tail36, scan, if_neg (by decide), root36Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail37
    (if 4 ≤ root36Candidates.length then search edge 4 [167] root36Candidates 167 winner else winner) = winner
  rw [if_pos (by decide), root36Branch]
  exact outerScan37
set_option maxRecDepth 4096 in
theorem outerScan35 : scan edge (search edge 4) 5 [] 0 tail35 winner = winner := by
  rw [tail35, scan, if_neg (by decide), root35Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail36
    (if 4 ≤ root35Candidates.length then search edge 4 [163] root35Candidates 163 winner else winner) = winner
  rw [if_pos (by decide), root35Branch]
  exact outerScan36
set_option maxRecDepth 4096 in
theorem outerScan34 : scan edge (search edge 4) 5 [] 0 tail34 winner = winner := by
  rw [tail34, scan, if_neg (by decide), root34Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail35
    (if 4 ≤ root34Candidates.length then search edge 4 [157] root34Candidates 157 winner else winner) = winner
  rw [if_pos (by decide), root34Branch]
  exact outerScan35
end ProjectEulerSolutions.PrimePairLarge.Search
