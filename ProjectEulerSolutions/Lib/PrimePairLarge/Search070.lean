import ProjectEulerSolutions.Lib.PrimePairLarge.Search069
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot122
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot123
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot124
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot125
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot126
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot127
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot128
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot129
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan129 : scan edge (search edge 4) 5 [] 0 tail129 winner = winner := by
  rw [tail129, scan, if_neg (by decide), root129Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail130
    (if 4 ≤ root129Candidates.length then search edge 4 [743] root129Candidates 743 winner else winner) = winner
  rw [if_pos (by decide), root129Branch]
  exact outerScan130
set_option maxRecDepth 4096 in
theorem outerScan128 : scan edge (search edge 4) 5 [] 0 tail128 winner = winner := by
  rw [tail128, scan, if_neg (by decide), root128Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail129
    (if 4 ≤ root128Candidates.length then search edge 4 [739] root128Candidates 739 winner else winner) = winner
  rw [if_pos (by decide), root128Branch]
  exact outerScan129
set_option maxRecDepth 4096 in
theorem outerScan127 : scan edge (search edge 4) 5 [] 0 tail127 winner = winner := by
  rw [tail127, scan, if_neg (by decide), root127Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail128
    (if 4 ≤ root127Candidates.length then search edge 4 [733] root127Candidates 733 winner else winner) = winner
  rw [if_pos (by decide), root127Branch]
  exact outerScan128
set_option maxRecDepth 4096 in
theorem outerScan126 : scan edge (search edge 4) 5 [] 0 tail126 winner = winner := by
  rw [tail126, scan, if_neg (by decide), root126Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail127
    (if 4 ≤ root126Candidates.length then search edge 4 [727] root126Candidates 727 winner else winner) = winner
  rw [if_pos (by decide), root126Branch]
  exact outerScan127
set_option maxRecDepth 4096 in
theorem outerScan125 : scan edge (search edge 4) 5 [] 0 tail125 winner = winner := by
  rw [tail125, scan, if_neg (by decide), root125Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail126
    (if 4 ≤ root125Candidates.length then search edge 4 [719] root125Candidates 719 winner else winner) = winner
  rw [if_pos (by decide), root125Branch]
  exact outerScan126
set_option maxRecDepth 4096 in
theorem outerScan124 : scan edge (search edge 4) 5 [] 0 tail124 winner = winner := by
  rw [tail124, scan, if_neg (by decide), root124Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail125
    (if 4 ≤ root124Candidates.length then search edge 4 [709] root124Candidates 709 winner else winner) = winner
  rw [if_pos (by decide), root124Branch]
  exact outerScan125
set_option maxRecDepth 4096 in
theorem outerScan123 : scan edge (search edge 4) 5 [] 0 tail123 winner = winner := by
  rw [tail123, scan, if_neg (by decide), root123Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail124
    (if 4 ≤ root123Candidates.length then search edge 4 [701] root123Candidates 701 winner else winner) = winner
  rw [if_pos (by decide), root123Branch]
  exact outerScan124
set_option maxRecDepth 4096 in
theorem outerScan122 : scan edge (search edge 4) 5 [] 0 tail122 winner = winner := by
  rw [tail122, scan, if_neg (by decide), root122Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail123
    (if 4 ≤ root122Candidates.length then search edge 4 [691] root122Candidates 691 winner else winner) = winner
  rw [if_pos (by decide), root122Branch]
  exact outerScan123
end ProjectEulerSolutions.PrimePairLarge.Search
