import ProjectEulerSolutions.Lib.PrimePairLarge.Search062
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot178
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot179
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot180
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot181
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot182
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot183
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot184
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot185
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan185 : scan edge (search edge 4) 5 [] 0 tail185 winner = winner := by
  rw [tail185, scan, if_neg (by decide), root185Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail186
    (if 4 ≤ root185Candidates.length then search edge 4 [1123] root185Candidates 1123 winner else winner) = winner
  rw [if_pos (by decide), root185Branch]
  exact outerScan186
set_option maxRecDepth 4096 in
theorem outerScan184 : scan edge (search edge 4) 5 [] 0 tail184 winner = winner := by
  rw [tail184, scan, if_neg (by decide), root184Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail185
    (if 4 ≤ root184Candidates.length then search edge 4 [1117] root184Candidates 1117 winner else winner) = winner
  rw [if_pos (by decide), root184Branch]
  exact outerScan185
set_option maxRecDepth 4096 in
theorem outerScan183 : scan edge (search edge 4) 5 [] 0 tail183 winner = winner := by
  rw [tail183, scan, if_neg (by decide), root183Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail184
    (if 4 ≤ root183Candidates.length then search edge 4 [1109] root183Candidates 1109 winner else winner) = winner
  rw [if_pos (by decide), root183Branch]
  exact outerScan184
set_option maxRecDepth 4096 in
theorem outerScan182 : scan edge (search edge 4) 5 [] 0 tail182 winner = winner := by
  rw [tail182, scan, if_neg (by decide), root182Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail183
    (if 4 ≤ root182Candidates.length then search edge 4 [1103] root182Candidates 1103 winner else winner) = winner
  rw [if_pos (by decide), root182Branch]
  exact outerScan183
set_option maxRecDepth 4096 in
theorem outerScan181 : scan edge (search edge 4) 5 [] 0 tail181 winner = winner := by
  rw [tail181, scan, if_neg (by decide), root181Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail182
    (if 4 ≤ root181Candidates.length then search edge 4 [1097] root181Candidates 1097 winner else winner) = winner
  rw [if_pos (by decide), root181Branch]
  exact outerScan182
set_option maxRecDepth 4096 in
theorem outerScan180 : scan edge (search edge 4) 5 [] 0 tail180 winner = winner := by
  rw [tail180, scan, if_neg (by decide), root180Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail181
    (if 4 ≤ root180Candidates.length then search edge 4 [1093] root180Candidates 1093 winner else winner) = winner
  rw [if_pos (by decide), root180Branch]
  exact outerScan181
set_option maxRecDepth 4096 in
theorem outerScan179 : scan edge (search edge 4) 5 [] 0 tail179 winner = winner := by
  rw [tail179, scan, if_neg (by decide), root179Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail180
    (if 4 ≤ root179Candidates.length then search edge 4 [1091] root179Candidates 1091 winner else winner) = winner
  rw [if_pos (by decide), root179Branch]
  exact outerScan180
set_option maxRecDepth 4096 in
theorem outerScan178 : scan edge (search edge 4) 5 [] 0 tail178 winner = winner := by
  rw [tail178, scan, if_neg (by decide), root178Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail179
    (if 4 ≤ root178Candidates.length then search edge 4 [1087] root178Candidates 1087 winner else winner) = winner
  rw [if_pos (by decide), root178Branch]
  exact outerScan179
end ProjectEulerSolutions.PrimePairLarge.Search
