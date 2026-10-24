import ProjectEulerSolutions.Lib.PrimePairLarge.Search061
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot186
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot187
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot188
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot189
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot190
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot191
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot192
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot193
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan193 : scan edge (search edge 4) 5 [] 0 tail193 winner = winner := by
  rw [tail193, scan, if_neg (by decide), root193Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail194
    (if 4 ≤ root193Candidates.length then search edge 4 [1193] root193Candidates 1193 winner else winner) = winner
  rw [if_pos (by decide), root193Branch]
  exact outerScan194
set_option maxRecDepth 4096 in
theorem outerScan192 : scan edge (search edge 4) 5 [] 0 tail192 winner = winner := by
  rw [tail192, scan, if_neg (by decide), root192Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail193
    (if 4 ≤ root192Candidates.length then search edge 4 [1187] root192Candidates 1187 winner else winner) = winner
  rw [if_pos (by decide), root192Branch]
  exact outerScan193
set_option maxRecDepth 4096 in
theorem outerScan191 : scan edge (search edge 4) 5 [] 0 tail191 winner = winner := by
  rw [tail191, scan, if_neg (by decide), root191Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail192
    (if 4 ≤ root191Candidates.length then search edge 4 [1181] root191Candidates 1181 winner else winner) = winner
  rw [if_pos (by decide), root191Branch]
  exact outerScan192
set_option maxRecDepth 4096 in
theorem outerScan190 : scan edge (search edge 4) 5 [] 0 tail190 winner = winner := by
  rw [tail190, scan, if_neg (by decide), root190Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail191
    (if 4 ≤ root190Candidates.length then search edge 4 [1171] root190Candidates 1171 winner else winner) = winner
  rw [if_pos (by decide), root190Branch]
  exact outerScan191
set_option maxRecDepth 4096 in
theorem outerScan189 : scan edge (search edge 4) 5 [] 0 tail189 winner = winner := by
  rw [tail189, scan, if_neg (by decide), root189Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail190
    (if 4 ≤ root189Candidates.length then search edge 4 [1163] root189Candidates 1163 winner else winner) = winner
  rw [if_pos (by decide), root189Branch]
  exact outerScan190
set_option maxRecDepth 4096 in
theorem outerScan188 : scan edge (search edge 4) 5 [] 0 tail188 winner = winner := by
  rw [tail188, scan, if_neg (by decide), root188Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail189
    (if 4 ≤ root188Candidates.length then search edge 4 [1153] root188Candidates 1153 winner else winner) = winner
  rw [if_pos (by decide), root188Branch]
  exact outerScan189
set_option maxRecDepth 4096 in
theorem outerScan187 : scan edge (search edge 4) 5 [] 0 tail187 winner = winner := by
  rw [tail187, scan, if_neg (by decide), root187Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail188
    (if 4 ≤ root187Candidates.length then search edge 4 [1151] root187Candidates 1151 winner else winner) = winner
  rw [if_pos (by decide), root187Branch]
  exact outerScan188
set_option maxRecDepth 4096 in
theorem outerScan186 : scan edge (search edge 4) 5 [] 0 tail186 winner = winner := by
  rw [tail186, scan, if_neg (by decide), root186Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail187
    (if 4 ≤ root186Candidates.length then search edge 4 [1129] root186Candidates 1129 winner else winner) = winner
  rw [if_pos (by decide), root186Branch]
  exact outerScan187
end ProjectEulerSolutions.PrimePairLarge.Search
