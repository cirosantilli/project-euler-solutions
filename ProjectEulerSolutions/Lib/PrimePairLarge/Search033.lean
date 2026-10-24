import ProjectEulerSolutions.Lib.PrimePairLarge.Search032
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot418
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot419
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot420
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot421
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot422
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot423
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot424
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot425
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan425 : scan edge (search edge 4) 5 [] 0 tail425 winner = winner := by
  rw [tail425, scan, if_neg (by decide), root425Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail426
    (if 4 ≤ root425Candidates.length then search edge 4 [2969] root425Candidates 2969 winner else winner) = winner
  rw [if_pos (by decide), root425Branch]
  exact outerScan426
set_option maxRecDepth 4096 in
theorem outerScan424 : scan edge (search edge 4) 5 [] 0 tail424 winner = winner := by
  rw [tail424, scan, if_neg (by decide), root424Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail425
    (if 4 ≤ root424Candidates.length then search edge 4 [2963] root424Candidates 2963 winner else winner) = winner
  rw [if_pos (by decide), root424Branch]
  exact outerScan425
set_option maxRecDepth 4096 in
theorem outerScan423 : scan edge (search edge 4) 5 [] 0 tail423 winner = winner := by
  rw [tail423, scan, if_neg (by decide), root423Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail424
    (if 4 ≤ root423Candidates.length then search edge 4 [2957] root423Candidates 2957 winner else winner) = winner
  rw [if_pos (by decide), root423Branch]
  exact outerScan424
set_option maxRecDepth 4096 in
theorem outerScan422 : scan edge (search edge 4) 5 [] 0 tail422 winner = winner := by
  rw [tail422, scan, if_neg (by decide), root422Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail423
    (if 4 ≤ root422Candidates.length then search edge 4 [2953] root422Candidates 2953 winner else winner) = winner
  rw [if_pos (by decide), root422Branch]
  exact outerScan423
set_option maxRecDepth 4096 in
theorem outerScan421 : scan edge (search edge 4) 5 [] 0 tail421 winner = winner := by
  rw [tail421, scan, if_neg (by decide), root421Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail422
    (if 4 ≤ root421Candidates.length then search edge 4 [2939] root421Candidates 2939 winner else winner) = winner
  rw [if_pos (by decide), root421Branch]
  exact outerScan422
set_option maxRecDepth 4096 in
theorem outerScan420 : scan edge (search edge 4) 5 [] 0 tail420 winner = winner := by
  rw [tail420, scan, if_neg (by decide), root420Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail421
    (if 4 ≤ root420Candidates.length then search edge 4 [2927] root420Candidates 2927 winner else winner) = winner
  rw [if_pos (by decide), root420Branch]
  exact outerScan421
set_option maxRecDepth 4096 in
theorem outerScan419 : scan edge (search edge 4) 5 [] 0 tail419 winner = winner := by
  rw [tail419, scan, if_neg (by decide), root419Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail420
    (if 4 ≤ root419Candidates.length then search edge 4 [2917] root419Candidates 2917 winner else winner) = winner
  rw [if_pos (by decide), root419Branch]
  exact outerScan420
set_option maxRecDepth 4096 in
theorem outerScan418 : scan edge (search edge 4) 5 [] 0 tail418 winner = winner := by
  rw [tail418, scan, if_neg (by decide), root418Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail419
    (if 4 ≤ root418Candidates.length then search edge 4 [2909] root418Candidates 2909 winner else winner) = winner
  rw [if_pos (by decide), root418Branch]
  exact outerScan419
end ProjectEulerSolutions.PrimePairLarge.Search
