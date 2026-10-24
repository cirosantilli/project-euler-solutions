import ProjectEulerSolutions.Lib.PrimePairLarge.Search037
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot378
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot379
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot380
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot381
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot382
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot383
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot384
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot385
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan385 : scan edge (search edge 4) 5 [] 0 tail385 winner = winner := by
  rw [tail385, scan, if_neg (by decide), root385Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail386
    (if 4 ≤ root385Candidates.length then search edge 4 [2677] root385Candidates 2677 winner else winner) = winner
  rw [if_pos (by decide), root385Branch]
  exact outerScan386
set_option maxRecDepth 4096 in
theorem outerScan384 : scan edge (search edge 4) 5 [] 0 tail384 winner = winner := by
  rw [tail384, scan, if_neg (by decide), root384Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail385
    (if 4 ≤ root384Candidates.length then search edge 4 [2671] root384Candidates 2671 winner else winner) = winner
  rw [if_pos (by decide), root384Branch]
  exact outerScan385
set_option maxRecDepth 4096 in
theorem outerScan383 : scan edge (search edge 4) 5 [] 0 tail383 winner = winner := by
  rw [tail383, scan, if_neg (by decide), root383Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail384
    (if 4 ≤ root383Candidates.length then search edge 4 [2663] root383Candidates 2663 winner else winner) = winner
  rw [if_pos (by decide), root383Branch]
  exact outerScan384
set_option maxRecDepth 4096 in
theorem outerScan382 : scan edge (search edge 4) 5 [] 0 tail382 winner = winner := by
  rw [tail382, scan, if_neg (by decide), root382Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail383
    (if 4 ≤ root382Candidates.length then search edge 4 [2659] root382Candidates 2659 winner else winner) = winner
  rw [if_pos (by decide), root382Branch]
  exact outerScan383
set_option maxRecDepth 4096 in
theorem outerScan381 : scan edge (search edge 4) 5 [] 0 tail381 winner = winner := by
  rw [tail381, scan, if_neg (by decide), root381Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail382
    (if 4 ≤ root381Candidates.length then search edge 4 [2657] root381Candidates 2657 winner else winner) = winner
  rw [if_pos (by decide), root381Branch]
  exact outerScan382
set_option maxRecDepth 4096 in
theorem outerScan380 : scan edge (search edge 4) 5 [] 0 tail380 winner = winner := by
  rw [tail380, scan, if_neg (by decide), root380Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail381
    (if 4 ≤ root380Candidates.length then search edge 4 [2647] root380Candidates 2647 winner else winner) = winner
  rw [if_pos (by decide), root380Branch]
  exact outerScan381
set_option maxRecDepth 4096 in
theorem outerScan379 : scan edge (search edge 4) 5 [] 0 tail379 winner = winner := by
  rw [tail379, scan, if_neg (by decide), root379Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail380
    (if 4 ≤ root379Candidates.length then search edge 4 [2633] root379Candidates 2633 winner else winner) = winner
  rw [if_pos (by decide), root379Branch]
  exact outerScan380
set_option maxRecDepth 4096 in
theorem outerScan378 : scan edge (search edge 4) 5 [] 0 tail378 winner = winner := by
  rw [tail378, scan, if_neg (by decide), root378Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail379
    (if 4 ≤ root378Candidates.length then search edge 4 [2621] root378Candidates 2621 winner else winner) = winner
  rw [if_pos (by decide), root378Branch]
  exact outerScan379
end ProjectEulerSolutions.PrimePairLarge.Search
