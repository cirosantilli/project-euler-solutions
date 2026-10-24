import ProjectEulerSolutions.Lib.PrimePairLarge.Search036
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot386
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot387
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot388
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot389
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot390
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot391
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot392
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot393
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan393 : scan edge (search edge 4) 5 [] 0 tail393 winner = winner := by
  rw [tail393, scan, if_neg (by decide), root393Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail394
    (if 4 ≤ root393Candidates.length then search edge 4 [2713] root393Candidates 2713 winner else winner) = winner
  rw [if_pos (by decide), root393Branch]
  exact outerScan394
set_option maxRecDepth 4096 in
theorem outerScan392 : scan edge (search edge 4) 5 [] 0 tail392 winner = winner := by
  rw [tail392, scan, if_neg (by decide), root392Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail393
    (if 4 ≤ root392Candidates.length then search edge 4 [2711] root392Candidates 2711 winner else winner) = winner
  rw [if_pos (by decide), root392Branch]
  exact outerScan393
set_option maxRecDepth 4096 in
theorem outerScan391 : scan edge (search edge 4) 5 [] 0 tail391 winner = winner := by
  rw [tail391, scan, if_neg (by decide), root391Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail392
    (if 4 ≤ root391Candidates.length then search edge 4 [2707] root391Candidates 2707 winner else winner) = winner
  rw [if_pos (by decide), root391Branch]
  exact outerScan392
set_option maxRecDepth 4096 in
theorem outerScan390 : scan edge (search edge 4) 5 [] 0 tail390 winner = winner := by
  rw [tail390, scan, if_neg (by decide), root390Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail391
    (if 4 ≤ root390Candidates.length then search edge 4 [2699] root390Candidates 2699 winner else winner) = winner
  rw [if_pos (by decide), root390Branch]
  exact outerScan391
set_option maxRecDepth 4096 in
theorem outerScan389 : scan edge (search edge 4) 5 [] 0 tail389 winner = winner := by
  rw [tail389, scan, if_neg (by decide), root389Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail390
    (if 4 ≤ root389Candidates.length then search edge 4 [2693] root389Candidates 2693 winner else winner) = winner
  rw [if_pos (by decide), root389Branch]
  exact outerScan390
set_option maxRecDepth 4096 in
theorem outerScan388 : scan edge (search edge 4) 5 [] 0 tail388 winner = winner := by
  rw [tail388, scan, if_neg (by decide), root388Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail389
    (if 4 ≤ root388Candidates.length then search edge 4 [2689] root388Candidates 2689 winner else winner) = winner
  rw [if_pos (by decide), root388Branch]
  exact outerScan389
set_option maxRecDepth 4096 in
theorem outerScan387 : scan edge (search edge 4) 5 [] 0 tail387 winner = winner := by
  rw [tail387, scan, if_neg (by decide), root387Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail388
    (if 4 ≤ root387Candidates.length then search edge 4 [2687] root387Candidates 2687 winner else winner) = winner
  rw [if_pos (by decide), root387Branch]
  exact outerScan388
set_option maxRecDepth 4096 in
theorem outerScan386 : scan edge (search edge 4) 5 [] 0 tail386 winner = winner := by
  rw [tail386, scan, if_neg (by decide), root386Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail387
    (if 4 ≤ root386Candidates.length then search edge 4 [2683] root386Candidates 2683 winner else winner) = winner
  rw [if_pos (by decide), root386Branch]
  exact outerScan387
end ProjectEulerSolutions.PrimePairLarge.Search
