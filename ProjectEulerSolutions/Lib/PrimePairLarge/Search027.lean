import ProjectEulerSolutions.Lib.PrimePairLarge.Search026
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot466
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot467
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot468
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot469
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot470
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot471
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot472
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot473
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan473 : scan edge (search edge 4) 5 [] 0 tail473 winner = winner := by
  rw [tail473, scan, if_neg (by decide), root473Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail474
    (if 4 ≤ root473Candidates.length then search edge 4 [3373] root473Candidates 3373 winner else winner) = winner
  rw [if_pos (by decide), root473Branch]
  exact outerScan474
set_option maxRecDepth 4096 in
theorem outerScan472 : scan edge (search edge 4) 5 [] 0 tail472 winner = winner := by
  rw [tail472, scan, if_neg (by decide), root472Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail473
    (if 4 ≤ root472Candidates.length then search edge 4 [3371] root472Candidates 3371 winner else winner) = winner
  rw [if_pos (by decide), root472Branch]
  exact outerScan473
set_option maxRecDepth 4096 in
theorem outerScan471 : scan edge (search edge 4) 5 [] 0 tail471 winner = winner := by
  rw [tail471, scan, if_neg (by decide), root471Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail472
    (if 4 ≤ root471Candidates.length then search edge 4 [3361] root471Candidates 3361 winner else winner) = winner
  rw [if_pos (by decide), root471Branch]
  exact outerScan472
set_option maxRecDepth 4096 in
theorem outerScan470 : scan edge (search edge 4) 5 [] 0 tail470 winner = winner := by
  rw [tail470, scan, if_neg (by decide), root470Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail471
    (if 4 ≤ root470Candidates.length then search edge 4 [3359] root470Candidates 3359 winner else winner) = winner
  rw [if_pos (by decide), root470Branch]
  exact outerScan471
set_option maxRecDepth 4096 in
theorem outerScan469 : scan edge (search edge 4) 5 [] 0 tail469 winner = winner := by
  rw [tail469, scan, if_neg (by decide), root469Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail470
    (if 4 ≤ root469Candidates.length then search edge 4 [3347] root469Candidates 3347 winner else winner) = winner
  rw [if_pos (by decide), root469Branch]
  exact outerScan470
set_option maxRecDepth 4096 in
theorem outerScan468 : scan edge (search edge 4) 5 [] 0 tail468 winner = winner := by
  rw [tail468, scan, if_neg (by decide), root468Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail469
    (if 4 ≤ root468Candidates.length then search edge 4 [3343] root468Candidates 3343 winner else winner) = winner
  rw [if_pos (by decide), root468Branch]
  exact outerScan469
set_option maxRecDepth 4096 in
theorem outerScan467 : scan edge (search edge 4) 5 [] 0 tail467 winner = winner := by
  rw [tail467, scan, if_neg (by decide), root467Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail468
    (if 4 ≤ root467Candidates.length then search edge 4 [3331] root467Candidates 3331 winner else winner) = winner
  rw [if_pos (by decide), root467Branch]
  exact outerScan468
set_option maxRecDepth 4096 in
theorem outerScan466 : scan edge (search edge 4) 5 [] 0 tail466 winner = winner := by
  rw [tail466, scan, if_neg (by decide), root466Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail467
    (if 4 ≤ root466Candidates.length then search edge 4 [3329] root466Candidates 3329 winner else winner) = winner
  rw [if_pos (by decide), root466Branch]
  exact outerScan467
end ProjectEulerSolutions.PrimePairLarge.Search
