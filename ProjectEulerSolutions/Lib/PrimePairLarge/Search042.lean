import ProjectEulerSolutions.Lib.PrimePairLarge.Search041
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot346
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot347
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot348
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot349
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot350
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot351
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot352
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot353
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan353 : scan edge (search edge 4) 5 [] 0 tail353 winner = winner := by
  rw [tail353, scan, if_neg (by decide), root353Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail354
    (if 4 ≤ root353Candidates.length then search edge 4 [2393] root353Candidates 2393 winner else winner) = winner
  rw [if_pos (by decide), root353Branch]
  exact outerScan354
set_option maxRecDepth 4096 in
theorem outerScan352 : scan edge (search edge 4) 5 [] 0 tail352 winner = winner := by
  rw [tail352, scan, if_neg (by decide), root352Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail353
    (if 4 ≤ root352Candidates.length then search edge 4 [2389] root352Candidates 2389 winner else winner) = winner
  rw [if_pos (by decide), root352Branch]
  exact outerScan353
set_option maxRecDepth 4096 in
theorem outerScan351 : scan edge (search edge 4) 5 [] 0 tail351 winner = winner := by
  rw [tail351, scan, if_neg (by decide), root351Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail352
    (if 4 ≤ root351Candidates.length then search edge 4 [2383] root351Candidates 2383 winner else winner) = winner
  rw [if_pos (by decide), root351Branch]
  exact outerScan352
set_option maxRecDepth 4096 in
theorem outerScan350 : scan edge (search edge 4) 5 [] 0 tail350 winner = winner := by
  rw [tail350, scan, if_neg (by decide), root350Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail351
    (if 4 ≤ root350Candidates.length then search edge 4 [2381] root350Candidates 2381 winner else winner) = winner
  rw [if_pos (by decide), root350Branch]
  exact outerScan351
set_option maxRecDepth 4096 in
theorem outerScan349 : scan edge (search edge 4) 5 [] 0 tail349 winner = winner := by
  rw [tail349, scan, if_neg (by decide), root349Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail350
    (if 4 ≤ root349Candidates.length then search edge 4 [2377] root349Candidates 2377 winner else winner) = winner
  rw [if_pos (by decide), root349Branch]
  exact outerScan350
set_option maxRecDepth 4096 in
theorem outerScan348 : scan edge (search edge 4) 5 [] 0 tail348 winner = winner := by
  rw [tail348, scan, if_neg (by decide), root348Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail349
    (if 4 ≤ root348Candidates.length then search edge 4 [2371] root348Candidates 2371 winner else winner) = winner
  rw [if_pos (by decide), root348Branch]
  exact outerScan349
set_option maxRecDepth 4096 in
theorem outerScan347 : scan edge (search edge 4) 5 [] 0 tail347 winner = winner := by
  rw [tail347, scan, if_neg (by decide), root347Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail348
    (if 4 ≤ root347Candidates.length then search edge 4 [2357] root347Candidates 2357 winner else winner) = winner
  rw [if_pos (by decide), root347Branch]
  exact outerScan348
set_option maxRecDepth 4096 in
theorem outerScan346 : scan edge (search edge 4) 5 [] 0 tail346 winner = winner := by
  rw [tail346, scan, if_neg (by decide), root346Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail347
    (if 4 ≤ root346Candidates.length then search edge 4 [2351] root346Candidates 2351 winner else winner) = winner
  rw [if_pos (by decide), root346Branch]
  exact outerScan347
end ProjectEulerSolutions.PrimePairLarge.Search
