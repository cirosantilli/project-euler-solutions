import ProjectEulerSolutions.Lib.PrimePairLarge.Search042
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot338
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot339
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot340
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot341
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot342
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot343
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot344
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot345
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan345 : scan edge (search edge 4) 5 [] 0 tail345 winner = winner := by
  rw [tail345, scan, if_neg (by decide), root345Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail346
    (if 4 ≤ root345Candidates.length then search edge 4 [2347] root345Candidates 2347 winner else winner) = winner
  rw [if_pos (by decide), root345Branch]
  exact outerScan346
set_option maxRecDepth 4096 in
theorem outerScan344 : scan edge (search edge 4) 5 [] 0 tail344 winner = winner := by
  rw [tail344, scan, if_neg (by decide), root344Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail345
    (if 4 ≤ root344Candidates.length then search edge 4 [2341] root344Candidates 2341 winner else winner) = winner
  rw [if_pos (by decide), root344Branch]
  exact outerScan345
set_option maxRecDepth 4096 in
theorem outerScan343 : scan edge (search edge 4) 5 [] 0 tail343 winner = winner := by
  rw [tail343, scan, if_neg (by decide), root343Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail344
    (if 4 ≤ root343Candidates.length then search edge 4 [2339] root343Candidates 2339 winner else winner) = winner
  rw [if_pos (by decide), root343Branch]
  exact outerScan344
set_option maxRecDepth 4096 in
theorem outerScan342 : scan edge (search edge 4) 5 [] 0 tail342 winner = winner := by
  rw [tail342, scan, if_neg (by decide), root342Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail343
    (if 4 ≤ root342Candidates.length then search edge 4 [2333] root342Candidates 2333 winner else winner) = winner
  rw [if_pos (by decide), root342Branch]
  exact outerScan343
set_option maxRecDepth 4096 in
theorem outerScan341 : scan edge (search edge 4) 5 [] 0 tail341 winner = winner := by
  rw [tail341, scan, if_neg (by decide), root341Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail342
    (if 4 ≤ root341Candidates.length then search edge 4 [2311] root341Candidates 2311 winner else winner) = winner
  rw [if_pos (by decide), root341Branch]
  exact outerScan342
set_option maxRecDepth 4096 in
theorem outerScan340 : scan edge (search edge 4) 5 [] 0 tail340 winner = winner := by
  rw [tail340, scan, if_neg (by decide), root340Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail341
    (if 4 ≤ root340Candidates.length then search edge 4 [2309] root340Candidates 2309 winner else winner) = winner
  rw [if_pos (by decide), root340Branch]
  exact outerScan341
set_option maxRecDepth 4096 in
theorem outerScan339 : scan edge (search edge 4) 5 [] 0 tail339 winner = winner := by
  rw [tail339, scan, if_neg (by decide), root339Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail340
    (if 4 ≤ root339Candidates.length then search edge 4 [2297] root339Candidates 2297 winner else winner) = winner
  rw [if_pos (by decide), root339Branch]
  exact outerScan340
set_option maxRecDepth 4096 in
theorem outerScan338 : scan edge (search edge 4) 5 [] 0 tail338 winner = winner := by
  rw [tail338, scan, if_neg (by decide), root338Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail339
    (if 4 ≤ root338Candidates.length then search edge 4 [2293] root338Candidates 2293 winner else winner) = winner
  rw [if_pos (by decide), root338Branch]
  exact outerScan339
end ProjectEulerSolutions.PrimePairLarge.Search
