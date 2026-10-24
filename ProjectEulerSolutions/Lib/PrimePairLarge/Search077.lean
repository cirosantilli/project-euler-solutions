import ProjectEulerSolutions.Lib.PrimePairLarge.Search076
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot066
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot067
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot068
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot069
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot070
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot071
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot072
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot073
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan73 : scan edge (search edge 4) 5 [] 0 tail73 winner = winner := by
  rw [tail73, scan, if_neg (by decide), root73Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail74
    (if 4 ≤ root73Candidates.length then search edge 4 [383] root73Candidates 383 winner else winner) = winner
  rw [if_pos (by decide), root73Branch]
  exact outerScan74
set_option maxRecDepth 4096 in
theorem outerScan72 : scan edge (search edge 4) 5 [] 0 tail72 winner = winner := by
  rw [tail72, scan, if_neg (by decide), root72Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail73
    (if 4 ≤ root72Candidates.length then search edge 4 [379] root72Candidates 379 winner else winner) = winner
  rw [if_pos (by decide), root72Branch]
  exact outerScan73
set_option maxRecDepth 4096 in
theorem outerScan71 : scan edge (search edge 4) 5 [] 0 tail71 winner = winner := by
  rw [tail71, scan, if_neg (by decide), root71Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail72
    (if 4 ≤ root71Candidates.length then search edge 4 [373] root71Candidates 373 winner else winner) = winner
  rw [if_pos (by decide), root71Branch]
  exact outerScan72
set_option maxRecDepth 4096 in
theorem outerScan70 : scan edge (search edge 4) 5 [] 0 tail70 winner = winner := by
  rw [tail70, scan, if_neg (by decide), root70Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail71
    (if 4 ≤ root70Candidates.length then search edge 4 [367] root70Candidates 367 winner else winner) = winner
  rw [if_pos (by decide), root70Branch]
  exact outerScan71
set_option maxRecDepth 4096 in
theorem outerScan69 : scan edge (search edge 4) 5 [] 0 tail69 winner = winner := by
  rw [tail69, scan, if_neg (by decide), root69Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail70
    (if 4 ≤ root69Candidates.length then search edge 4 [359] root69Candidates 359 winner else winner) = winner
  rw [if_pos (by decide), root69Branch]
  exact outerScan70
set_option maxRecDepth 4096 in
theorem outerScan68 : scan edge (search edge 4) 5 [] 0 tail68 winner = winner := by
  rw [tail68, scan, if_neg (by decide), root68Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail69
    (if 4 ≤ root68Candidates.length then search edge 4 [353] root68Candidates 353 winner else winner) = winner
  rw [if_pos (by decide), root68Branch]
  exact outerScan69
set_option maxRecDepth 4096 in
theorem outerScan67 : scan edge (search edge 4) 5 [] 0 tail67 winner = winner := by
  rw [tail67, scan, if_neg (by decide), root67Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail68
    (if 4 ≤ root67Candidates.length then search edge 4 [349] root67Candidates 349 winner else winner) = winner
  rw [if_pos (by decide), root67Branch]
  exact outerScan68
set_option maxRecDepth 4096 in
theorem outerScan66 : scan edge (search edge 4) 5 [] 0 tail66 winner = winner := by
  rw [tail66, scan, if_neg (by decide), root66Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail67
    (if 4 ≤ root66Candidates.length then search edge 4 [347] root66Candidates 347 winner else winner) = winner
  rw [if_pos (by decide), root66Branch]
  exact outerScan67
end ProjectEulerSolutions.PrimePairLarge.Search
