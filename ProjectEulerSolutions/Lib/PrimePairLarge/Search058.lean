import ProjectEulerSolutions.Lib.PrimePairLarge.Search057
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot218
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot219
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot220
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot221
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot222
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot223
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot224
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot225
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan225 : scan edge (search edge 4) 5 [] 0 tail225 winner = winner := by
  rw [tail225, scan, if_neg (by decide), root225Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail226
    (if 4 ≤ root225Candidates.length then search edge 4 [1439] root225Candidates 1439 winner else winner) = winner
  rw [if_pos (by decide), root225Branch]
  exact outerScan226
set_option maxRecDepth 4096 in
theorem outerScan224 : scan edge (search edge 4) 5 [] 0 tail224 winner = winner := by
  rw [tail224, scan, if_neg (by decide), root224Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail225
    (if 4 ≤ root224Candidates.length then search edge 4 [1433] root224Candidates 1433 winner else winner) = winner
  rw [if_pos (by decide), root224Branch]
  exact outerScan225
set_option maxRecDepth 4096 in
theorem outerScan223 : scan edge (search edge 4) 5 [] 0 tail223 winner = winner := by
  rw [tail223, scan, if_neg (by decide), root223Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail224
    (if 4 ≤ root223Candidates.length then search edge 4 [1429] root223Candidates 1429 winner else winner) = winner
  rw [if_pos (by decide), root223Branch]
  exact outerScan224
set_option maxRecDepth 4096 in
theorem outerScan222 : scan edge (search edge 4) 5 [] 0 tail222 winner = winner := by
  rw [tail222, scan, if_neg (by decide), root222Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail223
    (if 4 ≤ root222Candidates.length then search edge 4 [1427] root222Candidates 1427 winner else winner) = winner
  rw [if_pos (by decide), root222Branch]
  exact outerScan223
set_option maxRecDepth 4096 in
theorem outerScan221 : scan edge (search edge 4) 5 [] 0 tail221 winner = winner := by
  rw [tail221, scan, if_neg (by decide), root221Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail222
    (if 4 ≤ root221Candidates.length then search edge 4 [1423] root221Candidates 1423 winner else winner) = winner
  rw [if_pos (by decide), root221Branch]
  exact outerScan222
set_option maxRecDepth 4096 in
theorem outerScan220 : scan edge (search edge 4) 5 [] 0 tail220 winner = winner := by
  rw [tail220, scan, if_neg (by decide), root220Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail221
    (if 4 ≤ root220Candidates.length then search edge 4 [1409] root220Candidates 1409 winner else winner) = winner
  rw [if_pos (by decide), root220Branch]
  exact outerScan221
set_option maxRecDepth 4096 in
theorem outerScan219 : scan edge (search edge 4) 5 [] 0 tail219 winner = winner := by
  rw [tail219, scan, if_neg (by decide), root219Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail220
    (if 4 ≤ root219Candidates.length then search edge 4 [1399] root219Candidates 1399 winner else winner) = winner
  rw [if_pos (by decide), root219Branch]
  exact outerScan220
set_option maxRecDepth 4096 in
theorem outerScan218 : scan edge (search edge 4) 5 [] 0 tail218 winner = winner := by
  rw [tail218, scan, if_neg (by decide), root218Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail219
    (if 4 ≤ root218Candidates.length then search edge 4 [1381] root218Candidates 1381 winner else winner) = winner
  rw [if_pos (by decide), root218Branch]
  exact outerScan219
end ProjectEulerSolutions.PrimePairLarge.Search
