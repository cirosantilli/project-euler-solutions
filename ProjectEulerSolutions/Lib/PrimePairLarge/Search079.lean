import ProjectEulerSolutions.Lib.PrimePairLarge.Search078
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot050
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot051
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot052
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot053
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot054
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot055
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot056
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot057
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan57 : scan edge (search edge 4) 5 [] 0 tail57 winner = winner := by
  rw [tail57, scan, if_neg (by decide), root57Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail58
    (if 4 ≤ root57Candidates.length then search edge 4 [281] root57Candidates 281 winner else winner) = winner
  rw [if_pos (by decide), root57Branch]
  exact outerScan58
set_option maxRecDepth 4096 in
theorem outerScan56 : scan edge (search edge 4) 5 [] 0 tail56 winner = winner := by
  rw [tail56, scan, if_neg (by decide), root56Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail57
    (if 4 ≤ root56Candidates.length then search edge 4 [277] root56Candidates 277 winner else winner) = winner
  rw [if_pos (by decide), root56Branch]
  exact outerScan57
set_option maxRecDepth 4096 in
theorem outerScan55 : scan edge (search edge 4) 5 [] 0 tail55 winner = winner := by
  rw [tail55, scan, if_neg (by decide), root55Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail56
    (if 4 ≤ root55Candidates.length then search edge 4 [271] root55Candidates 271 winner else winner) = winner
  rw [if_pos (by decide), root55Branch]
  exact outerScan56
set_option maxRecDepth 4096 in
theorem outerScan54 : scan edge (search edge 4) 5 [] 0 tail54 winner = winner := by
  rw [tail54, scan, if_neg (by decide), root54Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail55
    (if 4 ≤ root54Candidates.length then search edge 4 [269] root54Candidates 269 winner else winner) = winner
  rw [if_pos (by decide), root54Branch]
  exact outerScan55
set_option maxRecDepth 4096 in
theorem outerScan53 : scan edge (search edge 4) 5 [] 0 tail53 winner = winner := by
  rw [tail53, scan, if_neg (by decide), root53Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail54
    (if 4 ≤ root53Candidates.length then search edge 4 [263] root53Candidates 263 winner else winner) = winner
  rw [if_pos (by decide), root53Branch]
  exact outerScan54
set_option maxRecDepth 4096 in
theorem outerScan52 : scan edge (search edge 4) 5 [] 0 tail52 winner = winner := by
  rw [tail52, scan, if_neg (by decide), root52Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail53
    (if 4 ≤ root52Candidates.length then search edge 4 [257] root52Candidates 257 winner else winner) = winner
  rw [if_pos (by decide), root52Branch]
  exact outerScan53
set_option maxRecDepth 4096 in
theorem outerScan51 : scan edge (search edge 4) 5 [] 0 tail51 winner = winner := by
  rw [tail51, scan, if_neg (by decide), root51Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail52
    (if 4 ≤ root51Candidates.length then search edge 4 [251] root51Candidates 251 winner else winner) = winner
  rw [if_pos (by decide), root51Branch]
  exact outerScan52
set_option maxRecDepth 4096 in
theorem outerScan50 : scan edge (search edge 4) 5 [] 0 tail50 winner = winner := by
  rw [tail50, scan, if_neg (by decide), root50Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail51
    (if 4 ≤ root50Candidates.length then search edge 4 [241] root50Candidates 241 winner else winner) = winner
  rw [if_pos (by decide), root50Branch]
  exact outerScan51
end ProjectEulerSolutions.PrimePairLarge.Search
