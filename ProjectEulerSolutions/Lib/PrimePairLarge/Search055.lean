import ProjectEulerSolutions.Lib.PrimePairLarge.Search054
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot242
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot243
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot244
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot245
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot246
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot247
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot248
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot249
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan249 : scan edge (search edge 4) 5 [] 0 tail249 winner = winner := by
  rw [tail249, scan, if_neg (by decide), root249Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail250
    (if 4 ≤ root249Candidates.length then search edge 4 [1601] root249Candidates 1601 winner else winner) = winner
  rw [if_pos (by decide), root249Branch]
  exact outerScan250
set_option maxRecDepth 4096 in
theorem outerScan248 : scan edge (search edge 4) 5 [] 0 tail248 winner = winner := by
  rw [tail248, scan, if_neg (by decide), root248Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail249
    (if 4 ≤ root248Candidates.length then search edge 4 [1597] root248Candidates 1597 winner else winner) = winner
  rw [if_pos (by decide), root248Branch]
  exact outerScan249
set_option maxRecDepth 4096 in
theorem outerScan247 : scan edge (search edge 4) 5 [] 0 tail247 winner = winner := by
  rw [tail247, scan, if_neg (by decide), root247Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail248
    (if 4 ≤ root247Candidates.length then search edge 4 [1583] root247Candidates 1583 winner else winner) = winner
  rw [if_pos (by decide), root247Branch]
  exact outerScan248
set_option maxRecDepth 4096 in
theorem outerScan246 : scan edge (search edge 4) 5 [] 0 tail246 winner = winner := by
  rw [tail246, scan, if_neg (by decide), root246Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail247
    (if 4 ≤ root246Candidates.length then search edge 4 [1579] root246Candidates 1579 winner else winner) = winner
  rw [if_pos (by decide), root246Branch]
  exact outerScan247
set_option maxRecDepth 4096 in
theorem outerScan245 : scan edge (search edge 4) 5 [] 0 tail245 winner = winner := by
  rw [tail245, scan, if_neg (by decide), root245Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail246
    (if 4 ≤ root245Candidates.length then search edge 4 [1571] root245Candidates 1571 winner else winner) = winner
  rw [if_pos (by decide), root245Branch]
  exact outerScan246
set_option maxRecDepth 4096 in
theorem outerScan244 : scan edge (search edge 4) 5 [] 0 tail244 winner = winner := by
  rw [tail244, scan, if_neg (by decide), root244Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail245
    (if 4 ≤ root244Candidates.length then search edge 4 [1567] root244Candidates 1567 winner else winner) = winner
  rw [if_pos (by decide), root244Branch]
  exact outerScan245
set_option maxRecDepth 4096 in
theorem outerScan243 : scan edge (search edge 4) 5 [] 0 tail243 winner = winner := by
  rw [tail243, scan, if_neg (by decide), root243Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail244
    (if 4 ≤ root243Candidates.length then search edge 4 [1559] root243Candidates 1559 winner else winner) = winner
  rw [if_pos (by decide), root243Branch]
  exact outerScan244
set_option maxRecDepth 4096 in
theorem outerScan242 : scan edge (search edge 4) 5 [] 0 tail242 winner = winner := by
  rw [tail242, scan, if_neg (by decide), root242Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail243
    (if 4 ≤ root242Candidates.length then search edge 4 [1553] root242Candidates 1553 winner else winner) = winner
  rw [if_pos (by decide), root242Branch]
  exact outerScan243
end ProjectEulerSolutions.PrimePairLarge.Search
