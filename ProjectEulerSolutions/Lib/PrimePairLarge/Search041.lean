import ProjectEulerSolutions.Lib.PrimePairLarge.Search040
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot354
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot355
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot356
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot357
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot358
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot359
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot360
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot361
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan361 : scan edge (search edge 4) 5 [] 0 tail361 winner = winner := by
  rw [tail361, scan, if_neg (by decide), root361Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail362
    (if 4 ≤ root361Candidates.length then search edge 4 [2459] root361Candidates 2459 winner else winner) = winner
  rw [if_pos (by decide), root361Branch]
  exact outerScan362
set_option maxRecDepth 4096 in
theorem outerScan360 : scan edge (search edge 4) 5 [] 0 tail360 winner = winner := by
  rw [tail360, scan, if_neg (by decide), root360Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail361
    (if 4 ≤ root360Candidates.length then search edge 4 [2447] root360Candidates 2447 winner else winner) = winner
  rw [if_pos (by decide), root360Branch]
  exact outerScan361
set_option maxRecDepth 4096 in
theorem outerScan359 : scan edge (search edge 4) 5 [] 0 tail359 winner = winner := by
  rw [tail359, scan, if_neg (by decide), root359Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail360
    (if 4 ≤ root359Candidates.length then search edge 4 [2441] root359Candidates 2441 winner else winner) = winner
  rw [if_pos (by decide), root359Branch]
  exact outerScan360
set_option maxRecDepth 4096 in
theorem outerScan358 : scan edge (search edge 4) 5 [] 0 tail358 winner = winner := by
  rw [tail358, scan, if_neg (by decide), root358Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail359
    (if 4 ≤ root358Candidates.length then search edge 4 [2437] root358Candidates 2437 winner else winner) = winner
  rw [if_pos (by decide), root358Branch]
  exact outerScan359
set_option maxRecDepth 4096 in
theorem outerScan357 : scan edge (search edge 4) 5 [] 0 tail357 winner = winner := by
  rw [tail357, scan, if_neg (by decide), root357Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail358
    (if 4 ≤ root357Candidates.length then search edge 4 [2423] root357Candidates 2423 winner else winner) = winner
  rw [if_pos (by decide), root357Branch]
  exact outerScan358
set_option maxRecDepth 4096 in
theorem outerScan356 : scan edge (search edge 4) 5 [] 0 tail356 winner = winner := by
  rw [tail356, scan, if_neg (by decide), root356Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail357
    (if 4 ≤ root356Candidates.length then search edge 4 [2417] root356Candidates 2417 winner else winner) = winner
  rw [if_pos (by decide), root356Branch]
  exact outerScan357
set_option maxRecDepth 4096 in
theorem outerScan355 : scan edge (search edge 4) 5 [] 0 tail355 winner = winner := by
  rw [tail355, scan, if_neg (by decide), root355Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail356
    (if 4 ≤ root355Candidates.length then search edge 4 [2411] root355Candidates 2411 winner else winner) = winner
  rw [if_pos (by decide), root355Branch]
  exact outerScan356
set_option maxRecDepth 4096 in
theorem outerScan354 : scan edge (search edge 4) 5 [] 0 tail354 winner = winner := by
  rw [tail354, scan, if_neg (by decide), root354Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail355
    (if 4 ≤ root354Candidates.length then search edge 4 [2399] root354Candidates 2399 winner else winner) = winner
  rw [if_pos (by decide), root354Branch]
  exact outerScan355
end ProjectEulerSolutions.PrimePairLarge.Search
