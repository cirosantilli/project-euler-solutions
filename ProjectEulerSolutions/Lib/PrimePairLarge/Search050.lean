import ProjectEulerSolutions.Lib.PrimePairLarge.Search049
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot282
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot283
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot284
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot285
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot286
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot287
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot288
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot289
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan289 : scan edge (search edge 4) 5 [] 0 tail289 winner = winner := by
  rw [tail289, scan, if_neg (by decide), root289Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail290
    (if 4 ≤ root289Candidates.length then search edge 4 [1907] root289Candidates 1907 winner else winner) = winner
  rw [if_pos (by decide), root289Branch]
  exact outerScan290
set_option maxRecDepth 4096 in
theorem outerScan288 : scan edge (search edge 4) 5 [] 0 tail288 winner = winner := by
  rw [tail288, scan, if_neg (by decide), root288Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail289
    (if 4 ≤ root288Candidates.length then search edge 4 [1901] root288Candidates 1901 winner else winner) = winner
  rw [if_pos (by decide), root288Branch]
  exact outerScan289
set_option maxRecDepth 4096 in
theorem outerScan287 : scan edge (search edge 4) 5 [] 0 tail287 winner = winner := by
  rw [tail287, scan, if_neg (by decide), root287Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail288
    (if 4 ≤ root287Candidates.length then search edge 4 [1889] root287Candidates 1889 winner else winner) = winner
  rw [if_pos (by decide), root287Branch]
  exact outerScan288
set_option maxRecDepth 4096 in
theorem outerScan286 : scan edge (search edge 4) 5 [] 0 tail286 winner = winner := by
  rw [tail286, scan, if_neg (by decide), root286Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail287
    (if 4 ≤ root286Candidates.length then search edge 4 [1879] root286Candidates 1879 winner else winner) = winner
  rw [if_pos (by decide), root286Branch]
  exact outerScan287
set_option maxRecDepth 4096 in
theorem outerScan285 : scan edge (search edge 4) 5 [] 0 tail285 winner = winner := by
  rw [tail285, scan, if_neg (by decide), root285Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail286
    (if 4 ≤ root285Candidates.length then search edge 4 [1877] root285Candidates 1877 winner else winner) = winner
  rw [if_pos (by decide), root285Branch]
  exact outerScan286
set_option maxRecDepth 4096 in
theorem outerScan284 : scan edge (search edge 4) 5 [] 0 tail284 winner = winner := by
  rw [tail284, scan, if_neg (by decide), root284Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail285
    (if 4 ≤ root284Candidates.length then search edge 4 [1873] root284Candidates 1873 winner else winner) = winner
  rw [if_pos (by decide), root284Branch]
  exact outerScan285
set_option maxRecDepth 4096 in
theorem outerScan283 : scan edge (search edge 4) 5 [] 0 tail283 winner = winner := by
  rw [tail283, scan, if_neg (by decide), root283Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail284
    (if 4 ≤ root283Candidates.length then search edge 4 [1871] root283Candidates 1871 winner else winner) = winner
  rw [if_pos (by decide), root283Branch]
  exact outerScan284
set_option maxRecDepth 4096 in
theorem outerScan282 : scan edge (search edge 4) 5 [] 0 tail282 winner = winner := by
  rw [tail282, scan, if_neg (by decide), root282Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail283
    (if 4 ≤ root282Candidates.length then search edge 4 [1867] root282Candidates 1867 winner else winner) = winner
  rw [if_pos (by decide), root282Branch]
  exact outerScan283
end ProjectEulerSolutions.PrimePairLarge.Search
