import ProjectEulerSolutions.Lib.PrimePairLarge.Search033
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot410
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot411
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot412
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot413
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot414
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot415
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot416
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot417
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan417 : scan edge (search edge 4) 5 [] 0 tail417 winner = winner := by
  rw [tail417, scan, if_neg (by decide), root417Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail418
    (if 4 ≤ root417Candidates.length then search edge 4 [2903] root417Candidates 2903 winner else winner) = winner
  rw [if_pos (by decide), root417Branch]
  exact outerScan418
set_option maxRecDepth 4096 in
theorem outerScan416 : scan edge (search edge 4) 5 [] 0 tail416 winner = winner := by
  rw [tail416, scan, if_neg (by decide), root416Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail417
    (if 4 ≤ root416Candidates.length then search edge 4 [2897] root416Candidates 2897 winner else winner) = winner
  rw [if_pos (by decide), root416Branch]
  exact outerScan417
set_option maxRecDepth 4096 in
theorem outerScan415 : scan edge (search edge 4) 5 [] 0 tail415 winner = winner := by
  rw [tail415, scan, if_neg (by decide), root415Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail416
    (if 4 ≤ root415Candidates.length then search edge 4 [2887] root415Candidates 2887 winner else winner) = winner
  rw [if_pos (by decide), root415Branch]
  exact outerScan416
set_option maxRecDepth 4096 in
theorem outerScan414 : scan edge (search edge 4) 5 [] 0 tail414 winner = winner := by
  rw [tail414, scan, if_neg (by decide), root414Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail415
    (if 4 ≤ root414Candidates.length then search edge 4 [2879] root414Candidates 2879 winner else winner) = winner
  rw [if_pos (by decide), root414Branch]
  exact outerScan415
set_option maxRecDepth 4096 in
theorem outerScan413 : scan edge (search edge 4) 5 [] 0 tail413 winner = winner := by
  rw [tail413, scan, if_neg (by decide), root413Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail414
    (if 4 ≤ root413Candidates.length then search edge 4 [2861] root413Candidates 2861 winner else winner) = winner
  rw [if_pos (by decide), root413Branch]
  exact outerScan414
set_option maxRecDepth 4096 in
theorem outerScan412 : scan edge (search edge 4) 5 [] 0 tail412 winner = winner := by
  rw [tail412, scan, if_neg (by decide), root412Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail413
    (if 4 ≤ root412Candidates.length then search edge 4 [2857] root412Candidates 2857 winner else winner) = winner
  rw [if_pos (by decide), root412Branch]
  exact outerScan413
set_option maxRecDepth 4096 in
theorem outerScan411 : scan edge (search edge 4) 5 [] 0 tail411 winner = winner := by
  rw [tail411, scan, if_neg (by decide), root411Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail412
    (if 4 ≤ root411Candidates.length then search edge 4 [2851] root411Candidates 2851 winner else winner) = winner
  rw [if_pos (by decide), root411Branch]
  exact outerScan412
set_option maxRecDepth 4096 in
theorem outerScan410 : scan edge (search edge 4) 5 [] 0 tail410 winner = winner := by
  rw [tail410, scan, if_neg (by decide), root410Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail411
    (if 4 ≤ root410Candidates.length then search edge 4 [2843] root410Candidates 2843 winner else winner) = winner
  rw [if_pos (by decide), root410Branch]
  exact outerScan411
end ProjectEulerSolutions.PrimePairLarge.Search
