import ProjectEulerSolutions.Lib.PrimePairLarge.Search048
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot290
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot291
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot292
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot293
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot294
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot295
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot296
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot297
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan297 : scan edge (search edge 4) 5 [] 0 tail297 winner = winner := by
  rw [tail297, scan, if_neg (by decide), root297Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail298
    (if 4 ≤ root297Candidates.length then search edge 4 [1987] root297Candidates 1987 winner else winner) = winner
  rw [if_pos (by decide), root297Branch]
  exact outerScan298
set_option maxRecDepth 4096 in
theorem outerScan296 : scan edge (search edge 4) 5 [] 0 tail296 winner = winner := by
  rw [tail296, scan, if_neg (by decide), root296Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail297
    (if 4 ≤ root296Candidates.length then search edge 4 [1979] root296Candidates 1979 winner else winner) = winner
  rw [if_pos (by decide), root296Branch]
  exact outerScan297
set_option maxRecDepth 4096 in
theorem outerScan295 : scan edge (search edge 4) 5 [] 0 tail295 winner = winner := by
  rw [tail295, scan, if_neg (by decide), root295Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail296
    (if 4 ≤ root295Candidates.length then search edge 4 [1973] root295Candidates 1973 winner else winner) = winner
  rw [if_pos (by decide), root295Branch]
  exact outerScan296
set_option maxRecDepth 4096 in
theorem outerScan294 : scan edge (search edge 4) 5 [] 0 tail294 winner = winner := by
  rw [tail294, scan, if_neg (by decide), root294Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail295
    (if 4 ≤ root294Candidates.length then search edge 4 [1951] root294Candidates 1951 winner else winner) = winner
  rw [if_pos (by decide), root294Branch]
  exact outerScan295
set_option maxRecDepth 4096 in
theorem outerScan293 : scan edge (search edge 4) 5 [] 0 tail293 winner = winner := by
  rw [tail293, scan, if_neg (by decide), root293Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail294
    (if 4 ≤ root293Candidates.length then search edge 4 [1949] root293Candidates 1949 winner else winner) = winner
  rw [if_pos (by decide), root293Branch]
  exact outerScan294
set_option maxRecDepth 4096 in
theorem outerScan292 : scan edge (search edge 4) 5 [] 0 tail292 winner = winner := by
  rw [tail292, scan, if_neg (by decide), root292Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail293
    (if 4 ≤ root292Candidates.length then search edge 4 [1933] root292Candidates 1933 winner else winner) = winner
  rw [if_pos (by decide), root292Branch]
  exact outerScan293
set_option maxRecDepth 4096 in
theorem outerScan291 : scan edge (search edge 4) 5 [] 0 tail291 winner = winner := by
  rw [tail291, scan, if_neg (by decide), root291Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail292
    (if 4 ≤ root291Candidates.length then search edge 4 [1931] root291Candidates 1931 winner else winner) = winner
  rw [if_pos (by decide), root291Branch]
  exact outerScan292
set_option maxRecDepth 4096 in
theorem outerScan290 : scan edge (search edge 4) 5 [] 0 tail290 winner = winner := by
  rw [tail290, scan, if_neg (by decide), root290Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail291
    (if 4 ≤ root290Candidates.length then search edge 4 [1913] root290Candidates 1913 winner else winner) = winner
  rw [if_pos (by decide), root290Branch]
  exact outerScan291
end ProjectEulerSolutions.PrimePairLarge.Search
