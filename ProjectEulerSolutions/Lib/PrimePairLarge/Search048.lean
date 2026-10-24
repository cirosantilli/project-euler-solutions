import ProjectEulerSolutions.Lib.PrimePairLarge.Search047
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot298
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot299
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot300
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot301
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot302
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot303
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot304
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot305
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan305 : scan edge (search edge 4) 5 [] 0 tail305 winner = winner := by
  rw [tail305, scan, if_neg (by decide), root305Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail306
    (if 4 ≤ root305Candidates.length then search edge 4 [2029] root305Candidates 2029 winner else winner) = winner
  rw [if_pos (by decide), root305Branch]
  exact outerScan306
set_option maxRecDepth 4096 in
theorem outerScan304 : scan edge (search edge 4) 5 [] 0 tail304 winner = winner := by
  rw [tail304, scan, if_neg (by decide), root304Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail305
    (if 4 ≤ root304Candidates.length then search edge 4 [2027] root304Candidates 2027 winner else winner) = winner
  rw [if_pos (by decide), root304Branch]
  exact outerScan305
set_option maxRecDepth 4096 in
theorem outerScan303 : scan edge (search edge 4) 5 [] 0 tail303 winner = winner := by
  rw [tail303, scan, if_neg (by decide), root303Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail304
    (if 4 ≤ root303Candidates.length then search edge 4 [2017] root303Candidates 2017 winner else winner) = winner
  rw [if_pos (by decide), root303Branch]
  exact outerScan304
set_option maxRecDepth 4096 in
theorem outerScan302 : scan edge (search edge 4) 5 [] 0 tail302 winner = winner := by
  rw [tail302, scan, if_neg (by decide), root302Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail303
    (if 4 ≤ root302Candidates.length then search edge 4 [2011] root302Candidates 2011 winner else winner) = winner
  rw [if_pos (by decide), root302Branch]
  exact outerScan303
set_option maxRecDepth 4096 in
theorem outerScan301 : scan edge (search edge 4) 5 [] 0 tail301 winner = winner := by
  rw [tail301, scan, if_neg (by decide), root301Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail302
    (if 4 ≤ root301Candidates.length then search edge 4 [2003] root301Candidates 2003 winner else winner) = winner
  rw [if_pos (by decide), root301Branch]
  exact outerScan302
set_option maxRecDepth 4096 in
theorem outerScan300 : scan edge (search edge 4) 5 [] 0 tail300 winner = winner := by
  rw [tail300, scan, if_neg (by decide), root300Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail301
    (if 4 ≤ root300Candidates.length then search edge 4 [1999] root300Candidates 1999 winner else winner) = winner
  rw [if_pos (by decide), root300Branch]
  exact outerScan301
set_option maxRecDepth 4096 in
theorem outerScan299 : scan edge (search edge 4) 5 [] 0 tail299 winner = winner := by
  rw [tail299, scan, if_neg (by decide), root299Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail300
    (if 4 ≤ root299Candidates.length then search edge 4 [1997] root299Candidates 1997 winner else winner) = winner
  rw [if_pos (by decide), root299Branch]
  exact outerScan300
set_option maxRecDepth 4096 in
theorem outerScan298 : scan edge (search edge 4) 5 [] 0 tail298 winner = winner := by
  rw [tail298, scan, if_neg (by decide), root298Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail299
    (if 4 ≤ root298Candidates.length then search edge 4 [1993] root298Candidates 1993 winner else winner) = winner
  rw [if_pos (by decide), root298Branch]
  exact outerScan299
end ProjectEulerSolutions.PrimePairLarge.Search
