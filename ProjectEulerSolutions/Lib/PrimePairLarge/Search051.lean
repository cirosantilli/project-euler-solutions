import ProjectEulerSolutions.Lib.PrimePairLarge.Search050
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot274
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot275
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot276
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot277
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot278
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot279
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot280
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot281
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan281 : scan edge (search edge 4) 5 [] 0 tail281 winner = winner := by
  rw [tail281, scan, if_neg (by decide), root281Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail282
    (if 4 ≤ root281Candidates.length then search edge 4 [1861] root281Candidates 1861 winner else winner) = winner
  rw [if_pos (by decide), root281Branch]
  exact outerScan282
set_option maxRecDepth 4096 in
theorem outerScan280 : scan edge (search edge 4) 5 [] 0 tail280 winner = winner := by
  rw [tail280, scan, if_neg (by decide), root280Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail281
    (if 4 ≤ root280Candidates.length then search edge 4 [1847] root280Candidates 1847 winner else winner) = winner
  rw [if_pos (by decide), root280Branch]
  exact outerScan281
set_option maxRecDepth 4096 in
theorem outerScan279 : scan edge (search edge 4) 5 [] 0 tail279 winner = winner := by
  rw [tail279, scan, if_neg (by decide), root279Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail280
    (if 4 ≤ root279Candidates.length then search edge 4 [1831] root279Candidates 1831 winner else winner) = winner
  rw [if_pos (by decide), root279Branch]
  exact outerScan280
set_option maxRecDepth 4096 in
theorem outerScan278 : scan edge (search edge 4) 5 [] 0 tail278 winner = winner := by
  rw [tail278, scan, if_neg (by decide), root278Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail279
    (if 4 ≤ root278Candidates.length then search edge 4 [1823] root278Candidates 1823 winner else winner) = winner
  rw [if_pos (by decide), root278Branch]
  exact outerScan279
set_option maxRecDepth 4096 in
theorem outerScan277 : scan edge (search edge 4) 5 [] 0 tail277 winner = winner := by
  rw [tail277, scan, if_neg (by decide), root277Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail278
    (if 4 ≤ root277Candidates.length then search edge 4 [1811] root277Candidates 1811 winner else winner) = winner
  rw [if_pos (by decide), root277Branch]
  exact outerScan278
set_option maxRecDepth 4096 in
theorem outerScan276 : scan edge (search edge 4) 5 [] 0 tail276 winner = winner := by
  rw [tail276, scan, if_neg (by decide), root276Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail277
    (if 4 ≤ root276Candidates.length then search edge 4 [1801] root276Candidates 1801 winner else winner) = winner
  rw [if_pos (by decide), root276Branch]
  exact outerScan277
set_option maxRecDepth 4096 in
theorem outerScan275 : scan edge (search edge 4) 5 [] 0 tail275 winner = winner := by
  rw [tail275, scan, if_neg (by decide), root275Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail276
    (if 4 ≤ root275Candidates.length then search edge 4 [1789] root275Candidates 1789 winner else winner) = winner
  rw [if_pos (by decide), root275Branch]
  exact outerScan276
set_option maxRecDepth 4096 in
theorem outerScan274 : scan edge (search edge 4) 5 [] 0 tail274 winner = winner := by
  rw [tail274, scan, if_neg (by decide), root274Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail275
    (if 4 ≤ root274Candidates.length then search edge 4 [1787] root274Candidates 1787 winner else winner) = winner
  rw [if_pos (by decide), root274Branch]
  exact outerScan275
end ProjectEulerSolutions.PrimePairLarge.Search
