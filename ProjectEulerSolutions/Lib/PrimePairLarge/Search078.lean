import ProjectEulerSolutions.Lib.PrimePairLarge.Search077
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot058
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot059
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot060
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot061
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot062
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot063
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot064
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot065
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan65 : scan edge (search edge 4) 5 [] 0 tail65 winner = winner := by
  rw [tail65, scan, if_neg (by decide), root65Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail66
    (if 4 ≤ root65Candidates.length then search edge 4 [337] root65Candidates 337 winner else winner) = winner
  rw [if_pos (by decide), root65Branch]
  exact outerScan66
set_option maxRecDepth 4096 in
theorem outerScan64 : scan edge (search edge 4) 5 [] 0 tail64 winner = winner := by
  rw [tail64, scan, if_neg (by decide), root64Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail65
    (if 4 ≤ root64Candidates.length then search edge 4 [331] root64Candidates 331 winner else winner) = winner
  rw [if_pos (by decide), root64Branch]
  exact outerScan65
set_option maxRecDepth 4096 in
theorem outerScan63 : scan edge (search edge 4) 5 [] 0 tail63 winner = winner := by
  rw [tail63, scan, if_neg (by decide), root63Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail64
    (if 4 ≤ root63Candidates.length then search edge 4 [317] root63Candidates 317 winner else winner) = winner
  rw [if_pos (by decide), root63Branch]
  exact outerScan64
set_option maxRecDepth 4096 in
theorem outerScan62 : scan edge (search edge 4) 5 [] 0 tail62 winner = winner := by
  rw [tail62, scan, if_neg (by decide), root62Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail63
    (if 4 ≤ root62Candidates.length then search edge 4 [313] root62Candidates 313 winner else winner) = winner
  rw [if_pos (by decide), root62Branch]
  exact outerScan63
set_option maxRecDepth 4096 in
theorem outerScan61 : scan edge (search edge 4) 5 [] 0 tail61 winner = winner := by
  rw [tail61, scan, if_neg (by decide), root61Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail62
    (if 4 ≤ root61Candidates.length then search edge 4 [311] root61Candidates 311 winner else winner) = winner
  rw [if_pos (by decide), root61Branch]
  exact outerScan62
set_option maxRecDepth 4096 in
theorem outerScan60 : scan edge (search edge 4) 5 [] 0 tail60 winner = winner := by
  rw [tail60, scan, if_neg (by decide), root60Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail61
    (if 4 ≤ root60Candidates.length then search edge 4 [307] root60Candidates 307 winner else winner) = winner
  rw [if_pos (by decide), root60Branch]
  exact outerScan61
set_option maxRecDepth 4096 in
theorem outerScan59 : scan edge (search edge 4) 5 [] 0 tail59 winner = winner := by
  rw [tail59, scan, if_neg (by decide), root59Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail60
    (if 4 ≤ root59Candidates.length then search edge 4 [293] root59Candidates 293 winner else winner) = winner
  rw [if_pos (by decide), root59Branch]
  exact outerScan60
set_option maxRecDepth 4096 in
theorem outerScan58 : scan edge (search edge 4) 5 [] 0 tail58 winner = winner := by
  rw [tail58, scan, if_neg (by decide), root58Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail59
    (if 4 ≤ root58Candidates.length then search edge 4 [283] root58Candidates 283 winner else winner) = winner
  rw [if_pos (by decide), root58Branch]
  exact outerScan59
end ProjectEulerSolutions.PrimePairLarge.Search
