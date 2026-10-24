import ProjectEulerSolutions.Lib.PrimePairLarge.Search075
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot074
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot075
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot076
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot077
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot078
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot079
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot080
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot081
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan81 : scan edge (search edge 4) 5 [] 0 tail81 winner = winner := by
  rw [tail81, scan, if_neg (by decide), root81Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail82
    (if 4 ≤ root81Candidates.length then search edge 4 [433] root81Candidates 433 winner else winner) = winner
  rw [if_pos (by decide), root81Branch]
  exact outerScan82
set_option maxRecDepth 4096 in
theorem outerScan80 : scan edge (search edge 4) 5 [] 0 tail80 winner = winner := by
  rw [tail80, scan, if_neg (by decide), root80Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail81
    (if 4 ≤ root80Candidates.length then search edge 4 [431] root80Candidates 431 winner else winner) = winner
  rw [if_pos (by decide), root80Branch]
  exact outerScan81
set_option maxRecDepth 4096 in
theorem outerScan79 : scan edge (search edge 4) 5 [] 0 tail79 winner = winner := by
  rw [tail79, scan, if_neg (by decide), root79Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail80
    (if 4 ≤ root79Candidates.length then search edge 4 [421] root79Candidates 421 winner else winner) = winner
  rw [if_pos (by decide), root79Branch]
  exact outerScan80
set_option maxRecDepth 4096 in
theorem outerScan78 : scan edge (search edge 4) 5 [] 0 tail78 winner = winner := by
  rw [tail78, scan, if_neg (by decide), root78Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail79
    (if 4 ≤ root78Candidates.length then search edge 4 [419] root78Candidates 419 winner else winner) = winner
  rw [if_pos (by decide), root78Branch]
  exact outerScan79
set_option maxRecDepth 4096 in
theorem outerScan77 : scan edge (search edge 4) 5 [] 0 tail77 winner = winner := by
  rw [tail77, scan, if_neg (by decide), root77Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail78
    (if 4 ≤ root77Candidates.length then search edge 4 [409] root77Candidates 409 winner else winner) = winner
  rw [if_pos (by decide), root77Branch]
  exact outerScan78
set_option maxRecDepth 4096 in
theorem outerScan76 : scan edge (search edge 4) 5 [] 0 tail76 winner = winner := by
  rw [tail76, scan, if_neg (by decide), root76Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail77
    (if 4 ≤ root76Candidates.length then search edge 4 [401] root76Candidates 401 winner else winner) = winner
  rw [if_pos (by decide), root76Branch]
  exact outerScan77
set_option maxRecDepth 4096 in
theorem outerScan75 : scan edge (search edge 4) 5 [] 0 tail75 winner = winner := by
  rw [tail75, scan, if_neg (by decide), root75Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail76
    (if 4 ≤ root75Candidates.length then search edge 4 [397] root75Candidates 397 winner else winner) = winner
  rw [if_pos (by decide), root75Branch]
  exact outerScan76
set_option maxRecDepth 4096 in
theorem outerScan74 : scan edge (search edge 4) 5 [] 0 tail74 winner = winner := by
  rw [tail74, scan, if_neg (by decide), root74Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail75
    (if 4 ≤ root74Candidates.length then search edge 4 [389] root74Candidates 389 winner else winner) = winner
  rw [if_pos (by decide), root74Branch]
  exact outerScan75
end ProjectEulerSolutions.PrimePairLarge.Search
