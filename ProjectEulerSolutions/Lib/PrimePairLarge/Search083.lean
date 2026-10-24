import ProjectEulerSolutions.Lib.PrimePairLarge.Search082
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot018
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot019
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot020
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot021
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot022
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot023
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot024
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot025
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan25 : scan edge (search edge 4) 5 [] 0 tail25 winner = winner := by
  rw [tail25, scan, if_neg (by decide), root25Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail26
    (if 4 ≤ root25Candidates.length then search edge 4 [107] root25Candidates 107 winner else winner) = winner
  rw [if_pos (by decide), root25Branch]
  exact outerScan26
set_option maxRecDepth 4096 in
theorem outerScan24 : scan edge (search edge 4) 5 [] 0 tail24 winner = winner := by
  rw [tail24, scan, if_neg (by decide), root24Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail25
    (if 4 ≤ root24Candidates.length then search edge 4 [103] root24Candidates 103 winner else winner) = winner
  rw [if_pos (by decide), root24Branch]
  exact outerScan25
set_option maxRecDepth 4096 in
theorem outerScan23 : scan edge (search edge 4) 5 [] 0 tail23 winner = winner := by
  rw [tail23, scan, if_neg (by decide), root23Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail24
    (if 4 ≤ root23Candidates.length then search edge 4 [101] root23Candidates 101 winner else winner) = winner
  rw [if_pos (by decide), root23Branch]
  exact outerScan24
set_option maxRecDepth 4096 in
theorem outerScan22 : scan edge (search edge 4) 5 [] 0 tail22 winner = winner := by
  rw [tail22, scan, if_neg (by decide), root22Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail23
    (if 4 ≤ root22Candidates.length then search edge 4 [97] root22Candidates 97 winner else winner) = winner
  rw [if_pos (by decide), root22Branch]
  exact outerScan23
set_option maxRecDepth 4096 in
theorem outerScan21 : scan edge (search edge 4) 5 [] 0 tail21 winner = winner := by
  rw [tail21, scan, if_neg (by decide), root21Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail22
    (if 4 ≤ root21Candidates.length then search edge 4 [89] root21Candidates 89 winner else winner) = winner
  rw [if_pos (by decide), root21Branch]
  exact outerScan22
set_option maxRecDepth 4096 in
theorem outerScan20 : scan edge (search edge 4) 5 [] 0 tail20 winner = winner := by
  rw [tail20, scan, if_neg (by decide), root20Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail21
    (if 4 ≤ root20Candidates.length then search edge 4 [83] root20Candidates 83 winner else winner) = winner
  rw [if_pos (by decide), root20Branch]
  exact outerScan21
set_option maxRecDepth 4096 in
theorem outerScan19 : scan edge (search edge 4) 5 [] 0 tail19 winner = winner := by
  rw [tail19, scan, if_neg (by decide), root19Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail20
    (if 4 ≤ root19Candidates.length then search edge 4 [79] root19Candidates 79 winner else winner) = winner
  rw [if_pos (by decide), root19Branch]
  exact outerScan20
set_option maxRecDepth 4096 in
theorem outerScan18 : scan edge (search edge 4) 5 [] 0 tail18 winner = winner := by
  rw [tail18, scan, if_neg (by decide), root18Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail19
    (if 4 ≤ root18Candidates.length then search edge 4 [73] root18Candidates 73 winner else winner) = winner
  rw [if_pos (by decide), root18Branch]
  exact outerScan19
end ProjectEulerSolutions.PrimePairLarge.Search
