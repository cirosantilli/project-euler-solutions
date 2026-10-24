import ProjectEulerSolutions.Lib.PrimePairLarge.Search027
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot458
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot459
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot460
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot461
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot462
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot463
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot464
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot465
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan465 : scan edge (search edge 4) 5 [] 0 tail465 winner = winner := by
  rw [tail465, scan, if_neg (by decide), root465Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail466
    (if 4 ≤ root465Candidates.length then search edge 4 [3323] root465Candidates 3323 winner else winner) = winner
  rw [if_pos (by decide), root465Branch]
  exact outerScan466
set_option maxRecDepth 4096 in
theorem outerScan464 : scan edge (search edge 4) 5 [] 0 tail464 winner = winner := by
  rw [tail464, scan, if_neg (by decide), root464Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail465
    (if 4 ≤ root464Candidates.length then search edge 4 [3319] root464Candidates 3319 winner else winner) = winner
  rw [if_pos (by decide), root464Branch]
  exact outerScan465
set_option maxRecDepth 4096 in
theorem outerScan463 : scan edge (search edge 4) 5 [] 0 tail463 winner = winner := by
  rw [tail463, scan, if_neg (by decide), root463Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail464
    (if 4 ≤ root463Candidates.length then search edge 4 [3313] root463Candidates 3313 winner else winner) = winner
  rw [if_pos (by decide), root463Branch]
  exact outerScan464
set_option maxRecDepth 4096 in
theorem outerScan462 : scan edge (search edge 4) 5 [] 0 tail462 winner = winner := by
  rw [tail462, scan, if_neg (by decide), root462Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail463
    (if 4 ≤ root462Candidates.length then search edge 4 [3307] root462Candidates 3307 winner else winner) = winner
  rw [if_pos (by decide), root462Branch]
  exact outerScan463
set_option maxRecDepth 4096 in
theorem outerScan461 : scan edge (search edge 4) 5 [] 0 tail461 winner = winner := by
  rw [tail461, scan, if_neg (by decide), root461Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail462
    (if 4 ≤ root461Candidates.length then search edge 4 [3301] root461Candidates 3301 winner else winner) = winner
  rw [if_pos (by decide), root461Branch]
  exact outerScan462
set_option maxRecDepth 4096 in
theorem outerScan460 : scan edge (search edge 4) 5 [] 0 tail460 winner = winner := by
  rw [tail460, scan, if_neg (by decide), root460Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail461
    (if 4 ≤ root460Candidates.length then search edge 4 [3299] root460Candidates 3299 winner else winner) = winner
  rw [if_pos (by decide), root460Branch]
  exact outerScan461
set_option maxRecDepth 4096 in
theorem outerScan459 : scan edge (search edge 4) 5 [] 0 tail459 winner = winner := by
  rw [tail459, scan, if_neg (by decide), root459Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail460
    (if 4 ≤ root459Candidates.length then search edge 4 [3271] root459Candidates 3271 winner else winner) = winner
  rw [if_pos (by decide), root459Branch]
  exact outerScan460
set_option maxRecDepth 4096 in
theorem outerScan458 : scan edge (search edge 4) 5 [] 0 tail458 winner = winner := by
  rw [tail458, scan, if_neg (by decide), root458Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail459
    (if 4 ≤ root458Candidates.length then search edge 4 [3259] root458Candidates 3259 winner else winner) = winner
  rw [if_pos (by decide), root458Branch]
  exact outerScan459
end ProjectEulerSolutions.PrimePairLarge.Search
