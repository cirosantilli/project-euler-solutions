import ProjectEulerSolutions.Lib.PrimePairLarge.Search030
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot434
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot435
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot436
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot437
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot438
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot439
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot440
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot441
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan441 : scan edge (search edge 4) 5 [] 0 tail441 winner = winner := by
  rw [tail441, scan, if_neg (by decide), root441Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail442
    (if 4 ≤ root441Candidates.length then search edge 4 [3119] root441Candidates 3119 winner else winner) = winner
  rw [if_pos (by decide), root441Branch]
  exact outerScan442
set_option maxRecDepth 4096 in
theorem outerScan440 : scan edge (search edge 4) 5 [] 0 tail440 winner = winner := by
  rw [tail440, scan, if_neg (by decide), root440Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail441
    (if 4 ≤ root440Candidates.length then search edge 4 [3109] root440Candidates 3109 winner else winner) = winner
  rw [if_pos (by decide), root440Branch]
  exact outerScan441
set_option maxRecDepth 4096 in
theorem outerScan439 : scan edge (search edge 4) 5 [] 0 tail439 winner = winner := by
  rw [tail439, scan, if_neg (by decide), root439Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail440
    (if 4 ≤ root439Candidates.length then search edge 4 [3089] root439Candidates 3089 winner else winner) = winner
  rw [if_pos (by decide), root439Branch]
  exact outerScan440
set_option maxRecDepth 4096 in
theorem outerScan438 : scan edge (search edge 4) 5 [] 0 tail438 winner = winner := by
  rw [tail438, scan, if_neg (by decide), root438Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail439
    (if 4 ≤ root438Candidates.length then search edge 4 [3083] root438Candidates 3083 winner else winner) = winner
  rw [if_pos (by decide), root438Branch]
  exact outerScan439
set_option maxRecDepth 4096 in
theorem outerScan437 : scan edge (search edge 4) 5 [] 0 tail437 winner = winner := by
  rw [tail437, scan, if_neg (by decide), root437Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail438
    (if 4 ≤ root437Candidates.length then search edge 4 [3079] root437Candidates 3079 winner else winner) = winner
  rw [if_pos (by decide), root437Branch]
  exact outerScan438
set_option maxRecDepth 4096 in
theorem outerScan436 : scan edge (search edge 4) 5 [] 0 tail436 winner = winner := by
  rw [tail436, scan, if_neg (by decide), root436Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail437
    (if 4 ≤ root436Candidates.length then search edge 4 [3067] root436Candidates 3067 winner else winner) = winner
  rw [if_pos (by decide), root436Branch]
  exact outerScan437
set_option maxRecDepth 4096 in
theorem outerScan435 : scan edge (search edge 4) 5 [] 0 tail435 winner = winner := by
  rw [tail435, scan, if_neg (by decide), root435Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail436
    (if 4 ≤ root435Candidates.length then search edge 4 [3061] root435Candidates 3061 winner else winner) = winner
  rw [if_pos (by decide), root435Branch]
  exact outerScan436
set_option maxRecDepth 4096 in
theorem outerScan434 : scan edge (search edge 4) 5 [] 0 tail434 winner = winner := by
  rw [tail434, scan, if_neg (by decide), root434Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail435
    (if 4 ≤ root434Candidates.length then search edge 4 [3049] root434Candidates 3049 winner else winner) = winner
  rw [if_pos (by decide), root434Branch]
  exact outerScan435
end ProjectEulerSolutions.PrimePairLarge.Search
