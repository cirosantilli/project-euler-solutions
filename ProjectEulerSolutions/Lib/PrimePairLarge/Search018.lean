import ProjectEulerSolutions.Lib.PrimePairLarge.Search017
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot538
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot539
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot540
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot541
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot542
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot543
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot544
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot545
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan545 : scan edge (search edge 4) 5 [] 0 tail545 winner = winner := by
  rw [tail545, scan, if_neg (by decide), root545Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail546
    (if 4 ≤ root545Candidates.length then search edge 4 [3947] root545Candidates 3947 winner else winner) = winner
  rw [if_pos (by decide), root545Branch]
  exact outerScan546
set_option maxRecDepth 4096 in
theorem outerScan544 : scan edge (search edge 4) 5 [] 0 tail544 winner = winner := by
  rw [tail544, scan, if_neg (by decide), root544Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail545
    (if 4 ≤ root544Candidates.length then search edge 4 [3943] root544Candidates 3943 winner else winner) = winner
  rw [if_pos (by decide), root544Branch]
  exact outerScan545
set_option maxRecDepth 4096 in
theorem outerScan543 : scan edge (search edge 4) 5 [] 0 tail543 winner = winner := by
  rw [tail543, scan, if_neg (by decide), root543Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail544
    (if 4 ≤ root543Candidates.length then search edge 4 [3931] root543Candidates 3931 winner else winner) = winner
  rw [if_pos (by decide), root543Branch]
  exact outerScan544
set_option maxRecDepth 4096 in
theorem outerScan542 : scan edge (search edge 4) 5 [] 0 tail542 winner = winner := by
  rw [tail542, scan, if_neg (by decide), root542Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail543
    (if 4 ≤ root542Candidates.length then search edge 4 [3929] root542Candidates 3929 winner else winner) = winner
  rw [if_pos (by decide), root542Branch]
  exact outerScan543
set_option maxRecDepth 4096 in
theorem outerScan541 : scan edge (search edge 4) 5 [] 0 tail541 winner = winner := by
  rw [tail541, scan, if_neg (by decide), root541Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail542
    (if 4 ≤ root541Candidates.length then search edge 4 [3923] root541Candidates 3923 winner else winner) = winner
  rw [if_pos (by decide), root541Branch]
  exact outerScan542
set_option maxRecDepth 4096 in
theorem outerScan540 : scan edge (search edge 4) 5 [] 0 tail540 winner = winner := by
  rw [tail540, scan, if_neg (by decide), root540Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail541
    (if 4 ≤ root540Candidates.length then search edge 4 [3919] root540Candidates 3919 winner else winner) = winner
  rw [if_pos (by decide), root540Branch]
  exact outerScan541
set_option maxRecDepth 4096 in
theorem outerScan539 : scan edge (search edge 4) 5 [] 0 tail539 winner = winner := by
  rw [tail539, scan, if_neg (by decide), root539Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail540
    (if 4 ≤ root539Candidates.length then search edge 4 [3917] root539Candidates 3917 winner else winner) = winner
  rw [if_pos (by decide), root539Branch]
  exact outerScan540
set_option maxRecDepth 4096 in
theorem outerScan538 : scan edge (search edge 4) 5 [] 0 tail538 winner = winner := by
  rw [tail538, scan, if_neg (by decide), root538Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail539
    (if 4 ≤ root538Candidates.length then search edge 4 [3911] root538Candidates 3911 winner else winner) = winner
  rw [if_pos (by decide), root538Branch]
  exact outerScan539
end ProjectEulerSolutions.PrimePairLarge.Search
