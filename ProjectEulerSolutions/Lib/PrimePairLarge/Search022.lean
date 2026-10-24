import ProjectEulerSolutions.Lib.PrimePairLarge.Search021
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot506
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot507
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot508
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot509
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot510
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot511
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot512
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot513
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan513 : scan edge (search edge 4) 5 [] 0 tail513 winner = winner := by
  rw [tail513, scan, if_neg (by decide), root513Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail514
    (if 4 ≤ root513Candidates.length then search edge 4 [3697] root513Candidates 3697 winner else winner) = winner
  rw [if_pos (by decide), root513Branch]
  exact outerScan514
set_option maxRecDepth 4096 in
theorem outerScan512 : scan edge (search edge 4) 5 [] 0 tail512 winner = winner := by
  rw [tail512, scan, if_neg (by decide), root512Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail513
    (if 4 ≤ root512Candidates.length then search edge 4 [3691] root512Candidates 3691 winner else winner) = winner
  rw [if_pos (by decide), root512Branch]
  exact outerScan513
set_option maxRecDepth 4096 in
theorem outerScan511 : scan edge (search edge 4) 5 [] 0 tail511 winner = winner := by
  rw [tail511, scan, if_neg (by decide), root511Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail512
    (if 4 ≤ root511Candidates.length then search edge 4 [3677] root511Candidates 3677 winner else winner) = winner
  rw [if_pos (by decide), root511Branch]
  exact outerScan512
set_option maxRecDepth 4096 in
theorem outerScan510 : scan edge (search edge 4) 5 [] 0 tail510 winner = winner := by
  rw [tail510, scan, if_neg (by decide), root510Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail511
    (if 4 ≤ root510Candidates.length then search edge 4 [3673] root510Candidates 3673 winner else winner) = winner
  rw [if_pos (by decide), root510Branch]
  exact outerScan511
set_option maxRecDepth 4096 in
theorem outerScan509 : scan edge (search edge 4) 5 [] 0 tail509 winner = winner := by
  rw [tail509, scan, if_neg (by decide), root509Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail510
    (if 4 ≤ root509Candidates.length then search edge 4 [3671] root509Candidates 3671 winner else winner) = winner
  rw [if_pos (by decide), root509Branch]
  exact outerScan510
set_option maxRecDepth 4096 in
theorem outerScan508 : scan edge (search edge 4) 5 [] 0 tail508 winner = winner := by
  rw [tail508, scan, if_neg (by decide), root508Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail509
    (if 4 ≤ root508Candidates.length then search edge 4 [3659] root508Candidates 3659 winner else winner) = winner
  rw [if_pos (by decide), root508Branch]
  exact outerScan509
set_option maxRecDepth 4096 in
theorem outerScan507 : scan edge (search edge 4) 5 [] 0 tail507 winner = winner := by
  rw [tail507, scan, if_neg (by decide), root507Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail508
    (if 4 ≤ root507Candidates.length then search edge 4 [3643] root507Candidates 3643 winner else winner) = winner
  rw [if_pos (by decide), root507Branch]
  exact outerScan508
set_option maxRecDepth 4096 in
theorem outerScan506 : scan edge (search edge 4) 5 [] 0 tail506 winner = winner := by
  rw [tail506, scan, if_neg (by decide), root506Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail507
    (if 4 ≤ root506Candidates.length then search edge 4 [3637] root506Candidates 3637 winner else winner) = winner
  rw [if_pos (by decide), root506Branch]
  exact outerScan507
end ProjectEulerSolutions.PrimePairLarge.Search
