import ProjectEulerSolutions.Lib.PrimePairLarge.Search016
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot546
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot547
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot548
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot549
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot550
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot551
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot552
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot553
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan553 : scan edge (search edge 4) 5 [] 0 tail553 winner = winner := by
  rw [tail553, scan, if_neg (by decide), root553Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail554
    (if 4 ≤ root553Candidates.length then search edge 4 [4021] root553Candidates 4021 winner else winner) = winner
  rw [if_pos (by decide), root553Branch]
  exact outerScan554
set_option maxRecDepth 4096 in
theorem outerScan552 : scan edge (search edge 4) 5 [] 0 tail552 winner = winner := by
  rw [tail552, scan, if_neg (by decide), root552Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail553
    (if 4 ≤ root552Candidates.length then search edge 4 [4019] root552Candidates 4019 winner else winner) = winner
  rw [if_pos (by decide), root552Branch]
  exact outerScan553
set_option maxRecDepth 4096 in
theorem outerScan551 : scan edge (search edge 4) 5 [] 0 tail551 winner = winner := by
  rw [tail551, scan, if_neg (by decide), root551Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail552
    (if 4 ≤ root551Candidates.length then search edge 4 [4013] root551Candidates 4013 winner else winner) = winner
  rw [if_pos (by decide), root551Branch]
  exact outerScan552
set_option maxRecDepth 4096 in
theorem outerScan550 : scan edge (search edge 4) 5 [] 0 tail550 winner = winner := by
  rw [tail550, scan, if_neg (by decide), root550Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail551
    (if 4 ≤ root550Candidates.length then search edge 4 [4007] root550Candidates 4007 winner else winner) = winner
  rw [if_pos (by decide), root550Branch]
  exact outerScan551
set_option maxRecDepth 4096 in
theorem outerScan549 : scan edge (search edge 4) 5 [] 0 tail549 winner = winner := by
  rw [tail549, scan, if_neg (by decide), root549Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail550
    (if 4 ≤ root549Candidates.length then search edge 4 [4003] root549Candidates 4003 winner else winner) = winner
  rw [if_pos (by decide), root549Branch]
  exact outerScan550
set_option maxRecDepth 4096 in
theorem outerScan548 : scan edge (search edge 4) 5 [] 0 tail548 winner = winner := by
  rw [tail548, scan, if_neg (by decide), root548Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail549
    (if 4 ≤ root548Candidates.length then search edge 4 [4001] root548Candidates 4001 winner else winner) = winner
  rw [if_pos (by decide), root548Branch]
  exact outerScan549
set_option maxRecDepth 4096 in
theorem outerScan547 : scan edge (search edge 4) 5 [] 0 tail547 winner = winner := by
  rw [tail547, scan, if_neg (by decide), root547Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail548
    (if 4 ≤ root547Candidates.length then search edge 4 [3989] root547Candidates 3989 winner else winner) = winner
  rw [if_pos (by decide), root547Branch]
  exact outerScan548
set_option maxRecDepth 4096 in
theorem outerScan546 : scan edge (search edge 4) 5 [] 0 tail546 winner = winner := by
  rw [tail546, scan, if_neg (by decide), root546Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail547
    (if 4 ≤ root546Candidates.length then search edge 4 [3967] root546Candidates 3967 winner else winner) = winner
  rw [if_pos (by decide), root546Branch]
  exact outerScan547
end ProjectEulerSolutions.PrimePairLarge.Search
