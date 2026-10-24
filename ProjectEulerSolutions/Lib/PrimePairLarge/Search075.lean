import ProjectEulerSolutions.Lib.PrimePairLarge.Search074
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot082
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot083
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot084
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot085
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot086
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot087
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot088
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot089
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan89 : scan edge (search edge 4) 5 [] 0 tail89 winner = winner := by
  rw [tail89, scan, if_neg (by decide), root89Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail90
    (if 4 ≤ root89Candidates.length then search edge 4 [479] root89Candidates 479 winner else winner) = winner
  rw [if_pos (by decide), root89Branch]
  exact outerScan90
set_option maxRecDepth 4096 in
theorem outerScan88 : scan edge (search edge 4) 5 [] 0 tail88 winner = winner := by
  rw [tail88, scan, if_neg (by decide), root88Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail89
    (if 4 ≤ root88Candidates.length then search edge 4 [467] root88Candidates 467 winner else winner) = winner
  rw [if_pos (by decide), root88Branch]
  exact outerScan89
set_option maxRecDepth 4096 in
theorem outerScan87 : scan edge (search edge 4) 5 [] 0 tail87 winner = winner := by
  rw [tail87, scan, if_neg (by decide), root87Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail88
    (if 4 ≤ root87Candidates.length then search edge 4 [463] root87Candidates 463 winner else winner) = winner
  rw [if_pos (by decide), root87Branch]
  exact outerScan88
set_option maxRecDepth 4096 in
theorem outerScan86 : scan edge (search edge 4) 5 [] 0 tail86 winner = winner := by
  rw [tail86, scan, if_neg (by decide), root86Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail87
    (if 4 ≤ root86Candidates.length then search edge 4 [461] root86Candidates 461 winner else winner) = winner
  rw [if_pos (by decide), root86Branch]
  exact outerScan87
set_option maxRecDepth 4096 in
theorem outerScan85 : scan edge (search edge 4) 5 [] 0 tail85 winner = winner := by
  rw [tail85, scan, if_neg (by decide), root85Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail86
    (if 4 ≤ root85Candidates.length then search edge 4 [457] root85Candidates 457 winner else winner) = winner
  rw [if_pos (by decide), root85Branch]
  exact outerScan86
set_option maxRecDepth 4096 in
theorem outerScan84 : scan edge (search edge 4) 5 [] 0 tail84 winner = winner := by
  rw [tail84, scan, if_neg (by decide), root84Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail85
    (if 4 ≤ root84Candidates.length then search edge 4 [449] root84Candidates 449 winner else winner) = winner
  rw [if_pos (by decide), root84Branch]
  exact outerScan85
set_option maxRecDepth 4096 in
theorem outerScan83 : scan edge (search edge 4) 5 [] 0 tail83 winner = winner := by
  rw [tail83, scan, if_neg (by decide), root83Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail84
    (if 4 ≤ root83Candidates.length then search edge 4 [443] root83Candidates 443 winner else winner) = winner
  rw [if_pos (by decide), root83Branch]
  exact outerScan84
set_option maxRecDepth 4096 in
theorem outerScan82 : scan edge (search edge 4) 5 [] 0 tail82 winner = winner := by
  rw [tail82, scan, if_neg (by decide), root82Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail83
    (if 4 ≤ root82Candidates.length then search edge 4 [439] root82Candidates 439 winner else winner) = winner
  rw [if_pos (by decide), root82Branch]
  exact outerScan83
end ProjectEulerSolutions.PrimePairLarge.Search
