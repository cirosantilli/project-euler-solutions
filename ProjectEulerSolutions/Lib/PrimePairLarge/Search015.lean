import ProjectEulerSolutions.Lib.PrimePairLarge.Search014
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot562
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot563
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot564
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot565
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot566
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot567
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot568
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot569
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan569 : scan edge (search edge 4) 5 [] 0 tail569 winner = winner := by
  rw [tail569, scan, if_neg (by decide), root569Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail570
    (if 4 ≤ root569Candidates.length then search edge 4 [4157] root569Candidates 4157 winner else winner) = winner
  rw [if_pos (by decide), root569Branch]
  exact outerScan570
set_option maxRecDepth 4096 in
theorem outerScan568 : scan edge (search edge 4) 5 [] 0 tail568 winner = winner := by
  rw [tail568, scan, if_neg (by decide), root568Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail569
    (if 4 ≤ root568Candidates.length then search edge 4 [4153] root568Candidates 4153 winner else winner) = winner
  rw [if_pos (by decide), root568Branch]
  exact outerScan569
set_option maxRecDepth 4096 in
theorem outerScan567 : scan edge (search edge 4) 5 [] 0 tail567 winner = winner := by
  rw [tail567, scan, if_neg (by decide), root567Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail568
    (if 4 ≤ root567Candidates.length then search edge 4 [4139] root567Candidates 4139 winner else winner) = winner
  rw [if_pos (by decide), root567Branch]
  exact outerScan568
set_option maxRecDepth 4096 in
theorem outerScan566 : scan edge (search edge 4) 5 [] 0 tail566 winner = winner := by
  rw [tail566, scan, if_neg (by decide), root566Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail567
    (if 4 ≤ root566Candidates.length then search edge 4 [4133] root566Candidates 4133 winner else winner) = winner
  rw [if_pos (by decide), root566Branch]
  exact outerScan567
set_option maxRecDepth 4096 in
theorem outerScan565 : scan edge (search edge 4) 5 [] 0 tail565 winner = winner := by
  rw [tail565, scan, if_neg (by decide), root565Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail566
    (if 4 ≤ root565Candidates.length then search edge 4 [4129] root565Candidates 4129 winner else winner) = winner
  rw [if_pos (by decide), root565Branch]
  exact outerScan566
set_option maxRecDepth 4096 in
theorem outerScan564 : scan edge (search edge 4) 5 [] 0 tail564 winner = winner := by
  rw [tail564, scan, if_neg (by decide), root564Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail565
    (if 4 ≤ root564Candidates.length then search edge 4 [4127] root564Candidates 4127 winner else winner) = winner
  rw [if_pos (by decide), root564Branch]
  exact outerScan565
set_option maxRecDepth 4096 in
theorem outerScan563 : scan edge (search edge 4) 5 [] 0 tail563 winner = winner := by
  rw [tail563, scan, if_neg (by decide), root563Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail564
    (if 4 ≤ root563Candidates.length then search edge 4 [4111] root563Candidates 4111 winner else winner) = winner
  rw [if_pos (by decide), root563Branch]
  exact outerScan564
set_option maxRecDepth 4096 in
theorem outerScan562 : scan edge (search edge 4) 5 [] 0 tail562 winner = winner := by
  rw [tail562, scan, if_neg (by decide), root562Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail563
    (if 4 ≤ root562Candidates.length then search edge 4 [4099] root562Candidates 4099 winner else winner) = winner
  rw [if_pos (by decide), root562Branch]
  exact outerScan563
end ProjectEulerSolutions.PrimePairLarge.Search
