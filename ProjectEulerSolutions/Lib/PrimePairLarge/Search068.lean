import ProjectEulerSolutions.Lib.PrimePairLarge.Search067
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot138
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot139
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot140
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot141
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot142
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot143
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot144
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot145
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan145 : scan edge (search edge 4) 5 [] 0 tail145 winner = winner := by
  rw [tail145, scan, if_neg (by decide), root145Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail146
    (if 4 ≤ root145Candidates.length then search edge 4 [857] root145Candidates 857 winner else winner) = winner
  rw [if_pos (by decide), root145Branch]
  exact outerScan146
set_option maxRecDepth 4096 in
theorem outerScan144 : scan edge (search edge 4) 5 [] 0 tail144 winner = winner := by
  rw [tail144, scan, if_neg (by decide), root144Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail145
    (if 4 ≤ root144Candidates.length then search edge 4 [853] root144Candidates 853 winner else winner) = winner
  rw [if_pos (by decide), root144Branch]
  exact outerScan145
set_option maxRecDepth 4096 in
theorem outerScan143 : scan edge (search edge 4) 5 [] 0 tail143 winner = winner := by
  rw [tail143, scan, if_neg (by decide), root143Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail144
    (if 4 ≤ root143Candidates.length then search edge 4 [839] root143Candidates 839 winner else winner) = winner
  rw [if_pos (by decide), root143Branch]
  exact outerScan144
set_option maxRecDepth 4096 in
theorem outerScan142 : scan edge (search edge 4) 5 [] 0 tail142 winner = winner := by
  rw [tail142, scan, if_neg (by decide), root142Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail143
    (if 4 ≤ root142Candidates.length then search edge 4 [829] root142Candidates 829 winner else winner) = winner
  rw [if_pos (by decide), root142Branch]
  exact outerScan143
set_option maxRecDepth 4096 in
theorem outerScan141 : scan edge (search edge 4) 5 [] 0 tail141 winner = winner := by
  rw [tail141, scan, if_neg (by decide), root141Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail142
    (if 4 ≤ root141Candidates.length then search edge 4 [827] root141Candidates 827 winner else winner) = winner
  rw [if_pos (by decide), root141Branch]
  exact outerScan142
set_option maxRecDepth 4096 in
theorem outerScan140 : scan edge (search edge 4) 5 [] 0 tail140 winner = winner := by
  rw [tail140, scan, if_neg (by decide), root140Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail141
    (if 4 ≤ root140Candidates.length then search edge 4 [823] root140Candidates 823 winner else winner) = winner
  rw [if_pos (by decide), root140Branch]
  exact outerScan141
set_option maxRecDepth 4096 in
theorem outerScan139 : scan edge (search edge 4) 5 [] 0 tail139 winner = winner := by
  rw [tail139, scan, if_neg (by decide), root139Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail140
    (if 4 ≤ root139Candidates.length then search edge 4 [821] root139Candidates 821 winner else winner) = winner
  rw [if_pos (by decide), root139Branch]
  exact outerScan140
set_option maxRecDepth 4096 in
theorem outerScan138 : scan edge (search edge 4) 5 [] 0 tail138 winner = winner := by
  rw [tail138, scan, if_neg (by decide), root138Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail139
    (if 4 ≤ root138Candidates.length then search edge 4 [811] root138Candidates 811 winner else winner) = winner
  rw [if_pos (by decide), root138Branch]
  exact outerScan139
end ProjectEulerSolutions.PrimePairLarge.Search
