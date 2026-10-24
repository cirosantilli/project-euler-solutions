import ProjectEulerSolutions.Lib.PrimePairLarge.Search070
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot114
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot115
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot116
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot117
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot118
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot119
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot120
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot121
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan121 : scan edge (search edge 4) 5 [] 0 tail121 winner = winner := by
  rw [tail121, scan, if_neg (by decide), root121Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail122
    (if 4 ≤ root121Candidates.length then search edge 4 [683] root121Candidates 683 winner else winner) = winner
  rw [if_pos (by decide), root121Branch]
  exact outerScan122
set_option maxRecDepth 4096 in
theorem outerScan120 : scan edge (search edge 4) 5 [] 0 tail120 winner = winner := by
  rw [tail120, scan, if_neg (by decide), root120Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail121
    (if 4 ≤ root120Candidates.length then search edge 4 [677] root120Candidates 677 winner else winner) = winner
  rw [if_pos (by decide), root120Branch]
  exact outerScan121
set_option maxRecDepth 4096 in
theorem outerScan119 : scan edge (search edge 4) 5 [] 0 tail119 winner = winner := by
  rw [tail119, scan, if_neg (by decide), root119Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail120
    (if 4 ≤ root119Candidates.length then search edge 4 [673] root119Candidates 673 winner else winner) = winner
  rw [if_pos (by decide), root119Branch]
  exact outerScan120
set_option maxRecDepth 4096 in
theorem outerScan118 : scan edge (search edge 4) 5 [] 0 tail118 winner = winner := by
  rw [tail118, scan, if_neg (by decide), root118Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail119
    (if 4 ≤ root118Candidates.length then search edge 4 [661] root118Candidates 661 winner else winner) = winner
  rw [if_pos (by decide), root118Branch]
  exact outerScan119
set_option maxRecDepth 4096 in
theorem outerScan117 : scan edge (search edge 4) 5 [] 0 tail117 winner = winner := by
  rw [tail117, scan, if_neg (by decide), root117Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail118
    (if 4 ≤ root117Candidates.length then search edge 4 [659] root117Candidates 659 winner else winner) = winner
  rw [if_pos (by decide), root117Branch]
  exact outerScan118
set_option maxRecDepth 4096 in
theorem outerScan116 : scan edge (search edge 4) 5 [] 0 tail116 winner = winner := by
  rw [tail116, scan, if_neg (by decide), root116Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail117
    (if 4 ≤ root116Candidates.length then search edge 4 [653] root116Candidates 653 winner else winner) = winner
  rw [if_pos (by decide), root116Branch]
  exact outerScan117
set_option maxRecDepth 4096 in
theorem outerScan115 : scan edge (search edge 4) 5 [] 0 tail115 winner = winner := by
  rw [tail115, scan, if_neg (by decide), root115Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail116
    (if 4 ≤ root115Candidates.length then search edge 4 [647] root115Candidates 647 winner else winner) = winner
  rw [if_pos (by decide), root115Branch]
  exact outerScan116
set_option maxRecDepth 4096 in
theorem outerScan114 : scan edge (search edge 4) 5 [] 0 tail114 winner = winner := by
  rw [tail114, scan, if_neg (by decide), root114Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail115
    (if 4 ≤ root114Candidates.length then search edge 4 [643] root114Candidates 643 winner else winner) = winner
  rw [if_pos (by decide), root114Branch]
  exact outerScan115
end ProjectEulerSolutions.PrimePairLarge.Search
