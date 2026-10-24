import ProjectEulerSolutions.Lib.PrimePairLarge.Search002
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot658
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot659
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot660
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot661
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot662
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot663
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot664
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot665
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan665 : scan edge (search edge 4) 5 [] 0 tail665 winner = winner := by
  rw [tail665, scan, if_neg (by decide), root665Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail666
    (if 4 ≤ root665Candidates.length then search edge 4 [4993] root665Candidates 4993 winner else winner) = winner
  rw [if_pos (by decide), root665Branch]
  exact outerScan666
set_option maxRecDepth 4096 in
theorem outerScan664 : scan edge (search edge 4) 5 [] 0 tail664 winner = winner := by
  rw [tail664, scan, if_neg (by decide), root664Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail665
    (if 4 ≤ root664Candidates.length then search edge 4 [4987] root664Candidates 4987 winner else winner) = winner
  rw [if_pos (by decide), root664Branch]
  exact outerScan665
set_option maxRecDepth 4096 in
theorem outerScan663 : scan edge (search edge 4) 5 [] 0 tail663 winner = winner := by
  rw [tail663, scan, if_neg (by decide), root663Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail664
    (if 4 ≤ root663Candidates.length then search edge 4 [4973] root663Candidates 4973 winner else winner) = winner
  rw [if_pos (by decide), root663Branch]
  exact outerScan664
set_option maxRecDepth 4096 in
theorem outerScan662 : scan edge (search edge 4) 5 [] 0 tail662 winner = winner := by
  rw [tail662, scan, if_neg (by decide), root662Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail663
    (if 4 ≤ root662Candidates.length then search edge 4 [4969] root662Candidates 4969 winner else winner) = winner
  rw [if_pos (by decide), root662Branch]
  exact outerScan663
set_option maxRecDepth 4096 in
theorem outerScan661 : scan edge (search edge 4) 5 [] 0 tail661 winner = winner := by
  rw [tail661, scan, if_neg (by decide), root661Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail662
    (if 4 ≤ root661Candidates.length then search edge 4 [4967] root661Candidates 4967 winner else winner) = winner
  rw [if_pos (by decide), root661Branch]
  exact outerScan662
set_option maxRecDepth 4096 in
theorem outerScan660 : scan edge (search edge 4) 5 [] 0 tail660 winner = winner := by
  rw [tail660, scan, if_neg (by decide), root660Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail661
    (if 4 ≤ root660Candidates.length then search edge 4 [4957] root660Candidates 4957 winner else winner) = winner
  rw [if_pos (by decide), root660Branch]
  exact outerScan661
set_option maxRecDepth 4096 in
theorem outerScan659 : scan edge (search edge 4) 5 [] 0 tail659 winner = winner := by
  rw [tail659, scan, if_neg (by decide), root659Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail660
    (if 4 ≤ root659Candidates.length then search edge 4 [4951] root659Candidates 4951 winner else winner) = winner
  rw [if_pos (by decide), root659Branch]
  exact outerScan660
set_option maxRecDepth 4096 in
theorem outerScan658 : scan edge (search edge 4) 5 [] 0 tail658 winner = winner := by
  rw [tail658, scan, if_neg (by decide), root658Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail659
    (if 4 ≤ root658Candidates.length then search edge 4 [4943] root658Candidates 4943 winner else winner) = winner
  rw [if_pos (by decide), root658Branch]
  exact outerScan659
end ProjectEulerSolutions.PrimePairLarge.Search
