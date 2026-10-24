import ProjectEulerSolutions.Lib.PrimePairLarge.Search003
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot650
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot651
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot652
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot653
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot654
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot655
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot656
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot657
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan657 : scan edge (search edge 4) 5 [] 0 tail657 winner = winner := by
  rw [tail657, scan, if_neg (by decide), root657Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail658
    (if 4 ≤ root657Candidates.length then search edge 4 [4937] root657Candidates 4937 winner else winner) = winner
  rw [if_pos (by decide), root657Branch]
  exact outerScan658
set_option maxRecDepth 4096 in
theorem outerScan656 : scan edge (search edge 4) 5 [] 0 tail656 winner = winner := by
  rw [tail656, scan, if_neg (by decide), root656Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail657
    (if 4 ≤ root656Candidates.length then search edge 4 [4933] root656Candidates 4933 winner else winner) = winner
  rw [if_pos (by decide), root656Branch]
  exact outerScan657
set_option maxRecDepth 4096 in
theorem outerScan655 : scan edge (search edge 4) 5 [] 0 tail655 winner = winner := by
  rw [tail655, scan, if_neg (by decide), root655Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail656
    (if 4 ≤ root655Candidates.length then search edge 4 [4931] root655Candidates 4931 winner else winner) = winner
  rw [if_pos (by decide), root655Branch]
  exact outerScan656
set_option maxRecDepth 4096 in
theorem outerScan654 : scan edge (search edge 4) 5 [] 0 tail654 winner = winner := by
  rw [tail654, scan, if_neg (by decide), root654Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail655
    (if 4 ≤ root654Candidates.length then search edge 4 [4919] root654Candidates 4919 winner else winner) = winner
  rw [if_pos (by decide), root654Branch]
  exact outerScan655
set_option maxRecDepth 4096 in
theorem outerScan653 : scan edge (search edge 4) 5 [] 0 tail653 winner = winner := by
  rw [tail653, scan, if_neg (by decide), root653Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail654
    (if 4 ≤ root653Candidates.length then search edge 4 [4909] root653Candidates 4909 winner else winner) = winner
  rw [if_pos (by decide), root653Branch]
  exact outerScan654
set_option maxRecDepth 4096 in
theorem outerScan652 : scan edge (search edge 4) 5 [] 0 tail652 winner = winner := by
  rw [tail652, scan, if_neg (by decide), root652Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail653
    (if 4 ≤ root652Candidates.length then search edge 4 [4903] root652Candidates 4903 winner else winner) = winner
  rw [if_pos (by decide), root652Branch]
  exact outerScan653
set_option maxRecDepth 4096 in
theorem outerScan651 : scan edge (search edge 4) 5 [] 0 tail651 winner = winner := by
  rw [tail651, scan, if_neg (by decide), root651Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail652
    (if 4 ≤ root651Candidates.length then search edge 4 [4889] root651Candidates 4889 winner else winner) = winner
  rw [if_pos (by decide), root651Branch]
  exact outerScan652
set_option maxRecDepth 4096 in
theorem outerScan650 : scan edge (search edge 4) 5 [] 0 tail650 winner = winner := by
  rw [tail650, scan, if_neg (by decide), root650Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail651
    (if 4 ≤ root650Candidates.length then search edge 4 [4877] root650Candidates 4877 winner else winner) = winner
  rw [if_pos (by decide), root650Branch]
  exact outerScan651
end ProjectEulerSolutions.PrimePairLarge.Search
