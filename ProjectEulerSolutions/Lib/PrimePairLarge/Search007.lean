import ProjectEulerSolutions.Lib.PrimePairLarge.Search006
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot626
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot627
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot628
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot629
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot630
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot631
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot632
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot633
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan633 : scan edge (search edge 4) 5 [] 0 tail633 winner = winner := by
  rw [tail633, scan, if_neg (by decide), root633Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail634
    (if 4 ≤ root633Candidates.length then search edge 4 [4721] root633Candidates 4721 winner else winner) = winner
  rw [if_pos (by decide), root633Branch]
  exact outerScan634
set_option maxRecDepth 4096 in
theorem outerScan632 : scan edge (search edge 4) 5 [] 0 tail632 winner = winner := by
  rw [tail632, scan, if_neg (by decide), root632Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail633
    (if 4 ≤ root632Candidates.length then search edge 4 [4703] root632Candidates 4703 winner else winner) = winner
  rw [if_pos (by decide), root632Branch]
  exact outerScan633
set_option maxRecDepth 4096 in
theorem outerScan631 : scan edge (search edge 4) 5 [] 0 tail631 winner = winner := by
  rw [tail631, scan, if_neg (by decide), root631Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail632
    (if 4 ≤ root631Candidates.length then search edge 4 [4691] root631Candidates 4691 winner else winner) = winner
  rw [if_pos (by decide), root631Branch]
  exact outerScan632
set_option maxRecDepth 4096 in
theorem outerScan630 : scan edge (search edge 4) 5 [] 0 tail630 winner = winner := by
  rw [tail630, scan, if_neg (by decide), root630Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail631
    (if 4 ≤ root630Candidates.length then search edge 4 [4679] root630Candidates 4679 winner else winner) = winner
  rw [if_pos (by decide), root630Branch]
  exact outerScan631
set_option maxRecDepth 4096 in
theorem outerScan629 : scan edge (search edge 4) 5 [] 0 tail629 winner = winner := by
  rw [tail629, scan, if_neg (by decide), root629Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail630
    (if 4 ≤ root629Candidates.length then search edge 4 [4673] root629Candidates 4673 winner else winner) = winner
  rw [if_pos (by decide), root629Branch]
  exact outerScan630
set_option maxRecDepth 4096 in
theorem outerScan628 : scan edge (search edge 4) 5 [] 0 tail628 winner = winner := by
  rw [tail628, scan, if_neg (by decide), root628Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail629
    (if 4 ≤ root628Candidates.length then search edge 4 [4663] root628Candidates 4663 winner else winner) = winner
  rw [if_pos (by decide), root628Branch]
  exact outerScan629
set_option maxRecDepth 4096 in
theorem outerScan627 : scan edge (search edge 4) 5 [] 0 tail627 winner = winner := by
  rw [tail627, scan, if_neg (by decide), root627Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail628
    (if 4 ≤ root627Candidates.length then search edge 4 [4657] root627Candidates 4657 winner else winner) = winner
  rw [if_pos (by decide), root627Branch]
  exact outerScan628
set_option maxRecDepth 4096 in
theorem outerScan626 : scan edge (search edge 4) 5 [] 0 tail626 winner = winner := by
  rw [tail626, scan, if_neg (by decide), root626Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail627
    (if 4 ≤ root626Candidates.length then search edge 4 [4651] root626Candidates 4651 winner else winner) = winner
  rw [if_pos (by decide), root626Branch]
  exact outerScan627
end ProjectEulerSolutions.PrimePairLarge.Search
