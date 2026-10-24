import ProjectEulerSolutions.Lib.PrimePairLarge.Search010
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot594
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot595
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot596
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot597
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot598
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot599
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot600
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot601
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan601 : scan edge (search edge 4) 5 [] 0 tail601 winner = winner := by
  rw [tail601, scan, if_neg (by decide), root601Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail602
    (if 4 ≤ root601Candidates.length then search edge 4 [4447] root601Candidates 4447 winner else winner) = winner
  rw [if_pos (by decide), root601Branch]
  exact outerScan602
set_option maxRecDepth 4096 in
theorem outerScan600 : scan edge (search edge 4) 5 [] 0 tail600 winner = winner := by
  rw [tail600, scan, if_neg (by decide), root600Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail601
    (if 4 ≤ root600Candidates.length then search edge 4 [4441] root600Candidates 4441 winner else winner) = winner
  rw [if_pos (by decide), root600Branch]
  exact outerScan601
set_option maxRecDepth 4096 in
theorem outerScan599 : scan edge (search edge 4) 5 [] 0 tail599 winner = winner := by
  rw [tail599, scan, if_neg (by decide), root599Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail600
    (if 4 ≤ root599Candidates.length then search edge 4 [4423] root599Candidates 4423 winner else winner) = winner
  rw [if_pos (by decide), root599Branch]
  exact outerScan600
set_option maxRecDepth 4096 in
theorem outerScan598 : scan edge (search edge 4) 5 [] 0 tail598 winner = winner := by
  rw [tail598, scan, if_neg (by decide), root598Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail599
    (if 4 ≤ root598Candidates.length then search edge 4 [4421] root598Candidates 4421 winner else winner) = winner
  rw [if_pos (by decide), root598Branch]
  exact outerScan599
set_option maxRecDepth 4096 in
theorem outerScan597 : scan edge (search edge 4) 5 [] 0 tail597 winner = winner := by
  rw [tail597, scan, if_neg (by decide), root597Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail598
    (if 4 ≤ root597Candidates.length then search edge 4 [4409] root597Candidates 4409 winner else winner) = winner
  rw [if_pos (by decide), root597Branch]
  exact outerScan598
set_option maxRecDepth 4096 in
theorem outerScan596 : scan edge (search edge 4) 5 [] 0 tail596 winner = winner := by
  rw [tail596, scan, if_neg (by decide), root596Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail597
    (if 4 ≤ root596Candidates.length then search edge 4 [4397] root596Candidates 4397 winner else winner) = winner
  rw [if_pos (by decide), root596Branch]
  exact outerScan597
set_option maxRecDepth 4096 in
theorem outerScan595 : scan edge (search edge 4) 5 [] 0 tail595 winner = winner := by
  rw [tail595, scan, if_neg (by decide), root595Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail596
    (if 4 ≤ root595Candidates.length then search edge 4 [4391] root595Candidates 4391 winner else winner) = winner
  rw [if_pos (by decide), root595Branch]
  exact outerScan596
set_option maxRecDepth 4096 in
theorem outerScan594 : scan edge (search edge 4) 5 [] 0 tail594 winner = winner := by
  rw [tail594, scan, if_neg (by decide), root594Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail595
    (if 4 ≤ root594Candidates.length then search edge 4 [4373] root594Candidates 4373 winner else winner) = winner
  rw [if_pos (by decide), root594Branch]
  exact outerScan595
end ProjectEulerSolutions.PrimePairLarge.Search
