import ProjectEulerSolutions.Lib.PrimePairLarge.Search072
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot098
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot099
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot100
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot101
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot102
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot103
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot104
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot105
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan105 : scan edge (search edge 4) 5 [] 0 tail105 winner = winner := by
  rw [tail105, scan, if_neg (by decide), root105Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail106
    (if 4 ≤ root105Candidates.length then search edge 4 [593] root105Candidates 593 winner else winner) = winner
  rw [if_pos (by decide), root105Branch]
  exact outerScan106
set_option maxRecDepth 4096 in
theorem outerScan104 : scan edge (search edge 4) 5 [] 0 tail104 winner = winner := by
  rw [tail104, scan, if_neg (by decide), root104Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail105
    (if 4 ≤ root104Candidates.length then search edge 4 [587] root104Candidates 587 winner else winner) = winner
  rw [if_pos (by decide), root104Branch]
  exact outerScan105
set_option maxRecDepth 4096 in
theorem outerScan103 : scan edge (search edge 4) 5 [] 0 tail103 winner = winner := by
  rw [tail103, scan, if_neg (by decide), root103Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail104
    (if 4 ≤ root103Candidates.length then search edge 4 [577] root103Candidates 577 winner else winner) = winner
  rw [if_pos (by decide), root103Branch]
  exact outerScan104
set_option maxRecDepth 4096 in
theorem outerScan102 : scan edge (search edge 4) 5 [] 0 tail102 winner = winner := by
  rw [tail102, scan, if_neg (by decide), root102Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail103
    (if 4 ≤ root102Candidates.length then search edge 4 [571] root102Candidates 571 winner else winner) = winner
  rw [if_pos (by decide), root102Branch]
  exact outerScan103
set_option maxRecDepth 4096 in
theorem outerScan101 : scan edge (search edge 4) 5 [] 0 tail101 winner = winner := by
  rw [tail101, scan, if_neg (by decide), root101Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail102
    (if 4 ≤ root101Candidates.length then search edge 4 [569] root101Candidates 569 winner else winner) = winner
  rw [if_pos (by decide), root101Branch]
  exact outerScan102
set_option maxRecDepth 4096 in
theorem outerScan100 : scan edge (search edge 4) 5 [] 0 tail100 winner = winner := by
  rw [tail100, scan, if_neg (by decide), root100Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail101
    (if 4 ≤ root100Candidates.length then search edge 4 [563] root100Candidates 563 winner else winner) = winner
  rw [if_pos (by decide), root100Branch]
  exact outerScan101
set_option maxRecDepth 4096 in
theorem outerScan99 : scan edge (search edge 4) 5 [] 0 tail99 winner = winner := by
  rw [tail99, scan, if_neg (by decide), root99Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail100
    (if 4 ≤ root99Candidates.length then search edge 4 [557] root99Candidates 557 winner else winner) = winner
  rw [if_pos (by decide), root99Branch]
  exact outerScan100
set_option maxRecDepth 4096 in
theorem outerScan98 : scan edge (search edge 4) 5 [] 0 tail98 winner = winner := by
  rw [tail98, scan, if_neg (by decide), root98Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail99
    (if 4 ≤ root98Candidates.length then search edge 4 [547] root98Candidates 547 winner else winner) = winner
  rw [if_pos (by decide), root98Branch]
  exact outerScan99
end ProjectEulerSolutions.PrimePairLarge.Search
