import ProjectEulerSolutions.Lib.PrimePairLarge.Search007
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot618
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot619
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot620
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot621
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot622
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot623
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot624
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot625
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan625 : scan edge (search edge 4) 5 [] 0 tail625 winner = winner := by
  rw [tail625, scan, if_neg (by decide), root625Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail626
    (if 4 ≤ root625Candidates.length then search edge 4 [4649] root625Candidates 4649 winner else winner) = winner
  rw [if_pos (by decide), root625Branch]
  exact outerScan626
set_option maxRecDepth 4096 in
theorem outerScan624 : scan edge (search edge 4) 5 [] 0 tail624 winner = winner := by
  rw [tail624, scan, if_neg (by decide), root624Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail625
    (if 4 ≤ root624Candidates.length then search edge 4 [4643] root624Candidates 4643 winner else winner) = winner
  rw [if_pos (by decide), root624Branch]
  exact outerScan625
set_option maxRecDepth 4096 in
theorem outerScan623 : scan edge (search edge 4) 5 [] 0 tail623 winner = winner := by
  rw [tail623, scan, if_neg (by decide), root623Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail624
    (if 4 ≤ root623Candidates.length then search edge 4 [4639] root623Candidates 4639 winner else winner) = winner
  rw [if_pos (by decide), root623Branch]
  exact outerScan624
set_option maxRecDepth 4096 in
theorem outerScan622 : scan edge (search edge 4) 5 [] 0 tail622 winner = winner := by
  rw [tail622, scan, if_neg (by decide), root622Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail623
    (if 4 ≤ root622Candidates.length then search edge 4 [4637] root622Candidates 4637 winner else winner) = winner
  rw [if_pos (by decide), root622Branch]
  exact outerScan623
set_option maxRecDepth 4096 in
theorem outerScan621 : scan edge (search edge 4) 5 [] 0 tail621 winner = winner := by
  rw [tail621, scan, if_neg (by decide), root621Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail622
    (if 4 ≤ root621Candidates.length then search edge 4 [4621] root621Candidates 4621 winner else winner) = winner
  rw [if_pos (by decide), root621Branch]
  exact outerScan622
set_option maxRecDepth 4096 in
theorem outerScan620 : scan edge (search edge 4) 5 [] 0 tail620 winner = winner := by
  rw [tail620, scan, if_neg (by decide), root620Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail621
    (if 4 ≤ root620Candidates.length then search edge 4 [4603] root620Candidates 4603 winner else winner) = winner
  rw [if_pos (by decide), root620Branch]
  exact outerScan621
set_option maxRecDepth 4096 in
theorem outerScan619 : scan edge (search edge 4) 5 [] 0 tail619 winner = winner := by
  rw [tail619, scan, if_neg (by decide), root619Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail620
    (if 4 ≤ root619Candidates.length then search edge 4 [4597] root619Candidates 4597 winner else winner) = winner
  rw [if_pos (by decide), root619Branch]
  exact outerScan620
set_option maxRecDepth 4096 in
theorem outerScan618 : scan edge (search edge 4) 5 [] 0 tail618 winner = winner := by
  rw [tail618, scan, if_neg (by decide), root618Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail619
    (if 4 ≤ root618Candidates.length then search edge 4 [4591] root618Candidates 4591 winner else winner) = winner
  rw [if_pos (by decide), root618Branch]
  exact outerScan619
end ProjectEulerSolutions.PrimePairLarge.Search
