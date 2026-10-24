import ProjectEulerSolutions.Lib.PrimePairLarge.Search000
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot674
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot675
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot676
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot677
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot678
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot679
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot680
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot681
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan681 : scan edge (search edge 4) 5 [] 0 tail681 winner = winner := by
  rw [tail681, scan, if_neg (by decide), root681Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail682
    (if 4 ≤ root681Candidates.length then search edge 4 [5113] root681Candidates 5113 winner else winner) = winner
  rw [if_pos (by decide), root681Branch]
  exact outerScan682
set_option maxRecDepth 4096 in
theorem outerScan680 : scan edge (search edge 4) 5 [] 0 tail680 winner = winner := by
  rw [tail680, scan, if_neg (by decide), root680Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail681
    (if 4 ≤ root680Candidates.length then search edge 4 [5107] root680Candidates 5107 winner else winner) = winner
  rw [if_pos (by decide), root680Branch]
  exact outerScan681
set_option maxRecDepth 4096 in
theorem outerScan679 : scan edge (search edge 4) 5 [] 0 tail679 winner = winner := by
  rw [tail679, scan, if_neg (by decide), root679Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail680
    (if 4 ≤ root679Candidates.length then search edge 4 [5101] root679Candidates 5101 winner else winner) = winner
  rw [if_pos (by decide), root679Branch]
  exact outerScan680
set_option maxRecDepth 4096 in
theorem outerScan678 : scan edge (search edge 4) 5 [] 0 tail678 winner = winner := by
  rw [tail678, scan, if_neg (by decide), root678Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail679
    (if 4 ≤ root678Candidates.length then search edge 4 [5099] root678Candidates 5099 winner else winner) = winner
  rw [if_pos (by decide), root678Branch]
  exact outerScan679
set_option maxRecDepth 4096 in
theorem outerScan677 : scan edge (search edge 4) 5 [] 0 tail677 winner = winner := by
  rw [tail677, scan, if_neg (by decide), root677Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail678
    (if 4 ≤ root677Candidates.length then search edge 4 [5087] root677Candidates 5087 winner else winner) = winner
  rw [if_pos (by decide), root677Branch]
  exact outerScan678
set_option maxRecDepth 4096 in
theorem outerScan676 : scan edge (search edge 4) 5 [] 0 tail676 winner = winner := by
  rw [tail676, scan, if_neg (by decide), root676Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail677
    (if 4 ≤ root676Candidates.length then search edge 4 [5081] root676Candidates 5081 winner else winner) = winner
  rw [if_pos (by decide), root676Branch]
  exact outerScan677
set_option maxRecDepth 4096 in
theorem outerScan675 : scan edge (search edge 4) 5 [] 0 tail675 winner = winner := by
  rw [tail675, scan, if_neg (by decide), root675Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail676
    (if 4 ≤ root675Candidates.length then search edge 4 [5077] root675Candidates 5077 winner else winner) = winner
  rw [if_pos (by decide), root675Branch]
  exact outerScan676
set_option maxRecDepth 4096 in
theorem outerScan674 : scan edge (search edge 4) 5 [] 0 tail674 winner = winner := by
  rw [tail674, scan, if_neg (by decide), root674Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail675
    (if 4 ≤ root674Candidates.length then search edge 4 [5059] root674Candidates 5059 winner else winner) = winner
  rw [if_pos (by decide), root674Branch]
  exact outerScan675
end ProjectEulerSolutions.PrimePairLarge.Search
