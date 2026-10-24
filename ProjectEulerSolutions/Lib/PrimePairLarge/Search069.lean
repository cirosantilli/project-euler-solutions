import ProjectEulerSolutions.Lib.PrimePairLarge.Search068
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot130
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot131
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot132
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot133
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot134
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot135
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot136
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot137
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan137 : scan edge (search edge 4) 5 [] 0 tail137 winner = winner := by
  rw [tail137, scan, if_neg (by decide), root137Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail138
    (if 4 ≤ root137Candidates.length then search edge 4 [809] root137Candidates 809 winner else winner) = winner
  rw [if_pos (by decide), root137Branch]
  exact outerScan138
set_option maxRecDepth 4096 in
theorem outerScan136 : scan edge (search edge 4) 5 [] 0 tail136 winner = winner := by
  rw [tail136, scan, if_neg (by decide), root136Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail137
    (if 4 ≤ root136Candidates.length then search edge 4 [797] root136Candidates 797 winner else winner) = winner
  rw [if_pos (by decide), root136Branch]
  exact outerScan137
set_option maxRecDepth 4096 in
theorem outerScan135 : scan edge (search edge 4) 5 [] 0 tail135 winner = winner := by
  rw [tail135, scan, if_neg (by decide), root135Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail136
    (if 4 ≤ root135Candidates.length then search edge 4 [787] root135Candidates 787 winner else winner) = winner
  rw [if_pos (by decide), root135Branch]
  exact outerScan136
set_option maxRecDepth 4096 in
theorem outerScan134 : scan edge (search edge 4) 5 [] 0 tail134 winner = winner := by
  rw [tail134, scan, if_neg (by decide), root134Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail135
    (if 4 ≤ root134Candidates.length then search edge 4 [773] root134Candidates 773 winner else winner) = winner
  rw [if_pos (by decide), root134Branch]
  exact outerScan135
set_option maxRecDepth 4096 in
theorem outerScan133 : scan edge (search edge 4) 5 [] 0 tail133 winner = winner := by
  rw [tail133, scan, if_neg (by decide), root133Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail134
    (if 4 ≤ root133Candidates.length then search edge 4 [769] root133Candidates 769 winner else winner) = winner
  rw [if_pos (by decide), root133Branch]
  exact outerScan134
set_option maxRecDepth 4096 in
theorem outerScan132 : scan edge (search edge 4) 5 [] 0 tail132 winner = winner := by
  rw [tail132, scan, if_neg (by decide), root132Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail133
    (if 4 ≤ root132Candidates.length then search edge 4 [761] root132Candidates 761 winner else winner) = winner
  rw [if_pos (by decide), root132Branch]
  exact outerScan133
set_option maxRecDepth 4096 in
theorem outerScan131 : scan edge (search edge 4) 5 [] 0 tail131 winner = winner := by
  rw [tail131, scan, if_neg (by decide), root131Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail132
    (if 4 ≤ root131Candidates.length then search edge 4 [757] root131Candidates 757 winner else winner) = winner
  rw [if_pos (by decide), root131Branch]
  exact outerScan132
set_option maxRecDepth 4096 in
theorem outerScan130 : scan edge (search edge 4) 5 [] 0 tail130 winner = winner := by
  rw [tail130, scan, if_neg (by decide), root130Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail131
    (if 4 ≤ root130Candidates.length then search edge 4 [751] root130Candidates 751 winner else winner) = winner
  rw [if_pos (by decide), root130Branch]
  exact outerScan131
end ProjectEulerSolutions.PrimePairLarge.Search
