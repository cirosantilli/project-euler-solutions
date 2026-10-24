import ProjectEulerSolutions.Lib.PrimePairLarge.Search066
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot146
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot147
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot148
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot149
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot150
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot151
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot152
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot153
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan153 : scan edge (search edge 4) 5 [] 0 tail153 winner = winner := by
  rw [tail153, scan, if_neg (by decide), root153Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail154
    (if 4 ≤ root153Candidates.length then search edge 4 [911] root153Candidates 911 winner else winner) = winner
  rw [if_pos (by decide), root153Branch]
  exact outerScan154
set_option maxRecDepth 4096 in
theorem outerScan152 : scan edge (search edge 4) 5 [] 0 tail152 winner = winner := by
  rw [tail152, scan, if_neg (by decide), root152Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail153
    (if 4 ≤ root152Candidates.length then search edge 4 [907] root152Candidates 907 winner else winner) = winner
  rw [if_pos (by decide), root152Branch]
  exact outerScan153
set_option maxRecDepth 4096 in
theorem outerScan151 : scan edge (search edge 4) 5 [] 0 tail151 winner = winner := by
  rw [tail151, scan, if_neg (by decide), root151Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail152
    (if 4 ≤ root151Candidates.length then search edge 4 [887] root151Candidates 887 winner else winner) = winner
  rw [if_pos (by decide), root151Branch]
  exact outerScan152
set_option maxRecDepth 4096 in
theorem outerScan150 : scan edge (search edge 4) 5 [] 0 tail150 winner = winner := by
  rw [tail150, scan, if_neg (by decide), root150Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail151
    (if 4 ≤ root150Candidates.length then search edge 4 [883] root150Candidates 883 winner else winner) = winner
  rw [if_pos (by decide), root150Branch]
  exact outerScan151
set_option maxRecDepth 4096 in
theorem outerScan149 : scan edge (search edge 4) 5 [] 0 tail149 winner = winner := by
  rw [tail149, scan, if_neg (by decide), root149Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail150
    (if 4 ≤ root149Candidates.length then search edge 4 [881] root149Candidates 881 winner else winner) = winner
  rw [if_pos (by decide), root149Branch]
  exact outerScan150
set_option maxRecDepth 4096 in
theorem outerScan148 : scan edge (search edge 4) 5 [] 0 tail148 winner = winner := by
  rw [tail148, scan, if_neg (by decide), root148Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail149
    (if 4 ≤ root148Candidates.length then search edge 4 [877] root148Candidates 877 winner else winner) = winner
  rw [if_pos (by decide), root148Branch]
  exact outerScan149
set_option maxRecDepth 4096 in
theorem outerScan147 : scan edge (search edge 4) 5 [] 0 tail147 winner = winner := by
  rw [tail147, scan, if_neg (by decide), root147Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail148
    (if 4 ≤ root147Candidates.length then search edge 4 [863] root147Candidates 863 winner else winner) = winner
  rw [if_pos (by decide), root147Branch]
  exact outerScan148
set_option maxRecDepth 4096 in
theorem outerScan146 : scan edge (search edge 4) 5 [] 0 tail146 winner = winner := by
  rw [tail146, scan, if_neg (by decide), root146Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail147
    (if 4 ≤ root146Candidates.length then search edge 4 [859] root146Candidates 859 winner else winner) = winner
  rw [if_pos (by decide), root146Branch]
  exact outerScan147
end ProjectEulerSolutions.PrimePairLarge.Search
