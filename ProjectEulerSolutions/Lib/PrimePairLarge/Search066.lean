import ProjectEulerSolutions.Lib.PrimePairLarge.Search065
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot154
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot155
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot156
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot157
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot158
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot159
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot160
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot161
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan161 : scan edge (search edge 4) 5 [] 0 tail161 winner = winner := by
  rw [tail161, scan, if_neg (by decide), root161Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail162
    (if 4 ≤ root161Candidates.length then search edge 4 [971] root161Candidates 971 winner else winner) = winner
  rw [if_pos (by decide), root161Branch]
  exact outerScan162
set_option maxRecDepth 4096 in
theorem outerScan160 : scan edge (search edge 4) 5 [] 0 tail160 winner = winner := by
  rw [tail160, scan, if_neg (by decide), root160Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail161
    (if 4 ≤ root160Candidates.length then search edge 4 [967] root160Candidates 967 winner else winner) = winner
  rw [if_pos (by decide), root160Branch]
  exact outerScan161
set_option maxRecDepth 4096 in
theorem outerScan159 : scan edge (search edge 4) 5 [] 0 tail159 winner = winner := by
  rw [tail159, scan, if_neg (by decide), root159Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail160
    (if 4 ≤ root159Candidates.length then search edge 4 [953] root159Candidates 953 winner else winner) = winner
  rw [if_pos (by decide), root159Branch]
  exact outerScan160
set_option maxRecDepth 4096 in
theorem outerScan158 : scan edge (search edge 4) 5 [] 0 tail158 winner = winner := by
  rw [tail158, scan, if_neg (by decide), root158Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail159
    (if 4 ≤ root158Candidates.length then search edge 4 [947] root158Candidates 947 winner else winner) = winner
  rw [if_pos (by decide), root158Branch]
  exact outerScan159
set_option maxRecDepth 4096 in
theorem outerScan157 : scan edge (search edge 4) 5 [] 0 tail157 winner = winner := by
  rw [tail157, scan, if_neg (by decide), root157Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail158
    (if 4 ≤ root157Candidates.length then search edge 4 [941] root157Candidates 941 winner else winner) = winner
  rw [if_pos (by decide), root157Branch]
  exact outerScan158
set_option maxRecDepth 4096 in
theorem outerScan156 : scan edge (search edge 4) 5 [] 0 tail156 winner = winner := by
  rw [tail156, scan, if_neg (by decide), root156Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail157
    (if 4 ≤ root156Candidates.length then search edge 4 [937] root156Candidates 937 winner else winner) = winner
  rw [if_pos (by decide), root156Branch]
  exact outerScan157
set_option maxRecDepth 4096 in
theorem outerScan155 : scan edge (search edge 4) 5 [] 0 tail155 winner = winner := by
  rw [tail155, scan, if_neg (by decide), root155Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail156
    (if 4 ≤ root155Candidates.length then search edge 4 [929] root155Candidates 929 winner else winner) = winner
  rw [if_pos (by decide), root155Branch]
  exact outerScan156
set_option maxRecDepth 4096 in
theorem outerScan154 : scan edge (search edge 4) 5 [] 0 tail154 winner = winner := by
  rw [tail154, scan, if_neg (by decide), root154Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail155
    (if 4 ≤ root154Candidates.length then search edge 4 [919] root154Candidates 919 winner else winner) = winner
  rw [if_pos (by decide), root154Branch]
  exact outerScan155
end ProjectEulerSolutions.PrimePairLarge.Search
