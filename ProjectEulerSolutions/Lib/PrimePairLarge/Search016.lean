import ProjectEulerSolutions.Lib.PrimePairLarge.Search015
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot554
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot555
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot556
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot557
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot558
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot559
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot560
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot561
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan561 : scan edge (search edge 4) 5 [] 0 tail561 winner = winner := by
  rw [tail561, scan, if_neg (by decide), root561Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail562
    (if 4 ≤ root561Candidates.length then search edge 4 [4093] root561Candidates 4093 winner else winner) = winner
  rw [if_pos (by decide), root561Branch]
  exact outerScan562
set_option maxRecDepth 4096 in
theorem outerScan560 : scan edge (search edge 4) 5 [] 0 tail560 winner = winner := by
  rw [tail560, scan, if_neg (by decide), root560Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail561
    (if 4 ≤ root560Candidates.length then search edge 4 [4091] root560Candidates 4091 winner else winner) = winner
  rw [if_pos (by decide), root560Branch]
  exact outerScan561
set_option maxRecDepth 4096 in
theorem outerScan559 : scan edge (search edge 4) 5 [] 0 tail559 winner = winner := by
  rw [tail559, scan, if_neg (by decide), root559Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail560
    (if 4 ≤ root559Candidates.length then search edge 4 [4079] root559Candidates 4079 winner else winner) = winner
  rw [if_pos (by decide), root559Branch]
  exact outerScan560
set_option maxRecDepth 4096 in
theorem outerScan558 : scan edge (search edge 4) 5 [] 0 tail558 winner = winner := by
  rw [tail558, scan, if_neg (by decide), root558Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail559
    (if 4 ≤ root558Candidates.length then search edge 4 [4073] root558Candidates 4073 winner else winner) = winner
  rw [if_pos (by decide), root558Branch]
  exact outerScan559
set_option maxRecDepth 4096 in
theorem outerScan557 : scan edge (search edge 4) 5 [] 0 tail557 winner = winner := by
  rw [tail557, scan, if_neg (by decide), root557Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail558
    (if 4 ≤ root557Candidates.length then search edge 4 [4057] root557Candidates 4057 winner else winner) = winner
  rw [if_pos (by decide), root557Branch]
  exact outerScan558
set_option maxRecDepth 4096 in
theorem outerScan556 : scan edge (search edge 4) 5 [] 0 tail556 winner = winner := by
  rw [tail556, scan, if_neg (by decide), root556Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail557
    (if 4 ≤ root556Candidates.length then search edge 4 [4051] root556Candidates 4051 winner else winner) = winner
  rw [if_pos (by decide), root556Branch]
  exact outerScan557
set_option maxRecDepth 4096 in
theorem outerScan555 : scan edge (search edge 4) 5 [] 0 tail555 winner = winner := by
  rw [tail555, scan, if_neg (by decide), root555Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail556
    (if 4 ≤ root555Candidates.length then search edge 4 [4049] root555Candidates 4049 winner else winner) = winner
  rw [if_pos (by decide), root555Branch]
  exact outerScan556
set_option maxRecDepth 4096 in
theorem outerScan554 : scan edge (search edge 4) 5 [] 0 tail554 winner = winner := by
  rw [tail554, scan, if_neg (by decide), root554Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail555
    (if 4 ≤ root554Candidates.length then search edge 4 [4027] root554Candidates 4027 winner else winner) = winner
  rw [if_pos (by decide), root554Branch]
  exact outerScan555
end ProjectEulerSolutions.PrimePairLarge.Search
