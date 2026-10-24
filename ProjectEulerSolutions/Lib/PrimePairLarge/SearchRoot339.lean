import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root339Candidates : List Nat := Data.row2297
set_option maxRecDepth 8192 in
theorem root339Candidates_checked : candidates edge 5 0 2297 tail340 winner.bound = root339Candidates := by
  apply candidates_eq_row edge 5 0 2297 tail340 root339Candidates winner.bound tail340_ordered
  · decide
  · decide
  · funext q
    rfl
def root339Tail31 : List Nat := []
def root339Tail30 : List Nat := 23633 :: root339Tail31
def root339Tail29 : List Nat := 23603 :: root339Tail30
def root339Tail28 : List Nat := 23447 :: root339Tail29
def root339Tail27 : List Nat := 20639 :: root339Tail28
def root339Tail26 : List Nat := 20549 :: root339Tail27
def root339Tail25 : List Nat := 19739 :: root339Tail26
def root339Tail24 : List Nat := 18257 :: root339Tail25
def root339Tail23 : List Nat := 16649 :: root339Tail24
def root339Tail22 : List Nat := 15287 :: root339Tail23
def root339Tail21 : List Nat := 14153 :: root339Tail22
def root339Tail20 : List Nat := 12611 :: root339Tail21
def root339Tail19 : List Nat := 12491 :: root339Tail20
def root339Tail18 : List Nat := 12107 :: root339Tail19
def root339Tail17 : List Nat := 11927 :: root339Tail18
def root339Tail16 : List Nat := 11243 :: root339Tail17
def root339Tail15 : List Nat := 11171 :: root339Tail16
def root339Tail14 : List Nat := 10781 :: root339Tail15
def root339Tail13 : List Nat := 10457 :: root339Tail14
def root339Tail12 : List Nat := 10037 :: root339Tail13
def root339Tail11 : List Nat := 9293 :: root339Tail12
def root339Tail10 : List Nat := 8807 :: root339Tail11
def root339Tail9 : List Nat := 7937 :: root339Tail10
def root339Tail8 : List Nat := 7127 :: root339Tail9
def root339Tail7 : List Nat := 6269 :: root339Tail8
def root339Tail6 : List Nat := 6089 :: root339Tail7
def root339Tail5 : List Nat := 5081 :: root339Tail6
def root339Tail4 : List Nat := 4967 :: root339Tail5
def root339Tail3 : List Nat := 4073 :: root339Tail4
def root339Tail2 : List Nat := 3581 :: root339Tail3
def root339Tail1 : List Nat := 2531 :: root339Tail2
def root339Tail0 : List Nat := 2381 :: root339Tail1
def root339Candidates0 : List Nat := [4073, 8807, 10781]
set_option maxRecDepth 8192 in
theorem root339Candidates0_checked : candidates edge 4 2297 2381 root339Tail1 winner.bound = root339Candidates0 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root339Child0 : search edge 3 [2297, 2381] root339Candidates0 4678 winner = winner := by decide
theorem root339Step0 : scan edge (search edge 3) 4 [2297] 2297 root339Tail0 winner =
    scan edge (search edge 3) 4 [2297] 2297 root339Tail1 winner := by
  rw [root339Tail0, scan, if_neg (by decide), root339Candidates0_checked]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail1
    (if 3 ≤ root339Candidates0.length then search edge 3 [2297, 2381] root339Candidates0 4678 winner else winner) = _
  rw [if_pos (by decide), root339Child0]
def root339Candidates1 : List Nat := [4967, 9293, 12611, 16649]
set_option maxRecDepth 8192 in
theorem root339Candidates1_checked : candidates edge 4 2297 2531 root339Tail2 winner.bound = root339Candidates1 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root339Child1 : search edge 3 [2297, 2531] root339Candidates1 4828 winner = winner := by decide
theorem root339Step1 : scan edge (search edge 3) 4 [2297] 2297 root339Tail1 winner =
    scan edge (search edge 3) 4 [2297] 2297 root339Tail2 winner := by
  rw [root339Tail1, scan, if_neg (by decide), root339Candidates1_checked]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail2
    (if 3 ≤ root339Candidates1.length then search edge 3 [2297, 2531] root339Candidates1 4828 winner else winner) = _
  rw [if_pos (by decide), root339Child1]
def root339Candidates2 : List Nat := [8807]
set_option maxRecDepth 8192 in
theorem root339Candidates2_checked : candidates edge 4 2297 3581 root339Tail3 winner.bound = root339Candidates2 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root339Step2 : scan edge (search edge 3) 4 [2297] 2297 root339Tail2 winner =
    scan edge (search edge 3) 4 [2297] 2297 root339Tail3 winner := by
  rw [root339Tail2, scan, if_neg (by decide), root339Candidates2_checked]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail3
    (if 3 ≤ root339Candidates2.length then search edge 3 [2297, 3581] root339Candidates2 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root339Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root339Candidates3_checked : candidates edge 4 2297 4073 root339Tail4 winner.bound = root339Candidates3 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root339Step3 : scan edge (search edge 3) 4 [2297] 2297 root339Tail3 winner =
    scan edge (search edge 3) 4 [2297] 2297 root339Tail4 winner := by
  rw [root339Tail3, scan, if_neg (by decide), root339Candidates3_checked]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail4
    (if 3 ≤ root339Candidates3.length then search edge 3 [2297, 4073] root339Candidates3 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root339Candidates4 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root339Candidates4_checked : candidates edge 4 2297 4967 root339Tail5 winner.bound = root339Candidates4 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root339Step4 : scan edge (search edge 3) 4 [2297] 2297 root339Tail4 winner =
    scan edge (search edge 3) 4 [2297] 2297 root339Tail5 winner := by
  rw [root339Tail4, scan, if_neg (by decide), root339Candidates4_checked]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail5
    (if 3 ≤ root339Candidates4.length then search edge 3 [2297, 4967] root339Candidates4 7264 winner else winner) = _
  rw [if_neg (by decide)]
def root339Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root339Candidates5_checked : candidates edge 4 2297 5081 root339Tail6 winner.bound = root339Candidates5 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root339Step5 : scan edge (search edge 3) 4 [2297] 2297 root339Tail5 winner =
    scan edge (search edge 3) 4 [2297] 2297 root339Tail6 winner := by
  rw [root339Tail5, scan, if_neg (by decide), root339Candidates5_checked]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail6
    (if 3 ≤ root339Candidates5.length then search edge 3 [2297, 5081] root339Candidates5 7378 winner else winner) = _
  rw [if_neg (by decide)]
theorem root339Scan6 : scan edge (search edge 3) 4 [2297] 2297 root339Tail6 winner = winner := by rw [root339Tail6, scan, if_pos (by decide)]
theorem root339Scan5 : scan edge (search edge 3) 4 [2297] 2297 root339Tail5 winner = winner :=
  (root339Step5).trans root339Scan6
theorem root339Scan4 : scan edge (search edge 3) 4 [2297] 2297 root339Tail4 winner = winner :=
  (root339Step4).trans root339Scan5
theorem root339Scan3 : scan edge (search edge 3) 4 [2297] 2297 root339Tail3 winner = winner :=
  (root339Step3).trans root339Scan4
theorem root339Scan2 : scan edge (search edge 3) 4 [2297] 2297 root339Tail2 winner = winner :=
  (root339Step2).trans root339Scan3
theorem root339Scan1 : scan edge (search edge 3) 4 [2297] 2297 root339Tail1 winner = winner :=
  (root339Step1).trans root339Scan2
theorem root339Scan0 : scan edge (search edge 3) 4 [2297] 2297 root339Tail0 winner = winner :=
  (root339Step0).trans root339Scan1
set_option maxRecDepth 4096 in
theorem root339Branch : search edge 4 [2297] root339Candidates 2297 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2297] 2297 root339Tail0 winner = winner
  exact root339Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
