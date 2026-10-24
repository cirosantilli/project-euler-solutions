import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root378Candidates : List Nat := Data.row2621
set_option maxRecDepth 8192 in
theorem root378Candidates_checked : candidates edge 5 0 2621 tail379 winner.bound = root378Candidates := by
  apply candidates_eq_row edge 5 0 2621 tail379 root378Candidates winner.bound tail379_ordered
  · decide
  · decide
  · funext q
    rfl
def root378Tail43 : List Nat := []
def root378Tail42 : List Nat := 23201 :: root378Tail43
def root378Tail41 : List Nat := 23117 :: root378Tail42
def root378Tail40 : List Nat := 22739 :: root378Tail41
def root378Tail39 : List Nat := 22397 :: root378Tail40
def root378Tail38 : List Nat := 21803 :: root378Tail39
def root378Tail37 : List Nat := 19469 :: root378Tail38
def root378Tail36 : List Nat := 19463 :: root378Tail37
def root378Tail35 : List Nat := 19373 :: root378Tail36
def root378Tail34 : List Nat := 18413 :: root378Tail35
def root378Tail33 : List Nat := 17987 :: root378Tail34
def root378Tail32 : List Nat := 16301 :: root378Tail33
def root378Tail31 : List Nat := 16097 :: root378Tail32
def root378Tail30 : List Nat := 15647 :: root378Tail31
def root378Tail29 : List Nat := 15629 :: root378Tail30
def root378Tail28 : List Nat := 15467 :: root378Tail29
def root378Tail27 : List Nat := 14081 :: root378Tail28
def root378Tail26 : List Nat := 13781 :: root378Tail27
def root378Tail25 : List Nat := 13757 :: root378Tail26
def root378Tail24 : List Nat := 13613 :: root378Tail25
def root378Tail23 : List Nat := 13457 :: root378Tail24
def root378Tail22 : List Nat := 13421 :: root378Tail23
def root378Tail21 : List Nat := 13367 :: root378Tail22
def root378Tail20 : List Nat := 13121 :: root378Tail21
def root378Tail19 : List Nat := 12569 :: root378Tail20
def root378Tail18 : List Nat := 12203 :: root378Tail19
def root378Tail17 : List Nat := 11813 :: root378Tail18
def root378Tail16 : List Nat := 11699 :: root378Tail17
def root378Tail15 : List Nat := 8861 :: root378Tail16
def root378Tail14 : List Nat := 8663 :: root378Tail15
def root378Tail13 : List Nat := 8363 :: root378Tail14
def root378Tail12 : List Nat := 7817 :: root378Tail13
def root378Tail11 : List Nat := 7151 :: root378Tail12
def root378Tail10 : List Nat := 6977 :: root378Tail11
def root378Tail9 : List Nat := 6521 :: root378Tail10
def root378Tail8 : List Nat := 6317 :: root378Tail9
def root378Tail7 : List Nat := 6263 :: root378Tail8
def root378Tail6 : List Nat := 6101 :: root378Tail7
def root378Tail5 : List Nat := 6053 :: root378Tail6
def root378Tail4 : List Nat := 5477 :: root378Tail5
def root378Tail3 : List Nat := 4973 :: root378Tail4
def root378Tail2 : List Nat := 4583 :: root378Tail3
def root378Tail1 : List Nat := 3623 :: root378Tail2
def root378Tail0 : List Nat := 2789 :: root378Tail1
def root378Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root378Candidates0_checked : candidates edge 4 2621 2789 root378Tail1 winner.bound = root378Candidates0 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root378Step0 : scan edge (search edge 3) 4 [2621] 2621 root378Tail0 winner =
    scan edge (search edge 3) 4 [2621] 2621 root378Tail1 winner := by
  rw [root378Tail0, scan, if_neg (by decide), root378Candidates0_checked]
  change scan edge (search edge 3) 4 [2621] 2621 root378Tail1
    (if 3 ≤ root378Candidates0.length then search edge 3 [2621, 2789] root378Candidates0 5410 winner else winner) = _
  rw [if_neg (by decide)]
def root378Candidates1 : List Nat := [6521, 18413, 19469]
set_option maxRecDepth 8192 in
theorem root378Candidates1_checked : candidates edge 4 2621 3623 root378Tail2 winner.bound = root378Candidates1 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root378Child1 : search edge 3 [2621, 3623] root378Candidates1 6244 winner = winner := by decide
theorem root378Step1 : scan edge (search edge 3) 4 [2621] 2621 root378Tail1 winner =
    scan edge (search edge 3) 4 [2621] 2621 root378Tail2 winner := by
  rw [root378Tail1, scan, if_neg (by decide), root378Candidates1_checked]
  change scan edge (search edge 3) 4 [2621] 2621 root378Tail2
    (if 3 ≤ root378Candidates1.length then search edge 3 [2621, 3623] root378Candidates1 6244 winner else winner) = _
  rw [if_pos (by decide), root378Child1]
def root378Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root378Candidates2_checked : candidates edge 4 2621 4583 root378Tail3 winner.bound = root378Candidates2 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root378Step2 : scan edge (search edge 3) 4 [2621] 2621 root378Tail2 winner =
    scan edge (search edge 3) 4 [2621] 2621 root378Tail3 winner := by
  rw [root378Tail2, scan, if_neg (by decide), root378Candidates2_checked]
  change scan edge (search edge 3) 4 [2621] 2621 root378Tail3
    (if 3 ≤ root378Candidates2.length then search edge 3 [2621, 4583] root378Candidates2 7204 winner else winner) = _
  rw [if_neg (by decide)]
def root378Candidates3 : List Nat := [11699, 13757, 15629]
set_option maxRecDepth 8192 in
theorem root378Candidates3_checked : candidates edge 4 2621 4973 root378Tail4 winner.bound = root378Candidates3 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root378Child3 : search edge 3 [2621, 4973] root378Candidates3 7594 winner = winner := by decide
theorem root378Step3 : scan edge (search edge 3) 4 [2621] 2621 root378Tail3 winner =
    scan edge (search edge 3) 4 [2621] 2621 root378Tail4 winner := by
  rw [root378Tail3, scan, if_neg (by decide), root378Candidates3_checked]
  change scan edge (search edge 3) 4 [2621] 2621 root378Tail4
    (if 3 ≤ root378Candidates3.length then search edge 3 [2621, 4973] root378Candidates3 7594 winner else winner) = _
  rw [if_pos (by decide), root378Child3]
def root378Candidates4 : List Nat := [6053, 13457]
set_option maxRecDepth 8192 in
theorem root378Candidates4_checked : candidates edge 4 2621 5477 root378Tail5 winner.bound = root378Candidates4 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root378Step4 : scan edge (search edge 3) 4 [2621] 2621 root378Tail4 winner =
    scan edge (search edge 3) 4 [2621] 2621 root378Tail5 winner := by
  rw [root378Tail4, scan, if_neg (by decide), root378Candidates4_checked]
  change scan edge (search edge 3) 4 [2621] 2621 root378Tail5
    (if 3 ≤ root378Candidates4.length then search edge 3 [2621, 5477] root378Candidates4 8098 winner else winner) = _
  rw [if_neg (by decide)]
theorem root378Scan5 : scan edge (search edge 3) 4 [2621] 2621 root378Tail5 winner = winner := by rw [root378Tail5, scan, if_pos (by decide)]
theorem root378Scan4 : scan edge (search edge 3) 4 [2621] 2621 root378Tail4 winner = winner :=
  (root378Step4).trans root378Scan5
theorem root378Scan3 : scan edge (search edge 3) 4 [2621] 2621 root378Tail3 winner = winner :=
  (root378Step3).trans root378Scan4
theorem root378Scan2 : scan edge (search edge 3) 4 [2621] 2621 root378Tail2 winner = winner :=
  (root378Step2).trans root378Scan3
theorem root378Scan1 : scan edge (search edge 3) 4 [2621] 2621 root378Tail1 winner = winner :=
  (root378Step1).trans root378Scan2
theorem root378Scan0 : scan edge (search edge 3) 4 [2621] 2621 root378Tail0 winner = winner :=
  (root378Step0).trans root378Scan1
set_option maxRecDepth 4096 in
theorem root378Branch : search edge 4 [2621] root378Candidates 2621 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2621] 2621 root378Tail0 winner = winner
  exact root378Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
