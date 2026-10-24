import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root404Candidates : List Nat := Data.row2797
set_option maxRecDepth 8192 in
theorem root404Candidates_checked : candidates edge 5 0 2797 tail405 winner.bound = root404Candidates := by
  apply candidates_eq_row edge 5 0 2797 tail405 root404Candidates winner.bound tail405_ordered
  · decide
  · decide
  · funext q
    rfl
def root404Tail42 : List Nat := []
def root404Tail41 : List Nat := 23209 :: root404Tail42
def root404Tail40 : List Nat := 23167 :: root404Tail41
def root404Tail39 : List Nat := 22447 :: root404Tail40
def root404Tail38 : List Nat := 22279 :: root404Tail39
def root404Tail37 : List Nat := 22171 :: root404Tail38
def root404Tail36 : List Nat := 21613 :: root404Tail37
def root404Tail35 : List Nat := 21397 :: root404Tail36
def root404Tail34 : List Nat := 21157 :: root404Tail35
def root404Tail33 : List Nat := 19051 :: root404Tail34
def root404Tail32 : List Nat := 19009 :: root404Tail33
def root404Tail31 : List Nat := 18979 :: root404Tail32
def root404Tail30 : List Nat := 18973 :: root404Tail31
def root404Tail29 : List Nat := 17959 :: root404Tail30
def root404Tail28 : List Nat := 17497 :: root404Tail29
def root404Tail27 : List Nat := 17419 :: root404Tail28
def root404Tail26 : List Nat := 15493 :: root404Tail27
def root404Tail25 : List Nat := 13963 :: root404Tail26
def root404Tail24 : List Nat := 13933 :: root404Tail25
def root404Tail23 : List Nat := 13249 :: root404Tail24
def root404Tail22 : List Nat := 12967 :: root404Tail23
def root404Tail21 : List Nat := 12583 :: root404Tail22
def root404Tail20 : List Nat := 10867 :: root404Tail21
def root404Tail19 : List Nat := 10111 :: root404Tail20
def root404Tail18 : List Nat := 10099 :: root404Tail19
def root404Tail17 : List Nat := 9871 :: root404Tail18
def root404Tail16 : List Nat := 9643 :: root404Tail17
def root404Tail15 : List Nat := 9181 :: root404Tail16
def root404Tail14 : List Nat := 8563 :: root404Tail15
def root404Tail13 : List Nat := 8017 :: root404Tail14
def root404Tail12 : List Nat := 7759 :: root404Tail13
def root404Tail11 : List Nat := 7681 :: root404Tail12
def root404Tail10 : List Nat := 7207 :: root404Tail11
def root404Tail9 : List Nat := 6841 :: root404Tail10
def root404Tail8 : List Nat := 6367 :: root404Tail9
def root404Tail7 : List Nat := 6163 :: root404Tail8
def root404Tail6 : List Nat := 5953 :: root404Tail7
def root404Tail5 : List Nat := 5449 :: root404Tail6
def root404Tail4 : List Nat := 5023 :: root404Tail5
def root404Tail3 : List Nat := 4861 :: root404Tail4
def root404Tail2 : List Nat := 4663 :: root404Tail3
def root404Tail1 : List Nat := 4567 :: root404Tail2
def root404Tail0 : List Nat := 4561 :: root404Tail1
def root404Candidates0 : List Nat := [8017, 12967, 13249]
set_option maxRecDepth 8192 in
theorem root404Candidates0_checked : candidates edge 4 2797 4561 root404Tail1 winner.bound = root404Candidates0 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root404Child0 : search edge 3 [2797, 4561] root404Candidates0 7358 winner = winner := by decide
theorem root404Step0 : scan edge (search edge 3) 4 [2797] 2797 root404Tail0 winner =
    scan edge (search edge 3) 4 [2797] 2797 root404Tail1 winner := by
  rw [root404Tail0, scan, if_neg (by decide), root404Candidates0_checked]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail1
    (if 3 ≤ root404Candidates0.length then search edge 3 [2797, 4561] root404Candidates0 7358 winner else winner) = _
  rw [if_pos (by decide), root404Child0]
def root404Candidates1 : List Nat := [9643]
set_option maxRecDepth 8192 in
theorem root404Candidates1_checked : candidates edge 4 2797 4567 root404Tail2 winner.bound = root404Candidates1 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root404Step1 : scan edge (search edge 3) 4 [2797] 2797 root404Tail1 winner =
    scan edge (search edge 3) 4 [2797] 2797 root404Tail2 winner := by
  rw [root404Tail1, scan, if_neg (by decide), root404Candidates1_checked]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail2
    (if 3 ≤ root404Candidates1.length then search edge 3 [2797, 4567] root404Candidates1 7364 winner else winner) = _
  rw [if_neg (by decide)]
def root404Candidates2 : List Nat := [8017]
set_option maxRecDepth 8192 in
theorem root404Candidates2_checked : candidates edge 4 2797 4663 root404Tail3 winner.bound = root404Candidates2 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root404Step2 : scan edge (search edge 3) 4 [2797] 2797 root404Tail2 winner =
    scan edge (search edge 3) 4 [2797] 2797 root404Tail3 winner := by
  rw [root404Tail2, scan, if_neg (by decide), root404Candidates2_checked]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail3
    (if 3 ≤ root404Candidates2.length then search edge 3 [2797, 4663] root404Candidates2 7460 winner else winner) = _
  rw [if_neg (by decide)]
def root404Candidates3 : List Nat := [5953, 6163]
set_option maxRecDepth 8192 in
theorem root404Candidates3_checked : candidates edge 4 2797 4861 root404Tail4 winner.bound = root404Candidates3 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root404Step3 : scan edge (search edge 3) 4 [2797] 2797 root404Tail3 winner =
    scan edge (search edge 3) 4 [2797] 2797 root404Tail4 winner := by
  rw [root404Tail3, scan, if_neg (by decide), root404Candidates3_checked]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail4
    (if 3 ≤ root404Candidates3.length then search edge 3 [2797, 4861] root404Candidates3 7658 winner else winner) = _
  rw [if_neg (by decide)]
def root404Candidates4 : List Nat := [6841]
set_option maxRecDepth 8192 in
theorem root404Candidates4_checked : candidates edge 4 2797 5023 root404Tail5 winner.bound = root404Candidates4 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root404Step4 : scan edge (search edge 3) 4 [2797] 2797 root404Tail4 winner =
    scan edge (search edge 3) 4 [2797] 2797 root404Tail5 winner := by
  rw [root404Tail4, scan, if_neg (by decide), root404Candidates4_checked]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail5
    (if 3 ≤ root404Candidates4.length then search edge 3 [2797, 5023] root404Candidates4 7820 winner else winner) = _
  rw [if_neg (by decide)]
def root404Candidates5 : List Nat := [13249]
set_option maxRecDepth 8192 in
theorem root404Candidates5_checked : candidates edge 4 2797 5449 root404Tail6 winner.bound = root404Candidates5 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root404Step5 : scan edge (search edge 3) 4 [2797] 2797 root404Tail5 winner =
    scan edge (search edge 3) 4 [2797] 2797 root404Tail6 winner := by
  rw [root404Tail5, scan, if_neg (by decide), root404Candidates5_checked]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail6
    (if 3 ≤ root404Candidates5.length then search edge 3 [2797, 5449] root404Candidates5 8246 winner else winner) = _
  rw [if_neg (by decide)]
theorem root404Scan6 : scan edge (search edge 3) 4 [2797] 2797 root404Tail6 winner = winner := by rw [root404Tail6, scan, if_pos (by decide)]
theorem root404Scan5 : scan edge (search edge 3) 4 [2797] 2797 root404Tail5 winner = winner :=
  (root404Step5).trans root404Scan6
theorem root404Scan4 : scan edge (search edge 3) 4 [2797] 2797 root404Tail4 winner = winner :=
  (root404Step4).trans root404Scan5
theorem root404Scan3 : scan edge (search edge 3) 4 [2797] 2797 root404Tail3 winner = winner :=
  (root404Step3).trans root404Scan4
theorem root404Scan2 : scan edge (search edge 3) 4 [2797] 2797 root404Tail2 winner = winner :=
  (root404Step2).trans root404Scan3
theorem root404Scan1 : scan edge (search edge 3) 4 [2797] 2797 root404Tail1 winner = winner :=
  (root404Step1).trans root404Scan2
theorem root404Scan0 : scan edge (search edge 3) 4 [2797] 2797 root404Tail0 winner = winner :=
  (root404Step0).trans root404Scan1
set_option maxRecDepth 4096 in
theorem root404Branch : search edge 4 [2797] root404Candidates 2797 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2797] 2797 root404Tail0 winner = winner
  exact root404Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
