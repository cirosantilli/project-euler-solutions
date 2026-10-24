import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root579Candidates : List Nat := Data.row4243
set_option maxRecDepth 8192 in
theorem root579Candidates_checked : candidates edge 5 0 4243 tail580 winner.bound = root579Candidates := by
  apply candidates_eq_row edge 5 0 4243 tail580 root579Candidates winner.bound tail580_ordered
  · decide
  · decide
  · funext q
    rfl
def root579Tail39 : List Nat := []
def root579Tail38 : List Nat := 20857 :: root579Tail39
def root579Tail37 : List Nat := 20707 :: root579Tail38
def root579Tail36 : List Nat := 19867 :: root579Tail37
def root579Tail35 : List Nat := 19309 :: root579Tail36
def root579Tail34 : List Nat := 19081 :: root579Tail35
def root579Tail33 : List Nat := 19009 :: root579Tail34
def root579Tail32 : List Nat := 18913 :: root579Tail33
def root579Tail31 : List Nat := 17209 :: root579Tail32
def root579Tail30 : List Nat := 16267 :: root579Tail31
def root579Tail29 : List Nat := 15361 :: root579Tail30
def root579Tail28 : List Nat := 14797 :: root579Tail29
def root579Tail27 : List Nat := 14461 :: root579Tail28
def root579Tail26 : List Nat := 13807 :: root579Tail27
def root579Tail25 : List Nat := 13063 :: root579Tail26
def root579Tail24 : List Nat := 12613 :: root579Tail25
def root579Tail23 : List Nat := 12457 :: root579Tail24
def root579Tail22 : List Nat := 12391 :: root579Tail23
def root579Tail21 : List Nat := 11491 :: root579Tail22
def root579Tail20 : List Nat := 11299 :: root579Tail21
def root579Tail19 : List Nat := 11197 :: root579Tail20
def root579Tail18 : List Nat := 11119 :: root579Tail19
def root579Tail17 : List Nat := 10723 :: root579Tail18
def root579Tail16 : List Nat := 10657 :: root579Tail17
def root579Tail15 : List Nat := 8581 :: root579Tail16
def root579Tail14 : List Nat := 7699 :: root579Tail15
def root579Tail13 : List Nat := 7669 :: root579Tail14
def root579Tail12 : List Nat := 7207 :: root579Tail13
def root579Tail11 : List Nat := 6997 :: root579Tail12
def root579Tail10 : List Nat := 6553 :: root579Tail11
def root579Tail9 : List Nat := 6469 :: root579Tail10
def root579Tail8 : List Nat := 6451 :: root579Tail9
def root579Tail7 : List Nat := 5527 :: root579Tail8
def root579Tail6 : List Nat := 5347 :: root579Tail7
def root579Tail5 : List Nat := 4831 :: root579Tail6
def root579Tail4 : List Nat := 4759 :: root579Tail5
def root579Tail3 : List Nat := 4621 :: root579Tail4
def root579Tail2 : List Nat := 4591 :: root579Tail3
def root579Tail1 : List Nat := 4423 :: root579Tail2
def root579Tail0 : List Nat := 4339 :: root579Tail1
def root579Candidates0 : List Nat := [4831]
set_option maxRecDepth 8192 in
theorem root579Candidates0_checked : candidates edge 4 4243 4339 root579Tail1 winner.bound = root579Candidates0 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root579Step0 : scan edge (search edge 3) 4 [4243] 4243 root579Tail0 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail1 winner := by
  rw [root579Tail0, scan, if_neg (by decide), root579Candidates0_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail1
    (if 3 ≤ root579Candidates0.length then search edge 3 [4243, 4339] root579Candidates0 8582 winner else winner) = _
  rw [if_neg (by decide)]
def root579Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root579Candidates1_checked : candidates edge 4 4243 4423 root579Tail2 winner.bound = root579Candidates1 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root579Step1 : scan edge (search edge 3) 4 [4243] 4243 root579Tail1 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail2 winner := by
  rw [root579Tail1, scan, if_neg (by decide), root579Candidates1_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail2
    (if 3 ≤ root579Candidates1.length then search edge 3 [4243, 4423] root579Candidates1 8666 winner else winner) = _
  rw [if_neg (by decide)]
def root579Candidates2 : List Nat := [14797]
set_option maxRecDepth 8192 in
theorem root579Candidates2_checked : candidates edge 4 4243 4591 root579Tail3 winner.bound = root579Candidates2 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root579Step2 : scan edge (search edge 3) 4 [4243] 4243 root579Tail2 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail3 winner := by
  rw [root579Tail2, scan, if_neg (by decide), root579Candidates2_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail3
    (if 3 ≤ root579Candidates2.length then search edge 3 [4243, 4591] root579Candidates2 8834 winner else winner) = _
  rw [if_neg (by decide)]
def root579Candidates3 : List Nat := [5347]
set_option maxRecDepth 8192 in
theorem root579Candidates3_checked : candidates edge 4 4243 4621 root579Tail4 winner.bound = root579Candidates3 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root579Step3 : scan edge (search edge 3) 4 [4243] 4243 root579Tail3 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail4 winner := by
  rw [root579Tail3, scan, if_neg (by decide), root579Candidates3_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail4
    (if 3 ≤ root579Candidates3.length then search edge 3 [4243, 4621] root579Candidates3 8864 winner else winner) = _
  rw [if_neg (by decide)]
def root579Candidates4 : List Nat := [14797]
set_option maxRecDepth 8192 in
theorem root579Candidates4_checked : candidates edge 4 4243 4759 root579Tail5 winner.bound = root579Candidates4 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root579Step4 : scan edge (search edge 3) 4 [4243] 4243 root579Tail4 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail5 winner := by
  rw [root579Tail4, scan, if_neg (by decide), root579Candidates4_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail5
    (if 3 ≤ root579Candidates4.length then search edge 3 [4243, 4759] root579Candidates4 9002 winner else winner) = _
  rw [if_neg (by decide)]
def root579Candidates5 : List Nat := [6451, 6553, 7699]
set_option maxRecDepth 8192 in
theorem root579Candidates5_checked : candidates edge 4 4243 4831 root579Tail6 winner.bound = root579Candidates5 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root579Child5 : search edge 3 [4243, 4831] root579Candidates5 9074 winner = winner := by decide
theorem root579Step5 : scan edge (search edge 3) 4 [4243] 4243 root579Tail5 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail6 winner := by
  rw [root579Tail5, scan, if_neg (by decide), root579Candidates5_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail6
    (if 3 ≤ root579Candidates5.length then search edge 3 [4243, 4831] root579Candidates5 9074 winner else winner) = _
  rw [if_pos (by decide), root579Child5]
def root579Candidates6 : List Nat := [11119]
set_option maxRecDepth 8192 in
theorem root579Candidates6_checked : candidates edge 4 4243 5347 root579Tail7 winner.bound = root579Candidates6 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root579Step6 : scan edge (search edge 3) 4 [4243] 4243 root579Tail6 winner =
    scan edge (search edge 3) 4 [4243] 4243 root579Tail7 winner := by
  rw [root579Tail6, scan, if_neg (by decide), root579Candidates6_checked]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail7
    (if 3 ≤ root579Candidates6.length then search edge 3 [4243, 5347] root579Candidates6 9590 winner else winner) = _
  rw [if_neg (by decide)]
theorem root579Scan7 : scan edge (search edge 3) 4 [4243] 4243 root579Tail7 winner = winner := by rw [root579Tail7, scan, if_pos (by decide)]
theorem root579Scan6 : scan edge (search edge 3) 4 [4243] 4243 root579Tail6 winner = winner :=
  (root579Step6).trans root579Scan7
theorem root579Scan5 : scan edge (search edge 3) 4 [4243] 4243 root579Tail5 winner = winner :=
  (root579Step5).trans root579Scan6
theorem root579Scan4 : scan edge (search edge 3) 4 [4243] 4243 root579Tail4 winner = winner :=
  (root579Step4).trans root579Scan5
theorem root579Scan3 : scan edge (search edge 3) 4 [4243] 4243 root579Tail3 winner = winner :=
  (root579Step3).trans root579Scan4
theorem root579Scan2 : scan edge (search edge 3) 4 [4243] 4243 root579Tail2 winner = winner :=
  (root579Step2).trans root579Scan3
theorem root579Scan1 : scan edge (search edge 3) 4 [4243] 4243 root579Tail1 winner = winner :=
  (root579Step1).trans root579Scan2
theorem root579Scan0 : scan edge (search edge 3) 4 [4243] 4243 root579Tail0 winner = winner :=
  (root579Step0).trans root579Scan1
set_option maxRecDepth 4096 in
theorem root579Branch : search edge 4 [4243] root579Candidates 4243 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4243] 4243 root579Tail0 winner = winner
  exact root579Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
