import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root234Candidates : List Nat := Data.row1489
set_option maxRecDepth 8192 in
theorem root234Candidates_checked : candidates edge 5 0 1489 tail235 winner.bound = root234Candidates := by
  apply candidates_eq_row edge 5 0 1489 tail235 root234Candidates winner.bound tail235_ordered
  · decide
  · decide
  · funext q
    rfl
def root234Tail48 : List Nat := []
def root234Tail47 : List Nat := 24499 :: root234Tail48
def root234Tail46 : List Nat := 23743 :: root234Tail47
def root234Tail45 : List Nat := 23059 :: root234Tail46
def root234Tail44 : List Nat := 22483 :: root234Tail45
def root234Tail43 : List Nat := 21937 :: root234Tail44
def root234Tail42 : List Nat := 21379 :: root234Tail43
def root234Tail41 : List Nat := 21013 :: root234Tail42
def root234Tail40 : List Nat := 20809 :: root234Tail41
def root234Tail39 : List Nat := 20749 :: root234Tail40
def root234Tail38 : List Nat := 18493 :: root234Tail39
def root234Tail37 : List Nat := 18481 :: root234Tail38
def root234Tail36 : List Nat := 17491 :: root234Tail37
def root234Tail35 : List Nat := 17053 :: root234Tail36
def root234Tail34 : List Nat := 17029 :: root234Tail35
def root234Tail33 : List Nat := 16981 :: root234Tail34
def root234Tail32 : List Nat := 16693 :: root234Tail33
def root234Tail31 : List Nat := 16249 :: root234Tail32
def root234Tail30 : List Nat := 15889 :: root234Tail31
def root234Tail29 : List Nat := 15643 :: root234Tail30
def root234Tail28 : List Nat := 15307 :: root234Tail29
def root234Tail27 : List Nat := 14929 :: root234Tail28
def root234Tail26 : List Nat := 14149 :: root234Tail27
def root234Tail25 : List Nat := 13903 :: root234Tail26
def root234Tail24 : List Nat := 13879 :: root234Tail25
def root234Tail23 : List Nat := 12889 :: root234Tail24
def root234Tail22 : List Nat := 12601 :: root234Tail23
def root234Tail21 : List Nat := 12343 :: root234Tail22
def root234Tail20 : List Nat := 11527 :: root234Tail21
def root234Tail19 : List Nat := 11251 :: root234Tail20
def root234Tail18 : List Nat := 10627 :: root234Tail19
def root234Tail17 : List Nat := 9697 :: root234Tail18
def root234Tail16 : List Nat := 9613 :: root234Tail17
def root234Tail15 : List Nat := 9397 :: root234Tail16
def root234Tail14 : List Nat := 9187 :: root234Tail15
def root234Tail13 : List Nat := 9007 :: root234Tail14
def root234Tail12 : List Nat := 8629 :: root234Tail13
def root234Tail11 : List Nat := 8623 :: root234Tail12
def root234Tail10 : List Nat := 8563 :: root234Tail11
def root234Tail9 : List Nat := 8329 :: root234Tail10
def root234Tail8 : List Nat := 8263 :: root234Tail9
def root234Tail7 : List Nat := 6883 :: root234Tail8
def root234Tail6 : List Nat := 6547 :: root234Tail7
def root234Tail5 : List Nat := 5857 :: root234Tail6
def root234Tail4 : List Nat := 5557 :: root234Tail5
def root234Tail3 : List Nat := 5119 :: root234Tail4
def root234Tail2 : List Nat := 4801 :: root234Tail3
def root234Tail1 : List Nat := 4597 :: root234Tail2
def root234Tail0 : List Nat := 3691 :: root234Tail1
def root234Candidates0 : List Nat := [12343]
set_option maxRecDepth 8192 in
theorem root234Candidates0_checked : candidates edge 4 1489 3691 root234Tail1 winner.bound = root234Candidates0 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
theorem root234Step0 : scan edge (search edge 3) 4 [1489] 1489 root234Tail0 winner =
    scan edge (search edge 3) 4 [1489] 1489 root234Tail1 winner := by
  rw [root234Tail0, scan, if_neg (by decide), root234Candidates0_checked]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail1
    (if 3 ≤ root234Candidates0.length then search edge 3 [1489, 3691] root234Candidates0 5180 winner else winner) = _
  rw [if_neg (by decide)]
def root234Candidates1 : List Nat := [8263, 9397]
set_option maxRecDepth 8192 in
theorem root234Candidates1_checked : candidates edge 4 1489 4597 root234Tail2 winner.bound = root234Candidates1 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root234Step1 : scan edge (search edge 3) 4 [1489] 1489 root234Tail1 winner =
    scan edge (search edge 3) 4 [1489] 1489 root234Tail2 winner := by
  rw [root234Tail1, scan, if_neg (by decide), root234Candidates1_checked]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail2
    (if 3 ≤ root234Candidates1.length then search edge 3 [1489, 4597] root234Candidates1 6086 winner else winner) = _
  rw [if_neg (by decide)]
def root234Candidates2 : List Nat := [9397, 18481]
set_option maxRecDepth 8192 in
theorem root234Candidates2_checked : candidates edge 4 1489 4801 root234Tail3 winner.bound = root234Candidates2 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root234Step2 : scan edge (search edge 3) 4 [1489] 1489 root234Tail2 winner =
    scan edge (search edge 3) 4 [1489] 1489 root234Tail3 winner := by
  rw [root234Tail2, scan, if_neg (by decide), root234Candidates2_checked]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail3
    (if 3 ≤ root234Candidates2.length then search edge 3 [1489, 4801] root234Candidates2 6290 winner else winner) = _
  rw [if_neg (by decide)]
def root234Candidates3 : List Nat := [12601, 18481]
set_option maxRecDepth 8192 in
theorem root234Candidates3_checked : candidates edge 4 1489 5119 root234Tail4 winner.bound = root234Candidates3 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root234Step3 : scan edge (search edge 3) 4 [1489] 1489 root234Tail3 winner =
    scan edge (search edge 3) 4 [1489] 1489 root234Tail4 winner := by
  rw [root234Tail3, scan, if_neg (by decide), root234Candidates3_checked]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail4
    (if 3 ≤ root234Candidates3.length then search edge 3 [1489, 5119] root234Candidates3 6608 winner else winner) = _
  rw [if_neg (by decide)]
def root234Candidates4 : List Nat := [10627]
set_option maxRecDepth 8192 in
theorem root234Candidates4_checked : candidates edge 4 1489 5557 root234Tail5 winner.bound = root234Candidates4 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root234Step4 : scan edge (search edge 3) 4 [1489] 1489 root234Tail4 winner =
    scan edge (search edge 3) 4 [1489] 1489 root234Tail5 winner := by
  rw [root234Tail4, scan, if_neg (by decide), root234Candidates4_checked]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail5
    (if 3 ≤ root234Candidates4.length then search edge 3 [1489, 5557] root234Candidates4 7046 winner else winner) = _
  rw [if_neg (by decide)]
def root234Candidates5 : List Nat := [9007, 14149]
set_option maxRecDepth 8192 in
theorem root234Candidates5_checked : candidates edge 4 1489 5857 root234Tail6 winner.bound = root234Candidates5 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root234Step5 : scan edge (search edge 3) 4 [1489] 1489 root234Tail5 winner =
    scan edge (search edge 3) 4 [1489] 1489 root234Tail6 winner := by
  rw [root234Tail5, scan, if_neg (by decide), root234Candidates5_checked]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail6
    (if 3 ≤ root234Candidates5.length then search edge 3 [1489, 5857] root234Candidates5 7346 winner else winner) = _
  rw [if_neg (by decide)]
theorem root234Scan6 : scan edge (search edge 3) 4 [1489] 1489 root234Tail6 winner = winner := by rw [root234Tail6, scan, if_pos (by decide)]
theorem root234Scan5 : scan edge (search edge 3) 4 [1489] 1489 root234Tail5 winner = winner :=
  (root234Step5).trans root234Scan6
theorem root234Scan4 : scan edge (search edge 3) 4 [1489] 1489 root234Tail4 winner = winner :=
  (root234Step4).trans root234Scan5
theorem root234Scan3 : scan edge (search edge 3) 4 [1489] 1489 root234Tail3 winner = winner :=
  (root234Step3).trans root234Scan4
theorem root234Scan2 : scan edge (search edge 3) 4 [1489] 1489 root234Tail2 winner = winner :=
  (root234Step2).trans root234Scan3
theorem root234Scan1 : scan edge (search edge 3) 4 [1489] 1489 root234Tail1 winner = winner :=
  (root234Step1).trans root234Scan2
theorem root234Scan0 : scan edge (search edge 3) 4 [1489] 1489 root234Tail0 winner = winner :=
  (root234Step0).trans root234Scan1
set_option maxRecDepth 4096 in
theorem root234Branch : search edge 4 [1489] root234Candidates 1489 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1489] 1489 root234Tail0 winner = winner
  exact root234Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
