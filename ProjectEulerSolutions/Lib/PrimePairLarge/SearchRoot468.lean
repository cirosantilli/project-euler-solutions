import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root468Candidates : List Nat := Data.row3343
set_option maxRecDepth 8192 in
theorem root468Candidates_checked : candidates edge 5 0 3343 tail469 winner.bound = root468Candidates := by
  apply candidates_eq_row edge 5 0 3343 tail469 root468Candidates winner.bound tail469_ordered
  · decide
  · decide
  · funext q
    rfl
def root468Tail38 : List Nat := []
def root468Tail37 : List Nat := 20773 :: root468Tail38
def root468Tail36 : List Nat := 20533 :: root468Tail37
def root468Tail35 : List Nat := 20407 :: root468Tail36
def root468Tail34 : List Nat := 19891 :: root468Tail35
def root468Tail33 : List Nat := 19819 :: root468Tail34
def root468Tail32 : List Nat := 19477 :: root468Tail33
def root468Tail31 : List Nat := 17791 :: root468Tail32
def root468Tail30 : List Nat := 17551 :: root468Tail31
def root468Tail29 : List Nat := 17449 :: root468Tail30
def root468Tail28 : List Nat := 17341 :: root468Tail29
def root468Tail27 : List Nat := 16903 :: root468Tail28
def root468Tail26 : List Nat := 16693 :: root468Tail27
def root468Tail25 : List Nat := 16573 :: root468Tail26
def root468Tail24 : List Nat := 16273 :: root468Tail25
def root468Tail23 : List Nat := 15901 :: root468Tail24
def root468Tail22 : List Nat := 15601 :: root468Tail23
def root468Tail21 : List Nat := 15307 :: root468Tail22
def root468Tail20 : List Nat := 14767 :: root468Tail21
def root468Tail19 : List Nat := 14593 :: root468Tail20
def root468Tail18 : List Nat := 13477 :: root468Tail19
def root468Tail17 : List Nat := 9811 :: root468Tail18
def root468Tail16 : List Nat := 9397 :: root468Tail17
def root468Tail15 : List Nat := 8467 :: root468Tail16
def root468Tail14 : List Nat := 7687 :: root468Tail15
def root468Tail13 : List Nat := 6961 :: root468Tail14
def root468Tail12 : List Nat := 6883 :: root468Tail13
def root468Tail11 : List Nat := 6451 :: root468Tail12
def root468Tail10 : List Nat := 6379 :: root468Tail11
def root468Tail9 : List Nat := 6373 :: root468Tail10
def root468Tail8 : List Nat := 5791 :: root468Tail9
def root468Tail7 : List Nat := 5449 :: root468Tail8
def root468Tail6 : List Nat := 4789 :: root468Tail7
def root468Tail5 : List Nat := 4567 :: root468Tail6
def root468Tail4 : List Nat := 4561 :: root468Tail5
def root468Tail3 : List Nat := 4483 :: root468Tail4
def root468Tail2 : List Nat := 3823 :: root468Tail3
def root468Tail1 : List Nat := 3643 :: root468Tail2
def root468Tail0 : List Nat := 3391 :: root468Tail1
def root468Candidates0 : List Nat := [3643, 14767, 16903, 17341]
set_option maxRecDepth 8192 in
theorem root468Candidates0_checked : candidates edge 4 3343 3391 root468Tail1 winner.bound = root468Candidates0 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root468Child0 : search edge 3 [3343, 3391] root468Candidates0 6734 winner = winner := by decide
theorem root468Step0 : scan edge (search edge 3) 4 [3343] 3343 root468Tail0 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail1 winner := by
  rw [root468Tail0, scan, if_neg (by decide), root468Candidates0_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail1
    (if 3 ≤ root468Candidates0.length then search edge 3 [3343, 3391] root468Candidates0 6734 winner else winner) = _
  rw [if_pos (by decide), root468Child0]
def root468Candidates1 : List Nat := [4561, 5449]
set_option maxRecDepth 8192 in
theorem root468Candidates1_checked : candidates edge 4 3343 3643 root468Tail2 winner.bound = root468Candidates1 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root468Step1 : scan edge (search edge 3) 4 [3343] 3343 root468Tail1 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail2 winner := by
  rw [root468Tail1, scan, if_neg (by decide), root468Candidates1_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail2
    (if 3 ≤ root468Candidates1.length then search edge 3 [3343, 3643] root468Candidates1 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root468Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root468Candidates2_checked : candidates edge 4 3343 3823 root468Tail3 winner.bound = root468Candidates2 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root468Step2 : scan edge (search edge 3) 4 [3343] 3343 root468Tail2 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail3 winner := by
  rw [root468Tail2, scan, if_neg (by decide), root468Candidates2_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail3
    (if 3 ≤ root468Candidates2.length then search edge 3 [3343, 3823] root468Candidates2 7166 winner else winner) = _
  rw [if_neg (by decide)]
def root468Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root468Candidates3_checked : candidates edge 4 3343 4483 root468Tail4 winner.bound = root468Candidates3 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root468Step3 : scan edge (search edge 3) 4 [3343] 3343 root468Tail3 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail4 winner := by
  rw [root468Tail3, scan, if_neg (by decide), root468Candidates3_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail4
    (if 3 ≤ root468Candidates3.length then search edge 3 [3343, 4483] root468Candidates3 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root468Candidates4 : List Nat := [6379, 9811]
set_option maxRecDepth 8192 in
theorem root468Candidates4_checked : candidates edge 4 3343 4561 root468Tail5 winner.bound = root468Candidates4 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root468Step4 : scan edge (search edge 3) 4 [3343] 3343 root468Tail4 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail5 winner := by
  rw [root468Tail4, scan, if_neg (by decide), root468Candidates4_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail5
    (if 3 ≤ root468Candidates4.length then search edge 3 [3343, 4561] root468Candidates4 7904 winner else winner) = _
  rw [if_neg (by decide)]
def root468Candidates5 : List Nat := [8467]
set_option maxRecDepth 8192 in
theorem root468Candidates5_checked : candidates edge 4 3343 4567 root468Tail6 winner.bound = root468Candidates5 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root468Step5 : scan edge (search edge 3) 4 [3343] 3343 root468Tail5 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail6 winner := by
  rw [root468Tail5, scan, if_neg (by decide), root468Candidates5_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail6
    (if 3 ≤ root468Candidates5.length then search edge 3 [3343, 4567] root468Candidates5 7910 winner else winner) = _
  rw [if_neg (by decide)]
def root468Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root468Candidates6_checked : candidates edge 4 3343 4789 root468Tail7 winner.bound = root468Candidates6 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root468Step6 : scan edge (search edge 3) 4 [3343] 3343 root468Tail6 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail7 winner := by
  rw [root468Tail6, scan, if_neg (by decide), root468Candidates6_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail7
    (if 3 ≤ root468Candidates6.length then search edge 3 [3343, 4789] root468Candidates6 8132 winner else winner) = _
  rw [if_neg (by decide)]
def root468Candidates7 : List Nat := [15901]
set_option maxRecDepth 8192 in
theorem root468Candidates7_checked : candidates edge 4 3343 5449 root468Tail8 winner.bound = root468Candidates7 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root468Step7 : scan edge (search edge 3) 4 [3343] 3343 root468Tail7 winner =
    scan edge (search edge 3) 4 [3343] 3343 root468Tail8 winner := by
  rw [root468Tail7, scan, if_neg (by decide), root468Candidates7_checked]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail8
    (if 3 ≤ root468Candidates7.length then search edge 3 [3343, 5449] root468Candidates7 8792 winner else winner) = _
  rw [if_neg (by decide)]
theorem root468Scan8 : scan edge (search edge 3) 4 [3343] 3343 root468Tail8 winner = winner := by rw [root468Tail8, scan, if_pos (by decide)]
theorem root468Scan7 : scan edge (search edge 3) 4 [3343] 3343 root468Tail7 winner = winner :=
  (root468Step7).trans root468Scan8
theorem root468Scan6 : scan edge (search edge 3) 4 [3343] 3343 root468Tail6 winner = winner :=
  (root468Step6).trans root468Scan7
theorem root468Scan5 : scan edge (search edge 3) 4 [3343] 3343 root468Tail5 winner = winner :=
  (root468Step5).trans root468Scan6
theorem root468Scan4 : scan edge (search edge 3) 4 [3343] 3343 root468Tail4 winner = winner :=
  (root468Step4).trans root468Scan5
theorem root468Scan3 : scan edge (search edge 3) 4 [3343] 3343 root468Tail3 winner = winner :=
  (root468Step3).trans root468Scan4
theorem root468Scan2 : scan edge (search edge 3) 4 [3343] 3343 root468Tail2 winner = winner :=
  (root468Step2).trans root468Scan3
theorem root468Scan1 : scan edge (search edge 3) 4 [3343] 3343 root468Tail1 winner = winner :=
  (root468Step1).trans root468Scan2
theorem root468Scan0 : scan edge (search edge 3) 4 [3343] 3343 root468Tail0 winner = winner :=
  (root468Step0).trans root468Scan1
set_option maxRecDepth 4096 in
theorem root468Branch : search edge 4 [3343] root468Candidates 3343 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3343] 3343 root468Tail0 winner = winner
  exact root468Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
