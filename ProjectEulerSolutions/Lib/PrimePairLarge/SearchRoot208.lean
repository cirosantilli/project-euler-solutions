import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root208Candidates : List Nat := Data.row1297
set_option maxRecDepth 8192 in
theorem root208Candidates_checked : candidates edge 5 0 1297 tail209 winner.bound = root208Candidates := by
  apply candidates_eq_row edge 5 0 1297 tail209 root208Candidates winner.bound tail209_ordered
  · decide
  · decide
  · funext q
    rfl
def root208Tail36 : List Nat := []
def root208Tail35 : List Nat := 24733 :: root208Tail36
def root208Tail34 : List Nat := 24169 :: root208Tail35
def root208Tail33 : List Nat := 24043 :: root208Tail34
def root208Tail32 : List Nat := 23899 :: root208Tail33
def root208Tail31 : List Nat := 23029 :: root208Tail32
def root208Tail30 : List Nat := 22993 :: root208Tail31
def root208Tail29 : List Nat := 20599 :: root208Tail30
def root208Tail28 : List Nat := 20323 :: root208Tail29
def root208Tail27 : List Nat := 19507 :: root208Tail28
def root208Tail26 : List Nat := 18661 :: root208Tail27
def root208Tail25 : List Nat := 18451 :: root208Tail26
def root208Tail24 : List Nat := 18307 :: root208Tail25
def root208Tail23 : List Nat := 17053 :: root208Tail24
def root208Tail22 : List Nat := 16603 :: root208Tail23
def root208Tail21 : List Nat := 16417 :: root208Tail22
def root208Tail20 : List Nat := 15199 :: root208Tail21
def root208Tail19 : List Nat := 15121 :: root208Tail20
def root208Tail18 : List Nat := 13159 :: root208Tail19
def root208Tail17 : List Nat := 12577 :: root208Tail18
def root208Tail16 : List Nat := 12049 :: root208Tail17
def root208Tail15 : List Nat := 12007 :: root208Tail16
def root208Tail14 : List Nat := 11383 :: root208Tail15
def root208Tail13 : List Nat := 10771 :: root208Tail14
def root208Tail12 : List Nat := 9697 :: root208Tail13
def root208Tail11 : List Nat := 9601 :: root208Tail12
def root208Tail10 : List Nat := 9433 :: root208Tail11
def root208Tail9 : List Nat := 8731 :: root208Tail10
def root208Tail8 : List Nat := 7603 :: root208Tail9
def root208Tail7 : List Nat := 6781 :: root208Tail8
def root208Tail6 : List Nat := 5821 :: root208Tail7
def root208Tail5 : List Nat := 4813 :: root208Tail6
def root208Tail4 : List Nat := 3547 :: root208Tail5
def root208Tail3 : List Nat := 3529 :: root208Tail4
def root208Tail2 : List Nat := 2671 :: root208Tail3
def root208Tail1 : List Nat := 2293 :: root208Tail2
def root208Tail0 : List Nat := 1549 :: root208Tail1
def root208Candidates0 : List Nat := [3529]
set_option maxRecDepth 8192 in
theorem root208Candidates0_checked : candidates edge 4 1297 1549 root208Tail1 winner.bound = root208Candidates0 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
theorem root208Step0 : scan edge (search edge 3) 4 [1297] 1297 root208Tail0 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail1 winner := by
  rw [root208Tail0, scan, if_neg (by decide), root208Candidates0_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail1
    (if 3 ≤ root208Candidates0.length then search edge 3 [1297, 1549] root208Candidates0 2846 winner else winner) = _
  rw [if_neg (by decide)]
def root208Candidates1 : List Nat := [2671]
set_option maxRecDepth 8192 in
theorem root208Candidates1_checked : candidates edge 4 1297 2293 root208Tail2 winner.bound = root208Candidates1 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root208Step1 : scan edge (search edge 3) 4 [1297] 1297 root208Tail1 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail2 winner := by
  rw [root208Tail1, scan, if_neg (by decide), root208Candidates1_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail2
    (if 3 ≤ root208Candidates1.length then search edge 3 [1297, 2293] root208Candidates1 3590 winner else winner) = _
  rw [if_neg (by decide)]
def root208Candidates2 : List Nat := [18307]
set_option maxRecDepth 8192 in
theorem root208Candidates2_checked : candidates edge 4 1297 2671 root208Tail3 winner.bound = root208Candidates2 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root208Step2 : scan edge (search edge 3) 4 [1297] 1297 root208Tail2 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail3 winner := by
  rw [root208Tail2, scan, if_neg (by decide), root208Candidates2_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail3
    (if 3 ≤ root208Candidates2.length then search edge 3 [1297, 2671] root208Candidates2 3968 winner else winner) = _
  rw [if_neg (by decide)]
def root208Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root208Candidates3_checked : candidates edge 4 1297 3529 root208Tail4 winner.bound = root208Candidates3 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root208Step3 : scan edge (search edge 3) 4 [1297] 1297 root208Tail3 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail4 winner := by
  rw [root208Tail3, scan, if_neg (by decide), root208Candidates3_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail4
    (if 3 ≤ root208Candidates3.length then search edge 3 [1297, 3529] root208Candidates3 4826 winner else winner) = _
  rw [if_neg (by decide)]
def root208Candidates4 : List Nat := [18451]
set_option maxRecDepth 8192 in
theorem root208Candidates4_checked : candidates edge 4 1297 3547 root208Tail5 winner.bound = root208Candidates4 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root208Step4 : scan edge (search edge 3) 4 [1297] 1297 root208Tail4 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail5 winner := by
  rw [root208Tail4, scan, if_neg (by decide), root208Candidates4_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail5
    (if 3 ≤ root208Candidates4.length then search edge 3 [1297, 3547] root208Candidates4 4844 winner else winner) = _
  rw [if_neg (by decide)]
def root208Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root208Candidates5_checked : candidates edge 4 1297 4813 root208Tail6 winner.bound = root208Candidates5 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root208Step5 : scan edge (search edge 3) 4 [1297] 1297 root208Tail5 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail6 winner := by
  rw [root208Tail5, scan, if_neg (by decide), root208Candidates5_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail6
    (if 3 ≤ root208Candidates5.length then search edge 3 [1297, 4813] root208Candidates5 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root208Candidates6 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root208Candidates6_checked : candidates edge 4 1297 5821 root208Tail7 winner.bound = root208Candidates6 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root208Step6 : scan edge (search edge 3) 4 [1297] 1297 root208Tail6 winner =
    scan edge (search edge 3) 4 [1297] 1297 root208Tail7 winner := by
  rw [root208Tail6, scan, if_neg (by decide), root208Candidates6_checked]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail7
    (if 3 ≤ root208Candidates6.length then search edge 3 [1297, 5821] root208Candidates6 7118 winner else winner) = _
  rw [if_neg (by decide)]
theorem root208Scan7 : scan edge (search edge 3) 4 [1297] 1297 root208Tail7 winner = winner := by rw [root208Tail7, scan, if_pos (by decide)]
theorem root208Scan6 : scan edge (search edge 3) 4 [1297] 1297 root208Tail6 winner = winner :=
  (root208Step6).trans root208Scan7
theorem root208Scan5 : scan edge (search edge 3) 4 [1297] 1297 root208Tail5 winner = winner :=
  (root208Step5).trans root208Scan6
theorem root208Scan4 : scan edge (search edge 3) 4 [1297] 1297 root208Tail4 winner = winner :=
  (root208Step4).trans root208Scan5
theorem root208Scan3 : scan edge (search edge 3) 4 [1297] 1297 root208Tail3 winner = winner :=
  (root208Step3).trans root208Scan4
theorem root208Scan2 : scan edge (search edge 3) 4 [1297] 1297 root208Tail2 winner = winner :=
  (root208Step2).trans root208Scan3
theorem root208Scan1 : scan edge (search edge 3) 4 [1297] 1297 root208Tail1 winner = winner :=
  (root208Step1).trans root208Scan2
theorem root208Scan0 : scan edge (search edge 3) 4 [1297] 1297 root208Tail0 winner = winner :=
  (root208Step0).trans root208Scan1
set_option maxRecDepth 4096 in
theorem root208Branch : search edge 4 [1297] root208Candidates 1297 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1297] 1297 root208Tail0 winner = winner
  exact root208Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
