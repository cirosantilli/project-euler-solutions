import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root255Candidates : List Nat := Data.row1627
set_option maxRecDepth 8192 in
theorem root255Candidates_checked : candidates edge 5 0 1627 tail256 winner.bound = root255Candidates := by
  apply candidates_eq_row edge 5 0 1627 tail256 root255Candidates winner.bound tail256_ordered
  · decide
  · decide
  · funext q
    rfl
def root255Tail40 : List Nat := []
def root255Tail39 : List Nat := 23719 :: root255Tail40
def root255Tail38 : List Nat := 21997 :: root255Tail39
def root255Tail37 : List Nat := 21313 :: root255Tail38
def root255Tail36 : List Nat := 21061 :: root255Tail37
def root255Tail35 : List Nat := 20071 :: root255Tail36
def root255Tail34 : List Nat := 18379 :: root255Tail35
def root255Tail33 : List Nat := 18301 :: root255Tail34
def root255Tail32 : List Nat := 17239 :: root255Tail33
def root255Tail31 : List Nat := 16903 :: root255Tail32
def root255Tail30 : List Nat := 16363 :: root255Tail31
def root255Tail29 : List Nat := 16033 :: root255Tail30
def root255Tail28 : List Nat := 14593 :: root255Tail29
def root255Tail27 : List Nat := 14479 :: root255Tail28
def root255Tail26 : List Nat := 14143 :: root255Tail27
def root255Tail25 : List Nat := 13381 :: root255Tail26
def root255Tail24 : List Nat := 12289 :: root255Tail25
def root255Tail23 : List Nat := 11833 :: root255Tail24
def root255Tail22 : List Nat := 11317 :: root255Tail23
def root255Tail21 : List Nat := 10903 :: root255Tail22
def root255Tail20 : List Nat := 10651 :: root255Tail21
def root255Tail19 : List Nat := 10267 :: root255Tail20
def root255Tail18 : List Nat := 9391 :: root255Tail19
def root255Tail17 : List Nat := 8761 :: root255Tail18
def root255Tail16 : List Nat := 8461 :: root255Tail17
def root255Tail15 : List Nat := 8287 :: root255Tail16
def root255Tail14 : List Nat := 7741 :: root255Tail15
def root255Tail13 : List Nat := 7507 :: root255Tail14
def root255Tail12 : List Nat := 7159 :: root255Tail13
def root255Tail11 : List Nat := 6961 :: root255Tail12
def root255Tail10 : List Nat := 6277 :: root255Tail11
def root255Tail9 : List Nat := 5821 :: root255Tail10
def root255Tail8 : List Nat := 5179 :: root255Tail9
def root255Tail7 : List Nat := 5011 :: root255Tail8
def root255Tail6 : List Nat := 4591 :: root255Tail7
def root255Tail5 : List Nat := 3769 :: root255Tail6
def root255Tail4 : List Nat := 3637 :: root255Tail5
def root255Tail3 : List Nat := 3463 :: root255Tail4
def root255Tail2 : List Nat := 3217 :: root255Tail3
def root255Tail1 : List Nat := 2917 :: root255Tail2
def root255Tail0 : List Nat := 1753 :: root255Tail1
def root255Candidates0 : List Nat := [21313]
set_option maxRecDepth 8192 in
theorem root255Candidates0_checked : candidates edge 4 1627 1753 root255Tail1 winner.bound = root255Candidates0 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
theorem root255Step0 : scan edge (search edge 3) 4 [1627] 1627 root255Tail0 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail1 winner := by
  rw [root255Tail0, scan, if_neg (by decide), root255Candidates0_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail1
    (if 3 ≤ root255Candidates0.length then search edge 3 [1627, 1753] root255Candidates0 3380 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates1 : List Nat := [5011, 10267]
set_option maxRecDepth 8192 in
theorem root255Candidates1_checked : candidates edge 4 1627 2917 root255Tail2 winner.bound = root255Candidates1 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root255Step1 : scan edge (search edge 3) 4 [1627] 1627 root255Tail1 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail2 winner := by
  rw [root255Tail1, scan, if_neg (by decide), root255Candidates1_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail2
    (if 3 ≤ root255Candidates1.length then search edge 3 [1627, 2917] root255Candidates1 4544 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates2 : List Nat := [5821]
set_option maxRecDepth 8192 in
theorem root255Candidates2_checked : candidates edge 4 1627 3217 root255Tail3 winner.bound = root255Candidates2 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root255Step2 : scan edge (search edge 3) 4 [1627] 1627 root255Tail2 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail3 winner := by
  rw [root255Tail2, scan, if_neg (by decide), root255Candidates2_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail3
    (if 3 ≤ root255Candidates2.length then search edge 3 [1627, 3217] root255Candidates2 4844 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates3 : List Nat := [10267]
set_option maxRecDepth 8192 in
theorem root255Candidates3_checked : candidates edge 4 1627 3463 root255Tail4 winner.bound = root255Candidates3 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root255Step3 : scan edge (search edge 3) 4 [1627] 1627 root255Tail3 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail4 winner := by
  rw [root255Tail3, scan, if_neg (by decide), root255Candidates3_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail4
    (if 3 ≤ root255Candidates3.length then search edge 3 [1627, 3463] root255Candidates3 5090 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root255Candidates4_checked : candidates edge 4 1627 3637 root255Tail5 winner.bound = root255Candidates4 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root255Step4 : scan edge (search edge 3) 4 [1627] 1627 root255Tail4 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail5 winner := by
  rw [root255Tail4, scan, if_neg (by decide), root255Candidates4_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail5
    (if 3 ≤ root255Candidates4.length then search edge 3 [1627, 3637] root255Candidates4 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates5 : List Nat := [18379]
set_option maxRecDepth 8192 in
theorem root255Candidates5_checked : candidates edge 4 1627 3769 root255Tail6 winner.bound = root255Candidates5 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root255Step5 : scan edge (search edge 3) 4 [1627] 1627 root255Tail5 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail6 winner := by
  rw [root255Tail5, scan, if_neg (by decide), root255Candidates5_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail6
    (if 3 ≤ root255Candidates5.length then search edge 3 [1627, 3769] root255Candidates5 5396 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates6 : List Nat := [8287, 10903, 11833]
set_option maxRecDepth 8192 in
theorem root255Candidates6_checked : candidates edge 4 1627 4591 root255Tail7 winner.bound = root255Candidates6 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root255Child6 : search edge 3 [1627, 4591] root255Candidates6 6218 winner = winner := by decide
theorem root255Step6 : scan edge (search edge 3) 4 [1627] 1627 root255Tail6 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail7 winner := by
  rw [root255Tail6, scan, if_neg (by decide), root255Candidates6_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail7
    (if 3 ≤ root255Candidates6.length then search edge 3 [1627, 4591] root255Candidates6 6218 winner else winner) = _
  rw [if_pos (by decide), root255Child6]
def root255Candidates7 : List Nat := [6961, 14479]
set_option maxRecDepth 8192 in
theorem root255Candidates7_checked : candidates edge 4 1627 5011 root255Tail8 winner.bound = root255Candidates7 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root255Step7 : scan edge (search edge 3) 4 [1627] 1627 root255Tail7 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail8 winner := by
  rw [root255Tail7, scan, if_neg (by decide), root255Candidates7_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail8
    (if 3 ≤ root255Candidates7.length then search edge 3 [1627, 5011] root255Candidates7 6638 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root255Candidates8_checked : candidates edge 4 1627 5179 root255Tail9 winner.bound = root255Candidates8 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root255Step8 : scan edge (search edge 3) 4 [1627] 1627 root255Tail8 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail9 winner := by
  rw [root255Tail8, scan, if_neg (by decide), root255Candidates8_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail9
    (if 3 ≤ root255Candidates8.length then search edge 3 [1627, 5179] root255Candidates8 6806 winner else winner) = _
  rw [if_neg (by decide)]
def root255Candidates9 : List Nat := [14479, 16903]
set_option maxRecDepth 8192 in
theorem root255Candidates9_checked : candidates edge 4 1627 5821 root255Tail10 winner.bound = root255Candidates9 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root255Step9 : scan edge (search edge 3) 4 [1627] 1627 root255Tail9 winner =
    scan edge (search edge 3) 4 [1627] 1627 root255Tail10 winner := by
  rw [root255Tail9, scan, if_neg (by decide), root255Candidates9_checked]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail10
    (if 3 ≤ root255Candidates9.length then search edge 3 [1627, 5821] root255Candidates9 7448 winner else winner) = _
  rw [if_neg (by decide)]
theorem root255Scan10 : scan edge (search edge 3) 4 [1627] 1627 root255Tail10 winner = winner := by rw [root255Tail10, scan, if_pos (by decide)]
theorem root255Scan9 : scan edge (search edge 3) 4 [1627] 1627 root255Tail9 winner = winner :=
  (root255Step9).trans root255Scan10
theorem root255Scan8 : scan edge (search edge 3) 4 [1627] 1627 root255Tail8 winner = winner :=
  (root255Step8).trans root255Scan9
theorem root255Scan7 : scan edge (search edge 3) 4 [1627] 1627 root255Tail7 winner = winner :=
  (root255Step7).trans root255Scan8
theorem root255Scan6 : scan edge (search edge 3) 4 [1627] 1627 root255Tail6 winner = winner :=
  (root255Step6).trans root255Scan7
theorem root255Scan5 : scan edge (search edge 3) 4 [1627] 1627 root255Tail5 winner = winner :=
  (root255Step5).trans root255Scan6
theorem root255Scan4 : scan edge (search edge 3) 4 [1627] 1627 root255Tail4 winner = winner :=
  (root255Step4).trans root255Scan5
theorem root255Scan3 : scan edge (search edge 3) 4 [1627] 1627 root255Tail3 winner = winner :=
  (root255Step3).trans root255Scan4
theorem root255Scan2 : scan edge (search edge 3) 4 [1627] 1627 root255Tail2 winner = winner :=
  (root255Step2).trans root255Scan3
theorem root255Scan1 : scan edge (search edge 3) 4 [1627] 1627 root255Tail1 winner = winner :=
  (root255Step1).trans root255Scan2
theorem root255Scan0 : scan edge (search edge 3) 4 [1627] 1627 root255Tail0 winner = winner :=
  (root255Step0).trans root255Scan1
set_option maxRecDepth 4096 in
theorem root255Branch : search edge 4 [1627] root255Candidates 1627 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1627] 1627 root255Tail0 winner = winner
  exact root255Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
