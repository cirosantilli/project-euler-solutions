import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root242Candidates : List Nat := Data.row1553
set_option maxRecDepth 8192 in
theorem root242Candidates_checked : candidates edge 5 0 1553 tail243 winner.bound = root242Candidates := by
  apply candidates_eq_row edge 5 0 1553 tail243 root242Candidates winner.bound tail243_ordered
  · decide
  · decide
  · funext q
    rfl
def root242Tail45 : List Nat := []
def root242Tail44 : List Nat := 23321 :: root242Tail45
def root242Tail43 : List Nat := 22091 :: root242Tail44
def root242Tail42 : List Nat := 21587 :: root242Tail43
def root242Tail41 : List Nat := 20147 :: root242Tail42
def root242Tail40 : List Nat := 19751 :: root242Tail41
def root242Tail39 : List Nat := 19709 :: root242Tail40
def root242Tail38 : List Nat := 19379 :: root242Tail39
def root242Tail37 : List Nat := 18803 :: root242Tail38
def root242Tail36 : List Nat := 18593 :: root242Tail37
def root242Tail35 : List Nat := 18089 :: root242Tail36
def root242Tail34 : List Nat := 18077 :: root242Tail35
def root242Tail33 : List Nat := 16451 :: root242Tail34
def root242Tail32 : List Nat := 16433 :: root242Tail33
def root242Tail31 : List Nat := 16253 :: root242Tail32
def root242Tail30 : List Nat := 15473 :: root242Tail31
def root242Tail29 : List Nat := 15287 :: root242Tail30
def root242Tail28 : List Nat := 14591 :: root242Tail29
def root242Tail27 : List Nat := 13859 :: root242Tail28
def root242Tail26 : List Nat := 12401 :: root242Tail27
def root242Tail25 : List Nat := 11621 :: root242Tail26
def root242Tail24 : List Nat := 11483 :: root242Tail25
def root242Tail23 : List Nat := 10949 :: root242Tail24
def root242Tail22 : List Nat := 10937 :: root242Tail23
def root242Tail21 : List Nat := 10709 :: root242Tail22
def root242Tail20 : List Nat := 10433 :: root242Tail21
def root242Tail19 : List Nat := 9791 :: root242Tail20
def root242Tail18 : List Nat := 8753 :: root242Tail19
def root242Tail17 : List Nat := 8147 :: root242Tail18
def root242Tail16 : List Nat := 8111 :: root242Tail17
def root242Tail15 : List Nat := 7883 :: root242Tail16
def root242Tail14 : List Nat := 7757 :: root242Tail15
def root242Tail13 : List Nat := 7211 :: root242Tail14
def root242Tail12 : List Nat := 7013 :: root242Tail13
def root242Tail11 : List Nat := 6947 :: root242Tail12
def root242Tail10 : List Nat := 6221 :: root242Tail11
def root242Tail9 : List Nat := 6203 :: root242Tail10
def root242Tail8 : List Nat := 4049 :: root242Tail9
def root242Tail7 : List Nat := 2687 :: root242Tail8
def root242Tail6 : List Nat := 2549 :: root242Tail7
def root242Tail5 : List Nat := 2393 :: root242Tail6
def root242Tail4 : List Nat := 2087 :: root242Tail5
def root242Tail3 : List Nat := 1811 :: root242Tail4
def root242Tail2 : List Nat := 1697 :: root242Tail3
def root242Tail1 : List Nat := 1613 :: root242Tail2
def root242Tail0 : List Nat := 1583 :: root242Tail1
def root242Candidates0 : List Nat := [10949]
set_option maxRecDepth 8192 in
theorem root242Candidates0_checked : candidates edge 4 1553 1583 root242Tail1 winner.bound = root242Candidates0 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
theorem root242Step0 : scan edge (search edge 3) 4 [1553] 1553 root242Tail0 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail1 winner := by
  rw [root242Tail0, scan, if_neg (by decide), root242Candidates0_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail1
    (if 3 ≤ root242Candidates0.length then search edge 3 [1553, 1583] root242Candidates0 3136 winner else winner) = _
  rw [if_neg (by decide)]
def root242Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root242Candidates1_checked : candidates edge 4 1553 1613 root242Tail2 winner.bound = root242Candidates1 := by
  unfold candidates
  have row : edge 1613 = (SortedNatTree.contains FastRows.row1613) := by
    change Data.row1613.contains = (SortedNatTree.contains FastRows.row1613)
    exact FastRows.row1613_checked.symm
  rw [row]
  decide
theorem root242Step1 : scan edge (search edge 3) 4 [1553] 1553 root242Tail1 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail2 winner := by
  rw [root242Tail1, scan, if_neg (by decide), root242Candidates1_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail2
    (if 3 ≤ root242Candidates1.length then search edge 3 [1553, 1613] root242Candidates1 3166 winner else winner) = _
  rw [if_neg (by decide)]
def root242Candidates2 : List Nat := [11483]
set_option maxRecDepth 8192 in
theorem root242Candidates2_checked : candidates edge 4 1553 1697 root242Tail3 winner.bound = root242Candidates2 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root242Step2 : scan edge (search edge 3) 4 [1553] 1553 root242Tail2 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail3 winner := by
  rw [root242Tail2, scan, if_neg (by decide), root242Candidates2_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail3
    (if 3 ≤ root242Candidates2.length then search edge 3 [1553, 1697] root242Candidates2 3250 winner else winner) = _
  rw [if_neg (by decide)]
def root242Candidates3 : List Nat := [13859, 15287, 16451]
set_option maxRecDepth 8192 in
theorem root242Candidates3_checked : candidates edge 4 1553 1811 root242Tail4 winner.bound = root242Candidates3 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root242Child3 : search edge 3 [1553, 1811] root242Candidates3 3364 winner = winner := by decide
theorem root242Step3 : scan edge (search edge 3) 4 [1553] 1553 root242Tail3 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail4 winner := by
  rw [root242Tail3, scan, if_neg (by decide), root242Candidates3_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail4
    (if 3 ≤ root242Candidates3.length then search edge 3 [1553, 1811] root242Candidates3 3364 winner else winner) = _
  rw [if_pos (by decide), root242Child3]
def root242Candidates4 : List Nat := [15473]
set_option maxRecDepth 8192 in
theorem root242Candidates4_checked : candidates edge 4 1553 2087 root242Tail5 winner.bound = root242Candidates4 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root242Step4 : scan edge (search edge 3) 4 [1553] 1553 root242Tail4 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail5 winner := by
  rw [root242Tail4, scan, if_neg (by decide), root242Candidates4_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail5
    (if 3 ≤ root242Candidates4.length then search edge 3 [1553, 2087] root242Candidates4 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root242Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root242Candidates5_checked : candidates edge 4 1553 2393 root242Tail6 winner.bound = root242Candidates5 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root242Step5 : scan edge (search edge 3) 4 [1553] 1553 root242Tail5 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail6 winner := by
  rw [root242Tail5, scan, if_neg (by decide), root242Candidates5_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail6
    (if 3 ≤ root242Candidates5.length then search edge 3 [1553, 2393] root242Candidates5 3946 winner else winner) = _
  rw [if_neg (by decide)]
def root242Candidates6 : List Nat := [7211, 10949, 12401, 18089]
set_option maxRecDepth 8192 in
theorem root242Candidates6_checked : candidates edge 4 1553 2549 root242Tail7 winner.bound = root242Candidates6 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root242Child6 : search edge 3 [1553, 2549] root242Candidates6 4102 winner = winner := by decide
theorem root242Step6 : scan edge (search edge 3) 4 [1553] 1553 root242Tail6 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail7 winner := by
  rw [root242Tail6, scan, if_neg (by decide), root242Candidates6_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail7
    (if 3 ≤ root242Candidates6.length then search edge 3 [1553, 2549] root242Candidates6 4102 winner else winner) = _
  rw [if_pos (by decide), root242Child6]
def root242Candidates7 : List Nat := [6221]
set_option maxRecDepth 8192 in
theorem root242Candidates7_checked : candidates edge 4 1553 2687 root242Tail8 winner.bound = root242Candidates7 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root242Step7 : scan edge (search edge 3) 4 [1553] 1553 root242Tail7 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail8 winner := by
  rw [root242Tail7, scan, if_neg (by decide), root242Candidates7_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail8
    (if 3 ≤ root242Candidates7.length then search edge 3 [1553, 2687] root242Candidates7 4240 winner else winner) = _
  rw [if_neg (by decide)]
def root242Candidates8 : List Nat := [8111, 19751]
set_option maxRecDepth 8192 in
theorem root242Candidates8_checked : candidates edge 4 1553 4049 root242Tail9 winner.bound = root242Candidates8 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root242Step8 : scan edge (search edge 3) 4 [1553] 1553 root242Tail8 winner =
    scan edge (search edge 3) 4 [1553] 1553 root242Tail9 winner := by
  rw [root242Tail8, scan, if_neg (by decide), root242Candidates8_checked]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail9
    (if 3 ≤ root242Candidates8.length then search edge 3 [1553, 4049] root242Candidates8 5602 winner else winner) = _
  rw [if_neg (by decide)]
theorem root242Scan9 : scan edge (search edge 3) 4 [1553] 1553 root242Tail9 winner = winner := by rw [root242Tail9, scan, if_pos (by decide)]
theorem root242Scan8 : scan edge (search edge 3) 4 [1553] 1553 root242Tail8 winner = winner :=
  (root242Step8).trans root242Scan9
theorem root242Scan7 : scan edge (search edge 3) 4 [1553] 1553 root242Tail7 winner = winner :=
  (root242Step7).trans root242Scan8
theorem root242Scan6 : scan edge (search edge 3) 4 [1553] 1553 root242Tail6 winner = winner :=
  (root242Step6).trans root242Scan7
theorem root242Scan5 : scan edge (search edge 3) 4 [1553] 1553 root242Tail5 winner = winner :=
  (root242Step5).trans root242Scan6
theorem root242Scan4 : scan edge (search edge 3) 4 [1553] 1553 root242Tail4 winner = winner :=
  (root242Step4).trans root242Scan5
theorem root242Scan3 : scan edge (search edge 3) 4 [1553] 1553 root242Tail3 winner = winner :=
  (root242Step3).trans root242Scan4
theorem root242Scan2 : scan edge (search edge 3) 4 [1553] 1553 root242Tail2 winner = winner :=
  (root242Step2).trans root242Scan3
theorem root242Scan1 : scan edge (search edge 3) 4 [1553] 1553 root242Tail1 winner = winner :=
  (root242Step1).trans root242Scan2
theorem root242Scan0 : scan edge (search edge 3) 4 [1553] 1553 root242Tail0 winner = winner :=
  (root242Step0).trans root242Scan1
set_option maxRecDepth 4096 in
theorem root242Branch : search edge 4 [1553] root242Candidates 1553 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1553] 1553 root242Tail0 winner = winner
  exact root242Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
