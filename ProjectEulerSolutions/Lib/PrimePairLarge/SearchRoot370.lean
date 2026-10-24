import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root370Candidates : List Nat := Data.row2549
set_option maxRecDepth 8192 in
theorem root370Candidates_checked : candidates edge 5 0 2549 tail371 winner.bound = root370Candidates := by
  apply candidates_eq_row edge 5 0 2549 tail371 root370Candidates winner.bound tail371_ordered
  · decide
  · decide
  · funext q
    rfl
def root370Tail45 : List Nat := []
def root370Tail44 : List Nat := 22901 :: root370Tail45
def root370Tail43 : List Nat := 22469 :: root370Tail44
def root370Tail42 : List Nat := 21683 :: root370Tail43
def root370Tail41 : List Nat := 21011 :: root370Tail42
def root370Tail40 : List Nat := 19961 :: root370Tail41
def root370Tail39 : List Nat := 19739 :: root370Tail40
def root370Tail38 : List Nat := 19013 :: root370Tail39
def root370Tail37 : List Nat := 19001 :: root370Tail38
def root370Tail36 : List Nat := 18899 :: root370Tail37
def root370Tail35 : List Nat := 18089 :: root370Tail36
def root370Tail34 : List Nat := 15467 :: root370Tail35
def root370Tail33 : List Nat := 14783 :: root370Tail34
def root370Tail32 : List Nat := 14717 :: root370Tail33
def root370Tail31 : List Nat := 14087 :: root370Tail32
def root370Tail30 : List Nat := 14081 :: root370Tail31
def root370Tail29 : List Nat := 13553 :: root370Tail30
def root370Tail28 : List Nat := 13397 :: root370Tail29
def root370Tail27 : List Nat := 12401 :: root370Tail28
def root370Tail26 : List Nat := 11897 :: root370Tail27
def root370Tail25 : List Nat := 10979 :: root370Tail26
def root370Tail24 : List Nat := 10949 :: root370Tail25
def root370Tail23 : List Nat := 10037 :: root370Tail24
def root370Tail22 : List Nat := 9743 :: root370Tail23
def root370Tail21 : List Nat := 9209 :: root370Tail22
def root370Tail20 : List Nat := 8747 :: root370Tail21
def root370Tail19 : List Nat := 8093 :: root370Tail20
def root370Tail18 : List Nat := 8009 :: root370Tail19
def root370Tail17 : List Nat := 7949 :: root370Tail18
def root370Tail16 : List Nat := 7937 :: root370Tail17
def root370Tail15 : List Nat := 7793 :: root370Tail16
def root370Tail14 : List Nat := 7673 :: root370Tail15
def root370Tail13 : List Nat := 7211 :: root370Tail14
def root370Tail12 : List Nat := 6833 :: root370Tail13
def root370Tail11 : List Nat := 6761 :: root370Tail12
def root370Tail10 : List Nat := 6323 :: root370Tail11
def root370Tail9 : List Nat := 5501 :: root370Tail10
def root370Tail8 : List Nat := 5471 :: root370Tail9
def root370Tail7 : List Nat := 4349 :: root370Tail8
def root370Tail6 : List Nat := 4241 :: root370Tail7
def root370Tail5 : List Nat := 4019 :: root370Tail6
def root370Tail4 : List Nat := 3947 :: root370Tail5
def root370Tail3 : List Nat := 3593 :: root370Tail4
def root370Tail2 : List Nat := 3329 :: root370Tail3
def root370Tail1 : List Nat := 3089 :: root370Tail2
def root370Tail0 : List Nat := 2729 :: root370Tail1
def root370Candidates0 : List Nat := [18089]
set_option maxRecDepth 8192 in
theorem root370Candidates0_checked : candidates edge 4 2549 2729 root370Tail1 winner.bound = root370Candidates0 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root370Step0 : scan edge (search edge 3) 4 [2549] 2549 root370Tail0 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail1 winner := by
  rw [root370Tail0, scan, if_neg (by decide), root370Candidates0_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail1
    (if 3 ≤ root370Candidates0.length then search edge 3 [2549, 2729] root370Candidates0 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates1 : List Nat := [8093]
set_option maxRecDepth 8192 in
theorem root370Candidates1_checked : candidates edge 4 2549 3089 root370Tail2 winner.bound = root370Candidates1 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root370Step1 : scan edge (search edge 3) 4 [2549] 2549 root370Tail1 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail2 winner := by
  rw [root370Tail1, scan, if_neg (by decide), root370Candidates1_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail2
    (if 3 ≤ root370Candidates1.length then search edge 3 [2549, 3089] root370Candidates1 5638 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root370Candidates2_checked : candidates edge 4 2549 3329 root370Tail3 winner.bound = root370Candidates2 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root370Step2 : scan edge (search edge 3) 4 [2549] 2549 root370Tail2 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail3 winner := by
  rw [root370Tail2, scan, if_neg (by decide), root370Candidates2_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail3
    (if 3 ≤ root370Candidates2.length then search edge 3 [2549, 3329] root370Candidates2 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root370Candidates3_checked : candidates edge 4 2549 3593 root370Tail4 winner.bound = root370Candidates3 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root370Step3 : scan edge (search edge 3) 4 [2549] 2549 root370Tail3 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail4 winner := by
  rw [root370Tail3, scan, if_neg (by decide), root370Candidates3_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail4
    (if 3 ≤ root370Candidates3.length then search edge 3 [2549, 3593] root370Candidates3 6142 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates4 : List Nat := [8093, 11897, 14081]
set_option maxRecDepth 8192 in
theorem root370Candidates4_checked : candidates edge 4 2549 3947 root370Tail5 winner.bound = root370Candidates4 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root370Child4 : search edge 3 [2549, 3947] root370Candidates4 6496 winner = winner := by decide
theorem root370Step4 : scan edge (search edge 3) 4 [2549] 2549 root370Tail4 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail5 winner := by
  rw [root370Tail4, scan, if_neg (by decide), root370Candidates4_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail5
    (if 3 ≤ root370Candidates4.length then search edge 3 [2549, 3947] root370Candidates4 6496 winner else winner) = _
  rw [if_pos (by decide), root370Child4]
def root370Candidates5 : List Nat := [5471, 7211]
set_option maxRecDepth 8192 in
theorem root370Candidates5_checked : candidates edge 4 2549 4019 root370Tail6 winner.bound = root370Candidates5 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root370Step5 : scan edge (search edge 3) 4 [2549] 2549 root370Tail5 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail6 winner := by
  rw [root370Tail5, scan, if_neg (by decide), root370Candidates5_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail6
    (if 3 ≤ root370Candidates5.length then search edge 3 [2549, 4019] root370Candidates5 6568 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates6 : List Nat := [7673, 8093, 10949]
set_option maxRecDepth 8192 in
theorem root370Candidates6_checked : candidates edge 4 2549 4241 root370Tail7 winner.bound = root370Candidates6 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root370Child6 : search edge 3 [2549, 4241] root370Candidates6 6790 winner = winner := by decide
theorem root370Step6 : scan edge (search edge 3) 4 [2549] 2549 root370Tail6 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail7 winner := by
  rw [root370Tail6, scan, if_neg (by decide), root370Candidates6_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail7
    (if 3 ≤ root370Candidates6.length then search edge 3 [2549, 4241] root370Candidates6 6790 winner else winner) = _
  rw [if_pos (by decide), root370Child6]
def root370Candidates7 : List Nat := [7793, 8093]
set_option maxRecDepth 8192 in
theorem root370Candidates7_checked : candidates edge 4 2549 4349 root370Tail8 winner.bound = root370Candidates7 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root370Step7 : scan edge (search edge 3) 4 [2549] 2549 root370Tail7 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail8 winner := by
  rw [root370Tail7, scan, if_neg (by decide), root370Candidates7_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail8
    (if 3 ≤ root370Candidates7.length then search edge 3 [2549, 4349] root370Candidates7 6898 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root370Candidates8_checked : candidates edge 4 2549 5471 root370Tail9 winner.bound = root370Candidates8 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root370Step8 : scan edge (search edge 3) 4 [2549] 2549 root370Tail8 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail9 winner := by
  rw [root370Tail8, scan, if_neg (by decide), root370Candidates8_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail9
    (if 3 ≤ root370Candidates8.length then search edge 3 [2549, 5471] root370Candidates8 8020 winner else winner) = _
  rw [if_neg (by decide)]
def root370Candidates9 : List Nat := [7211]
set_option maxRecDepth 8192 in
theorem root370Candidates9_checked : candidates edge 4 2549 5501 root370Tail10 winner.bound = root370Candidates9 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root370Step9 : scan edge (search edge 3) 4 [2549] 2549 root370Tail9 winner =
    scan edge (search edge 3) 4 [2549] 2549 root370Tail10 winner := by
  rw [root370Tail9, scan, if_neg (by decide), root370Candidates9_checked]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail10
    (if 3 ≤ root370Candidates9.length then search edge 3 [2549, 5501] root370Candidates9 8050 winner else winner) = _
  rw [if_neg (by decide)]
theorem root370Scan10 : scan edge (search edge 3) 4 [2549] 2549 root370Tail10 winner = winner := by rw [root370Tail10, scan, if_pos (by decide)]
theorem root370Scan9 : scan edge (search edge 3) 4 [2549] 2549 root370Tail9 winner = winner :=
  (root370Step9).trans root370Scan10
theorem root370Scan8 : scan edge (search edge 3) 4 [2549] 2549 root370Tail8 winner = winner :=
  (root370Step8).trans root370Scan9
theorem root370Scan7 : scan edge (search edge 3) 4 [2549] 2549 root370Tail7 winner = winner :=
  (root370Step7).trans root370Scan8
theorem root370Scan6 : scan edge (search edge 3) 4 [2549] 2549 root370Tail6 winner = winner :=
  (root370Step6).trans root370Scan7
theorem root370Scan5 : scan edge (search edge 3) 4 [2549] 2549 root370Tail5 winner = winner :=
  (root370Step5).trans root370Scan6
theorem root370Scan4 : scan edge (search edge 3) 4 [2549] 2549 root370Tail4 winner = winner :=
  (root370Step4).trans root370Scan5
theorem root370Scan3 : scan edge (search edge 3) 4 [2549] 2549 root370Tail3 winner = winner :=
  (root370Step3).trans root370Scan4
theorem root370Scan2 : scan edge (search edge 3) 4 [2549] 2549 root370Tail2 winner = winner :=
  (root370Step2).trans root370Scan3
theorem root370Scan1 : scan edge (search edge 3) 4 [2549] 2549 root370Tail1 winner = winner :=
  (root370Step1).trans root370Scan2
theorem root370Scan0 : scan edge (search edge 3) 4 [2549] 2549 root370Tail0 winner = winner :=
  (root370Step0).trans root370Scan1
set_option maxRecDepth 4096 in
theorem root370Branch : search edge 4 [2549] root370Candidates 2549 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2549] 2549 root370Tail0 winner = winner
  exact root370Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
