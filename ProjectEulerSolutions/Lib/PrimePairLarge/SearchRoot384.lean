import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root384Candidates : List Nat := Data.row2671
set_option maxRecDepth 8192 in
theorem root384Candidates_checked : candidates edge 5 0 2671 tail385 winner.bound = root384Candidates := by
  apply candidates_eq_row edge 5 0 2671 tail385 root384Candidates winner.bound tail385_ordered
  · decide
  · decide
  · funext q
    rfl
def root384Tail42 : List Nat := []
def root384Tail41 : List Nat := 22993 :: root384Tail42
def root384Tail40 : List Nat := 21319 :: root384Tail41
def root384Tail39 : List Nat := 21193 :: root384Tail40
def root384Tail38 : List Nat := 21121 :: root384Tail39
def root384Tail37 : List Nat := 20611 :: root384Tail38
def root384Tail36 : List Nat := 20233 :: root384Tail37
def root384Tail35 : List Nat := 19843 :: root384Tail36
def root384Tail34 : List Nat := 19597 :: root384Tail35
def root384Tail33 : List Nat := 18583 :: root384Tail34
def root384Tail32 : List Nat := 18541 :: root384Tail33
def root384Tail31 : List Nat := 18457 :: root384Tail32
def root384Tail30 : List Nat := 18307 :: root384Tail31
def root384Tail29 : List Nat := 17971 :: root384Tail30
def root384Tail28 : List Nat := 17623 :: root384Tail29
def root384Tail27 : List Nat := 15907 :: root384Tail28
def root384Tail26 : List Nat := 14983 :: root384Tail27
def root384Tail25 : List Nat := 14437 :: root384Tail26
def root384Tail24 : List Nat := 13633 :: root384Tail25
def root384Tail23 : List Nat := 13177 :: root384Tail24
def root384Tail22 : List Nat := 12889 :: root384Tail23
def root384Tail21 : List Nat := 11617 :: root384Tail22
def root384Tail20 : List Nat := 11287 :: root384Tail21
def root384Tail19 : List Nat := 10651 :: root384Tail20
def root384Tail18 : List Nat := 10159 :: root384Tail19
def root384Tail17 : List Nat := 9811 :: root384Tail18
def root384Tail16 : List Nat := 9067 :: root384Tail17
def root384Tail15 : List Nat := 8761 :: root384Tail16
def root384Tail14 : List Nat := 8053 :: root384Tail15
def root384Tail13 : List Nat := 7723 :: root384Tail14
def root384Tail12 : List Nat := 7507 :: root384Tail13
def root384Tail11 : List Nat := 7129 :: root384Tail12
def root384Tail10 : List Nat := 6829 :: root384Tail11
def root384Tail9 : List Nat := 6709 :: root384Tail10
def root384Tail8 : List Nat := 6343 :: root384Tail9
def root384Tail7 : List Nat := 6211 :: root384Tail8
def root384Tail6 : List Nat := 5077 :: root384Tail7
def root384Tail5 : List Nat := 4603 :: root384Tail6
def root384Tail4 : List Nat := 4231 :: root384Tail5
def root384Tail3 : List Nat := 3853 :: root384Tail4
def root384Tail2 : List Nat := 3769 :: root384Tail3
def root384Tail1 : List Nat := 3571 :: root384Tail2
def root384Tail0 : List Nat := 3361 :: root384Tail1
def root384Candidates0 : List Nat := [6343, 9811, 14983]
set_option maxRecDepth 8192 in
theorem root384Candidates0_checked : candidates edge 4 2671 3361 root384Tail1 winner.bound = root384Candidates0 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root384Child0 : search edge 3 [2671, 3361] root384Candidates0 6032 winner = winner := by decide
theorem root384Step0 : scan edge (search edge 3) 4 [2671] 2671 root384Tail0 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail1 winner := by
  rw [root384Tail0, scan, if_neg (by decide), root384Candidates0_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail1
    (if 3 ≤ root384Candidates0.length then search edge 3 [2671, 3361] root384Candidates0 6032 winner else winner) = _
  rw [if_pos (by decide), root384Child0]
def root384Candidates1 : List Nat := [9067, 15907]
set_option maxRecDepth 8192 in
theorem root384Candidates1_checked : candidates edge 4 2671 3571 root384Tail2 winner.bound = root384Candidates1 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root384Step1 : scan edge (search edge 3) 4 [2671] 2671 root384Tail1 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail2 winner := by
  rw [root384Tail1, scan, if_neg (by decide), root384Candidates1_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail2
    (if 3 ≤ root384Candidates1.length then search edge 3 [2671, 3571] root384Candidates1 6242 winner else winner) = _
  rw [if_neg (by decide)]
def root384Candidates2 : List Nat := [4603]
set_option maxRecDepth 8192 in
theorem root384Candidates2_checked : candidates edge 4 2671 3769 root384Tail3 winner.bound = root384Candidates2 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root384Step2 : scan edge (search edge 3) 4 [2671] 2671 root384Tail2 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail3 winner := by
  rw [root384Tail2, scan, if_neg (by decide), root384Candidates2_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail3
    (if 3 ≤ root384Candidates2.length then search edge 3 [2671, 3769] root384Candidates2 6440 winner else winner) = _
  rw [if_neg (by decide)]
def root384Candidates3 : List Nat := [6709, 18457]
set_option maxRecDepth 8192 in
theorem root384Candidates3_checked : candidates edge 4 2671 3853 root384Tail4 winner.bound = root384Candidates3 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root384Step3 : scan edge (search edge 3) 4 [2671] 2671 root384Tail3 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail4 winner := by
  rw [root384Tail3, scan, if_neg (by decide), root384Candidates3_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail4
    (if 3 ≤ root384Candidates3.length then search edge 3 [2671, 3853] root384Candidates3 6524 winner else winner) = _
  rw [if_neg (by decide)]
def root384Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root384Candidates4_checked : candidates edge 4 2671 4231 root384Tail5 winner.bound = root384Candidates4 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root384Step4 : scan edge (search edge 3) 4 [2671] 2671 root384Tail4 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail5 winner := by
  rw [root384Tail4, scan, if_neg (by decide), root384Candidates4_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail5
    (if 3 ≤ root384Candidates4.length then search edge 3 [2671, 4231] root384Candidates4 6902 winner else winner) = _
  rw [if_neg (by decide)]
def root384Candidates5 : List Nat := [8761, 14983, 18583]
set_option maxRecDepth 8192 in
theorem root384Candidates5_checked : candidates edge 4 2671 4603 root384Tail6 winner.bound = root384Candidates5 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root384Child5 : search edge 3 [2671, 4603] root384Candidates5 7274 winner = winner := by decide
theorem root384Step5 : scan edge (search edge 3) 4 [2671] 2671 root384Tail5 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail6 winner := by
  rw [root384Tail5, scan, if_neg (by decide), root384Candidates5_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail6
    (if 3 ≤ root384Candidates5.length then search edge 3 [2671, 4603] root384Candidates5 7274 winner else winner) = _
  rw [if_pos (by decide), root384Child5]
def root384Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root384Candidates6_checked : candidates edge 4 2671 5077 root384Tail7 winner.bound = root384Candidates6 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root384Step6 : scan edge (search edge 3) 4 [2671] 2671 root384Tail6 winner =
    scan edge (search edge 3) 4 [2671] 2671 root384Tail7 winner := by
  rw [root384Tail6, scan, if_neg (by decide), root384Candidates6_checked]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail7
    (if 3 ≤ root384Candidates6.length then search edge 3 [2671, 5077] root384Candidates6 7748 winner else winner) = _
  rw [if_neg (by decide)]
theorem root384Scan7 : scan edge (search edge 3) 4 [2671] 2671 root384Tail7 winner = winner := by rw [root384Tail7, scan, if_pos (by decide)]
theorem root384Scan6 : scan edge (search edge 3) 4 [2671] 2671 root384Tail6 winner = winner :=
  (root384Step6).trans root384Scan7
theorem root384Scan5 : scan edge (search edge 3) 4 [2671] 2671 root384Tail5 winner = winner :=
  (root384Step5).trans root384Scan6
theorem root384Scan4 : scan edge (search edge 3) 4 [2671] 2671 root384Tail4 winner = winner :=
  (root384Step4).trans root384Scan5
theorem root384Scan3 : scan edge (search edge 3) 4 [2671] 2671 root384Tail3 winner = winner :=
  (root384Step3).trans root384Scan4
theorem root384Scan2 : scan edge (search edge 3) 4 [2671] 2671 root384Tail2 winner = winner :=
  (root384Step2).trans root384Scan3
theorem root384Scan1 : scan edge (search edge 3) 4 [2671] 2671 root384Tail1 winner = winner :=
  (root384Step1).trans root384Scan2
theorem root384Scan0 : scan edge (search edge 3) 4 [2671] 2671 root384Tail0 winner = winner :=
  (root384Step0).trans root384Scan1
set_option maxRecDepth 4096 in
theorem root384Branch : search edge 4 [2671] root384Candidates 2671 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2671] 2671 root384Tail0 winner = winner
  exact root384Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
