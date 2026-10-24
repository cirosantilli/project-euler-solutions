import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root351Candidates : List Nat := Data.row2383
set_option maxRecDepth 8192 in
theorem root351Candidates_checked : candidates edge 5 0 2383 tail352 winner.bound = root351Candidates := by
  apply candidates_eq_row edge 5 0 2383 tail352 root351Candidates winner.bound tail352_ordered
  · decide
  · decide
  · funext q
    rfl
def root351Tail41 : List Nat := []
def root351Tail40 : List Nat := 21943 :: root351Tail41
def root351Tail39 : List Nat := 21649 :: root351Tail40
def root351Tail38 : List Nat := 21493 :: root351Tail39
def root351Tail37 : List Nat := 21319 :: root351Tail38
def root351Tail36 : List Nat := 21277 :: root351Tail37
def root351Tail35 : List Nat := 19777 :: root351Tail36
def root351Tail34 : List Nat := 19759 :: root351Tail35
def root351Tail33 : List Nat := 19597 :: root351Tail34
def root351Tail32 : List Nat := 19219 :: root351Tail33
def root351Tail31 : List Nat := 18043 :: root351Tail32
def root351Tail30 : List Nat := 17623 :: root351Tail31
def root351Tail29 : List Nat := 17341 :: root351Tail30
def root351Tail28 : List Nat := 16369 :: root351Tail29
def root351Tail27 : List Nat := 14437 :: root351Tail28
def root351Tail26 : List Nat := 13807 :: root351Tail27
def root351Tail25 : List Nat := 13681 :: root351Tail26
def root351Tail24 : List Nat := 13591 :: root351Tail25
def root351Tail23 : List Nat := 12757 :: root351Tail24
def root351Tail22 : List Nat := 11923 :: root351Tail23
def root351Tail21 : List Nat := 11779 :: root351Tail22
def root351Tail20 : List Nat := 11251 :: root351Tail21
def root351Tail19 : List Nat := 9973 :: root351Tail20
def root351Tail18 : List Nat := 9187 :: root351Tail19
def root351Tail17 : List Nat := 8761 :: root351Tail18
def root351Tail16 : List Nat := 8599 :: root351Tail17
def root351Tail15 : List Nat := 8431 :: root351Tail16
def root351Tail14 : List Nat := 8263 :: root351Tail15
def root351Tail13 : List Nat := 7873 :: root351Tail14
def root351Tail12 : List Nat := 7573 :: root351Tail13
def root351Tail11 : List Nat := 7129 :: root351Tail12
def root351Tail10 : List Nat := 6733 :: root351Tail11
def root351Tail9 : List Nat := 5527 :: root351Tail10
def root351Tail8 : List Nat := 5431 :: root351Tail9
def root351Tail7 : List Nat := 5113 :: root351Tail8
def root351Tail6 : List Nat := 5023 :: root351Tail7
def root351Tail5 : List Nat := 4639 :: root351Tail6
def root351Tail4 : List Nat := 4273 :: root351Tail5
def root351Tail3 : List Nat := 4099 :: root351Tail4
def root351Tail2 : List Nat := 4021 :: root351Tail3
def root351Tail1 : List Nat := 3583 :: root351Tail2
def root351Tail0 : List Nat := 2659 :: root351Tail1
def root351Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root351Candidates0_checked : candidates edge 4 2383 2659 root351Tail1 winner.bound = root351Candidates0 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
theorem root351Step0 : scan edge (search edge 3) 4 [2383] 2383 root351Tail0 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail1 winner := by
  rw [root351Tail0, scan, if_neg (by decide), root351Candidates0_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail1
    (if 3 ≤ root351Candidates0.length then search edge 3 [2383, 2659] root351Candidates0 5042 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root351Candidates1_checked : candidates edge 4 2383 3583 root351Tail2 winner.bound = root351Candidates1 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root351Step1 : scan edge (search edge 3) 4 [2383] 2383 root351Tail1 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail2 winner := by
  rw [root351Tail1, scan, if_neg (by decide), root351Candidates1_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail2
    (if 3 ≤ root351Candidates1.length then search edge 3 [2383, 3583] root351Candidates1 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates2 : List Nat := [6733, 11923]
set_option maxRecDepth 8192 in
theorem root351Candidates2_checked : candidates edge 4 2383 4021 root351Tail3 winner.bound = root351Candidates2 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root351Step2 : scan edge (search edge 3) 4 [2383] 2383 root351Tail2 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail3 winner := by
  rw [root351Tail2, scan, if_neg (by decide), root351Candidates2_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail3
    (if 3 ≤ root351Candidates2.length then search edge 3 [2383, 4021] root351Candidates2 6404 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates3 : List Nat := [12757, 19219]
set_option maxRecDepth 8192 in
theorem root351Candidates3_checked : candidates edge 4 2383 4099 root351Tail4 winner.bound = root351Candidates3 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root351Step3 : scan edge (search edge 3) 4 [2383] 2383 root351Tail3 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail4 winner := by
  rw [root351Tail3, scan, if_neg (by decide), root351Candidates3_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail4
    (if 3 ≤ root351Candidates3.length then search edge 3 [2383, 4099] root351Candidates3 6482 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates4 : List Nat := [5023, 7573]
set_option maxRecDepth 8192 in
theorem root351Candidates4_checked : candidates edge 4 2383 4273 root351Tail5 winner.bound = root351Candidates4 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root351Step4 : scan edge (search edge 3) 4 [2383] 2383 root351Tail4 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail5 winner := by
  rw [root351Tail4, scan, if_neg (by decide), root351Candidates4_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail5
    (if 3 ≤ root351Candidates4.length then search edge 3 [2383, 4273] root351Candidates4 6656 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates5 : List Nat := [9973]
set_option maxRecDepth 8192 in
theorem root351Candidates5_checked : candidates edge 4 2383 4639 root351Tail6 winner.bound = root351Candidates5 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root351Step5 : scan edge (search edge 3) 4 [2383] 2383 root351Tail5 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail6 winner := by
  rw [root351Tail5, scan, if_neg (by decide), root351Candidates5_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail6
    (if 3 ≤ root351Candidates5.length then search edge 3 [2383, 4639] root351Candidates5 7022 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root351Candidates6_checked : candidates edge 4 2383 5023 root351Tail7 winner.bound = root351Candidates6 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root351Step6 : scan edge (search edge 3) 4 [2383] 2383 root351Tail6 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail7 winner := by
  rw [root351Tail6, scan, if_neg (by decide), root351Candidates6_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail7
    (if 3 ≤ root351Candidates6.length then search edge 3 [2383, 5023] root351Candidates6 7406 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates7 : List Nat := [5431]
set_option maxRecDepth 8192 in
theorem root351Candidates7_checked : candidates edge 4 2383 5113 root351Tail8 winner.bound = root351Candidates7 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root351Step7 : scan edge (search edge 3) 4 [2383] 2383 root351Tail7 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail8 winner := by
  rw [root351Tail7, scan, if_neg (by decide), root351Candidates7_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail8
    (if 3 ≤ root351Candidates7.length then search edge 3 [2383, 5113] root351Candidates7 7496 winner else winner) = _
  rw [if_neg (by decide)]
def root351Candidates8 : List Nat := [7573, 8599, 9973]
set_option maxRecDepth 8192 in
theorem root351Candidates8_checked : candidates edge 4 2383 5431 root351Tail9 winner.bound = root351Candidates8 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root351Child8 : search edge 3 [2383, 5431] root351Candidates8 7814 winner = winner := by decide
theorem root351Step8 : scan edge (search edge 3) 4 [2383] 2383 root351Tail8 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail9 winner := by
  rw [root351Tail8, scan, if_neg (by decide), root351Candidates8_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail9
    (if 3 ≤ root351Candidates8.length then search edge 3 [2383, 5431] root351Candidates8 7814 winner else winner) = _
  rw [if_pos (by decide), root351Child8]
def root351Candidates9 : List Nat := [7129]
set_option maxRecDepth 8192 in
theorem root351Candidates9_checked : candidates edge 4 2383 5527 root351Tail10 winner.bound = root351Candidates9 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root351Step9 : scan edge (search edge 3) 4 [2383] 2383 root351Tail9 winner =
    scan edge (search edge 3) 4 [2383] 2383 root351Tail10 winner := by
  rw [root351Tail9, scan, if_neg (by decide), root351Candidates9_checked]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail10
    (if 3 ≤ root351Candidates9.length then search edge 3 [2383, 5527] root351Candidates9 7910 winner else winner) = _
  rw [if_neg (by decide)]
theorem root351Scan10 : scan edge (search edge 3) 4 [2383] 2383 root351Tail10 winner = winner := by rw [root351Tail10, scan, if_pos (by decide)]
theorem root351Scan9 : scan edge (search edge 3) 4 [2383] 2383 root351Tail9 winner = winner :=
  (root351Step9).trans root351Scan10
theorem root351Scan8 : scan edge (search edge 3) 4 [2383] 2383 root351Tail8 winner = winner :=
  (root351Step8).trans root351Scan9
theorem root351Scan7 : scan edge (search edge 3) 4 [2383] 2383 root351Tail7 winner = winner :=
  (root351Step7).trans root351Scan8
theorem root351Scan6 : scan edge (search edge 3) 4 [2383] 2383 root351Tail6 winner = winner :=
  (root351Step6).trans root351Scan7
theorem root351Scan5 : scan edge (search edge 3) 4 [2383] 2383 root351Tail5 winner = winner :=
  (root351Step5).trans root351Scan6
theorem root351Scan4 : scan edge (search edge 3) 4 [2383] 2383 root351Tail4 winner = winner :=
  (root351Step4).trans root351Scan5
theorem root351Scan3 : scan edge (search edge 3) 4 [2383] 2383 root351Tail3 winner = winner :=
  (root351Step3).trans root351Scan4
theorem root351Scan2 : scan edge (search edge 3) 4 [2383] 2383 root351Tail2 winner = winner :=
  (root351Step2).trans root351Scan3
theorem root351Scan1 : scan edge (search edge 3) 4 [2383] 2383 root351Tail1 winner = winner :=
  (root351Step1).trans root351Scan2
theorem root351Scan0 : scan edge (search edge 3) 4 [2383] 2383 root351Tail0 winner = winner :=
  (root351Step0).trans root351Scan1
set_option maxRecDepth 4096 in
theorem root351Branch : search edge 4 [2383] root351Candidates 2383 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2383] 2383 root351Tail0 winner = winner
  exact root351Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
