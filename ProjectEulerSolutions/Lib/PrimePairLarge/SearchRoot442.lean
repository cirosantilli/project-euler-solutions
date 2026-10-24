import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root442Candidates : List Nat := Data.row3121
set_option maxRecDepth 8192 in
theorem root442Candidates_checked : candidates edge 5 0 3121 tail443 winner.bound = root442Candidates := by
  apply candidates_eq_row edge 5 0 3121 tail443 root442Candidates winner.bound tail443_ordered
  · decide
  · decide
  · funext q
    rfl
def root442Tail38 : List Nat := []
def root442Tail37 : List Nat := 22483 :: root442Tail38
def root442Tail36 : List Nat := 21787 :: root442Tail37
def root442Tail35 : List Nat := 21319 :: root442Tail36
def root442Tail34 : List Nat := 19081 :: root442Tail35
def root442Tail33 : List Nat := 19009 :: root442Tail34
def root442Tail32 : List Nat := 18973 :: root442Tail33
def root442Tail31 : List Nat := 18913 :: root442Tail32
def root442Tail30 : List Nat := 18427 :: root442Tail31
def root442Tail29 : List Nat := 18379 :: root442Tail30
def root442Tail28 : List Nat := 16843 :: root442Tail29
def root442Tail27 : List Nat := 16759 :: root442Tail28
def root442Tail26 : List Nat := 15373 :: root442Tail27
def root442Tail25 : List Nat := 14731 :: root442Tail26
def root442Tail24 : List Nat := 14563 :: root442Tail25
def root442Tail23 : List Nat := 13903 :: root442Tail24
def root442Tail22 : List Nat := 11887 :: root442Tail23
def root442Tail21 : List Nat := 11677 :: root442Tail22
def root442Tail20 : List Nat := 11497 :: root442Tail21
def root442Tail19 : List Nat := 10993 :: root442Tail20
def root442Tail18 : List Nat := 10657 :: root442Tail19
def root442Tail17 : List Nat := 10639 :: root442Tail18
def root442Tail16 : List Nat := 9187 :: root442Tail17
def root442Tail15 : List Nat := 9043 :: root442Tail16
def root442Tail14 : List Nat := 8599 :: root442Tail15
def root442Tail13 : List Nat := 8221 :: root442Tail14
def root442Tail12 : List Nat := 7759 :: root442Tail13
def root442Tail11 : List Nat := 7507 :: root442Tail12
def root442Tail10 : List Nat := 7393 :: root442Tail11
def root442Tail9 : List Nat := 5851 :: root442Tail10
def root442Tail8 : List Nat := 4903 :: root442Tail9
def root442Tail7 : List Nat := 4639 :: root442Tail8
def root442Tail6 : List Nat := 4597 :: root442Tail7
def root442Tail5 : List Nat := 4507 :: root442Tail6
def root442Tail4 : List Nat := 4357 :: root442Tail5
def root442Tail3 : List Nat := 4219 :: root442Tail4
def root442Tail2 : List Nat := 4201 :: root442Tail3
def root442Tail1 : List Nat := 3673 :: root442Tail2
def root442Tail0 : List Nat := 3433 :: root442Tail1
def root442Candidates0 : List Nat := [8221, 18427]
set_option maxRecDepth 8192 in
theorem root442Candidates0_checked : candidates edge 4 3121 3433 root442Tail1 winner.bound = root442Candidates0 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root442Step0 : scan edge (search edge 3) 4 [3121] 3121 root442Tail0 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail1 winner := by
  rw [root442Tail0, scan, if_neg (by decide), root442Candidates0_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail1
    (if 3 ≤ root442Candidates0.length then search edge 3 [3121, 3433] root442Candidates0 6554 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root442Candidates1_checked : candidates edge 4 3121 3673 root442Tail2 winner.bound = root442Candidates1 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root442Step1 : scan edge (search edge 3) 4 [3121] 3121 root442Tail1 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail2 winner := by
  rw [root442Tail1, scan, if_neg (by decide), root442Candidates1_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail2
    (if 3 ≤ root442Candidates1.length then search edge 3 [3121, 3673] root442Candidates1 6794 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root442Candidates2_checked : candidates edge 4 3121 4201 root442Tail3 winner.bound = root442Candidates2 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root442Step2 : scan edge (search edge 3) 4 [3121] 3121 root442Tail2 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail3 winner := by
  rw [root442Tail2, scan, if_neg (by decide), root442Candidates2_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail3
    (if 3 ≤ root442Candidates2.length then search edge 3 [3121, 4201] root442Candidates2 7322 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates3 : List Nat := [4597]
set_option maxRecDepth 8192 in
theorem root442Candidates3_checked : candidates edge 4 3121 4219 root442Tail4 winner.bound = root442Candidates3 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
theorem root442Step3 : scan edge (search edge 3) 4 [3121] 3121 root442Tail3 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail4 winner := by
  rw [root442Tail3, scan, if_neg (by decide), root442Candidates3_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail4
    (if 3 ≤ root442Candidates3.length then search edge 3 [3121, 4219] root442Candidates3 7340 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root442Candidates4_checked : candidates edge 4 3121 4357 root442Tail5 winner.bound = root442Candidates4 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root442Step4 : scan edge (search edge 3) 4 [3121] 3121 root442Tail4 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail5 winner := by
  rw [root442Tail4, scan, if_neg (by decide), root442Candidates4_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail5
    (if 3 ≤ root442Candidates4.length then search edge 3 [3121, 4357] root442Candidates4 7478 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates5 : List Nat := [7759, 9187, 16759]
set_option maxRecDepth 8192 in
theorem root442Candidates5_checked : candidates edge 4 3121 4507 root442Tail6 winner.bound = root442Candidates5 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root442Child5 : search edge 3 [3121, 4507] root442Candidates5 7628 winner = winner := by decide
theorem root442Step5 : scan edge (search edge 3) 4 [3121] 3121 root442Tail5 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail6 winner := by
  rw [root442Tail5, scan, if_neg (by decide), root442Candidates5_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail6
    (if 3 ≤ root442Candidates5.length then search edge 3 [3121, 4507] root442Candidates5 7628 winner else winner) = _
  rw [if_pos (by decide), root442Child5]
def root442Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root442Candidates6_checked : candidates edge 4 3121 4597 root442Tail7 winner.bound = root442Candidates6 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root442Step6 : scan edge (search edge 3) 4 [3121] 3121 root442Tail6 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail7 winner := by
  rw [root442Tail6, scan, if_neg (by decide), root442Candidates6_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail7
    (if 3 ≤ root442Candidates6.length then search edge 3 [3121, 4597] root442Candidates6 7718 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root442Candidates7_checked : candidates edge 4 3121 4639 root442Tail8 winner.bound = root442Candidates7 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root442Step7 : scan edge (search edge 3) 4 [3121] 3121 root442Tail7 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail8 winner := by
  rw [root442Tail7, scan, if_neg (by decide), root442Candidates7_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail8
    (if 3 ≤ root442Candidates7.length then search edge 3 [3121, 4639] root442Candidates7 7760 winner else winner) = _
  rw [if_neg (by decide)]
def root442Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root442Candidates8_checked : candidates edge 4 3121 4903 root442Tail9 winner.bound = root442Candidates8 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root442Step8 : scan edge (search edge 3) 4 [3121] 3121 root442Tail8 winner =
    scan edge (search edge 3) 4 [3121] 3121 root442Tail9 winner := by
  rw [root442Tail8, scan, if_neg (by decide), root442Candidates8_checked]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail9
    (if 3 ≤ root442Candidates8.length then search edge 3 [3121, 4903] root442Candidates8 8024 winner else winner) = _
  rw [if_neg (by decide)]
theorem root442Scan9 : scan edge (search edge 3) 4 [3121] 3121 root442Tail9 winner = winner := by rw [root442Tail9, scan, if_pos (by decide)]
theorem root442Scan8 : scan edge (search edge 3) 4 [3121] 3121 root442Tail8 winner = winner :=
  (root442Step8).trans root442Scan9
theorem root442Scan7 : scan edge (search edge 3) 4 [3121] 3121 root442Tail7 winner = winner :=
  (root442Step7).trans root442Scan8
theorem root442Scan6 : scan edge (search edge 3) 4 [3121] 3121 root442Tail6 winner = winner :=
  (root442Step6).trans root442Scan7
theorem root442Scan5 : scan edge (search edge 3) 4 [3121] 3121 root442Tail5 winner = winner :=
  (root442Step5).trans root442Scan6
theorem root442Scan4 : scan edge (search edge 3) 4 [3121] 3121 root442Tail4 winner = winner :=
  (root442Step4).trans root442Scan5
theorem root442Scan3 : scan edge (search edge 3) 4 [3121] 3121 root442Tail3 winner = winner :=
  (root442Step3).trans root442Scan4
theorem root442Scan2 : scan edge (search edge 3) 4 [3121] 3121 root442Tail2 winner = winner :=
  (root442Step2).trans root442Scan3
theorem root442Scan1 : scan edge (search edge 3) 4 [3121] 3121 root442Tail1 winner = winner :=
  (root442Step1).trans root442Scan2
theorem root442Scan0 : scan edge (search edge 3) 4 [3121] 3121 root442Tail0 winner = winner :=
  (root442Step0).trans root442Scan1
set_option maxRecDepth 4096 in
theorem root442Branch : search edge 4 [3121] root442Candidates 3121 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3121] 3121 root442Tail0 winner = winner
  exact root442Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
