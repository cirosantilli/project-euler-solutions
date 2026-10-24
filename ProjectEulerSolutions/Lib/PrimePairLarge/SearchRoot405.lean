import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root405Candidates : List Nat := Data.row2801
set_option maxRecDepth 8192 in
theorem root405Candidates_checked : candidates edge 5 0 2801 tail406 winner.bound = root405Candidates := by
  apply candidates_eq_row edge 5 0 2801 tail406 root405Candidates winner.bound tail406_ordered
  · decide
  · decide
  · funext q
    rfl
def root405Tail54 : List Nat := []
def root405Tail53 : List Nat := 23201 :: root405Tail54
def root405Tail52 : List Nat := 22901 :: root405Tail53
def root405Tail51 : List Nat := 22613 :: root405Tail52
def root405Tail50 : List Nat := 21713 :: root405Tail51
def root405Tail49 : List Nat := 21599 :: root405Tail50
def root405Tail48 : List Nat := 20177 :: root405Tail49
def root405Tail47 : List Nat := 19571 :: root405Tail48
def root405Tail46 : List Nat := 19403 :: root405Tail47
def root405Tail45 : List Nat := 18719 :: root405Tail46
def root405Tail44 : List Nat := 17837 :: root405Tail45
def root405Tail43 : List Nat := 17471 :: root405Tail44
def root405Tail42 : List Nat := 16619 :: root405Tail43
def root405Tail41 : List Nat := 16433 :: root405Tail42
def root405Tail40 : List Nat := 16349 :: root405Tail41
def root405Tail39 : List Nat := 15497 :: root405Tail40
def root405Tail38 : List Nat := 14759 :: root405Tail39
def root405Tail37 : List Nat := 14633 :: root405Tail38
def root405Tail36 : List Nat := 13829 :: root405Tail37
def root405Tail35 : List Nat := 13451 :: root405Tail36
def root405Tail34 : List Nat := 13187 :: root405Tail35
def root405Tail33 : List Nat := 12011 :: root405Tail34
def root405Tail32 : List Nat := 11987 :: root405Tail33
def root405Tail31 : List Nat := 11483 :: root405Tail32
def root405Tail30 : List Nat := 11369 :: root405Tail31
def root405Tail29 : List Nat := 10667 :: root405Tail30
def root405Tail28 : List Nat := 9857 :: root405Tail29
def root405Tail27 : List Nat := 9743 :: root405Tail28
def root405Tail26 : List Nat := 9491 :: root405Tail27
def root405Tail25 : List Nat := 9461 :: root405Tail26
def root405Tail24 : List Nat := 8387 :: root405Tail25
def root405Tail23 : List Nat := 8171 :: root405Tail24
def root405Tail22 : List Nat := 8087 :: root405Tail23
def root405Tail21 : List Nat := 7919 :: root405Tail22
def root405Tail20 : List Nat := 7331 :: root405Tail21
def root405Tail19 : List Nat := 7193 :: root405Tail20
def root405Tail18 : List Nat := 7043 :: root405Tail19
def root405Tail17 : List Nat := 6803 :: root405Tail18
def root405Tail16 : List Nat := 6689 :: root405Tail17
def root405Tail15 : List Nat := 6311 :: root405Tail16
def root405Tail14 : List Nat := 6101 :: root405Tail15
def root405Tail13 : List Nat := 5987 :: root405Tail14
def root405Tail12 : List Nat := 5693 :: root405Tail13
def root405Tail11 : List Nat := 5573 :: root405Tail12
def root405Tail10 : List Nat := 5417 :: root405Tail11
def root405Tail9 : List Nat := 5051 :: root405Tail10
def root405Tail8 : List Nat := 4733 :: root405Tail9
def root405Tail7 : List Nat := 4229 :: root405Tail8
def root405Tail6 : List Nat := 4073 :: root405Tail7
def root405Tail5 : List Nat := 3917 :: root405Tail6
def root405Tail4 : List Nat := 3851 :: root405Tail5
def root405Tail3 : List Nat := 3803 :: root405Tail4
def root405Tail2 : List Nat := 3527 :: root405Tail3
def root405Tail1 : List Nat := 3221 :: root405Tail2
def root405Tail0 : List Nat := 2843 :: root405Tail1
def root405Candidates0 : List Nat := [8171, 9857, 10667, 17471]
set_option maxRecDepth 8192 in
theorem root405Candidates0_checked : candidates edge 4 2801 2843 root405Tail1 winner.bound = root405Candidates0 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root405Child0 : search edge 3 [2801, 2843] root405Candidates0 5644 winner = winner := by decide
theorem root405Step0 : scan edge (search edge 3) 4 [2801] 2801 root405Tail0 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail1 winner := by
  rw [root405Tail0, scan, if_neg (by decide), root405Candidates0_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail1
    (if 3 ≤ root405Candidates0.length then search edge 3 [2801, 2843] root405Candidates0 5644 winner else winner) = _
  rw [if_pos (by decide), root405Child0]
def root405Candidates1 : List Nat := [3851, 6311]
set_option maxRecDepth 8192 in
theorem root405Candidates1_checked : candidates edge 4 2801 3221 root405Tail2 winner.bound = root405Candidates1 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root405Step1 : scan edge (search edge 3) 4 [2801] 2801 root405Tail1 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail2 winner := by
  rw [root405Tail1, scan, if_neg (by decide), root405Candidates1_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail2
    (if 3 ≤ root405Candidates1.length then search edge 3 [2801, 3221] root405Candidates1 6022 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates2 : List Nat := [6803, 17471, 17837, 19571]
set_option maxRecDepth 8192 in
theorem root405Candidates2_checked : candidates edge 4 2801 3527 root405Tail3 winner.bound = root405Candidates2 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root405Child2 : search edge 3 [2801, 3527] root405Candidates2 6328 winner = winner := by decide
theorem root405Step2 : scan edge (search edge 3) 4 [2801] 2801 root405Tail2 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail3 winner := by
  rw [root405Tail2, scan, if_neg (by decide), root405Candidates2_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail3
    (if 3 ≤ root405Candidates2.length then search edge 3 [2801, 3527] root405Candidates2 6328 winner else winner) = _
  rw [if_pos (by decide), root405Child2]
def root405Candidates3 : List Nat := [5051]
set_option maxRecDepth 8192 in
theorem root405Candidates3_checked : candidates edge 4 2801 3803 root405Tail4 winner.bound = root405Candidates3 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root405Step3 : scan edge (search edge 3) 4 [2801] 2801 root405Tail3 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail4 winner := by
  rw [root405Tail3, scan, if_neg (by decide), root405Candidates3_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail4
    (if 3 ≤ root405Candidates3.length then search edge 3 [2801, 3803] root405Candidates3 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates4 : List Nat := [7043, 9491]
set_option maxRecDepth 8192 in
theorem root405Candidates4_checked : candidates edge 4 2801 3851 root405Tail5 winner.bound = root405Candidates4 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root405Step4 : scan edge (search edge 3) 4 [2801] 2801 root405Tail4 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail5 winner := by
  rw [root405Tail4, scan, if_neg (by decide), root405Candidates4_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail5
    (if 3 ≤ root405Candidates4.length then search edge 3 [2801, 3851] root405Candidates4 6652 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates5 : List Nat := [5987, 7331]
set_option maxRecDepth 8192 in
theorem root405Candidates5_checked : candidates edge 4 2801 3917 root405Tail6 winner.bound = root405Candidates5 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root405Step5 : scan edge (search edge 3) 4 [2801] 2801 root405Tail5 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail6 winner := by
  rw [root405Tail5, scan, if_neg (by decide), root405Candidates5_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail6
    (if 3 ≤ root405Candidates5.length then search edge 3 [2801, 3917] root405Candidates5 6718 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates6 : List Nat := [17837]
set_option maxRecDepth 8192 in
theorem root405Candidates6_checked : candidates edge 4 2801 4073 root405Tail7 winner.bound = root405Candidates6 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root405Step6 : scan edge (search edge 3) 4 [2801] 2801 root405Tail6 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail7 winner := by
  rw [root405Tail6, scan, if_neg (by decide), root405Candidates6_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail7
    (if 3 ≤ root405Candidates6.length then search edge 3 [2801, 4073] root405Candidates6 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates7 : List Nat := [6689, 16349, 16619, 18719]
set_option maxRecDepth 8192 in
theorem root405Candidates7_checked : candidates edge 4 2801 4229 root405Tail8 winner.bound = root405Candidates7 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root405Child7 : search edge 3 [2801, 4229] root405Candidates7 7030 winner = winner := by decide
theorem root405Step7 : scan edge (search edge 3) 4 [2801] 2801 root405Tail7 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail8 winner := by
  rw [root405Tail7, scan, if_neg (by decide), root405Candidates7_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail8
    (if 3 ≤ root405Candidates7.length then search edge 3 [2801, 4229] root405Candidates7 7030 winner else winner) = _
  rw [if_pos (by decide), root405Child7]
def root405Candidates8 : List Nat := [6689, 7331, 7919, 13187]
set_option maxRecDepth 8192 in
theorem root405Candidates8_checked : candidates edge 4 2801 4733 root405Tail9 winner.bound = root405Candidates8 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root405Child8 : search edge 3 [2801, 4733] root405Candidates8 7534 winner = winner := by decide
theorem root405Step8 : scan edge (search edge 3) 4 [2801] 2801 root405Tail8 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail9 winner := by
  rw [root405Tail8, scan, if_neg (by decide), root405Candidates8_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail9
    (if 3 ≤ root405Candidates8.length then search edge 3 [2801, 4733] root405Candidates8 7534 winner else winner) = _
  rw [if_pos (by decide), root405Child8]
def root405Candidates9 : List Nat := [6101]
set_option maxRecDepth 8192 in
theorem root405Candidates9_checked : candidates edge 4 2801 5051 root405Tail10 winner.bound = root405Candidates9 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root405Step9 : scan edge (search edge 3) 4 [2801] 2801 root405Tail9 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail10 winner := by
  rw [root405Tail9, scan, if_neg (by decide), root405Candidates9_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail10
    (if 3 ≤ root405Candidates9.length then search edge 3 [2801, 5051] root405Candidates9 7852 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates10 : List Nat := [13187]
set_option maxRecDepth 8192 in
theorem root405Candidates10_checked : candidates edge 4 2801 5417 root405Tail11 winner.bound = root405Candidates10 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root405Step10 : scan edge (search edge 3) 4 [2801] 2801 root405Tail10 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail11 winner := by
  rw [root405Tail10, scan, if_neg (by decide), root405Candidates10_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail11
    (if 3 ≤ root405Candidates10.length then search edge 3 [2801, 5417] root405Candidates10 8218 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root405Candidates11_checked : candidates edge 4 2801 5573 root405Tail12 winner.bound = root405Candidates11 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root405Step11 : scan edge (search edge 3) 4 [2801] 2801 root405Tail11 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail12 winner := by
  rw [root405Tail11, scan, if_neg (by decide), root405Candidates11_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail12
    (if 3 ≤ root405Candidates11.length then search edge 3 [2801, 5573] root405Candidates11 8374 winner else winner) = _
  rw [if_neg (by decide)]
def root405Candidates12 : List Nat := [8171]
set_option maxRecDepth 8192 in
theorem root405Candidates12_checked : candidates edge 4 2801 5693 root405Tail13 winner.bound = root405Candidates12 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root405Step12 : scan edge (search edge 3) 4 [2801] 2801 root405Tail12 winner =
    scan edge (search edge 3) 4 [2801] 2801 root405Tail13 winner := by
  rw [root405Tail12, scan, if_neg (by decide), root405Candidates12_checked]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail13
    (if 3 ≤ root405Candidates12.length then search edge 3 [2801, 5693] root405Candidates12 8494 winner else winner) = _
  rw [if_neg (by decide)]
theorem root405Scan13 : scan edge (search edge 3) 4 [2801] 2801 root405Tail13 winner = winner := by rw [root405Tail13, scan, if_pos (by decide)]
theorem root405Scan12 : scan edge (search edge 3) 4 [2801] 2801 root405Tail12 winner = winner :=
  (root405Step12).trans root405Scan13
theorem root405Scan11 : scan edge (search edge 3) 4 [2801] 2801 root405Tail11 winner = winner :=
  (root405Step11).trans root405Scan12
theorem root405Scan10 : scan edge (search edge 3) 4 [2801] 2801 root405Tail10 winner = winner :=
  (root405Step10).trans root405Scan11
theorem root405Scan9 : scan edge (search edge 3) 4 [2801] 2801 root405Tail9 winner = winner :=
  (root405Step9).trans root405Scan10
theorem root405Scan8 : scan edge (search edge 3) 4 [2801] 2801 root405Tail8 winner = winner :=
  (root405Step8).trans root405Scan9
theorem root405Scan7 : scan edge (search edge 3) 4 [2801] 2801 root405Tail7 winner = winner :=
  (root405Step7).trans root405Scan8
theorem root405Scan6 : scan edge (search edge 3) 4 [2801] 2801 root405Tail6 winner = winner :=
  (root405Step6).trans root405Scan7
theorem root405Scan5 : scan edge (search edge 3) 4 [2801] 2801 root405Tail5 winner = winner :=
  (root405Step5).trans root405Scan6
theorem root405Scan4 : scan edge (search edge 3) 4 [2801] 2801 root405Tail4 winner = winner :=
  (root405Step4).trans root405Scan5
theorem root405Scan3 : scan edge (search edge 3) 4 [2801] 2801 root405Tail3 winner = winner :=
  (root405Step3).trans root405Scan4
theorem root405Scan2 : scan edge (search edge 3) 4 [2801] 2801 root405Tail2 winner = winner :=
  (root405Step2).trans root405Scan3
theorem root405Scan1 : scan edge (search edge 3) 4 [2801] 2801 root405Tail1 winner = winner :=
  (root405Step1).trans root405Scan2
theorem root405Scan0 : scan edge (search edge 3) 4 [2801] 2801 root405Tail0 winner = winner :=
  (root405Step0).trans root405Scan1
set_option maxRecDepth 4096 in
theorem root405Branch : search edge 4 [2801] root405Candidates 2801 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2801] 2801 root405Tail0 winner = winner
  exact root405Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
