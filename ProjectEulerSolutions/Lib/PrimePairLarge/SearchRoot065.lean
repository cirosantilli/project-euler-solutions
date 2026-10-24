import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root65Candidates : List Nat := Data.row337
set_option maxRecDepth 8192 in
theorem root65Candidates_checked : candidates edge 5 0 337 tail66 winner.bound = root65Candidates := by
  apply candidates_eq_row edge 5 0 337 tail66 root65Candidates winner.bound tail66_ordered
  · decide
  · decide
  · funext q
    rfl
def root65Tail62 : List Nat := []
def root65Tail61 : List Nat := 24421 :: root65Tail62
def root65Tail60 : List Nat := 24337 :: root65Tail61
def root65Tail59 : List Nat := 23623 :: root65Tail60
def root65Tail58 : List Nat := 23293 :: root65Tail59
def root65Tail57 : List Nat := 23203 :: root65Tail58
def root65Tail56 : List Nat := 22483 :: root65Tail57
def root65Tail55 : List Nat := 22063 :: root65Tail56
def root65Tail54 : List Nat := 21517 :: root65Tail55
def root65Tail53 : List Nat := 21139 :: root65Tail54
def root65Tail52 : List Nat := 20719 :: root65Tail53
def root65Tail51 : List Nat := 20509 :: root65Tail52
def root65Tail50 : List Nat := 20107 :: root65Tail51
def root65Tail49 : List Nat := 19597 :: root65Tail50
def root65Tail48 : List Nat := 19429 :: root65Tail49
def root65Tail47 : List Nat := 18757 :: root65Tail48
def root65Tail46 : List Nat := 18541 :: root65Tail47
def root65Tail45 : List Nat := 17737 :: root65Tail46
def root65Tail44 : List Nat := 17707 :: root65Tail45
def root65Tail43 : List Nat := 17449 :: root65Tail44
def root65Tail42 : List Nat := 16987 :: root65Tail43
def root65Tail41 : List Nat := 16519 :: root65Tail42
def root65Tail40 : List Nat := 16273 :: root65Tail41
def root65Tail39 : List Nat := 16231 :: root65Tail40
def root65Tail38 : List Nat := 15559 :: root65Tail39
def root65Tail37 : List Nat := 15307 :: root65Tail38
def root65Tail36 : List Nat := 14011 :: root65Tail37
def root65Tail35 : List Nat := 13963 :: root65Tail36
def root65Tail34 : List Nat := 13759 :: root65Tail35
def root65Tail33 : List Nat := 13681 :: root65Tail34
def root65Tail32 : List Nat := 13513 :: root65Tail33
def root65Tail31 : List Nat := 13411 :: root65Tail32
def root65Tail30 : List Nat := 12967 :: root65Tail31
def root65Tail29 : List Nat := 12781 :: root65Tail30
def root65Tail28 : List Nat := 12163 :: root65Tail29
def root65Tail27 : List Nat := 11863 :: root65Tail28
def root65Tail26 : List Nat := 11617 :: root65Tail27
def root65Tail25 : List Nat := 10723 :: root65Tail26
def root65Tail24 : List Nat := 10429 :: root65Tail25
def root65Tail23 : List Nat := 9613 :: root65Tail24
def root65Tail22 : List Nat := 9127 :: root65Tail23
def root65Tail21 : List Nat := 9001 :: root65Tail22
def root65Tail20 : List Nat := 7723 :: root65Tail21
def root65Tail19 : List Nat := 6949 :: root65Tail20
def root65Tail18 : List Nat := 6781 :: root65Tail19
def root65Tail17 : List Nat := 5857 :: root65Tail18
def root65Tail16 : List Nat := 5431 :: root65Tail17
def root65Tail15 : List Nat := 4729 :: root65Tail16
def root65Tail14 : List Nat := 4507 :: root65Tail15
def root65Tail13 : List Nat := 3499 :: root65Tail14
def root65Tail12 : List Nat := 3343 :: root65Tail13
def root65Tail11 : List Nat := 3217 :: root65Tail12
def root65Tail10 : List Nat := 3169 :: root65Tail11
def root65Tail9 : List Nat := 2797 :: root65Tail10
def root65Tail8 : List Nat := 2371 :: root65Tail9
def root65Tail7 : List Nat := 1867 :: root65Tail8
def root65Tail6 : List Nat := 1297 :: root65Tail7
def root65Tail5 : List Nat := 1153 :: root65Tail6
def root65Tail4 : List Nat := 919 :: root65Tail5
def root65Tail3 : List Nat := 691 :: root65Tail4
def root65Tail2 : List Nat := 607 :: root65Tail3
def root65Tail1 : List Nat := 397 :: root65Tail2
def root65Tail0 : List Nat := 349 :: root65Tail1
def root65Candidates0 : List Nat := [919]
set_option maxRecDepth 8192 in
theorem root65Candidates0_checked : candidates edge 4 337 349 root65Tail1 winner.bound = root65Candidates0 := by
  unfold candidates
  have row : edge 349 = (SortedNatTree.contains FastRows.row349) := by
    change Data.row349.contains = (SortedNatTree.contains FastRows.row349)
    exact FastRows.row349_checked.symm
  rw [row]
  decide
theorem root65Step0 : scan edge (search edge 3) 4 [337] 337 root65Tail0 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail1 winner := by
  rw [root65Tail0, scan, if_neg (by decide), root65Candidates0_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail1
    (if 3 ≤ root65Candidates0.length then search edge 3 [337, 349] root65Candidates0 686 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates1 : List Nat := [17707, 21139]
set_option maxRecDepth 8192 in
theorem root65Candidates1_checked : candidates edge 4 337 397 root65Tail2 winner.bound = root65Candidates1 := by
  unfold candidates
  have row : edge 397 = (SortedNatTree.contains FastRows.row397) := by
    change Data.row397.contains = (SortedNatTree.contains FastRows.row397)
    exact FastRows.row397_checked.symm
  rw [row]
  decide
theorem root65Step1 : scan edge (search edge 3) 4 [337] 337 root65Tail1 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail2 winner := by
  rw [root65Tail1, scan, if_neg (by decide), root65Candidates1_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail2
    (if 3 ≤ root65Candidates1.length then search edge 3 [337, 397] root65Candidates1 734 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates2 : List Nat := [1297, 12781]
set_option maxRecDepth 8192 in
theorem root65Candidates2_checked : candidates edge 4 337 607 root65Tail3 winner.bound = root65Candidates2 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
theorem root65Step2 : scan edge (search edge 3) 4 [337] 337 root65Tail2 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail3 winner := by
  rw [root65Tail2, scan, if_neg (by decide), root65Candidates2_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail3
    (if 3 ≤ root65Candidates2.length then search edge 3 [337, 607] root65Candidates2 944 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates3 : List Nat := [919, 4729, 5431, 16519, 23293]
set_option maxRecDepth 8192 in
theorem root65Candidates3_checked : candidates edge 4 337 691 root65Tail4 winner.bound = root65Candidates3 := by
  unfold candidates
  have row : edge 691 = (SortedNatTree.contains FastRows.row691) := by
    change Data.row691.contains = (SortedNatTree.contains FastRows.row691)
    exact FastRows.row691_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root65Child3 : search edge 3 [337, 691] root65Candidates3 1028 winner = winner := by decide
theorem root65Step3 : scan edge (search edge 3) 4 [337] 337 root65Tail3 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail4 winner := by
  rw [root65Tail3, scan, if_neg (by decide), root65Candidates3_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail4
    (if 3 ≤ root65Candidates3.length then search edge 3 [337, 691] root65Candidates3 1028 winner else winner) = _
  rw [if_pos (by decide), root65Child3]
def root65Candidates4 : List Nat := [20509]
set_option maxRecDepth 8192 in
theorem root65Candidates4_checked : candidates edge 4 337 919 root65Tail5 winner.bound = root65Candidates4 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
theorem root65Step4 : scan edge (search edge 3) 4 [337] 337 root65Tail4 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail5 winner := by
  rw [root65Tail4, scan, if_neg (by decide), root65Candidates4_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail5
    (if 3 ≤ root65Candidates4.length then search edge 3 [337, 919] root65Candidates4 1256 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates5 : List Nat := [9613]
set_option maxRecDepth 8192 in
theorem root65Candidates5_checked : candidates edge 4 337 1153 root65Tail6 winner.bound = root65Candidates5 := by
  unfold candidates
  have row : edge 1153 = (SortedNatTree.contains FastRows.row1153) := by
    change Data.row1153.contains = (SortedNatTree.contains FastRows.row1153)
    exact FastRows.row1153_checked.symm
  rw [row]
  decide
theorem root65Step5 : scan edge (search edge 3) 4 [337] 337 root65Tail5 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail6 winner := by
  rw [root65Tail5, scan, if_neg (by decide), root65Candidates5_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail6
    (if 3 ≤ root65Candidates5.length then search edge 3 [337, 1153] root65Candidates5 1490 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates6 : List Nat := [6781]
set_option maxRecDepth 8192 in
theorem root65Candidates6_checked : candidates edge 4 337 1297 root65Tail7 winner.bound = root65Candidates6 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
theorem root65Step6 : scan edge (search edge 3) 4 [337] 337 root65Tail6 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail7 winner := by
  rw [root65Tail6, scan, if_neg (by decide), root65Candidates6_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail7
    (if 3 ≤ root65Candidates6.length then search edge 3 [337, 1297] root65Candidates6 1634 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root65Candidates7_checked : candidates edge 4 337 1867 root65Tail8 winner.bound = root65Candidates7 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
theorem root65Step7 : scan edge (search edge 3) 4 [337] 337 root65Tail7 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail8 winner := by
  rw [root65Tail7, scan, if_neg (by decide), root65Candidates7_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail8
    (if 3 ≤ root65Candidates7.length then search edge 3 [337, 1867] root65Candidates7 2204 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates8 : List Nat := [3169]
set_option maxRecDepth 8192 in
theorem root65Candidates8_checked : candidates edge 4 337 2371 root65Tail9 winner.bound = root65Candidates8 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root65Step8 : scan edge (search edge 3) 4 [337] 337 root65Tail8 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail9 winner := by
  rw [root65Tail8, scan, if_neg (by decide), root65Candidates8_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail9
    (if 3 ≤ root65Candidates8.length then search edge 3 [337, 2371] root65Candidates8 2708 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates9 : List Nat := [12967, 13963]
set_option maxRecDepth 8192 in
theorem root65Candidates9_checked : candidates edge 4 337 2797 root65Tail10 winner.bound = root65Candidates9 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root65Step9 : scan edge (search edge 3) 4 [337] 337 root65Tail9 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail10 winner := by
  rw [root65Tail9, scan, if_neg (by decide), root65Candidates9_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail10
    (if 3 ≤ root65Candidates9.length then search edge 3 [337, 2797] root65Candidates9 3134 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates10 : List Nat := [15559]
set_option maxRecDepth 8192 in
theorem root65Candidates10_checked : candidates edge 4 337 3169 root65Tail11 winner.bound = root65Candidates10 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
theorem root65Step10 : scan edge (search edge 3) 4 [337] 337 root65Tail10 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail11 winner := by
  rw [root65Tail10, scan, if_neg (by decide), root65Candidates10_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail11
    (if 3 ≤ root65Candidates10.length then search edge 3 [337, 3169] root65Candidates10 3506 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates11 : List Nat := [13759]
set_option maxRecDepth 8192 in
theorem root65Candidates11_checked : candidates edge 4 337 3217 root65Tail12 winner.bound = root65Candidates11 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root65Step11 : scan edge (search edge 3) 4 [337] 337 root65Tail11 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail12 winner := by
  rw [root65Tail11, scan, if_neg (by decide), root65Candidates11_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail12
    (if 3 ≤ root65Candidates11.length then search edge 3 [337, 3217] root65Candidates11 3554 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates12 : List Nat := [15307, 16273, 17449]
set_option maxRecDepth 8192 in
theorem root65Candidates12_checked : candidates edge 4 337 3343 root65Tail13 winner.bound = root65Candidates12 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root65Child12 : search edge 3 [337, 3343] root65Candidates12 3680 winner = winner := by decide
theorem root65Step12 : scan edge (search edge 3) 4 [337] 337 root65Tail12 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail13 winner := by
  rw [root65Tail12, scan, if_neg (by decide), root65Candidates12_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail13
    (if 3 ≤ root65Candidates12.length then search edge 3 [337, 3343] root65Candidates12 3680 winner else winner) = _
  rw [if_pos (by decide), root65Child12]
def root65Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root65Candidates13_checked : candidates edge 4 337 3499 root65Tail14 winner.bound = root65Candidates13 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root65Step13 : scan edge (search edge 3) 4 [337] 337 root65Tail13 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail14 winner := by
  rw [root65Tail13, scan, if_neg (by decide), root65Candidates13_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail14
    (if 3 ≤ root65Candidates13.length then search edge 3 [337, 3499] root65Candidates13 3836 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root65Candidates14_checked : candidates edge 4 337 4507 root65Tail15 winner.bound = root65Candidates14 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root65Step14 : scan edge (search edge 3) 4 [337] 337 root65Tail14 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail15 winner := by
  rw [root65Tail14, scan, if_neg (by decide), root65Candidates14_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail15
    (if 3 ≤ root65Candidates14.length then search edge 3 [337, 4507] root65Candidates14 4844 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates15 : List Nat := [15559, 19597]
set_option maxRecDepth 8192 in
theorem root65Candidates15_checked : candidates edge 4 337 4729 root65Tail16 winner.bound = root65Candidates15 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
theorem root65Step15 : scan edge (search edge 3) 4 [337] 337 root65Tail15 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail16 winner := by
  rw [root65Tail15, scan, if_neg (by decide), root65Candidates15_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail16
    (if 3 ≤ root65Candidates15.length then search edge 3 [337, 4729] root65Candidates15 5066 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root65Candidates16_checked : candidates edge 4 337 5431 root65Tail17 winner.bound = root65Candidates16 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root65Step16 : scan edge (search edge 3) 4 [337] 337 root65Tail16 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail17 winner := by
  rw [root65Tail16, scan, if_neg (by decide), root65Candidates16_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail17
    (if 3 ≤ root65Candidates16.length then search edge 3 [337, 5431] root65Candidates16 5768 winner else winner) = _
  rw [if_neg (by decide)]
def root65Candidates17 : List Nat := [7723, 16987]
set_option maxRecDepth 8192 in
theorem root65Candidates17_checked : candidates edge 4 337 5857 root65Tail18 winner.bound = root65Candidates17 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root65Step17 : scan edge (search edge 3) 4 [337] 337 root65Tail17 winner =
    scan edge (search edge 3) 4 [337] 337 root65Tail18 winner := by
  rw [root65Tail17, scan, if_neg (by decide), root65Candidates17_checked]
  change scan edge (search edge 3) 4 [337] 337 root65Tail18
    (if 3 ≤ root65Candidates17.length then search edge 3 [337, 5857] root65Candidates17 6194 winner else winner) = _
  rw [if_neg (by decide)]
theorem root65Scan18 : scan edge (search edge 3) 4 [337] 337 root65Tail18 winner = winner := by rw [root65Tail18, scan, if_pos (by decide)]
theorem root65Scan17 : scan edge (search edge 3) 4 [337] 337 root65Tail17 winner = winner :=
  (root65Step17).trans root65Scan18
theorem root65Scan16 : scan edge (search edge 3) 4 [337] 337 root65Tail16 winner = winner :=
  (root65Step16).trans root65Scan17
theorem root65Scan15 : scan edge (search edge 3) 4 [337] 337 root65Tail15 winner = winner :=
  (root65Step15).trans root65Scan16
theorem root65Scan14 : scan edge (search edge 3) 4 [337] 337 root65Tail14 winner = winner :=
  (root65Step14).trans root65Scan15
theorem root65Scan13 : scan edge (search edge 3) 4 [337] 337 root65Tail13 winner = winner :=
  (root65Step13).trans root65Scan14
theorem root65Scan12 : scan edge (search edge 3) 4 [337] 337 root65Tail12 winner = winner :=
  (root65Step12).trans root65Scan13
theorem root65Scan11 : scan edge (search edge 3) 4 [337] 337 root65Tail11 winner = winner :=
  (root65Step11).trans root65Scan12
theorem root65Scan10 : scan edge (search edge 3) 4 [337] 337 root65Tail10 winner = winner :=
  (root65Step10).trans root65Scan11
theorem root65Scan9 : scan edge (search edge 3) 4 [337] 337 root65Tail9 winner = winner :=
  (root65Step9).trans root65Scan10
theorem root65Scan8 : scan edge (search edge 3) 4 [337] 337 root65Tail8 winner = winner :=
  (root65Step8).trans root65Scan9
theorem root65Scan7 : scan edge (search edge 3) 4 [337] 337 root65Tail7 winner = winner :=
  (root65Step7).trans root65Scan8
theorem root65Scan6 : scan edge (search edge 3) 4 [337] 337 root65Tail6 winner = winner :=
  (root65Step6).trans root65Scan7
theorem root65Scan5 : scan edge (search edge 3) 4 [337] 337 root65Tail5 winner = winner :=
  (root65Step5).trans root65Scan6
theorem root65Scan4 : scan edge (search edge 3) 4 [337] 337 root65Tail4 winner = winner :=
  (root65Step4).trans root65Scan5
theorem root65Scan3 : scan edge (search edge 3) 4 [337] 337 root65Tail3 winner = winner :=
  (root65Step3).trans root65Scan4
theorem root65Scan2 : scan edge (search edge 3) 4 [337] 337 root65Tail2 winner = winner :=
  (root65Step2).trans root65Scan3
theorem root65Scan1 : scan edge (search edge 3) 4 [337] 337 root65Tail1 winner = winner :=
  (root65Step1).trans root65Scan2
theorem root65Scan0 : scan edge (search edge 3) 4 [337] 337 root65Tail0 winner = winner :=
  (root65Step0).trans root65Scan1
set_option maxRecDepth 4096 in
theorem root65Branch : search edge 4 [337] root65Candidates 337 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [337] 337 root65Tail0 winner = winner
  exact root65Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
