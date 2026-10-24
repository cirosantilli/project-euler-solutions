import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root119Candidates : List Nat := Data.row673
set_option maxRecDepth 8192 in
theorem root119Candidates_checked : candidates edge 5 0 673 tail120 winner.bound = root119Candidates := by
  apply candidates_eq_row edge 5 0 673 tail120 root119Candidates winner.bound tail120_ordered
  · decide
  · decide
  · funext q
    rfl
def root119Tail61 : List Nat := []
def root119Tail60 : List Nat := 24919 :: root119Tail61
def root119Tail59 : List Nat := 23827 :: root119Tail60
def root119Tail58 : List Nat := 23623 :: root119Tail59
def root119Tail57 : List Nat := 23497 :: root119Tail58
def root119Tail56 : List Nat := 22921 :: root119Tail57
def root119Tail55 : List Nat := 22441 :: root119Tail56
def root119Tail54 : List Nat := 22027 :: root119Tail55
def root119Tail53 : List Nat := 20887 :: root119Tail54
def root119Tail52 : List Nat := 20347 :: root119Tail53
def root119Tail51 : List Nat := 20101 :: root119Tail52
def root119Tail50 : List Nat := 19249 :: root119Tail51
def root119Tail49 : List Nat := 19141 :: root119Tail50
def root119Tail48 : List Nat := 19009 :: root119Tail49
def root119Tail47 : List Nat := 18583 :: root119Tail48
def root119Tail46 : List Nat := 17749 :: root119Tail47
def root119Tail45 : List Nat := 17713 :: root119Tail46
def root119Tail44 : List Nat := 17623 :: root119Tail45
def root119Tail43 : List Nat := 17191 :: root119Tail44
def root119Tail42 : List Nat := 15349 :: root119Tail43
def root119Tail41 : List Nat := 15217 :: root119Tail42
def root119Tail40 : List Nat := 14983 :: root119Tail41
def root119Tail39 : List Nat := 14683 :: root119Tail40
def root119Tail38 : List Nat := 14593 :: root119Tail39
def root119Tail37 : List Nat := 14563 :: root119Tail38
def root119Tail36 : List Nat := 14437 :: root119Tail37
def root119Tail35 : List Nat := 14173 :: root119Tail36
def root119Tail34 : List Nat := 13681 :: root119Tail35
def root119Tail33 : List Nat := 13537 :: root119Tail34
def root119Tail32 : List Nat := 13291 :: root119Tail33
def root119Tail31 : List Nat := 13177 :: root119Tail32
def root119Tail30 : List Nat := 12577 :: root119Tail31
def root119Tail29 : List Nat := 12241 :: root119Tail30
def root119Tail28 : List Nat := 12211 :: root119Tail29
def root119Tail27 : List Nat := 11821 :: root119Tail28
def root119Tail26 : List Nat := 11701 :: root119Tail27
def root119Tail25 : List Nat := 11689 :: root119Tail26
def root119Tail24 : List Nat := 11527 :: root119Tail25
def root119Tail23 : List Nat := 10729 :: root119Tail24
def root119Tail22 : List Nat := 8929 :: root119Tail23
def root119Tail21 : List Nat := 8689 :: root119Tail22
def root119Tail20 : List Nat := 8089 :: root119Tail21
def root119Tail19 : List Nat := 7417 :: root119Tail20
def root119Tail18 : List Nat := 7177 :: root119Tail19
def root119Tail17 : List Nat := 6841 :: root119Tail18
def root119Tail16 : List Nat := 6271 :: root119Tail17
def root119Tail15 : List Nat := 4993 :: root119Tail16
def root119Tail14 : List Nat := 4813 :: root119Tail15
def root119Tail13 : List Nat := 4597 :: root119Tail14
def root119Tail12 : List Nat := 4297 :: root119Tail13
def root119Tail11 : List Nat := 3271 :: root119Tail12
def root119Tail10 : List Nat := 2659 :: root119Tail11
def root119Tail9 : List Nat := 2293 :: root119Tail10
def root119Tail8 : List Nat := 2281 :: root119Tail9
def root119Tail7 : List Nat := 1759 :: root119Tail8
def root119Tail6 : List Nat := 1657 :: root119Tail7
def root119Tail5 : List Nat := 1579 :: root119Tail6
def root119Tail4 : List Nat := 1489 :: root119Tail5
def root119Tail3 : List Nat := 1327 :: root119Tail4
def root119Tail2 : List Nat := 1279 :: root119Tail3
def root119Tail1 : List Nat := 1117 :: root119Tail2
def root119Tail0 : List Nat := 769 :: root119Tail1
def root119Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates0_checked : candidates edge 4 673 769 root119Tail1 winner.bound = root119Candidates0 := by
  unfold candidates
  have row : edge 769 = (SortedNatTree.contains FastRows.row769) := by
    change Data.row769.contains = (SortedNatTree.contains FastRows.row769)
    exact FastRows.row769_checked.symm
  rw [row]
  decide
theorem root119Step0 : scan edge (search edge 3) 4 [673] 673 root119Tail0 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail1 winner := by
  rw [root119Tail0, scan, if_neg (by decide), root119Candidates0_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail1
    (if 3 ≤ root119Candidates0.length then search edge 3 [673, 769] root119Candidates0 1442 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates1 : List Nat := [1327, 4993, 14593, 18583, 23497]
set_option maxRecDepth 8192 in
theorem root119Candidates1_checked : candidates edge 4 673 1117 root119Tail2 winner.bound = root119Candidates1 := by
  unfold candidates
  have row : edge 1117 = (SortedNatTree.contains FastRows.row1117) := by
    change Data.row1117.contains = (SortedNatTree.contains FastRows.row1117)
    exact FastRows.row1117_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root119Child1 : search edge 3 [673, 1117] root119Candidates1 1790 winner = winner := by decide
theorem root119Step1 : scan edge (search edge 3) 4 [673] 673 root119Tail1 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail2 winner := by
  rw [root119Tail1, scan, if_neg (by decide), root119Candidates1_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail2
    (if 3 ≤ root119Candidates1.length then search edge 3 [673, 1117] root119Candidates1 1790 winner else winner) = _
  rw [if_pos (by decide), root119Child1]
def root119Candidates2 : List Nat := [6271]
set_option maxRecDepth 8192 in
theorem root119Candidates2_checked : candidates edge 4 673 1279 root119Tail3 winner.bound = root119Candidates2 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
theorem root119Step2 : scan edge (search edge 3) 4 [673] 673 root119Tail2 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail3 winner := by
  rw [root119Tail2, scan, if_neg (by decide), root119Candidates2_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail3
    (if 3 ≤ root119Candidates2.length then search edge 3 [673, 1279] root119Candidates2 1952 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates3 : List Nat := [2293, 19141, 22921, 23497]
set_option maxRecDepth 8192 in
theorem root119Candidates3_checked : candidates edge 4 673 1327 root119Tail4 winner.bound = root119Candidates3 := by
  unfold candidates
  have row : edge 1327 = (SortedNatTree.contains FastRows.row1327) := by
    change Data.row1327.contains = (SortedNatTree.contains FastRows.row1327)
    exact FastRows.row1327_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root119Child3 : search edge 3 [673, 1327] root119Candidates3 2000 winner = winner := by decide
theorem root119Step3 : scan edge (search edge 3) 4 [673] 673 root119Tail3 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail4 winner := by
  rw [root119Tail3, scan, if_neg (by decide), root119Candidates3_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail4
    (if 3 ≤ root119Candidates3.length then search edge 3 [673, 1327] root119Candidates3 2000 winner else winner) = _
  rw [if_pos (by decide), root119Child3]
def root119Candidates4 : List Nat := [4597, 11527]
set_option maxRecDepth 8192 in
theorem root119Candidates4_checked : candidates edge 4 673 1489 root119Tail5 winner.bound = root119Candidates4 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
theorem root119Step4 : scan edge (search edge 3) 4 [673] 673 root119Tail4 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail5 winner := by
  rw [root119Tail4, scan, if_neg (by decide), root119Candidates4_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail5
    (if 3 ≤ root119Candidates4.length then search edge 3 [673, 1489] root119Candidates4 2162 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates5 : List Nat := [11701, 13177]
set_option maxRecDepth 8192 in
theorem root119Candidates5_checked : candidates edge 4 673 1579 root119Tail6 winner.bound = root119Candidates5 := by
  unfold candidates
  have row : edge 1579 = (SortedNatTree.contains FastRows.row1579) := by
    change Data.row1579.contains = (SortedNatTree.contains FastRows.row1579)
    exact FastRows.row1579_checked.symm
  rw [row]
  decide
theorem root119Step5 : scan edge (search edge 3) 4 [673] 673 root119Tail5 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail6 winner := by
  rw [root119Tail5, scan, if_neg (by decide), root119Candidates5_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail6
    (if 3 ≤ root119Candidates5.length then search edge 3 [673, 1579] root119Candidates5 2252 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates6_checked : candidates edge 4 673 1657 root119Tail7 winner.bound = root119Candidates6 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
theorem root119Step6 : scan edge (search edge 3) 4 [673] 673 root119Tail6 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail7 winner := by
  rw [root119Tail6, scan, if_neg (by decide), root119Candidates6_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail7
    (if 3 ≤ root119Candidates6.length then search edge 3 [673, 1657] root119Candidates6 2330 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates7 : List Nat := [7417]
set_option maxRecDepth 8192 in
theorem root119Candidates7_checked : candidates edge 4 673 1759 root119Tail8 winner.bound = root119Candidates7 := by
  unfold candidates
  have row : edge 1759 = (SortedNatTree.contains FastRows.row1759) := by
    change Data.row1759.contains = (SortedNatTree.contains FastRows.row1759)
    exact FastRows.row1759_checked.symm
  rw [row]
  decide
theorem root119Step7 : scan edge (search edge 3) 4 [673] 673 root119Tail7 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail8 winner := by
  rw [root119Tail7, scan, if_neg (by decide), root119Candidates7_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail8
    (if 3 ≤ root119Candidates7.length then search edge 3 [673, 1759] root119Candidates7 2432 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates8_checked : candidates edge 4 673 2281 root119Tail9 winner.bound = root119Candidates8 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root119Step8 : scan edge (search edge 3) 4 [673] 673 root119Tail8 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail9 winner := by
  rw [root119Tail8, scan, if_neg (by decide), root119Candidates8_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail9
    (if 3 ≤ root119Candidates8.length then search edge 3 [673, 2281] root119Candidates8 2954 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates9 : List Nat := [12211, 14563]
set_option maxRecDepth 8192 in
theorem root119Candidates9_checked : candidates edge 4 673 2293 root119Tail10 winner.bound = root119Candidates9 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root119Step9 : scan edge (search edge 3) 4 [673] 673 root119Tail9 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail10 winner := by
  rw [root119Tail9, scan, if_neg (by decide), root119Candidates9_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail10
    (if 3 ≤ root119Candidates9.length then search edge 3 [673, 2293] root119Candidates9 2966 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates10_checked : candidates edge 4 673 2659 root119Tail11 winner.bound = root119Candidates10 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
theorem root119Step10 : scan edge (search edge 3) 4 [673] 673 root119Tail10 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail11 winner := by
  rw [root119Tail10, scan, if_neg (by decide), root119Candidates10_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail11
    (if 3 ≤ root119Candidates10.length then search edge 3 [673, 2659] root119Candidates10 3332 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates11_checked : candidates edge 4 673 3271 root119Tail12 winner.bound = root119Candidates11 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root119Step11 : scan edge (search edge 3) 4 [673] 673 root119Tail11 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail12 winner := by
  rw [root119Tail11, scan, if_neg (by decide), root119Candidates11_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail12
    (if 3 ≤ root119Candidates11.length then search edge 3 [673, 3271] root119Candidates11 3944 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates12_checked : candidates edge 4 673 4297 root119Tail13 winner.bound = root119Candidates12 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root119Step12 : scan edge (search edge 3) 4 [673] 673 root119Tail12 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail13 winner := by
  rw [root119Tail12, scan, if_neg (by decide), root119Candidates12_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail13
    (if 3 ≤ root119Candidates12.length then search edge 3 [673, 4297] root119Candidates12 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates13_checked : candidates edge 4 673 4597 root119Tail14 winner.bound = root119Candidates13 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root119Step13 : scan edge (search edge 3) 4 [673] 673 root119Tail13 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail14 winner := by
  rw [root119Tail13, scan, if_neg (by decide), root119Candidates13_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail14
    (if 3 ≤ root119Candidates13.length then search edge 3 [673, 4597] root119Candidates13 5270 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates14 : List Nat := [12241]
set_option maxRecDepth 8192 in
theorem root119Candidates14_checked : candidates edge 4 673 4813 root119Tail15 winner.bound = root119Candidates14 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root119Step14 : scan edge (search edge 3) 4 [673] 673 root119Tail14 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail15 winner := by
  rw [root119Tail14, scan, if_neg (by decide), root119Candidates14_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail15
    (if 3 ≤ root119Candidates14.length then search edge 3 [673, 4813] root119Candidates14 5486 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates15_checked : candidates edge 4 673 4993 root119Tail16 winner.bound = root119Candidates15 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root119Step15 : scan edge (search edge 3) 4 [673] 673 root119Tail15 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail16 winner := by
  rw [root119Tail15, scan, if_neg (by decide), root119Candidates15_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail16
    (if 3 ≤ root119Candidates15.length then search edge 3 [673, 4993] root119Candidates15 5666 winner else winner) = _
  rw [if_neg (by decide)]
def root119Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root119Candidates16_checked : candidates edge 4 673 6271 root119Tail17 winner.bound = root119Candidates16 := by
  unfold candidates
  have row : edge 6271 = (SortedNatTree.contains FastRows.row6271) := by
    change Data.row6271.contains = (SortedNatTree.contains FastRows.row6271)
    exact FastRows.row6271_checked.symm
  rw [row]
  decide
theorem root119Step16 : scan edge (search edge 3) 4 [673] 673 root119Tail16 winner =
    scan edge (search edge 3) 4 [673] 673 root119Tail17 winner := by
  rw [root119Tail16, scan, if_neg (by decide), root119Candidates16_checked]
  change scan edge (search edge 3) 4 [673] 673 root119Tail17
    (if 3 ≤ root119Candidates16.length then search edge 3 [673, 6271] root119Candidates16 6944 winner else winner) = _
  rw [if_neg (by decide)]
theorem root119Scan17 : scan edge (search edge 3) 4 [673] 673 root119Tail17 winner = winner := by rw [root119Tail17, scan, if_pos (by decide)]
theorem root119Scan16 : scan edge (search edge 3) 4 [673] 673 root119Tail16 winner = winner :=
  (root119Step16).trans root119Scan17
theorem root119Scan15 : scan edge (search edge 3) 4 [673] 673 root119Tail15 winner = winner :=
  (root119Step15).trans root119Scan16
theorem root119Scan14 : scan edge (search edge 3) 4 [673] 673 root119Tail14 winner = winner :=
  (root119Step14).trans root119Scan15
theorem root119Scan13 : scan edge (search edge 3) 4 [673] 673 root119Tail13 winner = winner :=
  (root119Step13).trans root119Scan14
theorem root119Scan12 : scan edge (search edge 3) 4 [673] 673 root119Tail12 winner = winner :=
  (root119Step12).trans root119Scan13
theorem root119Scan11 : scan edge (search edge 3) 4 [673] 673 root119Tail11 winner = winner :=
  (root119Step11).trans root119Scan12
theorem root119Scan10 : scan edge (search edge 3) 4 [673] 673 root119Tail10 winner = winner :=
  (root119Step10).trans root119Scan11
theorem root119Scan9 : scan edge (search edge 3) 4 [673] 673 root119Tail9 winner = winner :=
  (root119Step9).trans root119Scan10
theorem root119Scan8 : scan edge (search edge 3) 4 [673] 673 root119Tail8 winner = winner :=
  (root119Step8).trans root119Scan9
theorem root119Scan7 : scan edge (search edge 3) 4 [673] 673 root119Tail7 winner = winner :=
  (root119Step7).trans root119Scan8
theorem root119Scan6 : scan edge (search edge 3) 4 [673] 673 root119Tail6 winner = winner :=
  (root119Step6).trans root119Scan7
theorem root119Scan5 : scan edge (search edge 3) 4 [673] 673 root119Tail5 winner = winner :=
  (root119Step5).trans root119Scan6
theorem root119Scan4 : scan edge (search edge 3) 4 [673] 673 root119Tail4 winner = winner :=
  (root119Step4).trans root119Scan5
theorem root119Scan3 : scan edge (search edge 3) 4 [673] 673 root119Tail3 winner = winner :=
  (root119Step3).trans root119Scan4
theorem root119Scan2 : scan edge (search edge 3) 4 [673] 673 root119Tail2 winner = winner :=
  (root119Step2).trans root119Scan3
theorem root119Scan1 : scan edge (search edge 3) 4 [673] 673 root119Tail1 winner = winner :=
  (root119Step1).trans root119Scan2
theorem root119Scan0 : scan edge (search edge 3) 4 [673] 673 root119Tail0 winner = winner :=
  (root119Step0).trans root119Scan1
set_option maxRecDepth 4096 in
theorem root119Branch : search edge 4 [673] root119Candidates 673 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [673] 673 root119Tail0 winner = winner
  exact root119Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
