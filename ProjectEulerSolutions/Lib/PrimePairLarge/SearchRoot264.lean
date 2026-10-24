import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root264Candidates : List Nat := Data.row1709
set_option maxRecDepth 8192 in
theorem root264Candidates_checked : candidates edge 5 0 1709 tail265 winner.bound = root264Candidates := by
  apply candidates_eq_row edge 5 0 1709 tail265 root264Candidates winner.bound tail265_ordered
  · decide
  · decide
  · funext q
    rfl
def root264Tail56 : List Nat := []
def root264Tail55 : List Nat := 24107 :: root264Tail56
def root264Tail54 : List Nat := 23201 :: root264Tail55
def root264Tail53 : List Nat := 22751 :: root264Tail54
def root264Tail52 : List Nat := 22541 :: root264Tail53
def root264Tail51 : List Nat := 22133 :: root264Tail52
def root264Tail50 : List Nat := 22091 :: root264Tail51
def root264Tail49 : List Nat := 21977 :: root264Tail50
def root264Tail48 : List Nat := 21323 :: root264Tail49
def root264Tail47 : List Nat := 20117 :: root264Tail48
def root264Tail46 : List Nat := 19949 :: root264Tail47
def root264Tail45 : List Nat := 19559 :: root264Tail46
def root264Tail44 : List Nat := 19379 :: root264Tail45
def root264Tail43 : List Nat := 19181 :: root264Tail44
def root264Tail42 : List Nat := 15971 :: root264Tail43
def root264Tail41 : List Nat := 15137 :: root264Tail42
def root264Tail40 : List Nat := 15131 :: root264Tail41
def root264Tail39 : List Nat := 14699 :: root264Tail40
def root264Tail38 : List Nat := 14591 :: root264Tail39
def root264Tail37 : List Nat := 14543 :: root264Tail38
def root264Tail36 : List Nat := 13913 :: root264Tail37
def root264Tail35 : List Nat := 12671 :: root264Tail36
def root264Tail34 : List Nat := 12281 :: root264Tail35
def root264Tail33 : List Nat := 11003 :: root264Tail34
def root264Tail32 : List Nat := 10139 :: root264Tail33
def root264Tail31 : List Nat := 9923 :: root264Tail32
def root264Tail30 : List Nat := 9521 :: root264Tail31
def root264Tail29 : List Nat := 9491 :: root264Tail30
def root264Tail28 : List Nat := 9431 :: root264Tail29
def root264Tail27 : List Nat := 9209 :: root264Tail28
def root264Tail26 : List Nat := 8831 :: root264Tail27
def root264Tail25 : List Nat := 8819 :: root264Tail26
def root264Tail24 : List Nat := 8297 :: root264Tail25
def root264Tail23 : List Nat := 8231 :: root264Tail24
def root264Tail22 : List Nat := 7877 :: root264Tail23
def root264Tail21 : List Nat := 7823 :: root264Tail22
def root264Tail20 : List Nat := 7247 :: root264Tail21
def root264Tail19 : List Nat := 7001 :: root264Tail20
def root264Tail18 : List Nat := 6869 :: root264Tail19
def root264Tail17 : List Nat := 6857 :: root264Tail18
def root264Tail16 : List Nat := 6551 :: root264Tail17
def root264Tail15 : List Nat := 4523 :: root264Tail16
def root264Tail14 : List Nat := 4397 :: root264Tail15
def root264Tail13 : List Nat := 4283 :: root264Tail14
def root264Tail12 : List Nat := 4013 :: root264Tail13
def root264Tail11 : List Nat := 4001 :: root264Tail12
def root264Tail10 : List Nat := 3917 :: root264Tail11
def root264Tail9 : List Nat := 3389 :: root264Tail10
def root264Tail8 : List Nat := 3299 :: root264Tail9
def root264Tail7 : List Nat := 3137 :: root264Tail8
def root264Tail6 : List Nat := 2843 :: root264Tail7
def root264Tail5 : List Nat := 2591 :: root264Tail6
def root264Tail4 : List Nat := 2423 :: root264Tail5
def root264Tail3 : List Nat := 2393 :: root264Tail4
def root264Tail2 : List Nat := 2333 :: root264Tail3
def root264Tail1 : List Nat := 2213 :: root264Tail2
def root264Tail0 : List Nat := 1913 :: root264Tail1
def root264Candidates0 : List Nat := [2213, 2843, 8297, 22091]
set_option maxRecDepth 8192 in
theorem root264Candidates0_checked : candidates edge 4 1709 1913 root264Tail1 winner.bound = root264Candidates0 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root264Child0 : search edge 3 [1709, 1913] root264Candidates0 3622 winner = winner := by decide
theorem root264Step0 : scan edge (search edge 3) 4 [1709] 1709 root264Tail0 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail1 winner := by
  rw [root264Tail0, scan, if_neg (by decide), root264Candidates0_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail1
    (if 3 ≤ root264Candidates0.length then search edge 3 [1709, 1913] root264Candidates0 3622 winner else winner) = _
  rw [if_pos (by decide), root264Child0]
def root264Candidates1 : List Nat := [12671, 19181, 21323]
set_option maxRecDepth 8192 in
theorem root264Candidates1_checked : candidates edge 4 1709 2213 root264Tail2 winner.bound = root264Candidates1 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root264Child1 : search edge 3 [1709, 2213] root264Candidates1 3922 winner = winner := by decide
theorem root264Step1 : scan edge (search edge 3) 4 [1709] 1709 root264Tail1 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail2 winner := by
  rw [root264Tail1, scan, if_neg (by decide), root264Candidates1_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail2
    (if 3 ≤ root264Candidates1.length then search edge 3 [1709, 2213] root264Candidates1 3922 winner else winner) = _
  rw [if_pos (by decide), root264Child1]
def root264Candidates2 : List Nat := [2423, 2843, 3137, 8831, 19949]
set_option maxRecDepth 8192 in
theorem root264Candidates2_checked : candidates edge 4 1709 2333 root264Tail3 winner.bound = root264Candidates2 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root264Child2 : search edge 3 [1709, 2333] root264Candidates2 4042 winner = winner := by decide
theorem root264Step2 : scan edge (search edge 3) 4 [1709] 1709 root264Tail2 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail3 winner := by
  rw [root264Tail2, scan, if_neg (by decide), root264Candidates2_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail3
    (if 3 ≤ root264Candidates2.length then search edge 3 [1709, 2333] root264Candidates2 4042 winner else winner) = _
  rw [if_pos (by decide), root264Child2]
def root264Candidates3 : List Nat := [2423, 9491, 11003]
set_option maxRecDepth 8192 in
theorem root264Candidates3_checked : candidates edge 4 1709 2393 root264Tail4 winner.bound = root264Candidates3 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root264Child3 : search edge 3 [1709, 2393] root264Candidates3 4102 winner = winner := by decide
theorem root264Step3 : scan edge (search edge 3) 4 [1709] 1709 root264Tail3 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail4 winner := by
  rw [root264Tail3, scan, if_neg (by decide), root264Candidates3_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail4
    (if 3 ≤ root264Candidates3.length then search edge 3 [1709, 2393] root264Candidates3 4102 winner else winner) = _
  rw [if_pos (by decide), root264Child3]
def root264Candidates4 : List Nat := [9209, 13913]
set_option maxRecDepth 8192 in
theorem root264Candidates4_checked : candidates edge 4 1709 2423 root264Tail5 winner.bound = root264Candidates4 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root264Step4 : scan edge (search edge 3) 4 [1709] 1709 root264Tail4 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail5 winner := by
  rw [root264Tail4, scan, if_neg (by decide), root264Candidates4_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail5
    (if 3 ≤ root264Candidates4.length then search edge 3 [1709, 2423] root264Candidates4 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates5 : List Nat := [4283, 8231]
set_option maxRecDepth 8192 in
theorem root264Candidates5_checked : candidates edge 4 1709 2591 root264Tail6 winner.bound = root264Candidates5 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root264Step5 : scan edge (search edge 3) 4 [1709] 1709 root264Tail5 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail6 winner := by
  rw [root264Tail5, scan, if_neg (by decide), root264Candidates5_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail6
    (if 3 ≤ root264Candidates5.length then search edge 3 [1709, 2591] root264Candidates5 4300 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates6 : List Nat := [3389, 8831, 10139, 15137]
set_option maxRecDepth 8192 in
theorem root264Candidates6_checked : candidates edge 4 1709 2843 root264Tail7 winner.bound = root264Candidates6 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root264Child6 : search edge 3 [1709, 2843] root264Candidates6 4552 winner = winner := by decide
theorem root264Step6 : scan edge (search edge 3) 4 [1709] 1709 root264Tail6 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail7 winner := by
  rw [root264Tail6, scan, if_neg (by decide), root264Candidates6_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail7
    (if 3 ≤ root264Candidates6.length then search edge 3 [1709, 2843] root264Candidates6 4552 winner else winner) = _
  rw [if_pos (by decide), root264Child6]
def root264Candidates7 : List Nat := [9923]
set_option maxRecDepth 8192 in
theorem root264Candidates7_checked : candidates edge 4 1709 3137 root264Tail8 winner.bound = root264Candidates7 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root264Step7 : scan edge (search edge 3) 4 [1709] 1709 root264Tail7 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail8 winner := by
  rw [root264Tail7, scan, if_neg (by decide), root264Candidates7_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail8
    (if 3 ≤ root264Candidates7.length then search edge 3 [1709, 3137] root264Candidates7 4846 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates8 : List Nat := [4523, 6869, 20117]
set_option maxRecDepth 8192 in
theorem root264Candidates8_checked : candidates edge 4 1709 3299 root264Tail9 winner.bound = root264Candidates8 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root264Child8 : search edge 3 [1709, 3299] root264Candidates8 5008 winner = winner := by decide
theorem root264Step8 : scan edge (search edge 3) 4 [1709] 1709 root264Tail8 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail9 winner := by
  rw [root264Tail8, scan, if_neg (by decide), root264Candidates8_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail9
    (if 3 ≤ root264Candidates8.length then search edge 3 [1709, 3299] root264Candidates8 5008 winner else winner) = _
  rw [if_pos (by decide), root264Child8]
def root264Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root264Candidates9_checked : candidates edge 4 1709 3389 root264Tail10 winner.bound = root264Candidates9 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
theorem root264Step9 : scan edge (search edge 3) 4 [1709] 1709 root264Tail9 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail10 winner := by
  rw [root264Tail9, scan, if_neg (by decide), root264Candidates9_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail10
    (if 3 ≤ root264Candidates9.length then search edge 3 [1709, 3389] root264Candidates9 5098 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root264Candidates10_checked : candidates edge 4 1709 3917 root264Tail11 winner.bound = root264Candidates10 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root264Step10 : scan edge (search edge 3) 4 [1709] 1709 root264Tail10 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail11 winner := by
  rw [root264Tail10, scan, if_neg (by decide), root264Candidates10_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail11
    (if 3 ≤ root264Candidates10.length then search edge 3 [1709, 3917] root264Candidates10 5626 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates11 : List Nat := [7877]
set_option maxRecDepth 8192 in
theorem root264Candidates11_checked : candidates edge 4 1709 4001 root264Tail12 winner.bound = root264Candidates11 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root264Step11 : scan edge (search edge 3) 4 [1709] 1709 root264Tail11 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail12 winner := by
  rw [root264Tail11, scan, if_neg (by decide), root264Candidates11_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail12
    (if 3 ≤ root264Candidates11.length then search edge 3 [1709, 4001] root264Candidates11 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root264Candidates12_checked : candidates edge 4 1709 4013 root264Tail13 winner.bound = root264Candidates12 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root264Step12 : scan edge (search edge 3) 4 [1709] 1709 root264Tail12 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail13 winner := by
  rw [root264Tail12, scan, if_neg (by decide), root264Candidates12_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail13
    (if 3 ≤ root264Candidates12.length then search edge 3 [1709, 4013] root264Candidates12 5722 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates13 : List Nat := [4523, 19379]
set_option maxRecDepth 8192 in
theorem root264Candidates13_checked : candidates edge 4 1709 4283 root264Tail14 winner.bound = root264Candidates13 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root264Step13 : scan edge (search edge 3) 4 [1709] 1709 root264Tail13 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail14 winner := by
  rw [root264Tail13, scan, if_neg (by decide), root264Candidates13_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail14
    (if 3 ≤ root264Candidates13.length then search edge 3 [1709, 4283] root264Candidates13 5992 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates14 : List Nat := [7823, 9431]
set_option maxRecDepth 8192 in
theorem root264Candidates14_checked : candidates edge 4 1709 4397 root264Tail15 winner.bound = root264Candidates14 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root264Step14 : scan edge (search edge 3) 4 [1709] 1709 root264Tail14 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail15 winner := by
  rw [root264Tail14, scan, if_neg (by decide), root264Candidates14_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail15
    (if 3 ≤ root264Candidates14.length then search edge 3 [1709, 4397] root264Candidates14 6106 winner else winner) = _
  rw [if_neg (by decide)]
def root264Candidates15 : List Nat := [6551]
set_option maxRecDepth 8192 in
theorem root264Candidates15_checked : candidates edge 4 1709 4523 root264Tail16 winner.bound = root264Candidates15 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root264Step15 : scan edge (search edge 3) 4 [1709] 1709 root264Tail15 winner =
    scan edge (search edge 3) 4 [1709] 1709 root264Tail16 winner := by
  rw [root264Tail15, scan, if_neg (by decide), root264Candidates15_checked]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail16
    (if 3 ≤ root264Candidates15.length then search edge 3 [1709, 4523] root264Candidates15 6232 winner else winner) = _
  rw [if_neg (by decide)]
theorem root264Scan16 : scan edge (search edge 3) 4 [1709] 1709 root264Tail16 winner = winner := by rw [root264Tail16, scan, if_pos (by decide)]
theorem root264Scan15 : scan edge (search edge 3) 4 [1709] 1709 root264Tail15 winner = winner :=
  (root264Step15).trans root264Scan16
theorem root264Scan14 : scan edge (search edge 3) 4 [1709] 1709 root264Tail14 winner = winner :=
  (root264Step14).trans root264Scan15
theorem root264Scan13 : scan edge (search edge 3) 4 [1709] 1709 root264Tail13 winner = winner :=
  (root264Step13).trans root264Scan14
theorem root264Scan12 : scan edge (search edge 3) 4 [1709] 1709 root264Tail12 winner = winner :=
  (root264Step12).trans root264Scan13
theorem root264Scan11 : scan edge (search edge 3) 4 [1709] 1709 root264Tail11 winner = winner :=
  (root264Step11).trans root264Scan12
theorem root264Scan10 : scan edge (search edge 3) 4 [1709] 1709 root264Tail10 winner = winner :=
  (root264Step10).trans root264Scan11
theorem root264Scan9 : scan edge (search edge 3) 4 [1709] 1709 root264Tail9 winner = winner :=
  (root264Step9).trans root264Scan10
theorem root264Scan8 : scan edge (search edge 3) 4 [1709] 1709 root264Tail8 winner = winner :=
  (root264Step8).trans root264Scan9
theorem root264Scan7 : scan edge (search edge 3) 4 [1709] 1709 root264Tail7 winner = winner :=
  (root264Step7).trans root264Scan8
theorem root264Scan6 : scan edge (search edge 3) 4 [1709] 1709 root264Tail6 winner = winner :=
  (root264Step6).trans root264Scan7
theorem root264Scan5 : scan edge (search edge 3) 4 [1709] 1709 root264Tail5 winner = winner :=
  (root264Step5).trans root264Scan6
theorem root264Scan4 : scan edge (search edge 3) 4 [1709] 1709 root264Tail4 winner = winner :=
  (root264Step4).trans root264Scan5
theorem root264Scan3 : scan edge (search edge 3) 4 [1709] 1709 root264Tail3 winner = winner :=
  (root264Step3).trans root264Scan4
theorem root264Scan2 : scan edge (search edge 3) 4 [1709] 1709 root264Tail2 winner = winner :=
  (root264Step2).trans root264Scan3
theorem root264Scan1 : scan edge (search edge 3) 4 [1709] 1709 root264Tail1 winner = winner :=
  (root264Step1).trans root264Scan2
theorem root264Scan0 : scan edge (search edge 3) 4 [1709] 1709 root264Tail0 winner = winner :=
  (root264Step0).trans root264Scan1
set_option maxRecDepth 4096 in
theorem root264Branch : search edge 4 [1709] root264Candidates 1709 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1709] 1709 root264Tail0 winner = winner
  exact root264Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
