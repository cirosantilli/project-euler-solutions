import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root129Candidates : List Nat := Data.row743
set_option maxRecDepth 8192 in
theorem root129Candidates_checked : candidates edge 5 0 743 tail130 winner.bound = root129Candidates := by
  apply candidates_eq_row edge 5 0 743 tail130 root129Candidates winner.bound tail130_ordered
  · decide
  · decide
  · funext q
    rfl
def root129Tail61 : List Nat := []
def root129Tail60 : List Nat := 24953 :: root129Tail61
def root129Tail59 : List Nat := 24821 :: root129Tail60
def root129Tail58 : List Nat := 24443 :: root129Tail59
def root129Tail57 : List Nat := 23663 :: root129Tail58
def root129Tail56 : List Nat := 23369 :: root129Tail57
def root129Tail55 : List Nat := 23357 :: root129Tail56
def root129Tail54 : List Nat := 23057 :: root129Tail55
def root129Tail53 : List Nat := 22697 :: root129Tail54
def root129Tail52 : List Nat := 22571 :: root129Tail53
def root129Tail51 : List Nat := 22481 :: root129Tail52
def root129Tail50 : List Nat := 22397 :: root129Tail51
def root129Tail49 : List Nat := 22277 :: root129Tail50
def root129Tail48 : List Nat := 22133 :: root129Tail49
def root129Tail47 : List Nat := 22013 :: root129Tail48
def root129Tail46 : List Nat := 20759 :: root129Tail47
def root129Tail45 : List Nat := 19997 :: root129Tail46
def root129Tail44 : List Nat := 19841 :: root129Tail45
def root129Tail43 : List Nat := 18869 :: root129Tail44
def root129Tail42 : List Nat := 18731 :: root129Tail43
def root129Tail41 : List Nat := 18119 :: root129Tail42
def root129Tail40 : List Nat := 17729 :: root129Tail41
def root129Tail39 : List Nat := 17681 :: root129Tail40
def root129Tail38 : List Nat := 17657 :: root129Tail39
def root129Tail37 : List Nat := 16901 :: root129Tail38
def root129Tail36 : List Nat := 16139 :: root129Tail37
def root129Tail35 : List Nat := 16007 :: root129Tail36
def root129Tail34 : List Nat := 15803 :: root129Tail35
def root129Tail33 : List Nat := 14243 :: root129Tail34
def root129Tail32 : List Nat := 14087 :: root129Tail33
def root129Tail31 : List Nat := 13907 :: root129Tail32
def root129Tail30 : List Nat := 13751 :: root129Tail31
def root129Tail29 : List Nat := 13121 :: root129Tail30
def root129Tail28 : List Nat := 12569 :: root129Tail29
def root129Tail27 : List Nat := 11927 :: root129Tail28
def root129Tail26 : List Nat := 11093 :: root129Tail27
def root129Tail25 : List Nat := 11057 :: root129Tail26
def root129Tail24 : List Nat := 10589 :: root129Tail25
def root129Tail23 : List Nat := 9749 :: root129Tail24
def root129Tail22 : List Nat := 9623 :: root129Tail23
def root129Tail21 : List Nat := 8693 :: root129Tail22
def root129Tail20 : List Nat := 8447 :: root129Tail21
def root129Tail19 : List Nat := 8117 :: root129Tail20
def root129Tail18 : List Nat := 6323 :: root129Tail19
def root129Tail17 : List Nat := 6113 :: root129Tail18
def root129Tail16 : List Nat := 5591 :: root129Tail17
def root129Tail15 : List Nat := 5081 :: root129Tail16
def root129Tail14 : List Nat := 4283 :: root129Tail15
def root129Tail13 : List Nat := 4001 :: root129Tail14
def root129Tail12 : List Nat := 3527 :: root129Tail13
def root129Tail11 : List Nat := 3359 :: root129Tail12
def root129Tail10 : List Nat := 2753 :: root129Tail11
def root129Tail9 : List Nat := 2309 :: root129Tail10
def root129Tail8 : List Nat := 2273 :: root129Tail9
def root129Tail7 : List Nat := 2099 :: root129Tail8
def root129Tail6 : List Nat := 1889 :: root129Tail7
def root129Tail5 : List Nat := 1871 :: root129Tail6
def root129Tail4 : List Nat := 1493 :: root129Tail5
def root129Tail3 : List Nat := 1481 :: root129Tail4
def root129Tail2 : List Nat := 1097 :: root129Tail3
def root129Tail1 : List Nat := 947 :: root129Tail2
def root129Tail0 : List Nat := 881 :: root129Tail1
def root129Candidates0 : List Nat := [5591, 17657, 22277]
set_option maxRecDepth 8192 in
theorem root129Candidates0_checked : candidates edge 4 743 881 root129Tail1 winner.bound = root129Candidates0 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root129Child0 : search edge 3 [743, 881] root129Candidates0 1624 winner = winner := by decide
theorem root129Step0 : scan edge (search edge 3) 4 [743] 743 root129Tail0 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail1 winner := by
  rw [root129Tail0, scan, if_neg (by decide), root129Candidates0_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail1
    (if 3 ≤ root129Candidates0.length then search edge 3 [743, 881] root129Candidates0 1624 winner else winner) = _
  rw [if_pos (by decide), root129Child0]
def root129Candidates1 : List Nat := [1097, 16901, 17729]
set_option maxRecDepth 8192 in
theorem root129Candidates1_checked : candidates edge 4 743 947 root129Tail2 winner.bound = root129Candidates1 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root129Child1 : search edge 3 [743, 947] root129Candidates1 1690 winner = winner := by decide
theorem root129Step1 : scan edge (search edge 3) 4 [743] 743 root129Tail1 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail2 winner := by
  rw [root129Tail1, scan, if_neg (by decide), root129Candidates1_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail2
    (if 3 ≤ root129Candidates1.length then search edge 3 [743, 947] root129Candidates1 1690 winner else winner) = _
  rw [if_pos (by decide), root129Child1]
def root129Candidates2 : List Nat := [4283, 6113, 6323]
set_option maxRecDepth 8192 in
theorem root129Candidates2_checked : candidates edge 4 743 1097 root129Tail3 winner.bound = root129Candidates2 := by
  unfold candidates
  have row : edge 1097 = (SortedNatTree.contains FastRows.row1097) := by
    change Data.row1097.contains = (SortedNatTree.contains FastRows.row1097)
    exact FastRows.row1097_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root129Child2 : search edge 3 [743, 1097] root129Candidates2 1840 winner = winner := by decide
theorem root129Step2 : scan edge (search edge 3) 4 [743] 743 root129Tail2 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail3 winner := by
  rw [root129Tail2, scan, if_neg (by decide), root129Candidates2_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail3
    (if 3 ≤ root129Candidates2.length then search edge 3 [743, 1097] root129Candidates2 1840 winner else winner) = _
  rw [if_pos (by decide), root129Child2]
def root129Candidates3 : List Nat := [13907, 22571]
set_option maxRecDepth 8192 in
theorem root129Candidates3_checked : candidates edge 4 743 1481 root129Tail4 winner.bound = root129Candidates3 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
theorem root129Step3 : scan edge (search edge 3) 4 [743] 743 root129Tail3 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail4 winner := by
  rw [root129Tail3, scan, if_neg (by decide), root129Candidates3_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail4
    (if 3 ≤ root129Candidates3.length then search edge 3 [743, 1481] root129Candidates3 2224 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root129Candidates4_checked : candidates edge 4 743 1493 root129Tail5 winner.bound = root129Candidates4 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root129Step4 : scan edge (search edge 3) 4 [743] 743 root129Tail4 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail5 winner := by
  rw [root129Tail4, scan, if_neg (by decide), root129Candidates4_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail5
    (if 3 ≤ root129Candidates4.length then search edge 3 [743, 1493] root129Candidates4 2236 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates5 : List Nat := [6323, 11927, 17729]
set_option maxRecDepth 8192 in
theorem root129Candidates5_checked : candidates edge 4 743 1871 root129Tail6 winner.bound = root129Candidates5 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root129Child5 : search edge 3 [743, 1871] root129Candidates5 2614 winner = winner := by decide
theorem root129Step5 : scan edge (search edge 3) 4 [743] 743 root129Tail5 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail6 winner := by
  rw [root129Tail5, scan, if_neg (by decide), root129Candidates5_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail6
    (if 3 ≤ root129Candidates5.length then search edge 3 [743, 1871] root129Candidates5 2614 winner else winner) = _
  rw [if_pos (by decide), root129Child5]
def root129Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root129Candidates6_checked : candidates edge 4 743 1889 root129Tail7 winner.bound = root129Candidates6 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root129Step6 : scan edge (search edge 3) 4 [743] 743 root129Tail6 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail7 winner := by
  rw [root129Tail6, scan, if_neg (by decide), root129Candidates6_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail7
    (if 3 ≤ root129Candidates6.length then search edge 3 [743, 1889] root129Candidates6 2632 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates7 : List Nat := [3527]
set_option maxRecDepth 8192 in
theorem root129Candidates7_checked : candidates edge 4 743 2099 root129Tail8 winner.bound = root129Candidates7 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
theorem root129Step7 : scan edge (search edge 3) 4 [743] 743 root129Tail7 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail8 winner := by
  rw [root129Tail7, scan, if_neg (by decide), root129Candidates7_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail8
    (if 3 ≤ root129Candidates7.length then search edge 3 [743, 2099] root129Candidates7 2842 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates8 : List Nat := [19997]
set_option maxRecDepth 8192 in
theorem root129Candidates8_checked : candidates edge 4 743 2273 root129Tail9 winner.bound = root129Candidates8 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
theorem root129Step8 : scan edge (search edge 3) 4 [743] 743 root129Tail8 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail9 winner := by
  rw [root129Tail8, scan, if_neg (by decide), root129Candidates8_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail9
    (if 3 ≤ root129Candidates8.length then search edge 3 [743, 2273] root129Candidates8 3016 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates9 : List Nat := [5081, 18119, 19841]
set_option maxRecDepth 8192 in
theorem root129Candidates9_checked : candidates edge 4 743 2309 root129Tail10 winner.bound = root129Candidates9 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root129Child9 : search edge 3 [743, 2309] root129Candidates9 3052 winner = winner := by decide
theorem root129Step9 : scan edge (search edge 3) 4 [743] 743 root129Tail9 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail10 winner := by
  rw [root129Tail9, scan, if_neg (by decide), root129Candidates9_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail10
    (if 3 ≤ root129Candidates9.length then search edge 3 [743, 2309] root129Candidates9 3052 winner else winner) = _
  rw [if_pos (by decide), root129Child9]
def root129Candidates10 : List Nat := [14243]
set_option maxRecDepth 8192 in
theorem root129Candidates10_checked : candidates edge 4 743 2753 root129Tail11 winner.bound = root129Candidates10 := by
  unfold candidates
  have row : edge 2753 = (SortedNatTree.contains FastRows.row2753) := by
    change Data.row2753.contains = (SortedNatTree.contains FastRows.row2753)
    exact FastRows.row2753_checked.symm
  rw [row]
  decide
theorem root129Step10 : scan edge (search edge 3) 4 [743] 743 root129Tail10 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail11 winner := by
  rw [root129Tail10, scan, if_neg (by decide), root129Candidates10_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail11
    (if 3 ≤ root129Candidates10.length then search edge 3 [743, 2753] root129Candidates10 3496 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates11 : List Nat := [8447, 16901]
set_option maxRecDepth 8192 in
theorem root129Candidates11_checked : candidates edge 4 743 3359 root129Tail12 winner.bound = root129Candidates11 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root129Step11 : scan edge (search edge 3) 4 [743] 743 root129Tail11 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail12 winner := by
  rw [root129Tail11, scan, if_neg (by decide), root129Candidates11_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail12
    (if 3 ≤ root129Candidates11.length then search edge 3 [743, 3359] root129Candidates11 4102 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root129Candidates12_checked : candidates edge 4 743 3527 root129Tail13 winner.bound = root129Candidates12 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root129Step12 : scan edge (search edge 3) 4 [743] 743 root129Tail12 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail13 winner := by
  rw [root129Tail12, scan, if_neg (by decide), root129Candidates12_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail13
    (if 3 ≤ root129Candidates12.length then search edge 3 [743, 3527] root129Candidates12 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates13 : List Nat := [6323, 12569, 19997, 20759]
set_option maxRecDepth 8192 in
theorem root129Candidates13_checked : candidates edge 4 743 4001 root129Tail14 winner.bound = root129Candidates13 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root129Child13 : search edge 3 [743, 4001] root129Candidates13 4744 winner = winner := by decide
theorem root129Step13 : scan edge (search edge 3) 4 [743] 743 root129Tail13 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail14 winner := by
  rw [root129Tail13, scan, if_neg (by decide), root129Candidates13_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail14
    (if 3 ≤ root129Candidates13.length then search edge 3 [743, 4001] root129Candidates13 4744 winner else winner) = _
  rw [if_pos (by decide), root129Child13]
def root129Candidates14 : List Nat := [8447, 16901]
set_option maxRecDepth 8192 in
theorem root129Candidates14_checked : candidates edge 4 743 4283 root129Tail15 winner.bound = root129Candidates14 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root129Step14 : scan edge (search edge 3) 4 [743] 743 root129Tail14 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail15 winner := by
  rw [root129Tail14, scan, if_neg (by decide), root129Candidates14_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail15
    (if 3 ≤ root129Candidates14.length then search edge 3 [743, 4283] root129Candidates14 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates15 : List Nat := [17657]
set_option maxRecDepth 8192 in
theorem root129Candidates15_checked : candidates edge 4 743 5081 root129Tail16 winner.bound = root129Candidates15 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root129Step15 : scan edge (search edge 3) 4 [743] 743 root129Tail15 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail16 winner := by
  rw [root129Tail15, scan, if_neg (by decide), root129Candidates15_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail16
    (if 3 ≤ root129Candidates15.length then search edge 3 [743, 5081] root129Candidates15 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root129Candidates16_checked : candidates edge 4 743 5591 root129Tail17 winner.bound = root129Candidates16 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root129Step16 : scan edge (search edge 3) 4 [743] 743 root129Tail16 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail17 winner := by
  rw [root129Tail16, scan, if_neg (by decide), root129Candidates16_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail17
    (if 3 ≤ root129Candidates16.length then search edge 3 [743, 5591] root129Candidates16 6334 winner else winner) = _
  rw [if_neg (by decide)]
def root129Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root129Candidates17_checked : candidates edge 4 743 6113 root129Tail18 winner.bound = root129Candidates17 := by
  unfold candidates
  have row : edge 6113 = (SortedNatTree.contains FastRows.row6113) := by
    change Data.row6113.contains = (SortedNatTree.contains FastRows.row6113)
    exact FastRows.row6113_checked.symm
  rw [row]
  decide
theorem root129Step17 : scan edge (search edge 3) 4 [743] 743 root129Tail17 winner =
    scan edge (search edge 3) 4 [743] 743 root129Tail18 winner := by
  rw [root129Tail17, scan, if_neg (by decide), root129Candidates17_checked]
  change scan edge (search edge 3) 4 [743] 743 root129Tail18
    (if 3 ≤ root129Candidates17.length then search edge 3 [743, 6113] root129Candidates17 6856 winner else winner) = _
  rw [if_neg (by decide)]
theorem root129Scan18 : scan edge (search edge 3) 4 [743] 743 root129Tail18 winner = winner := by rw [root129Tail18, scan, if_pos (by decide)]
theorem root129Scan17 : scan edge (search edge 3) 4 [743] 743 root129Tail17 winner = winner :=
  (root129Step17).trans root129Scan18
theorem root129Scan16 : scan edge (search edge 3) 4 [743] 743 root129Tail16 winner = winner :=
  (root129Step16).trans root129Scan17
theorem root129Scan15 : scan edge (search edge 3) 4 [743] 743 root129Tail15 winner = winner :=
  (root129Step15).trans root129Scan16
theorem root129Scan14 : scan edge (search edge 3) 4 [743] 743 root129Tail14 winner = winner :=
  (root129Step14).trans root129Scan15
theorem root129Scan13 : scan edge (search edge 3) 4 [743] 743 root129Tail13 winner = winner :=
  (root129Step13).trans root129Scan14
theorem root129Scan12 : scan edge (search edge 3) 4 [743] 743 root129Tail12 winner = winner :=
  (root129Step12).trans root129Scan13
theorem root129Scan11 : scan edge (search edge 3) 4 [743] 743 root129Tail11 winner = winner :=
  (root129Step11).trans root129Scan12
theorem root129Scan10 : scan edge (search edge 3) 4 [743] 743 root129Tail10 winner = winner :=
  (root129Step10).trans root129Scan11
theorem root129Scan9 : scan edge (search edge 3) 4 [743] 743 root129Tail9 winner = winner :=
  (root129Step9).trans root129Scan10
theorem root129Scan8 : scan edge (search edge 3) 4 [743] 743 root129Tail8 winner = winner :=
  (root129Step8).trans root129Scan9
theorem root129Scan7 : scan edge (search edge 3) 4 [743] 743 root129Tail7 winner = winner :=
  (root129Step7).trans root129Scan8
theorem root129Scan6 : scan edge (search edge 3) 4 [743] 743 root129Tail6 winner = winner :=
  (root129Step6).trans root129Scan7
theorem root129Scan5 : scan edge (search edge 3) 4 [743] 743 root129Tail5 winner = winner :=
  (root129Step5).trans root129Scan6
theorem root129Scan4 : scan edge (search edge 3) 4 [743] 743 root129Tail4 winner = winner :=
  (root129Step4).trans root129Scan5
theorem root129Scan3 : scan edge (search edge 3) 4 [743] 743 root129Tail3 winner = winner :=
  (root129Step3).trans root129Scan4
theorem root129Scan2 : scan edge (search edge 3) 4 [743] 743 root129Tail2 winner = winner :=
  (root129Step2).trans root129Scan3
theorem root129Scan1 : scan edge (search edge 3) 4 [743] 743 root129Tail1 winner = winner :=
  (root129Step1).trans root129Scan2
theorem root129Scan0 : scan edge (search edge 3) 4 [743] 743 root129Tail0 winner = winner :=
  (root129Step0).trans root129Scan1
set_option maxRecDepth 4096 in
theorem root129Branch : search edge 4 [743] root129Candidates 743 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [743] 743 root129Tail0 winner = winner
  exact root129Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
