import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root149Candidates : List Nat := Data.row881
set_option maxRecDepth 8192 in
theorem root149Candidates_checked : candidates edge 5 0 881 tail150 winner.bound = root149Candidates := by
  apply candidates_eq_row edge 5 0 881 tail150 root149Candidates winner.bound tail150_ordered
  · decide
  · decide
  · funext q
    rfl
def root149Tail68 : List Nat := []
def root149Tail67 : List Nat := 24767 :: root149Tail68
def root149Tail66 : List Nat := 24623 :: root149Tail67
def root149Tail65 : List Nat := 24251 :: root149Tail66
def root149Tail64 : List Nat := 24137 :: root149Tail65
def root149Tail63 : List Nat := 23753 :: root149Tail64
def root149Tail62 : List Nat := 22943 :: root149Tail63
def root149Tail61 : List Nat := 22907 :: root149Tail62
def root149Tail60 : List Nat := 22277 :: root149Tail61
def root149Tail59 : List Nat := 22037 :: root149Tail60
def root149Tail58 : List Nat := 19793 :: root149Tail59
def root149Tail57 : List Nat := 19709 :: root149Tail58
def root149Tail56 : List Nat := 19301 :: root149Tail57
def root149Tail55 : List Nat := 19079 :: root149Tail56
def root149Tail54 : List Nat := 18671 :: root149Tail55
def root149Tail53 : List Nat := 17657 :: root149Tail54
def root149Tail52 : List Nat := 16937 :: root149Tail53
def root149Tail51 : List Nat := 16811 :: root149Tail52
def root149Tail50 : List Nat := 16487 :: root149Tail51
def root149Tail49 : List Nat := 16103 :: root149Tail50
def root149Tail48 : List Nat := 16097 :: root149Tail49
def root149Tail47 : List Nat := 15761 :: root149Tail48
def root149Tail46 : List Nat := 15101 :: root149Tail47
def root149Tail45 : List Nat := 15017 :: root149Tail46
def root149Tail44 : List Nat := 14813 :: root149Tail45
def root149Tail43 : List Nat := 14591 :: root149Tail44
def root149Tail42 : List Nat := 13841 :: root149Tail43
def root149Tail41 : List Nat := 13721 :: root149Tail42
def root149Tail40 : List Nat := 13577 :: root149Tail41
def root149Tail39 : List Nat := 12539 :: root149Tail40
def root149Tail38 : List Nat := 12107 :: root149Tail39
def root149Tail37 : List Nat := 11351 :: root149Tail38
def root149Tail36 : List Nat := 10973 :: root149Tail37
def root149Tail35 : List Nat := 10937 :: root149Tail36
def root149Tail34 : List Nat := 10859 :: root149Tail35
def root149Tail33 : List Nat := 10739 :: root149Tail34
def root149Tail32 : List Nat := 10691 :: root149Tail33
def root149Tail31 : List Nat := 10487 :: root149Tail32
def root149Tail30 : List Nat := 9011 :: root149Tail31
def root149Tail29 : List Nat := 8297 :: root149Tail30
def root149Tail28 : List Nat := 7607 :: root149Tail29
def root149Tail27 : List Nat := 7541 :: root149Tail28
def root149Tail26 : List Nat := 7331 :: root149Tail27
def root149Tail25 : List Nat := 7253 :: root149Tail26
def root149Tail24 : List Nat := 7103 :: root149Tail25
def root149Tail23 : List Nat := 6959 :: root149Tail24
def root149Tail22 : List Nat := 6827 :: root149Tail23
def root149Tail21 : List Nat := 6791 :: root149Tail22
def root149Tail20 : List Nat := 6701 :: root149Tail21
def root149Tail19 : List Nat := 6473 :: root149Tail20
def root149Tail18 : List Nat := 6449 :: root149Tail19
def root149Tail17 : List Nat := 6053 :: root149Tail18
def root149Tail16 : List Nat := 5801 :: root149Tail17
def root149Tail15 : List Nat := 5591 :: root149Tail16
def root149Tail14 : List Nat := 5021 :: root149Tail15
def root149Tail13 : List Nat := 4931 :: root149Tail14
def root149Tail12 : List Nat := 4733 :: root149Tail13
def root149Tail11 : List Nat := 4457 :: root149Tail12
def root149Tail10 : List Nat := 4229 :: root149Tail11
def root149Tail9 : List Nat := 4217 :: root149Tail10
def root149Tail8 : List Nat := 3407 :: root149Tail9
def root149Tail7 : List Nat := 3011 :: root149Tail8
def root149Tail6 : List Nat := 2411 :: root149Tail7
def root149Tail5 : List Nat := 1949 :: root149Tail6
def root149Tail4 : List Nat := 1667 :: root149Tail5
def root149Tail3 : List Nat := 1499 :: root149Tail4
def root149Tail2 : List Nat := 1373 :: root149Tail3
def root149Tail1 : List Nat := 983 :: root149Tail2
def root149Tail0 : List Nat := 953 :: root149Tail1
def root149Candidates0 : List Nat := [14591, 16937]
set_option maxRecDepth 8192 in
theorem root149Candidates0_checked : candidates edge 4 881 953 root149Tail1 winner.bound = root149Candidates0 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
theorem root149Step0 : scan edge (search edge 3) 4 [881] 881 root149Tail0 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail1 winner := by
  rw [root149Tail0, scan, if_neg (by decide), root149Candidates0_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail1
    (if 3 ≤ root149Candidates0.length then search edge 3 [881, 953] root149Candidates0 1834 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates1 : List Nat := [19301]
set_option maxRecDepth 8192 in
theorem root149Candidates1_checked : candidates edge 4 881 983 root149Tail2 winner.bound = root149Candidates1 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root149Step1 : scan edge (search edge 3) 4 [881] 881 root149Tail1 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail2 winner := by
  rw [root149Tail1, scan, if_neg (by decide), root149Candidates1_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail2
    (if 3 ≤ root149Candidates1.length then search edge 3 [881, 983] root149Candidates1 1864 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates2 : List Nat := [1499, 7541, 10487, 14591, 22943]
set_option maxRecDepth 8192 in
theorem root149Candidates2_checked : candidates edge 4 881 1373 root149Tail3 winner.bound = root149Candidates2 := by
  unfold candidates
  have row : edge 1373 = (SortedNatTree.contains FastRows.row1373) := by
    change Data.row1373.contains = (SortedNatTree.contains FastRows.row1373)
    exact FastRows.row1373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root149Child2 : search edge 3 [881, 1373] root149Candidates2 2254 winner = winner := by decide
theorem root149Step2 : scan edge (search edge 3) 4 [881] 881 root149Tail2 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail3 winner := by
  rw [root149Tail2, scan, if_neg (by decide), root149Candidates2_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail3
    (if 3 ≤ root149Candidates2.length then search edge 3 [881, 1373] root149Candidates2 2254 winner else winner) = _
  rw [if_pos (by decide), root149Child2]
def root149Candidates3 : List Nat := [7331, 10487, 19793]
set_option maxRecDepth 8192 in
theorem root149Candidates3_checked : candidates edge 4 881 1499 root149Tail4 winner.bound = root149Candidates3 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root149Child3 : search edge 3 [881, 1499] root149Candidates3 2380 winner = winner := by decide
theorem root149Step3 : scan edge (search edge 3) 4 [881] 881 root149Tail3 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail4 winner := by
  rw [root149Tail3, scan, if_neg (by decide), root149Candidates3_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail4
    (if 3 ≤ root149Candidates3.length then search edge 3 [881, 1499] root149Candidates3 2380 winner else winner) = _
  rw [if_pos (by decide), root149Child3]
def root149Candidates4 : List Nat := [6701]
set_option maxRecDepth 8192 in
theorem root149Candidates4_checked : candidates edge 4 881 1667 root149Tail5 winner.bound = root149Candidates4 := by
  unfold candidates
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row]
  decide
theorem root149Step4 : scan edge (search edge 3) 4 [881] 881 root149Tail4 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail5 winner := by
  rw [root149Tail4, scan, if_neg (by decide), root149Candidates4_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail5
    (if 3 ≤ root149Candidates4.length then search edge 3 [881, 1667] root149Candidates4 2548 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root149Candidates5_checked : candidates edge 4 881 1949 root149Tail6 winner.bound = root149Candidates5 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root149Step5 : scan edge (search edge 3) 4 [881] 881 root149Tail5 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail6 winner := by
  rw [root149Tail5, scan, if_neg (by decide), root149Candidates5_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail6
    (if 3 ≤ root149Candidates5.length then search edge 3 [881, 1949] root149Candidates5 2830 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates6 : List Nat := [12107]
set_option maxRecDepth 8192 in
theorem root149Candidates6_checked : candidates edge 4 881 2411 root149Tail7 winner.bound = root149Candidates6 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root149Step6 : scan edge (search edge 3) 4 [881] 881 root149Tail6 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail7 winner := by
  rw [root149Tail6, scan, if_neg (by decide), root149Candidates6_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail7
    (if 3 ≤ root149Candidates6.length then search edge 3 [881, 2411] root149Candidates6 3292 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root149Candidates7_checked : candidates edge 4 881 3011 root149Tail8 winner.bound = root149Candidates7 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root149Step7 : scan edge (search edge 3) 4 [881] 881 root149Tail7 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail8 winner := by
  rw [root149Tail7, scan, if_neg (by decide), root149Candidates7_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail8
    (if 3 ≤ root149Candidates7.length then search edge 3 [881, 3011] root149Candidates7 3892 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root149Candidates8_checked : candidates edge 4 881 3407 root149Tail9 winner.bound = root149Candidates8 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root149Step8 : scan edge (search edge 3) 4 [881] 881 root149Tail8 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail9 winner := by
  rw [root149Tail8, scan, if_neg (by decide), root149Candidates8_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail9
    (if 3 ≤ root149Candidates8.length then search edge 3 [881, 3407] root149Candidates8 4288 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root149Candidates9_checked : candidates edge 4 881 4217 root149Tail10 winner.bound = root149Candidates9 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root149Step9 : scan edge (search edge 3) 4 [881] 881 root149Tail9 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail10 winner := by
  rw [root149Tail9, scan, if_neg (by decide), root149Candidates9_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail10
    (if 3 ≤ root149Candidates9.length then search edge 3 [881, 4217] root149Candidates9 5098 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root149Candidates10_checked : candidates edge 4 881 4229 root149Tail11 winner.bound = root149Candidates10 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root149Step10 : scan edge (search edge 3) 4 [881] 881 root149Tail10 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail11 winner := by
  rw [root149Tail10, scan, if_neg (by decide), root149Candidates10_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail11
    (if 3 ≤ root149Candidates10.length then search edge 3 [881, 4229] root149Candidates10 5110 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates11 : List Nat := [5591, 6473, 7331, 7607]
set_option maxRecDepth 8192 in
theorem root149Candidates11_checked : candidates edge 4 881 4457 root149Tail12 winner.bound = root149Candidates11 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root149Child11 : search edge 3 [881, 4457] root149Candidates11 5338 winner = winner := by decide
theorem root149Step11 : scan edge (search edge 3) 4 [881] 881 root149Tail11 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail12 winner := by
  rw [root149Tail11, scan, if_neg (by decide), root149Candidates11_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail12
    (if 3 ≤ root149Candidates11.length then search edge 3 [881, 4457] root149Candidates11 5338 winner else winner) = _
  rw [if_pos (by decide), root149Child11]
def root149Candidates12 : List Nat := [7331]
set_option maxRecDepth 8192 in
theorem root149Candidates12_checked : candidates edge 4 881 4733 root149Tail13 winner.bound = root149Candidates12 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root149Step12 : scan edge (search edge 3) 4 [881] 881 root149Tail12 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail13 winner := by
  rw [root149Tail12, scan, if_neg (by decide), root149Candidates12_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail13
    (if 3 ≤ root149Candidates12.length then search edge 3 [881, 4733] root149Candidates12 5614 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates13 : List Nat := [10973, 15017]
set_option maxRecDepth 8192 in
theorem root149Candidates13_checked : candidates edge 4 881 4931 root149Tail14 winner.bound = root149Candidates13 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root149Step13 : scan edge (search edge 3) 4 [881] 881 root149Tail13 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail14 winner := by
  rw [root149Tail13, scan, if_neg (by decide), root149Candidates13_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail14
    (if 3 ≤ root149Candidates13.length then search edge 3 [881, 4931] root149Candidates13 5812 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates14 : List Nat := [6053]
set_option maxRecDepth 8192 in
theorem root149Candidates14_checked : candidates edge 4 881 5021 root149Tail15 winner.bound = root149Candidates14 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root149Step14 : scan edge (search edge 3) 4 [881] 881 root149Tail14 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail15 winner := by
  rw [root149Tail14, scan, if_neg (by decide), root149Candidates14_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail15
    (if 3 ≤ root149Candidates14.length then search edge 3 [881, 5021] root149Candidates14 5902 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates15 : List Nat := [6701]
set_option maxRecDepth 8192 in
theorem root149Candidates15_checked : candidates edge 4 881 5591 root149Tail16 winner.bound = root149Candidates15 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root149Step15 : scan edge (search edge 3) 4 [881] 881 root149Tail15 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail16 winner := by
  rw [root149Tail15, scan, if_neg (by decide), root149Candidates15_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail16
    (if 3 ≤ root149Candidates15.length then search edge 3 [881, 5591] root149Candidates15 6472 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates16 : List Nat := [13841]
set_option maxRecDepth 8192 in
theorem root149Candidates16_checked : candidates edge 4 881 5801 root149Tail17 winner.bound = root149Candidates16 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root149Step16 : scan edge (search edge 3) 4 [881] 881 root149Tail16 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail17 winner := by
  rw [root149Tail16, scan, if_neg (by decide), root149Candidates16_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail17
    (if 3 ≤ root149Candidates16.length then search edge 3 [881, 5801] root149Candidates16 6682 winner else winner) = _
  rw [if_neg (by decide)]
def root149Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root149Candidates17_checked : candidates edge 4 881 6053 root149Tail18 winner.bound = root149Candidates17 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root149Step17 : scan edge (search edge 3) 4 [881] 881 root149Tail17 winner =
    scan edge (search edge 3) 4 [881] 881 root149Tail18 winner := by
  rw [root149Tail17, scan, if_neg (by decide), root149Candidates17_checked]
  change scan edge (search edge 3) 4 [881] 881 root149Tail18
    (if 3 ≤ root149Candidates17.length then search edge 3 [881, 6053] root149Candidates17 6934 winner else winner) = _
  rw [if_neg (by decide)]
theorem root149Scan18 : scan edge (search edge 3) 4 [881] 881 root149Tail18 winner = winner := by rw [root149Tail18, scan, if_pos (by decide)]
theorem root149Scan17 : scan edge (search edge 3) 4 [881] 881 root149Tail17 winner = winner :=
  (root149Step17).trans root149Scan18
theorem root149Scan16 : scan edge (search edge 3) 4 [881] 881 root149Tail16 winner = winner :=
  (root149Step16).trans root149Scan17
theorem root149Scan15 : scan edge (search edge 3) 4 [881] 881 root149Tail15 winner = winner :=
  (root149Step15).trans root149Scan16
theorem root149Scan14 : scan edge (search edge 3) 4 [881] 881 root149Tail14 winner = winner :=
  (root149Step14).trans root149Scan15
theorem root149Scan13 : scan edge (search edge 3) 4 [881] 881 root149Tail13 winner = winner :=
  (root149Step13).trans root149Scan14
theorem root149Scan12 : scan edge (search edge 3) 4 [881] 881 root149Tail12 winner = winner :=
  (root149Step12).trans root149Scan13
theorem root149Scan11 : scan edge (search edge 3) 4 [881] 881 root149Tail11 winner = winner :=
  (root149Step11).trans root149Scan12
theorem root149Scan10 : scan edge (search edge 3) 4 [881] 881 root149Tail10 winner = winner :=
  (root149Step10).trans root149Scan11
theorem root149Scan9 : scan edge (search edge 3) 4 [881] 881 root149Tail9 winner = winner :=
  (root149Step9).trans root149Scan10
theorem root149Scan8 : scan edge (search edge 3) 4 [881] 881 root149Tail8 winner = winner :=
  (root149Step8).trans root149Scan9
theorem root149Scan7 : scan edge (search edge 3) 4 [881] 881 root149Tail7 winner = winner :=
  (root149Step7).trans root149Scan8
theorem root149Scan6 : scan edge (search edge 3) 4 [881] 881 root149Tail6 winner = winner :=
  (root149Step6).trans root149Scan7
theorem root149Scan5 : scan edge (search edge 3) 4 [881] 881 root149Tail5 winner = winner :=
  (root149Step5).trans root149Scan6
theorem root149Scan4 : scan edge (search edge 3) 4 [881] 881 root149Tail4 winner = winner :=
  (root149Step4).trans root149Scan5
theorem root149Scan3 : scan edge (search edge 3) 4 [881] 881 root149Tail3 winner = winner :=
  (root149Step3).trans root149Scan4
theorem root149Scan2 : scan edge (search edge 3) 4 [881] 881 root149Tail2 winner = winner :=
  (root149Step2).trans root149Scan3
theorem root149Scan1 : scan edge (search edge 3) 4 [881] 881 root149Tail1 winner = winner :=
  (root149Step1).trans root149Scan2
theorem root149Scan0 : scan edge (search edge 3) 4 [881] 881 root149Tail0 winner = winner :=
  (root149Step0).trans root149Scan1
set_option maxRecDepth 4096 in
theorem root149Branch : search edge 4 [881] root149Candidates 881 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [881] 881 root149Tail0 winner = winner
  exact root149Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
