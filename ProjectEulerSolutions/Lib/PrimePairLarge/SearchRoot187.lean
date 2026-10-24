import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root187Candidates : List Nat := Data.row1151
set_option maxRecDepth 8192 in
theorem root187Candidates_checked : candidates edge 5 0 1151 tail188 winner.bound = root187Candidates := by
  apply candidates_eq_row edge 5 0 1151 tail188 root187Candidates winner.bound tail188_ordered
  · decide
  · decide
  · funext q
    rfl
def root187Tail55 : List Nat := []
def root187Tail54 : List Nat := 24593 :: root187Tail55
def root187Tail53 : List Nat := 23747 :: root187Tail54
def root187Tail52 : List Nat := 22937 :: root187Tail53
def root187Tail51 : List Nat := 21341 :: root187Tail52
def root187Tail50 : List Nat := 21089 :: root187Tail51
def root187Tail49 : List Nat := 20939 :: root187Tail50
def root187Tail48 : List Nat := 20177 :: root187Tail49
def root187Tail47 : List Nat := 19997 :: root187Tail48
def root187Tail46 : List Nat := 19889 :: root187Tail47
def root187Tail45 : List Nat := 19379 :: root187Tail46
def root187Tail44 : List Nat := 18251 :: root187Tail45
def root187Tail43 : List Nat := 16931 :: root187Tail44
def root187Tail42 : List Nat := 16223 :: root187Tail43
def root187Tail41 : List Nat := 15971 :: root187Tail42
def root187Tail40 : List Nat := 14969 :: root187Tail41
def root187Tail39 : List Nat := 14423 :: root187Tail40
def root187Tail38 : List Nat := 12983 :: root187Tail39
def root187Tail37 : List Nat := 12197 :: root187Tail38
def root187Tail36 : List Nat := 12113 :: root187Tail37
def root187Tail35 : List Nat := 11783 :: root187Tail36
def root187Tail34 : List Nat := 11483 :: root187Tail35
def root187Tail33 : List Nat := 11393 :: root187Tail34
def root187Tail32 : List Nat := 10733 :: root187Tail33
def root187Tail31 : List Nat := 10601 :: root187Tail32
def root187Tail30 : List Nat := 10427 :: root187Tail31
def root187Tail29 : List Nat := 10271 :: root187Tail30
def root187Tail28 : List Nat := 9461 :: root187Tail29
def root187Tail27 : List Nat := 9239 :: root187Tail28
def root187Tail26 : List Nat := 8669 :: root187Tail27
def root187Tail25 : List Nat := 8609 :: root187Tail26
def root187Tail24 : List Nat := 7577 :: root187Tail25
def root187Tail23 : List Nat := 7019 :: root187Tail24
def root187Tail22 : List Nat := 6863 :: root187Tail23
def root187Tail21 : List Nat := 6521 :: root187Tail22
def root187Tail20 : List Nat := 6473 :: root187Tail21
def root187Tail19 : List Nat := 6101 :: root187Tail20
def root187Tail18 : List Nat := 5897 :: root187Tail19
def root187Tail17 : List Nat := 5417 :: root187Tail18
def root187Tail16 : List Nat := 5333 :: root187Tail17
def root187Tail15 : List Nat := 4871 :: root187Tail16
def root187Tail14 : List Nat := 4337 :: root187Tail15
def root187Tail13 : List Nat := 4127 :: root187Tail14
def root187Tail12 : List Nat := 4001 :: root187Tail13
def root187Tail11 : List Nat := 3527 :: root187Tail12
def root187Tail10 : List Nat := 3371 :: root187Tail11
def root187Tail9 : List Nat := 2789 :: root187Tail10
def root187Tail8 : List Nat := 2399 :: root187Tail9
def root187Tail7 : List Nat := 1823 :: root187Tail8
def root187Tail6 : List Nat := 1607 :: root187Tail7
def root187Tail5 : List Nat := 1559 :: root187Tail6
def root187Tail4 : List Nat := 1481 :: root187Tail5
def root187Tail3 : List Nat := 1433 :: root187Tail4
def root187Tail2 : List Nat := 1361 :: root187Tail3
def root187Tail1 : List Nat := 1277 :: root187Tail2
def root187Tail0 : List Nat := 1217 :: root187Tail1
def root187Candidates0 : List Nat := [1361, 1433, 7019, 8609]
set_option maxRecDepth 8192 in
theorem root187Candidates0_checked : candidates edge 4 1151 1217 root187Tail1 winner.bound = root187Candidates0 := by
  unfold candidates
  have row : edge 1217 = (SortedNatTree.contains FastRows.row1217) := by
    change Data.row1217.contains = (SortedNatTree.contains FastRows.row1217)
    exact FastRows.row1217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root187Child0 : search edge 3 [1151, 1217] root187Candidates0 2368 winner = winner := by decide
theorem root187Step0 : scan edge (search edge 3) 4 [1151] 1151 root187Tail0 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail1 winner := by
  rw [root187Tail0, scan, if_neg (by decide), root187Candidates0_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail1
    (if 3 ≤ root187Candidates0.length then search edge 3 [1151, 1217] root187Candidates0 2368 winner else winner) = _
  rw [if_pos (by decide), root187Child0]
def root187Candidates1 : List Nat := [7577, 10601]
set_option maxRecDepth 8192 in
theorem root187Candidates1_checked : candidates edge 4 1151 1277 root187Tail2 winner.bound = root187Candidates1 := by
  unfold candidates
  have row : edge 1277 = (SortedNatTree.contains FastRows.row1277) := by
    change Data.row1277.contains = (SortedNatTree.contains FastRows.row1277)
    exact FastRows.row1277_checked.symm
  rw [row]
  decide
theorem root187Step1 : scan edge (search edge 3) 4 [1151] 1151 root187Tail1 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail2 winner := by
  rw [root187Tail1, scan, if_neg (by decide), root187Candidates1_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail2
    (if 3 ≤ root187Candidates1.length then search edge 3 [1151, 1277] root187Candidates1 2428 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates2 : List Nat := [14969]
set_option maxRecDepth 8192 in
theorem root187Candidates2_checked : candidates edge 4 1151 1361 root187Tail3 winner.bound = root187Candidates2 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
theorem root187Step2 : scan edge (search edge 3) 4 [1151] 1151 root187Tail2 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail3 winner := by
  rw [root187Tail2, scan, if_neg (by decide), root187Candidates2_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail3
    (if 3 ≤ root187Candidates2.length then search edge 3 [1151, 1361] root187Candidates2 2512 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates3 : List Nat := [21341]
set_option maxRecDepth 8192 in
theorem root187Candidates3_checked : candidates edge 4 1151 1433 root187Tail4 winner.bound = root187Candidates3 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
theorem root187Step3 : scan edge (search edge 3) 4 [1151] 1151 root187Tail3 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail4 winner := by
  rw [root187Tail3, scan, if_neg (by decide), root187Candidates3_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail4
    (if 3 ≤ root187Candidates3.length then search edge 3 [1151, 1433] root187Candidates3 2584 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates4 : List Nat := [4127, 10427]
set_option maxRecDepth 8192 in
theorem root187Candidates4_checked : candidates edge 4 1151 1481 root187Tail5 winner.bound = root187Candidates4 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
theorem root187Step4 : scan edge (search edge 3) 4 [1151] 1151 root187Tail4 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail5 winner := by
  rw [root187Tail4, scan, if_neg (by decide), root187Candidates4_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail5
    (if 3 ≤ root187Candidates4.length then search edge 3 [1151, 1481] root187Candidates4 2632 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates5 : List Nat := [5897, 19889]
set_option maxRecDepth 8192 in
theorem root187Candidates5_checked : candidates edge 4 1151 1559 root187Tail6 winner.bound = root187Candidates5 := by
  unfold candidates
  have row : edge 1559 = (SortedNatTree.contains FastRows.row1559) := by
    change Data.row1559.contains = (SortedNatTree.contains FastRows.row1559)
    exact FastRows.row1559_checked.symm
  rw [row]
  decide
theorem root187Step5 : scan edge (search edge 3) 4 [1151] 1151 root187Tail5 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail6 winner := by
  rw [root187Tail5, scan, if_neg (by decide), root187Candidates5_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail6
    (if 3 ≤ root187Candidates5.length then search edge 3 [1151, 1559] root187Candidates5 2710 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates6 : List Nat := [20939, 22937]
set_option maxRecDepth 8192 in
theorem root187Candidates6_checked : candidates edge 4 1151 1607 root187Tail7 winner.bound = root187Candidates6 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
theorem root187Step6 : scan edge (search edge 3) 4 [1151] 1151 root187Tail6 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail7 winner := by
  rw [root187Tail6, scan, if_neg (by decide), root187Candidates6_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail7
    (if 3 ≤ root187Candidates6.length then search edge 3 [1151, 1607] root187Candidates6 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates7 : List Nat := [4001, 14423]
set_option maxRecDepth 8192 in
theorem root187Candidates7_checked : candidates edge 4 1151 1823 root187Tail8 winner.bound = root187Candidates7 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
theorem root187Step7 : scan edge (search edge 3) 4 [1151] 1151 root187Tail7 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail8 winner := by
  rw [root187Tail7, scan, if_neg (by decide), root187Candidates7_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail8
    (if 3 ≤ root187Candidates7.length then search edge 3 [1151, 1823] root187Candidates7 2974 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates8 : List Nat := [20939]
set_option maxRecDepth 8192 in
theorem root187Candidates8_checked : candidates edge 4 1151 2399 root187Tail9 winner.bound = root187Candidates8 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
theorem root187Step8 : scan edge (search edge 3) 4 [1151] 1151 root187Tail8 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail9 winner := by
  rw [root187Tail8, scan, if_neg (by decide), root187Candidates8_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail9
    (if 3 ≤ root187Candidates8.length then search edge 3 [1151, 2399] root187Candidates8 3550 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates9 : List Nat := [10601, 12197, 21089]
set_option maxRecDepth 8192 in
theorem root187Candidates9_checked : candidates edge 4 1151 2789 root187Tail10 winner.bound = root187Candidates9 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root187Child9 : search edge 3 [1151, 2789] root187Candidates9 3940 winner = winner := by decide
theorem root187Step9 : scan edge (search edge 3) 4 [1151] 1151 root187Tail9 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail10 winner := by
  rw [root187Tail9, scan, if_neg (by decide), root187Candidates9_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail10
    (if 3 ≤ root187Candidates9.length then search edge 3 [1151, 2789] root187Candidates9 3940 winner else winner) = _
  rw [if_pos (by decide), root187Child9]
def root187Candidates10 : List Nat := [16931]
set_option maxRecDepth 8192 in
theorem root187Candidates10_checked : candidates edge 4 1151 3371 root187Tail11 winner.bound = root187Candidates10 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
theorem root187Step10 : scan edge (search edge 3) 4 [1151] 1151 root187Tail10 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail11 winner := by
  rw [root187Tail10, scan, if_neg (by decide), root187Candidates10_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail11
    (if 3 ≤ root187Candidates10.length then search edge 3 [1151, 3371] root187Candidates10 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates11 : List Nat := [8669, 9239]
set_option maxRecDepth 8192 in
theorem root187Candidates11_checked : candidates edge 4 1151 3527 root187Tail12 winner.bound = root187Candidates11 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root187Step11 : scan edge (search edge 3) 4 [1151] 1151 root187Tail11 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail12 winner := by
  rw [root187Tail11, scan, if_neg (by decide), root187Candidates11_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail12
    (if 3 ≤ root187Candidates11.length then search edge 3 [1151, 3527] root187Candidates11 4678 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates12 : List Nat := [7577, 19997]
set_option maxRecDepth 8192 in
theorem root187Candidates12_checked : candidates edge 4 1151 4001 root187Tail13 winner.bound = root187Candidates12 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root187Step12 : scan edge (search edge 3) 4 [1151] 1151 root187Tail12 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail13 winner := by
  rw [root187Tail12, scan, if_neg (by decide), root187Candidates12_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail13
    (if 3 ≤ root187Candidates12.length then search edge 3 [1151, 4001] root187Candidates12 5152 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates13 : List Nat := [12983]
set_option maxRecDepth 8192 in
theorem root187Candidates13_checked : candidates edge 4 1151 4127 root187Tail14 winner.bound = root187Candidates13 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root187Step13 : scan edge (search edge 3) 4 [1151] 1151 root187Tail13 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail14 winner := by
  rw [root187Tail13, scan, if_neg (by decide), root187Candidates13_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail14
    (if 3 ≤ root187Candidates13.length then search edge 3 [1151, 4127] root187Candidates13 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates14 : List Nat := [18251]
set_option maxRecDepth 8192 in
theorem root187Candidates14_checked : candidates edge 4 1151 4337 root187Tail15 winner.bound = root187Candidates14 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root187Step14 : scan edge (search edge 3) 4 [1151] 1151 root187Tail14 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail15 winner := by
  rw [root187Tail14, scan, if_neg (by decide), root187Candidates14_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail15
    (if 3 ≤ root187Candidates14.length then search edge 3 [1151, 4337] root187Candidates14 5488 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root187Candidates15_checked : candidates edge 4 1151 4871 root187Tail16 winner.bound = root187Candidates15 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root187Step15 : scan edge (search edge 3) 4 [1151] 1151 root187Tail15 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail16 winner := by
  rw [root187Tail15, scan, if_neg (by decide), root187Candidates15_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail16
    (if 3 ≤ root187Candidates15.length then search edge 3 [1151, 4871] root187Candidates15 6022 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates16 : List Nat := [11483]
set_option maxRecDepth 8192 in
theorem root187Candidates16_checked : candidates edge 4 1151 5333 root187Tail17 winner.bound = root187Candidates16 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root187Step16 : scan edge (search edge 3) 4 [1151] 1151 root187Tail16 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail17 winner := by
  rw [root187Tail16, scan, if_neg (by decide), root187Candidates16_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail17
    (if 3 ≤ root187Candidates16.length then search edge 3 [1151, 5333] root187Candidates16 6484 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates17 : List Nat := [9239]
set_option maxRecDepth 8192 in
theorem root187Candidates17_checked : candidates edge 4 1151 5417 root187Tail18 winner.bound = root187Candidates17 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root187Step17 : scan edge (search edge 3) 4 [1151] 1151 root187Tail17 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail18 winner := by
  rw [root187Tail17, scan, if_neg (by decide), root187Candidates17_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail18
    (if 3 ≤ root187Candidates17.length then search edge 3 [1151, 5417] root187Candidates17 6568 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates18 : List Nat := [7019, 12197]
set_option maxRecDepth 8192 in
theorem root187Candidates18_checked : candidates edge 4 1151 5897 root187Tail19 winner.bound = root187Candidates18 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root187Step18 : scan edge (search edge 3) 4 [1151] 1151 root187Tail18 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail19 winner := by
  rw [root187Tail18, scan, if_neg (by decide), root187Candidates18_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail19
    (if 3 ≤ root187Candidates18.length then search edge 3 [1151, 5897] root187Candidates18 7048 winner else winner) = _
  rw [if_neg (by decide)]
def root187Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root187Candidates19_checked : candidates edge 4 1151 6101 root187Tail20 winner.bound = root187Candidates19 := by
  unfold candidates
  have row : edge 6101 = (SortedNatTree.contains FastRows.row6101) := by
    change Data.row6101.contains = (SortedNatTree.contains FastRows.row6101)
    exact FastRows.row6101_checked.symm
  rw [row]
  decide
theorem root187Step19 : scan edge (search edge 3) 4 [1151] 1151 root187Tail19 winner =
    scan edge (search edge 3) 4 [1151] 1151 root187Tail20 winner := by
  rw [root187Tail19, scan, if_neg (by decide), root187Candidates19_checked]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail20
    (if 3 ≤ root187Candidates19.length then search edge 3 [1151, 6101] root187Candidates19 7252 winner else winner) = _
  rw [if_neg (by decide)]
theorem root187Scan20 : scan edge (search edge 3) 4 [1151] 1151 root187Tail20 winner = winner := by rw [root187Tail20, scan, if_pos (by decide)]
theorem root187Scan19 : scan edge (search edge 3) 4 [1151] 1151 root187Tail19 winner = winner :=
  (root187Step19).trans root187Scan20
theorem root187Scan18 : scan edge (search edge 3) 4 [1151] 1151 root187Tail18 winner = winner :=
  (root187Step18).trans root187Scan19
theorem root187Scan17 : scan edge (search edge 3) 4 [1151] 1151 root187Tail17 winner = winner :=
  (root187Step17).trans root187Scan18
theorem root187Scan16 : scan edge (search edge 3) 4 [1151] 1151 root187Tail16 winner = winner :=
  (root187Step16).trans root187Scan17
theorem root187Scan15 : scan edge (search edge 3) 4 [1151] 1151 root187Tail15 winner = winner :=
  (root187Step15).trans root187Scan16
theorem root187Scan14 : scan edge (search edge 3) 4 [1151] 1151 root187Tail14 winner = winner :=
  (root187Step14).trans root187Scan15
theorem root187Scan13 : scan edge (search edge 3) 4 [1151] 1151 root187Tail13 winner = winner :=
  (root187Step13).trans root187Scan14
theorem root187Scan12 : scan edge (search edge 3) 4 [1151] 1151 root187Tail12 winner = winner :=
  (root187Step12).trans root187Scan13
theorem root187Scan11 : scan edge (search edge 3) 4 [1151] 1151 root187Tail11 winner = winner :=
  (root187Step11).trans root187Scan12
theorem root187Scan10 : scan edge (search edge 3) 4 [1151] 1151 root187Tail10 winner = winner :=
  (root187Step10).trans root187Scan11
theorem root187Scan9 : scan edge (search edge 3) 4 [1151] 1151 root187Tail9 winner = winner :=
  (root187Step9).trans root187Scan10
theorem root187Scan8 : scan edge (search edge 3) 4 [1151] 1151 root187Tail8 winner = winner :=
  (root187Step8).trans root187Scan9
theorem root187Scan7 : scan edge (search edge 3) 4 [1151] 1151 root187Tail7 winner = winner :=
  (root187Step7).trans root187Scan8
theorem root187Scan6 : scan edge (search edge 3) 4 [1151] 1151 root187Tail6 winner = winner :=
  (root187Step6).trans root187Scan7
theorem root187Scan5 : scan edge (search edge 3) 4 [1151] 1151 root187Tail5 winner = winner :=
  (root187Step5).trans root187Scan6
theorem root187Scan4 : scan edge (search edge 3) 4 [1151] 1151 root187Tail4 winner = winner :=
  (root187Step4).trans root187Scan5
theorem root187Scan3 : scan edge (search edge 3) 4 [1151] 1151 root187Tail3 winner = winner :=
  (root187Step3).trans root187Scan4
theorem root187Scan2 : scan edge (search edge 3) 4 [1151] 1151 root187Tail2 winner = winner :=
  (root187Step2).trans root187Scan3
theorem root187Scan1 : scan edge (search edge 3) 4 [1151] 1151 root187Tail1 winner = winner :=
  (root187Step1).trans root187Scan2
theorem root187Scan0 : scan edge (search edge 3) 4 [1151] 1151 root187Tail0 winner = winner :=
  (root187Step0).trans root187Scan1
set_option maxRecDepth 4096 in
theorem root187Branch : search edge 4 [1151] root187Candidates 1151 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1151] 1151 root187Tail0 winner = winner
  exact root187Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
