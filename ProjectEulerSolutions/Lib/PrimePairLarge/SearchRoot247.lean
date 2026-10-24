import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root247Candidates : List Nat := Data.row1583
set_option maxRecDepth 8192 in
theorem root247Candidates_checked : candidates edge 5 0 1583 tail248 winner.bound = root247Candidates := by
  apply candidates_eq_row edge 5 0 1583 tail248 root247Candidates winner.bound tail248_ordered
  · decide
  · decide
  · funext q
    rfl
def root247Tail51 : List Nat := []
def root247Tail50 : List Nat := 23831 :: root247Tail51
def root247Tail49 : List Nat := 22481 :: root247Tail50
def root247Tail48 : List Nat := 21563 :: root247Tail49
def root247Tail47 : List Nat := 20849 :: root247Tail48
def root247Tail46 : List Nat := 20693 :: root247Tail47
def root247Tail45 : List Nat := 20117 :: root247Tail46
def root247Tail44 : List Nat := 19853 :: root247Tail45
def root247Tail43 : List Nat := 18869 :: root247Tail44
def root247Tail42 : List Nat := 17921 :: root247Tail43
def root247Tail41 : List Nat := 17657 :: root247Tail42
def root247Tail40 : List Nat := 17123 :: root247Tail41
def root247Tail39 : List Nat := 16943 :: root247Tail40
def root247Tail38 : List Nat := 16811 :: root247Tail39
def root247Tail37 : List Nat := 16421 :: root247Tail38
def root247Tail36 : List Nat := 16001 :: root247Tail37
def root247Tail35 : List Nat := 14153 :: root247Tail36
def root247Tail34 : List Nat := 13037 :: root247Tail35
def root247Tail33 : List Nat := 10949 :: root247Tail34
def root247Tail32 : List Nat := 10301 :: root247Tail33
def root247Tail31 : List Nat := 9377 :: root247Tail32
def root247Tail30 : List Nat := 8369 :: root247Tail31
def root247Tail29 : List Nat := 8171 :: root247Tail30
def root247Tail28 : List Nat := 7949 :: root247Tail29
def root247Tail27 : List Nat := 7643 :: root247Tail28
def root247Tail26 : List Nat := 7079 :: root247Tail27
def root247Tail25 : List Nat := 7043 :: root247Tail26
def root247Tail24 : List Nat := 6869 :: root247Tail25
def root247Tail23 : List Nat := 6827 :: root247Tail24
def root247Tail22 : List Nat := 6803 :: root247Tail23
def root247Tail21 : List Nat := 6659 :: root247Tail22
def root247Tail20 : List Nat := 6551 :: root247Tail21
def root247Tail19 : List Nat := 6287 :: root247Tail20
def root247Tail18 : List Nat := 6143 :: root247Tail19
def root247Tail17 : List Nat := 6113 :: root247Tail18
def root247Tail16 : List Nat := 5879 :: root247Tail17
def root247Tail15 : List Nat := 5861 :: root247Tail16
def root247Tail14 : List Nat := 5333 :: root247Tail15
def root247Tail13 : List Nat := 5009 :: root247Tail14
def root247Tail12 : List Nat := 4799 :: root247Tail13
def root247Tail11 : List Nat := 4451 :: root247Tail12
def root247Tail10 : List Nat := 4283 :: root247Tail11
def root247Tail9 : List Nat := 4229 :: root247Tail10
def root247Tail8 : List Nat := 4001 :: root247Tail9
def root247Tail7 : List Nat := 3929 :: root247Tail8
def root247Tail6 : List Nat := 3557 :: root247Tail7
def root247Tail5 : List Nat := 3347 :: root247Tail6
def root247Tail4 : List Nat := 2447 :: root247Tail5
def root247Tail3 : List Nat := 1973 :: root247Tail4
def root247Tail2 : List Nat := 1877 :: root247Tail3
def root247Tail1 : List Nat := 1637 :: root247Tail2
def root247Tail0 : List Nat := 1619 :: root247Tail1
def root247Candidates0 : List Nat := [6143, 7949]
set_option maxRecDepth 8192 in
theorem root247Candidates0_checked : candidates edge 4 1583 1619 root247Tail1 winner.bound = root247Candidates0 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
theorem root247Step0 : scan edge (search edge 3) 4 [1583] 1583 root247Tail0 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail1 winner := by
  rw [root247Tail0, scan, if_neg (by decide), root247Candidates0_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail1
    (if 3 ≤ root247Candidates0.length then search edge 3 [1583, 1619] root247Candidates0 3202 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root247Candidates1_checked : candidates edge 4 1583 1637 root247Tail2 winner.bound = root247Candidates1 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
theorem root247Step1 : scan edge (search edge 3) 4 [1583] 1583 root247Tail1 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail2 winner := by
  rw [root247Tail1, scan, if_neg (by decide), root247Candidates1_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail2
    (if 3 ≤ root247Candidates1.length then search edge 3 [1583, 1637] root247Candidates1 3220 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates2 : List Nat := [2447, 6113, 6287, 13037]
set_option maxRecDepth 8192 in
theorem root247Candidates2_checked : candidates edge 4 1583 1877 root247Tail3 winner.bound = root247Candidates2 := by
  unfold candidates
  have row : edge 1877 = (SortedNatTree.contains FastRows.row1877) := by
    change Data.row1877.contains = (SortedNatTree.contains FastRows.row1877)
    exact FastRows.row1877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root247Child2 : search edge 3 [1583, 1877] root247Candidates2 3460 winner = winner := by decide
theorem root247Step2 : scan edge (search edge 3) 4 [1583] 1583 root247Tail2 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail3 winner := by
  rw [root247Tail2, scan, if_neg (by decide), root247Candidates2_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail3
    (if 3 ≤ root247Candidates2.length then search edge 3 [1583, 1877] root247Candidates2 3460 winner else winner) = _
  rw [if_pos (by decide), root247Child2]
def root247Candidates3 : List Nat := [21563]
set_option maxRecDepth 8192 in
theorem root247Candidates3_checked : candidates edge 4 1583 1973 root247Tail4 winner.bound = root247Candidates3 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
theorem root247Step3 : scan edge (search edge 3) 4 [1583] 1583 root247Tail3 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail4 winner := by
  rw [root247Tail3, scan, if_neg (by decide), root247Candidates3_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail4
    (if 3 ≤ root247Candidates3.length then search edge 3 [1583, 1973] root247Candidates3 3556 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates4 : List Nat := [6113]
set_option maxRecDepth 8192 in
theorem root247Candidates4_checked : candidates edge 4 1583 2447 root247Tail5 winner.bound = root247Candidates4 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root247Step4 : scan edge (search edge 3) 4 [1583] 1583 root247Tail4 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail5 winner := by
  rw [root247Tail4, scan, if_neg (by decide), root247Candidates4_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail5
    (if 3 ≤ root247Candidates4.length then search edge 3 [1583, 2447] root247Candidates4 4030 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates5 : List Nat := [5333, 6143, 7043, 18869]
set_option maxRecDepth 8192 in
theorem root247Candidates5_checked : candidates edge 4 1583 3347 root247Tail6 winner.bound = root247Candidates5 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root247Child5 : search edge 3 [1583, 3347] root247Candidates5 4930 winner = winner := by decide
theorem root247Step5 : scan edge (search edge 3) 4 [1583] 1583 root247Tail5 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail6 winner := by
  rw [root247Tail5, scan, if_neg (by decide), root247Candidates5_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail6
    (if 3 ≤ root247Candidates5.length then search edge 3 [1583, 3347] root247Candidates5 4930 winner else winner) = _
  rw [if_pos (by decide), root247Child5]
def root247Candidates6 : List Nat := [6551, 13037, 16811]
set_option maxRecDepth 8192 in
theorem root247Candidates6_checked : candidates edge 4 1583 3557 root247Tail7 winner.bound = root247Candidates6 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root247Child6 : search edge 3 [1583, 3557] root247Candidates6 5140 winner = winner := by decide
theorem root247Step6 : scan edge (search edge 3) 4 [1583] 1583 root247Tail6 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail7 winner := by
  rw [root247Tail6, scan, if_neg (by decide), root247Candidates6_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail7
    (if 3 ≤ root247Candidates6.length then search edge 3 [1583, 3557] root247Candidates6 5140 winner else winner) = _
  rw [if_pos (by decide), root247Child6]
def root247Candidates7 : List Nat := [6143, 7043]
set_option maxRecDepth 8192 in
theorem root247Candidates7_checked : candidates edge 4 1583 3929 root247Tail8 winner.bound = root247Candidates7 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root247Step7 : scan edge (search edge 3) 4 [1583] 1583 root247Tail7 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail8 winner := by
  rw [root247Tail7, scan, if_neg (by decide), root247Candidates7_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail8
    (if 3 ≤ root247Candidates7.length then search edge 3 [1583, 3929] root247Candidates7 5512 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root247Candidates8_checked : candidates edge 4 1583 4001 root247Tail9 winner.bound = root247Candidates8 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root247Step8 : scan edge (search edge 3) 4 [1583] 1583 root247Tail8 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail9 winner := by
  rw [root247Tail8, scan, if_neg (by decide), root247Candidates8_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail9
    (if 3 ≤ root247Candidates8.length then search edge 3 [1583, 4001] root247Candidates8 5584 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates9 : List Nat := [7643]
set_option maxRecDepth 8192 in
theorem root247Candidates9_checked : candidates edge 4 1583 4229 root247Tail10 winner.bound = root247Candidates9 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root247Step9 : scan edge (search edge 3) 4 [1583] 1583 root247Tail9 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail10 winner := by
  rw [root247Tail9, scan, if_neg (by decide), root247Candidates9_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail10
    (if 3 ≤ root247Candidates9.length then search edge 3 [1583, 4229] root247Candidates9 5812 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root247Candidates10_checked : candidates edge 4 1583 4283 root247Tail11 winner.bound = root247Candidates10 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root247Step10 : scan edge (search edge 3) 4 [1583] 1583 root247Tail10 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail11 winner := by
  rw [root247Tail10, scan, if_neg (by decide), root247Candidates10_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail11
    (if 3 ≤ root247Candidates10.length then search edge 3 [1583, 4283] root247Candidates10 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates11 : List Nat := [6803, 16943]
set_option maxRecDepth 8192 in
theorem root247Candidates11_checked : candidates edge 4 1583 4451 root247Tail12 winner.bound = root247Candidates11 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
theorem root247Step11 : scan edge (search edge 3) 4 [1583] 1583 root247Tail11 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail12 winner := by
  rw [root247Tail11, scan, if_neg (by decide), root247Candidates11_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail12
    (if 3 ≤ root247Candidates11.length then search edge 3 [1583, 4451] root247Candidates11 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root247Candidates12_checked : candidates edge 4 1583 4799 root247Tail13 winner.bound = root247Candidates12 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root247Step12 : scan edge (search edge 3) 4 [1583] 1583 root247Tail12 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail13 winner := by
  rw [root247Tail12, scan, if_neg (by decide), root247Candidates12_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail13
    (if 3 ≤ root247Candidates12.length then search edge 3 [1583, 4799] root247Candidates12 6382 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates13 : List Nat := [5861]
set_option maxRecDepth 8192 in
theorem root247Candidates13_checked : candidates edge 4 1583 5009 root247Tail14 winner.bound = root247Candidates13 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root247Step13 : scan edge (search edge 3) 4 [1583] 1583 root247Tail13 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail14 winner := by
  rw [root247Tail13, scan, if_neg (by decide), root247Candidates13_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail14
    (if 3 ≤ root247Candidates13.length then search edge 3 [1583, 5009] root247Candidates13 6592 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root247Candidates14_checked : candidates edge 4 1583 5333 root247Tail15 winner.bound = root247Candidates14 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root247Step14 : scan edge (search edge 3) 4 [1583] 1583 root247Tail14 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail15 winner := by
  rw [root247Tail14, scan, if_neg (by decide), root247Candidates14_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail15
    (if 3 ≤ root247Candidates14.length then search edge 3 [1583, 5333] root247Candidates14 6916 winner else winner) = _
  rw [if_neg (by decide)]
def root247Candidates15 : List Nat := [6143, 6827, 7043, 8171]
set_option maxRecDepth 8192 in
theorem root247Candidates15_checked : candidates edge 4 1583 5861 root247Tail16 winner.bound = root247Candidates15 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root247Child15 : search edge 3 [1583, 5861] root247Candidates15 7444 winner = winner := by decide
theorem root247Step15 : scan edge (search edge 3) 4 [1583] 1583 root247Tail15 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail16 winner := by
  rw [root247Tail15, scan, if_neg (by decide), root247Candidates15_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail16
    (if 3 ≤ root247Candidates15.length then search edge 3 [1583, 5861] root247Candidates15 7444 winner else winner) = _
  rw [if_pos (by decide), root247Child15]
def root247Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root247Candidates16_checked : candidates edge 4 1583 5879 root247Tail17 winner.bound = root247Candidates16 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root247Step16 : scan edge (search edge 3) 4 [1583] 1583 root247Tail16 winner =
    scan edge (search edge 3) 4 [1583] 1583 root247Tail17 winner := by
  rw [root247Tail16, scan, if_neg (by decide), root247Candidates16_checked]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail17
    (if 3 ≤ root247Candidates16.length then search edge 3 [1583, 5879] root247Candidates16 7462 winner else winner) = _
  rw [if_neg (by decide)]
theorem root247Scan17 : scan edge (search edge 3) 4 [1583] 1583 root247Tail17 winner = winner := by rw [root247Tail17, scan, if_pos (by decide)]
theorem root247Scan16 : scan edge (search edge 3) 4 [1583] 1583 root247Tail16 winner = winner :=
  (root247Step16).trans root247Scan17
theorem root247Scan15 : scan edge (search edge 3) 4 [1583] 1583 root247Tail15 winner = winner :=
  (root247Step15).trans root247Scan16
theorem root247Scan14 : scan edge (search edge 3) 4 [1583] 1583 root247Tail14 winner = winner :=
  (root247Step14).trans root247Scan15
theorem root247Scan13 : scan edge (search edge 3) 4 [1583] 1583 root247Tail13 winner = winner :=
  (root247Step13).trans root247Scan14
theorem root247Scan12 : scan edge (search edge 3) 4 [1583] 1583 root247Tail12 winner = winner :=
  (root247Step12).trans root247Scan13
theorem root247Scan11 : scan edge (search edge 3) 4 [1583] 1583 root247Tail11 winner = winner :=
  (root247Step11).trans root247Scan12
theorem root247Scan10 : scan edge (search edge 3) 4 [1583] 1583 root247Tail10 winner = winner :=
  (root247Step10).trans root247Scan11
theorem root247Scan9 : scan edge (search edge 3) 4 [1583] 1583 root247Tail9 winner = winner :=
  (root247Step9).trans root247Scan10
theorem root247Scan8 : scan edge (search edge 3) 4 [1583] 1583 root247Tail8 winner = winner :=
  (root247Step8).trans root247Scan9
theorem root247Scan7 : scan edge (search edge 3) 4 [1583] 1583 root247Tail7 winner = winner :=
  (root247Step7).trans root247Scan8
theorem root247Scan6 : scan edge (search edge 3) 4 [1583] 1583 root247Tail6 winner = winner :=
  (root247Step6).trans root247Scan7
theorem root247Scan5 : scan edge (search edge 3) 4 [1583] 1583 root247Tail5 winner = winner :=
  (root247Step5).trans root247Scan6
theorem root247Scan4 : scan edge (search edge 3) 4 [1583] 1583 root247Tail4 winner = winner :=
  (root247Step4).trans root247Scan5
theorem root247Scan3 : scan edge (search edge 3) 4 [1583] 1583 root247Tail3 winner = winner :=
  (root247Step3).trans root247Scan4
theorem root247Scan2 : scan edge (search edge 3) 4 [1583] 1583 root247Tail2 winner = winner :=
  (root247Step2).trans root247Scan3
theorem root247Scan1 : scan edge (search edge 3) 4 [1583] 1583 root247Tail1 winner = winner :=
  (root247Step1).trans root247Scan2
theorem root247Scan0 : scan edge (search edge 3) 4 [1583] 1583 root247Tail0 winner = winner :=
  (root247Step0).trans root247Scan1
set_option maxRecDepth 4096 in
theorem root247Branch : search edge 4 [1583] root247Candidates 1583 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1583] 1583 root247Tail0 winner = winner
  exact root247Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
