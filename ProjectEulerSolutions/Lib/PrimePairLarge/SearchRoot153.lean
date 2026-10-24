import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root153Candidates : List Nat := Data.row911
set_option maxRecDepth 8192 in
theorem root153Candidates_checked : candidates edge 5 0 911 tail154 winner.bound = root153Candidates := by
  apply candidates_eq_row edge 5 0 911 tail154 root153Candidates winner.bound tail154_ordered
  · decide
  · decide
  · funext q
    rfl
def root153Tail57 : List Nat := []
def root153Tail56 : List Nat := 24281 :: root153Tail57
def root153Tail55 : List Nat := 23831 :: root153Tail56
def root153Tail54 : List Nat := 23561 :: root153Tail55
def root153Tail53 : List Nat := 23399 :: root153Tail54
def root153Tail52 : List Nat := 22817 :: root153Tail53
def root153Tail51 : List Nat := 22679 :: root153Tail52
def root153Tail50 : List Nat := 22391 :: root153Tail51
def root153Tail49 : List Nat := 21647 :: root153Tail50
def root153Tail48 : List Nat := 21227 :: root153Tail49
def root153Tail47 : List Nat := 21179 :: root153Tail48
def root153Tail46 : List Nat := 21143 :: root153Tail47
def root153Tail45 : List Nat := 20807 :: root153Tail46
def root153Tail44 : List Nat := 18917 :: root153Tail45
def root153Tail43 : List Nat := 17987 :: root153Tail44
def root153Tail42 : List Nat := 17231 :: root153Tail43
def root153Tail41 : List Nat := 16187 :: root153Tail42
def root153Tail40 : List Nat := 15287 :: root153Tail41
def root153Tail39 : List Nat := 15077 :: root153Tail40
def root153Tail38 : List Nat := 14783 :: root153Tail39
def root153Tail37 : List Nat := 14411 :: root153Tail38
def root153Tail36 : List Nat := 14369 :: root153Tail37
def root153Tail35 : List Nat := 14159 :: root153Tail36
def root153Tail34 : List Nat := 13751 :: root153Tail35
def root153Tail33 : List Nat := 13613 :: root153Tail34
def root153Tail32 : List Nat := 12893 :: root153Tail33
def root153Tail31 : List Nat := 12479 :: root153Tail32
def root153Tail30 : List Nat := 12473 :: root153Tail31
def root153Tail29 : List Nat := 12347 :: root153Tail30
def root153Tail28 : List Nat := 11903 :: root153Tail29
def root153Tail27 : List Nat := 11681 :: root153Tail28
def root153Tail26 : List Nat := 11261 :: root153Tail27
def root153Tail25 : List Nat := 9749 :: root153Tail26
def root153Tail24 : List Nat := 9629 :: root153Tail25
def root153Tail23 : List Nat := 9533 :: root153Tail24
def root153Tail22 : List Nat := 8819 :: root153Tail23
def root153Tail21 : List Nat := 8609 :: root153Tail22
def root153Tail20 : List Nat := 8111 :: root153Tail21
def root153Tail19 : List Nat := 7109 :: root153Tail20
def root153Tail18 : List Nat := 6971 :: root153Tail19
def root153Tail17 : List Nat := 6269 :: root153Tail18
def root153Tail16 : List Nat := 6113 :: root153Tail17
def root153Tail15 : List Nat := 6089 :: root153Tail16
def root153Tail14 : List Nat := 5471 :: root153Tail15
def root153Tail13 : List Nat := 5303 :: root153Tail14
def root153Tail12 : List Nat := 4643 :: root153Tail13
def root153Tail11 : List Nat := 4583 :: root153Tail12
def root153Tail10 : List Nat := 4517 :: root153Tail11
def root153Tail9 : List Nat := 4337 :: root153Tail10
def root153Tail8 : List Nat := 3821 :: root153Tail9
def root153Tail7 : List Nat := 3413 :: root153Tail8
def root153Tail6 : List Nat := 2657 :: root153Tail7
def root153Tail5 : List Nat := 2399 :: root153Tail6
def root153Tail4 : List Nat := 2309 :: root153Tail5
def root153Tail3 : List Nat := 1511 :: root153Tail4
def root153Tail2 : List Nat := 1229 :: root153Tail3
def root153Tail1 : List Nat := 1103 :: root153Tail2
def root153Tail0 : List Nat := 947 :: root153Tail1
def root153Candidates0 : List Nat := [2657, 8111, 14159, 16187, 20807]
set_option maxRecDepth 8192 in
theorem root153Candidates0_checked : candidates edge 4 911 947 root153Tail1 winner.bound = root153Candidates0 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root153Child0 : search edge 3 [911, 947] root153Candidates0 1858 winner = winner := by decide
theorem root153Step0 : scan edge (search edge 3) 4 [911] 911 root153Tail0 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail1 winner := by
  rw [root153Tail0, scan, if_neg (by decide), root153Candidates0_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail1
    (if 3 ≤ root153Candidates0.length then search edge 3 [911, 947] root153Candidates0 1858 winner else winner) = _
  rw [if_pos (by decide), root153Child0]
def root153Candidates1 : List Nat := [16187, 23399]
set_option maxRecDepth 8192 in
theorem root153Candidates1_checked : candidates edge 4 911 1103 root153Tail2 winner.bound = root153Candidates1 := by
  unfold candidates
  have row : edge 1103 = (SortedNatTree.contains FastRows.row1103) := by
    change Data.row1103.contains = (SortedNatTree.contains FastRows.row1103)
    exact FastRows.row1103_checked.symm
  rw [row]
  decide
theorem root153Step1 : scan edge (search edge 3) 4 [911] 911 root153Tail1 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail2 winner := by
  rw [root153Tail1, scan, if_neg (by decide), root153Candidates1_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail2
    (if 3 ≤ root153Candidates1.length then search edge 3 [911, 1103] root153Candidates1 2014 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates2 : List Nat := [11681]
set_option maxRecDepth 8192 in
theorem root153Candidates2_checked : candidates edge 4 911 1229 root153Tail3 winner.bound = root153Candidates2 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
theorem root153Step2 : scan edge (search edge 3) 4 [911] 911 root153Tail2 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail3 winner := by
  rw [root153Tail2, scan, if_neg (by decide), root153Candidates2_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail3
    (if 3 ≤ root153Candidates2.length then search edge 3 [911, 1229] root153Candidates2 2140 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates3 : List Nat := [18917]
set_option maxRecDepth 8192 in
theorem root153Candidates3_checked : candidates edge 4 911 1511 root153Tail4 winner.bound = root153Candidates3 := by
  unfold candidates
  have row : edge 1511 = (SortedNatTree.contains FastRows.row1511) := by
    change Data.row1511.contains = (SortedNatTree.contains FastRows.row1511)
    exact FastRows.row1511_checked.symm
  rw [row]
  decide
theorem root153Step3 : scan edge (search edge 3) 4 [911] 911 root153Tail3 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail4 winner := by
  rw [root153Tail3, scan, if_neg (by decide), root153Candidates3_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail4
    (if 3 ≤ root153Candidates3.length then search edge 3 [911, 1511] root153Candidates3 2422 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates4 : List Nat := [21227]
set_option maxRecDepth 8192 in
theorem root153Candidates4_checked : candidates edge 4 911 2309 root153Tail5 winner.bound = root153Candidates4 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
theorem root153Step4 : scan edge (search edge 3) 4 [911] 911 root153Tail4 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail5 winner := by
  rw [root153Tail4, scan, if_neg (by decide), root153Candidates4_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail5
    (if 3 ≤ root153Candidates4.length then search edge 3 [911, 2309] root153Candidates4 3220 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates5 : List Nat := [3821, 4517, 15077]
set_option maxRecDepth 8192 in
theorem root153Candidates5_checked : candidates edge 4 911 2399 root153Tail6 winner.bound = root153Candidates5 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root153Child5 : search edge 3 [911, 2399] root153Candidates5 3310 winner = winner := by decide
theorem root153Step5 : scan edge (search edge 3) 4 [911] 911 root153Tail5 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail6 winner := by
  rw [root153Tail5, scan, if_neg (by decide), root153Candidates5_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail6
    (if 3 ≤ root153Candidates5.length then search edge 3 [911, 2399] root153Candidates5 3310 winner else winner) = _
  rw [if_pos (by decide), root153Child5]
def root153Candidates6 : List Nat := [4517, 18917, 21143]
set_option maxRecDepth 8192 in
theorem root153Candidates6_checked : candidates edge 4 911 2657 root153Tail7 winner.bound = root153Candidates6 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root153Child6 : search edge 3 [911, 2657] root153Candidates6 3568 winner = winner := by decide
theorem root153Step6 : scan edge (search edge 3) 4 [911] 911 root153Tail6 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail7 winner := by
  rw [root153Tail6, scan, if_neg (by decide), root153Candidates6_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail7
    (if 3 ≤ root153Candidates6.length then search edge 3 [911, 2657] root153Candidates6 3568 winner else winner) = _
  rw [if_pos (by decide), root153Child6]
def root153Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root153Candidates7_checked : candidates edge 4 911 3413 root153Tail8 winner.bound = root153Candidates7 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root153Step7 : scan edge (search edge 3) 4 [911] 911 root153Tail7 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail8 winner := by
  rw [root153Tail7, scan, if_neg (by decide), root153Candidates7_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail8
    (if 3 ≤ root153Candidates7.length then search edge 3 [911, 3413] root153Candidates7 4324 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates8 : List Nat := [18917]
set_option maxRecDepth 8192 in
theorem root153Candidates8_checked : candidates edge 4 911 3821 root153Tail9 winner.bound = root153Candidates8 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root153Step8 : scan edge (search edge 3) 4 [911] 911 root153Tail8 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail9 winner := by
  rw [root153Tail8, scan, if_neg (by decide), root153Candidates8_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail9
    (if 3 ≤ root153Candidates8.length then search edge 3 [911, 3821] root153Candidates8 4732 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates9 : List Nat := [5471, 11903, 13613]
set_option maxRecDepth 8192 in
theorem root153Candidates9_checked : candidates edge 4 911 4337 root153Tail10 winner.bound = root153Candidates9 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root153Child9 : search edge 3 [911, 4337] root153Candidates9 5248 winner = winner := by decide
theorem root153Step9 : scan edge (search edge 3) 4 [911] 911 root153Tail9 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail10 winner := by
  rw [root153Tail9, scan, if_neg (by decide), root153Candidates9_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail10
    (if 3 ≤ root153Candidates9.length then search edge 3 [911, 4337] root153Candidates9 5248 winner else winner) = _
  rw [if_pos (by decide), root153Child9]
def root153Candidates10 : List Nat := [12473]
set_option maxRecDepth 8192 in
theorem root153Candidates10_checked : candidates edge 4 911 4517 root153Tail11 winner.bound = root153Candidates10 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root153Step10 : scan edge (search edge 3) 4 [911] 911 root153Tail10 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail11 winner := by
  rw [root153Tail10, scan, if_neg (by decide), root153Candidates10_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail11
    (if 3 ≤ root153Candidates10.length then search edge 3 [911, 4517] root153Candidates10 5428 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root153Candidates11_checked : candidates edge 4 911 4583 root153Tail12 winner.bound = root153Candidates11 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root153Step11 : scan edge (search edge 3) 4 [911] 911 root153Tail11 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail12 winner := by
  rw [root153Tail11, scan, if_neg (by decide), root153Candidates11_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail12
    (if 3 ≤ root153Candidates11.length then search edge 3 [911, 4583] root153Candidates11 5494 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates12 : List Nat := [5471]
set_option maxRecDepth 8192 in
theorem root153Candidates12_checked : candidates edge 4 911 4643 root153Tail13 winner.bound = root153Candidates12 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root153Step12 : scan edge (search edge 3) 4 [911] 911 root153Tail12 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail13 winner := by
  rw [root153Tail12, scan, if_neg (by decide), root153Candidates12_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail13
    (if 3 ≤ root153Candidates12.length then search edge 3 [911, 4643] root153Candidates12 5554 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates13 : List Nat := [12479]
set_option maxRecDepth 8192 in
theorem root153Candidates13_checked : candidates edge 4 911 5303 root153Tail14 winner.bound = root153Candidates13 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root153Step13 : scan edge (search edge 3) 4 [911] 911 root153Tail13 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail14 winner := by
  rw [root153Tail13, scan, if_neg (by decide), root153Candidates13_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail14
    (if 3 ≤ root153Candidates13.length then search edge 3 [911, 5303] root153Candidates13 6214 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates14 : List Nat := [13751]
set_option maxRecDepth 8192 in
theorem root153Candidates14_checked : candidates edge 4 911 5471 root153Tail15 winner.bound = root153Candidates14 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root153Step14 : scan edge (search edge 3) 4 [911] 911 root153Tail14 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail15 winner := by
  rw [root153Tail14, scan, if_neg (by decide), root153Candidates14_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail15
    (if 3 ≤ root153Candidates14.length then search edge 3 [911, 5471] root153Candidates14 6382 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates15 : List Nat := [8111, 9749]
set_option maxRecDepth 8192 in
theorem root153Candidates15_checked : candidates edge 4 911 6089 root153Tail16 winner.bound = root153Candidates15 := by
  unfold candidates
  have row : edge 6089 = (SortedNatTree.contains FastRows.row6089) := by
    change Data.row6089.contains = (SortedNatTree.contains FastRows.row6089)
    exact FastRows.row6089_checked.symm
  rw [row]
  decide
theorem root153Step15 : scan edge (search edge 3) 4 [911] 911 root153Tail15 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail16 winner := by
  rw [root153Tail15, scan, if_neg (by decide), root153Candidates15_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail16
    (if 3 ≤ root153Candidates15.length then search edge 3 [911, 6089] root153Candidates15 7000 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates16 : List Nat := [15077]
set_option maxRecDepth 8192 in
theorem root153Candidates16_checked : candidates edge 4 911 6113 root153Tail17 winner.bound = root153Candidates16 := by
  unfold candidates
  have row : edge 6113 = (SortedNatTree.contains FastRows.row6113) := by
    change Data.row6113.contains = (SortedNatTree.contains FastRows.row6113)
    exact FastRows.row6113_checked.symm
  rw [row]
  decide
theorem root153Step16 : scan edge (search edge 3) 4 [911] 911 root153Tail16 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail17 winner := by
  rw [root153Tail16, scan, if_neg (by decide), root153Candidates16_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail17
    (if 3 ≤ root153Candidates16.length then search edge 3 [911, 6113] root153Candidates16 7024 winner else winner) = _
  rw [if_neg (by decide)]
def root153Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root153Candidates17_checked : candidates edge 4 911 6269 root153Tail18 winner.bound = root153Candidates17 := by
  unfold candidates
  have row : edge 6269 = (SortedNatTree.contains FastRows.row6269) := by
    change Data.row6269.contains = (SortedNatTree.contains FastRows.row6269)
    exact FastRows.row6269_checked.symm
  rw [row]
  decide
theorem root153Step17 : scan edge (search edge 3) 4 [911] 911 root153Tail17 winner =
    scan edge (search edge 3) 4 [911] 911 root153Tail18 winner := by
  rw [root153Tail17, scan, if_neg (by decide), root153Candidates17_checked]
  change scan edge (search edge 3) 4 [911] 911 root153Tail18
    (if 3 ≤ root153Candidates17.length then search edge 3 [911, 6269] root153Candidates17 7180 winner else winner) = _
  rw [if_neg (by decide)]
theorem root153Scan18 : scan edge (search edge 3) 4 [911] 911 root153Tail18 winner = winner := by rw [root153Tail18, scan, if_pos (by decide)]
theorem root153Scan17 : scan edge (search edge 3) 4 [911] 911 root153Tail17 winner = winner :=
  (root153Step17).trans root153Scan18
theorem root153Scan16 : scan edge (search edge 3) 4 [911] 911 root153Tail16 winner = winner :=
  (root153Step16).trans root153Scan17
theorem root153Scan15 : scan edge (search edge 3) 4 [911] 911 root153Tail15 winner = winner :=
  (root153Step15).trans root153Scan16
theorem root153Scan14 : scan edge (search edge 3) 4 [911] 911 root153Tail14 winner = winner :=
  (root153Step14).trans root153Scan15
theorem root153Scan13 : scan edge (search edge 3) 4 [911] 911 root153Tail13 winner = winner :=
  (root153Step13).trans root153Scan14
theorem root153Scan12 : scan edge (search edge 3) 4 [911] 911 root153Tail12 winner = winner :=
  (root153Step12).trans root153Scan13
theorem root153Scan11 : scan edge (search edge 3) 4 [911] 911 root153Tail11 winner = winner :=
  (root153Step11).trans root153Scan12
theorem root153Scan10 : scan edge (search edge 3) 4 [911] 911 root153Tail10 winner = winner :=
  (root153Step10).trans root153Scan11
theorem root153Scan9 : scan edge (search edge 3) 4 [911] 911 root153Tail9 winner = winner :=
  (root153Step9).trans root153Scan10
theorem root153Scan8 : scan edge (search edge 3) 4 [911] 911 root153Tail8 winner = winner :=
  (root153Step8).trans root153Scan9
theorem root153Scan7 : scan edge (search edge 3) 4 [911] 911 root153Tail7 winner = winner :=
  (root153Step7).trans root153Scan8
theorem root153Scan6 : scan edge (search edge 3) 4 [911] 911 root153Tail6 winner = winner :=
  (root153Step6).trans root153Scan7
theorem root153Scan5 : scan edge (search edge 3) 4 [911] 911 root153Tail5 winner = winner :=
  (root153Step5).trans root153Scan6
theorem root153Scan4 : scan edge (search edge 3) 4 [911] 911 root153Tail4 winner = winner :=
  (root153Step4).trans root153Scan5
theorem root153Scan3 : scan edge (search edge 3) 4 [911] 911 root153Tail3 winner = winner :=
  (root153Step3).trans root153Scan4
theorem root153Scan2 : scan edge (search edge 3) 4 [911] 911 root153Tail2 winner = winner :=
  (root153Step2).trans root153Scan3
theorem root153Scan1 : scan edge (search edge 3) 4 [911] 911 root153Tail1 winner = winner :=
  (root153Step1).trans root153Scan2
theorem root153Scan0 : scan edge (search edge 3) 4 [911] 911 root153Tail0 winner = winner :=
  (root153Step0).trans root153Scan1
set_option maxRecDepth 4096 in
theorem root153Branch : search edge 4 [911] root153Candidates 911 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [911] 911 root153Tail0 winner = winner
  exact root153Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
