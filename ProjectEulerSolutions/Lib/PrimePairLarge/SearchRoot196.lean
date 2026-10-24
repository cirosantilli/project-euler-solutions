import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root196Candidates : List Nat := Data.row1217
set_option maxRecDepth 8192 in
theorem root196Candidates_checked : candidates edge 5 0 1217 tail197 winner.bound = root196Candidates := by
  apply candidates_eq_row edge 5 0 1217 tail197 root196Candidates winner.bound tail197_ordered
  · decide
  · decide
  · funext q
    rfl
def root196Tail60 : List Nat := []
def root196Tail59 : List Nat := 24419 :: root196Tail60
def root196Tail58 : List Nat := 24359 :: root196Tail59
def root196Tail57 : List Nat := 24077 :: root196Tail58
def root196Tail56 : List Nat := 23603 :: root196Tail57
def root196Tail55 : List Nat := 23201 :: root196Tail56
def root196Tail54 : List Nat := 23099 :: root196Tail55
def root196Tail53 : List Nat := 21503 :: root196Tail54
def root196Tail52 : List Nat := 21101 :: root196Tail53
def root196Tail51 : List Nat := 20873 :: root196Tail52
def root196Tail50 : List Nat := 20399 :: root196Tail51
def root196Tail49 : List Nat := 20129 :: root196Tail50
def root196Tail48 : List Nat := 19457 :: root196Tail49
def root196Tail47 : List Nat := 18719 :: root196Tail48
def root196Tail46 : List Nat := 18329 :: root196Tail47
def root196Tail45 : List Nat := 17891 :: root196Tail46
def root196Tail44 : List Nat := 17333 :: root196Tail45
def root196Tail43 : List Nat := 16937 :: root196Tail44
def root196Tail42 : List Nat := 16883 :: root196Tail43
def root196Tail41 : List Nat := 16811 :: root196Tail42
def root196Tail40 : List Nat := 16631 :: root196Tail41
def root196Tail39 : List Nat := 15959 :: root196Tail40
def root196Tail38 : List Nat := 15551 :: root196Tail39
def root196Tail37 : List Nat := 15263 :: root196Tail38
def root196Tail36 : List Nat := 13367 :: root196Tail37
def root196Tail35 : List Nat := 12653 :: root196Tail36
def root196Tail34 : List Nat := 9749 :: root196Tail35
def root196Tail33 : List Nat := 9689 :: root196Tail34
def root196Tail32 : List Nat := 9491 :: root196Tail33
def root196Tail31 : List Nat := 8861 :: root196Tail32
def root196Tail30 : List Nat := 8627 :: root196Tail31
def root196Tail29 : List Nat := 8609 :: root196Tail30
def root196Tail28 : List Nat := 8093 :: root196Tail29
def root196Tail27 : List Nat := 7817 :: root196Tail28
def root196Tail26 : List Nat := 7019 :: root196Tail27
def root196Tail25 : List Nat := 7001 :: root196Tail26
def root196Tail24 : List Nat := 6173 :: root196Tail25
def root196Tail23 : List Nat := 5879 :: root196Tail24
def root196Tail22 : List Nat := 5573 :: root196Tail23
def root196Tail21 : List Nat := 5477 :: root196Tail22
def root196Tail20 : List Nat := 5393 :: root196Tail21
def root196Tail19 : List Nat := 5153 :: root196Tail20
def root196Tail18 : List Nat := 5081 :: root196Tail19
def root196Tail17 : List Nat := 5003 :: root196Tail18
def root196Tail16 : List Nat := 4931 :: root196Tail17
def root196Tail15 : List Nat := 3671 :: root196Tail16
def root196Tail14 : List Nat := 2903 :: root196Tail15
def root196Tail13 : List Nat := 2843 :: root196Tail14
def root196Tail12 : List Nat := 2591 :: root196Tail13
def root196Tail11 : List Nat := 2477 :: root196Tail12
def root196Tail10 : List Nat := 2423 :: root196Tail11
def root196Tail9 : List Nat := 2393 :: root196Tail10
def root196Tail8 : List Nat := 2309 :: root196Tail9
def root196Tail7 : List Nat := 2273 :: root196Tail8
def root196Tail6 : List Nat := 2087 :: root196Tail7
def root196Tail5 : List Nat := 2003 :: root196Tail6
def root196Tail4 : List Nat := 1721 :: root196Tail5
def root196Tail3 : List Nat := 1583 :: root196Tail4
def root196Tail2 : List Nat := 1433 :: root196Tail3
def root196Tail1 : List Nat := 1427 :: root196Tail2
def root196Tail0 : List Nat := 1361 :: root196Tail1
def root196Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root196Candidates0_checked : candidates edge 4 1217 1361 root196Tail1 winner.bound = root196Candidates0 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
theorem root196Step0 : scan edge (search edge 3) 4 [1217] 1217 root196Tail0 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail1 winner := by
  rw [root196Tail0, scan, if_neg (by decide), root196Candidates0_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail1
    (if 3 ≤ root196Candidates0.length then search edge 3 [1217, 1361] root196Candidates0 2578 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates1 : List Nat := [5477]
set_option maxRecDepth 8192 in
theorem root196Candidates1_checked : candidates edge 4 1217 1427 root196Tail2 winner.bound = root196Candidates1 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
theorem root196Step1 : scan edge (search edge 3) 4 [1217] 1217 root196Tail1 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail2 winner := by
  rw [root196Tail1, scan, if_neg (by decide), root196Candidates1_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail2
    (if 3 ≤ root196Candidates1.length then search edge 3 [1217, 1427] root196Candidates1 2644 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates2 : List Nat := [5393]
set_option maxRecDepth 8192 in
theorem root196Candidates2_checked : candidates edge 4 1217 1433 root196Tail3 winner.bound = root196Candidates2 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
theorem root196Step2 : scan edge (search edge 3) 4 [1217] 1217 root196Tail2 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail3 winner := by
  rw [root196Tail2, scan, if_neg (by decide), root196Candidates2_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail3
    (if 3 ≤ root196Candidates2.length then search edge 3 [1217, 1433] root196Candidates2 2650 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates3 : List Nat := [5879, 16811]
set_option maxRecDepth 8192 in
theorem root196Candidates3_checked : candidates edge 4 1217 1583 root196Tail4 winner.bound = root196Candidates3 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
theorem root196Step3 : scan edge (search edge 3) 4 [1217] 1217 root196Tail3 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail4 winner := by
  rw [root196Tail3, scan, if_neg (by decide), root196Candidates3_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail4
    (if 3 ≤ root196Candidates3.length then search edge 3 [1217, 1583] root196Candidates3 2800 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates4 : List Nat := [8627, 9749, 18329]
set_option maxRecDepth 8192 in
theorem root196Candidates4_checked : candidates edge 4 1217 1721 root196Tail5 winner.bound = root196Candidates4 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root196Child4 : search edge 3 [1217, 1721] root196Candidates4 2938 winner = winner := by decide
theorem root196Step4 : scan edge (search edge 3) 4 [1217] 1217 root196Tail4 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail5 winner := by
  rw [root196Tail4, scan, if_neg (by decide), root196Candidates4_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail5
    (if 3 ≤ root196Candidates4.length then search edge 3 [1217, 1721] root196Candidates4 2938 winner else winner) = _
  rw [if_pos (by decide), root196Child4]
def root196Candidates5 : List Nat := [2087, 5153, 8609, 15959]
set_option maxRecDepth 8192 in
theorem root196Candidates5_checked : candidates edge 4 1217 2003 root196Tail6 winner.bound = root196Candidates5 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root196Child5 : search edge 3 [1217, 2003] root196Candidates5 3220 winner = winner := by decide
theorem root196Step5 : scan edge (search edge 3) 4 [1217] 1217 root196Tail5 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail6 winner := by
  rw [root196Tail5, scan, if_neg (by decide), root196Candidates5_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail6
    (if 3 ≤ root196Candidates5.length then search edge 3 [1217, 2003] root196Candidates5 3220 winner else winner) = _
  rw [if_pos (by decide), root196Child5]
def root196Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root196Candidates6_checked : candidates edge 4 1217 2087 root196Tail7 winner.bound = root196Candidates6 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root196Step6 : scan edge (search edge 3) 4 [1217] 1217 root196Tail6 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail7 winner := by
  rw [root196Tail6, scan, if_neg (by decide), root196Candidates6_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail7
    (if 3 ≤ root196Candidates6.length then search edge 3 [1217, 2087] root196Candidates6 3304 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates7 : List Nat := [2903, 8627, 8861]
set_option maxRecDepth 8192 in
theorem root196Candidates7_checked : candidates edge 4 1217 2273 root196Tail8 winner.bound = root196Candidates7 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root196Child7 : search edge 3 [1217, 2273] root196Candidates7 3490 winner = winner := by decide
theorem root196Step7 : scan edge (search edge 3) 4 [1217] 1217 root196Tail7 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail8 winner := by
  rw [root196Tail7, scan, if_neg (by decide), root196Candidates7_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail8
    (if 3 ≤ root196Candidates7.length then search edge 3 [1217, 2273] root196Candidates7 3490 winner else winner) = _
  rw [if_pos (by decide), root196Child7]
def root196Candidates8 : List Nat := [5081, 7817, 8861, 15551]
set_option maxRecDepth 8192 in
theorem root196Candidates8_checked : candidates edge 4 1217 2309 root196Tail9 winner.bound = root196Candidates8 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root196Child8 : search edge 3 [1217, 2309] root196Candidates8 3526 winner = winner := by decide
theorem root196Step8 : scan edge (search edge 3) 4 [1217] 1217 root196Tail8 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail9 winner := by
  rw [root196Tail8, scan, if_neg (by decide), root196Candidates8_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail9
    (if 3 ≤ root196Candidates8.length then search edge 3 [1217, 2309] root196Candidates8 3526 winner else winner) = _
  rw [if_pos (by decide), root196Child8]
def root196Candidates9 : List Nat := [2423, 9491]
set_option maxRecDepth 8192 in
theorem root196Candidates9_checked : candidates edge 4 1217 2393 root196Tail10 winner.bound = root196Candidates9 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root196Step9 : scan edge (search edge 3) 4 [1217] 1217 root196Tail9 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail10 winner := by
  rw [root196Tail9, scan, if_neg (by decide), root196Candidates9_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail10
    (if 3 ≤ root196Candidates9.length then search edge 3 [1217, 2393] root196Candidates9 3610 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates10 : List Nat := [8609, 21503]
set_option maxRecDepth 8192 in
theorem root196Candidates10_checked : candidates edge 4 1217 2423 root196Tail11 winner.bound = root196Candidates10 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root196Step10 : scan edge (search edge 3) 4 [1217] 1217 root196Tail10 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail11 winner := by
  rw [root196Tail10, scan, if_neg (by decide), root196Candidates10_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail11
    (if 3 ≤ root196Candidates10.length then search edge 3 [1217, 2423] root196Candidates10 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates11 : List Nat := [17891]
set_option maxRecDepth 8192 in
theorem root196Candidates11_checked : candidates edge 4 1217 2477 root196Tail12 winner.bound = root196Candidates11 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
theorem root196Step11 : scan edge (search edge 3) 4 [1217] 1217 root196Tail11 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail12 winner := by
  rw [root196Tail11, scan, if_neg (by decide), root196Candidates11_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail12
    (if 3 ≤ root196Candidates11.length then search edge 3 [1217, 2477] root196Candidates11 3694 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates12 : List Nat := [15959]
set_option maxRecDepth 8192 in
theorem root196Candidates12_checked : candidates edge 4 1217 2591 root196Tail13 winner.bound = root196Candidates12 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root196Step12 : scan edge (search edge 3) 4 [1217] 1217 root196Tail12 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail13 winner := by
  rw [root196Tail12, scan, if_neg (by decide), root196Candidates12_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail13
    (if 3 ≤ root196Candidates12.length then search edge 3 [1217, 2591] root196Candidates12 3808 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates13 : List Nat := [12653]
set_option maxRecDepth 8192 in
theorem root196Candidates13_checked : candidates edge 4 1217 2843 root196Tail14 winner.bound = root196Candidates13 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root196Step13 : scan edge (search edge 3) 4 [1217] 1217 root196Tail13 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail14 winner := by
  rw [root196Tail13, scan, if_neg (by decide), root196Candidates13_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail14
    (if 3 ≤ root196Candidates13.length then search edge 3 [1217, 2843] root196Candidates13 4060 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root196Candidates14_checked : candidates edge 4 1217 2903 root196Tail15 winner.bound = root196Candidates14 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root196Step14 : scan edge (search edge 3) 4 [1217] 1217 root196Tail14 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail15 winner := by
  rw [root196Tail14, scan, if_neg (by decide), root196Candidates14_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail15
    (if 3 ≤ root196Candidates14.length then search edge 3 [1217, 2903] root196Candidates14 4120 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates15 : List Nat := [7019]
set_option maxRecDepth 8192 in
theorem root196Candidates15_checked : candidates edge 4 1217 3671 root196Tail16 winner.bound = root196Candidates15 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root196Step15 : scan edge (search edge 3) 4 [1217] 1217 root196Tail15 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail16 winner := by
  rw [root196Tail15, scan, if_neg (by decide), root196Candidates15_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail16
    (if 3 ≤ root196Candidates15.length then search edge 3 [1217, 3671] root196Candidates15 4888 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates16 : List Nat := [7817]
set_option maxRecDepth 8192 in
theorem root196Candidates16_checked : candidates edge 4 1217 4931 root196Tail17 winner.bound = root196Candidates16 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root196Step16 : scan edge (search edge 3) 4 [1217] 1217 root196Tail16 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail17 winner := by
  rw [root196Tail16, scan, if_neg (by decide), root196Candidates16_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail17
    (if 3 ≤ root196Candidates16.length then search edge 3 [1217, 4931] root196Candidates16 6148 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root196Candidates17_checked : candidates edge 4 1217 5003 root196Tail18 winner.bound = root196Candidates17 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root196Step17 : scan edge (search edge 3) 4 [1217] 1217 root196Tail17 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail18 winner := by
  rw [root196Tail17, scan, if_neg (by decide), root196Candidates17_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail18
    (if 3 ≤ root196Candidates17.length then search edge 3 [1217, 5003] root196Candidates17 6220 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates18 : List Nat := [7817, 18329]
set_option maxRecDepth 8192 in
theorem root196Candidates18_checked : candidates edge 4 1217 5081 root196Tail19 winner.bound = root196Candidates18 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root196Step18 : scan edge (search edge 3) 4 [1217] 1217 root196Tail18 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail19 winner := by
  rw [root196Tail18, scan, if_neg (by decide), root196Candidates18_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail19
    (if 3 ≤ root196Candidates18.length then search edge 3 [1217, 5081] root196Candidates18 6298 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates19 : List Nat := [9491]
set_option maxRecDepth 8192 in
theorem root196Candidates19_checked : candidates edge 4 1217 5153 root196Tail20 winner.bound = root196Candidates19 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root196Step19 : scan edge (search edge 3) 4 [1217] 1217 root196Tail19 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail20 winner := by
  rw [root196Tail19, scan, if_neg (by decide), root196Candidates19_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail20
    (if 3 ≤ root196Candidates19.length then search edge 3 [1217, 5153] root196Candidates19 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates20 : List Nat := [7817]
set_option maxRecDepth 8192 in
theorem root196Candidates20_checked : candidates edge 4 1217 5393 root196Tail21 winner.bound = root196Candidates20 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root196Step20 : scan edge (search edge 3) 4 [1217] 1217 root196Tail20 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail21 winner := by
  rw [root196Tail20, scan, if_neg (by decide), root196Candidates20_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail21
    (if 3 ≤ root196Candidates20.length then search edge 3 [1217, 5393] root196Candidates20 6610 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates21 : List Nat := [15263]
set_option maxRecDepth 8192 in
theorem root196Candidates21_checked : candidates edge 4 1217 5477 root196Tail22 winner.bound = root196Candidates21 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root196Step21 : scan edge (search edge 3) 4 [1217] 1217 root196Tail21 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail22 winner := by
  rw [root196Tail21, scan, if_neg (by decide), root196Candidates21_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail22
    (if 3 ≤ root196Candidates21.length then search edge 3 [1217, 5477] root196Candidates21 6694 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root196Candidates22_checked : candidates edge 4 1217 5573 root196Tail23 winner.bound = root196Candidates22 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root196Step22 : scan edge (search edge 3) 4 [1217] 1217 root196Tail22 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail23 winner := by
  rw [root196Tail22, scan, if_neg (by decide), root196Candidates22_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail23
    (if 3 ≤ root196Candidates22.length then search edge 3 [1217, 5573] root196Candidates22 6790 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates23 : List Nat := [16883]
set_option maxRecDepth 8192 in
theorem root196Candidates23_checked : candidates edge 4 1217 5879 root196Tail24 winner.bound = root196Candidates23 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root196Step23 : scan edge (search edge 3) 4 [1217] 1217 root196Tail23 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail24 winner := by
  rw [root196Tail23, scan, if_neg (by decide), root196Candidates23_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail24
    (if 3 ≤ root196Candidates23.length then search edge 3 [1217, 5879] root196Candidates23 7096 winner else winner) = _
  rw [if_neg (by decide)]
def root196Candidates24 : List Nat := [15551]
set_option maxRecDepth 8192 in
theorem root196Candidates24_checked : candidates edge 4 1217 6173 root196Tail25 winner.bound = root196Candidates24 := by
  unfold candidates
  have row : edge 6173 = (SortedNatTree.contains FastRows.row6173) := by
    change Data.row6173.contains = (SortedNatTree.contains FastRows.row6173)
    exact FastRows.row6173_checked.symm
  rw [row]
  decide
theorem root196Step24 : scan edge (search edge 3) 4 [1217] 1217 root196Tail24 winner =
    scan edge (search edge 3) 4 [1217] 1217 root196Tail25 winner := by
  rw [root196Tail24, scan, if_neg (by decide), root196Candidates24_checked]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail25
    (if 3 ≤ root196Candidates24.length then search edge 3 [1217, 6173] root196Candidates24 7390 winner else winner) = _
  rw [if_neg (by decide)]
theorem root196Scan25 : scan edge (search edge 3) 4 [1217] 1217 root196Tail25 winner = winner := by rw [root196Tail25, scan, if_pos (by decide)]
theorem root196Scan24 : scan edge (search edge 3) 4 [1217] 1217 root196Tail24 winner = winner :=
  (root196Step24).trans root196Scan25
theorem root196Scan23 : scan edge (search edge 3) 4 [1217] 1217 root196Tail23 winner = winner :=
  (root196Step23).trans root196Scan24
theorem root196Scan22 : scan edge (search edge 3) 4 [1217] 1217 root196Tail22 winner = winner :=
  (root196Step22).trans root196Scan23
theorem root196Scan21 : scan edge (search edge 3) 4 [1217] 1217 root196Tail21 winner = winner :=
  (root196Step21).trans root196Scan22
theorem root196Scan20 : scan edge (search edge 3) 4 [1217] 1217 root196Tail20 winner = winner :=
  (root196Step20).trans root196Scan21
theorem root196Scan19 : scan edge (search edge 3) 4 [1217] 1217 root196Tail19 winner = winner :=
  (root196Step19).trans root196Scan20
theorem root196Scan18 : scan edge (search edge 3) 4 [1217] 1217 root196Tail18 winner = winner :=
  (root196Step18).trans root196Scan19
theorem root196Scan17 : scan edge (search edge 3) 4 [1217] 1217 root196Tail17 winner = winner :=
  (root196Step17).trans root196Scan18
theorem root196Scan16 : scan edge (search edge 3) 4 [1217] 1217 root196Tail16 winner = winner :=
  (root196Step16).trans root196Scan17
theorem root196Scan15 : scan edge (search edge 3) 4 [1217] 1217 root196Tail15 winner = winner :=
  (root196Step15).trans root196Scan16
theorem root196Scan14 : scan edge (search edge 3) 4 [1217] 1217 root196Tail14 winner = winner :=
  (root196Step14).trans root196Scan15
theorem root196Scan13 : scan edge (search edge 3) 4 [1217] 1217 root196Tail13 winner = winner :=
  (root196Step13).trans root196Scan14
theorem root196Scan12 : scan edge (search edge 3) 4 [1217] 1217 root196Tail12 winner = winner :=
  (root196Step12).trans root196Scan13
theorem root196Scan11 : scan edge (search edge 3) 4 [1217] 1217 root196Tail11 winner = winner :=
  (root196Step11).trans root196Scan12
theorem root196Scan10 : scan edge (search edge 3) 4 [1217] 1217 root196Tail10 winner = winner :=
  (root196Step10).trans root196Scan11
theorem root196Scan9 : scan edge (search edge 3) 4 [1217] 1217 root196Tail9 winner = winner :=
  (root196Step9).trans root196Scan10
theorem root196Scan8 : scan edge (search edge 3) 4 [1217] 1217 root196Tail8 winner = winner :=
  (root196Step8).trans root196Scan9
theorem root196Scan7 : scan edge (search edge 3) 4 [1217] 1217 root196Tail7 winner = winner :=
  (root196Step7).trans root196Scan8
theorem root196Scan6 : scan edge (search edge 3) 4 [1217] 1217 root196Tail6 winner = winner :=
  (root196Step6).trans root196Scan7
theorem root196Scan5 : scan edge (search edge 3) 4 [1217] 1217 root196Tail5 winner = winner :=
  (root196Step5).trans root196Scan6
theorem root196Scan4 : scan edge (search edge 3) 4 [1217] 1217 root196Tail4 winner = winner :=
  (root196Step4).trans root196Scan5
theorem root196Scan3 : scan edge (search edge 3) 4 [1217] 1217 root196Tail3 winner = winner :=
  (root196Step3).trans root196Scan4
theorem root196Scan2 : scan edge (search edge 3) 4 [1217] 1217 root196Tail2 winner = winner :=
  (root196Step2).trans root196Scan3
theorem root196Scan1 : scan edge (search edge 3) 4 [1217] 1217 root196Tail1 winner = winner :=
  (root196Step1).trans root196Scan2
theorem root196Scan0 : scan edge (search edge 3) 4 [1217] 1217 root196Tail0 winner = winner :=
  (root196Step0).trans root196Scan1
set_option maxRecDepth 4096 in
theorem root196Branch : search edge 4 [1217] root196Candidates 1217 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1217] 1217 root196Tail0 winner = winner
  exact root196Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
