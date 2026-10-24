import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root108Candidates : List Nat := Data.row607
set_option maxRecDepth 8192 in
theorem root108Candidates_checked : candidates edge 5 0 607 tail109 winner.bound = root108Candidates := by
  apply candidates_eq_row edge 5 0 607 tail109 root108Candidates winner.bound tail109_ordered
  · decide
  · decide
  · funext q
    rfl
def root108Tail75 : List Nat := []
def root108Tail74 : List Nat := 25243 :: root108Tail75
def root108Tail73 : List Nat := 25237 :: root108Tail74
def root108Tail72 : List Nat := 25153 :: root108Tail73
def root108Tail71 : List Nat := 25117 :: root108Tail72
def root108Tail70 : List Nat := 24151 :: root108Tail71
def root108Tail69 : List Nat := 24121 :: root108Tail70
def root108Tail68 : List Nat := 24049 :: root108Tail69
def root108Tail67 : List Nat := 23431 :: root108Tail68
def root108Tail66 : List Nat := 23251 :: root108Tail67
def root108Tail65 : List Nat := 22303 :: root108Tail66
def root108Tail64 : List Nat := 21559 :: root108Tail65
def root108Tail63 : List Nat := 21391 :: root108Tail64
def root108Tail62 : List Nat := 21319 :: root108Tail63
def root108Tail61 : List Nat := 19819 :: root108Tail62
def root108Tail60 : List Nat := 19753 :: root108Tail61
def root108Tail59 : List Nat := 19501 :: root108Tail60
def root108Tail58 : List Nat := 19273 :: root108Tail59
def root108Tail57 : List Nat := 18979 :: root108Tail58
def root108Tail56 : List Nat := 18787 :: root108Tail57
def root108Tail55 : List Nat := 18523 :: root108Tail56
def root108Tail54 : List Nat := 18439 :: root108Tail55
def root108Tail53 : List Nat := 17971 :: root108Tail54
def root108Tail52 : List Nat := 15121 :: root108Tail53
def root108Tail51 : List Nat := 14713 :: root108Tail52
def root108Tail50 : List Nat := 14683 :: root108Tail51
def root108Tail49 : List Nat := 14629 :: root108Tail50
def root108Tail48 : List Nat := 14389 :: root108Tail49
def root108Tail47 : List Nat := 12889 :: root108Tail48
def root108Tail46 : List Nat := 12781 :: root108Tail47
def root108Tail45 : List Nat := 12373 :: root108Tail46
def root108Tail44 : List Nat := 11971 :: root108Tail45
def root108Tail43 : List Nat := 11941 :: root108Tail44
def root108Tail42 : List Nat := 11923 :: root108Tail43
def root108Tail41 : List Nat := 11383 :: root108Tail42
def root108Tail40 : List Nat := 11197 :: root108Tail41
def root108Tail39 : List Nat := 10831 :: root108Tail40
def root108Tail38 : List Nat := 10333 :: root108Tail39
def root108Tail37 : List Nat := 10159 :: root108Tail38
def root108Tail36 : List Nat := 9547 :: root108Tail37
def root108Tail35 : List Nat := 9391 :: root108Tail36
def root108Tail34 : List Nat := 9043 :: root108Tail35
def root108Tail33 : List Nat := 8731 :: root108Tail34
def root108Tail32 : List Nat := 8599 :: root108Tail33
def root108Tail31 : List Nat := 8581 :: root108Tail32
def root108Tail30 : List Nat := 8389 :: root108Tail31
def root108Tail29 : List Nat := 8353 :: root108Tail30
def root108Tail28 : List Nat := 8287 :: root108Tail29
def root108Tail27 : List Nat := 7639 :: root108Tail28
def root108Tail26 : List Nat := 7573 :: root108Tail27
def root108Tail25 : List Nat := 7039 :: root108Tail26
def root108Tail24 : List Nat := 6883 :: root108Tail25
def root108Tail23 : List Nat := 6709 :: root108Tail24
def root108Tail22 : List Nat := 6673 :: root108Tail23
def root108Tail21 : List Nat := 6337 :: root108Tail22
def root108Tail20 : List Nat := 6121 :: root108Tail21
def root108Tail19 : List Nat := 5107 :: root108Tail20
def root108Tail18 : List Nat := 5077 :: root108Tail19
def root108Tail17 : List Nat := 4603 :: root108Tail18
def root108Tail16 : List Nat := 4591 :: root108Tail17
def root108Tail15 : List Nat := 4561 :: root108Tail16
def root108Tail14 : List Nat := 4483 :: root108Tail15
def root108Tail13 : List Nat := 4447 :: root108Tail14
def root108Tail12 : List Nat := 4363 :: root108Tail13
def root108Tail11 : List Nat := 3889 :: root108Tail12
def root108Tail10 : List Nat := 3733 :: root108Tail11
def root108Tail9 : List Nat := 3037 :: root108Tail10
def root108Tail8 : List Nat := 2887 :: root108Tail9
def root108Tail7 : List Nat := 2467 :: root108Tail8
def root108Tail6 : List Nat := 1999 :: root108Tail7
def root108Tail5 : List Nat := 1693 :: root108Tail6
def root108Tail4 : List Nat := 1453 :: root108Tail5
def root108Tail3 : List Nat := 1297 :: root108Tail4
def root108Tail2 : List Nat := 991 :: root108Tail3
def root108Tail1 : List Nat := 967 :: root108Tail2
def root108Tail0 : List Nat := 619 :: root108Tail1
def root108Candidates0 : List Nat := [10831, 14683, 24151]
set_option maxRecDepth 8192 in
theorem root108Candidates0_checked : candidates edge 4 607 619 root108Tail1 winner.bound = root108Candidates0 := by
  unfold candidates
  have row : edge 619 = (SortedNatTree.contains FastRows.row619) := by
    change Data.row619.contains = (SortedNatTree.contains FastRows.row619)
    exact FastRows.row619_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root108Child0 : search edge 3 [607, 619] root108Candidates0 1226 winner = winner := by decide
theorem root108Step0 : scan edge (search edge 3) 4 [607] 607 root108Tail0 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail1 winner := by
  rw [root108Tail0, scan, if_neg (by decide), root108Candidates0_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail1
    (if 3 ≤ root108Candidates0.length then search edge 3 [607, 619] root108Candidates0 1226 winner else winner) = _
  rw [if_pos (by decide), root108Child0]
def root108Candidates1 : List Nat := [12373, 14713, 19501, 24049]
set_option maxRecDepth 8192 in
theorem root108Candidates1_checked : candidates edge 4 607 967 root108Tail2 winner.bound = root108Candidates1 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root108Child1 : search edge 3 [607, 967] root108Candidates1 1574 winner = winner := by decide
theorem root108Step1 : scan edge (search edge 3) 4 [607] 607 root108Tail1 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail2 winner := by
  rw [root108Tail1, scan, if_neg (by decide), root108Candidates1_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail2
    (if 3 ≤ root108Candidates1.length then search edge 3 [607, 967] root108Candidates1 1574 winner else winner) = _
  rw [if_pos (by decide), root108Child1]
def root108Candidates2 : List Nat := [18787]
set_option maxRecDepth 8192 in
theorem root108Candidates2_checked : candidates edge 4 607 991 root108Tail3 winner.bound = root108Candidates2 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
theorem root108Step2 : scan edge (search edge 3) 4 [607] 607 root108Tail2 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail3 winner := by
  rw [root108Tail2, scan, if_neg (by decide), root108Candidates2_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail3
    (if 3 ≤ root108Candidates2.length then search edge 3 [607, 991] root108Candidates2 1598 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates3 : List Nat := [8731, 11383, 15121]
set_option maxRecDepth 8192 in
theorem root108Candidates3_checked : candidates edge 4 607 1297 root108Tail4 winner.bound = root108Candidates3 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root108Child3 : search edge 3 [607, 1297] root108Candidates3 1904 winner = winner := by decide
theorem root108Step3 : scan edge (search edge 3) 4 [607] 607 root108Tail3 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail4 winner := by
  rw [root108Tail3, scan, if_neg (by decide), root108Candidates3_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail4
    (if 3 ≤ root108Candidates3.length then search edge 3 [607, 1297] root108Candidates3 1904 winner else winner) = _
  rw [if_pos (by decide), root108Child3]
def root108Candidates4 : List Nat := [4483, 17971]
set_option maxRecDepth 8192 in
theorem root108Candidates4_checked : candidates edge 4 607 1453 root108Tail5 winner.bound = root108Candidates4 := by
  unfold candidates
  have row : edge 1453 = (SortedNatTree.contains FastRows.row1453) := by
    change Data.row1453.contains = (SortedNatTree.contains FastRows.row1453)
    exact FastRows.row1453_checked.symm
  rw [row]
  decide
theorem root108Step4 : scan edge (search edge 3) 4 [607] 607 root108Tail4 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail5 winner := by
  rw [root108Tail4, scan, if_neg (by decide), root108Candidates4_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail5
    (if 3 ≤ root108Candidates4.length then search edge 3 [607, 1453] root108Candidates4 2060 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates5 : List Nat := [18439, 19753]
set_option maxRecDepth 8192 in
theorem root108Candidates5_checked : candidates edge 4 607 1693 root108Tail6 winner.bound = root108Candidates5 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
theorem root108Step5 : scan edge (search edge 3) 4 [607] 607 root108Tail5 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail6 winner := by
  rw [root108Tail5, scan, if_neg (by decide), root108Candidates5_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail6
    (if 3 ≤ root108Candidates5.length then search edge 3 [607, 1693] root108Candidates5 2300 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates6 : List Nat := [3733, 6121]
set_option maxRecDepth 8192 in
theorem root108Candidates6_checked : candidates edge 4 607 1999 root108Tail7 winner.bound = root108Candidates6 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root108Step6 : scan edge (search edge 3) 4 [607] 607 root108Tail6 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail7 winner := by
  rw [root108Tail6, scan, if_neg (by decide), root108Candidates6_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail7
    (if 3 ≤ root108Candidates6.length then search edge 3 [607, 1999] root108Candidates6 2606 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root108Candidates7_checked : candidates edge 4 607 2467 root108Tail8 winner.bound = root108Candidates7 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root108Step7 : scan edge (search edge 3) 4 [607] 607 root108Tail7 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail8 winner := by
  rw [root108Tail7, scan, if_neg (by decide), root108Candidates7_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail8
    (if 3 ≤ root108Candidates7.length then search edge 3 [607, 2467] root108Candidates7 3074 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates8 : List Nat := [6709, 7573]
set_option maxRecDepth 8192 in
theorem root108Candidates8_checked : candidates edge 4 607 2887 root108Tail9 winner.bound = root108Candidates8 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root108Step8 : scan edge (search edge 3) 4 [607] 607 root108Tail8 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail9 winner := by
  rw [root108Tail8, scan, if_neg (by decide), root108Candidates8_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail9
    (if 3 ≤ root108Candidates8.length then search edge 3 [607, 2887] root108Candidates8 3494 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates9 : List Nat := [11971]
set_option maxRecDepth 8192 in
theorem root108Candidates9_checked : candidates edge 4 607 3037 root108Tail10 winner.bound = root108Candidates9 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
theorem root108Step9 : scan edge (search edge 3) 4 [607] 607 root108Tail9 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail10 winner := by
  rw [root108Tail9, scan, if_neg (by decide), root108Candidates9_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail10
    (if 3 ≤ root108Candidates9.length then search edge 3 [607, 3037] root108Candidates9 3644 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates10 : List Nat := [10159, 11941, 11971, 14629]
set_option maxRecDepth 8192 in
theorem root108Candidates10_checked : candidates edge 4 607 3733 root108Tail11 winner.bound = root108Candidates10 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root108Child10 : search edge 3 [607, 3733] root108Candidates10 4340 winner = winner := by decide
theorem root108Step10 : scan edge (search edge 3) 4 [607] 607 root108Tail10 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail11 winner := by
  rw [root108Tail10, scan, if_neg (by decide), root108Candidates10_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail11
    (if 3 ≤ root108Candidates10.length then search edge 3 [607, 3733] root108Candidates10 4340 winner else winner) = _
  rw [if_pos (by decide), root108Child10]
def root108Candidates11 : List Nat := [12889, 19819]
set_option maxRecDepth 8192 in
theorem root108Candidates11_checked : candidates edge 4 607 3889 root108Tail12 winner.bound = root108Candidates11 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root108Step11 : scan edge (search edge 3) 4 [607] 607 root108Tail11 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail12 winner := by
  rw [root108Tail11, scan, if_neg (by decide), root108Candidates11_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail12
    (if 3 ≤ root108Candidates11.length then search edge 3 [607, 3889] root108Candidates11 4496 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root108Candidates12_checked : candidates edge 4 607 4363 root108Tail13 winner.bound = root108Candidates12 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root108Step12 : scan edge (search edge 3) 4 [607] 607 root108Tail12 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail13 winner := by
  rw [root108Tail12, scan, if_neg (by decide), root108Candidates12_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail13
    (if 3 ≤ root108Candidates12.length then search edge 3 [607, 4363] root108Candidates12 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates13 : List Nat := [10831, 11941, 18787]
set_option maxRecDepth 8192 in
theorem root108Candidates13_checked : candidates edge 4 607 4447 root108Tail14 winner.bound = root108Candidates13 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root108Child13 : search edge 3 [607, 4447] root108Candidates13 5054 winner = winner := by decide
theorem root108Step13 : scan edge (search edge 3) 4 [607] 607 root108Tail13 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail14 winner := by
  rw [root108Tail13, scan, if_neg (by decide), root108Candidates13_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail14
    (if 3 ≤ root108Candidates13.length then search edge 3 [607, 4447] root108Candidates13 5054 winner else winner) = _
  rw [if_pos (by decide), root108Child13]
def root108Candidates14 : List Nat := [11923]
set_option maxRecDepth 8192 in
theorem root108Candidates14_checked : candidates edge 4 607 4483 root108Tail15 winner.bound = root108Candidates14 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root108Step14 : scan edge (search edge 3) 4 [607] 607 root108Tail14 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail15 winner := by
  rw [root108Tail14, scan, if_neg (by decide), root108Candidates14_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail15
    (if 3 ≤ root108Candidates14.length then search edge 3 [607, 4483] root108Candidates14 5090 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates15 : List Nat := [8389]
set_option maxRecDepth 8192 in
theorem root108Candidates15_checked : candidates edge 4 607 4561 root108Tail16 winner.bound = root108Candidates15 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root108Step15 : scan edge (search edge 3) 4 [607] 607 root108Tail15 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail16 winner := by
  rw [root108Tail15, scan, if_neg (by decide), root108Candidates15_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail16
    (if 3 ≤ root108Candidates15.length then search edge 3 [607, 4561] root108Candidates15 5168 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates16 : List Nat := [6121, 8287]
set_option maxRecDepth 8192 in
theorem root108Candidates16_checked : candidates edge 4 607 4591 root108Tail17 winner.bound = root108Candidates16 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root108Step16 : scan edge (search edge 3) 4 [607] 607 root108Tail16 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail17 winner := by
  rw [root108Tail16, scan, if_neg (by decide), root108Candidates16_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail17
    (if 3 ≤ root108Candidates16.length then search edge 3 [607, 4591] root108Candidates16 5198 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates17 : List Nat := [8599, 8731]
set_option maxRecDepth 8192 in
theorem root108Candidates17_checked : candidates edge 4 607 4603 root108Tail18 winner.bound = root108Candidates17 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root108Step17 : scan edge (search edge 3) 4 [607] 607 root108Tail17 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail18 winner := by
  rw [root108Tail17, scan, if_neg (by decide), root108Candidates17_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail18
    (if 3 ≤ root108Candidates17.length then search edge 3 [607, 4603] root108Candidates17 5210 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root108Candidates18_checked : candidates edge 4 607 5077 root108Tail19 winner.bound = root108Candidates18 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root108Step18 : scan edge (search edge 3) 4 [607] 607 root108Tail18 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail19 winner := by
  rw [root108Tail18, scan, if_neg (by decide), root108Candidates18_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail19
    (if 3 ≤ root108Candidates18.length then search edge 3 [607, 5077] root108Candidates18 5684 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates19 : List Nat := [12373]
set_option maxRecDepth 8192 in
theorem root108Candidates19_checked : candidates edge 4 607 5107 root108Tail20 winner.bound = root108Candidates19 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root108Step19 : scan edge (search edge 3) 4 [607] 607 root108Tail19 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail20 winner := by
  rw [root108Tail19, scan, if_neg (by decide), root108Candidates19_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail20
    (if 3 ≤ root108Candidates19.length then search edge 3 [607, 5107] root108Candidates19 5714 winner else winner) = _
  rw [if_neg (by decide)]
def root108Candidates20 : List Nat := [8389, 9043, 10831]
set_option maxRecDepth 8192 in
theorem root108Candidates20_checked : candidates edge 4 607 6121 root108Tail21 winner.bound = root108Candidates20 := by
  unfold candidates
  have row : edge 6121 = (SortedNatTree.contains FastRows.row6121) := by
    change Data.row6121.contains = (SortedNatTree.contains FastRows.row6121)
    exact FastRows.row6121_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root108Child20 : search edge 3 [607, 6121] root108Candidates20 6728 winner = winner := by decide
theorem root108Step20 : scan edge (search edge 3) 4 [607] 607 root108Tail20 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail21 winner := by
  rw [root108Tail20, scan, if_neg (by decide), root108Candidates20_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail21
    (if 3 ≤ root108Candidates20.length then search edge 3 [607, 6121] root108Candidates20 6728 winner else winner) = _
  rw [if_pos (by decide), root108Child20]
def root108Candidates21 : List Nat := [10159]
set_option maxRecDepth 8192 in
theorem root108Candidates21_checked : candidates edge 4 607 6337 root108Tail22 winner.bound = root108Candidates21 := by
  unfold candidates
  have row : edge 6337 = (SortedNatTree.contains FastRows.row6337) := by
    change Data.row6337.contains = (SortedNatTree.contains FastRows.row6337)
    exact FastRows.row6337_checked.symm
  rw [row]
  decide
theorem root108Step21 : scan edge (search edge 3) 4 [607] 607 root108Tail21 winner =
    scan edge (search edge 3) 4 [607] 607 root108Tail22 winner := by
  rw [root108Tail21, scan, if_neg (by decide), root108Candidates21_checked]
  change scan edge (search edge 3) 4 [607] 607 root108Tail22
    (if 3 ≤ root108Candidates21.length then search edge 3 [607, 6337] root108Candidates21 6944 winner else winner) = _
  rw [if_neg (by decide)]
theorem root108Scan22 : scan edge (search edge 3) 4 [607] 607 root108Tail22 winner = winner := by rw [root108Tail22, scan, if_pos (by decide)]
theorem root108Scan21 : scan edge (search edge 3) 4 [607] 607 root108Tail21 winner = winner :=
  (root108Step21).trans root108Scan22
theorem root108Scan20 : scan edge (search edge 3) 4 [607] 607 root108Tail20 winner = winner :=
  (root108Step20).trans root108Scan21
theorem root108Scan19 : scan edge (search edge 3) 4 [607] 607 root108Tail19 winner = winner :=
  (root108Step19).trans root108Scan20
theorem root108Scan18 : scan edge (search edge 3) 4 [607] 607 root108Tail18 winner = winner :=
  (root108Step18).trans root108Scan19
theorem root108Scan17 : scan edge (search edge 3) 4 [607] 607 root108Tail17 winner = winner :=
  (root108Step17).trans root108Scan18
theorem root108Scan16 : scan edge (search edge 3) 4 [607] 607 root108Tail16 winner = winner :=
  (root108Step16).trans root108Scan17
theorem root108Scan15 : scan edge (search edge 3) 4 [607] 607 root108Tail15 winner = winner :=
  (root108Step15).trans root108Scan16
theorem root108Scan14 : scan edge (search edge 3) 4 [607] 607 root108Tail14 winner = winner :=
  (root108Step14).trans root108Scan15
theorem root108Scan13 : scan edge (search edge 3) 4 [607] 607 root108Tail13 winner = winner :=
  (root108Step13).trans root108Scan14
theorem root108Scan12 : scan edge (search edge 3) 4 [607] 607 root108Tail12 winner = winner :=
  (root108Step12).trans root108Scan13
theorem root108Scan11 : scan edge (search edge 3) 4 [607] 607 root108Tail11 winner = winner :=
  (root108Step11).trans root108Scan12
theorem root108Scan10 : scan edge (search edge 3) 4 [607] 607 root108Tail10 winner = winner :=
  (root108Step10).trans root108Scan11
theorem root108Scan9 : scan edge (search edge 3) 4 [607] 607 root108Tail9 winner = winner :=
  (root108Step9).trans root108Scan10
theorem root108Scan8 : scan edge (search edge 3) 4 [607] 607 root108Tail8 winner = winner :=
  (root108Step8).trans root108Scan9
theorem root108Scan7 : scan edge (search edge 3) 4 [607] 607 root108Tail7 winner = winner :=
  (root108Step7).trans root108Scan8
theorem root108Scan6 : scan edge (search edge 3) 4 [607] 607 root108Tail6 winner = winner :=
  (root108Step6).trans root108Scan7
theorem root108Scan5 : scan edge (search edge 3) 4 [607] 607 root108Tail5 winner = winner :=
  (root108Step5).trans root108Scan6
theorem root108Scan4 : scan edge (search edge 3) 4 [607] 607 root108Tail4 winner = winner :=
  (root108Step4).trans root108Scan5
theorem root108Scan3 : scan edge (search edge 3) 4 [607] 607 root108Tail3 winner = winner :=
  (root108Step3).trans root108Scan4
theorem root108Scan2 : scan edge (search edge 3) 4 [607] 607 root108Tail2 winner = winner :=
  (root108Step2).trans root108Scan3
theorem root108Scan1 : scan edge (search edge 3) 4 [607] 607 root108Tail1 winner = winner :=
  (root108Step1).trans root108Scan2
theorem root108Scan0 : scan edge (search edge 3) 4 [607] 607 root108Tail0 winner = winner :=
  (root108Step0).trans root108Scan1
set_option maxRecDepth 4096 in
theorem root108Branch : search edge 4 [607] root108Candidates 607 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [607] 607 root108Tail0 winner = winner
  exact root108Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
