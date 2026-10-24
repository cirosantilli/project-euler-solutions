import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root50Candidates : List Nat := Data.row241
set_option maxRecDepth 8192 in
theorem root50Candidates_checked : candidates edge 5 0 241 tail51 winner.bound = root50Candidates := by
  apply candidates_eq_row edge 5 0 241 tail51 root50Candidates winner.bound tail51_ordered
  · decide
  · decide
  · funext q
    rfl
def root50Tail81 : List Nat := []
def root50Tail80 : List Nat := 25639 :: root50Tail81
def root50Tail79 : List Nat := 25243 :: root50Tail80
def root50Tail78 : List Nat := 25237 :: root50Tail79
def root50Tail77 : List Nat := 24943 :: root50Tail78
def root50Tail76 : List Nat := 24229 :: root50Tail77
def root50Tail75 : List Nat := 23773 :: root50Tail76
def root50Tail74 : List Nat := 23623 :: root50Tail75
def root50Tail73 : List Nat := 23581 :: root50Tail74
def root50Tail72 : List Nat := 23497 :: root50Tail73
def root50Tail71 : List Nat := 22807 :: root50Tail72
def root50Tail70 : List Nat := 22741 :: root50Tail71
def root50Tail69 : List Nat := 22573 :: root50Tail70
def root50Tail68 : List Nat := 21529 :: root50Tail69
def root50Tail67 : List Nat := 20731 :: root50Tail68
def root50Tail66 : List Nat := 20143 :: root50Tail67
def root50Tail65 : List Nat := 20029 :: root50Tail66
def root50Tail64 : List Nat := 19699 :: root50Tail65
def root50Tail63 : List Nat := 19681 :: root50Tail64
def root50Tail62 : List Nat := 19423 :: root50Tail63
def root50Tail61 : List Nat := 18229 :: root50Tail62
def root50Tail60 : List Nat := 17683 :: root50Tail61
def root50Tail59 : List Nat := 17551 :: root50Tail60
def root50Tail58 : List Nat := 17257 :: root50Tail59
def root50Tail57 : List Nat := 16993 :: root50Tail58
def root50Tail56 : List Nat := 16879 :: root50Tail57
def root50Tail55 : List Nat := 16363 :: root50Tail56
def root50Tail54 : List Nat := 16333 :: root50Tail55
def root50Tail53 : List Nat := 16273 :: root50Tail54
def root50Tail52 : List Nat := 16087 :: root50Tail53
def root50Tail51 : List Nat := 15859 :: root50Tail52
def root50Tail50 : List Nat := 15439 :: root50Tail51
def root50Tail49 : List Nat := 15307 :: root50Tail50
def root50Tail48 : List Nat := 15031 :: root50Tail49
def root50Tail47 : List Nat := 14821 :: root50Tail48
def root50Tail46 : List Nat := 14533 :: root50Tail47
def root50Tail45 : List Nat := 14221 :: root50Tail46
def root50Tail44 : List Nat := 14107 :: root50Tail45
def root50Tail43 : List Nat := 13963 :: root50Tail44
def root50Tail42 : List Nat := 12973 :: root50Tail43
def root50Tail41 : List Nat := 11587 :: root50Tail42
def root50Tail40 : List Nat := 10903 :: root50Tail41
def root50Tail39 : List Nat := 10789 :: root50Tail40
def root50Tail38 : List Nat := 10729 :: root50Tail39
def root50Tail37 : List Nat := 10453 :: root50Tail38
def root50Tail36 : List Nat := 10399 :: root50Tail37
def root50Tail35 : List Nat := 10159 :: root50Tail36
def root50Tail34 : List Nat := 9883 :: root50Tail35
def root50Tail33 : List Nat := 9721 :: root50Tail34
def root50Tail32 : List Nat := 9463 :: root50Tail33
def root50Tail31 : List Nat := 9433 :: root50Tail32
def root50Tail30 : List Nat := 9007 :: root50Tail31
def root50Tail29 : List Nat := 8329 :: root50Tail30
def root50Tail28 : List Nat := 7741 :: root50Tail29
def root50Tail27 : List Nat := 6763 :: root50Tail28
def root50Tail26 : List Nat := 6607 :: root50Tail27
def root50Tail25 : List Nat := 6361 :: root50Tail26
def root50Tail24 : List Nat := 5689 :: root50Tail25
def root50Tail23 : List Nat := 5503 :: root50Tail24
def root50Tail22 : List Nat := 5431 :: root50Tail23
def root50Tail21 : List Nat := 4549 :: root50Tail22
def root50Tail20 : List Nat := 4129 :: root50Tail21
def root50Tail19 : List Nat := 3823 :: root50Tail20
def root50Tail18 : List Nat := 3511 :: root50Tail19
def root50Tail17 : List Nat := 3253 :: root50Tail18
def root50Tail16 : List Nat := 3217 :: root50Tail17
def root50Tail15 : List Nat := 2833 :: root50Tail16
def root50Tail14 : List Nat := 2797 :: root50Tail15
def root50Tail13 : List Nat := 2731 :: root50Tail14
def root50Tail12 : List Nat := 2671 :: root50Tail13
def root50Tail11 : List Nat := 2029 :: root50Tail12
def root50Tail10 : List Nat := 1867 :: root50Tail11
def root50Tail9 : List Nat := 883 :: root50Tail10
def root50Tail8 : List Nat := 823 :: root50Tail9
def root50Tail7 : List Nat := 811 :: root50Tail8
def root50Tail6 : List Nat := 739 :: root50Tail7
def root50Tail5 : List Nat := 727 :: root50Tail6
def root50Tail4 : List Nat := 691 :: root50Tail5
def root50Tail3 : List Nat := 601 :: root50Tail4
def root50Tail2 : List Nat := 421 :: root50Tail3
def root50Tail1 : List Nat := 313 :: root50Tail2
def root50Tail0 : List Nat := 271 :: root50Tail1
def root50Candidates0 : List Nat := [9433, 15031, 16087, 25243]
set_option maxRecDepth 8192 in
theorem root50Candidates0_checked : candidates edge 4 241 271 root50Tail1 winner.bound = root50Candidates0 := by
  unfold candidates
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child0 : search edge 3 [241, 271] root50Candidates0 512 winner = winner := by decide
theorem root50Step0 : scan edge (search edge 3) 4 [241] 241 root50Tail0 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail1 winner := by
  rw [root50Tail0, scan, if_neg (by decide), root50Candidates0_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail1
    (if 3 ≤ root50Candidates0.length then search edge 3 [241, 271] root50Candidates0 512 winner else winner) = _
  rw [if_pos (by decide), root50Child0]
def root50Candidates1 : List Nat := [727, 8329, 11587, 17683]
set_option maxRecDepth 8192 in
theorem root50Candidates1_checked : candidates edge 4 241 313 root50Tail2 winner.bound = root50Candidates1 := by
  unfold candidates
  have row : edge 313 = (SortedNatTree.contains FastRows.row313) := by
    change Data.row313.contains = (SortedNatTree.contains FastRows.row313)
    exact FastRows.row313_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child1 : search edge 3 [241, 313] root50Candidates1 554 winner = winner := by decide
theorem root50Step1 : scan edge (search edge 3) 4 [241] 241 root50Tail1 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail2 winner := by
  rw [root50Tail1, scan, if_neg (by decide), root50Candidates1_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail2
    (if 3 ≤ root50Candidates1.length then search edge 3 [241, 313] root50Candidates1 554 winner else winner) = _
  rw [if_pos (by decide), root50Child1]
def root50Candidates2 : List Nat := [3217]
set_option maxRecDepth 8192 in
theorem root50Candidates2_checked : candidates edge 4 241 421 root50Tail3 winner.bound = root50Candidates2 := by
  unfold candidates
  have row : edge 421 = (SortedNatTree.contains FastRows.row421) := by
    change Data.row421.contains = (SortedNatTree.contains FastRows.row421)
    exact FastRows.row421_checked.symm
  rw [row]
  decide
theorem root50Step2 : scan edge (search edge 3) 4 [241] 241 root50Tail2 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail3 winner := by
  rw [root50Tail2, scan, if_neg (by decide), root50Candidates2_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail3
    (if 3 ≤ root50Candidates2.length then search edge 3 [241, 421] root50Candidates2 662 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates3 : List Nat := [823, 1867, 16087, 19681]
set_option maxRecDepth 8192 in
theorem root50Candidates3_checked : candidates edge 4 241 601 root50Tail4 winner.bound = root50Candidates3 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child3 : search edge 3 [241, 601] root50Candidates3 842 winner = winner := by decide
theorem root50Step3 : scan edge (search edge 3) 4 [241] 241 root50Tail3 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail4 winner := by
  rw [root50Tail3, scan, if_neg (by decide), root50Candidates3_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail4
    (if 3 ≤ root50Candidates3.length then search edge 3 [241, 601] root50Candidates3 842 winner else winner) = _
  rw [if_pos (by decide), root50Child3]
def root50Candidates4 : List Nat := [727, 5431]
set_option maxRecDepth 8192 in
theorem root50Candidates4_checked : candidates edge 4 241 691 root50Tail5 winner.bound = root50Candidates4 := by
  unfold candidates
  have row : edge 691 = (SortedNatTree.contains FastRows.row691) := by
    change Data.row691.contains = (SortedNatTree.contains FastRows.row691)
    exact FastRows.row691_checked.symm
  rw [row]
  decide
theorem root50Step4 : scan edge (search edge 3) 4 [241] 241 root50Tail4 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail5 winner := by
  rw [root50Tail4, scan, if_neg (by decide), root50Candidates4_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail5
    (if 3 ≤ root50Candidates4.length then search edge 3 [241, 691] root50Candidates4 932 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates5 : List Nat := [823, 1867, 5503, 20143]
set_option maxRecDepth 8192 in
theorem root50Candidates5_checked : candidates edge 4 241 727 root50Tail6 winner.bound = root50Candidates5 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child5 : search edge 3 [241, 727] root50Candidates5 968 winner = winner := by decide
theorem root50Step5 : scan edge (search edge 3) 4 [241] 241 root50Tail5 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail6 winner := by
  rw [root50Tail5, scan, if_neg (by decide), root50Candidates5_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail6
    (if 3 ≤ root50Candidates5.length then search edge 3 [241, 727] root50Candidates5 968 winner else winner) = _
  rw [if_pos (by decide), root50Child5]
def root50Candidates6 : List Nat := [4549, 9463, 16333]
set_option maxRecDepth 8192 in
theorem root50Candidates6_checked : candidates edge 4 241 739 root50Tail7 winner.bound = root50Candidates6 := by
  unfold candidates
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child6 : search edge 3 [241, 739] root50Candidates6 980 winner = winner := by decide
theorem root50Step6 : scan edge (search edge 3) 4 [241] 241 root50Tail6 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail7 winner := by
  rw [root50Tail6, scan, if_neg (by decide), root50Candidates6_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail7
    (if 3 ≤ root50Candidates6.length then search edge 3 [241, 739] root50Candidates6 980 winner else winner) = _
  rw [if_pos (by decide), root50Child6]
def root50Candidates7 : List Nat := [15439, 17551, 18229]
set_option maxRecDepth 8192 in
theorem root50Candidates7_checked : candidates edge 4 241 811 root50Tail8 winner.bound = root50Candidates7 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child7 : search edge 3 [241, 811] root50Candidates7 1052 winner = winner := by decide
theorem root50Step7 : scan edge (search edge 3) 4 [241] 241 root50Tail7 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail8 winner := by
  rw [root50Tail7, scan, if_neg (by decide), root50Candidates7_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail8
    (if 3 ≤ root50Candidates7.length then search edge 3 [241, 811] root50Candidates7 1052 winner else winner) = _
  rw [if_pos (by decide), root50Child7]
def root50Candidates8 : List Nat := [13963]
set_option maxRecDepth 8192 in
theorem root50Candidates8_checked : candidates edge 4 241 823 root50Tail9 winner.bound = root50Candidates8 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
theorem root50Step8 : scan edge (search edge 3) 4 [241] 241 root50Tail8 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail9 winner := by
  rw [root50Tail8, scan, if_neg (by decide), root50Candidates8_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail9
    (if 3 ≤ root50Candidates8.length then search edge 3 [241, 823] root50Candidates8 1064 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates9 : List Nat := [9007, 22573]
set_option maxRecDepth 8192 in
theorem root50Candidates9_checked : candidates edge 4 241 883 root50Tail10 winner.bound = root50Candidates9 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
theorem root50Step9 : scan edge (search edge 3) 4 [241] 241 root50Tail9 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail10 winner := by
  rw [root50Tail9, scan, if_neg (by decide), root50Candidates9_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail10
    (if 3 ≤ root50Candidates9.length then search edge 3 [241, 883] root50Candidates9 1124 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates10 : List Nat := [10789, 15031]
set_option maxRecDepth 8192 in
theorem root50Candidates10_checked : candidates edge 4 241 1867 root50Tail11 winner.bound = root50Candidates10 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
theorem root50Step10 : scan edge (search edge 3) 4 [241] 241 root50Tail10 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail11 winner := by
  rw [root50Tail10, scan, if_neg (by decide), root50Candidates10_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail11
    (if 3 ≤ root50Candidates10.length then search edge 3 [241, 1867] root50Candidates10 2108 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates11 : List Nat := [5689, 19681]
set_option maxRecDepth 8192 in
theorem root50Candidates11_checked : candidates edge 4 241 2029 root50Tail12 winner.bound = root50Candidates11 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
theorem root50Step11 : scan edge (search edge 3) 4 [241] 241 root50Tail11 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail12 winner := by
  rw [root50Tail11, scan, if_neg (by decide), root50Candidates11_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail12
    (if 3 ≤ root50Candidates11.length then search edge 3 [241, 2029] root50Candidates11 2270 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates12 : List Nat := [10159]
set_option maxRecDepth 8192 in
theorem root50Candidates12_checked : candidates edge 4 241 2671 root50Tail13 winner.bound = root50Candidates12 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root50Step12 : scan edge (search edge 3) 4 [241] 241 root50Tail12 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail13 winner := by
  rw [root50Tail12, scan, if_neg (by decide), root50Candidates12_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail13
    (if 3 ≤ root50Candidates12.length then search edge 3 [241, 2671] root50Candidates12 2912 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates13 : List Nat := [4129, 14221, 15439, 18229]
set_option maxRecDepth 8192 in
theorem root50Candidates13_checked : candidates edge 4 241 2731 root50Tail14 winner.bound = root50Candidates13 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child13 : search edge 3 [241, 2731] root50Candidates13 2972 winner = winner := by decide
theorem root50Step13 : scan edge (search edge 3) 4 [241] 241 root50Tail13 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail14 winner := by
  rw [root50Tail13, scan, if_neg (by decide), root50Candidates13_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail14
    (if 3 ≤ root50Candidates13.length then search edge 3 [241, 2731] root50Candidates13 2972 winner else winner) = _
  rw [if_pos (by decide), root50Child13]
def root50Candidates14 : List Nat := [13963]
set_option maxRecDepth 8192 in
theorem root50Candidates14_checked : candidates edge 4 241 2797 root50Tail15 winner.bound = root50Candidates14 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root50Step14 : scan edge (search edge 3) 4 [241] 241 root50Tail14 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail15 winner := by
  rw [root50Tail14, scan, if_neg (by decide), root50Candidates14_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail15
    (if 3 ≤ root50Candidates14.length then search edge 3 [241, 2797] root50Candidates14 3038 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates15 : List Nat := [6361]
set_option maxRecDepth 8192 in
theorem root50Candidates15_checked : candidates edge 4 241 2833 root50Tail16 winner.bound = root50Candidates15 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root50Step15 : scan edge (search edge 3) 4 [241] 241 root50Tail15 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail16 winner := by
  rw [root50Tail15, scan, if_neg (by decide), root50Candidates15_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail16
    (if 3 ≤ root50Candidates15.length then search edge 3 [241, 2833] root50Candidates15 3074 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates16 : List Nat := [3511, 19681, 20731]
set_option maxRecDepth 8192 in
theorem root50Candidates16_checked : candidates edge 4 241 3217 root50Tail17 winner.bound = root50Candidates16 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child16 : search edge 3 [241, 3217] root50Candidates16 3458 winner = winner := by decide
theorem root50Step16 : scan edge (search edge 3) 4 [241] 241 root50Tail16 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail17 winner := by
  rw [root50Tail16, scan, if_neg (by decide), root50Candidates16_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail17
    (if 3 ≤ root50Candidates16.length then search edge 3 [241, 3217] root50Candidates16 3458 winner else winner) = _
  rw [if_pos (by decide), root50Child16]
def root50Candidates17 : List Nat := [17257]
set_option maxRecDepth 8192 in
theorem root50Candidates17_checked : candidates edge 4 241 3253 root50Tail18 winner.bound = root50Candidates17 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root50Step17 : scan edge (search edge 3) 4 [241] 241 root50Tail17 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail18 winner := by
  rw [root50Tail17, scan, if_neg (by decide), root50Candidates17_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail18
    (if 3 ≤ root50Candidates17.length then search edge 3 [241, 3253] root50Candidates17 3494 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root50Candidates18_checked : candidates edge 4 241 3511 root50Tail19 winner.bound = root50Candidates18 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root50Step18 : scan edge (search edge 3) 4 [241] 241 root50Tail18 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail19 winner := by
  rw [root50Tail18, scan, if_neg (by decide), root50Candidates18_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail19
    (if 3 ≤ root50Candidates18.length then search edge 3 [241, 3511] root50Candidates18 3752 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates19 : List Nat := [4549, 15859]
set_option maxRecDepth 8192 in
theorem root50Candidates19_checked : candidates edge 4 241 3823 root50Tail20 winner.bound = root50Candidates19 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root50Step19 : scan edge (search edge 3) 4 [241] 241 root50Tail19 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail20 winner := by
  rw [root50Tail19, scan, if_neg (by decide), root50Candidates19_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail20
    (if 3 ≤ root50Candidates19.length then search edge 3 [241, 3823] root50Candidates19 4064 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates20 : List Nat := [6361, 14533, 15859]
set_option maxRecDepth 8192 in
theorem root50Candidates20_checked : candidates edge 4 241 4129 root50Tail21 winner.bound = root50Candidates20 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child20 : search edge 3 [241, 4129] root50Candidates20 4370 winner = winner := by decide
theorem root50Step20 : scan edge (search edge 3) 4 [241] 241 root50Tail20 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail21 winner := by
  rw [root50Tail20, scan, if_neg (by decide), root50Candidates20_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail21
    (if 3 ≤ root50Candidates20.length then search edge 3 [241, 4129] root50Candidates20 4370 winner else winner) = _
  rw [if_pos (by decide), root50Child20]
def root50Candidates21 : List Nat := [6361, 15859, 19681]
set_option maxRecDepth 8192 in
theorem root50Candidates21_checked : candidates edge 4 241 4549 root50Tail22 winner.bound = root50Candidates21 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root50Child21 : search edge 3 [241, 4549] root50Candidates21 4790 winner = winner := by decide
theorem root50Step21 : scan edge (search edge 3) 4 [241] 241 root50Tail21 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail22 winner := by
  rw [root50Tail21, scan, if_neg (by decide), root50Candidates21_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail22
    (if 3 ≤ root50Candidates21.length then search edge 3 [241, 4549] root50Candidates21 4790 winner else winner) = _
  rw [if_pos (by decide), root50Child21]
def root50Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root50Candidates22_checked : candidates edge 4 241 5431 root50Tail23 winner.bound = root50Candidates22 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root50Step22 : scan edge (search edge 3) 4 [241] 241 root50Tail22 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail23 winner := by
  rw [root50Tail22, scan, if_neg (by decide), root50Candidates22_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail23
    (if 3 ≤ root50Candidates22.length then search edge 3 [241, 5431] root50Candidates22 5672 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root50Candidates23_checked : candidates edge 4 241 5503 root50Tail24 winner.bound = root50Candidates23 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root50Step23 : scan edge (search edge 3) 4 [241] 241 root50Tail23 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail24 winner := by
  rw [root50Tail23, scan, if_neg (by decide), root50Candidates23_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail24
    (if 3 ≤ root50Candidates23.length then search edge 3 [241, 5503] root50Candidates23 5744 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates24 : List Nat := [8329, 15031]
set_option maxRecDepth 8192 in
theorem root50Candidates24_checked : candidates edge 4 241 5689 root50Tail25 winner.bound = root50Candidates24 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root50Step24 : scan edge (search edge 3) 4 [241] 241 root50Tail24 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail25 winner := by
  rw [root50Tail24, scan, if_neg (by decide), root50Candidates24_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail25
    (if 3 ≤ root50Candidates24.length then search edge 3 [241, 5689] root50Candidates24 5930 winner else winner) = _
  rw [if_neg (by decide)]
def root50Candidates25 : List Nat := [10789, 12973]
set_option maxRecDepth 8192 in
theorem root50Candidates25_checked : candidates edge 4 241 6361 root50Tail26 winner.bound = root50Candidates25 := by
  unfold candidates
  have row : edge 6361 = (SortedNatTree.contains FastRows.row6361) := by
    change Data.row6361.contains = (SortedNatTree.contains FastRows.row6361)
    exact FastRows.row6361_checked.symm
  rw [row]
  decide
theorem root50Step25 : scan edge (search edge 3) 4 [241] 241 root50Tail25 winner =
    scan edge (search edge 3) 4 [241] 241 root50Tail26 winner := by
  rw [root50Tail25, scan, if_neg (by decide), root50Candidates25_checked]
  change scan edge (search edge 3) 4 [241] 241 root50Tail26
    (if 3 ≤ root50Candidates25.length then search edge 3 [241, 6361] root50Candidates25 6602 winner else winner) = _
  rw [if_neg (by decide)]
theorem root50Scan26 : scan edge (search edge 3) 4 [241] 241 root50Tail26 winner = winner := by rw [root50Tail26, scan, if_pos (by decide)]
theorem root50Scan25 : scan edge (search edge 3) 4 [241] 241 root50Tail25 winner = winner :=
  (root50Step25).trans root50Scan26
theorem root50Scan24 : scan edge (search edge 3) 4 [241] 241 root50Tail24 winner = winner :=
  (root50Step24).trans root50Scan25
theorem root50Scan23 : scan edge (search edge 3) 4 [241] 241 root50Tail23 winner = winner :=
  (root50Step23).trans root50Scan24
theorem root50Scan22 : scan edge (search edge 3) 4 [241] 241 root50Tail22 winner = winner :=
  (root50Step22).trans root50Scan23
theorem root50Scan21 : scan edge (search edge 3) 4 [241] 241 root50Tail21 winner = winner :=
  (root50Step21).trans root50Scan22
theorem root50Scan20 : scan edge (search edge 3) 4 [241] 241 root50Tail20 winner = winner :=
  (root50Step20).trans root50Scan21
theorem root50Scan19 : scan edge (search edge 3) 4 [241] 241 root50Tail19 winner = winner :=
  (root50Step19).trans root50Scan20
theorem root50Scan18 : scan edge (search edge 3) 4 [241] 241 root50Tail18 winner = winner :=
  (root50Step18).trans root50Scan19
theorem root50Scan17 : scan edge (search edge 3) 4 [241] 241 root50Tail17 winner = winner :=
  (root50Step17).trans root50Scan18
theorem root50Scan16 : scan edge (search edge 3) 4 [241] 241 root50Tail16 winner = winner :=
  (root50Step16).trans root50Scan17
theorem root50Scan15 : scan edge (search edge 3) 4 [241] 241 root50Tail15 winner = winner :=
  (root50Step15).trans root50Scan16
theorem root50Scan14 : scan edge (search edge 3) 4 [241] 241 root50Tail14 winner = winner :=
  (root50Step14).trans root50Scan15
theorem root50Scan13 : scan edge (search edge 3) 4 [241] 241 root50Tail13 winner = winner :=
  (root50Step13).trans root50Scan14
theorem root50Scan12 : scan edge (search edge 3) 4 [241] 241 root50Tail12 winner = winner :=
  (root50Step12).trans root50Scan13
theorem root50Scan11 : scan edge (search edge 3) 4 [241] 241 root50Tail11 winner = winner :=
  (root50Step11).trans root50Scan12
theorem root50Scan10 : scan edge (search edge 3) 4 [241] 241 root50Tail10 winner = winner :=
  (root50Step10).trans root50Scan11
theorem root50Scan9 : scan edge (search edge 3) 4 [241] 241 root50Tail9 winner = winner :=
  (root50Step9).trans root50Scan10
theorem root50Scan8 : scan edge (search edge 3) 4 [241] 241 root50Tail8 winner = winner :=
  (root50Step8).trans root50Scan9
theorem root50Scan7 : scan edge (search edge 3) 4 [241] 241 root50Tail7 winner = winner :=
  (root50Step7).trans root50Scan8
theorem root50Scan6 : scan edge (search edge 3) 4 [241] 241 root50Tail6 winner = winner :=
  (root50Step6).trans root50Scan7
theorem root50Scan5 : scan edge (search edge 3) 4 [241] 241 root50Tail5 winner = winner :=
  (root50Step5).trans root50Scan6
theorem root50Scan4 : scan edge (search edge 3) 4 [241] 241 root50Tail4 winner = winner :=
  (root50Step4).trans root50Scan5
theorem root50Scan3 : scan edge (search edge 3) 4 [241] 241 root50Tail3 winner = winner :=
  (root50Step3).trans root50Scan4
theorem root50Scan2 : scan edge (search edge 3) 4 [241] 241 root50Tail2 winner = winner :=
  (root50Step2).trans root50Scan3
theorem root50Scan1 : scan edge (search edge 3) 4 [241] 241 root50Tail1 winner = winner :=
  (root50Step1).trans root50Scan2
theorem root50Scan0 : scan edge (search edge 3) 4 [241] 241 root50Tail0 winner = winner :=
  (root50Step0).trans root50Scan1
set_option maxRecDepth 4096 in
theorem root50Branch : search edge 4 [241] root50Candidates 241 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [241] 241 root50Tail0 winner = winner
  exact root50Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
