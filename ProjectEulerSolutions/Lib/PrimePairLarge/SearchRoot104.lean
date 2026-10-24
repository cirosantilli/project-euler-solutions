import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root104Candidates : List Nat := Data.row587
set_option maxRecDepth 8192 in
theorem root104Candidates_checked : candidates edge 5 0 587 tail105 winner.bound = root104Candidates := by
  apply candidates_eq_row edge 5 0 587 tail105 root104Candidates winner.bound tail105_ordered
  · decide
  · decide
  · funext q
    rfl
def root104Tail73 : List Nat := []
def root104Tail72 : List Nat := 25097 :: root104Tail73
def root104Tail71 : List Nat := 25013 :: root104Tail72
def root104Tail70 : List Nat := 24593 :: root104Tail71
def root104Tail69 : List Nat := 23333 :: root104Tail70
def root104Tail68 : List Nat := 23201 :: root104Tail69
def root104Tail67 : List Nat := 23087 :: root104Tail68
def root104Tail66 : List Nat := 23003 :: root104Tail67
def root104Tail65 : List Nat := 22691 :: root104Tail66
def root104Tail64 : List Nat := 22397 :: root104Tail65
def root104Tail63 : List Nat := 22247 :: root104Tail64
def root104Tail62 : List Nat := 22073 :: root104Tail63
def root104Tail61 : List Nat := 21683 :: root104Tail62
def root104Tail60 : List Nat := 21521 :: root104Tail61
def root104Tail59 : List Nat := 20759 :: root104Tail60
def root104Tail58 : List Nat := 19889 :: root104Tail59
def root104Tail57 : List Nat := 19841 :: root104Tail58
def root104Tail56 : List Nat := 19457 :: root104Tail57
def root104Tail55 : List Nat := 17783 :: root104Tail56
def root104Tail54 : List Nat := 16937 :: root104Tail55
def root104Tail53 : List Nat := 16703 :: root104Tail54
def root104Tail52 : List Nat := 16649 :: root104Tail53
def root104Tail51 : List Nat := 16067 :: root104Tail52
def root104Tail50 : List Nat := 15809 :: root104Tail51
def root104Tail49 : List Nat := 15299 :: root104Tail50
def root104Tail48 : List Nat := 14639 :: root104Tail49
def root104Tail47 : List Nat := 13883 :: root104Tail48
def root104Tail46 : List Nat := 13799 :: root104Tail47
def root104Tail45 : List Nat := 13367 :: root104Tail46
def root104Tail44 : List Nat := 12671 :: root104Tail45
def root104Tail43 : List Nat := 12323 :: root104Tail44
def root104Tail42 : List Nat := 12149 :: root104Tail43
def root104Tail41 : List Nat := 11981 :: root104Tail42
def root104Tail40 : List Nat := 11867 :: root104Tail41
def root104Tail39 : List Nat := 11831 :: root104Tail40
def root104Tail38 : List Nat := 11657 :: root104Tail39
def root104Tail37 : List Nat := 11519 :: root104Tail38
def root104Tail36 : List Nat := 10937 :: root104Tail37
def root104Tail35 : List Nat := 10859 :: root104Tail36
def root104Tail34 : List Nat := 10739 :: root104Tail35
def root104Tail33 : List Nat := 9749 :: root104Tail34
def root104Tail32 : List Nat := 9623 :: root104Tail33
def root104Tail31 : List Nat := 9521 :: root104Tail32
def root104Tail30 : List Nat := 9473 :: root104Tail31
def root104Tail29 : List Nat := 8681 :: root104Tail30
def root104Tail28 : List Nat := 8123 :: root104Tail29
def root104Tail27 : List Nat := 8069 :: root104Tail28
def root104Tail26 : List Nat := 7829 :: root104Tail27
def root104Tail25 : List Nat := 7523 :: root104Tail26
def root104Tail24 : List Nat := 7481 :: root104Tail25
def root104Tail23 : List Nat := 7247 :: root104Tail24
def root104Tail22 : List Nat := 7127 :: root104Tail23
def root104Tail21 : List Nat := 6977 :: root104Tail22
def root104Tail20 : List Nat := 6911 :: root104Tail21
def root104Tail19 : List Nat := 6323 :: root104Tail20
def root104Tail18 : List Nat := 6011 :: root104Tail19
def root104Tail17 : List Nat := 5087 :: root104Tail18
def root104Tail16 : List Nat := 5021 :: root104Tail17
def root104Tail15 : List Nat := 4793 :: root104Tail16
def root104Tail14 : List Nat := 4397 :: root104Tail15
def root104Tail13 : List Nat := 4229 :: root104Tail14
def root104Tail12 : List Nat := 4133 :: root104Tail13
def root104Tail11 : List Nat := 3851 :: root104Tail12
def root104Tail10 : List Nat := 3767 :: root104Tail11
def root104Tail9 : List Nat := 3719 :: root104Tail10
def root104Tail8 : List Nat := 3011 :: root104Tail9
def root104Tail7 : List Nat := 2903 :: root104Tail8
def root104Tail6 : List Nat := 2549 :: root104Tail7
def root104Tail5 : List Nat := 2357 :: root104Tail6
def root104Tail4 : List Nat := 1571 :: root104Tail5
def root104Tail3 : List Nat := 1223 :: root104Tail4
def root104Tail2 : List Nat := 1163 :: root104Tail3
def root104Tail1 : List Nat := 677 :: root104Tail2
def root104Tail0 : List Nat := 617 :: root104Tail1
def root104Candidates0 : List Nat := [6011, 6323, 9473, 10937, 16649]
set_option maxRecDepth 8192 in
theorem root104Candidates0_checked : candidates edge 4 587 617 root104Tail1 winner.bound = root104Candidates0 := by
  unfold candidates
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root104Child0 : search edge 3 [587, 617] root104Candidates0 1204 winner = winner := by decide
theorem root104Step0 : scan edge (search edge 3) 4 [587] 587 root104Tail0 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail1 winner := by
  rw [root104Tail0, scan, if_neg (by decide), root104Candidates0_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail1
    (if 3 ≤ root104Candidates0.length then search edge 3 [587, 617] root104Candidates0 1204 winner else winner) = _
  rw [if_pos (by decide), root104Child0]
def root104Candidates1 : List Nat := [16937]
set_option maxRecDepth 8192 in
theorem root104Candidates1_checked : candidates edge 4 587 677 root104Tail2 winner.bound = root104Candidates1 := by
  unfold candidates
  have row : edge 677 = (SortedNatTree.contains FastRows.row677) := by
    change Data.row677.contains = (SortedNatTree.contains FastRows.row677)
    exact FastRows.row677_checked.symm
  rw [row]
  decide
theorem root104Step1 : scan edge (search edge 3) 4 [587] 587 root104Tail1 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail2 winner := by
  rw [root104Tail1, scan, if_neg (by decide), root104Candidates1_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail2
    (if 3 ≤ root104Candidates1.length then search edge 3 [587, 677] root104Candidates1 1264 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates2 : List Nat := [23087]
set_option maxRecDepth 8192 in
theorem root104Candidates2_checked : candidates edge 4 587 1163 root104Tail3 winner.bound = root104Candidates2 := by
  unfold candidates
  have row : edge 1163 = (SortedNatTree.contains FastRows.row1163) := by
    change Data.row1163.contains = (SortedNatTree.contains FastRows.row1163)
    exact FastRows.row1163_checked.symm
  rw [row]
  decide
theorem root104Step2 : scan edge (search edge 3) 4 [587] 587 root104Tail2 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail3 winner := by
  rw [root104Tail2, scan, if_neg (by decide), root104Candidates2_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail3
    (if 3 ≤ root104Candidates2.length then search edge 3 [587, 1163] root104Candidates2 1750 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates3 : List Nat := [4793, 7523]
set_option maxRecDepth 8192 in
theorem root104Candidates3_checked : candidates edge 4 587 1223 root104Tail4 winner.bound = root104Candidates3 := by
  unfold candidates
  have row : edge 1223 = (SortedNatTree.contains FastRows.row1223) := by
    change Data.row1223.contains = (SortedNatTree.contains FastRows.row1223)
    exact FastRows.row1223_checked.symm
  rw [row]
  decide
theorem root104Step3 : scan edge (search edge 3) 4 [587] 587 root104Tail3 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail4 winner := by
  rw [root104Tail3, scan, if_neg (by decide), root104Candidates3_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail4
    (if 3 ≤ root104Candidates3.length then search edge 3 [587, 1223] root104Candidates3 1810 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates4 : List Nat := [4133, 19889]
set_option maxRecDepth 8192 in
theorem root104Candidates4_checked : candidates edge 4 587 1571 root104Tail5 winner.bound = root104Candidates4 := by
  unfold candidates
  have row : edge 1571 = (SortedNatTree.contains FastRows.row1571) := by
    change Data.row1571.contains = (SortedNatTree.contains FastRows.row1571)
    exact FastRows.row1571_checked.symm
  rw [row]
  decide
theorem root104Step4 : scan edge (search edge 3) 4 [587] 587 root104Tail4 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail5 winner := by
  rw [root104Tail4, scan, if_neg (by decide), root104Candidates4_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail5
    (if 3 ≤ root104Candidates4.length then search edge 3 [587, 1571] root104Candidates4 2158 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates5 : List Nat := [2903, 5087, 6323]
set_option maxRecDepth 8192 in
theorem root104Candidates5_checked : candidates edge 4 587 2357 root104Tail6 winner.bound = root104Candidates5 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root104Child5 : search edge 3 [587, 2357] root104Candidates5 2944 winner = winner := by decide
theorem root104Step5 : scan edge (search edge 3) 4 [587] 587 root104Tail5 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail6 winner := by
  rw [root104Tail5, scan, if_neg (by decide), root104Candidates5_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail6
    (if 3 ≤ root104Candidates5.length then search edge 3 [587, 2357] root104Candidates5 2944 winner else winner) = _
  rw [if_pos (by decide), root104Child5]
def root104Candidates6 : List Nat := [6323, 21683]
set_option maxRecDepth 8192 in
theorem root104Candidates6_checked : candidates edge 4 587 2549 root104Tail7 winner.bound = root104Candidates6 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root104Step6 : scan edge (search edge 3) 4 [587] 587 root104Tail6 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail7 winner := by
  rw [root104Tail6, scan, if_neg (by decide), root104Candidates6_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail7
    (if 3 ≤ root104Candidates6.length then search edge 3 [587, 2549] root104Candidates6 3136 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates7 : List Nat := [8123, 13799]
set_option maxRecDepth 8192 in
theorem root104Candidates7_checked : candidates edge 4 587 2903 root104Tail8 winner.bound = root104Candidates7 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root104Step7 : scan edge (search edge 3) 4 [587] 587 root104Tail7 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail8 winner := by
  rw [root104Tail7, scan, if_neg (by decide), root104Candidates7_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail8
    (if 3 ≤ root104Candidates7.length then search edge 3 [587, 2903] root104Candidates7 3490 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates8 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root104Candidates8_checked : candidates edge 4 587 3011 root104Tail9 winner.bound = root104Candidates8 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root104Step8 : scan edge (search edge 3) 4 [587] 587 root104Tail8 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail9 winner := by
  rw [root104Tail8, scan, if_neg (by decide), root104Candidates8_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail9
    (if 3 ≤ root104Candidates8.length then search edge 3 [587, 3011] root104Candidates8 3598 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates9 : List Nat := [16067]
set_option maxRecDepth 8192 in
theorem root104Candidates9_checked : candidates edge 4 587 3719 root104Tail10 winner.bound = root104Candidates9 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root104Step9 : scan edge (search edge 3) 4 [587] 587 root104Tail9 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail10 winner := by
  rw [root104Tail9, scan, if_neg (by decide), root104Candidates9_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail10
    (if 3 ≤ root104Candidates9.length then search edge 3 [587, 3719] root104Candidates9 4306 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates10 : List Nat := [6911]
set_option maxRecDepth 8192 in
theorem root104Candidates10_checked : candidates edge 4 587 3767 root104Tail11 winner.bound = root104Candidates10 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root104Step10 : scan edge (search edge 3) 4 [587] 587 root104Tail10 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail11 winner := by
  rw [root104Tail10, scan, if_neg (by decide), root104Candidates10_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail11
    (if 3 ≤ root104Candidates10.length then search edge 3 [587, 3767] root104Candidates10 4354 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates11 : List Nat := [9521, 11981, 16703]
set_option maxRecDepth 8192 in
theorem root104Candidates11_checked : candidates edge 4 587 3851 root104Tail12 winner.bound = root104Candidates11 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root104Child11 : search edge 3 [587, 3851] root104Candidates11 4438 winner = winner := by decide
theorem root104Step11 : scan edge (search edge 3) 4 [587] 587 root104Tail11 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail12 winner := by
  rw [root104Tail11, scan, if_neg (by decide), root104Candidates11_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail12
    (if 3 ≤ root104Candidates11.length then search edge 3 [587, 3851] root104Candidates11 4438 winner else winner) = _
  rw [if_pos (by decide), root104Child11]
def root104Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root104Candidates12_checked : candidates edge 4 587 4133 root104Tail13 winner.bound = root104Candidates12 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root104Step12 : scan edge (search edge 3) 4 [587] 587 root104Tail12 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail13 winner := by
  rw [root104Tail12, scan, if_neg (by decide), root104Candidates12_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail13
    (if 3 ≤ root104Candidates12.length then search edge 3 [587, 4133] root104Candidates12 4720 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root104Candidates13_checked : candidates edge 4 587 4229 root104Tail14 winner.bound = root104Candidates13 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root104Step13 : scan edge (search edge 3) 4 [587] 587 root104Tail13 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail14 winner := by
  rw [root104Tail13, scan, if_neg (by decide), root104Candidates13_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail14
    (if 3 ≤ root104Candidates13.length then search edge 3 [587, 4229] root104Candidates13 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates14 : List Nat := [12149]
set_option maxRecDepth 8192 in
theorem root104Candidates14_checked : candidates edge 4 587 4397 root104Tail15 winner.bound = root104Candidates14 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root104Step14 : scan edge (search edge 3) 4 [587] 587 root104Tail14 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail15 winner := by
  rw [root104Tail14, scan, if_neg (by decide), root104Candidates14_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail15
    (if 3 ≤ root104Candidates14.length then search edge 3 [587, 4397] root104Candidates14 4984 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates15 : List Nat := [7481, 8123]
set_option maxRecDepth 8192 in
theorem root104Candidates15_checked : candidates edge 4 587 4793 root104Tail16 winner.bound = root104Candidates15 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root104Step15 : scan edge (search edge 3) 4 [587] 587 root104Tail15 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail16 winner := by
  rw [root104Tail15, scan, if_neg (by decide), root104Candidates15_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail16
    (if 3 ≤ root104Candidates15.length then search edge 3 [587, 4793] root104Candidates15 5380 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates16 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root104Candidates16_checked : candidates edge 4 587 5021 root104Tail17 winner.bound = root104Candidates16 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root104Step16 : scan edge (search edge 3) 4 [587] 587 root104Tail16 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail17 winner := by
  rw [root104Tail16, scan, if_neg (by decide), root104Candidates16_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail17
    (if 3 ≤ root104Candidates16.length then search edge 3 [587, 5021] root104Candidates16 5608 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates17 : List Nat := [7247]
set_option maxRecDepth 8192 in
theorem root104Candidates17_checked : candidates edge 4 587 5087 root104Tail18 winner.bound = root104Candidates17 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root104Step17 : scan edge (search edge 3) 4 [587] 587 root104Tail17 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail18 winner := by
  rw [root104Tail17, scan, if_neg (by decide), root104Candidates17_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail18
    (if 3 ≤ root104Candidates17.length then search edge 3 [587, 5087] root104Candidates17 5674 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root104Candidates18_checked : candidates edge 4 587 6011 root104Tail19 winner.bound = root104Candidates18 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root104Step18 : scan edge (search edge 3) 4 [587] 587 root104Tail18 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail19 winner := by
  rw [root104Tail18, scan, if_neg (by decide), root104Candidates18_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail19
    (if 3 ≤ root104Candidates18.length then search edge 3 [587, 6011] root104Candidates18 6598 winner else winner) = _
  rw [if_neg (by decide)]
def root104Candidates19 : List Nat := [7523]
set_option maxRecDepth 8192 in
theorem root104Candidates19_checked : candidates edge 4 587 6323 root104Tail20 winner.bound = root104Candidates19 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root104Step19 : scan edge (search edge 3) 4 [587] 587 root104Tail19 winner =
    scan edge (search edge 3) 4 [587] 587 root104Tail20 winner := by
  rw [root104Tail19, scan, if_neg (by decide), root104Candidates19_checked]
  change scan edge (search edge 3) 4 [587] 587 root104Tail20
    (if 3 ≤ root104Candidates19.length then search edge 3 [587, 6323] root104Candidates19 6910 winner else winner) = _
  rw [if_neg (by decide)]
theorem root104Scan20 : scan edge (search edge 3) 4 [587] 587 root104Tail20 winner = winner := by rw [root104Tail20, scan, if_pos (by decide)]
theorem root104Scan19 : scan edge (search edge 3) 4 [587] 587 root104Tail19 winner = winner :=
  (root104Step19).trans root104Scan20
theorem root104Scan18 : scan edge (search edge 3) 4 [587] 587 root104Tail18 winner = winner :=
  (root104Step18).trans root104Scan19
theorem root104Scan17 : scan edge (search edge 3) 4 [587] 587 root104Tail17 winner = winner :=
  (root104Step17).trans root104Scan18
theorem root104Scan16 : scan edge (search edge 3) 4 [587] 587 root104Tail16 winner = winner :=
  (root104Step16).trans root104Scan17
theorem root104Scan15 : scan edge (search edge 3) 4 [587] 587 root104Tail15 winner = winner :=
  (root104Step15).trans root104Scan16
theorem root104Scan14 : scan edge (search edge 3) 4 [587] 587 root104Tail14 winner = winner :=
  (root104Step14).trans root104Scan15
theorem root104Scan13 : scan edge (search edge 3) 4 [587] 587 root104Tail13 winner = winner :=
  (root104Step13).trans root104Scan14
theorem root104Scan12 : scan edge (search edge 3) 4 [587] 587 root104Tail12 winner = winner :=
  (root104Step12).trans root104Scan13
theorem root104Scan11 : scan edge (search edge 3) 4 [587] 587 root104Tail11 winner = winner :=
  (root104Step11).trans root104Scan12
theorem root104Scan10 : scan edge (search edge 3) 4 [587] 587 root104Tail10 winner = winner :=
  (root104Step10).trans root104Scan11
theorem root104Scan9 : scan edge (search edge 3) 4 [587] 587 root104Tail9 winner = winner :=
  (root104Step9).trans root104Scan10
theorem root104Scan8 : scan edge (search edge 3) 4 [587] 587 root104Tail8 winner = winner :=
  (root104Step8).trans root104Scan9
theorem root104Scan7 : scan edge (search edge 3) 4 [587] 587 root104Tail7 winner = winner :=
  (root104Step7).trans root104Scan8
theorem root104Scan6 : scan edge (search edge 3) 4 [587] 587 root104Tail6 winner = winner :=
  (root104Step6).trans root104Scan7
theorem root104Scan5 : scan edge (search edge 3) 4 [587] 587 root104Tail5 winner = winner :=
  (root104Step5).trans root104Scan6
theorem root104Scan4 : scan edge (search edge 3) 4 [587] 587 root104Tail4 winner = winner :=
  (root104Step4).trans root104Scan5
theorem root104Scan3 : scan edge (search edge 3) 4 [587] 587 root104Tail3 winner = winner :=
  (root104Step3).trans root104Scan4
theorem root104Scan2 : scan edge (search edge 3) 4 [587] 587 root104Tail2 winner = winner :=
  (root104Step2).trans root104Scan3
theorem root104Scan1 : scan edge (search edge 3) 4 [587] 587 root104Tail1 winner = winner :=
  (root104Step1).trans root104Scan2
theorem root104Scan0 : scan edge (search edge 3) 4 [587] 587 root104Tail0 winner = winner :=
  (root104Step0).trans root104Scan1
set_option maxRecDepth 4096 in
theorem root104Branch : search edge 4 [587] root104Candidates 587 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [587] 587 root104Tail0 winner = winner
  exact root104Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
