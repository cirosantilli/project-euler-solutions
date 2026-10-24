import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root46Candidates : List Nat := Data.row227
set_option maxRecDepth 8192 in
theorem root46Candidates_checked : candidates edge 5 0 227 tail47 winner.bound = root46Candidates := by
  apply candidates_eq_row edge 5 0 227 tail47 root46Candidates winner.bound tail47_ordered
  · decide
  · decide
  · funext q
    rfl
def root46Tail64 : List Nat := []
def root46Tail63 : List Nat := 25643 :: root46Tail64
def root46Tail62 : List Nat := 25097 :: root46Tail63
def root46Tail61 : List Nat := 24749 :: root46Tail62
def root46Tail60 : List Nat := 24071 :: root46Tail61
def root46Tail59 : List Nat := 23801 :: root46Tail60
def root46Tail58 : List Nat := 23417 :: root46Tail59
def root46Tail57 : List Nat := 23399 :: root46Tail58
def root46Tail56 : List Nat := 23087 :: root46Tail57
def root46Tail55 : List Nat := 23021 :: root46Tail56
def root46Tail54 : List Nat := 22961 :: root46Tail55
def root46Tail53 : List Nat := 22709 :: root46Tail54
def root46Tail52 : List Nat := 22079 :: root46Tail53
def root46Tail51 : List Nat := 21893 :: root46Tail52
def root46Tail50 : List Nat := 21851 :: root46Tail51
def root46Tail49 : List Nat := 21767 :: root46Tail50
def root46Tail48 : List Nat := 21467 :: root46Tail49
def root46Tail47 : List Nat := 20921 :: root46Tail48
def root46Tail46 : List Nat := 19997 :: root46Tail47
def root46Tail45 : List Nat := 19937 :: root46Tail46
def root46Tail44 : List Nat := 19163 :: root46Tail45
def root46Tail43 : List Nat := 18671 :: root46Tail44
def root46Tail42 : List Nat := 17789 :: root46Tail43
def root46Tail41 : List Nat := 17609 :: root46Tail42
def root46Tail40 : List Nat := 17417 :: root46Tail41
def root46Tail39 : List Nat := 17333 :: root46Tail40
def root46Tail38 : List Nat := 17099 :: root46Tail39
def root46Tail37 : List Nat := 15401 :: root46Tail38
def root46Tail36 : List Nat := 15233 :: root46Tail37
def root46Tail35 : List Nat := 14327 :: root46Tail36
def root46Tail34 : List Nat := 13883 :: root46Tail35
def root46Tail33 : List Nat := 13829 :: root46Tail34
def root46Tail32 : List Nat := 13781 :: root46Tail33
def root46Tail31 : List Nat := 12101 :: root46Tail32
def root46Tail30 : List Nat := 11867 :: root46Tail31
def root46Tail29 : List Nat := 11657 :: root46Tail30
def root46Tail28 : List Nat := 11411 :: root46Tail29
def root46Tail27 : List Nat := 11393 :: root46Tail28
def root46Tail26 : List Nat := 10847 :: root46Tail27
def root46Tail25 : List Nat := 10733 :: root46Tail26
def root46Tail24 : List Nat := 9839 :: root46Tail25
def root46Tail23 : List Nat := 9293 :: root46Tail24
def root46Tail22 : List Nat := 8693 :: root46Tail23
def root46Tail21 : List Nat := 8093 :: root46Tail22
def root46Tail20 : List Nat := 7823 :: root46Tail21
def root46Tail19 : List Nat := 5591 :: root46Tail20
def root46Tail18 : List Nat := 5333 :: root46Tail19
def root46Tail17 : List Nat := 4793 :: root46Tail18
def root46Tail16 : List Nat := 4691 :: root46Tail17
def root46Tail15 : List Nat := 4421 :: root46Tail16
def root46Tail14 : List Nat := 4271 :: root46Tail15
def root46Tail13 : List Nat := 3911 :: root46Tail14
def root46Tail12 : List Nat := 3701 :: root46Tail13
def root46Tail11 : List Nat := 2939 :: root46Tail12
def root46Tail10 : List Nat := 2903 :: root46Tail11
def root46Tail9 : List Nat := 2729 :: root46Tail10
def root46Tail8 : List Nat := 2129 :: root46Tail9
def root46Tail7 : List Nat := 1721 :: root46Tail8
def root46Tail6 : List Nat := 1499 :: root46Tail7
def root46Tail5 : List Nat := 1427 :: root46Tail6
def root46Tail4 : List Nat := 1229 :: root46Tail5
def root46Tail3 : List Nat := 827 :: root46Tail4
def root46Tail2 : List Nat := 719 :: root46Tail3
def root46Tail1 : List Nat := 593 :: root46Tail2
def root46Tail0 : List Nat := 281 :: root46Tail1
def root46Candidates0 : List Nat := [719, 1427]
set_option maxRecDepth 8192 in
theorem root46Candidates0_checked : candidates edge 4 227 281 root46Tail1 winner.bound = root46Candidates0 := by
  unfold candidates
  have row : edge 281 = (SortedNatTree.contains FastRows.row281) := by
    change Data.row281.contains = (SortedNatTree.contains FastRows.row281)
    exact FastRows.row281_checked.symm
  rw [row]
  decide
theorem root46Step0 : scan edge (search edge 3) 4 [227] 227 root46Tail0 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail1 winner := by
  rw [root46Tail0, scan, if_neg (by decide), root46Candidates0_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail1
    (if 3 ≤ root46Candidates0.length then search edge 3 [227, 281] root46Candidates0 508 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates1 : List Nat := [9839, 11393]
set_option maxRecDepth 8192 in
theorem root46Candidates1_checked : candidates edge 4 227 593 root46Tail2 winner.bound = root46Candidates1 := by
  unfold candidates
  have row : edge 593 = (SortedNatTree.contains FastRows.row593) := by
    change Data.row593.contains = (SortedNatTree.contains FastRows.row593)
    exact FastRows.row593_checked.symm
  rw [row]
  decide
theorem root46Step1 : scan edge (search edge 3) 4 [227] 227 root46Tail1 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail2 winner := by
  rw [root46Tail1, scan, if_neg (by decide), root46Candidates1_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail2
    (if 3 ≤ root46Candidates1.length then search edge 3 [227, 593] root46Candidates1 820 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates2 : List Nat := [17609]
set_option maxRecDepth 8192 in
theorem root46Candidates2_checked : candidates edge 4 227 719 root46Tail3 winner.bound = root46Candidates2 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
theorem root46Step2 : scan edge (search edge 3) 4 [227] 227 root46Tail2 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail3 winner := by
  rw [root46Tail2, scan, if_neg (by decide), root46Candidates2_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail3
    (if 3 ≤ root46Candidates2.length then search edge 3 [227, 719] root46Candidates2 946 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates3 : List Nat := [1721, 21767, 21893]
set_option maxRecDepth 8192 in
theorem root46Candidates3_checked : candidates edge 4 227 827 root46Tail4 winner.bound = root46Candidates3 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root46Child3 : search edge 3 [227, 827] root46Candidates3 1054 winner = winner := by decide
theorem root46Step3 : scan edge (search edge 3) 4 [227] 227 root46Tail3 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail4 winner := by
  rw [root46Tail3, scan, if_neg (by decide), root46Candidates3_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail4
    (if 3 ≤ root46Candidates3.length then search edge 3 [227, 827] root46Candidates3 1054 winner else winner) = _
  rw [if_pos (by decide), root46Child3]
def root46Candidates4 : List Nat := [21767]
set_option maxRecDepth 8192 in
theorem root46Candidates4_checked : candidates edge 4 227 1229 root46Tail5 winner.bound = root46Candidates4 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
theorem root46Step4 : scan edge (search edge 3) 4 [227] 227 root46Tail4 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail5 winner := by
  rw [root46Tail4, scan, if_neg (by decide), root46Candidates4_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail5
    (if 3 ≤ root46Candidates4.length then search edge 3 [227, 1229] root46Candidates4 1456 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates5 : List Nat := [13829]
set_option maxRecDepth 8192 in
theorem root46Candidates5_checked : candidates edge 4 227 1427 root46Tail6 winner.bound = root46Candidates5 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
theorem root46Step5 : scan edge (search edge 3) 4 [227] 227 root46Tail5 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail6 winner := by
  rw [root46Tail5, scan, if_neg (by decide), root46Candidates5_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail6
    (if 3 ≤ root46Candidates5.length then search edge 3 [227, 1427] root46Candidates5 1654 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates6 : List Nat := [2129, 8693, 22709]
set_option maxRecDepth 8192 in
theorem root46Candidates6_checked : candidates edge 4 227 1499 root46Tail7 winner.bound = root46Candidates6 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root46Child6 : search edge 3 [227, 1499] root46Candidates6 1726 winner = winner := by decide
theorem root46Step6 : scan edge (search edge 3) 4 [227] 227 root46Tail6 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail7 winner := by
  rw [root46Tail6, scan, if_neg (by decide), root46Candidates6_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail7
    (if 3 ≤ root46Candidates6.length then search edge 3 [227, 1499] root46Candidates6 1726 winner else winner) = _
  rw [if_pos (by decide), root46Child6]
def root46Candidates7 : List Nat := [21767]
set_option maxRecDepth 8192 in
theorem root46Candidates7_checked : candidates edge 4 227 1721 root46Tail8 winner.bound = root46Candidates7 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root46Step7 : scan edge (search edge 3) 4 [227] 227 root46Tail7 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail8 winner := by
  rw [root46Tail7, scan, if_neg (by decide), root46Candidates7_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail8
    (if 3 ≤ root46Candidates7.length then search edge 3 [227, 1721] root46Candidates7 1948 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates8 : List Nat := [2729, 4271]
set_option maxRecDepth 8192 in
theorem root46Candidates8_checked : candidates edge 4 227 2129 root46Tail9 winner.bound = root46Candidates8 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root46Step8 : scan edge (search edge 3) 4 [227] 227 root46Tail8 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail9 winner := by
  rw [root46Tail8, scan, if_neg (by decide), root46Candidates8_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail9
    (if 3 ≤ root46Candidates8.length then search edge 3 [227, 2129] root46Candidates8 2356 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates9 : List Nat := [13829, 22961]
set_option maxRecDepth 8192 in
theorem root46Candidates9_checked : candidates edge 4 227 2729 root46Tail10 winner.bound = root46Candidates9 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root46Step9 : scan edge (search edge 3) 4 [227] 227 root46Tail9 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail10 winner := by
  rw [root46Tail9, scan, if_neg (by decide), root46Candidates9_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail10
    (if 3 ≤ root46Candidates9.length then search edge 3 [227, 2729] root46Candidates9 2956 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates10 : List Nat := [19937]
set_option maxRecDepth 8192 in
theorem root46Candidates10_checked : candidates edge 4 227 2903 root46Tail11 winner.bound = root46Candidates10 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root46Step10 : scan edge (search edge 3) 4 [227] 227 root46Tail10 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail11 winner := by
  rw [root46Tail10, scan, if_neg (by decide), root46Candidates10_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail11
    (if 3 ≤ root46Candidates10.length then search edge 3 [227, 2903] root46Candidates10 3130 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates11 : List Nat := [10847]
set_option maxRecDepth 8192 in
theorem root46Candidates11_checked : candidates edge 4 227 2939 root46Tail12 winner.bound = root46Candidates11 := by
  unfold candidates
  have row : edge 2939 = (SortedNatTree.contains FastRows.row2939) := by
    change Data.row2939.contains = (SortedNatTree.contains FastRows.row2939)
    exact FastRows.row2939_checked.symm
  rw [row]
  decide
theorem root46Step11 : scan edge (search edge 3) 4 [227] 227 root46Tail11 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail12 winner := by
  rw [root46Tail11, scan, if_neg (by decide), root46Candidates11_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail12
    (if 3 ≤ root46Candidates11.length then search edge 3 [227, 2939] root46Candidates11 3166 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates12 : List Nat := [11867, 17333]
set_option maxRecDepth 8192 in
theorem root46Candidates12_checked : candidates edge 4 227 3701 root46Tail13 winner.bound = root46Candidates12 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
theorem root46Step12 : scan edge (search edge 3) 4 [227] 227 root46Tail12 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail13 winner := by
  rw [root46Tail12, scan, if_neg (by decide), root46Candidates12_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail13
    (if 3 ≤ root46Candidates12.length then search edge 3 [227, 3701] root46Candidates12 3928 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates13 : List Nat := [15233]
set_option maxRecDepth 8192 in
theorem root46Candidates13_checked : candidates edge 4 227 3911 root46Tail14 winner.bound = root46Candidates13 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root46Step13 : scan edge (search edge 3) 4 [227] 227 root46Tail13 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail14 winner := by
  rw [root46Tail13, scan, if_neg (by decide), root46Candidates13_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail14
    (if 3 ≤ root46Candidates13.length then search edge 3 [227, 3911] root46Candidates13 4138 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates14 : List Nat := [7823, 8693]
set_option maxRecDepth 8192 in
theorem root46Candidates14_checked : candidates edge 4 227 4271 root46Tail15 winner.bound = root46Candidates14 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root46Step14 : scan edge (search edge 3) 4 [227] 227 root46Tail14 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail15 winner := by
  rw [root46Tail14, scan, if_neg (by decide), root46Candidates14_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail15
    (if 3 ≤ root46Candidates14.length then search edge 3 [227, 4271] root46Candidates14 4498 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates15 : List Nat := [11393, 18671]
set_option maxRecDepth 8192 in
theorem root46Candidates15_checked : candidates edge 4 227 4421 root46Tail16 winner.bound = root46Candidates15 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root46Step15 : scan edge (search edge 3) 4 [227] 227 root46Tail15 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail16 winner := by
  rw [root46Tail15, scan, if_neg (by decide), root46Candidates15_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail16
    (if 3 ≤ root46Candidates15.length then search edge 3 [227, 4421] root46Candidates15 4648 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates16 : List Nat := [17609, 19163]
set_option maxRecDepth 8192 in
theorem root46Candidates16_checked : candidates edge 4 227 4691 root46Tail17 winner.bound = root46Candidates16 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root46Step16 : scan edge (search edge 3) 4 [227] 227 root46Tail16 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail17 winner := by
  rw [root46Tail16, scan, if_neg (by decide), root46Candidates16_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail17
    (if 3 ≤ root46Candidates16.length then search edge 3 [227, 4691] root46Candidates16 4918 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root46Candidates17_checked : candidates edge 4 227 4793 root46Tail18 winner.bound = root46Candidates17 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root46Step17 : scan edge (search edge 3) 4 [227] 227 root46Tail17 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail18 winner := by
  rw [root46Tail17, scan, if_neg (by decide), root46Candidates17_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail18
    (if 3 ≤ root46Candidates17.length then search edge 3 [227, 4793] root46Candidates17 5020 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates18 : List Nat := [8093]
set_option maxRecDepth 8192 in
theorem root46Candidates18_checked : candidates edge 4 227 5333 root46Tail19 winner.bound = root46Candidates18 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root46Step18 : scan edge (search edge 3) 4 [227] 227 root46Tail18 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail19 winner := by
  rw [root46Tail18, scan, if_neg (by decide), root46Candidates18_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail19
    (if 3 ≤ root46Candidates18.length then search edge 3 [227, 5333] root46Candidates18 5560 winner else winner) = _
  rw [if_neg (by decide)]
def root46Candidates19 : List Nat := [11867, 15233]
set_option maxRecDepth 8192 in
theorem root46Candidates19_checked : candidates edge 4 227 5591 root46Tail20 winner.bound = root46Candidates19 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root46Step19 : scan edge (search edge 3) 4 [227] 227 root46Tail19 winner =
    scan edge (search edge 3) 4 [227] 227 root46Tail20 winner := by
  rw [root46Tail19, scan, if_neg (by decide), root46Candidates19_checked]
  change scan edge (search edge 3) 4 [227] 227 root46Tail20
    (if 3 ≤ root46Candidates19.length then search edge 3 [227, 5591] root46Candidates19 5818 winner else winner) = _
  rw [if_neg (by decide)]
theorem root46Scan20 : scan edge (search edge 3) 4 [227] 227 root46Tail20 winner = winner := by rw [root46Tail20, scan, if_pos (by decide)]
theorem root46Scan19 : scan edge (search edge 3) 4 [227] 227 root46Tail19 winner = winner :=
  (root46Step19).trans root46Scan20
theorem root46Scan18 : scan edge (search edge 3) 4 [227] 227 root46Tail18 winner = winner :=
  (root46Step18).trans root46Scan19
theorem root46Scan17 : scan edge (search edge 3) 4 [227] 227 root46Tail17 winner = winner :=
  (root46Step17).trans root46Scan18
theorem root46Scan16 : scan edge (search edge 3) 4 [227] 227 root46Tail16 winner = winner :=
  (root46Step16).trans root46Scan17
theorem root46Scan15 : scan edge (search edge 3) 4 [227] 227 root46Tail15 winner = winner :=
  (root46Step15).trans root46Scan16
theorem root46Scan14 : scan edge (search edge 3) 4 [227] 227 root46Tail14 winner = winner :=
  (root46Step14).trans root46Scan15
theorem root46Scan13 : scan edge (search edge 3) 4 [227] 227 root46Tail13 winner = winner :=
  (root46Step13).trans root46Scan14
theorem root46Scan12 : scan edge (search edge 3) 4 [227] 227 root46Tail12 winner = winner :=
  (root46Step12).trans root46Scan13
theorem root46Scan11 : scan edge (search edge 3) 4 [227] 227 root46Tail11 winner = winner :=
  (root46Step11).trans root46Scan12
theorem root46Scan10 : scan edge (search edge 3) 4 [227] 227 root46Tail10 winner = winner :=
  (root46Step10).trans root46Scan11
theorem root46Scan9 : scan edge (search edge 3) 4 [227] 227 root46Tail9 winner = winner :=
  (root46Step9).trans root46Scan10
theorem root46Scan8 : scan edge (search edge 3) 4 [227] 227 root46Tail8 winner = winner :=
  (root46Step8).trans root46Scan9
theorem root46Scan7 : scan edge (search edge 3) 4 [227] 227 root46Tail7 winner = winner :=
  (root46Step7).trans root46Scan8
theorem root46Scan6 : scan edge (search edge 3) 4 [227] 227 root46Tail6 winner = winner :=
  (root46Step6).trans root46Scan7
theorem root46Scan5 : scan edge (search edge 3) 4 [227] 227 root46Tail5 winner = winner :=
  (root46Step5).trans root46Scan6
theorem root46Scan4 : scan edge (search edge 3) 4 [227] 227 root46Tail4 winner = winner :=
  (root46Step4).trans root46Scan5
theorem root46Scan3 : scan edge (search edge 3) 4 [227] 227 root46Tail3 winner = winner :=
  (root46Step3).trans root46Scan4
theorem root46Scan2 : scan edge (search edge 3) 4 [227] 227 root46Tail2 winner = winner :=
  (root46Step2).trans root46Scan3
theorem root46Scan1 : scan edge (search edge 3) 4 [227] 227 root46Tail1 winner = winner :=
  (root46Step1).trans root46Scan2
theorem root46Scan0 : scan edge (search edge 3) 4 [227] 227 root46Tail0 winner = winner :=
  (root46Step0).trans root46Scan1
set_option maxRecDepth 4096 in
theorem root46Branch : search edge 4 [227] root46Candidates 227 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [227] 227 root46Tail0 winner = winner
  exact root46Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
