import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root147Candidates : List Nat := Data.row863
set_option maxRecDepth 8192 in
theorem root147Candidates_checked : candidates edge 5 0 863 tail148 winner.bound = root147Candidates := by
  apply candidates_eq_row edge 5 0 863 tail148 root147Candidates winner.bound tail148_ordered
  · decide
  · decide
  · funext q
    rfl
def root147Tail70 : List Nat := []
def root147Tail69 : List Nat := 24443 :: root147Tail70
def root147Tail68 : List Nat := 23801 :: root147Tail69
def root147Tail67 : List Nat := 22697 :: root147Tail68
def root147Tail66 : List Nat := 22307 :: root147Tail67
def root147Tail65 : List Nat := 22037 :: root147Tail66
def root147Tail64 : List Nat := 21839 :: root147Tail65
def root147Tail63 : List Nat := 21227 :: root147Tail64
def root147Tail62 : List Nat := 21089 :: root147Tail63
def root147Tail61 : List Nat := 20879 :: root147Tail62
def root147Tail60 : List Nat := 20639 :: root147Tail61
def root147Tail59 : List Nat := 20369 :: root147Tail60
def root147Tail58 : List Nat := 20231 :: root147Tail59
def root147Tail57 : List Nat := 19991 :: root147Tail58
def root147Tail56 : List Nat := 19913 :: root147Tail57
def root147Tail55 : List Nat := 19463 :: root147Tail56
def root147Tail54 : List Nat := 19433 :: root147Tail55
def root147Tail53 : List Nat := 19157 :: root147Tail54
def root147Tail52 : List Nat := 18731 :: root147Tail53
def root147Tail51 : List Nat := 18143 :: root147Tail52
def root147Tail50 : List Nat := 17939 :: root147Tail51
def root147Tail49 : List Nat := 17921 :: root147Tail50
def root147Tail48 : List Nat := 16421 :: root147Tail49
def root147Tail47 : List Nat := 16187 :: root147Tail48
def root147Tail46 : List Nat := 15809 :: root147Tail47
def root147Tail45 : List Nat := 15401 :: root147Tail46
def root147Tail44 : List Nat := 15263 :: root147Tail45
def root147Tail43 : List Nat := 15149 :: root147Tail44
def root147Tail42 : List Nat := 14957 :: root147Tail43
def root147Tail41 : List Nat := 13907 :: root147Tail42
def root147Tail40 : List Nat := 13649 :: root147Tail41
def root147Tail39 : List Nat := 13613 :: root147Tail40
def root147Tail38 : List Nat := 13397 :: root147Tail39
def root147Tail37 : List Nat := 13121 :: root147Tail38
def root147Tail36 : List Nat := 13037 :: root147Tail37
def root147Tail35 : List Nat := 12713 :: root147Tail36
def root147Tail34 : List Nat := 11987 :: root147Tail35
def root147Tail33 : List Nat := 11579 :: root147Tail34
def root147Tail32 : List Nat := 11369 :: root147Tail33
def root147Tail31 : List Nat := 10883 :: root147Tail32
def root147Tail30 : List Nat := 10487 :: root147Tail31
def root147Tail29 : List Nat := 10463 :: root147Tail30
def root147Tail28 : List Nat := 10247 :: root147Tail29
def root147Tail27 : List Nat := 9803 :: root147Tail28
def root147Tail26 : List Nat := 9539 :: root147Tail27
def root147Tail25 : List Nat := 8951 :: root147Tail26
def root147Tail24 : List Nat := 8807 :: root147Tail25
def root147Tail23 : List Nat := 8447 :: root147Tail24
def root147Tail22 : List Nat := 7523 :: root147Tail23
def root147Tail21 : List Nat := 6959 :: root147Tail22
def root147Tail20 : List Nat := 6899 :: root147Tail21
def root147Tail19 : List Nat := 6779 :: root147Tail20
def root147Tail18 : List Nat := 5849 :: root147Tail19
def root147Tail17 : List Nat := 5717 :: root147Tail18
def root147Tail16 : List Nat := 5381 :: root147Tail17
def root147Tail15 : List Nat := 4421 :: root147Tail16
def root147Tail14 : List Nat := 4397 :: root147Tail15
def root147Tail13 : List Nat := 4259 :: root147Tail14
def root147Tail12 : List Nat := 4253 :: root147Tail13
def root147Tail11 : List Nat := 3137 :: root147Tail12
def root147Tail10 : List Nat := 2927 :: root147Tail11
def root147Tail9 : List Nat := 2801 :: root147Tail10
def root147Tail8 : List Nat := 2621 :: root147Tail9
def root147Tail7 : List Nat := 2357 :: root147Tail8
def root147Tail6 : List Nat := 2297 :: root147Tail7
def root147Tail5 : List Nat := 2003 :: root147Tail6
def root147Tail4 : List Nat := 1907 :: root147Tail5
def root147Tail3 : List Nat := 1823 :: root147Tail4
def root147Tail2 : List Nat := 1307 :: root147Tail3
def root147Tail1 : List Nat := 1277 :: root147Tail2
def root147Tail0 : List Nat := 983 :: root147Tail1
def root147Candidates0 : List Nat := [21089]
set_option maxRecDepth 8192 in
theorem root147Candidates0_checked : candidates edge 4 863 983 root147Tail1 winner.bound = root147Candidates0 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root147Step0 : scan edge (search edge 3) 4 [863] 863 root147Tail0 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail1 winner := by
  rw [root147Tail0, scan, if_neg (by decide), root147Candidates0_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail1
    (if 3 ≤ root147Candidates0.length then search edge 3 [863, 983] root147Candidates0 1846 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates1 : List Nat := [9539, 20639]
set_option maxRecDepth 8192 in
theorem root147Candidates1_checked : candidates edge 4 863 1277 root147Tail2 winner.bound = root147Candidates1 := by
  unfold candidates
  have row : edge 1277 = (SortedNatTree.contains FastRows.row1277) := by
    change Data.row1277.contains = (SortedNatTree.contains FastRows.row1277)
    exact FastRows.row1277_checked.symm
  rw [row]
  decide
theorem root147Step1 : scan edge (search edge 3) 4 [863] 863 root147Tail1 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail2 winner := by
  rw [root147Tail1, scan, if_neg (by decide), root147Candidates1_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail2
    (if 3 ≤ root147Candidates1.length then search edge 3 [863, 1277] root147Candidates1 2140 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates2 : List Nat := [2357, 13397, 15401, 19991, 21839]
set_option maxRecDepth 8192 in
theorem root147Candidates2_checked : candidates edge 4 863 1307 root147Tail3 winner.bound = root147Candidates2 := by
  unfold candidates
  have row : edge 1307 = (SortedNatTree.contains FastRows.row1307) := by
    change Data.row1307.contains = (SortedNatTree.contains FastRows.row1307)
    exact FastRows.row1307_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root147Child2 : search edge 3 [863, 1307] root147Candidates2 2170 winner = winner := by decide
theorem root147Step2 : scan edge (search edge 3) 4 [863] 863 root147Tail2 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail3 winner := by
  rw [root147Tail2, scan, if_neg (by decide), root147Candidates2_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail3
    (if 3 ≤ root147Candidates2.length then search edge 3 [863, 1307] root147Candidates2 2170 winner else winner) = _
  rw [if_pos (by decide), root147Child2]
def root147Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root147Candidates3_checked : candidates edge 4 863 1823 root147Tail4 winner.bound = root147Candidates3 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
theorem root147Step3 : scan edge (search edge 3) 4 [863] 863 root147Tail3 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail4 winner := by
  rw [root147Tail3, scan, if_neg (by decide), root147Candidates3_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail4
    (if 3 ≤ root147Candidates3.length then search edge 3 [863, 1823] root147Candidates3 2686 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates4 : List Nat := [2621]
set_option maxRecDepth 8192 in
theorem root147Candidates4_checked : candidates edge 4 863 1907 root147Tail5 winner.bound = root147Candidates4 := by
  unfold candidates
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row]
  decide
theorem root147Step4 : scan edge (search edge 3) 4 [863] 863 root147Tail4 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail5 winner := by
  rw [root147Tail4, scan, if_neg (by decide), root147Candidates4_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail5
    (if 3 ≤ root147Candidates4.length then search edge 3 [863, 1907] root147Candidates4 2770 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates5 : List Nat := [12713]
set_option maxRecDepth 8192 in
theorem root147Candidates5_checked : candidates edge 4 863 2003 root147Tail6 winner.bound = root147Candidates5 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
theorem root147Step5 : scan edge (search edge 3) 4 [863] 863 root147Tail5 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail6 winner := by
  rw [root147Tail5, scan, if_neg (by decide), root147Candidates5_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail6
    (if 3 ≤ root147Candidates5.length then search edge 3 [863, 2003] root147Candidates5 2866 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates6 : List Nat := [8807, 20639]
set_option maxRecDepth 8192 in
theorem root147Candidates6_checked : candidates edge 4 863 2297 root147Tail7 winner.bound = root147Candidates6 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
theorem root147Step6 : scan edge (search edge 3) 4 [863] 863 root147Tail6 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail7 winner := by
  rw [root147Tail6, scan, if_neg (by decide), root147Candidates6_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail7
    (if 3 ≤ root147Candidates6.length then search edge 3 [863, 2297] root147Candidates6 3160 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates7 : List Nat := [8447, 11369]
set_option maxRecDepth 8192 in
theorem root147Candidates7_checked : candidates edge 4 863 2357 root147Tail8 winner.bound = root147Candidates7 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
theorem root147Step7 : scan edge (search edge 3) 4 [863] 863 root147Tail7 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail8 winner := by
  rw [root147Tail7, scan, if_neg (by decide), root147Candidates7_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail8
    (if 3 ≤ root147Candidates7.length then search edge 3 [863, 2357] root147Candidates7 3220 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates8 : List Nat := [13121, 13613, 19463]
set_option maxRecDepth 8192 in
theorem root147Candidates8_checked : candidates edge 4 863 2621 root147Tail9 winner.bound = root147Candidates8 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root147Child8 : search edge 3 [863, 2621] root147Candidates8 3484 winner = winner := by decide
theorem root147Step8 : scan edge (search edge 3) 4 [863] 863 root147Tail8 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail9 winner := by
  rw [root147Tail8, scan, if_neg (by decide), root147Candidates8_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail9
    (if 3 ≤ root147Candidates8.length then search edge 3 [863, 2621] root147Candidates8 3484 winner else winner) = _
  rw [if_pos (by decide), root147Child8]
def root147Candidates9 : List Nat := [11369, 11987]
set_option maxRecDepth 8192 in
theorem root147Candidates9_checked : candidates edge 4 863 2801 root147Tail10 winner.bound = root147Candidates9 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
theorem root147Step9 : scan edge (search edge 3) 4 [863] 863 root147Tail9 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail10 winner := by
  rw [root147Tail9, scan, if_neg (by decide), root147Candidates9_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail10
    (if 3 ≤ root147Candidates9.length then search edge 3 [863, 2801] root147Candidates9 3664 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates10 : List Nat := [3137, 11987, 15263]
set_option maxRecDepth 8192 in
theorem root147Candidates10_checked : candidates edge 4 863 2927 root147Tail11 winner.bound = root147Candidates10 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root147Child10 : search edge 3 [863, 2927] root147Candidates10 3790 winner = winner := by decide
theorem root147Step10 : scan edge (search edge 3) 4 [863] 863 root147Tail10 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail11 winner := by
  rw [root147Tail10, scan, if_neg (by decide), root147Candidates10_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail11
    (if 3 ≤ root147Candidates10.length then search edge 3 [863, 2927] root147Candidates10 3790 winner else winner) = _
  rw [if_pos (by decide), root147Child10]
def root147Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root147Candidates11_checked : candidates edge 4 863 3137 root147Tail12 winner.bound = root147Candidates11 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root147Step11 : scan edge (search edge 3) 4 [863] 863 root147Tail11 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail12 winner := by
  rw [root147Tail11, scan, if_neg (by decide), root147Candidates11_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail12
    (if 3 ≤ root147Candidates11.length then search edge 3 [863, 3137] root147Candidates11 4000 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates12 : List Nat := [4397]
set_option maxRecDepth 8192 in
theorem root147Candidates12_checked : candidates edge 4 863 4253 root147Tail13 winner.bound = root147Candidates12 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root147Step12 : scan edge (search edge 3) 4 [863] 863 root147Tail12 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail13 winner := by
  rw [root147Tail12, scan, if_neg (by decide), root147Candidates12_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail13
    (if 3 ≤ root147Candidates12.length then search edge 3 [863, 4253] root147Candidates12 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root147Candidates13_checked : candidates edge 4 863 4259 root147Tail14 winner.bound = root147Candidates13 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root147Step13 : scan edge (search edge 3) 4 [863] 863 root147Tail13 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail14 winner := by
  rw [root147Tail13, scan, if_neg (by decide), root147Candidates13_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail14
    (if 3 ≤ root147Candidates13.length then search edge 3 [863, 4259] root147Candidates13 5122 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates14 : List Nat := [4421, 10487, 13121]
set_option maxRecDepth 8192 in
theorem root147Candidates14_checked : candidates edge 4 863 4397 root147Tail15 winner.bound = root147Candidates14 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root147Child14 : search edge 3 [863, 4397] root147Candidates14 5260 winner = winner := by decide
theorem root147Step14 : scan edge (search edge 3) 4 [863] 863 root147Tail14 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail15 winner := by
  rw [root147Tail14, scan, if_neg (by decide), root147Candidates14_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail15
    (if 3 ≤ root147Candidates14.length then search edge 3 [863, 4397] root147Candidates14 5260 winner else winner) = _
  rw [if_pos (by decide), root147Child14]
def root147Candidates15 : List Nat := [6899, 9803, 10463, 10883]
set_option maxRecDepth 8192 in
theorem root147Candidates15_checked : candidates edge 4 863 4421 root147Tail16 winner.bound = root147Candidates15 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root147Child15 : search edge 3 [863, 4421] root147Candidates15 5284 winner = winner := by decide
theorem root147Step15 : scan edge (search edge 3) 4 [863] 863 root147Tail15 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail16 winner := by
  rw [root147Tail15, scan, if_neg (by decide), root147Candidates15_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail16
    (if 3 ≤ root147Candidates15.length then search edge 3 [863, 4421] root147Candidates15 5284 winner else winner) = _
  rw [if_pos (by decide), root147Child15]
def root147Candidates16 : List Nat := [10883, 13613]
set_option maxRecDepth 8192 in
theorem root147Candidates16_checked : candidates edge 4 863 5381 root147Tail17 winner.bound = root147Candidates16 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root147Step16 : scan edge (search edge 3) 4 [863] 863 root147Tail16 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail17 winner := by
  rw [root147Tail16, scan, if_neg (by decide), root147Candidates16_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail17
    (if 3 ≤ root147Candidates16.length then search edge 3 [863, 5381] root147Candidates16 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates17 : List Nat := [10487, 13397]
set_option maxRecDepth 8192 in
theorem root147Candidates17_checked : candidates edge 4 863 5717 root147Tail18 winner.bound = root147Candidates17 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root147Step17 : scan edge (search edge 3) 4 [863] 863 root147Tail17 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail18 winner := by
  rw [root147Tail17, scan, if_neg (by decide), root147Candidates17_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail18
    (if 3 ≤ root147Candidates17.length then search edge 3 [863, 5717] root147Candidates17 6580 winner else winner) = _
  rw [if_neg (by decide)]
def root147Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root147Candidates18_checked : candidates edge 4 863 5849 root147Tail19 winner.bound = root147Candidates18 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root147Step18 : scan edge (search edge 3) 4 [863] 863 root147Tail18 winner =
    scan edge (search edge 3) 4 [863] 863 root147Tail19 winner := by
  rw [root147Tail18, scan, if_neg (by decide), root147Candidates18_checked]
  change scan edge (search edge 3) 4 [863] 863 root147Tail19
    (if 3 ≤ root147Candidates18.length then search edge 3 [863, 5849] root147Candidates18 6712 winner else winner) = _
  rw [if_neg (by decide)]
theorem root147Scan19 : scan edge (search edge 3) 4 [863] 863 root147Tail19 winner = winner := by rw [root147Tail19, scan, if_pos (by decide)]
theorem root147Scan18 : scan edge (search edge 3) 4 [863] 863 root147Tail18 winner = winner :=
  (root147Step18).trans root147Scan19
theorem root147Scan17 : scan edge (search edge 3) 4 [863] 863 root147Tail17 winner = winner :=
  (root147Step17).trans root147Scan18
theorem root147Scan16 : scan edge (search edge 3) 4 [863] 863 root147Tail16 winner = winner :=
  (root147Step16).trans root147Scan17
theorem root147Scan15 : scan edge (search edge 3) 4 [863] 863 root147Tail15 winner = winner :=
  (root147Step15).trans root147Scan16
theorem root147Scan14 : scan edge (search edge 3) 4 [863] 863 root147Tail14 winner = winner :=
  (root147Step14).trans root147Scan15
theorem root147Scan13 : scan edge (search edge 3) 4 [863] 863 root147Tail13 winner = winner :=
  (root147Step13).trans root147Scan14
theorem root147Scan12 : scan edge (search edge 3) 4 [863] 863 root147Tail12 winner = winner :=
  (root147Step12).trans root147Scan13
theorem root147Scan11 : scan edge (search edge 3) 4 [863] 863 root147Tail11 winner = winner :=
  (root147Step11).trans root147Scan12
theorem root147Scan10 : scan edge (search edge 3) 4 [863] 863 root147Tail10 winner = winner :=
  (root147Step10).trans root147Scan11
theorem root147Scan9 : scan edge (search edge 3) 4 [863] 863 root147Tail9 winner = winner :=
  (root147Step9).trans root147Scan10
theorem root147Scan8 : scan edge (search edge 3) 4 [863] 863 root147Tail8 winner = winner :=
  (root147Step8).trans root147Scan9
theorem root147Scan7 : scan edge (search edge 3) 4 [863] 863 root147Tail7 winner = winner :=
  (root147Step7).trans root147Scan8
theorem root147Scan6 : scan edge (search edge 3) 4 [863] 863 root147Tail6 winner = winner :=
  (root147Step6).trans root147Scan7
theorem root147Scan5 : scan edge (search edge 3) 4 [863] 863 root147Tail5 winner = winner :=
  (root147Step5).trans root147Scan6
theorem root147Scan4 : scan edge (search edge 3) 4 [863] 863 root147Tail4 winner = winner :=
  (root147Step4).trans root147Scan5
theorem root147Scan3 : scan edge (search edge 3) 4 [863] 863 root147Tail3 winner = winner :=
  (root147Step3).trans root147Scan4
theorem root147Scan2 : scan edge (search edge 3) 4 [863] 863 root147Tail2 winner = winner :=
  (root147Step2).trans root147Scan3
theorem root147Scan1 : scan edge (search edge 3) 4 [863] 863 root147Tail1 winner = winner :=
  (root147Step1).trans root147Scan2
theorem root147Scan0 : scan edge (search edge 3) 4 [863] 863 root147Tail0 winner = winner :=
  (root147Step0).trans root147Scan1
set_option maxRecDepth 4096 in
theorem root147Branch : search edge 4 [863] root147Candidates 863 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [863] 863 root147Tail0 winner = winner
  exact root147Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
