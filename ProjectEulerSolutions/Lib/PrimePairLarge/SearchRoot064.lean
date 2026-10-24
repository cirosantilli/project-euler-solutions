import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root64Candidates : List Nat := Data.row331
set_option maxRecDepth 8192 in
theorem root64Candidates_checked : candidates edge 5 0 331 tail65 winner.bound = root64Candidates := by
  apply candidates_eq_row edge 5 0 331 tail65 root64Candidates winner.bound tail65_ordered
  · decide
  · decide
  · funext q
    rfl
def root64Tail81 : List Nat := []
def root64Tail80 : List Nat := 25357 :: root64Tail81
def root64Tail79 : List Nat := 24907 :: root64Tail80
def root64Tail78 : List Nat := 24799 :: root64Tail79
def root64Tail77 : List Nat := 24373 :: root64Tail78
def root64Tail76 : List Nat := 23911 :: root64Tail77
def root64Tail75 : List Nat := 23773 :: root64Tail76
def root64Tail74 : List Nat := 23689 :: root64Tail75
def root64Tail73 : List Nat := 23509 :: root64Tail74
def root64Tail72 : List Nat := 23227 :: root64Tail73
def root64Tail71 : List Nat := 22501 :: root64Tail72
def root64Tail70 : List Nat := 22189 :: root64Tail71
def root64Tail69 : List Nat := 22153 :: root64Tail70
def root64Tail68 : List Nat := 22039 :: root64Tail69
def root64Tail67 : List Nat := 21589 :: root64Tail68
def root64Tail66 : List Nat := 21493 :: root64Tail67
def root64Tail65 : List Nat := 21169 :: root64Tail66
def root64Tail64 : List Nat := 20749 :: root64Tail65
def root64Tail63 : List Nat := 19543 :: root64Tail64
def root64Tail62 : List Nat := 19087 :: root64Tail63
def root64Tail61 : List Nat := 18637 :: root64Tail62
def root64Tail60 : List Nat := 18313 :: root64Tail61
def root64Tail59 : List Nat := 17977 :: root64Tail60
def root64Tail58 : List Nat := 17911 :: root64Tail59
def root64Tail57 : List Nat := 17599 :: root64Tail58
def root64Tail56 : List Nat := 17443 :: root64Tail57
def root64Tail55 : List Nat := 17317 :: root64Tail56
def root64Tail54 : List Nat := 16651 :: root64Tail55
def root64Tail53 : List Nat := 16519 :: root64Tail54
def root64Tail52 : List Nat := 16189 :: root64Tail53
def root64Tail51 : List Nat := 15679 :: root64Tail52
def root64Tail50 : List Nat := 15373 :: root64Tail51
def root64Tail49 : List Nat := 13441 :: root64Tail50
def root64Tail48 : List Nat := 13009 :: root64Tail49
def root64Tail47 : List Nat := 12853 :: root64Tail48
def root64Tail46 : List Nat := 12043 :: root64Tail47
def root64Tail45 : List Nat := 11779 :: root64Tail46
def root64Tail44 : List Nat := 11527 :: root64Tail45
def root64Tail43 : List Nat := 11497 :: root64Tail44
def root64Tail42 : List Nat := 10993 :: root64Tail43
def root64Tail41 : List Nat := 10771 :: root64Tail42
def root64Tail40 : List Nat := 10687 :: root64Tail41
def root64Tail39 : List Nat := 10321 :: root64Tail40
def root64Tail38 : List Nat := 10303 :: root64Tail39
def root64Tail37 : List Nat := 10093 :: root64Tail38
def root64Tail36 : List Nat := 8929 :: root64Tail37
def root64Tail35 : List Nat := 8431 :: root64Tail36
def root64Tail34 : List Nat := 8377 :: root64Tail35
def root64Tail33 : List Nat := 8221 :: root64Tail34
def root64Tail32 : List Nat := 7537 :: root64Tail33
def root64Tail31 : List Nat := 7417 :: root64Tail32
def root64Tail30 : List Nat := 6949 :: root64Tail31
def root64Tail29 : List Nat := 6553 :: root64Tail30
def root64Tail28 : List Nat := 6451 :: root64Tail29
def root64Tail27 : List Nat := 6427 :: root64Tail28
def root64Tail26 : List Nat := 6361 :: root64Tail27
def root64Tail25 : List Nat := 5779 :: root64Tail26
def root64Tail24 : List Nat := 5407 :: root64Tail25
def root64Tail23 : List Nat := 5107 :: root64Tail24
def root64Tail22 : List Nat := 4651 :: root64Tail23
def root64Tail21 : List Nat := 4639 :: root64Tail22
def root64Tail20 : List Nat := 4513 :: root64Tail21
def root64Tail19 : List Nat := 4261 :: root64Tail20
def root64Tail18 : List Nat := 3769 :: root64Tail19
def root64Tail17 : List Nat := 3637 :: root64Tail18
def root64Tail16 : List Nat := 3361 :: root64Tail17
def root64Tail15 : List Nat := 3067 :: root64Tail16
def root64Tail14 : List Nat := 2833 :: root64Tail15
def root64Tail13 : List Nat := 2731 :: root64Tail14
def root64Tail12 : List Nat := 2719 :: root64Tail13
def root64Tail11 : List Nat := 1999 :: root64Tail12
def root64Tail10 : List Nat := 1933 :: root64Tail11
def root64Tail9 : List Nat := 1831 :: root64Tail10
def root64Tail8 : List Nat := 1669 :: root64Tail9
def root64Tail7 : List Nat := 1531 :: root64Tail8
def root64Tail6 : List Nat := 1303 :: root64Tail7
def root64Tail5 : List Nat := 937 :: root64Tail6
def root64Tail4 : List Nat := 907 :: root64Tail5
def root64Tail3 : List Nat := 883 :: root64Tail4
def root64Tail2 : List Nat := 739 :: root64Tail3
def root64Tail1 : List Nat := 577 :: root64Tail2
def root64Tail0 : List Nat := 349 :: root64Tail1
def root64Candidates0 : List Nat := [2719, 8377, 21589]
set_option maxRecDepth 8192 in
theorem root64Candidates0_checked : candidates edge 4 331 349 root64Tail1 winner.bound = root64Candidates0 := by
  unfold candidates
  have row : edge 349 = (SortedNatTree.contains FastRows.row349) := by
    change Data.row349.contains = (SortedNatTree.contains FastRows.row349)
    exact FastRows.row349_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child0 : search edge 3 [331, 349] root64Candidates0 680 winner = winner := by decide
theorem root64Step0 : scan edge (search edge 3) 4 [331] 331 root64Tail0 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail1 winner := by
  rw [root64Tail0, scan, if_neg (by decide), root64Candidates0_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail1
    (if 3 ≤ root64Candidates0.length then search edge 3 [331, 349] root64Candidates0 680 winner else winner) = _
  rw [if_pos (by decide), root64Child0]
def root64Candidates1 : List Nat := [937, 6949, 11779]
set_option maxRecDepth 8192 in
theorem root64Candidates1_checked : candidates edge 4 331 577 root64Tail2 winner.bound = root64Candidates1 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child1 : search edge 3 [331, 577] root64Candidates1 908 winner = winner := by decide
theorem root64Step1 : scan edge (search edge 3) 4 [331] 331 root64Tail1 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail2 winner := by
  rw [root64Tail1, scan, if_neg (by decide), root64Candidates1_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail2
    (if 3 ≤ root64Candidates1.length then search edge 3 [331, 577] root64Candidates1 908 winner else winner) = _
  rw [if_pos (by decide), root64Child1]
def root64Candidates2 : List Nat := [8431, 15679, 16189, 17911]
set_option maxRecDepth 8192 in
theorem root64Candidates2_checked : candidates edge 4 331 739 root64Tail3 winner.bound = root64Candidates2 := by
  unfold candidates
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child2 : search edge 3 [331, 739] root64Candidates2 1070 winner = winner := by decide
theorem root64Step2 : scan edge (search edge 3) 4 [331] 331 root64Tail2 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail3 winner := by
  rw [root64Tail2, scan, if_neg (by decide), root64Candidates2_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail3
    (if 3 ≤ root64Candidates2.length then search edge 3 [331, 739] root64Candidates2 1070 winner else winner) = _
  rw [if_pos (by decide), root64Child2]
def root64Candidates3 : List Nat := [11779]
set_option maxRecDepth 8192 in
theorem root64Candidates3_checked : candidates edge 4 331 883 root64Tail4 winner.bound = root64Candidates3 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
theorem root64Step3 : scan edge (search edge 3) 4 [331] 331 root64Tail3 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail4 winner := by
  rw [root64Tail3, scan, if_neg (by decide), root64Candidates3_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail4
    (if 3 ≤ root64Candidates3.length then search edge 3 [331, 883] root64Candidates3 1214 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates4 : List Nat := [2731, 11527, 13009, 16519]
set_option maxRecDepth 8192 in
theorem root64Candidates4_checked : candidates edge 4 331 907 root64Tail5 winner.bound = root64Candidates4 := by
  unfold candidates
  have row : edge 907 = (SortedNatTree.contains FastRows.row907) := by
    change Data.row907.contains = (SortedNatTree.contains FastRows.row907)
    exact FastRows.row907_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child4 : search edge 3 [331, 907] root64Candidates4 1238 winner = winner := by decide
theorem root64Step4 : scan edge (search edge 3) 4 [331] 331 root64Tail4 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail5 winner := by
  rw [root64Tail4, scan, if_neg (by decide), root64Candidates4_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail5
    (if 3 ≤ root64Candidates4.length then search edge 3 [331, 907] root64Candidates4 1238 winner else winner) = _
  rw [if_pos (by decide), root64Child4]
def root64Candidates5 : List Nat := [5407, 10321, 21169, 22039]
set_option maxRecDepth 8192 in
theorem root64Candidates5_checked : candidates edge 4 331 937 root64Tail6 winner.bound = root64Candidates5 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child5 : search edge 3 [331, 937] root64Candidates5 1268 winner = winner := by decide
theorem root64Step5 : scan edge (search edge 3) 4 [331] 331 root64Tail5 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail6 winner := by
  rw [root64Tail5, scan, if_neg (by decide), root64Candidates5_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail6
    (if 3 ≤ root64Candidates5.length then search edge 3 [331, 937] root64Candidates5 1268 winner else winner) = _
  rw [if_pos (by decide), root64Child5]
def root64Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root64Candidates6_checked : candidates edge 4 331 1303 root64Tail7 winner.bound = root64Candidates6 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
theorem root64Step6 : scan edge (search edge 3) 4 [331] 331 root64Tail6 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail7 winner := by
  rw [root64Tail6, scan, if_neg (by decide), root64Candidates6_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail7
    (if 3 ≤ root64Candidates6.length then search edge 3 [331, 1303] root64Candidates6 1634 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates7 : List Nat := [6361, 7537, 17443]
set_option maxRecDepth 8192 in
theorem root64Candidates7_checked : candidates edge 4 331 1531 root64Tail8 winner.bound = root64Candidates7 := by
  unfold candidates
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child7 : search edge 3 [331, 1531] root64Candidates7 1862 winner = winner := by decide
theorem root64Step7 : scan edge (search edge 3) 4 [331] 331 root64Tail7 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail8 winner := by
  rw [root64Tail7, scan, if_neg (by decide), root64Candidates7_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail8
    (if 3 ≤ root64Candidates7.length then search edge 3 [331, 1531] root64Candidates7 1862 winner else winner) = _
  rw [if_pos (by decide), root64Child7]
def root64Candidates8 : List Nat := [4651, 11497, 22153]
set_option maxRecDepth 8192 in
theorem root64Candidates8_checked : candidates edge 4 331 1669 root64Tail9 winner.bound = root64Candidates8 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root64Child8 : search edge 3 [331, 1669] root64Candidates8 2000 winner = winner := by decide
theorem root64Step8 : scan edge (search edge 3) 4 [331] 331 root64Tail8 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail9 winner := by
  rw [root64Tail8, scan, if_neg (by decide), root64Candidates8_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail9
    (if 3 ≤ root64Candidates8.length then search edge 3 [331, 1669] root64Candidates8 2000 winner else winner) = _
  rw [if_pos (by decide), root64Child8]
def root64Candidates9 : List Nat := [3769, 16651]
set_option maxRecDepth 8192 in
theorem root64Candidates9_checked : candidates edge 4 331 1831 root64Tail10 winner.bound = root64Candidates9 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
theorem root64Step9 : scan edge (search edge 3) 4 [331] 331 root64Tail9 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail10 winner := by
  rw [root64Tail9, scan, if_neg (by decide), root64Candidates9_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail10
    (if 3 ≤ root64Candidates9.length then search edge 3 [331, 1831] root64Candidates9 2162 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root64Candidates10_checked : candidates edge 4 331 1933 root64Tail11 winner.bound = root64Candidates10 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root64Step10 : scan edge (search edge 3) 4 [331] 331 root64Tail10 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail11 winner := by
  rw [root64Tail10, scan, if_neg (by decide), root64Candidates10_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail11
    (if 3 ≤ root64Candidates10.length then search edge 3 [331, 1933] root64Candidates10 2264 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates11 : List Nat := [3361, 22501]
set_option maxRecDepth 8192 in
theorem root64Candidates11_checked : candidates edge 4 331 1999 root64Tail12 winner.bound = root64Candidates11 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root64Step11 : scan edge (search edge 3) 4 [331] 331 root64Tail11 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail12 winner := by
  rw [root64Tail11, scan, if_neg (by decide), root64Candidates11_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail12
    (if 3 ≤ root64Candidates11.length then search edge 3 [331, 1999] root64Candidates11 2330 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates12 : List Nat := [10687]
set_option maxRecDepth 8192 in
theorem root64Candidates12_checked : candidates edge 4 331 2719 root64Tail13 winner.bound = root64Candidates12 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root64Step12 : scan edge (search edge 3) 4 [331] 331 root64Tail12 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail13 winner := by
  rw [root64Tail12, scan, if_neg (by decide), root64Candidates12_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail13
    (if 3 ≤ root64Candidates12.length then search edge 3 [331, 2719] root64Candidates12 3050 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates13 : List Nat := [6451]
set_option maxRecDepth 8192 in
theorem root64Candidates13_checked : candidates edge 4 331 2731 root64Tail14 winner.bound = root64Candidates13 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
theorem root64Step13 : scan edge (search edge 3) 4 [331] 331 root64Tail13 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail14 winner := by
  rw [root64Tail13, scan, if_neg (by decide), root64Candidates13_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail14
    (if 3 ≤ root64Candidates13.length then search edge 3 [331, 2731] root64Candidates13 3062 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates14 : List Nat := [6361]
set_option maxRecDepth 8192 in
theorem root64Candidates14_checked : candidates edge 4 331 2833 root64Tail15 winner.bound = root64Candidates14 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root64Step14 : scan edge (search edge 3) 4 [331] 331 root64Tail14 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail15 winner := by
  rw [root64Tail14, scan, if_neg (by decide), root64Candidates14_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail15
    (if 3 ≤ root64Candidates14.length then search edge 3 [331, 2833] root64Candidates14 3164 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates15 : List Nat := [8929]
set_option maxRecDepth 8192 in
theorem root64Candidates15_checked : candidates edge 4 331 3067 root64Tail16 winner.bound = root64Candidates15 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root64Step15 : scan edge (search edge 3) 4 [331] 331 root64Tail15 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail16 winner := by
  rw [root64Tail15, scan, if_neg (by decide), root64Candidates15_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail16
    (if 3 ≤ root64Candidates15.length then search edge 3 [331, 3067] root64Candidates15 3398 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root64Candidates16_checked : candidates edge 4 331 3361 root64Tail17 winner.bound = root64Candidates16 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root64Step16 : scan edge (search edge 3) 4 [331] 331 root64Tail16 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail17 winner := by
  rw [root64Tail16, scan, if_neg (by decide), root64Candidates16_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail17
    (if 3 ≤ root64Candidates16.length then search edge 3 [331, 3361] root64Candidates16 3692 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates17 : List Nat := [5407, 10771]
set_option maxRecDepth 8192 in
theorem root64Candidates17_checked : candidates edge 4 331 3637 root64Tail18 winner.bound = root64Candidates17 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root64Step17 : scan edge (search edge 3) 4 [331] 331 root64Tail17 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail18 winner := by
  rw [root64Tail17, scan, if_neg (by decide), root64Candidates17_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail18
    (if 3 ≤ root64Candidates17.length then search edge 3 [331, 3637] root64Candidates17 3968 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates18 : List Nat := [6949]
set_option maxRecDepth 8192 in
theorem root64Candidates18_checked : candidates edge 4 331 3769 root64Tail19 winner.bound = root64Candidates18 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root64Step18 : scan edge (search edge 3) 4 [331] 331 root64Tail18 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail19 winner := by
  rw [root64Tail18, scan, if_neg (by decide), root64Candidates18_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail19
    (if 3 ≤ root64Candidates18.length then search edge 3 [331, 3769] root64Candidates18 4100 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates19 : List Nat := [15373]
set_option maxRecDepth 8192 in
theorem root64Candidates19_checked : candidates edge 4 331 4261 root64Tail20 winner.bound = root64Candidates19 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root64Step19 : scan edge (search edge 3) 4 [331] 331 root64Tail19 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail20 winner := by
  rw [root64Tail19, scan, if_neg (by decide), root64Candidates19_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail20
    (if 3 ≤ root64Candidates19.length then search edge 3 [331, 4261] root64Candidates19 4592 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates20 : List Nat := [10687, 10771]
set_option maxRecDepth 8192 in
theorem root64Candidates20_checked : candidates edge 4 331 4513 root64Tail21 winner.bound = root64Candidates20 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root64Step20 : scan edge (search edge 3) 4 [331] 331 root64Tail20 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail21 winner := by
  rw [root64Tail20, scan, if_neg (by decide), root64Candidates20_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail21
    (if 3 ≤ root64Candidates20.length then search edge 3 [331, 4513] root64Candidates20 4844 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates21 : List Nat := [15679, 17599]
set_option maxRecDepth 8192 in
theorem root64Candidates21_checked : candidates edge 4 331 4639 root64Tail22 winner.bound = root64Candidates21 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root64Step21 : scan edge (search edge 3) 4 [331] 331 root64Tail21 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail22 winner := by
  rw [root64Tail21, scan, if_neg (by decide), root64Candidates21_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail22
    (if 3 ≤ root64Candidates21.length then search edge 3 [331, 4639] root64Candidates21 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates22 : List Nat := [11527]
set_option maxRecDepth 8192 in
theorem root64Candidates22_checked : candidates edge 4 331 4651 root64Tail23 winner.bound = root64Candidates22 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root64Step22 : scan edge (search edge 3) 4 [331] 331 root64Tail22 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail23 winner := by
  rw [root64Tail22, scan, if_neg (by decide), root64Candidates22_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail23
    (if 3 ≤ root64Candidates22.length then search edge 3 [331, 4651] root64Candidates22 4982 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates23 : List Nat := [5779]
set_option maxRecDepth 8192 in
theorem root64Candidates23_checked : candidates edge 4 331 5107 root64Tail24 winner.bound = root64Candidates23 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root64Step23 : scan edge (search edge 3) 4 [331] 331 root64Tail23 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail24 winner := by
  rw [root64Tail23, scan, if_neg (by decide), root64Candidates23_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail24
    (if 3 ≤ root64Candidates23.length then search edge 3 [331, 5107] root64Candidates23 5438 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates24 : List Nat := [5779, 18637]
set_option maxRecDepth 8192 in
theorem root64Candidates24_checked : candidates edge 4 331 5407 root64Tail25 winner.bound = root64Candidates24 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root64Step24 : scan edge (search edge 3) 4 [331] 331 root64Tail24 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail25 winner := by
  rw [root64Tail24, scan, if_neg (by decide), root64Candidates24_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail25
    (if 3 ≤ root64Candidates24.length then search edge 3 [331, 5407] root64Candidates24 5738 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root64Candidates25_checked : candidates edge 4 331 5779 root64Tail26 winner.bound = root64Candidates25 := by
  unfold candidates
  have row : edge 5779 = (SortedNatTree.contains FastRows.row5779) := by
    change Data.row5779.contains = (SortedNatTree.contains FastRows.row5779)
    exact FastRows.row5779_checked.symm
  rw [row]
  decide
theorem root64Step25 : scan edge (search edge 3) 4 [331] 331 root64Tail25 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail26 winner := by
  rw [root64Tail25, scan, if_neg (by decide), root64Candidates25_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail26
    (if 3 ≤ root64Candidates25.length then search edge 3 [331, 5779] root64Candidates25 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root64Candidates26 : List Nat := [15373]
set_option maxRecDepth 8192 in
theorem root64Candidates26_checked : candidates edge 4 331 6361 root64Tail27 winner.bound = root64Candidates26 := by
  unfold candidates
  have row : edge 6361 = (SortedNatTree.contains FastRows.row6361) := by
    change Data.row6361.contains = (SortedNatTree.contains FastRows.row6361)
    exact FastRows.row6361_checked.symm
  rw [row]
  decide
theorem root64Step26 : scan edge (search edge 3) 4 [331] 331 root64Tail26 winner =
    scan edge (search edge 3) 4 [331] 331 root64Tail27 winner := by
  rw [root64Tail26, scan, if_neg (by decide), root64Candidates26_checked]
  change scan edge (search edge 3) 4 [331] 331 root64Tail27
    (if 3 ≤ root64Candidates26.length then search edge 3 [331, 6361] root64Candidates26 6692 winner else winner) = _
  rw [if_neg (by decide)]
theorem root64Scan27 : scan edge (search edge 3) 4 [331] 331 root64Tail27 winner = winner := by rw [root64Tail27, scan, if_pos (by decide)]
theorem root64Scan26 : scan edge (search edge 3) 4 [331] 331 root64Tail26 winner = winner :=
  (root64Step26).trans root64Scan27
theorem root64Scan25 : scan edge (search edge 3) 4 [331] 331 root64Tail25 winner = winner :=
  (root64Step25).trans root64Scan26
theorem root64Scan24 : scan edge (search edge 3) 4 [331] 331 root64Tail24 winner = winner :=
  (root64Step24).trans root64Scan25
theorem root64Scan23 : scan edge (search edge 3) 4 [331] 331 root64Tail23 winner = winner :=
  (root64Step23).trans root64Scan24
theorem root64Scan22 : scan edge (search edge 3) 4 [331] 331 root64Tail22 winner = winner :=
  (root64Step22).trans root64Scan23
theorem root64Scan21 : scan edge (search edge 3) 4 [331] 331 root64Tail21 winner = winner :=
  (root64Step21).trans root64Scan22
theorem root64Scan20 : scan edge (search edge 3) 4 [331] 331 root64Tail20 winner = winner :=
  (root64Step20).trans root64Scan21
theorem root64Scan19 : scan edge (search edge 3) 4 [331] 331 root64Tail19 winner = winner :=
  (root64Step19).trans root64Scan20
theorem root64Scan18 : scan edge (search edge 3) 4 [331] 331 root64Tail18 winner = winner :=
  (root64Step18).trans root64Scan19
theorem root64Scan17 : scan edge (search edge 3) 4 [331] 331 root64Tail17 winner = winner :=
  (root64Step17).trans root64Scan18
theorem root64Scan16 : scan edge (search edge 3) 4 [331] 331 root64Tail16 winner = winner :=
  (root64Step16).trans root64Scan17
theorem root64Scan15 : scan edge (search edge 3) 4 [331] 331 root64Tail15 winner = winner :=
  (root64Step15).trans root64Scan16
theorem root64Scan14 : scan edge (search edge 3) 4 [331] 331 root64Tail14 winner = winner :=
  (root64Step14).trans root64Scan15
theorem root64Scan13 : scan edge (search edge 3) 4 [331] 331 root64Tail13 winner = winner :=
  (root64Step13).trans root64Scan14
theorem root64Scan12 : scan edge (search edge 3) 4 [331] 331 root64Tail12 winner = winner :=
  (root64Step12).trans root64Scan13
theorem root64Scan11 : scan edge (search edge 3) 4 [331] 331 root64Tail11 winner = winner :=
  (root64Step11).trans root64Scan12
theorem root64Scan10 : scan edge (search edge 3) 4 [331] 331 root64Tail10 winner = winner :=
  (root64Step10).trans root64Scan11
theorem root64Scan9 : scan edge (search edge 3) 4 [331] 331 root64Tail9 winner = winner :=
  (root64Step9).trans root64Scan10
theorem root64Scan8 : scan edge (search edge 3) 4 [331] 331 root64Tail8 winner = winner :=
  (root64Step8).trans root64Scan9
theorem root64Scan7 : scan edge (search edge 3) 4 [331] 331 root64Tail7 winner = winner :=
  (root64Step7).trans root64Scan8
theorem root64Scan6 : scan edge (search edge 3) 4 [331] 331 root64Tail6 winner = winner :=
  (root64Step6).trans root64Scan7
theorem root64Scan5 : scan edge (search edge 3) 4 [331] 331 root64Tail5 winner = winner :=
  (root64Step5).trans root64Scan6
theorem root64Scan4 : scan edge (search edge 3) 4 [331] 331 root64Tail4 winner = winner :=
  (root64Step4).trans root64Scan5
theorem root64Scan3 : scan edge (search edge 3) 4 [331] 331 root64Tail3 winner = winner :=
  (root64Step3).trans root64Scan4
theorem root64Scan2 : scan edge (search edge 3) 4 [331] 331 root64Tail2 winner = winner :=
  (root64Step2).trans root64Scan3
theorem root64Scan1 : scan edge (search edge 3) 4 [331] 331 root64Tail1 winner = winner :=
  (root64Step1).trans root64Scan2
theorem root64Scan0 : scan edge (search edge 3) 4 [331] 331 root64Tail0 winner = winner :=
  (root64Step0).trans root64Scan1
set_option maxRecDepth 4096 in
theorem root64Branch : search edge 4 [331] root64Candidates 331 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [331] 331 root64Tail0 winner = winner
  exact root64Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
