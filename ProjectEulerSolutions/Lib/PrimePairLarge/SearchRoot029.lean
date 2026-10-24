import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root29Candidates : List Nat := Data.row131
set_option maxRecDepth 8192 in
theorem root29Candidates_checked : candidates edge 5 0 131 tail30 winner.bound = root29Candidates := by
  apply candidates_eq_row edge 5 0 131 tail30 root29Candidates winner.bound tail30_ordered
  · decide
  · decide
  · funext q
    rfl
def root29Tail74 : List Nat := []
def root29Tail73 : List Nat := 25349 :: root29Tail74
def root29Tail72 : List Nat := 24851 :: root29Tail73
def root29Tail71 : List Nat := 24179 :: root29Tail72
def root29Tail70 : List Nat := 23603 :: root29Tail71
def root29Tail69 : List Nat := 23339 :: root29Tail70
def root29Tail68 : List Nat := 23081 :: root29Tail69
def root29Tail67 : List Nat := 23003 :: root29Tail68
def root29Tail66 : List Nat := 22409 :: root29Tail67
def root29Tail65 : List Nat := 22229 :: root29Tail66
def root29Tail64 : List Nat := 21881 :: root29Tail65
def root29Tail63 : List Nat := 21683 :: root29Tail64
def root29Tail62 : List Nat := 20981 :: root29Tail63
def root29Tail61 : List Nat := 20759 :: root29Tail62
def root29Tail60 : List Nat := 20717 :: root29Tail61
def root29Tail59 : List Nat := 20297 :: root29Tail60
def root29Tail58 : List Nat := 19961 :: root29Tail59
def root29Tail57 : List Nat := 19853 :: root29Tail58
def root29Tail56 : List Nat := 19763 :: root29Tail57
def root29Tail55 : List Nat := 18269 :: root29Tail56
def root29Tail54 : List Nat := 17891 :: root29Tail55
def root29Tail53 : List Nat := 17231 :: root29Tail54
def root29Tail52 : List Nat := 15527 :: root29Tail53
def root29Tail51 : List Nat := 15227 :: root29Tail52
def root29Tail50 : List Nat := 15173 :: root29Tail51
def root29Tail49 : List Nat := 15161 :: root29Tail50
def root29Tail48 : List Nat := 15017 :: root29Tail49
def root29Tail47 : List Nat := 14897 :: root29Tail48
def root29Tail46 : List Nat := 13679 :: root29Tail47
def root29Tail45 : List Nat := 13619 :: root29Tail46
def root29Tail44 : List Nat := 13127 :: root29Tail45
def root29Tail43 : List Nat := 11789 :: root29Tail44
def root29Tail42 : List Nat := 11621 :: root29Tail43
def root29Tail41 : List Nat := 10949 :: root29Tail42
def root29Tail40 : List Nat := 10781 :: root29Tail41
def root29Tail39 : List Nat := 10607 :: root29Tail40
def root29Tail38 : List Nat := 10457 :: root29Tail39
def root29Tail37 : List Nat := 9743 :: root29Tail38
def root29Tail36 : List Nat := 9137 :: root29Tail37
def root29Tail35 : List Nat := 7727 :: root29Tail36
def root29Tail34 : List Nat := 7703 :: root29Tail35
def root29Tail33 : List Nat := 7487 :: root29Tail34
def root29Tail32 : List Nat := 7013 :: root29Tail33
def root29Tail31 : List Nat := 6971 :: root29Tail32
def root29Tail30 : List Nat := 6779 :: root29Tail31
def root29Tail29 : List Nat := 6299 :: root29Tail30
def root29Tail28 : List Nat := 5651 :: root29Tail29
def root29Tail27 : List Nat := 5519 :: root29Tail28
def root29Tail26 : List Nat := 5441 :: root29Tail27
def root29Tail25 : List Nat := 5399 :: root29Tail26
def root29Tail24 : List Nat := 5231 :: root29Tail25
def root29Tail23 : List Nat := 4673 :: root29Tail24
def root29Tail22 : List Nat := 4283 :: root29Tail23
def root29Tail21 : List Nat := 4259 :: root29Tail22
def root29Tail20 : List Nat := 4217 :: root29Tail21
def root29Tail19 : List Nat := 4133 :: root29Tail20
def root29Tail18 : List Nat := 4127 :: root29Tail19
def root29Tail17 : List Nat := 3929 :: root29Tail18
def root29Tail16 : List Nat := 3881 :: root29Tail17
def root29Tail15 : List Nat := 3467 :: root29Tail16
def root29Tail14 : List Nat := 3083 :: root29Tail15
def root29Tail13 : List Nat := 2963 :: root29Tail14
def root29Tail12 : List Nat := 2411 :: root29Tail13
def root29Tail11 : List Nat := 1901 :: root29Tail12
def root29Tail10 : List Nat := 1523 :: root29Tail11
def root29Tail9 : List Nat := 1229 :: root29Tail10
def root29Tail8 : List Nat := 1217 :: root29Tail9
def root29Tail7 : List Nat := 941 :: root29Tail8
def root29Tail6 : List Nat := 839 :: root29Tail7
def root29Tail5 : List Nat := 797 :: root29Tail6
def root29Tail4 : List Nat := 743 :: root29Tail5
def root29Tail3 : List Nat := 641 :: root29Tail4
def root29Tail2 : List Nat := 617 :: root29Tail3
def root29Tail1 : List Nat := 479 :: root29Tail2
def root29Tail0 : List Nat := 449 :: root29Tail1
def root29Candidates0 : List Nat := [941]
set_option maxRecDepth 8192 in
theorem root29Candidates0_checked : candidates edge 4 131 449 root29Tail1 winner.bound = root29Candidates0 := by
  unfold candidates
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row]
  decide
theorem root29Step0 : scan edge (search edge 3) 4 [131] 131 root29Tail0 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail1 winner := by
  rw [root29Tail0, scan, if_neg (by decide), root29Candidates0_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail1
    (if 3 ≤ root29Candidates0.length then search edge 3 [131, 449] root29Candidates0 580 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates1 : List Nat := [19763]
set_option maxRecDepth 8192 in
theorem root29Candidates1_checked : candidates edge 4 131 479 root29Tail2 winner.bound = root29Candidates1 := by
  unfold candidates
  have row : edge 479 = (SortedNatTree.contains FastRows.row479) := by
    change Data.row479.contains = (SortedNatTree.contains FastRows.row479)
    exact FastRows.row479_checked.symm
  rw [row]
  decide
theorem root29Step1 : scan edge (search edge 3) 4 [131] 131 root29Tail1 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail2 winner := by
  rw [root29Tail1, scan, if_neg (by decide), root29Candidates1_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail2
    (if 3 ≤ root29Candidates1.length then search edge 3 [131, 479] root29Candidates1 610 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates2_checked : candidates edge 4 131 617 root29Tail3 winner.bound = root29Candidates2 := by
  unfold candidates
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row]
  decide
theorem root29Step2 : scan edge (search edge 3) 4 [131] 131 root29Tail2 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail3 winner := by
  rw [root29Tail2, scan, if_neg (by decide), root29Candidates2_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail3
    (if 3 ≤ root29Candidates2.length then search edge 3 [131, 617] root29Candidates2 748 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates3 : List Nat := [5441]
set_option maxRecDepth 8192 in
theorem root29Candidates3_checked : candidates edge 4 131 641 root29Tail4 winner.bound = root29Candidates3 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
theorem root29Step3 : scan edge (search edge 3) 4 [131] 131 root29Tail3 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail4 winner := by
  rw [root29Tail3, scan, if_neg (by decide), root29Candidates3_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail4
    (if 3 ≤ root29Candidates3.length then search edge 3 [131, 641] root29Candidates3 772 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates4 : List Nat := [4283, 20759]
set_option maxRecDepth 8192 in
theorem root29Candidates4_checked : candidates edge 4 131 743 root29Tail5 winner.bound = root29Candidates4 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
theorem root29Step4 : scan edge (search edge 3) 4 [131] 131 root29Tail4 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail5 winner := by
  rw [root29Tail4, scan, if_neg (by decide), root29Candidates4_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail5
    (if 3 ≤ root29Candidates4.length then search edge 3 [131, 743] root29Candidates4 874 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates5 : List Nat := [2411, 3929, 4259, 5519, 20759, 21683, 24851]
set_option maxRecDepth 8192 in
theorem root29Candidates5_checked : candidates edge 4 131 797 root29Tail6 winner.bound = root29Candidates5 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root29Child5 : search edge 3 [131, 797] root29Candidates5 928 winner = winner := by decide
theorem root29Step5 : scan edge (search edge 3) 4 [131] 131 root29Tail5 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail6 winner := by
  rw [root29Tail5, scan, if_neg (by decide), root29Candidates5_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail6
    (if 3 ≤ root29Candidates5.length then search edge 3 [131, 797] root29Candidates5 928 winner else winner) = _
  rw [if_pos (by decide), root29Child5]
def root29Candidates6 : List Nat := [10781, 13127, 20717, 24179]
set_option maxRecDepth 8192 in
theorem root29Candidates6_checked : candidates edge 4 131 839 root29Tail7 winner.bound = root29Candidates6 := by
  unfold candidates
  have row : edge 839 = (SortedNatTree.contains FastRows.row839) := by
    change Data.row839.contains = (SortedNatTree.contains FastRows.row839)
    exact FastRows.row839_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root29Child6 : search edge 3 [131, 839] root29Candidates6 970 winner = winner := by decide
theorem root29Step6 : scan edge (search edge 3) 4 [131] 131 root29Tail6 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail7 winner := by
  rw [root29Tail6, scan, if_neg (by decide), root29Candidates6_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail7
    (if 3 ≤ root29Candidates6.length then search edge 3 [131, 839] root29Candidates6 970 winner else winner) = _
  rw [if_pos (by decide), root29Child6]
def root29Candidates7 : List Nat := [1229, 6299, 6971, 10607, 14897, 24179]
set_option maxRecDepth 8192 in
theorem root29Candidates7_checked : candidates edge 4 131 941 root29Tail8 winner.bound = root29Candidates7 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root29Child7 : search edge 3 [131, 941] root29Candidates7 1072 winner = winner := by decide
theorem root29Step7 : scan edge (search edge 3) 4 [131] 131 root29Tail7 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail8 winner := by
  rw [root29Tail7, scan, if_neg (by decide), root29Candidates7_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail8
    (if 3 ≤ root29Candidates7.length then search edge 3 [131, 941] root29Candidates7 1072 winner else winner) = _
  rw [if_pos (by decide), root29Child7]
def root29Candidates8 : List Nat := [17891, 23603]
set_option maxRecDepth 8192 in
theorem root29Candidates8_checked : candidates edge 4 131 1217 root29Tail9 winner.bound = root29Candidates8 := by
  unfold candidates
  have row : edge 1217 = (SortedNatTree.contains FastRows.row1217) := by
    change Data.row1217.contains = (SortedNatTree.contains FastRows.row1217)
    exact FastRows.row1217_checked.symm
  rw [row]
  decide
theorem root29Step8 : scan edge (search edge 3) 4 [131] 131 root29Tail8 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail9 winner := by
  rw [root29Tail8, scan, if_neg (by decide), root29Candidates8_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail9
    (if 3 ≤ root29Candidates8.length then search edge 3 [131, 1217] root29Candidates8 1348 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates9 : List Nat := [2963, 3881, 4127, 5399]
set_option maxRecDepth 8192 in
theorem root29Candidates9_checked : candidates edge 4 131 1229 root29Tail10 winner.bound = root29Candidates9 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root29Child9 : search edge 3 [131, 1229] root29Candidates9 1360 winner = winner := by decide
theorem root29Step9 : scan edge (search edge 3) 4 [131] 131 root29Tail9 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail10 winner := by
  rw [root29Tail9, scan, if_neg (by decide), root29Candidates9_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail10
    (if 3 ≤ root29Candidates9.length then search edge 3 [131, 1229] root29Candidates9 1360 winner else winner) = _
  rw [if_pos (by decide), root29Child9]
def root29Candidates10 : List Nat := [3929, 19763, 22409]
set_option maxRecDepth 8192 in
theorem root29Candidates10_checked : candidates edge 4 131 1523 root29Tail11 winner.bound = root29Candidates10 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root29Child10 : search edge 3 [131, 1523] root29Candidates10 1654 winner = winner := by decide
theorem root29Step10 : scan edge (search edge 3) 4 [131] 131 root29Tail10 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail11 winner := by
  rw [root29Tail10, scan, if_neg (by decide), root29Candidates10_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail11
    (if 3 ≤ root29Candidates10.length then search edge 3 [131, 1523] root29Candidates10 1654 winner else winner) = _
  rw [if_pos (by decide), root29Child10]
def root29Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates11_checked : candidates edge 4 131 1901 root29Tail12 winner.bound = root29Candidates11 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root29Step11 : scan edge (search edge 3) 4 [131] 131 root29Tail11 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail12 winner := by
  rw [root29Tail11, scan, if_neg (by decide), root29Candidates11_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail12
    (if 3 ≤ root29Candidates11.length then search edge 3 [131, 1901] root29Candidates11 2032 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates12 : List Nat := [5519]
set_option maxRecDepth 8192 in
theorem root29Candidates12_checked : candidates edge 4 131 2411 root29Tail13 winner.bound = root29Candidates12 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root29Step12 : scan edge (search edge 3) 4 [131] 131 root29Tail12 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail13 winner := by
  rw [root29Tail12, scan, if_neg (by decide), root29Candidates12_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail13
    (if 3 ≤ root29Candidates12.length then search edge 3 [131, 2411] root29Candidates12 2542 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates13 : List Nat := [13679, 15161]
set_option maxRecDepth 8192 in
theorem root29Candidates13_checked : candidates edge 4 131 2963 root29Tail14 winner.bound = root29Candidates13 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root29Step13 : scan edge (search edge 3) 4 [131] 131 root29Tail13 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail14 winner := by
  rw [root29Tail13, scan, if_neg (by decide), root29Candidates13_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail14
    (if 3 ≤ root29Candidates13.length then search edge 3 [131, 2963] root29Candidates13 3094 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates14 : List Nat := [20759]
set_option maxRecDepth 8192 in
theorem root29Candidates14_checked : candidates edge 4 131 3083 root29Tail15 winner.bound = root29Candidates14 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root29Step14 : scan edge (search edge 3) 4 [131] 131 root29Tail14 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail15 winner := by
  rw [root29Tail14, scan, if_neg (by decide), root29Candidates14_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail15
    (if 3 ≤ root29Candidates14.length then search edge 3 [131, 3083] root29Candidates14 3214 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates15 : List Nat := [5231]
set_option maxRecDepth 8192 in
theorem root29Candidates15_checked : candidates edge 4 131 3467 root29Tail16 winner.bound = root29Candidates15 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root29Step15 : scan edge (search edge 3) 4 [131] 131 root29Tail15 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail16 winner := by
  rw [root29Tail15, scan, if_neg (by decide), root29Candidates15_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail16
    (if 3 ≤ root29Candidates15.length then search edge 3 [131, 3467] root29Candidates15 3598 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates16 : List Nat := [5519]
set_option maxRecDepth 8192 in
theorem root29Candidates16_checked : candidates edge 4 131 3881 root29Tail17 winner.bound = root29Candidates16 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root29Step16 : scan edge (search edge 3) 4 [131] 131 root29Tail16 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail17 winner := by
  rw [root29Tail16, scan, if_neg (by decide), root29Candidates16_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail17
    (if 3 ≤ root29Candidates16.length then search edge 3 [131, 3881] root29Candidates16 4012 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates17_checked : candidates edge 4 131 3929 root29Tail18 winner.bound = root29Candidates17 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root29Step17 : scan edge (search edge 3) 4 [131] 131 root29Tail17 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail18 winner := by
  rw [root29Tail17, scan, if_neg (by decide), root29Candidates17_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail18
    (if 3 ≤ root29Candidates17.length then search edge 3 [131, 3929] root29Candidates17 4060 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates18 : List Nat := [5519, 20981]
set_option maxRecDepth 8192 in
theorem root29Candidates18_checked : candidates edge 4 131 4127 root29Tail19 winner.bound = root29Candidates18 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root29Step18 : scan edge (search edge 3) 4 [131] 131 root29Tail18 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail19 winner := by
  rw [root29Tail18, scan, if_neg (by decide), root29Candidates18_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail19
    (if 3 ≤ root29Candidates18.length then search edge 3 [131, 4127] root29Candidates18 4258 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates19 : List Nat := [17231, 19763]
set_option maxRecDepth 8192 in
theorem root29Candidates19_checked : candidates edge 4 131 4133 root29Tail20 winner.bound = root29Candidates19 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root29Step19 : scan edge (search edge 3) 4 [131] 131 root29Tail19 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail20 winner := by
  rw [root29Tail19, scan, if_neg (by decide), root29Candidates19_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail20
    (if 3 ≤ root29Candidates19.length then search edge 3 [131, 4133] root29Candidates19 4264 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates20 : List Nat := [4673, 9743]
set_option maxRecDepth 8192 in
theorem root29Candidates20_checked : candidates edge 4 131 4217 root29Tail21 winner.bound = root29Candidates20 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root29Step20 : scan edge (search edge 3) 4 [131] 131 root29Tail20 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail21 winner := by
  rw [root29Tail20, scan, if_neg (by decide), root29Candidates20_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail21
    (if 3 ≤ root29Candidates20.length then search edge 3 [131, 4217] root29Candidates20 4348 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates21_checked : candidates edge 4 131 4259 root29Tail22 winner.bound = root29Candidates21 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root29Step21 : scan edge (search edge 3) 4 [131] 131 root29Tail21 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail22 winner := by
  rw [root29Tail21, scan, if_neg (by decide), root29Candidates21_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail22
    (if 3 ≤ root29Candidates21.length then search edge 3 [131, 4259] root29Candidates21 4390 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates22_checked : candidates edge 4 131 4283 root29Tail23 winner.bound = root29Candidates22 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root29Step22 : scan edge (search edge 3) 4 [131] 131 root29Tail22 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail23 winner := by
  rw [root29Tail22, scan, if_neg (by decide), root29Candidates22_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail23
    (if 3 ≤ root29Candidates22.length then search edge 3 [131, 4283] root29Candidates22 4414 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates23 : List Nat := [5519, 6779]
set_option maxRecDepth 8192 in
theorem root29Candidates23_checked : candidates edge 4 131 4673 root29Tail24 winner.bound = root29Candidates23 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root29Step23 : scan edge (search edge 3) 4 [131] 131 root29Tail23 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail24 winner := by
  rw [root29Tail23, scan, if_neg (by decide), root29Candidates23_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail24
    (if 3 ≤ root29Candidates23.length then search edge 3 [131, 4673] root29Candidates23 4804 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates24 : List Nat := [10949, 15017]
set_option maxRecDepth 8192 in
theorem root29Candidates24_checked : candidates edge 4 131 5231 root29Tail25 winner.bound = root29Candidates24 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root29Step24 : scan edge (search edge 3) 4 [131] 131 root29Tail24 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail25 winner := by
  rw [root29Tail24, scan, if_neg (by decide), root29Candidates24_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail25
    (if 3 ≤ root29Candidates24.length then search edge 3 [131, 5231] root29Candidates24 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates25_checked : candidates edge 4 131 5399 root29Tail26 winner.bound = root29Candidates25 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root29Step25 : scan edge (search edge 3) 4 [131] 131 root29Tail25 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail26 winner := by
  rw [root29Tail25, scan, if_neg (by decide), root29Candidates25_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail26
    (if 3 ≤ root29Candidates25.length then search edge 3 [131, 5399] root29Candidates25 5530 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root29Candidates26_checked : candidates edge 4 131 5441 root29Tail27 winner.bound = root29Candidates26 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root29Step26 : scan edge (search edge 3) 4 [131] 131 root29Tail26 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail27 winner := by
  rw [root29Tail26, scan, if_neg (by decide), root29Candidates26_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail27
    (if 3 ≤ root29Candidates26.length then search edge 3 [131, 5441] root29Candidates26 5572 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates27 : List Nat := [6299, 19763, 20297]
set_option maxRecDepth 8192 in
theorem root29Candidates27_checked : candidates edge 4 131 5519 root29Tail28 winner.bound = root29Candidates27 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root29Child27 : search edge 3 [131, 5519] root29Candidates27 5650 winner = winner := by decide
theorem root29Step27 : scan edge (search edge 3) 4 [131] 131 root29Tail27 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail28 winner := by
  rw [root29Tail27, scan, if_neg (by decide), root29Candidates27_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail28
    (if 3 ≤ root29Candidates27.length then search edge 3 [131, 5519] root29Candidates27 5650 winner else winner) = _
  rw [if_pos (by decide), root29Child27]
def root29Candidates28 : List Nat := [19853]
set_option maxRecDepth 8192 in
theorem root29Candidates28_checked : candidates edge 4 131 5651 root29Tail29 winner.bound = root29Candidates28 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root29Step28 : scan edge (search edge 3) 4 [131] 131 root29Tail28 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail29 winner := by
  rw [root29Tail28, scan, if_neg (by decide), root29Candidates28_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail29
    (if 3 ≤ root29Candidates28.length then search edge 3 [131, 5651] root29Candidates28 5782 winner else winner) = _
  rw [if_neg (by decide)]
def root29Candidates29 : List Nat := [13679]
set_option maxRecDepth 8192 in
theorem root29Candidates29_checked : candidates edge 4 131 6299 root29Tail30 winner.bound = root29Candidates29 := by
  unfold candidates
  have row : edge 6299 = (SortedNatTree.contains FastRows.row6299) := by
    change Data.row6299.contains = (SortedNatTree.contains FastRows.row6299)
    exact FastRows.row6299_checked.symm
  rw [row]
  decide
theorem root29Step29 : scan edge (search edge 3) 4 [131] 131 root29Tail29 winner =
    scan edge (search edge 3) 4 [131] 131 root29Tail30 winner := by
  rw [root29Tail29, scan, if_neg (by decide), root29Candidates29_checked]
  change scan edge (search edge 3) 4 [131] 131 root29Tail30
    (if 3 ≤ root29Candidates29.length then search edge 3 [131, 6299] root29Candidates29 6430 winner else winner) = _
  rw [if_neg (by decide)]
theorem root29Scan30 : scan edge (search edge 3) 4 [131] 131 root29Tail30 winner = winner := by rw [root29Tail30, scan, if_pos (by decide)]
theorem root29Scan29 : scan edge (search edge 3) 4 [131] 131 root29Tail29 winner = winner :=
  (root29Step29).trans root29Scan30
theorem root29Scan28 : scan edge (search edge 3) 4 [131] 131 root29Tail28 winner = winner :=
  (root29Step28).trans root29Scan29
theorem root29Scan27 : scan edge (search edge 3) 4 [131] 131 root29Tail27 winner = winner :=
  (root29Step27).trans root29Scan28
theorem root29Scan26 : scan edge (search edge 3) 4 [131] 131 root29Tail26 winner = winner :=
  (root29Step26).trans root29Scan27
theorem root29Scan25 : scan edge (search edge 3) 4 [131] 131 root29Tail25 winner = winner :=
  (root29Step25).trans root29Scan26
theorem root29Scan24 : scan edge (search edge 3) 4 [131] 131 root29Tail24 winner = winner :=
  (root29Step24).trans root29Scan25
theorem root29Scan23 : scan edge (search edge 3) 4 [131] 131 root29Tail23 winner = winner :=
  (root29Step23).trans root29Scan24
theorem root29Scan22 : scan edge (search edge 3) 4 [131] 131 root29Tail22 winner = winner :=
  (root29Step22).trans root29Scan23
theorem root29Scan21 : scan edge (search edge 3) 4 [131] 131 root29Tail21 winner = winner :=
  (root29Step21).trans root29Scan22
theorem root29Scan20 : scan edge (search edge 3) 4 [131] 131 root29Tail20 winner = winner :=
  (root29Step20).trans root29Scan21
theorem root29Scan19 : scan edge (search edge 3) 4 [131] 131 root29Tail19 winner = winner :=
  (root29Step19).trans root29Scan20
theorem root29Scan18 : scan edge (search edge 3) 4 [131] 131 root29Tail18 winner = winner :=
  (root29Step18).trans root29Scan19
theorem root29Scan17 : scan edge (search edge 3) 4 [131] 131 root29Tail17 winner = winner :=
  (root29Step17).trans root29Scan18
theorem root29Scan16 : scan edge (search edge 3) 4 [131] 131 root29Tail16 winner = winner :=
  (root29Step16).trans root29Scan17
theorem root29Scan15 : scan edge (search edge 3) 4 [131] 131 root29Tail15 winner = winner :=
  (root29Step15).trans root29Scan16
theorem root29Scan14 : scan edge (search edge 3) 4 [131] 131 root29Tail14 winner = winner :=
  (root29Step14).trans root29Scan15
theorem root29Scan13 : scan edge (search edge 3) 4 [131] 131 root29Tail13 winner = winner :=
  (root29Step13).trans root29Scan14
theorem root29Scan12 : scan edge (search edge 3) 4 [131] 131 root29Tail12 winner = winner :=
  (root29Step12).trans root29Scan13
theorem root29Scan11 : scan edge (search edge 3) 4 [131] 131 root29Tail11 winner = winner :=
  (root29Step11).trans root29Scan12
theorem root29Scan10 : scan edge (search edge 3) 4 [131] 131 root29Tail10 winner = winner :=
  (root29Step10).trans root29Scan11
theorem root29Scan9 : scan edge (search edge 3) 4 [131] 131 root29Tail9 winner = winner :=
  (root29Step9).trans root29Scan10
theorem root29Scan8 : scan edge (search edge 3) 4 [131] 131 root29Tail8 winner = winner :=
  (root29Step8).trans root29Scan9
theorem root29Scan7 : scan edge (search edge 3) 4 [131] 131 root29Tail7 winner = winner :=
  (root29Step7).trans root29Scan8
theorem root29Scan6 : scan edge (search edge 3) 4 [131] 131 root29Tail6 winner = winner :=
  (root29Step6).trans root29Scan7
theorem root29Scan5 : scan edge (search edge 3) 4 [131] 131 root29Tail5 winner = winner :=
  (root29Step5).trans root29Scan6
theorem root29Scan4 : scan edge (search edge 3) 4 [131] 131 root29Tail4 winner = winner :=
  (root29Step4).trans root29Scan5
theorem root29Scan3 : scan edge (search edge 3) 4 [131] 131 root29Tail3 winner = winner :=
  (root29Step3).trans root29Scan4
theorem root29Scan2 : scan edge (search edge 3) 4 [131] 131 root29Tail2 winner = winner :=
  (root29Step2).trans root29Scan3
theorem root29Scan1 : scan edge (search edge 3) 4 [131] 131 root29Tail1 winner = winner :=
  (root29Step1).trans root29Scan2
theorem root29Scan0 : scan edge (search edge 3) 4 [131] 131 root29Tail0 winner = winner :=
  (root29Step0).trans root29Scan1
set_option maxRecDepth 4096 in
theorem root29Branch : search edge 4 [131] root29Candidates 131 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [131] 131 root29Tail0 winner = winner
  exact root29Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
