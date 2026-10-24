import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root72Candidates : List Nat := Data.row379
set_option maxRecDepth 8192 in
theorem root72Candidates_checked : candidates edge 5 0 379 tail73 winner.bound = root72Candidates := by
  apply candidates_eq_row edge 5 0 379 tail73 root72Candidates winner.bound tail73_ordered
  · decide
  · decide
  · funext q
    rfl
def root72Tail77 : List Nat := []
def root72Tail76 : List Nat := 25603 :: root72Tail77
def root72Tail75 : List Nat := 24847 :: root72Tail76
def root72Tail74 : List Nat := 24709 :: root72Tail75
def root72Tail73 : List Nat := 23833 :: root72Tail74
def root72Tail72 : List Nat := 23293 :: root72Tail73
def root72Tail71 : List Nat := 23203 :: root72Tail72
def root72Tail70 : List Nat := 23173 :: root72Tail71
def root72Tail69 : List Nat := 22483 :: root72Tail70
def root72Tail68 : List Nat := 21139 :: root72Tail69
def root72Tail67 : List Nat := 20899 :: root72Tail68
def root72Tail66 : List Nat := 20611 :: root72Tail67
def root72Tail65 : List Nat := 20521 :: root72Tail66
def root72Tail64 : List Nat := 19051 :: root72Tail65
def root72Tail63 : List Nat := 18427 :: root72Tail64
def root72Tail62 : List Nat := 17923 :: root72Tail63
def root72Tail61 : List Nat := 17881 :: root72Tail62
def root72Tail60 : List Nat := 17569 :: root72Tail61
def root72Tail59 : List Nat := 17401 :: root72Tail60
def root72Tail58 : List Nat := 16981 :: root72Tail59
def root72Tail57 : List Nat := 16831 :: root72Tail58
def root72Tail56 : List Nat := 16561 :: root72Tail57
def root72Tail55 : List Nat := 16189 :: root72Tail56
def root72Tail54 : List Nat := 16183 :: root72Tail55
def root72Tail53 : List Nat := 15817 :: root72Tail54
def root72Tail52 : List Nat := 15259 :: root72Tail53
def root72Tail51 : List Nat := 15187 :: root72Tail52
def root72Tail50 : List Nat := 14983 :: root72Tail51
def root72Tail49 : List Nat := 14713 :: root72Tail50
def root72Tail48 : List Nat := 14593 :: root72Tail49
def root72Tail47 : List Nat := 13159 :: root72Tail48
def root72Tail46 : List Nat := 12829 :: root72Tail47
def root72Tail45 : List Nat := 12037 :: root72Tail46
def root72Tail44 : List Nat := 11701 :: root72Tail45
def root72Tail43 : List Nat := 11353 :: root72Tail44
def root72Tail42 : List Nat := 10891 :: root72Tail43
def root72Tail41 : List Nat := 10657 :: root72Tail42
def root72Tail40 : List Nat := 10429 :: root72Tail41
def root72Tail39 : List Nat := 9931 :: root72Tail40
def root72Tail38 : List Nat := 9811 :: root72Tail39
def root72Tail37 : List Nat := 7933 :: root72Tail38
def root72Tail36 : List Nat := 7879 :: root72Tail37
def root72Tail35 : List Nat := 7873 :: root72Tail36
def root72Tail34 : List Nat := 7723 :: root72Tail35
def root72Tail33 : List Nat := 7699 :: root72Tail34
def root72Tail32 : List Nat := 6997 :: root72Tail33
def root72Tail31 : List Nat := 6733 :: root72Tail32
def root72Tail30 : List Nat := 6673 :: root72Tail31
def root72Tail29 : List Nat := 6379 :: root72Tail30
def root72Tail28 : List Nat := 6367 :: root72Tail29
def root72Tail27 : List Nat := 5581 :: root72Tail28
def root72Tail26 : List Nat := 5569 :: root72Tail27
def root72Tail25 : List Nat := 5563 :: root72Tail26
def root72Tail24 : List Nat := 4951 :: root72Tail25
def root72Tail23 : List Nat := 4423 :: root72Tail24
def root72Tail22 : List Nat := 4051 :: root72Tail23
def root72Tail21 : List Nat := 3931 :: root72Tail22
def root72Tail20 : List Nat := 3259 :: root72Tail21
def root72Tail19 : List Nat := 3121 :: root72Tail20
def root72Tail18 : List Nat := 3079 :: root72Tail19
def root72Tail17 : List Nat := 3061 :: root72Tail18
def root72Tail16 : List Nat := 2791 :: root72Tail17
def root72Tail15 : List Nat := 2281 :: root72Tail16
def root72Tail14 : List Nat := 2221 :: root72Tail15
def root72Tail13 : List Nat := 1873 :: root72Tail14
def root72Tail12 : List Nat := 1741 :: root72Tail13
def root72Tail11 : List Nat := 1423 :: root72Tail12
def root72Tail10 : List Nat := 1321 :: root72Tail11
def root72Tail9 : List Nat := 1231 :: root72Tail10
def root72Tail8 : List Nat := 1213 :: root72Tail9
def root72Tail7 : List Nat := 1087 :: root72Tail8
def root72Tail6 : List Nat := 1069 :: root72Tail7
def root72Tail5 : List Nat := 1063 :: root72Tail6
def root72Tail4 : List Nat := 997 :: root72Tail5
def root72Tail3 : List Nat := 877 :: root72Tail4
def root72Tail2 : List Nat := 811 :: root72Tail3
def root72Tail1 : List Nat := 751 :: root72Tail2
def root72Tail0 : List Nat := 397 :: root72Tail1
def root72Candidates0 : List Nat := [1063, 7933, 15817, 17569, 21139]
set_option maxRecDepth 8192 in
theorem root72Candidates0_checked : candidates edge 4 379 397 root72Tail1 winner.bound = root72Candidates0 := by
  unfold candidates
  have row : edge 397 = (SortedNatTree.contains FastRows.row397) := by
    change Data.row397.contains = (SortedNatTree.contains FastRows.row397)
    exact FastRows.row397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child0 : search edge 3 [379, 397] root72Candidates0 776 winner = winner := by decide
theorem root72Step0 : scan edge (search edge 3) 4 [379] 379 root72Tail0 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail1 winner := by
  rw [root72Tail0, scan, if_neg (by decide), root72Candidates0_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail1
    (if 3 ≤ root72Candidates0.length then search edge 3 [379, 397] root72Candidates0 776 winner else winner) = _
  rw [if_pos (by decide), root72Child0]
def root72Candidates1 : List Nat := [997, 17401, 23173]
set_option maxRecDepth 8192 in
theorem root72Candidates1_checked : candidates edge 4 379 751 root72Tail2 winner.bound = root72Candidates1 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child1 : search edge 3 [379, 751] root72Candidates1 1130 winner = winner := by decide
theorem root72Step1 : scan edge (search edge 3) 4 [379] 379 root72Tail1 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail2 winner := by
  rw [root72Tail1, scan, if_neg (by decide), root72Candidates1_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail2
    (if 3 ≤ root72Candidates1.length then search edge 3 [379, 751] root72Candidates1 1130 winner else winner) = _
  rw [if_pos (by decide), root72Child1]
def root72Candidates2 : List Nat := [997, 17923]
set_option maxRecDepth 8192 in
theorem root72Candidates2_checked : candidates edge 4 379 811 root72Tail3 winner.bound = root72Candidates2 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
theorem root72Step2 : scan edge (search edge 3) 4 [379] 379 root72Tail2 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail3 winner := by
  rw [root72Tail2, scan, if_neg (by decide), root72Candidates2_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail3
    (if 3 ≤ root72Candidates2.length then search edge 3 [379, 811] root72Candidates2 1190 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates3 : List Nat := [997, 14983, 23833]
set_option maxRecDepth 8192 in
theorem root72Candidates3_checked : candidates edge 4 379 877 root72Tail4 winner.bound = root72Candidates3 := by
  unfold candidates
  have row : edge 877 = (SortedNatTree.contains FastRows.row877) := by
    change Data.row877.contains = (SortedNatTree.contains FastRows.row877)
    exact FastRows.row877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child3 : search edge 3 [379, 877] root72Candidates3 1256 winner = winner := by decide
theorem root72Step3 : scan edge (search edge 3) 4 [379] 379 root72Tail3 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail4 winner := by
  rw [root72Tail3, scan, if_neg (by decide), root72Candidates3_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail4
    (if 3 ≤ root72Candidates3.length then search edge 3 [379, 877] root72Candidates3 1256 winner else winner) = _
  rw [if_pos (by decide), root72Child3]
def root72Candidates4 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root72Candidates4_checked : candidates edge 4 379 997 root72Tail5 winner.bound = root72Candidates4 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
theorem root72Step4 : scan edge (search edge 3) 4 [379] 379 root72Tail4 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail5 winner := by
  rw [root72Tail4, scan, if_neg (by decide), root72Candidates4_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail5
    (if 3 ≤ root72Candidates4.length then search edge 3 [379, 997] root72Candidates4 1376 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates5 : List Nat := [9931]
set_option maxRecDepth 8192 in
theorem root72Candidates5_checked : candidates edge 4 379 1063 root72Tail6 winner.bound = root72Candidates5 := by
  unfold candidates
  have row : edge 1063 = (SortedNatTree.contains FastRows.row1063) := by
    change Data.row1063.contains = (SortedNatTree.contains FastRows.row1063)
    exact FastRows.row1063_checked.symm
  rw [row]
  decide
theorem root72Step5 : scan edge (search edge 3) 4 [379] 379 root72Tail5 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail6 winner := by
  rw [root72Tail5, scan, if_neg (by decide), root72Candidates5_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail6
    (if 3 ≤ root72Candidates5.length then search edge 3 [379, 1063] root72Candidates5 1442 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates6 : List Nat := [1321, 2281, 3061, 11353, 17401]
set_option maxRecDepth 8192 in
theorem root72Candidates6_checked : candidates edge 4 379 1069 root72Tail7 winner.bound = root72Candidates6 := by
  unfold candidates
  have row : edge 1069 = (SortedNatTree.contains FastRows.row1069) := by
    change Data.row1069.contains = (SortedNatTree.contains FastRows.row1069)
    exact FastRows.row1069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child6 : search edge 3 [379, 1069] root72Candidates6 1448 winner = winner := by decide
theorem root72Step6 : scan edge (search edge 3) 4 [379] 379 root72Tail6 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail7 winner := by
  rw [root72Tail6, scan, if_neg (by decide), root72Candidates6_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail7
    (if 3 ≤ root72Candidates6.length then search edge 3 [379, 1069] root72Candidates6 1448 winner else winner) = _
  rw [if_pos (by decide), root72Child6]
def root72Candidates7 : List Nat := [5569, 10429, 12829, 21139]
set_option maxRecDepth 8192 in
theorem root72Candidates7_checked : candidates edge 4 379 1087 root72Tail8 winner.bound = root72Candidates7 := by
  unfold candidates
  have row : edge 1087 = (SortedNatTree.contains FastRows.row1087) := by
    change Data.row1087.contains = (SortedNatTree.contains FastRows.row1087)
    exact FastRows.row1087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child7 : search edge 3 [379, 1087] root72Candidates7 1466 winner = winner := by decide
theorem root72Step7 : scan edge (search edge 3) 4 [379] 379 root72Tail7 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail8 winner := by
  rw [root72Tail7, scan, if_neg (by decide), root72Candidates7_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail8
    (if 3 ≤ root72Candidates7.length then search edge 3 [379, 1087] root72Candidates7 1466 winner else winner) = _
  rw [if_pos (by decide), root72Child7]
def root72Candidates8 : List Nat := [3061, 16183, 17569, 22483, 23833]
set_option maxRecDepth 8192 in
theorem root72Candidates8_checked : candidates edge 4 379 1213 root72Tail9 winner.bound = root72Candidates8 := by
  unfold candidates
  have row : edge 1213 = (SortedNatTree.contains FastRows.row1213) := by
    change Data.row1213.contains = (SortedNatTree.contains FastRows.row1213)
    exact FastRows.row1213_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child8 : search edge 3 [379, 1213] root72Candidates8 1592 winner = winner := by decide
theorem root72Step8 : scan edge (search edge 3) 4 [379] 379 root72Tail8 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail9 winner := by
  rw [root72Tail8, scan, if_neg (by decide), root72Candidates8_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail9
    (if 3 ≤ root72Candidates8.length then search edge 3 [379, 1213] root72Candidates8 1592 winner else winner) = _
  rw [if_pos (by decide), root72Child8]
def root72Candidates9 : List Nat := [20521]
set_option maxRecDepth 8192 in
theorem root72Candidates9_checked : candidates edge 4 379 1231 root72Tail10 winner.bound = root72Candidates9 := by
  unfold candidates
  have row : edge 1231 = (SortedNatTree.contains FastRows.row1231) := by
    change Data.row1231.contains = (SortedNatTree.contains FastRows.row1231)
    exact FastRows.row1231_checked.symm
  rw [row]
  decide
theorem root72Step9 : scan edge (search edge 3) 4 [379] 379 root72Tail9 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail10 winner := by
  rw [root72Tail9, scan, if_neg (by decide), root72Candidates9_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail10
    (if 3 ≤ root72Candidates9.length then search edge 3 [379, 1231] root72Candidates9 1610 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates10 : List Nat := [5581, 15259]
set_option maxRecDepth 8192 in
theorem root72Candidates10_checked : candidates edge 4 379 1321 root72Tail11 winner.bound = root72Candidates10 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
theorem root72Step10 : scan edge (search edge 3) 4 [379] 379 root72Tail10 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail11 winner := by
  rw [root72Tail10, scan, if_neg (by decide), root72Candidates10_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail11
    (if 3 ≤ root72Candidates10.length then search edge 3 [379, 1321] root72Candidates10 1700 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates11 : List Nat := [6367, 16189, 21139]
set_option maxRecDepth 8192 in
theorem root72Candidates11_checked : candidates edge 4 379 1423 root72Tail12 winner.bound = root72Candidates11 := by
  unfold candidates
  have row : edge 1423 = (SortedNatTree.contains FastRows.row1423) := by
    change Data.row1423.contains = (SortedNatTree.contains FastRows.row1423)
    exact FastRows.row1423_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child11 : search edge 3 [379, 1423] root72Candidates11 1802 winner = winner := by decide
theorem root72Step11 : scan edge (search edge 3) 4 [379] 379 root72Tail11 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail12 winner := by
  rw [root72Tail11, scan, if_neg (by decide), root72Candidates11_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail12
    (if 3 ≤ root72Candidates11.length then search edge 3 [379, 1423] root72Candidates11 1802 winner else winner) = _
  rw [if_pos (by decide), root72Child11]
def root72Candidates12 : List Nat := [2281, 5581]
set_option maxRecDepth 8192 in
theorem root72Candidates12_checked : candidates edge 4 379 1741 root72Tail13 winner.bound = root72Candidates12 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
theorem root72Step12 : scan edge (search edge 3) 4 [379] 379 root72Tail12 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail13 winner := by
  rw [root72Tail12, scan, if_neg (by decide), root72Candidates12_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail13
    (if 3 ≤ root72Candidates12.length then search edge 3 [379, 1741] root72Candidates12 2120 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates13 : List Nat := [5569, 6997, 10657]
set_option maxRecDepth 8192 in
theorem root72Candidates13_checked : candidates edge 4 379 1873 root72Tail14 winner.bound = root72Candidates13 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child13 : search edge 3 [379, 1873] root72Candidates13 2252 winner = winner := by decide
theorem root72Step13 : scan edge (search edge 3) 4 [379] 379 root72Tail13 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail14 winner := by
  rw [root72Tail13, scan, if_neg (by decide), root72Candidates13_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail14
    (if 3 ≤ root72Candidates13.length then search edge 3 [379, 1873] root72Candidates13 2252 winner else winner) = _
  rw [if_pos (by decide), root72Child13]
def root72Candidates14 : List Nat := [5581, 6673, 7873, 12829, 14713]
set_option maxRecDepth 8192 in
theorem root72Candidates14_checked : candidates edge 4 379 2221 root72Tail15 winner.bound = root72Candidates14 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child14 : search edge 3 [379, 2221] root72Candidates14 2600 winner = winner := by decide
theorem root72Step14 : scan edge (search edge 3) 4 [379] 379 root72Tail14 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail15 winner := by
  rw [root72Tail14, scan, if_neg (by decide), root72Candidates14_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail15
    (if 3 ≤ root72Candidates14.length then search edge 3 [379, 2221] root72Candidates14 2600 winner else winner) = _
  rw [if_pos (by decide), root72Child14]
def root72Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root72Candidates15_checked : candidates edge 4 379 2281 root72Tail16 winner.bound = root72Candidates15 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root72Step15 : scan edge (search edge 3) 4 [379] 379 root72Tail15 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail16 winner := by
  rw [root72Tail15, scan, if_neg (by decide), root72Candidates15_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail16
    (if 3 ≤ root72Candidates15.length then search edge 3 [379, 2281] root72Candidates15 2660 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates16 : List Nat := [6379, 11353]
set_option maxRecDepth 8192 in
theorem root72Candidates16_checked : candidates edge 4 379 2791 root72Tail17 winner.bound = root72Candidates16 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root72Step16 : scan edge (search edge 3) 4 [379] 379 root72Tail16 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail17 winner := by
  rw [root72Tail16, scan, if_neg (by decide), root72Candidates16_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail17
    (if 3 ≤ root72Candidates16.length then search edge 3 [379, 2791] root72Candidates16 3170 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates17 : List Nat := [6673, 15259]
set_option maxRecDepth 8192 in
theorem root72Candidates17_checked : candidates edge 4 379 3061 root72Tail18 winner.bound = root72Candidates17 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root72Step17 : scan edge (search edge 3) 4 [379] 379 root72Tail17 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail18 winner := by
  rw [root72Tail17, scan, if_neg (by decide), root72Candidates17_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail18
    (if 3 ≤ root72Candidates17.length then search edge 3 [379, 3061] root72Candidates17 3440 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates18 : List Nat := [7699, 12037]
set_option maxRecDepth 8192 in
theorem root72Candidates18_checked : candidates edge 4 379 3079 root72Tail19 winner.bound = root72Candidates18 := by
  unfold candidates
  have row : edge 3079 = (SortedNatTree.contains FastRows.row3079) := by
    change Data.row3079.contains = (SortedNatTree.contains FastRows.row3079)
    exact FastRows.row3079_checked.symm
  rw [row]
  decide
theorem root72Step18 : scan edge (search edge 3) 4 [379] 379 root72Tail18 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail19 winner := by
  rw [root72Tail18, scan, if_neg (by decide), root72Candidates18_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail19
    (if 3 ≤ root72Candidates18.length then search edge 3 [379, 3079] root72Candidates18 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates19 : List Nat := [10657, 18427, 22483]
set_option maxRecDepth 8192 in
theorem root72Candidates19_checked : candidates edge 4 379 3121 root72Tail20 winner.bound = root72Candidates19 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root72Child19 : search edge 3 [379, 3121] root72Candidates19 3500 winner = winner := by decide
theorem root72Step19 : scan edge (search edge 3) 4 [379] 379 root72Tail19 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail20 winner := by
  rw [root72Tail19, scan, if_neg (by decide), root72Candidates19_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail20
    (if 3 ≤ root72Candidates19.length then search edge 3 [379, 3121] root72Candidates19 3500 winner else winner) = _
  rw [if_pos (by decide), root72Child19]
def root72Candidates20 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root72Candidates20_checked : candidates edge 4 379 3259 root72Tail21 winner.bound = root72Candidates20 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
theorem root72Step20 : scan edge (search edge 3) 4 [379] 379 root72Tail20 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail21 winner := by
  rw [root72Tail20, scan, if_neg (by decide), root72Candidates20_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail21
    (if 3 ≤ root72Candidates20.length then search edge 3 [379, 3259] root72Candidates20 3638 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates21 : List Nat := [16189]
set_option maxRecDepth 8192 in
theorem root72Candidates21_checked : candidates edge 4 379 3931 root72Tail22 winner.bound = root72Candidates21 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root72Step21 : scan edge (search edge 3) 4 [379] 379 root72Tail21 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail22 winner := by
  rw [root72Tail21, scan, if_neg (by decide), root72Candidates21_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail22
    (if 3 ≤ root72Candidates21.length then search edge 3 [379, 3931] root72Candidates21 4310 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates22 : List Nat := [15259]
set_option maxRecDepth 8192 in
theorem root72Candidates22_checked : candidates edge 4 379 4051 root72Tail23 winner.bound = root72Candidates22 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root72Step22 : scan edge (search edge 3) 4 [379] 379 root72Tail22 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail23 winner := by
  rw [root72Tail22, scan, if_neg (by decide), root72Candidates22_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail23
    (if 3 ≤ root72Candidates22.length then search edge 3 [379, 4051] root72Candidates22 4430 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates23 : List Nat := [15187]
set_option maxRecDepth 8192 in
theorem root72Candidates23_checked : candidates edge 4 379 4423 root72Tail24 winner.bound = root72Candidates23 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root72Step23 : scan edge (search edge 3) 4 [379] 379 root72Tail23 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail24 winner := by
  rw [root72Tail23, scan, if_neg (by decide), root72Candidates23_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail24
    (if 3 ≤ root72Candidates23.length then search edge 3 [379, 4423] root72Candidates23 4802 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates24 : List Nat := [5563]
set_option maxRecDepth 8192 in
theorem root72Candidates24_checked : candidates edge 4 379 4951 root72Tail25 winner.bound = root72Candidates24 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root72Step24 : scan edge (search edge 3) 4 [379] 379 root72Tail24 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail25 winner := by
  rw [root72Tail24, scan, if_neg (by decide), root72Candidates24_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail25
    (if 3 ≤ root72Candidates24.length then search edge 3 [379, 4951] root72Candidates24 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root72Candidates25_checked : candidates edge 4 379 5563 root72Tail26 winner.bound = root72Candidates25 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root72Step25 : scan edge (search edge 3) 4 [379] 379 root72Tail25 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail26 winner := by
  rw [root72Tail25, scan, if_neg (by decide), root72Candidates25_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail26
    (if 3 ≤ root72Candidates25.length then search edge 3 [379, 5563] root72Candidates25 5942 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root72Candidates26_checked : candidates edge 4 379 5569 root72Tail27 winner.bound = root72Candidates26 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root72Step26 : scan edge (search edge 3) 4 [379] 379 root72Tail26 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail27 winner := by
  rw [root72Tail26, scan, if_neg (by decide), root72Candidates26_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail27
    (if 3 ≤ root72Candidates26.length then search edge 3 [379, 5569] root72Candidates26 5948 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates27 : List Nat := []
set_option maxRecDepth 8192 in
theorem root72Candidates27_checked : candidates edge 4 379 5581 root72Tail28 winner.bound = root72Candidates27 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root72Step27 : scan edge (search edge 3) 4 [379] 379 root72Tail27 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail28 winner := by
  rw [root72Tail27, scan, if_neg (by decide), root72Candidates27_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail28
    (if 3 ≤ root72Candidates27.length then search edge 3 [379, 5581] root72Candidates27 5960 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates28 : List Nat := [12829]
set_option maxRecDepth 8192 in
theorem root72Candidates28_checked : candidates edge 4 379 6367 root72Tail29 winner.bound = root72Candidates28 := by
  unfold candidates
  have row : edge 6367 = (SortedNatTree.contains FastRows.row6367) := by
    change Data.row6367.contains = (SortedNatTree.contains FastRows.row6367)
    exact FastRows.row6367_checked.symm
  rw [row]
  decide
theorem root72Step28 : scan edge (search edge 3) 4 [379] 379 root72Tail28 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail29 winner := by
  rw [root72Tail28, scan, if_neg (by decide), root72Candidates28_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail29
    (if 3 ≤ root72Candidates28.length then search edge 3 [379, 6367] root72Candidates28 6746 winner else winner) = _
  rw [if_neg (by decide)]
def root72Candidates29 : List Nat := [9811, 16561]
set_option maxRecDepth 8192 in
theorem root72Candidates29_checked : candidates edge 4 379 6379 root72Tail30 winner.bound = root72Candidates29 := by
  unfold candidates
  have row : edge 6379 = (SortedNatTree.contains FastRows.row6379) := by
    change Data.row6379.contains = (SortedNatTree.contains FastRows.row6379)
    exact FastRows.row6379_checked.symm
  rw [row]
  decide
theorem root72Step29 : scan edge (search edge 3) 4 [379] 379 root72Tail29 winner =
    scan edge (search edge 3) 4 [379] 379 root72Tail30 winner := by
  rw [root72Tail29, scan, if_neg (by decide), root72Candidates29_checked]
  change scan edge (search edge 3) 4 [379] 379 root72Tail30
    (if 3 ≤ root72Candidates29.length then search edge 3 [379, 6379] root72Candidates29 6758 winner else winner) = _
  rw [if_neg (by decide)]
theorem root72Scan30 : scan edge (search edge 3) 4 [379] 379 root72Tail30 winner = winner := by rw [root72Tail30, scan, if_pos (by decide)]
theorem root72Scan29 : scan edge (search edge 3) 4 [379] 379 root72Tail29 winner = winner :=
  (root72Step29).trans root72Scan30
theorem root72Scan28 : scan edge (search edge 3) 4 [379] 379 root72Tail28 winner = winner :=
  (root72Step28).trans root72Scan29
theorem root72Scan27 : scan edge (search edge 3) 4 [379] 379 root72Tail27 winner = winner :=
  (root72Step27).trans root72Scan28
theorem root72Scan26 : scan edge (search edge 3) 4 [379] 379 root72Tail26 winner = winner :=
  (root72Step26).trans root72Scan27
theorem root72Scan25 : scan edge (search edge 3) 4 [379] 379 root72Tail25 winner = winner :=
  (root72Step25).trans root72Scan26
theorem root72Scan24 : scan edge (search edge 3) 4 [379] 379 root72Tail24 winner = winner :=
  (root72Step24).trans root72Scan25
theorem root72Scan23 : scan edge (search edge 3) 4 [379] 379 root72Tail23 winner = winner :=
  (root72Step23).trans root72Scan24
theorem root72Scan22 : scan edge (search edge 3) 4 [379] 379 root72Tail22 winner = winner :=
  (root72Step22).trans root72Scan23
theorem root72Scan21 : scan edge (search edge 3) 4 [379] 379 root72Tail21 winner = winner :=
  (root72Step21).trans root72Scan22
theorem root72Scan20 : scan edge (search edge 3) 4 [379] 379 root72Tail20 winner = winner :=
  (root72Step20).trans root72Scan21
theorem root72Scan19 : scan edge (search edge 3) 4 [379] 379 root72Tail19 winner = winner :=
  (root72Step19).trans root72Scan20
theorem root72Scan18 : scan edge (search edge 3) 4 [379] 379 root72Tail18 winner = winner :=
  (root72Step18).trans root72Scan19
theorem root72Scan17 : scan edge (search edge 3) 4 [379] 379 root72Tail17 winner = winner :=
  (root72Step17).trans root72Scan18
theorem root72Scan16 : scan edge (search edge 3) 4 [379] 379 root72Tail16 winner = winner :=
  (root72Step16).trans root72Scan17
theorem root72Scan15 : scan edge (search edge 3) 4 [379] 379 root72Tail15 winner = winner :=
  (root72Step15).trans root72Scan16
theorem root72Scan14 : scan edge (search edge 3) 4 [379] 379 root72Tail14 winner = winner :=
  (root72Step14).trans root72Scan15
theorem root72Scan13 : scan edge (search edge 3) 4 [379] 379 root72Tail13 winner = winner :=
  (root72Step13).trans root72Scan14
theorem root72Scan12 : scan edge (search edge 3) 4 [379] 379 root72Tail12 winner = winner :=
  (root72Step12).trans root72Scan13
theorem root72Scan11 : scan edge (search edge 3) 4 [379] 379 root72Tail11 winner = winner :=
  (root72Step11).trans root72Scan12
theorem root72Scan10 : scan edge (search edge 3) 4 [379] 379 root72Tail10 winner = winner :=
  (root72Step10).trans root72Scan11
theorem root72Scan9 : scan edge (search edge 3) 4 [379] 379 root72Tail9 winner = winner :=
  (root72Step9).trans root72Scan10
theorem root72Scan8 : scan edge (search edge 3) 4 [379] 379 root72Tail8 winner = winner :=
  (root72Step8).trans root72Scan9
theorem root72Scan7 : scan edge (search edge 3) 4 [379] 379 root72Tail7 winner = winner :=
  (root72Step7).trans root72Scan8
theorem root72Scan6 : scan edge (search edge 3) 4 [379] 379 root72Tail6 winner = winner :=
  (root72Step6).trans root72Scan7
theorem root72Scan5 : scan edge (search edge 3) 4 [379] 379 root72Tail5 winner = winner :=
  (root72Step5).trans root72Scan6
theorem root72Scan4 : scan edge (search edge 3) 4 [379] 379 root72Tail4 winner = winner :=
  (root72Step4).trans root72Scan5
theorem root72Scan3 : scan edge (search edge 3) 4 [379] 379 root72Tail3 winner = winner :=
  (root72Step3).trans root72Scan4
theorem root72Scan2 : scan edge (search edge 3) 4 [379] 379 root72Tail2 winner = winner :=
  (root72Step2).trans root72Scan3
theorem root72Scan1 : scan edge (search edge 3) 4 [379] 379 root72Tail1 winner = winner :=
  (root72Step1).trans root72Scan2
theorem root72Scan0 : scan edge (search edge 3) 4 [379] 379 root72Tail0 winner = winner :=
  (root72Step0).trans root72Scan1
set_option maxRecDepth 4096 in
theorem root72Branch : search edge 4 [379] root72Candidates 379 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [379] 379 root72Tail0 winner = winner
  exact root72Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
