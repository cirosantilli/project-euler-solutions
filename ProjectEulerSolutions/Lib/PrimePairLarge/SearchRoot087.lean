import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root87Candidates : List Nat := Data.row463
set_option maxRecDepth 8192 in
theorem root87Candidates_checked : candidates edge 5 0 463 tail88 winner.bound = root87Candidates := by
  apply candidates_eq_row edge 5 0 463 tail88 root87Candidates winner.bound tail88_ordered
  · decide
  · decide
  · funext q
    rfl
def root87Tail61 : List Nat := []
def root87Tail60 : List Nat := 25171 :: root87Tail61
def root87Tail59 : List Nat := 24517 :: root87Tail60
def root87Tail58 : List Nat := 24379 :: root87Tail59
def root87Tail57 : List Nat := 23911 :: root87Tail58
def root87Tail56 : List Nat := 23887 :: root87Tail57
def root87Tail55 : List Nat := 23869 :: root87Tail56
def root87Tail54 : List Nat := 22279 :: root87Tail55
def root87Tail53 : List Nat := 22027 :: root87Tail54
def root87Tail52 : List Nat := 21859 :: root87Tail53
def root87Tail51 : List Nat := 20773 :: root87Tail52
def root87Tail50 : List Nat := 20173 :: root87Tail51
def root87Tail49 : List Nat := 20023 :: root87Tail50
def root87Tail48 : List Nat := 20011 :: root87Tail49
def root87Tail47 : List Nat := 19753 :: root87Tail48
def root87Tail46 : List Nat := 19501 :: root87Tail47
def root87Tail45 : List Nat := 19333 :: root87Tail46
def root87Tail44 : List Nat := 16981 :: root87Tail45
def root87Tail43 : List Nat := 16657 :: root87Tail44
def root87Tail42 : List Nat := 15013 :: root87Tail43
def root87Tail41 : List Nat := 14143 :: root87Tail42
def root87Tail40 : List Nat := 13807 :: root87Tail41
def root87Tail39 : List Nat := 13759 :: root87Tail40
def root87Tail38 : List Nat := 13249 :: root87Tail39
def root87Tail37 : List Nat := 12697 :: root87Tail38
def root87Tail36 : List Nat := 11491 :: root87Tail37
def root87Tail35 : List Nat := 11119 :: root87Tail36
def root87Tail34 : List Nat := 10909 :: root87Tail35
def root87Tail33 : List Nat := 10867 :: root87Tail34
def root87Tail32 : List Nat := 10723 :: root87Tail33
def root87Tail31 : List Nat := 10009 :: root87Tail32
def root87Tail30 : List Nat := 9931 :: root87Tail31
def root87Tail29 : List Nat := 9883 :: root87Tail30
def root87Tail28 : List Nat := 9127 :: root87Tail29
def root87Tail27 : List Nat := 8761 :: root87Tail28
def root87Tail26 : List Nat := 8461 :: root87Tail27
def root87Tail25 : List Nat := 7669 :: root87Tail26
def root87Tail24 : List Nat := 7489 :: root87Tail25
def root87Tail23 : List Nat := 7333 :: root87Tail24
def root87Tail22 : List Nat := 6967 :: root87Tail23
def root87Tail21 : List Nat := 6451 :: root87Tail22
def root87Tail20 : List Nat := 5623 :: root87Tail21
def root87Tail19 : List Nat := 5569 :: root87Tail20
def root87Tail18 : List Nat := 5437 :: root87Tail19
def root87Tail17 : List Nat := 4933 :: root87Tail18
def root87Tail16 : List Nat := 4723 :: root87Tail17
def root87Tail15 : List Nat := 4111 :: root87Tail16
def root87Tail14 : List Nat := 3931 :: root87Tail15
def root87Tail13 : List Nat := 3793 :: root87Tail14
def root87Tail12 : List Nat := 3457 :: root87Tail13
def root87Tail11 : List Nat := 3331 :: root87Tail12
def root87Tail10 : List Nat := 3259 :: root87Tail11
def root87Tail9 : List Nat := 3001 :: root87Tail10
def root87Tail8 : List Nat := 2917 :: root87Tail9
def root87Tail7 : List Nat := 2473 :: root87Tail8
def root87Tail6 : List Nat := 1663 :: root87Tail7
def root87Tail5 : List Nat := 1489 :: root87Tail6
def root87Tail4 : List Nat := 1171 :: root87Tail5
def root87Tail3 : List Nat := 829 :: root87Tail4
def root87Tail2 : List Nat := 643 :: root87Tail3
def root87Tail1 : List Nat := 613 :: root87Tail2
def root87Tail0 : List Nat := 523 :: root87Tail1
def root87Candidates0 : List Nat := [1663, 6967, 23911]
set_option maxRecDepth 8192 in
theorem root87Candidates0_checked : candidates edge 4 463 523 root87Tail1 winner.bound = root87Candidates0 := by
  unfold candidates
  have row : edge 523 = (SortedNatTree.contains FastRows.row523) := by
    change Data.row523.contains = (SortedNatTree.contains FastRows.row523)
    exact FastRows.row523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root87Child0 : search edge 3 [463, 523] root87Candidates0 986 winner = winner := by decide
theorem root87Step0 : scan edge (search edge 3) 4 [463] 463 root87Tail0 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail1 winner := by
  rw [root87Tail0, scan, if_neg (by decide), root87Candidates0_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail1
    (if 3 ≤ root87Candidates0.length then search edge 3 [463, 523] root87Candidates0 986 winner else winner) = _
  rw [if_pos (by decide), root87Child0]
def root87Candidates1 : List Nat := [829, 6967, 19501, 24379]
set_option maxRecDepth 8192 in
theorem root87Candidates1_checked : candidates edge 4 463 613 root87Tail2 winner.bound = root87Candidates1 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root87Child1 : search edge 3 [463, 613] root87Candidates1 1076 winner = winner := by decide
theorem root87Step1 : scan edge (search edge 3) 4 [463] 463 root87Tail1 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail2 winner := by
  rw [root87Tail1, scan, if_neg (by decide), root87Candidates1_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail2
    (if 3 ≤ root87Candidates1.length then search edge 3 [463, 613] root87Candidates1 1076 winner else winner) = _
  rw [if_pos (by decide), root87Child1]
def root87Candidates2 : List Nat := [2917, 4111, 16657]
set_option maxRecDepth 8192 in
theorem root87Candidates2_checked : candidates edge 4 463 643 root87Tail3 winner.bound = root87Candidates2 := by
  unfold candidates
  have row : edge 643 = (SortedNatTree.contains FastRows.row643) := by
    change Data.row643.contains = (SortedNatTree.contains FastRows.row643)
    exact FastRows.row643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root87Child2 : search edge 3 [463, 643] root87Candidates2 1106 winner = winner := by decide
theorem root87Step2 : scan edge (search edge 3) 4 [463] 463 root87Tail2 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail3 winner := by
  rw [root87Tail2, scan, if_neg (by decide), root87Candidates2_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail3
    (if 3 ≤ root87Candidates2.length then search edge 3 [463, 643] root87Candidates2 1106 winner else winner) = _
  rw [if_pos (by decide), root87Child2]
def root87Candidates3 : List Nat := [22027]
set_option maxRecDepth 8192 in
theorem root87Candidates3_checked : candidates edge 4 463 829 root87Tail4 winner.bound = root87Candidates3 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
theorem root87Step3 : scan edge (search edge 3) 4 [463] 463 root87Tail3 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail4 winner := by
  rw [root87Tail3, scan, if_neg (by decide), root87Candidates3_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail4
    (if 3 ≤ root87Candidates3.length then search edge 3 [463, 829] root87Candidates3 1292 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates4 : List Nat := [4933, 8461]
set_option maxRecDepth 8192 in
theorem root87Candidates4_checked : candidates edge 4 463 1171 root87Tail5 winner.bound = root87Candidates4 := by
  unfold candidates
  have row : edge 1171 = (SortedNatTree.contains FastRows.row1171) := by
    change Data.row1171.contains = (SortedNatTree.contains FastRows.row1171)
    exact FastRows.row1171_checked.symm
  rw [row]
  decide
theorem root87Step4 : scan edge (search edge 3) 4 [463] 463 root87Tail4 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail5 winner := by
  rw [root87Tail4, scan, if_neg (by decide), root87Candidates4_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail5
    (if 3 ≤ root87Candidates4.length then search edge 3 [463, 1171] root87Candidates4 1634 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates5 : List Nat := [16981]
set_option maxRecDepth 8192 in
theorem root87Candidates5_checked : candidates edge 4 463 1489 root87Tail6 winner.bound = root87Candidates5 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
theorem root87Step5 : scan edge (search edge 3) 4 [463] 463 root87Tail5 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail6 winner := by
  rw [root87Tail5, scan, if_neg (by decide), root87Candidates5_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail6
    (if 3 ≤ root87Candidates5.length then search edge 3 [463, 1489] root87Candidates5 1952 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates6 : List Nat := [10909, 14143]
set_option maxRecDepth 8192 in
theorem root87Candidates6_checked : candidates edge 4 463 1663 root87Tail7 winner.bound = root87Candidates6 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
theorem root87Step6 : scan edge (search edge 3) 4 [463] 463 root87Tail6 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail7 winner := by
  rw [root87Tail6, scan, if_neg (by decide), root87Candidates6_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail7
    (if 3 ≤ root87Candidates6.length then search edge 3 [463, 1663] root87Candidates6 2126 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates7 : List Nat := [3001, 5437, 13249]
set_option maxRecDepth 8192 in
theorem root87Candidates7_checked : candidates edge 4 463 2473 root87Tail8 winner.bound = root87Candidates7 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root87Child7 : search edge 3 [463, 2473] root87Candidates7 2936 winner = winner := by decide
theorem root87Step7 : scan edge (search edge 3) 4 [463] 463 root87Tail7 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail8 winner := by
  rw [root87Tail7, scan, if_neg (by decide), root87Candidates7_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail8
    (if 3 ≤ root87Candidates7.length then search edge 3 [463, 2473] root87Candidates7 2936 winner else winner) = _
  rw [if_pos (by decide), root87Child7]
def root87Candidates8 : List Nat := [5437]
set_option maxRecDepth 8192 in
theorem root87Candidates8_checked : candidates edge 4 463 2917 root87Tail9 winner.bound = root87Candidates8 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root87Step8 : scan edge (search edge 3) 4 [463] 463 root87Tail8 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail9 winner := by
  rw [root87Tail8, scan, if_neg (by decide), root87Candidates8_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail9
    (if 3 ≤ root87Candidates8.length then search edge 3 [463, 2917] root87Candidates8 3380 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates9 : List Nat := [9931, 16981]
set_option maxRecDepth 8192 in
theorem root87Candidates9_checked : candidates edge 4 463 3001 root87Tail10 winner.bound = root87Candidates9 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
theorem root87Step9 : scan edge (search edge 3) 4 [463] 463 root87Tail9 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail10 winner := by
  rw [root87Tail9, scan, if_neg (by decide), root87Candidates9_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail10
    (if 3 ≤ root87Candidates9.length then search edge 3 [463, 3001] root87Candidates9 3464 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates10 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root87Candidates10_checked : candidates edge 4 463 3259 root87Tail11 winner.bound = root87Candidates10 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
theorem root87Step10 : scan edge (search edge 3) 4 [463] 463 root87Tail10 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail11 winner := by
  rw [root87Tail10, scan, if_neg (by decide), root87Candidates10_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail11
    (if 3 ≤ root87Candidates10.length then search edge 3 [463, 3259] root87Candidates10 3722 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root87Candidates11_checked : candidates edge 4 463 3331 root87Tail12 winner.bound = root87Candidates11 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root87Step11 : scan edge (search edge 3) 4 [463] 463 root87Tail11 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail12 winner := by
  rw [root87Tail11, scan, if_neg (by decide), root87Candidates11_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail12
    (if 3 ≤ root87Candidates11.length then search edge 3 [463, 3331] root87Candidates11 3794 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates12 : List Nat := [9931]
set_option maxRecDepth 8192 in
theorem root87Candidates12_checked : candidates edge 4 463 3457 root87Tail13 winner.bound = root87Candidates12 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root87Step12 : scan edge (search edge 3) 4 [463] 463 root87Tail12 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail13 winner := by
  rw [root87Tail12, scan, if_neg (by decide), root87Candidates12_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail13
    (if 3 ≤ root87Candidates12.length then search edge 3 [463, 3457] root87Candidates12 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates13 : List Nat := [16981]
set_option maxRecDepth 8192 in
theorem root87Candidates13_checked : candidates edge 4 463 3793 root87Tail14 winner.bound = root87Candidates13 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
theorem root87Step13 : scan edge (search edge 3) 4 [463] 463 root87Tail13 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail14 winner := by
  rw [root87Tail13, scan, if_neg (by decide), root87Candidates13_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail14
    (if 3 ≤ root87Candidates13.length then search edge 3 [463, 3793] root87Candidates13 4256 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates14 : List Nat := [20011]
set_option maxRecDepth 8192 in
theorem root87Candidates14_checked : candidates edge 4 463 3931 root87Tail15 winner.bound = root87Candidates14 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root87Step14 : scan edge (search edge 3) 4 [463] 463 root87Tail14 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail15 winner := by
  rw [root87Tail14, scan, if_neg (by decide), root87Candidates14_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail15
    (if 3 ≤ root87Candidates14.length then search edge 3 [463, 3931] root87Candidates14 4394 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates15 : List Nat := [13249]
set_option maxRecDepth 8192 in
theorem root87Candidates15_checked : candidates edge 4 463 4111 root87Tail16 winner.bound = root87Candidates15 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root87Step15 : scan edge (search edge 3) 4 [463] 463 root87Tail15 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail16 winner := by
  rw [root87Tail15, scan, if_neg (by decide), root87Candidates15_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail16
    (if 3 ≤ root87Candidates15.length then search edge 3 [463, 4111] root87Candidates15 4574 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates16 : List Nat := [5623, 13249]
set_option maxRecDepth 8192 in
theorem root87Candidates16_checked : candidates edge 4 463 4723 root87Tail17 winner.bound = root87Candidates16 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root87Step16 : scan edge (search edge 3) 4 [463] 463 root87Tail16 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail17 winner := by
  rw [root87Tail16, scan, if_neg (by decide), root87Candidates16_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail17
    (if 3 ≤ root87Candidates16.length then search edge 3 [463, 4723] root87Candidates16 5186 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates17 : List Nat := [10009]
set_option maxRecDepth 8192 in
theorem root87Candidates17_checked : candidates edge 4 463 4933 root87Tail18 winner.bound = root87Candidates17 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root87Step17 : scan edge (search edge 3) 4 [463] 463 root87Tail17 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail18 winner := by
  rw [root87Tail17, scan, if_neg (by decide), root87Candidates17_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail18
    (if 3 ≤ root87Candidates17.length then search edge 3 [463, 4933] root87Candidates17 5396 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates18 : List Nat := [9883, 20011]
set_option maxRecDepth 8192 in
theorem root87Candidates18_checked : candidates edge 4 463 5437 root87Tail19 winner.bound = root87Candidates18 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root87Step18 : scan edge (search edge 3) 4 [463] 463 root87Tail18 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail19 winner := by
  rw [root87Tail18, scan, if_neg (by decide), root87Candidates18_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail19
    (if 3 ≤ root87Candidates18.length then search edge 3 [463, 5437] root87Candidates18 5900 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates19 : List Nat := [6451, 11119]
set_option maxRecDepth 8192 in
theorem root87Candidates19_checked : candidates edge 4 463 5569 root87Tail20 winner.bound = root87Candidates19 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root87Step19 : scan edge (search edge 3) 4 [463] 463 root87Tail19 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail20 winner := by
  rw [root87Tail19, scan, if_neg (by decide), root87Candidates19_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail20
    (if 3 ≤ root87Candidates19.length then search edge 3 [463, 5569] root87Candidates19 6032 winner else winner) = _
  rw [if_neg (by decide)]
def root87Candidates20 : List Nat := [10009]
set_option maxRecDepth 8192 in
theorem root87Candidates20_checked : candidates edge 4 463 5623 root87Tail21 winner.bound = root87Candidates20 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root87Step20 : scan edge (search edge 3) 4 [463] 463 root87Tail20 winner =
    scan edge (search edge 3) 4 [463] 463 root87Tail21 winner := by
  rw [root87Tail20, scan, if_neg (by decide), root87Candidates20_checked]
  change scan edge (search edge 3) 4 [463] 463 root87Tail21
    (if 3 ≤ root87Candidates20.length then search edge 3 [463, 5623] root87Candidates20 6086 winner else winner) = _
  rw [if_neg (by decide)]
theorem root87Scan21 : scan edge (search edge 3) 4 [463] 463 root87Tail21 winner = winner := by rw [root87Tail21, scan, if_pos (by decide)]
theorem root87Scan20 : scan edge (search edge 3) 4 [463] 463 root87Tail20 winner = winner :=
  (root87Step20).trans root87Scan21
theorem root87Scan19 : scan edge (search edge 3) 4 [463] 463 root87Tail19 winner = winner :=
  (root87Step19).trans root87Scan20
theorem root87Scan18 : scan edge (search edge 3) 4 [463] 463 root87Tail18 winner = winner :=
  (root87Step18).trans root87Scan19
theorem root87Scan17 : scan edge (search edge 3) 4 [463] 463 root87Tail17 winner = winner :=
  (root87Step17).trans root87Scan18
theorem root87Scan16 : scan edge (search edge 3) 4 [463] 463 root87Tail16 winner = winner :=
  (root87Step16).trans root87Scan17
theorem root87Scan15 : scan edge (search edge 3) 4 [463] 463 root87Tail15 winner = winner :=
  (root87Step15).trans root87Scan16
theorem root87Scan14 : scan edge (search edge 3) 4 [463] 463 root87Tail14 winner = winner :=
  (root87Step14).trans root87Scan15
theorem root87Scan13 : scan edge (search edge 3) 4 [463] 463 root87Tail13 winner = winner :=
  (root87Step13).trans root87Scan14
theorem root87Scan12 : scan edge (search edge 3) 4 [463] 463 root87Tail12 winner = winner :=
  (root87Step12).trans root87Scan13
theorem root87Scan11 : scan edge (search edge 3) 4 [463] 463 root87Tail11 winner = winner :=
  (root87Step11).trans root87Scan12
theorem root87Scan10 : scan edge (search edge 3) 4 [463] 463 root87Tail10 winner = winner :=
  (root87Step10).trans root87Scan11
theorem root87Scan9 : scan edge (search edge 3) 4 [463] 463 root87Tail9 winner = winner :=
  (root87Step9).trans root87Scan10
theorem root87Scan8 : scan edge (search edge 3) 4 [463] 463 root87Tail8 winner = winner :=
  (root87Step8).trans root87Scan9
theorem root87Scan7 : scan edge (search edge 3) 4 [463] 463 root87Tail7 winner = winner :=
  (root87Step7).trans root87Scan8
theorem root87Scan6 : scan edge (search edge 3) 4 [463] 463 root87Tail6 winner = winner :=
  (root87Step6).trans root87Scan7
theorem root87Scan5 : scan edge (search edge 3) 4 [463] 463 root87Tail5 winner = winner :=
  (root87Step5).trans root87Scan6
theorem root87Scan4 : scan edge (search edge 3) 4 [463] 463 root87Tail4 winner = winner :=
  (root87Step4).trans root87Scan5
theorem root87Scan3 : scan edge (search edge 3) 4 [463] 463 root87Tail3 winner = winner :=
  (root87Step3).trans root87Scan4
theorem root87Scan2 : scan edge (search edge 3) 4 [463] 463 root87Tail2 winner = winner :=
  (root87Step2).trans root87Scan3
theorem root87Scan1 : scan edge (search edge 3) 4 [463] 463 root87Tail1 winner = winner :=
  (root87Step1).trans root87Scan2
theorem root87Scan0 : scan edge (search edge 3) 4 [463] 463 root87Tail0 winner = winner :=
  (root87Step0).trans root87Scan1
set_option maxRecDepth 4096 in
theorem root87Branch : search edge 4 [463] root87Candidates 463 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [463] 463 root87Tail0 winner = winner
  exact root87Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
