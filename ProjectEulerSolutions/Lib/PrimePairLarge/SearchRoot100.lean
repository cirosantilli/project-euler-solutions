import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root100Candidates : List Nat := Data.row563
set_option maxRecDepth 8192 in
theorem root100Candidates_checked : candidates edge 5 0 563 tail101 winner.bound = root100Candidates := by
  apply candidates_eq_row edge 5 0 563 tail101 root100Candidates winner.bound tail101_ordered
  · decide
  · decide
  · funext q
    rfl
def root100Tail61 : List Nat := []
def root100Tail60 : List Nat := 25463 :: root100Tail61
def root100Tail59 : List Nat := 25013 :: root100Tail60
def root100Tail58 : List Nat := 23909 :: root100Tail59
def root100Tail57 : List Nat := 23741 :: root100Tail58
def root100Tail56 : List Nat := 22073 :: root100Tail57
def root100Tail55 : List Nat := 21893 :: root100Tail56
def root100Tail54 : List Nat := 21569 :: root100Tail55
def root100Tail53 : List Nat := 21191 :: root100Tail54
def root100Tail52 : List Nat := 19577 :: root100Tail53
def root100Tail51 : List Nat := 19559 :: root100Tail52
def root100Tail50 : List Nat := 17291 :: root100Tail51
def root100Tail49 : List Nat := 17189 :: root100Tail50
def root100Tail48 : List Nat := 16829 :: root100Tail49
def root100Tail47 : List Nat := 16661 :: root100Tail48
def root100Tail46 : List Nat := 16097 :: root100Tail47
def root100Tail45 : List Nat := 15269 :: root100Tail46
def root100Tail44 : List Nat := 15131 :: root100Tail45
def root100Tail43 : List Nat := 14939 :: root100Tail44
def root100Tail42 : List Nat := 14831 :: root100Tail43
def root100Tail41 : List Nat := 14747 :: root100Tail42
def root100Tail40 : List Nat := 14543 :: root100Tail41
def root100Tail39 : List Nat := 13841 :: root100Tail40
def root100Tail38 : List Nat := 13781 :: root100Tail39
def root100Tail37 : List Nat := 13421 :: root100Tail38
def root100Tail36 : List Nat := 13151 :: root100Tail37
def root100Tail35 : List Nat := 12527 :: root100Tail36
def root100Tail34 : List Nat := 12401 :: root100Tail35
def root100Tail33 : List Nat := 11699 :: root100Tail34
def root100Tail32 : List Nat := 11681 :: root100Tail33
def root100Tail31 : List Nat := 10733 :: root100Tail32
def root100Tail30 : List Nat := 10061 :: root100Tail31
def root100Tail29 : List Nat := 9749 :: root100Tail30
def root100Tail28 : List Nat := 9479 :: root100Tail29
def root100Tail27 : List Nat := 8537 :: root100Tail28
def root100Tail26 : List Nat := 7481 :: root100Tail27
def root100Tail25 : List Nat := 7043 :: root100Tail26
def root100Tail24 : List Nat := 7019 :: root100Tail25
def root100Tail23 : List Nat := 6791 :: root100Tail24
def root100Tail22 : List Nat := 6581 :: root100Tail23
def root100Tail21 : List Nat := 6551 :: root100Tail22
def root100Tail20 : List Nat := 5717 :: root100Tail21
def root100Tail19 : List Nat := 5501 :: root100Tail20
def root100Tail18 : List Nat := 5297 :: root100Tail19
def root100Tail17 : List Nat := 5261 :: root100Tail18
def root100Tail16 : List Nat := 5237 :: root100Tail17
def root100Tail15 : List Nat := 4799 :: root100Tail16
def root100Tail14 : List Nat := 4493 :: root100Tail15
def root100Tail13 : List Nat := 3821 :: root100Tail14
def root100Tail12 : List Nat := 3119 :: root100Tail13
def root100Tail11 : List Nat := 2909 :: root100Tail12
def root100Tail10 : List Nat := 2267 :: root100Tail11
def root100Tail9 : List Nat := 1979 :: root100Tail10
def root100Tail8 : List Nat := 1973 :: root100Tail9
def root100Tail7 : List Nat := 1601 :: root100Tail8
def root100Tail6 : List Nat := 1523 :: root100Tail7
def root100Tail5 : List Nat := 1427 :: root100Tail6
def root100Tail4 : List Nat := 1259 :: root100Tail5
def root100Tail3 : List Nat := 1013 :: root100Tail4
def root100Tail2 : List Nat := 971 :: root100Tail3
def root100Tail1 : List Nat := 809 :: root100Tail2
def root100Tail0 : List Nat := 587 :: root100Tail1
def root100Candidates0 : List Nat := [7481, 9749, 22073]
set_option maxRecDepth 8192 in
theorem root100Candidates0_checked : candidates edge 4 563 587 root100Tail1 winner.bound = root100Candidates0 := by
  unfold candidates
  have row : edge 587 = (SortedNatTree.contains FastRows.row587) := by
    change Data.row587.contains = (SortedNatTree.contains FastRows.row587)
    exact FastRows.row587_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root100Child0 : search edge 3 [563, 587] root100Candidates0 1150 winner = winner := by decide
theorem root100Step0 : scan edge (search edge 3) 4 [563] 563 root100Tail0 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail1 winner := by
  rw [root100Tail0, scan, if_neg (by decide), root100Candidates0_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail1
    (if 3 ≤ root100Candidates0.length then search edge 3 [563, 587] root100Candidates0 1150 winner else winner) = _
  rw [if_pos (by decide), root100Child0]
def root100Candidates1 : List Nat := [1427, 5237]
set_option maxRecDepth 8192 in
theorem root100Candidates1_checked : candidates edge 4 563 809 root100Tail2 winner.bound = root100Candidates1 := by
  unfold candidates
  have row : edge 809 = (SortedNatTree.contains FastRows.row809) := by
    change Data.row809.contains = (SortedNatTree.contains FastRows.row809)
    exact FastRows.row809_checked.symm
  rw [row]
  decide
theorem root100Step1 : scan edge (search edge 3) 4 [563] 563 root100Tail1 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail2 winner := by
  rw [root100Tail1, scan, if_neg (by decide), root100Candidates1_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail2
    (if 3 ≤ root100Candidates1.length then search edge 3 [563, 809] root100Candidates1 1372 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates2 : List Nat := [5501]
set_option maxRecDepth 8192 in
theorem root100Candidates2_checked : candidates edge 4 563 971 root100Tail3 winner.bound = root100Candidates2 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
theorem root100Step2 : scan edge (search edge 3) 4 [563] 563 root100Tail2 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail3 winner := by
  rw [root100Tail2, scan, if_neg (by decide), root100Candidates2_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail3
    (if 3 ≤ root100Candidates2.length then search edge 3 [563, 971] root100Candidates2 1534 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates3 : List Nat := [21569]
set_option maxRecDepth 8192 in
theorem root100Candidates3_checked : candidates edge 4 563 1013 root100Tail4 winner.bound = root100Candidates3 := by
  unfold candidates
  have row : edge 1013 = (SortedNatTree.contains FastRows.row1013) := by
    change Data.row1013.contains = (SortedNatTree.contains FastRows.row1013)
    exact FastRows.row1013_checked.symm
  rw [row]
  decide
theorem root100Step3 : scan edge (search edge 3) 4 [563] 563 root100Tail3 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail4 winner := by
  rw [root100Tail3, scan, if_neg (by decide), root100Candidates3_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail4
    (if 3 ≤ root100Candidates3.length then search edge 3 [563, 1013] root100Candidates3 1576 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root100Candidates4_checked : candidates edge 4 563 1259 root100Tail5 winner.bound = root100Candidates4 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
theorem root100Step4 : scan edge (search edge 3) 4 [563] 563 root100Tail4 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail5 winner := by
  rw [root100Tail4, scan, if_neg (by decide), root100Candidates4_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail5
    (if 3 ≤ root100Candidates4.length then search edge 3 [563, 1259] root100Candidates4 1822 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates5 : List Nat := [5297, 5501, 9479, 19559]
set_option maxRecDepth 8192 in
theorem root100Candidates5_checked : candidates edge 4 563 1427 root100Tail6 winner.bound = root100Candidates5 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root100Child5 : search edge 3 [563, 1427] root100Candidates5 1990 winner = winner := by decide
theorem root100Step5 : scan edge (search edge 3) 4 [563] 563 root100Tail5 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail6 winner := by
  rw [root100Tail5, scan, if_neg (by decide), root100Candidates5_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail6
    (if 3 ≤ root100Candidates5.length then search edge 3 [563, 1427] root100Candidates5 1990 winner else winner) = _
  rw [if_pos (by decide), root100Child5]
def root100Candidates6 : List Nat := [2909, 7043, 19577]
set_option maxRecDepth 8192 in
theorem root100Candidates6_checked : candidates edge 4 563 1523 root100Tail7 winner.bound = root100Candidates6 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root100Child6 : search edge 3 [563, 1523] root100Candidates6 2086 winner = winner := by decide
theorem root100Step6 : scan edge (search edge 3) 4 [563] 563 root100Tail6 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail7 winner := by
  rw [root100Tail6, scan, if_neg (by decide), root100Candidates6_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail7
    (if 3 ≤ root100Candidates6.length then search edge 3 [563, 1523] root100Candidates6 2086 winner else winner) = _
  rw [if_pos (by decide), root100Child6]
def root100Candidates7 : List Nat := [9749]
set_option maxRecDepth 8192 in
theorem root100Candidates7_checked : candidates edge 4 563 1601 root100Tail8 winner.bound = root100Candidates7 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root100Step7 : scan edge (search edge 3) 4 [563] 563 root100Tail7 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail8 winner := by
  rw [root100Tail7, scan, if_neg (by decide), root100Candidates7_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail8
    (if 3 ≤ root100Candidates7.length then search edge 3 [563, 1601] root100Candidates7 2164 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates8 : List Nat := [1979, 4493]
set_option maxRecDepth 8192 in
theorem root100Candidates8_checked : candidates edge 4 563 1973 root100Tail9 winner.bound = root100Candidates8 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
theorem root100Step8 : scan edge (search edge 3) 4 [563] 563 root100Tail8 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail9 winner := by
  rw [root100Tail8, scan, if_neg (by decide), root100Candidates8_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail9
    (if 3 ≤ root100Candidates8.length then search edge 3 [563, 1973] root100Candidates8 2536 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates9 : List Nat := [16661]
set_option maxRecDepth 8192 in
theorem root100Candidates9_checked : candidates edge 4 563 1979 root100Tail10 winner.bound = root100Candidates9 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root100Step9 : scan edge (search edge 3) 4 [563] 563 root100Tail9 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail10 winner := by
  rw [root100Tail9, scan, if_neg (by decide), root100Candidates9_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail10
    (if 3 ≤ root100Candidates9.length then search edge 3 [563, 1979] root100Candidates9 2542 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates10 : List Nat := [6791]
set_option maxRecDepth 8192 in
theorem root100Candidates10_checked : candidates edge 4 563 2267 root100Tail11 winner.bound = root100Candidates10 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root100Step10 : scan edge (search edge 3) 4 [563] 563 root100Tail10 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail11 winner := by
  rw [root100Tail10, scan, if_neg (by decide), root100Candidates10_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail11
    (if 3 ≤ root100Candidates10.length then search edge 3 [563, 2267] root100Candidates10 2830 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root100Candidates11_checked : candidates edge 4 563 2909 root100Tail12 winner.bound = root100Candidates11 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
theorem root100Step11 : scan edge (search edge 3) 4 [563] 563 root100Tail11 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail12 winner := by
  rw [root100Tail11, scan, if_neg (by decide), root100Candidates11_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail12
    (if 3 ≤ root100Candidates11.length then search edge 3 [563, 2909] root100Candidates11 3472 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates12 : List Nat := [5237, 11681, 21569]
set_option maxRecDepth 8192 in
theorem root100Candidates12_checked : candidates edge 4 563 3119 root100Tail13 winner.bound = root100Candidates12 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root100Child12 : search edge 3 [563, 3119] root100Candidates12 3682 winner = winner := by decide
theorem root100Step12 : scan edge (search edge 3) 4 [563] 563 root100Tail12 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail13 winner := by
  rw [root100Tail12, scan, if_neg (by decide), root100Candidates12_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail13
    (if 3 ≤ root100Candidates12.length then search edge 3 [563, 3119] root100Candidates12 3682 winner else winner) = _
  rw [if_pos (by decide), root100Child12]
def root100Candidates13 : List Nat := [7019]
set_option maxRecDepth 8192 in
theorem root100Candidates13_checked : candidates edge 4 563 3821 root100Tail14 winner.bound = root100Candidates13 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root100Step13 : scan edge (search edge 3) 4 [563] 563 root100Tail13 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail14 winner := by
  rw [root100Tail13, scan, if_neg (by decide), root100Candidates13_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail14
    (if 3 ≤ root100Candidates13.length then search edge 3 [563, 3821] root100Candidates13 4384 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates14 : List Nat := [5717]
set_option maxRecDepth 8192 in
theorem root100Candidates14_checked : candidates edge 4 563 4493 root100Tail15 winner.bound = root100Candidates14 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root100Step14 : scan edge (search edge 3) 4 [563] 563 root100Tail14 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail15 winner := by
  rw [root100Tail14, scan, if_neg (by decide), root100Candidates14_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail15
    (if 3 ≤ root100Candidates14.length then search edge 3 [563, 4493] root100Candidates14 5056 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates15 : List Nat := [17189]
set_option maxRecDepth 8192 in
theorem root100Candidates15_checked : candidates edge 4 563 4799 root100Tail16 winner.bound = root100Candidates15 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root100Step15 : scan edge (search edge 3) 4 [563] 563 root100Tail15 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail16 winner := by
  rw [root100Tail15, scan, if_neg (by decide), root100Candidates15_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail16
    (if 3 ≤ root100Candidates15.length then search edge 3 [563, 4799] root100Candidates15 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates16 : List Nat := [7043, 8537]
set_option maxRecDepth 8192 in
theorem root100Candidates16_checked : candidates edge 4 563 5237 root100Tail17 winner.bound = root100Candidates16 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root100Step16 : scan edge (search edge 3) 4 [563] 563 root100Tail16 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail17 winner := by
  rw [root100Tail16, scan, if_neg (by decide), root100Candidates16_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail17
    (if 3 ≤ root100Candidates16.length then search edge 3 [563, 5237] root100Candidates16 5800 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root100Candidates17_checked : candidates edge 4 563 5261 root100Tail18 winner.bound = root100Candidates17 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root100Step17 : scan edge (search edge 3) 4 [563] 563 root100Tail17 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail18 winner := by
  rw [root100Tail17, scan, if_neg (by decide), root100Candidates17_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail18
    (if 3 ≤ root100Candidates17.length then search edge 3 [563, 5261] root100Candidates17 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root100Candidates18_checked : candidates edge 4 563 5297 root100Tail19 winner.bound = root100Candidates18 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root100Step18 : scan edge (search edge 3) 4 [563] 563 root100Tail18 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail19 winner := by
  rw [root100Tail18, scan, if_neg (by decide), root100Candidates18_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail19
    (if 3 ≤ root100Candidates18.length then search edge 3 [563, 5297] root100Candidates18 5860 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates19 : List Nat := [7481]
set_option maxRecDepth 8192 in
theorem root100Candidates19_checked : candidates edge 4 563 5501 root100Tail20 winner.bound = root100Candidates19 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root100Step19 : scan edge (search edge 3) 4 [563] 563 root100Tail19 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail20 winner := by
  rw [root100Tail19, scan, if_neg (by decide), root100Candidates19_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail20
    (if 3 ≤ root100Candidates19.length then search edge 3 [563, 5501] root100Candidates19 6064 winner else winner) = _
  rw [if_neg (by decide)]
def root100Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root100Candidates20_checked : candidates edge 4 563 5717 root100Tail21 winner.bound = root100Candidates20 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root100Step20 : scan edge (search edge 3) 4 [563] 563 root100Tail20 winner =
    scan edge (search edge 3) 4 [563] 563 root100Tail21 winner := by
  rw [root100Tail20, scan, if_neg (by decide), root100Candidates20_checked]
  change scan edge (search edge 3) 4 [563] 563 root100Tail21
    (if 3 ≤ root100Candidates20.length then search edge 3 [563, 5717] root100Candidates20 6280 winner else winner) = _
  rw [if_neg (by decide)]
theorem root100Scan21 : scan edge (search edge 3) 4 [563] 563 root100Tail21 winner = winner := by rw [root100Tail21, scan, if_pos (by decide)]
theorem root100Scan20 : scan edge (search edge 3) 4 [563] 563 root100Tail20 winner = winner :=
  (root100Step20).trans root100Scan21
theorem root100Scan19 : scan edge (search edge 3) 4 [563] 563 root100Tail19 winner = winner :=
  (root100Step19).trans root100Scan20
theorem root100Scan18 : scan edge (search edge 3) 4 [563] 563 root100Tail18 winner = winner :=
  (root100Step18).trans root100Scan19
theorem root100Scan17 : scan edge (search edge 3) 4 [563] 563 root100Tail17 winner = winner :=
  (root100Step17).trans root100Scan18
theorem root100Scan16 : scan edge (search edge 3) 4 [563] 563 root100Tail16 winner = winner :=
  (root100Step16).trans root100Scan17
theorem root100Scan15 : scan edge (search edge 3) 4 [563] 563 root100Tail15 winner = winner :=
  (root100Step15).trans root100Scan16
theorem root100Scan14 : scan edge (search edge 3) 4 [563] 563 root100Tail14 winner = winner :=
  (root100Step14).trans root100Scan15
theorem root100Scan13 : scan edge (search edge 3) 4 [563] 563 root100Tail13 winner = winner :=
  (root100Step13).trans root100Scan14
theorem root100Scan12 : scan edge (search edge 3) 4 [563] 563 root100Tail12 winner = winner :=
  (root100Step12).trans root100Scan13
theorem root100Scan11 : scan edge (search edge 3) 4 [563] 563 root100Tail11 winner = winner :=
  (root100Step11).trans root100Scan12
theorem root100Scan10 : scan edge (search edge 3) 4 [563] 563 root100Tail10 winner = winner :=
  (root100Step10).trans root100Scan11
theorem root100Scan9 : scan edge (search edge 3) 4 [563] 563 root100Tail9 winner = winner :=
  (root100Step9).trans root100Scan10
theorem root100Scan8 : scan edge (search edge 3) 4 [563] 563 root100Tail8 winner = winner :=
  (root100Step8).trans root100Scan9
theorem root100Scan7 : scan edge (search edge 3) 4 [563] 563 root100Tail7 winner = winner :=
  (root100Step7).trans root100Scan8
theorem root100Scan6 : scan edge (search edge 3) 4 [563] 563 root100Tail6 winner = winner :=
  (root100Step6).trans root100Scan7
theorem root100Scan5 : scan edge (search edge 3) 4 [563] 563 root100Tail5 winner = winner :=
  (root100Step5).trans root100Scan6
theorem root100Scan4 : scan edge (search edge 3) 4 [563] 563 root100Tail4 winner = winner :=
  (root100Step4).trans root100Scan5
theorem root100Scan3 : scan edge (search edge 3) 4 [563] 563 root100Tail3 winner = winner :=
  (root100Step3).trans root100Scan4
theorem root100Scan2 : scan edge (search edge 3) 4 [563] 563 root100Tail2 winner = winner :=
  (root100Step2).trans root100Scan3
theorem root100Scan1 : scan edge (search edge 3) 4 [563] 563 root100Tail1 winner = winner :=
  (root100Step1).trans root100Scan2
theorem root100Scan0 : scan edge (search edge 3) 4 [563] 563 root100Tail0 winner = winner :=
  (root100Step0).trans root100Scan1
set_option maxRecDepth 4096 in
theorem root100Branch : search edge 4 [563] root100Candidates 563 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [563] 563 root100Tail0 winner = winner
  exact root100Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
