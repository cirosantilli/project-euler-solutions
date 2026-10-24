import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root95Candidates : List Nat := Data.row521
set_option maxRecDepth 8192 in
theorem root95Candidates_checked : candidates edge 5 0 521 tail96 winner.bound = root95Candidates := by
  apply candidates_eq_row edge 5 0 521 tail96 root95Candidates winner.bound tail96_ordered
  · decide
  · decide
  · funext q
    rfl
def root95Tail69 : List Nat := []
def root95Tail68 : List Nat := 25439 :: root95Tail69
def root95Tail67 : List Nat := 25121 :: root95Tail68
def root95Tail66 : List Nat := 24071 :: root95Tail67
def root95Tail65 : List Nat := 23399 :: root95Tail66
def root95Tail64 : List Nat := 23081 :: root95Tail65
def root95Tail63 : List Nat := 22817 :: root95Tail64
def root95Tail62 : List Nat := 22709 :: root95Tail63
def root95Tail61 : List Nat := 22481 :: root95Tail62
def root95Tail60 : List Nat := 21851 :: root95Tail61
def root95Tail59 : List Nat := 21467 :: root95Tail60
def root95Tail58 : List Nat := 21317 :: root95Tail59
def root95Tail57 : List Nat := 21221 :: root95Tail58
def root95Tail56 : List Nat := 20771 :: root95Tail57
def root95Tail55 : List Nat := 19997 :: root95Tail56
def root95Tail54 : List Nat := 19853 :: root95Tail55
def root95Tail53 : List Nat := 19073 :: root95Tail54
def root95Tail52 : List Nat := 18947 :: root95Tail53
def root95Tail51 : List Nat := 17981 :: root95Tail52
def root95Tail50 : List Nat := 17159 :: root95Tail51
def root95Tail49 : List Nat := 17099 :: root95Tail50
def root95Tail48 : List Nat := 16931 :: root95Tail49
def root95Tail47 : List Nat := 16319 :: root95Tail48
def root95Tail46 : List Nat := 16073 :: root95Tail47
def root95Tail45 : List Nat := 15881 :: root95Tail46
def root95Tail44 : List Nat := 15737 :: root95Tail45
def root95Tail43 : List Nat := 14897 :: root95Tail44
def root95Tail42 : List Nat := 14747 :: root95Tail43
def root95Tail41 : List Nat := 13913 :: root95Tail42
def root95Tail40 : List Nat := 13907 :: root95Tail41
def root95Tail39 : List Nat := 13487 :: root95Tail40
def root95Tail38 : List Nat := 13421 :: root95Tail39
def root95Tail37 : List Nat := 12821 :: root95Tail38
def root95Tail36 : List Nat := 12713 :: root95Tail37
def root95Tail35 : List Nat := 12161 :: root95Tail36
def root95Tail34 : List Nat := 11177 :: root95Tail35
def root95Tail33 : List Nat := 10337 :: root95Tail34
def root95Tail32 : List Nat := 10313 :: root95Tail33
def root95Tail31 : List Nat := 10133 :: root95Tail32
def root95Tail30 : List Nat := 9941 :: root95Tail31
def root95Tail29 : List Nat := 9791 :: root95Tail30
def root95Tail28 : List Nat := 7907 :: root95Tail29
def root95Tail27 : List Nat := 7673 :: root95Tail28
def root95Tail26 : List Nat := 7607 :: root95Tail27
def root95Tail25 : List Nat := 6521 :: root95Tail26
def root95Tail24 : List Nat := 6353 :: root95Tail25
def root95Tail23 : List Nat := 6311 :: root95Tail24
def root95Tail22 : List Nat := 6269 :: root95Tail23
def root95Tail21 : List Nat := 6053 :: root95Tail22
def root95Tail20 : List Nat := 5417 :: root95Tail21
def root95Tail19 : List Nat := 5381 :: root95Tail20
def root95Tail18 : List Nat := 4691 :: root95Tail19
def root95Tail17 : List Nat := 4337 :: root95Tail18
def root95Tail16 : List Nat := 3539 :: root95Tail17
def root95Tail15 : List Nat := 3389 :: root95Tail16
def root95Tail14 : List Nat := 3137 :: root95Tail15
def root95Tail13 : List Nat := 2777 :: root95Tail14
def root95Tail12 : List Nat := 2477 :: root95Tail13
def root95Tail11 : List Nat := 2447 :: root95Tail12
def root95Tail10 : List Nat := 2381 :: root95Tail11
def root95Tail9 : List Nat := 1721 :: root95Tail10
def root95Tail8 : List Nat := 1637 :: root95Tail9
def root95Tail7 : List Nat := 1523 :: root95Tail8
def root95Tail6 : List Nat := 1373 :: root95Tail7
def root95Tail5 : List Nat := 1229 :: root95Tail6
def root95Tail4 : List Nat := 1091 :: root95Tail5
def root95Tail3 : List Nat := 809 :: root95Tail4
def root95Tail2 : List Nat := 659 :: root95Tail3
def root95Tail1 : List Nat := 641 :: root95Tail2
def root95Tail0 : List Nat := 557 :: root95Tail1
def root95Candidates0 : List Nat := [1721, 3389, 16319]
set_option maxRecDepth 8192 in
theorem root95Candidates0_checked : candidates edge 4 521 557 root95Tail1 winner.bound = root95Candidates0 := by
  unfold candidates
  have row : edge 557 = (SortedNatTree.contains FastRows.row557) := by
    change Data.row557.contains = (SortedNatTree.contains FastRows.row557)
    exact FastRows.row557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child0 : search edge 3 [521, 557] root95Candidates0 1078 winner = winner := by decide
theorem root95Step0 : scan edge (search edge 3) 4 [521] 521 root95Tail0 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail1 winner := by
  rw [root95Tail0, scan, if_neg (by decide), root95Candidates0_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail1
    (if 3 ≤ root95Candidates0.length then search edge 3 [521, 557] root95Candidates0 1078 winner else winner) = _
  rw [if_pos (by decide), root95Child0]
def root95Candidates1 : List Nat := [23399]
set_option maxRecDepth 8192 in
theorem root95Candidates1_checked : candidates edge 4 521 641 root95Tail2 winner.bound = root95Candidates1 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
theorem root95Step1 : scan edge (search edge 3) 4 [521] 521 root95Tail1 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail2 winner := by
  rw [root95Tail1, scan, if_neg (by decide), root95Candidates1_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail2
    (if 3 ≤ root95Candidates1.length then search edge 3 [521, 641] root95Candidates1 1162 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root95Candidates2_checked : candidates edge 4 521 659 root95Tail3 winner.bound = root95Candidates2 := by
  unfold candidates
  have row : edge 659 = (SortedNatTree.contains FastRows.row659) := by
    change Data.row659.contains = (SortedNatTree.contains FastRows.row659)
    exact FastRows.row659_checked.symm
  rw [row]
  decide
theorem root95Step2 : scan edge (search edge 3) 4 [521] 521 root95Tail2 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail3 winner := by
  rw [root95Tail2, scan, if_neg (by decide), root95Candidates2_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail3
    (if 3 ≤ root95Candidates2.length then search edge 3 [521, 659] root95Candidates2 1180 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates3 : List Nat := [1091]
set_option maxRecDepth 8192 in
theorem root95Candidates3_checked : candidates edge 4 521 809 root95Tail4 winner.bound = root95Candidates3 := by
  unfold candidates
  have row : edge 809 = (SortedNatTree.contains FastRows.row809) := by
    change Data.row809.contains = (SortedNatTree.contains FastRows.row809)
    exact FastRows.row809_checked.symm
  rw [row]
  decide
theorem root95Step3 : scan edge (search edge 3) 4 [521] 521 root95Tail3 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail4 winner := by
  rw [root95Tail3, scan, if_neg (by decide), root95Candidates3_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail4
    (if 3 ≤ root95Candidates3.length then search edge 3 [521, 809] root95Candidates3 1330 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates4 : List Nat := [9791]
set_option maxRecDepth 8192 in
theorem root95Candidates4_checked : candidates edge 4 521 1091 root95Tail5 winner.bound = root95Candidates4 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
theorem root95Step4 : scan edge (search edge 3) 4 [521] 521 root95Tail4 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail5 winner := by
  rw [root95Tail4, scan, if_neg (by decide), root95Candidates4_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail5
    (if 3 ≤ root95Candidates4.length then search edge 3 [521, 1091] root95Candidates4 1612 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates5 : List Nat := [3389, 6353, 7907, 18947]
set_option maxRecDepth 8192 in
theorem root95Candidates5_checked : candidates edge 4 521 1229 root95Tail6 winner.bound = root95Candidates5 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child5 : search edge 3 [521, 1229] root95Candidates5 1750 winner = winner := by decide
theorem root95Step5 : scan edge (search edge 3) 4 [521] 521 root95Tail5 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail6 winner := by
  rw [root95Tail5, scan, if_neg (by decide), root95Candidates5_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail6
    (if 3 ≤ root95Candidates5.length then search edge 3 [521, 1229] root95Candidates5 1750 winner else winner) = _
  rw [if_pos (by decide), root95Child5]
def root95Candidates6 : List Nat := [1637, 3539, 7907, 9941]
set_option maxRecDepth 8192 in
theorem root95Candidates6_checked : candidates edge 4 521 1373 root95Tail7 winner.bound = root95Candidates6 := by
  unfold candidates
  have row : edge 1373 = (SortedNatTree.contains FastRows.row1373) := by
    change Data.row1373.contains = (SortedNatTree.contains FastRows.row1373)
    exact FastRows.row1373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child6 : search edge 3 [521, 1373] root95Candidates6 1894 winner = winner := by decide
theorem root95Step6 : scan edge (search edge 3) 4 [521] 521 root95Tail6 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail7 winner := by
  rw [root95Tail6, scan, if_neg (by decide), root95Candidates6_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail7
    (if 3 ≤ root95Candidates6.length then search edge 3 [521, 1373] root95Candidates6 1894 winner else winner) = _
  rw [if_pos (by decide), root95Child6]
def root95Candidates7 : List Nat := [2447, 6311, 7907, 21317]
set_option maxRecDepth 8192 in
theorem root95Candidates7_checked : candidates edge 4 521 1523 root95Tail8 winner.bound = root95Candidates7 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child7 : search edge 3 [521, 1523] root95Candidates7 2044 winner = winner := by decide
theorem root95Step7 : scan edge (search edge 3) 4 [521] 521 root95Tail7 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail8 winner := by
  rw [root95Tail7, scan, if_neg (by decide), root95Candidates7_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail8
    (if 3 ≤ root95Candidates7.length then search edge 3 [521, 1523] root95Candidates7 2044 winner else winner) = _
  rw [if_pos (by decide), root95Child7]
def root95Candidates8 : List Nat := [6053, 7607, 21851]
set_option maxRecDepth 8192 in
theorem root95Candidates8_checked : candidates edge 4 521 1637 root95Tail9 winner.bound = root95Candidates8 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child8 : search edge 3 [521, 1637] root95Candidates8 2158 winner = winner := by decide
theorem root95Step8 : scan edge (search edge 3) 4 [521] 521 root95Tail8 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail9 winner := by
  rw [root95Tail8, scan, if_neg (by decide), root95Candidates8_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail9
    (if 3 ≤ root95Candidates8.length then search edge 3 [521, 1637] root95Candidates8 2158 winner else winner) = _
  rw [if_pos (by decide), root95Child8]
def root95Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root95Candidates9_checked : candidates edge 4 521 1721 root95Tail10 winner.bound = root95Candidates9 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root95Step9 : scan edge (search edge 3) 4 [521] 521 root95Tail9 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail10 winner := by
  rw [root95Tail9, scan, if_neg (by decide), root95Candidates9_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail10
    (if 3 ≤ root95Candidates9.length then search edge 3 [521, 1721] root95Candidates9 2242 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates10 : List Nat := [2447, 3539]
set_option maxRecDepth 8192 in
theorem root95Candidates10_checked : candidates edge 4 521 2381 root95Tail11 winner.bound = root95Candidates10 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
theorem root95Step10 : scan edge (search edge 3) 4 [521] 521 root95Tail10 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail11 winner := by
  rw [root95Tail10, scan, if_neg (by decide), root95Candidates10_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail11
    (if 3 ≤ root95Candidates10.length then search edge 3 [521, 2381] root95Candidates10 2902 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates11 : List Nat := [7907]
set_option maxRecDepth 8192 in
theorem root95Candidates11_checked : candidates edge 4 521 2447 root95Tail12 winner.bound = root95Candidates11 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root95Step11 : scan edge (search edge 3) 4 [521] 521 root95Tail11 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail12 winner := by
  rw [root95Tail11, scan, if_neg (by decide), root95Candidates11_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail12
    (if 3 ≤ root95Candidates11.length then search edge 3 [521, 2447] root95Candidates11 2968 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates12 : List Nat := [5381, 12821, 15737]
set_option maxRecDepth 8192 in
theorem root95Candidates12_checked : candidates edge 4 521 2477 root95Tail13 winner.bound = root95Candidates12 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child12 : search edge 3 [521, 2477] root95Candidates12 2998 winner = winner := by decide
theorem root95Step12 : scan edge (search edge 3) 4 [521] 521 root95Tail12 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail13 winner := by
  rw [root95Tail12, scan, if_neg (by decide), root95Candidates12_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail13
    (if 3 ≤ root95Candidates12.length then search edge 3 [521, 2477] root95Candidates12 2998 winner else winner) = _
  rw [if_pos (by decide), root95Child12]
def root95Candidates13 : List Nat := [19073]
set_option maxRecDepth 8192 in
theorem root95Candidates13_checked : candidates edge 4 521 2777 root95Tail14 winner.bound = root95Candidates13 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root95Step13 : scan edge (search edge 3) 4 [521] 521 root95Tail13 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail14 winner := by
  rw [root95Tail13, scan, if_neg (by decide), root95Candidates13_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail14
    (if 3 ≤ root95Candidates13.length then search edge 3 [521, 2777] root95Candidates13 3298 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates14 : List Nat := [6311]
set_option maxRecDepth 8192 in
theorem root95Candidates14_checked : candidates edge 4 521 3137 root95Tail15 winner.bound = root95Candidates14 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root95Step14 : scan edge (search edge 3) 4 [521] 521 root95Tail14 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail15 winner := by
  rw [root95Tail14, scan, if_neg (by decide), root95Candidates14_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail15
    (if 3 ≤ root95Candidates14.length then search edge 3 [521, 3137] root95Candidates14 3658 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root95Candidates15_checked : candidates edge 4 521 3389 root95Tail16 winner.bound = root95Candidates15 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
theorem root95Step15 : scan edge (search edge 3) 4 [521] 521 root95Tail15 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail16 winner := by
  rw [root95Tail15, scan, if_neg (by decide), root95Candidates15_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail16
    (if 3 ≤ root95Candidates15.length then search edge 3 [521, 3389] root95Candidates15 3910 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates16 : List Nat := [6521]
set_option maxRecDepth 8192 in
theorem root95Candidates16_checked : candidates edge 4 521 3539 root95Tail17 winner.bound = root95Candidates16 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root95Step16 : scan edge (search edge 3) 4 [521] 521 root95Tail16 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail17 winner := by
  rw [root95Tail16, scan, if_neg (by decide), root95Candidates16_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail17
    (if 3 ≤ root95Candidates16.length then search edge 3 [521, 3539] root95Candidates16 4060 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates17 : List Nat := [17981]
set_option maxRecDepth 8192 in
theorem root95Candidates17_checked : candidates edge 4 521 4337 root95Tail18 winner.bound = root95Candidates17 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root95Step17 : scan edge (search edge 3) 4 [521] 521 root95Tail17 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail18 winner := by
  rw [root95Tail17, scan, if_neg (by decide), root95Candidates17_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail18
    (if 3 ≤ root95Candidates17.length then search edge 3 [521, 4337] root95Candidates17 4858 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates18 : List Nat := [6521]
set_option maxRecDepth 8192 in
theorem root95Candidates18_checked : candidates edge 4 521 4691 root95Tail19 winner.bound = root95Candidates18 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root95Step18 : scan edge (search edge 3) 4 [521] 521 root95Tail18 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail19 winner := by
  rw [root95Tail18, scan, if_neg (by decide), root95Candidates18_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail19
    (if 3 ≤ root95Candidates18.length then search edge 3 [521, 4691] root95Candidates18 5212 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates19 : List Nat := [9791, 14747, 19073, 19997]
set_option maxRecDepth 8192 in
theorem root95Candidates19_checked : candidates edge 4 521 5381 root95Tail20 winner.bound = root95Candidates19 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root95Child19 : search edge 3 [521, 5381] root95Candidates19 5902 winner = winner := by decide
theorem root95Step19 : scan edge (search edge 3) 4 [521] 521 root95Tail19 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail20 winner := by
  rw [root95Tail19, scan, if_neg (by decide), root95Candidates19_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail20
    (if 3 ≤ root95Candidates19.length then search edge 3 [521, 5381] root95Candidates19 5902 winner else winner) = _
  rw [if_pos (by decide), root95Child19]
def root95Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root95Candidates20_checked : candidates edge 4 521 5417 root95Tail21 winner.bound = root95Candidates20 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root95Step20 : scan edge (search edge 3) 4 [521] 521 root95Tail20 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail21 winner := by
  rw [root95Tail20, scan, if_neg (by decide), root95Candidates20_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail21
    (if 3 ≤ root95Candidates20.length then search edge 3 [521, 5417] root95Candidates20 5938 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates21 : List Nat := [13421, 19073]
set_option maxRecDepth 8192 in
theorem root95Candidates21_checked : candidates edge 4 521 6053 root95Tail22 winner.bound = root95Candidates21 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root95Step21 : scan edge (search edge 3) 4 [521] 521 root95Tail21 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail22 winner := by
  rw [root95Tail21, scan, if_neg (by decide), root95Candidates21_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail22
    (if 3 ≤ root95Candidates21.length then search edge 3 [521, 6053] root95Candidates21 6574 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates22 : List Nat := [12161]
set_option maxRecDepth 8192 in
theorem root95Candidates22_checked : candidates edge 4 521 6269 root95Tail23 winner.bound = root95Candidates22 := by
  unfold candidates
  have row : edge 6269 = (SortedNatTree.contains FastRows.row6269) := by
    change Data.row6269.contains = (SortedNatTree.contains FastRows.row6269)
    exact FastRows.row6269_checked.symm
  rw [row]
  decide
theorem root95Step22 : scan edge (search edge 3) 4 [521] 521 root95Tail22 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail23 winner := by
  rw [root95Tail22, scan, if_neg (by decide), root95Candidates22_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail23
    (if 3 ≤ root95Candidates22.length then search edge 3 [521, 6269] root95Candidates22 6790 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates23 : List Nat := [9941]
set_option maxRecDepth 8192 in
theorem root95Candidates23_checked : candidates edge 4 521 6311 root95Tail24 winner.bound = root95Candidates23 := by
  unfold candidates
  have row : edge 6311 = (SortedNatTree.contains FastRows.row6311) := by
    change Data.row6311.contains = (SortedNatTree.contains FastRows.row6311)
    exact FastRows.row6311_checked.symm
  rw [row]
  decide
theorem root95Step23 : scan edge (search edge 3) 4 [521] 521 root95Tail23 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail24 winner := by
  rw [root95Tail23, scan, if_neg (by decide), root95Candidates23_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail24
    (if 3 ≤ root95Candidates23.length then search edge 3 [521, 6311] root95Candidates23 6832 winner else winner) = _
  rw [if_neg (by decide)]
def root95Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root95Candidates24_checked : candidates edge 4 521 6353 root95Tail25 winner.bound = root95Candidates24 := by
  unfold candidates
  have row : edge 6353 = (SortedNatTree.contains FastRows.row6353) := by
    change Data.row6353.contains = (SortedNatTree.contains FastRows.row6353)
    exact FastRows.row6353_checked.symm
  rw [row]
  decide
theorem root95Step24 : scan edge (search edge 3) 4 [521] 521 root95Tail24 winner =
    scan edge (search edge 3) 4 [521] 521 root95Tail25 winner := by
  rw [root95Tail24, scan, if_neg (by decide), root95Candidates24_checked]
  change scan edge (search edge 3) 4 [521] 521 root95Tail25
    (if 3 ≤ root95Candidates24.length then search edge 3 [521, 6353] root95Candidates24 6874 winner else winner) = _
  rw [if_neg (by decide)]
theorem root95Scan25 : scan edge (search edge 3) 4 [521] 521 root95Tail25 winner = winner := by rw [root95Tail25, scan, if_pos (by decide)]
theorem root95Scan24 : scan edge (search edge 3) 4 [521] 521 root95Tail24 winner = winner :=
  (root95Step24).trans root95Scan25
theorem root95Scan23 : scan edge (search edge 3) 4 [521] 521 root95Tail23 winner = winner :=
  (root95Step23).trans root95Scan24
theorem root95Scan22 : scan edge (search edge 3) 4 [521] 521 root95Tail22 winner = winner :=
  (root95Step22).trans root95Scan23
theorem root95Scan21 : scan edge (search edge 3) 4 [521] 521 root95Tail21 winner = winner :=
  (root95Step21).trans root95Scan22
theorem root95Scan20 : scan edge (search edge 3) 4 [521] 521 root95Tail20 winner = winner :=
  (root95Step20).trans root95Scan21
theorem root95Scan19 : scan edge (search edge 3) 4 [521] 521 root95Tail19 winner = winner :=
  (root95Step19).trans root95Scan20
theorem root95Scan18 : scan edge (search edge 3) 4 [521] 521 root95Tail18 winner = winner :=
  (root95Step18).trans root95Scan19
theorem root95Scan17 : scan edge (search edge 3) 4 [521] 521 root95Tail17 winner = winner :=
  (root95Step17).trans root95Scan18
theorem root95Scan16 : scan edge (search edge 3) 4 [521] 521 root95Tail16 winner = winner :=
  (root95Step16).trans root95Scan17
theorem root95Scan15 : scan edge (search edge 3) 4 [521] 521 root95Tail15 winner = winner :=
  (root95Step15).trans root95Scan16
theorem root95Scan14 : scan edge (search edge 3) 4 [521] 521 root95Tail14 winner = winner :=
  (root95Step14).trans root95Scan15
theorem root95Scan13 : scan edge (search edge 3) 4 [521] 521 root95Tail13 winner = winner :=
  (root95Step13).trans root95Scan14
theorem root95Scan12 : scan edge (search edge 3) 4 [521] 521 root95Tail12 winner = winner :=
  (root95Step12).trans root95Scan13
theorem root95Scan11 : scan edge (search edge 3) 4 [521] 521 root95Tail11 winner = winner :=
  (root95Step11).trans root95Scan12
theorem root95Scan10 : scan edge (search edge 3) 4 [521] 521 root95Tail10 winner = winner :=
  (root95Step10).trans root95Scan11
theorem root95Scan9 : scan edge (search edge 3) 4 [521] 521 root95Tail9 winner = winner :=
  (root95Step9).trans root95Scan10
theorem root95Scan8 : scan edge (search edge 3) 4 [521] 521 root95Tail8 winner = winner :=
  (root95Step8).trans root95Scan9
theorem root95Scan7 : scan edge (search edge 3) 4 [521] 521 root95Tail7 winner = winner :=
  (root95Step7).trans root95Scan8
theorem root95Scan6 : scan edge (search edge 3) 4 [521] 521 root95Tail6 winner = winner :=
  (root95Step6).trans root95Scan7
theorem root95Scan5 : scan edge (search edge 3) 4 [521] 521 root95Tail5 winner = winner :=
  (root95Step5).trans root95Scan6
theorem root95Scan4 : scan edge (search edge 3) 4 [521] 521 root95Tail4 winner = winner :=
  (root95Step4).trans root95Scan5
theorem root95Scan3 : scan edge (search edge 3) 4 [521] 521 root95Tail3 winner = winner :=
  (root95Step3).trans root95Scan4
theorem root95Scan2 : scan edge (search edge 3) 4 [521] 521 root95Tail2 winner = winner :=
  (root95Step2).trans root95Scan3
theorem root95Scan1 : scan edge (search edge 3) 4 [521] 521 root95Tail1 winner = winner :=
  (root95Step1).trans root95Scan2
theorem root95Scan0 : scan edge (search edge 3) 4 [521] 521 root95Tail0 winner = winner :=
  (root95Step0).trans root95Scan1
set_option maxRecDepth 4096 in
theorem root95Branch : search edge 4 [521] root95Candidates 521 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [521] 521 root95Tail0 winner = winner
  exact root95Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
