import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root82Candidates : List Nat := Data.row439
set_option maxRecDepth 8192 in
theorem root82Candidates_checked : candidates edge 5 0 439 tail83 winner.bound = root82Candidates := by
  apply candidates_eq_row edge 5 0 439 tail83 root82Candidates winner.bound tail83_ordered
  · decide
  · decide
  · funext q
    rfl
def root82Tail67 : List Nat := []
def root82Tail66 : List Nat := 25237 :: root82Tail67
def root82Tail65 : List Nat := 24781 :: root82Tail66
def root82Tail64 : List Nat := 24469 :: root82Tail65
def root82Tail63 : List Nat := 24373 :: root82Tail64
def root82Tail62 : List Nat := 24109 :: root82Tail63
def root82Tail61 : List Nat := 23827 :: root82Tail62
def root82Tail60 : List Nat := 23071 :: root82Tail61
def root82Tail59 : List Nat := 22717 :: root82Tail60
def root82Tail58 : List Nat := 22639 :: root82Tail59
def root82Tail57 : List Nat := 22621 :: root82Tail58
def root82Tail56 : List Nat := 22273 :: root82Tail57
def root82Tail55 : List Nat := 21991 :: root82Tail56
def root82Tail54 : List Nat := 21799 :: root82Tail55
def root82Tail53 : List Nat := 19441 :: root82Tail54
def root82Tail52 : List Nat := 18451 :: root82Tail53
def root82Tail51 : List Nat := 18211 :: root82Tail52
def root82Tail50 : List Nat := 18181 :: root82Tail51
def root82Tail49 : List Nat := 18097 :: root82Tail50
def root82Tail48 : List Nat := 17539 :: root82Tail49
def root82Tail47 : List Nat := 17389 :: root82Tail48
def root82Tail46 : List Nat := 17257 :: root82Tail47
def root82Tail45 : List Nat := 16993 :: root82Tail46
def root82Tail44 : List Nat := 16963 :: root82Tail45
def root82Tail43 : List Nat := 15679 :: root82Tail44
def root82Tail42 : List Nat := 15271 :: root82Tail43
def root82Tail41 : List Nat := 15031 :: root82Tail42
def root82Tail40 : List Nat := 14389 :: root82Tail41
def root82Tail39 : List Nat := 14293 :: root82Tail40
def root82Tail38 : List Nat := 13789 :: root82Tail39
def root82Tail37 : List Nat := 13687 :: root82Tail38
def root82Tail36 : List Nat := 13063 :: root82Tail37
def root82Tail35 : List Nat := 12301 :: root82Tail36
def root82Tail34 : List Nat := 11971 :: root82Tail35
def root82Tail33 : List Nat := 11953 :: root82Tail34
def root82Tail32 : List Nat := 11689 :: root82Tail33
def root82Tail31 : List Nat := 10903 :: root82Tail32
def root82Tail30 : List Nat := 10597 :: root82Tail31
def root82Tail29 : List Nat := 10459 :: root82Tail30
def root82Tail28 : List Nat := 10429 :: root82Tail29
def root82Tail27 : List Nat := 10357 :: root82Tail28
def root82Tail26 : List Nat := 9067 :: root82Tail27
def root82Tail25 : List Nat := 8011 :: root82Tail26
def root82Tail24 : List Nat := 6481 :: root82Tail25
def root82Tail23 : List Nat := 5827 :: root82Tail24
def root82Tail22 : List Nat := 5701 :: root82Tail23
def root82Tail21 : List Nat := 5557 :: root82Tail22
def root82Tail20 : List Nat := 5527 :: root82Tail21
def root82Tail19 : List Nat := 4783 :: root82Tail20
def root82Tail18 : List Nat := 4447 :: root82Tail19
def root82Tail17 : List Nat := 4441 :: root82Tail18
def root82Tail16 : List Nat := 4021 :: root82Tail17
def root82Tail15 : List Nat := 3511 :: root82Tail16
def root82Tail14 : List Nat := 2887 :: root82Tail15
def root82Tail13 : List Nat := 2797 :: root82Tail14
def root82Tail12 : List Nat := 2593 :: root82Tail13
def root82Tail11 : List Nat := 2503 :: root82Tail12
def root82Tail10 : List Nat := 2383 :: root82Tail11
def root82Tail9 : List Nat := 1579 :: root82Tail10
def root82Tail8 : List Nat := 1549 :: root82Tail9
def root82Tail7 : List Nat := 1483 :: root82Tail8
def root82Tail6 : List Nat := 1399 :: root82Tail7
def root82Tail5 : List Nat := 853 :: root82Tail6
def root82Tail4 : List Nat := 787 :: root82Tail5
def root82Tail3 : List Nat := 661 :: root82Tail4
def root82Tail2 : List Nat := 613 :: root82Tail3
def root82Tail1 : List Nat := 601 :: root82Tail2
def root82Tail0 : List Nat := 541 :: root82Tail1
def root82Candidates0 : List Nat := [661, 2887, 24373, 24781]
set_option maxRecDepth 8192 in
theorem root82Candidates0_checked : candidates edge 4 439 541 root82Tail1 winner.bound = root82Candidates0 := by
  unfold candidates
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root82Child0 : search edge 3 [439, 541] root82Candidates0 980 winner = winner := by decide
theorem root82Step0 : scan edge (search edge 3) 4 [439] 439 root82Tail0 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail1 winner := by
  rw [root82Tail0, scan, if_neg (by decide), root82Candidates0_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail1
    (if 3 ≤ root82Candidates0.length then search edge 3 [439, 541] root82Candidates0 980 winner else winner) = _
  rw [if_pos (by decide), root82Child0]
def root82Candidates1 : List Nat := [18181]
set_option maxRecDepth 8192 in
theorem root82Candidates1_checked : candidates edge 4 439 601 root82Tail2 winner.bound = root82Candidates1 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
theorem root82Step1 : scan edge (search edge 3) 4 [439] 439 root82Tail1 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail2 winner := by
  rw [root82Tail1, scan, if_neg (by decide), root82Candidates1_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail2
    (if 3 ≤ root82Candidates1.length then search edge 3 [439, 601] root82Candidates1 1040 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates2 : List Nat := [661, 4021, 13789, 23071]
set_option maxRecDepth 8192 in
theorem root82Candidates2_checked : candidates edge 4 439 613 root82Tail3 winner.bound = root82Candidates2 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root82Child2 : search edge 3 [439, 613] root82Candidates2 1052 winner = winner := by decide
theorem root82Step2 : scan edge (search edge 3) 4 [439] 439 root82Tail2 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail3 winner := by
  rw [root82Tail2, scan, if_neg (by decide), root82Candidates2_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail3
    (if 3 ≤ root82Candidates2.length then search edge 3 [439, 613] root82Candidates2 1052 winner else winner) = _
  rw [if_pos (by decide), root82Child2]
def root82Candidates3 : List Nat := [19441]
set_option maxRecDepth 8192 in
theorem root82Candidates3_checked : candidates edge 4 439 661 root82Tail4 winner.bound = root82Candidates3 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
theorem root82Step3 : scan edge (search edge 3) 4 [439] 439 root82Tail3 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail4 winner := by
  rw [root82Tail3, scan, if_neg (by decide), root82Candidates3_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail4
    (if 3 ≤ root82Candidates3.length then search edge 3 [439, 661] root82Candidates3 1100 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates4 : List Nat := [1483, 15031]
set_option maxRecDepth 8192 in
theorem root82Candidates4_checked : candidates edge 4 439 787 root82Tail5 winner.bound = root82Candidates4 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
theorem root82Step4 : scan edge (search edge 3) 4 [439] 439 root82Tail4 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail5 winner := by
  rw [root82Tail4, scan, if_neg (by decide), root82Candidates4_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail5
    (if 3 ≤ root82Candidates4.length then search edge 3 [439, 787] root82Candidates4 1226 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates5 : List Nat := [2593, 4447, 13687, 17257]
set_option maxRecDepth 8192 in
theorem root82Candidates5_checked : candidates edge 4 439 853 root82Tail6 winner.bound = root82Candidates5 := by
  unfold candidates
  have row : edge 853 = (SortedNatTree.contains FastRows.row853) := by
    change Data.row853.contains = (SortedNatTree.contains FastRows.row853)
    exact FastRows.row853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root82Child5 : search edge 3 [439, 853] root82Candidates5 1292 winner = winner := by decide
theorem root82Step5 : scan edge (search edge 3) 4 [439] 439 root82Tail5 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail6 winner := by
  rw [root82Tail5, scan, if_neg (by decide), root82Candidates5_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail6
    (if 3 ≤ root82Candidates5.length then search edge 3 [439, 853] root82Candidates5 1292 winner else winner) = _
  rw [if_pos (by decide), root82Child5]
def root82Candidates6 : List Nat := [6481, 23071, 23827]
set_option maxRecDepth 8192 in
theorem root82Candidates6_checked : candidates edge 4 439 1399 root82Tail7 winner.bound = root82Candidates6 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root82Child6 : search edge 3 [439, 1399] root82Candidates6 1838 winner = winner := by decide
theorem root82Step6 : scan edge (search edge 3) 4 [439] 439 root82Tail6 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail7 winner := by
  rw [root82Tail6, scan, if_neg (by decide), root82Candidates6_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail7
    (if 3 ≤ root82Candidates6.length then search edge 3 [439, 1399] root82Candidates6 1838 winner else winner) = _
  rw [if_pos (by decide), root82Child6]
def root82Candidates7 : List Nat := [2593, 10429]
set_option maxRecDepth 8192 in
theorem root82Candidates7_checked : candidates edge 4 439 1483 root82Tail8 winner.bound = root82Candidates7 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
theorem root82Step7 : scan edge (search edge 3) 4 [439] 439 root82Tail7 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail8 winner := by
  rw [root82Tail7, scan, if_neg (by decide), root82Candidates7_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail8
    (if 3 ≤ root82Candidates7.length then search edge 3 [439, 1483] root82Candidates7 1922 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates8 : List Nat := [4783, 15031]
set_option maxRecDepth 8192 in
theorem root82Candidates8_checked : candidates edge 4 439 1549 root82Tail9 winner.bound = root82Candidates8 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
theorem root82Step8 : scan edge (search edge 3) 4 [439] 439 root82Tail8 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail9 winner := by
  rw [root82Tail8, scan, if_neg (by decide), root82Candidates8_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail9
    (if 3 ≤ root82Candidates8.length then search edge 3 [439, 1549] root82Candidates8 1988 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates9 : List Nat := [4783]
set_option maxRecDepth 8192 in
theorem root82Candidates9_checked : candidates edge 4 439 1579 root82Tail10 winner.bound = root82Candidates9 := by
  unfold candidates
  have row : edge 1579 = (SortedNatTree.contains FastRows.row1579) := by
    change Data.row1579.contains = (SortedNatTree.contains FastRows.row1579)
    exact FastRows.row1579_checked.symm
  rw [row]
  decide
theorem root82Step9 : scan edge (search edge 3) 4 [439] 439 root82Tail9 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail10 winner := by
  rw [root82Tail9, scan, if_neg (by decide), root82Candidates9_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail10
    (if 3 ≤ root82Candidates9.length then search edge 3 [439, 1579] root82Candidates9 2018 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates10 : List Nat := [4021, 5527]
set_option maxRecDepth 8192 in
theorem root82Candidates10_checked : candidates edge 4 439 2383 root82Tail11 winner.bound = root82Candidates10 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root82Step10 : scan edge (search edge 3) 4 [439] 439 root82Tail10 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail11 winner := by
  rw [root82Tail10, scan, if_neg (by decide), root82Candidates10_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail11
    (if 3 ≤ root82Candidates10.length then search edge 3 [439, 2383] root82Candidates10 2822 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates11 : List Nat := [4447, 18181, 22717]
set_option maxRecDepth 8192 in
theorem root82Candidates11_checked : candidates edge 4 439 2503 root82Tail12 winner.bound = root82Candidates11 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root82Child11 : search edge 3 [439, 2503] root82Candidates11 2942 winner = winner := by decide
theorem root82Step11 : scan edge (search edge 3) 4 [439] 439 root82Tail11 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail12 winner := by
  rw [root82Tail11, scan, if_neg (by decide), root82Candidates11_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail12
    (if 3 ≤ root82Candidates11.length then search edge 3 [439, 2503] root82Candidates11 2942 winner else winner) = _
  rw [if_pos (by decide), root82Child11]
def root82Candidates12 : List Nat := [2887, 19441]
set_option maxRecDepth 8192 in
theorem root82Candidates12_checked : candidates edge 4 439 2593 root82Tail13 winner.bound = root82Candidates12 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
theorem root82Step12 : scan edge (search edge 3) 4 [439] 439 root82Tail12 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail13 winner := by
  rw [root82Tail12, scan, if_neg (by decide), root82Candidates12_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail13
    (if 3 ≤ root82Candidates12.length then search edge 3 [439, 2593] root82Candidates12 3032 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root82Candidates13_checked : candidates edge 4 439 2797 root82Tail14 winner.bound = root82Candidates13 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root82Step13 : scan edge (search edge 3) 4 [439] 439 root82Tail13 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail14 winner := by
  rw [root82Tail13, scan, if_neg (by decide), root82Candidates13_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail14
    (if 3 ≤ root82Candidates13.length then search edge 3 [439, 2797] root82Candidates13 3236 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates14 : List Nat := [18181]
set_option maxRecDepth 8192 in
theorem root82Candidates14_checked : candidates edge 4 439 2887 root82Tail15 winner.bound = root82Candidates14 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root82Step14 : scan edge (search edge 3) 4 [439] 439 root82Tail14 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail15 winner := by
  rw [root82Tail14, scan, if_neg (by decide), root82Candidates14_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail15
    (if 3 ≤ root82Candidates14.length then search edge 3 [439, 2887] root82Candidates14 3326 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates15 : List Nat := [8011]
set_option maxRecDepth 8192 in
theorem root82Candidates15_checked : candidates edge 4 439 3511 root82Tail16 winner.bound = root82Candidates15 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root82Step15 : scan edge (search edge 3) 4 [439] 439 root82Tail15 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail16 winner := by
  rw [root82Tail15, scan, if_neg (by decide), root82Candidates15_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail16
    (if 3 ≤ root82Candidates15.length then search edge 3 [439, 3511] root82Candidates15 3950 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates16 : List Nat := [4441, 18097]
set_option maxRecDepth 8192 in
theorem root82Candidates16_checked : candidates edge 4 439 4021 root82Tail17 winner.bound = root82Candidates16 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root82Step16 : scan edge (search edge 3) 4 [439] 439 root82Tail16 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail17 winner := by
  rw [root82Tail16, scan, if_neg (by decide), root82Candidates16_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail17
    (if 3 ≤ root82Candidates16.length then search edge 3 [439, 4021] root82Candidates16 4460 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates17 : List Nat := [6481]
set_option maxRecDepth 8192 in
theorem root82Candidates17_checked : candidates edge 4 439 4441 root82Tail18 winner.bound = root82Candidates17 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root82Step17 : scan edge (search edge 3) 4 [439] 439 root82Tail17 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail18 winner := by
  rw [root82Tail17, scan, if_neg (by decide), root82Candidates17_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail18
    (if 3 ≤ root82Candidates17.length then search edge 3 [439, 4441] root82Candidates17 4880 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates18 : List Nat := [15031, 19441]
set_option maxRecDepth 8192 in
theorem root82Candidates18_checked : candidates edge 4 439 4447 root82Tail19 winner.bound = root82Candidates18 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root82Step18 : scan edge (search edge 3) 4 [439] 439 root82Tail18 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail19 winner := by
  rw [root82Tail18, scan, if_neg (by decide), root82Candidates18_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail19
    (if 3 ≤ root82Candidates18.length then search edge 3 [439, 4447] root82Candidates18 4886 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates19 : List Nat := [14293]
set_option maxRecDepth 8192 in
theorem root82Candidates19_checked : candidates edge 4 439 4783 root82Tail20 winner.bound = root82Candidates19 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root82Step19 : scan edge (search edge 3) 4 [439] 439 root82Tail19 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail20 winner := by
  rw [root82Tail19, scan, if_neg (by decide), root82Candidates19_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail20
    (if 3 ≤ root82Candidates19.length then search edge 3 [439, 4783] root82Candidates19 5222 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates20 : List Nat := [13687]
set_option maxRecDepth 8192 in
theorem root82Candidates20_checked : candidates edge 4 439 5527 root82Tail21 winner.bound = root82Candidates20 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root82Step20 : scan edge (search edge 3) 4 [439] 439 root82Tail20 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail21 winner := by
  rw [root82Tail20, scan, if_neg (by decide), root82Candidates20_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail21
    (if 3 ≤ root82Candidates20.length then search edge 3 [439, 5527] root82Candidates20 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root82Candidates21_checked : candidates edge 4 439 5557 root82Tail22 winner.bound = root82Candidates21 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root82Step21 : scan edge (search edge 3) 4 [439] 439 root82Tail21 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail22 winner := by
  rw [root82Tail21, scan, if_neg (by decide), root82Candidates21_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail22
    (if 3 ≤ root82Candidates21.length then search edge 3 [439, 5557] root82Candidates21 5996 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates22 : List Nat := [6481]
set_option maxRecDepth 8192 in
theorem root82Candidates22_checked : candidates edge 4 439 5701 root82Tail23 winner.bound = root82Candidates22 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root82Step22 : scan edge (search edge 3) 4 [439] 439 root82Tail22 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail23 winner := by
  rw [root82Tail22, scan, if_neg (by decide), root82Candidates22_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail23
    (if 3 ≤ root82Candidates22.length then search edge 3 [439, 5701] root82Candidates22 6140 winner else winner) = _
  rw [if_neg (by decide)]
def root82Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root82Candidates23_checked : candidates edge 4 439 5827 root82Tail24 winner.bound = root82Candidates23 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root82Step23 : scan edge (search edge 3) 4 [439] 439 root82Tail23 winner =
    scan edge (search edge 3) 4 [439] 439 root82Tail24 winner := by
  rw [root82Tail23, scan, if_neg (by decide), root82Candidates23_checked]
  change scan edge (search edge 3) 4 [439] 439 root82Tail24
    (if 3 ≤ root82Candidates23.length then search edge 3 [439, 5827] root82Candidates23 6266 winner else winner) = _
  rw [if_neg (by decide)]
theorem root82Scan24 : scan edge (search edge 3) 4 [439] 439 root82Tail24 winner = winner := by rw [root82Tail24, scan, if_pos (by decide)]
theorem root82Scan23 : scan edge (search edge 3) 4 [439] 439 root82Tail23 winner = winner :=
  (root82Step23).trans root82Scan24
theorem root82Scan22 : scan edge (search edge 3) 4 [439] 439 root82Tail22 winner = winner :=
  (root82Step22).trans root82Scan23
theorem root82Scan21 : scan edge (search edge 3) 4 [439] 439 root82Tail21 winner = winner :=
  (root82Step21).trans root82Scan22
theorem root82Scan20 : scan edge (search edge 3) 4 [439] 439 root82Tail20 winner = winner :=
  (root82Step20).trans root82Scan21
theorem root82Scan19 : scan edge (search edge 3) 4 [439] 439 root82Tail19 winner = winner :=
  (root82Step19).trans root82Scan20
theorem root82Scan18 : scan edge (search edge 3) 4 [439] 439 root82Tail18 winner = winner :=
  (root82Step18).trans root82Scan19
theorem root82Scan17 : scan edge (search edge 3) 4 [439] 439 root82Tail17 winner = winner :=
  (root82Step17).trans root82Scan18
theorem root82Scan16 : scan edge (search edge 3) 4 [439] 439 root82Tail16 winner = winner :=
  (root82Step16).trans root82Scan17
theorem root82Scan15 : scan edge (search edge 3) 4 [439] 439 root82Tail15 winner = winner :=
  (root82Step15).trans root82Scan16
theorem root82Scan14 : scan edge (search edge 3) 4 [439] 439 root82Tail14 winner = winner :=
  (root82Step14).trans root82Scan15
theorem root82Scan13 : scan edge (search edge 3) 4 [439] 439 root82Tail13 winner = winner :=
  (root82Step13).trans root82Scan14
theorem root82Scan12 : scan edge (search edge 3) 4 [439] 439 root82Tail12 winner = winner :=
  (root82Step12).trans root82Scan13
theorem root82Scan11 : scan edge (search edge 3) 4 [439] 439 root82Tail11 winner = winner :=
  (root82Step11).trans root82Scan12
theorem root82Scan10 : scan edge (search edge 3) 4 [439] 439 root82Tail10 winner = winner :=
  (root82Step10).trans root82Scan11
theorem root82Scan9 : scan edge (search edge 3) 4 [439] 439 root82Tail9 winner = winner :=
  (root82Step9).trans root82Scan10
theorem root82Scan8 : scan edge (search edge 3) 4 [439] 439 root82Tail8 winner = winner :=
  (root82Step8).trans root82Scan9
theorem root82Scan7 : scan edge (search edge 3) 4 [439] 439 root82Tail7 winner = winner :=
  (root82Step7).trans root82Scan8
theorem root82Scan6 : scan edge (search edge 3) 4 [439] 439 root82Tail6 winner = winner :=
  (root82Step6).trans root82Scan7
theorem root82Scan5 : scan edge (search edge 3) 4 [439] 439 root82Tail5 winner = winner :=
  (root82Step5).trans root82Scan6
theorem root82Scan4 : scan edge (search edge 3) 4 [439] 439 root82Tail4 winner = winner :=
  (root82Step4).trans root82Scan5
theorem root82Scan3 : scan edge (search edge 3) 4 [439] 439 root82Tail3 winner = winner :=
  (root82Step3).trans root82Scan4
theorem root82Scan2 : scan edge (search edge 3) 4 [439] 439 root82Tail2 winner = winner :=
  (root82Step2).trans root82Scan3
theorem root82Scan1 : scan edge (search edge 3) 4 [439] 439 root82Tail1 winner = winner :=
  (root82Step1).trans root82Scan2
theorem root82Scan0 : scan edge (search edge 3) 4 [439] 439 root82Tail0 winner = winner :=
  (root82Step0).trans root82Scan1
set_option maxRecDepth 4096 in
theorem root82Branch : search edge 4 [439] root82Candidates 439 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [439] 439 root82Tail0 winner = winner
  exact root82Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
