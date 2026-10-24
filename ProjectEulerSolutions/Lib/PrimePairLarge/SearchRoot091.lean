import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root91Candidates : List Nat := Data.row491
set_option maxRecDepth 8192 in
theorem root91Candidates_checked : candidates edge 5 0 491 tail92 winner.bound = root91Candidates := by
  apply candidates_eq_row edge 5 0 491 tail92 root91Candidates winner.bound tail92_ordered
  · decide
  · decide
  · funext q
    rfl
def root91Tail69 : List Nat := []
def root91Tail68 : List Nat := 25367 :: root91Tail69
def root91Tail67 : List Nat := 24953 :: root91Tail68
def root91Tail66 : List Nat := 24371 :: root91Tail67
def root91Tail65 : List Nat := 23819 :: root91Tail66
def root91Tail64 : List Nat := 23609 :: root91Tail65
def root91Tail63 : List Nat := 23447 :: root91Tail64
def root91Tail62 : List Nat := 23201 :: root91Tail63
def root91Tail61 : List Nat := 22811 :: root91Tail62
def root91Tail60 : List Nat := 22613 :: root91Tail61
def root91Tail59 : List Nat := 22109 :: root91Tail60
def root91Tail58 : List Nat := 22013 :: root91Tail59
def root91Tail57 : List Nat := 21713 :: root91Tail58
def root91Tail56 : List Nat := 21587 :: root91Tail57
def root91Tail55 : List Nat := 21341 :: root91Tail56
def root91Tail54 : List Nat := 20549 :: root91Tail55
def root91Tail53 : List Nat := 19163 :: root91Tail54
def root91Tail52 : List Nat := 18947 :: root91Tail53
def root91Tail51 : List Nat := 18233 :: root91Tail52
def root91Tail50 : List Nat := 17477 :: root91Tail51
def root91Tail49 : List Nat := 16673 :: root91Tail50
def root91Tail48 : List Nat := 15887 :: root91Tail49
def root91Tail47 : List Nat := 15497 :: root91Tail48
def root91Tail46 : List Nat := 15083 :: root91Tail47
def root91Tail45 : List Nat := 14969 :: root91Tail46
def root91Tail44 : List Nat := 14831 :: root91Tail45
def root91Tail43 : List Nat := 14543 :: root91Tail44
def root91Tail42 : List Nat := 14159 :: root91Tail43
def root91Tail41 : List Nat := 13997 :: root91Tail42
def root91Tail40 : List Nat := 13781 :: root91Tail41
def root91Tail39 : List Nat := 13619 :: root91Tail40
def root91Tail38 : List Nat := 12941 :: root91Tail39
def root91Tail37 : List Nat := 12197 :: root91Tail38
def root91Tail36 : List Nat := 11813 :: root91Tail37
def root91Tail35 : List Nat := 11093 :: root91Tail36
def root91Tail34 : List Nat := 10889 :: root91Tail35
def root91Tail33 : List Nat := 10667 :: root91Tail34
def root91Tail32 : List Nat := 10559 :: root91Tail33
def root91Tail31 : List Nat := 10499 :: root91Tail32
def root91Tail30 : List Nat := 9623 :: root91Tail31
def root91Tail29 : List Nat := 9209 :: root91Tail30
def root91Tail28 : List Nat := 8999 :: root91Tail29
def root91Tail27 : List Nat := 8861 :: root91Tail28
def root91Tail26 : List Nat := 7853 :: root91Tail27
def root91Tail25 : List Nat := 7481 :: root91Tail26
def root91Tail24 : List Nat := 7079 :: root91Tail25
def root91Tail23 : List Nat := 6803 :: root91Tail24
def root91Tail22 : List Nat := 6719 :: root91Tail23
def root91Tail21 : List Nat := 6323 :: root91Tail22
def root91Tail20 : List Nat := 6257 :: root91Tail21
def root91Tail19 : List Nat := 6053 :: root91Tail20
def root91Tail18 : List Nat := 5843 :: root91Tail19
def root91Tail17 : List Nat := 4457 :: root91Tail18
def root91Tail16 : List Nat := 4211 :: root91Tail17
def root91Tail15 : List Nat := 3947 :: root91Tail16
def root91Tail14 : List Nat := 3917 :: root91Tail15
def root91Tail13 : List Nat := 3617 :: root91Tail14
def root91Tail12 : List Nat := 3449 :: root91Tail13
def root91Tail11 : List Nat := 2837 :: root91Tail12
def root91Tail10 : List Nat := 2459 :: root91Tail11
def root91Tail9 : List Nat := 2447 :: root91Tail10
def root91Tail8 : List Nat := 2309 :: root91Tail9
def root91Tail7 : List Nat := 1871 :: root91Tail8
def root91Tail6 : List Nat := 1607 :: root91Tail7
def root91Tail5 : List Nat := 1523 :: root91Tail6
def root91Tail4 : List Nat := 1409 :: root91Tail5
def root91Tail3 : List Nat := 983 :: root91Tail4
def root91Tail2 : List Nat := 773 :: root91Tail3
def root91Tail1 : List Nat := 653 :: root91Tail2
def root91Tail0 : List Nat := 593 :: root91Tail1
def root91Candidates0 : List Nat := [4211, 9209, 20549, 24371]
set_option maxRecDepth 8192 in
theorem root91Candidates0_checked : candidates edge 4 491 593 root91Tail1 winner.bound = root91Candidates0 := by
  unfold candidates
  have row : edge 593 = (SortedNatTree.contains FastRows.row593) := by
    change Data.row593.contains = (SortedNatTree.contains FastRows.row593)
    exact FastRows.row593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child0 : search edge 3 [491, 593] root91Candidates0 1084 winner = winner := by decide
theorem root91Step0 : scan edge (search edge 3) 4 [491] 491 root91Tail0 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail1 winner := by
  rw [root91Tail0, scan, if_neg (by decide), root91Candidates0_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail1
    (if 3 ≤ root91Candidates0.length then search edge 3 [491, 593] root91Candidates0 1084 winner else winner) = _
  rw [if_pos (by decide), root91Child0]
def root91Candidates1 : List Nat := [1523]
set_option maxRecDepth 8192 in
theorem root91Candidates1_checked : candidates edge 4 491 653 root91Tail2 winner.bound = root91Candidates1 := by
  unfold candidates
  have row : edge 653 = (SortedNatTree.contains FastRows.row653) := by
    change Data.row653.contains = (SortedNatTree.contains FastRows.row653)
    exact FastRows.row653_checked.symm
  rw [row]
  decide
theorem root91Step1 : scan edge (search edge 3) 4 [491] 491 root91Tail1 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail2 winner := by
  rw [root91Tail1, scan, if_neg (by decide), root91Candidates1_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail2
    (if 3 ≤ root91Candidates1.length then search edge 3 [491, 653] root91Candidates1 1144 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates2 : List Nat := [14543, 15083, 17477]
set_option maxRecDepth 8192 in
theorem root91Candidates2_checked : candidates edge 4 491 773 root91Tail3 winner.bound = root91Candidates2 := by
  unfold candidates
  have row : edge 773 = (SortedNatTree.contains FastRows.row773) := by
    change Data.row773.contains = (SortedNatTree.contains FastRows.row773)
    exact FastRows.row773_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child2 : search edge 3 [491, 773] root91Candidates2 1264 winner = winner := by decide
theorem root91Step2 : scan edge (search edge 3) 4 [491] 491 root91Tail2 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail3 winner := by
  rw [root91Tail2, scan, if_neg (by decide), root91Candidates2_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail3
    (if 3 ≤ root91Candidates2.length then search edge 3 [491, 773] root91Candidates2 1264 winner else winner) = _
  rw [if_pos (by decide), root91Child2]
def root91Candidates3 : List Nat := [3617, 6719, 7481, 15083]
set_option maxRecDepth 8192 in
theorem root91Candidates3_checked : candidates edge 4 491 983 root91Tail4 winner.bound = root91Candidates3 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child3 : search edge 3 [491, 983] root91Candidates3 1474 winner = winner := by decide
theorem root91Step3 : scan edge (search edge 3) 4 [491] 491 root91Tail3 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail4 winner := by
  rw [root91Tail3, scan, if_neg (by decide), root91Candidates3_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail4
    (if 3 ≤ root91Candidates3.length then search edge 3 [491, 983] root91Candidates3 1474 winner else winner) = _
  rw [if_pos (by decide), root91Child3]
def root91Candidates4 : List Nat := [15083, 22811, 23609, 23819]
set_option maxRecDepth 8192 in
theorem root91Candidates4_checked : candidates edge 4 491 1409 root91Tail5 winner.bound = root91Candidates4 := by
  unfold candidates
  have row : edge 1409 = (SortedNatTree.contains FastRows.row1409) := by
    change Data.row1409.contains = (SortedNatTree.contains FastRows.row1409)
    exact FastRows.row1409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child4 : search edge 3 [491, 1409] root91Candidates4 1900 winner = winner := by decide
theorem root91Step4 : scan edge (search edge 3) 4 [491] 491 root91Tail4 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail5 winner := by
  rw [root91Tail4, scan, if_neg (by decide), root91Candidates4_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail5
    (if 3 ≤ root91Candidates4.length then search edge 3 [491, 1409] root91Candidates4 1900 winner else winner) = _
  rw [if_pos (by decide), root91Child4]
def root91Candidates5 : List Nat := [2447, 19163, 22013]
set_option maxRecDepth 8192 in
theorem root91Candidates5_checked : candidates edge 4 491 1523 root91Tail6 winner.bound = root91Candidates5 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child5 : search edge 3 [491, 1523] root91Candidates5 2014 winner = winner := by decide
theorem root91Step5 : scan edge (search edge 3) 4 [491] 491 root91Tail5 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail6 winner := by
  rw [root91Tail5, scan, if_neg (by decide), root91Candidates5_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail6
    (if 3 ≤ root91Candidates5.length then search edge 3 [491, 1523] root91Candidates5 2014 winner else winner) = _
  rw [if_pos (by decide), root91Child5]
def root91Candidates6 : List Nat := [22013]
set_option maxRecDepth 8192 in
theorem root91Candidates6_checked : candidates edge 4 491 1607 root91Tail7 winner.bound = root91Candidates6 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
theorem root91Step6 : scan edge (search edge 3) 4 [491] 491 root91Tail6 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail7 winner := by
  rw [root91Tail6, scan, if_neg (by decide), root91Candidates6_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail7
    (if 3 ≤ root91Candidates6.length then search edge 3 [491, 1607] root91Candidates6 2098 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates7 : List Nat := [6323, 9209]
set_option maxRecDepth 8192 in
theorem root91Candidates7_checked : candidates edge 4 491 1871 root91Tail8 winner.bound = root91Candidates7 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
theorem root91Step7 : scan edge (search edge 3) 4 [491] 491 root91Tail7 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail8 winner := by
  rw [root91Tail7, scan, if_neg (by decide), root91Candidates7_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail8
    (if 3 ≤ root91Candidates7.length then search edge 3 [491, 1871] root91Candidates7 2362 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates8 : List Nat := [6257, 8861, 12197]
set_option maxRecDepth 8192 in
theorem root91Candidates8_checked : candidates edge 4 491 2309 root91Tail9 winner.bound = root91Candidates8 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child8 : search edge 3 [491, 2309] root91Candidates8 2800 winner = winner := by decide
theorem root91Step8 : scan edge (search edge 3) 4 [491] 491 root91Tail8 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail9 winner := by
  rw [root91Tail8, scan, if_neg (by decide), root91Candidates8_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail9
    (if 3 ≤ root91Candidates8.length then search edge 3 [491, 2309] root91Candidates8 2800 winner else winner) = _
  rw [if_pos (by decide), root91Child8]
def root91Candidates9 : List Nat := [6323, 10889, 12197]
set_option maxRecDepth 8192 in
theorem root91Candidates9_checked : candidates edge 4 491 2447 root91Tail10 winner.bound = root91Candidates9 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child9 : search edge 3 [491, 2447] root91Candidates9 2938 winner = winner := by decide
theorem root91Step9 : scan edge (search edge 3) 4 [491] 491 root91Tail9 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail10 winner := by
  rw [root91Tail9, scan, if_neg (by decide), root91Candidates9_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail10
    (if 3 ≤ root91Candidates9.length then search edge 3 [491, 2447] root91Candidates9 2938 winner else winner) = _
  rw [if_pos (by decide), root91Child9]
def root91Candidates10 : List Nat := [5843, 14159]
set_option maxRecDepth 8192 in
theorem root91Candidates10_checked : candidates edge 4 491 2459 root91Tail11 winner.bound = root91Candidates10 := by
  unfold candidates
  have row : edge 2459 = (SortedNatTree.contains FastRows.row2459) := by
    change Data.row2459.contains = (SortedNatTree.contains FastRows.row2459)
    exact FastRows.row2459_checked.symm
  rw [row]
  decide
theorem root91Step10 : scan edge (search edge 3) 4 [491] 491 root91Tail10 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail11 winner := by
  rw [root91Tail10, scan, if_neg (by decide), root91Candidates10_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail11
    (if 3 ≤ root91Candidates10.length then search edge 3 [491, 2459] root91Candidates10 2950 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates11 : List Nat := [3617, 9209]
set_option maxRecDepth 8192 in
theorem root91Candidates11_checked : candidates edge 4 491 2837 root91Tail12 winner.bound = root91Candidates11 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
theorem root91Step11 : scan edge (search edge 3) 4 [491] 491 root91Tail11 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail12 winner := by
  rw [root91Tail11, scan, if_neg (by decide), root91Candidates11_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail12
    (if 3 ≤ root91Candidates11.length then search edge 3 [491, 2837] root91Candidates11 3328 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates12 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root91Candidates12_checked : candidates edge 4 491 3449 root91Tail13 winner.bound = root91Candidates12 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root91Step12 : scan edge (search edge 3) 4 [491] 491 root91Tail12 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail13 winner := by
  rw [root91Tail12, scan, if_neg (by decide), root91Candidates12_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail13
    (if 3 ≤ root91Candidates12.length then search edge 3 [491, 3449] root91Candidates12 3940 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates13 : List Nat := [10499, 16673]
set_option maxRecDepth 8192 in
theorem root91Candidates13_checked : candidates edge 4 491 3617 root91Tail14 winner.bound = root91Candidates13 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root91Step13 : scan edge (search edge 3) 4 [491] 491 root91Tail13 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail14 winner := by
  rw [root91Tail13, scan, if_neg (by decide), root91Candidates13_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail14
    (if 3 ≤ root91Candidates13.length then search edge 3 [491, 3617] root91Candidates13 4108 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates14 : List Nat := [4211, 6323]
set_option maxRecDepth 8192 in
theorem root91Candidates14_checked : candidates edge 4 491 3917 root91Tail15 winner.bound = root91Candidates14 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root91Step14 : scan edge (search edge 3) 4 [491] 491 root91Tail14 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail15 winner := by
  rw [root91Tail14, scan, if_neg (by decide), root91Candidates14_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail15
    (if 3 ≤ root91Candidates14.length then search edge 3 [491, 3917] root91Candidates14 4408 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates15 : List Nat := [6803, 8861, 11813]
set_option maxRecDepth 8192 in
theorem root91Candidates15_checked : candidates edge 4 491 3947 root91Tail16 winner.bound = root91Candidates15 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root91Child15 : search edge 3 [491, 3947] root91Candidates15 4438 winner = winner := by decide
theorem root91Step15 : scan edge (search edge 3) 4 [491] 491 root91Tail15 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail16 winner := by
  rw [root91Tail15, scan, if_neg (by decide), root91Candidates15_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail16
    (if 3 ≤ root91Candidates15.length then search edge 3 [491, 3947] root91Candidates15 4438 winner else winner) = _
  rw [if_pos (by decide), root91Child15]
def root91Candidates16 : List Nat := [13619]
set_option maxRecDepth 8192 in
theorem root91Candidates16_checked : candidates edge 4 491 4211 root91Tail17 winner.bound = root91Candidates16 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root91Step16 : scan edge (search edge 3) 4 [491] 491 root91Tail16 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail17 winner := by
  rw [root91Tail16, scan, if_neg (by decide), root91Candidates16_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail17
    (if 3 ≤ root91Candidates16.length then search edge 3 [491, 4211] root91Candidates16 4702 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates17 : List Nat := [7853]
set_option maxRecDepth 8192 in
theorem root91Candidates17_checked : candidates edge 4 491 4457 root91Tail18 winner.bound = root91Candidates17 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root91Step17 : scan edge (search edge 3) 4 [491] 491 root91Tail17 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail18 winner := by
  rw [root91Tail17, scan, if_neg (by decide), root91Candidates17_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail18
    (if 3 ≤ root91Candidates17.length then search edge 3 [491, 4457] root91Candidates17 4948 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root91Candidates18_checked : candidates edge 4 491 5843 root91Tail19 winner.bound = root91Candidates18 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root91Step18 : scan edge (search edge 3) 4 [491] 491 root91Tail18 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail19 winner := by
  rw [root91Tail18, scan, if_neg (by decide), root91Candidates18_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail19
    (if 3 ≤ root91Candidates18.length then search edge 3 [491, 5843] root91Candidates18 6334 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates19 : List Nat := [17477]
set_option maxRecDepth 8192 in
theorem root91Candidates19_checked : candidates edge 4 491 6053 root91Tail20 winner.bound = root91Candidates19 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root91Step19 : scan edge (search edge 3) 4 [491] 491 root91Tail19 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail20 winner := by
  rw [root91Tail19, scan, if_neg (by decide), root91Candidates19_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail20
    (if 3 ≤ root91Candidates19.length then search edge 3 [491, 6053] root91Candidates19 6544 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root91Candidates20_checked : candidates edge 4 491 6257 root91Tail21 winner.bound = root91Candidates20 := by
  unfold candidates
  have row : edge 6257 = (SortedNatTree.contains FastRows.row6257) := by
    change Data.row6257.contains = (SortedNatTree.contains FastRows.row6257)
    exact FastRows.row6257_checked.symm
  rw [row]
  decide
theorem root91Step20 : scan edge (search edge 3) 4 [491] 491 root91Tail20 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail21 winner := by
  rw [root91Tail20, scan, if_neg (by decide), root91Candidates20_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail21
    (if 3 ≤ root91Candidates20.length then search edge 3 [491, 6257] root91Candidates20 6748 winner else winner) = _
  rw [if_neg (by decide)]
def root91Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root91Candidates21_checked : candidates edge 4 491 6323 root91Tail22 winner.bound = root91Candidates21 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root91Step21 : scan edge (search edge 3) 4 [491] 491 root91Tail21 winner =
    scan edge (search edge 3) 4 [491] 491 root91Tail22 winner := by
  rw [root91Tail21, scan, if_neg (by decide), root91Candidates21_checked]
  change scan edge (search edge 3) 4 [491] 491 root91Tail22
    (if 3 ≤ root91Candidates21.length then search edge 3 [491, 6323] root91Candidates21 6814 winner else winner) = _
  rw [if_neg (by decide)]
theorem root91Scan22 : scan edge (search edge 3) 4 [491] 491 root91Tail22 winner = winner := by rw [root91Tail22, scan, if_pos (by decide)]
theorem root91Scan21 : scan edge (search edge 3) 4 [491] 491 root91Tail21 winner = winner :=
  (root91Step21).trans root91Scan22
theorem root91Scan20 : scan edge (search edge 3) 4 [491] 491 root91Tail20 winner = winner :=
  (root91Step20).trans root91Scan21
theorem root91Scan19 : scan edge (search edge 3) 4 [491] 491 root91Tail19 winner = winner :=
  (root91Step19).trans root91Scan20
theorem root91Scan18 : scan edge (search edge 3) 4 [491] 491 root91Tail18 winner = winner :=
  (root91Step18).trans root91Scan19
theorem root91Scan17 : scan edge (search edge 3) 4 [491] 491 root91Tail17 winner = winner :=
  (root91Step17).trans root91Scan18
theorem root91Scan16 : scan edge (search edge 3) 4 [491] 491 root91Tail16 winner = winner :=
  (root91Step16).trans root91Scan17
theorem root91Scan15 : scan edge (search edge 3) 4 [491] 491 root91Tail15 winner = winner :=
  (root91Step15).trans root91Scan16
theorem root91Scan14 : scan edge (search edge 3) 4 [491] 491 root91Tail14 winner = winner :=
  (root91Step14).trans root91Scan15
theorem root91Scan13 : scan edge (search edge 3) 4 [491] 491 root91Tail13 winner = winner :=
  (root91Step13).trans root91Scan14
theorem root91Scan12 : scan edge (search edge 3) 4 [491] 491 root91Tail12 winner = winner :=
  (root91Step12).trans root91Scan13
theorem root91Scan11 : scan edge (search edge 3) 4 [491] 491 root91Tail11 winner = winner :=
  (root91Step11).trans root91Scan12
theorem root91Scan10 : scan edge (search edge 3) 4 [491] 491 root91Tail10 winner = winner :=
  (root91Step10).trans root91Scan11
theorem root91Scan9 : scan edge (search edge 3) 4 [491] 491 root91Tail9 winner = winner :=
  (root91Step9).trans root91Scan10
theorem root91Scan8 : scan edge (search edge 3) 4 [491] 491 root91Tail8 winner = winner :=
  (root91Step8).trans root91Scan9
theorem root91Scan7 : scan edge (search edge 3) 4 [491] 491 root91Tail7 winner = winner :=
  (root91Step7).trans root91Scan8
theorem root91Scan6 : scan edge (search edge 3) 4 [491] 491 root91Tail6 winner = winner :=
  (root91Step6).trans root91Scan7
theorem root91Scan5 : scan edge (search edge 3) 4 [491] 491 root91Tail5 winner = winner :=
  (root91Step5).trans root91Scan6
theorem root91Scan4 : scan edge (search edge 3) 4 [491] 491 root91Tail4 winner = winner :=
  (root91Step4).trans root91Scan5
theorem root91Scan3 : scan edge (search edge 3) 4 [491] 491 root91Tail3 winner = winner :=
  (root91Step3).trans root91Scan4
theorem root91Scan2 : scan edge (search edge 3) 4 [491] 491 root91Tail2 winner = winner :=
  (root91Step2).trans root91Scan3
theorem root91Scan1 : scan edge (search edge 3) 4 [491] 491 root91Tail1 winner = winner :=
  (root91Step1).trans root91Scan2
theorem root91Scan0 : scan edge (search edge 3) 4 [491] 491 root91Tail0 winner = winner :=
  (root91Step0).trans root91Scan1
set_option maxRecDepth 4096 in
theorem root91Branch : search edge 4 [491] root91Candidates 491 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [491] 491 root91Tail0 winner = winner
  exact root91Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
