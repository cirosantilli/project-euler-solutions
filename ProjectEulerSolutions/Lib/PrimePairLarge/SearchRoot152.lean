import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root152Candidates : List Nat := Data.row907
set_option maxRecDepth 8192 in
theorem root152Candidates_checked : candidates edge 5 0 907 tail153 winner.bound = root152Candidates := by
  apply candidates_eq_row edge 5 0 907 tail153 root152Candidates winner.bound tail153_ordered
  · decide
  · decide
  · funext q
    rfl
def root152Tail63 : List Nat := []
def root152Tail62 : List Nat := 24793 :: root152Tail63
def root152Tail61 : List Nat := 24169 :: root152Tail62
def root152Tail60 : List Nat := 24043 :: root152Tail61
def root152Tail59 : List Nat := 23557 :: root152Tail60
def root152Tail58 : List Nat := 22861 :: root152Tail59
def root152Tail57 : List Nat := 22717 :: root152Tail58
def root152Tail56 : List Nat := 22147 :: root152Tail57
def root152Tail55 : List Nat := 21559 :: root152Tail56
def root152Tail54 : List Nat := 20599 :: root152Tail55
def root152Tail53 : List Nat := 20023 :: root152Tail54
def root152Tail52 : List Nat := 17449 :: root152Tail53
def root152Tail51 : List Nat := 17383 :: root152Tail52
def root152Tail50 : List Nat := 17299 :: root152Tail51
def root152Tail49 : List Nat := 17047 :: root152Tail50
def root152Tail48 : List Nat := 16729 :: root152Tail49
def root152Tail47 : List Nat := 16519 :: root152Tail48
def root152Tail46 : List Nat := 16141 :: root152Tail47
def root152Tail45 : List Nat := 15643 :: root152Tail46
def root152Tail44 : List Nat := 15619 :: root152Tail45
def root152Tail43 : List Nat := 15193 :: root152Tail44
def root152Tail42 : List Nat := 15073 :: root152Tail43
def root152Tail41 : List Nat := 14779 :: root152Tail42
def root152Tail40 : List Nat := 14551 :: root152Tail41
def root152Tail39 : List Nat := 14149 :: root152Tail40
def root152Tail38 : List Nat := 13879 :: root152Tail39
def root152Tail37 : List Nat := 13009 :: root152Tail38
def root152Tail36 : List Nat := 12823 :: root152Tail37
def root152Tail35 : List Nat := 12703 :: root152Tail36
def root152Tail34 : List Nat := 12343 :: root152Tail35
def root152Tail33 : List Nat := 12301 :: root152Tail34
def root152Tail32 : List Nat := 12073 :: root152Tail33
def root152Tail31 : List Nat := 12007 :: root152Tail32
def root152Tail30 : List Nat := 11527 :: root152Tail31
def root152Tail29 : List Nat := 10099 :: root152Tail30
def root152Tail28 : List Nat := 9337 :: root152Tail29
def root152Tail27 : List Nat := 8539 :: root152Tail28
def root152Tail26 : List Nat := 7993 :: root152Tail27
def root152Tail25 : List Nat := 7723 :: root152Tail26
def root152Tail24 : List Nat := 7591 :: root152Tail25
def root152Tail23 : List Nat := 7027 :: root152Tail24
def root152Tail22 : List Nat := 6961 :: root152Tail23
def root152Tail21 : List Nat := 6841 :: root152Tail22
def root152Tail20 : List Nat := 6733 :: root152Tail21
def root152Tail19 : List Nat := 6337 :: root152Tail20
def root152Tail18 : List Nat := 6229 :: root152Tail19
def root152Tail17 : List Nat := 5851 :: root152Tail18
def root152Tail16 : List Nat := 5503 :: root152Tail17
def root152Tail15 : List Nat := 4993 :: root152Tail16
def root152Tail14 : List Nat := 4987 :: root152Tail15
def root152Tail13 : List Nat := 4861 :: root152Tail14
def root152Tail12 : List Nat := 4567 :: root152Tail13
def root152Tail11 : List Nat := 4423 :: root152Tail12
def root152Tail10 : List Nat := 4357 :: root152Tail11
def root152Tail9 : List Nat := 3919 :: root152Tail10
def root152Tail8 : List Nat := 3709 :: root152Tail9
def root152Tail7 : List Nat := 2731 :: root152Tail8
def root152Tail6 : List Nat := 2473 :: root152Tail7
def root152Tail5 : List Nat := 2383 :: root152Tail6
def root152Tail4 : List Nat := 1801 :: root152Tail5
def root152Tail3 : List Nat := 1471 :: root152Tail4
def root152Tail2 : List Nat := 1459 :: root152Tail3
def root152Tail1 : List Nat := 1429 :: root152Tail2
def root152Tail0 : List Nat := 1213 :: root152Tail1
def root152Candidates0 : List Nat := [12007]
set_option maxRecDepth 8192 in
theorem root152Candidates0_checked : candidates edge 4 907 1213 root152Tail1 winner.bound = root152Candidates0 := by
  unfold candidates
  have row : edge 1213 = (SortedNatTree.contains FastRows.row1213) := by
    change Data.row1213.contains = (SortedNatTree.contains FastRows.row1213)
    exact FastRows.row1213_checked.symm
  rw [row]
  decide
theorem root152Step0 : scan edge (search edge 3) 4 [907] 907 root152Tail0 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail1 winner := by
  rw [root152Tail0, scan, if_neg (by decide), root152Candidates0_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail1
    (if 3 ≤ root152Candidates0.length then search edge 3 [907, 1213] root152Candidates0 2120 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates1 : List Nat := [3919, 22861]
set_option maxRecDepth 8192 in
theorem root152Candidates1_checked : candidates edge 4 907 1429 root152Tail2 winner.bound = root152Candidates1 := by
  unfold candidates
  have row : edge 1429 = (SortedNatTree.contains FastRows.row1429) := by
    change Data.row1429.contains = (SortedNatTree.contains FastRows.row1429)
    exact FastRows.row1429_checked.symm
  rw [row]
  decide
theorem root152Step1 : scan edge (search edge 3) 4 [907] 907 root152Tail1 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail2 winner := by
  rw [root152Tail1, scan, if_neg (by decide), root152Candidates1_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail2
    (if 3 ≤ root152Candidates1.length then search edge 3 [907, 1429] root152Candidates1 2336 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates2 : List Nat := [7027, 12301, 20023]
set_option maxRecDepth 8192 in
theorem root152Candidates2_checked : candidates edge 4 907 1459 root152Tail3 winner.bound = root152Candidates2 := by
  unfold candidates
  have row : edge 1459 = (SortedNatTree.contains FastRows.row1459) := by
    change Data.row1459.contains = (SortedNatTree.contains FastRows.row1459)
    exact FastRows.row1459_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root152Child2 : search edge 3 [907, 1459] root152Candidates2 2366 winner = winner := by decide
theorem root152Step2 : scan edge (search edge 3) 4 [907] 907 root152Tail2 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail3 winner := by
  rw [root152Tail2, scan, if_neg (by decide), root152Candidates2_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail3
    (if 3 ≤ root152Candidates2.length then search edge 3 [907, 1459] root152Candidates2 2366 winner else winner) = _
  rw [if_pos (by decide), root152Child2]
def root152Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root152Candidates3_checked : candidates edge 4 907 1471 root152Tail4 winner.bound = root152Candidates3 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
theorem root152Step3 : scan edge (search edge 3) 4 [907] 907 root152Tail3 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail4 winner := by
  rw [root152Tail3, scan, if_neg (by decide), root152Candidates3_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail4
    (if 3 ≤ root152Candidates3.length then search edge 3 [907, 1471] root152Candidates3 2378 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root152Candidates4_checked : candidates edge 4 907 1801 root152Tail5 winner.bound = root152Candidates4 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root152Step4 : scan edge (search edge 3) 4 [907] 907 root152Tail4 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail5 winner := by
  rw [root152Tail4, scan, if_neg (by decide), root152Candidates4_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail5
    (if 3 ≤ root152Candidates4.length then search edge 3 [907, 1801] root152Candidates4 2708 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates5 : List Nat := [6733]
set_option maxRecDepth 8192 in
theorem root152Candidates5_checked : candidates edge 4 907 2383 root152Tail6 winner.bound = root152Candidates5 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root152Step5 : scan edge (search edge 3) 4 [907] 907 root152Tail5 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail6 winner := by
  rw [root152Tail5, scan, if_neg (by decide), root152Candidates5_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail6
    (if 3 ≤ root152Candidates5.length then search edge 3 [907, 2383] root152Candidates5 3290 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates6 : List Nat := [12301, 14149]
set_option maxRecDepth 8192 in
theorem root152Candidates6_checked : candidates edge 4 907 2473 root152Tail7 winner.bound = root152Candidates6 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
theorem root152Step6 : scan edge (search edge 3) 4 [907] 907 root152Tail6 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail7 winner := by
  rw [root152Tail6, scan, if_neg (by decide), root152Candidates6_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail7
    (if 3 ≤ root152Candidates6.length then search edge 3 [907, 2473] root152Candidates6 3380 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates7 : List Nat := [4423, 7993]
set_option maxRecDepth 8192 in
theorem root152Candidates7_checked : candidates edge 4 907 2731 root152Tail8 winner.bound = root152Candidates7 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
theorem root152Step7 : scan edge (search edge 3) 4 [907] 907 root152Tail7 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail8 winner := by
  rw [root152Tail7, scan, if_neg (by decide), root152Candidates7_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail8
    (if 3 ≤ root152Candidates7.length then search edge 3 [907, 2731] root152Candidates7 3638 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root152Candidates8_checked : candidates edge 4 907 3709 root152Tail9 winner.bound = root152Candidates8 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root152Step8 : scan edge (search edge 3) 4 [907] 907 root152Tail8 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail9 winner := by
  rw [root152Tail8, scan, if_neg (by decide), root152Candidates8_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail9
    (if 3 ≤ root152Candidates8.length then search edge 3 [907, 3709] root152Candidates8 4616 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates9 : List Nat := [4567, 5503, 6961, 16141]
set_option maxRecDepth 8192 in
theorem root152Candidates9_checked : candidates edge 4 907 3919 root152Tail10 winner.bound = root152Candidates9 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root152Child9 : search edge 3 [907, 3919] root152Candidates9 4826 winner = winner := by decide
theorem root152Step9 : scan edge (search edge 3) 4 [907] 907 root152Tail9 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail10 winner := by
  rw [root152Tail9, scan, if_neg (by decide), root152Candidates9_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail10
    (if 3 ≤ root152Candidates9.length then search edge 3 [907, 3919] root152Candidates9 4826 winner else winner) = _
  rw [if_pos (by decide), root152Child9]
def root152Candidates10 : List Nat := [14551, 17383]
set_option maxRecDepth 8192 in
theorem root152Candidates10_checked : candidates edge 4 907 4357 root152Tail11 winner.bound = root152Candidates10 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root152Step10 : scan edge (search edge 3) 4 [907] 907 root152Tail10 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail11 winner := by
  rw [root152Tail10, scan, if_neg (by decide), root152Candidates10_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail11
    (if 3 ≤ root152Candidates10.length then search edge 3 [907, 4357] root152Candidates10 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates11 : List Nat := [9337, 16141]
set_option maxRecDepth 8192 in
theorem root152Candidates11_checked : candidates edge 4 907 4423 root152Tail12 winner.bound = root152Candidates11 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root152Step11 : scan edge (search edge 3) 4 [907] 907 root152Tail11 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail12 winner := by
  rw [root152Tail11, scan, if_neg (by decide), root152Candidates11_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail12
    (if 3 ≤ root152Candidates11.length then search edge 3 [907, 4423] root152Candidates11 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates12 : List Nat := [7993, 14779, 15073]
set_option maxRecDepth 8192 in
theorem root152Candidates12_checked : candidates edge 4 907 4567 root152Tail13 winner.bound = root152Candidates12 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root152Child12 : search edge 3 [907, 4567] root152Candidates12 5474 winner = winner := by decide
theorem root152Step12 : scan edge (search edge 3) 4 [907] 907 root152Tail12 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail13 winner := by
  rw [root152Tail12, scan, if_neg (by decide), root152Candidates12_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail13
    (if 3 ≤ root152Candidates12.length then search edge 3 [907, 4567] root152Candidates12 5474 winner else winner) = _
  rw [if_pos (by decide), root152Child12]
def root152Candidates13 : List Nat := [6229, 17449]
set_option maxRecDepth 8192 in
theorem root152Candidates13_checked : candidates edge 4 907 4861 root152Tail14 winner.bound = root152Candidates13 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root152Step13 : scan edge (search edge 3) 4 [907] 907 root152Tail13 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail14 winner := by
  rw [root152Tail13, scan, if_neg (by decide), root152Candidates13_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail14
    (if 3 ≤ root152Candidates13.length then search edge 3 [907, 4861] root152Candidates13 5768 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates14 : List Nat := [14149]
set_option maxRecDepth 8192 in
theorem root152Candidates14_checked : candidates edge 4 907 4987 root152Tail15 winner.bound = root152Candidates14 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root152Step14 : scan edge (search edge 3) 4 [907] 907 root152Tail14 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail15 winner := by
  rw [root152Tail14, scan, if_neg (by decide), root152Candidates14_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail15
    (if 3 ≤ root152Candidates14.length then search edge 3 [907, 4987] root152Candidates14 5894 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates15 : List Nat := [17383]
set_option maxRecDepth 8192 in
theorem root152Candidates15_checked : candidates edge 4 907 4993 root152Tail16 winner.bound = root152Candidates15 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root152Step15 : scan edge (search edge 3) 4 [907] 907 root152Tail15 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail16 winner := by
  rw [root152Tail15, scan, if_neg (by decide), root152Candidates15_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail16
    (if 3 ≤ root152Candidates15.length then search edge 3 [907, 4993] root152Candidates15 5900 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates16 : List Nat := [7027, 16141]
set_option maxRecDepth 8192 in
theorem root152Candidates16_checked : candidates edge 4 907 5503 root152Tail17 winner.bound = root152Candidates16 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root152Step16 : scan edge (search edge 3) 4 [907] 907 root152Tail16 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail17 winner := by
  rw [root152Tail16, scan, if_neg (by decide), root152Candidates16_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail17
    (if 3 ≤ root152Candidates16.length then search edge 3 [907, 5503] root152Candidates16 6410 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates17 : List Nat := [6961]
set_option maxRecDepth 8192 in
theorem root152Candidates17_checked : candidates edge 4 907 5851 root152Tail18 winner.bound = root152Candidates17 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root152Step17 : scan edge (search edge 3) 4 [907] 907 root152Tail17 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail18 winner := by
  rw [root152Tail17, scan, if_neg (by decide), root152Candidates17_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail18
    (if 3 ≤ root152Candidates17.length then search edge 3 [907, 5851] root152Candidates17 6758 winner else winner) = _
  rw [if_neg (by decide)]
def root152Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root152Candidates18_checked : candidates edge 4 907 6229 root152Tail19 winner.bound = root152Candidates18 := by
  unfold candidates
  have row : edge 6229 = (SortedNatTree.contains FastRows.row6229) := by
    change Data.row6229.contains = (SortedNatTree.contains FastRows.row6229)
    exact FastRows.row6229_checked.symm
  rw [row]
  decide
theorem root152Step18 : scan edge (search edge 3) 4 [907] 907 root152Tail18 winner =
    scan edge (search edge 3) 4 [907] 907 root152Tail19 winner := by
  rw [root152Tail18, scan, if_neg (by decide), root152Candidates18_checked]
  change scan edge (search edge 3) 4 [907] 907 root152Tail19
    (if 3 ≤ root152Candidates18.length then search edge 3 [907, 6229] root152Candidates18 7136 winner else winner) = _
  rw [if_neg (by decide)]
theorem root152Scan19 : scan edge (search edge 3) 4 [907] 907 root152Tail19 winner = winner := by rw [root152Tail19, scan, if_pos (by decide)]
theorem root152Scan18 : scan edge (search edge 3) 4 [907] 907 root152Tail18 winner = winner :=
  (root152Step18).trans root152Scan19
theorem root152Scan17 : scan edge (search edge 3) 4 [907] 907 root152Tail17 winner = winner :=
  (root152Step17).trans root152Scan18
theorem root152Scan16 : scan edge (search edge 3) 4 [907] 907 root152Tail16 winner = winner :=
  (root152Step16).trans root152Scan17
theorem root152Scan15 : scan edge (search edge 3) 4 [907] 907 root152Tail15 winner = winner :=
  (root152Step15).trans root152Scan16
theorem root152Scan14 : scan edge (search edge 3) 4 [907] 907 root152Tail14 winner = winner :=
  (root152Step14).trans root152Scan15
theorem root152Scan13 : scan edge (search edge 3) 4 [907] 907 root152Tail13 winner = winner :=
  (root152Step13).trans root152Scan14
theorem root152Scan12 : scan edge (search edge 3) 4 [907] 907 root152Tail12 winner = winner :=
  (root152Step12).trans root152Scan13
theorem root152Scan11 : scan edge (search edge 3) 4 [907] 907 root152Tail11 winner = winner :=
  (root152Step11).trans root152Scan12
theorem root152Scan10 : scan edge (search edge 3) 4 [907] 907 root152Tail10 winner = winner :=
  (root152Step10).trans root152Scan11
theorem root152Scan9 : scan edge (search edge 3) 4 [907] 907 root152Tail9 winner = winner :=
  (root152Step9).trans root152Scan10
theorem root152Scan8 : scan edge (search edge 3) 4 [907] 907 root152Tail8 winner = winner :=
  (root152Step8).trans root152Scan9
theorem root152Scan7 : scan edge (search edge 3) 4 [907] 907 root152Tail7 winner = winner :=
  (root152Step7).trans root152Scan8
theorem root152Scan6 : scan edge (search edge 3) 4 [907] 907 root152Tail6 winner = winner :=
  (root152Step6).trans root152Scan7
theorem root152Scan5 : scan edge (search edge 3) 4 [907] 907 root152Tail5 winner = winner :=
  (root152Step5).trans root152Scan6
theorem root152Scan4 : scan edge (search edge 3) 4 [907] 907 root152Tail4 winner = winner :=
  (root152Step4).trans root152Scan5
theorem root152Scan3 : scan edge (search edge 3) 4 [907] 907 root152Tail3 winner = winner :=
  (root152Step3).trans root152Scan4
theorem root152Scan2 : scan edge (search edge 3) 4 [907] 907 root152Tail2 winner = winner :=
  (root152Step2).trans root152Scan3
theorem root152Scan1 : scan edge (search edge 3) 4 [907] 907 root152Tail1 winner = winner :=
  (root152Step1).trans root152Scan2
theorem root152Scan0 : scan edge (search edge 3) 4 [907] 907 root152Tail0 winner = winner :=
  (root152Step0).trans root152Scan1
set_option maxRecDepth 4096 in
theorem root152Branch : search edge 4 [907] root152Candidates 907 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [907] 907 root152Tail0 winner = winner
  exact root152Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
