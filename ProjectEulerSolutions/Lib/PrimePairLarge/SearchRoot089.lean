import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root89Candidates : List Nat := Data.row479
set_option maxRecDepth 8192 in
theorem root89Candidates_checked : candidates edge 5 0 479 tail90 winner.bound = root89Candidates := by
  apply candidates_eq_row edge 5 0 479 tail90 root89Candidates winner.bound tail90_ordered
  · decide
  · decide
  · funext q
    rfl
def root89Tail64 : List Nat := []
def root89Tail63 : List Nat := 25253 :: root89Tail64
def root89Tail62 : List Nat := 24623 :: root89Tail63
def root89Tail61 : List Nat := 24029 :: root89Tail62
def root89Tail60 : List Nat := 23609 :: root89Tail61
def root89Tail59 : List Nat := 23417 :: root89Tail60
def root89Tail58 : List Nat := 23279 :: root89Tail59
def root89Tail57 : List Nat := 22073 :: root89Tail58
def root89Tail56 : List Nat := 21893 :: root89Tail57
def root89Tail55 : List Nat := 21323 :: root89Tail56
def root89Tail54 : List Nat := 20183 :: root89Tail55
def root89Tail53 : List Nat := 20123 :: root89Tail54
def root89Tail52 : List Nat := 19973 :: root89Tail53
def root89Tail51 : List Nat := 19793 :: root89Tail52
def root89Tail50 : List Nat := 19763 :: root89Tail51
def root89Tail49 : List Nat := 18719 :: root89Tail50
def root89Tail48 : List Nat := 17417 :: root89Tail49
def root89Tail47 : List Nat := 17291 :: root89Tail48
def root89Tail46 : List Nat := 16811 :: root89Tail47
def root89Tail45 : List Nat := 15761 :: root89Tail46
def root89Tail44 : List Nat := 15467 :: root89Tail45
def root89Tail43 : List Nat := 15383 :: root89Tail44
def root89Tail42 : List Nat := 14813 :: root89Tail43
def root89Tail41 : List Nat := 14249 :: root89Tail42
def root89Tail40 : List Nat := 13799 :: root89Tail41
def root89Tail39 : List Nat := 13721 :: root89Tail40
def root89Tail38 : List Nat := 13337 :: root89Tail39
def root89Tail37 : List Nat := 12671 :: root89Tail38
def root89Tail36 : List Nat := 12653 :: root89Tail37
def root89Tail35 : List Nat := 12539 :: root89Tail36
def root89Tail34 : List Nat := 12203 :: root89Tail35
def root89Tail33 : List Nat := 12041 :: root89Tail34
def root89Tail32 : List Nat := 11909 :: root89Tail33
def root89Tail31 : List Nat := 11813 :: root89Tail32
def root89Tail30 : List Nat := 11351 :: root89Tail31
def root89Tail29 : List Nat := 11261 :: root89Tail30
def root89Tail28 : List Nat := 10883 :: root89Tail29
def root89Tail27 : List Nat := 10859 :: root89Tail28
def root89Tail26 : List Nat := 10529 :: root89Tail27
def root89Tail25 : List Nat := 10091 :: root89Tail26
def root89Tail24 : List Nat := 9941 :: root89Tail25
def root89Tail23 : List Nat := 9857 :: root89Tail24
def root89Tail22 : List Nat := 9689 :: root89Tail23
def root89Tail21 : List Nat := 9227 :: root89Tail22
def root89Tail20 : List Nat := 8867 :: root89Tail21
def root89Tail19 : List Nat := 7517 :: root89Tail20
def root89Tail18 : List Nat := 6203 :: root89Tail19
def root89Tail17 : List Nat := 5807 :: root89Tail18
def root89Tail16 : List Nat := 4817 :: root89Tail17
def root89Tail15 : List Nat := 4733 :: root89Tail16
def root89Tail14 : List Nat := 4481 :: root89Tail15
def root89Tail13 : List Nat := 4337 :: root89Tail14
def root89Tail12 : List Nat := 3533 :: root89Tail13
def root89Tail11 : List Nat := 2297 :: root89Tail12
def root89Tail10 : List Nat := 1559 :: root89Tail11
def root89Tail9 : List Nat := 1307 :: root89Tail10
def root89Tail8 : List Nat := 1187 :: root89Tail9
def root89Tail7 : List Nat := 1049 :: root89Tail8
def root89Tail6 : List Nat := 971 :: root89Tail7
def root89Tail5 : List Nat := 881 :: root89Tail6
def root89Tail4 : List Nat := 821 :: root89Tail5
def root89Tail3 : List Nat := 701 :: root89Tail4
def root89Tail2 : List Nat := 599 :: root89Tail3
def root89Tail1 : List Nat := 593 :: root89Tail2
def root89Tail0 : List Nat := 569 :: root89Tail1
def root89Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root89Candidates0_checked : candidates edge 4 479 569 root89Tail1 winner.bound = root89Candidates0 := by
  unfold candidates
  have row : edge 569 = (SortedNatTree.contains FastRows.row569) := by
    change Data.row569.contains = (SortedNatTree.contains FastRows.row569)
    exact FastRows.row569_checked.symm
  rw [row]
  decide
theorem root89Step0 : scan edge (search edge 3) 4 [479] 479 root89Tail0 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail1 winner := by
  rw [root89Tail0, scan, if_neg (by decide), root89Candidates0_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail1
    (if 3 ≤ root89Candidates0.length then search edge 3 [479, 569] root89Candidates0 1048 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root89Candidates1_checked : candidates edge 4 479 593 root89Tail2 winner.bound = root89Candidates1 := by
  unfold candidates
  have row : edge 593 = (SortedNatTree.contains FastRows.row593) := by
    change Data.row593.contains = (SortedNatTree.contains FastRows.row593)
    exact FastRows.row593_checked.symm
  rw [row]
  decide
theorem root89Step1 : scan edge (search edge 3) 4 [479] 479 root89Tail1 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail2 winner := by
  rw [root89Tail1, scan, if_neg (by decide), root89Candidates1_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail2
    (if 3 ≤ root89Candidates1.length then search edge 3 [479, 593] root89Candidates1 1072 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates2 : List Nat := [7517, 20183]
set_option maxRecDepth 8192 in
theorem root89Candidates2_checked : candidates edge 4 479 599 root89Tail3 winner.bound = root89Candidates2 := by
  unfold candidates
  have row : edge 599 = (SortedNatTree.contains FastRows.row599) := by
    change Data.row599.contains = (SortedNatTree.contains FastRows.row599)
    exact FastRows.row599_checked.symm
  rw [row]
  decide
theorem root89Step2 : scan edge (search edge 3) 4 [479] 479 root89Tail2 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail3 winner := by
  rw [root89Tail2, scan, if_neg (by decide), root89Candidates2_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail3
    (if 3 ≤ root89Candidates2.length then search edge 3 [479, 599] root89Candidates2 1078 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates3 : List Nat := [1187, 1307, 8867, 11351]
set_option maxRecDepth 8192 in
theorem root89Candidates3_checked : candidates edge 4 479 701 root89Tail4 winner.bound = root89Candidates3 := by
  unfold candidates
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root89Child3 : search edge 3 [479, 701] root89Candidates3 1180 winner = winner := by decide
theorem root89Step3 : scan edge (search edge 3) 4 [479] 479 root89Tail3 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail4 winner := by
  rw [root89Tail3, scan, if_neg (by decide), root89Candidates3_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail4
    (if 3 ≤ root89Candidates3.length then search edge 3 [479, 701] root89Candidates3 1180 winner else winner) = _
  rw [if_pos (by decide), root89Child3]
def root89Candidates4 : List Nat := [971, 11261]
set_option maxRecDepth 8192 in
theorem root89Candidates4_checked : candidates edge 4 479 821 root89Tail5 winner.bound = root89Candidates4 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
theorem root89Step4 : scan edge (search edge 3) 4 [479] 479 root89Tail4 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail5 winner := by
  rw [root89Tail4, scan, if_neg (by decide), root89Candidates4_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail5
    (if 3 ≤ root89Candidates4.length then search edge 3 [479, 821] root89Candidates4 1300 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates5 : List Nat := [4733, 10859, 11351, 12539, 13721, 14813, 15761, 16811, 19793, 24623]
set_option maxRecDepth 8192 in
theorem root89Candidates5_checked : candidates edge 4 479 881 root89Tail6 winner.bound = root89Candidates5 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root89Child5 : search edge 3 [479, 881] root89Candidates5 1360 winner = winner := by decide
theorem root89Step5 : scan edge (search edge 3) 4 [479] 479 root89Tail5 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail6 winner := by
  rw [root89Tail5, scan, if_neg (by decide), root89Candidates5_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail6
    (if 3 ≤ root89Candidates5.length then search edge 3 [479, 881] root89Candidates5 1360 winner else winner) = _
  rw [if_pos (by decide), root89Child5]
def root89Candidates6 : List Nat := [7517, 23417]
set_option maxRecDepth 8192 in
theorem root89Candidates6_checked : candidates edge 4 479 971 root89Tail7 winner.bound = root89Candidates6 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
theorem root89Step6 : scan edge (search edge 3) 4 [479] 479 root89Tail6 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail7 winner := by
  rw [root89Tail6, scan, if_neg (by decide), root89Candidates6_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail7
    (if 3 ≤ root89Candidates6.length then search edge 3 [479, 971] root89Candidates6 1450 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates7 : List Nat := [2297, 6203]
set_option maxRecDepth 8192 in
theorem root89Candidates7_checked : candidates edge 4 479 1049 root89Tail8 winner.bound = root89Candidates7 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
theorem root89Step7 : scan edge (search edge 3) 4 [479] 479 root89Tail7 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail8 winner := by
  rw [root89Tail7, scan, if_neg (by decide), root89Candidates7_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail8
    (if 3 ≤ root89Candidates7.length then search edge 3 [479, 1049] root89Candidates7 1528 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates8 : List Nat := [6203, 11909, 15383]
set_option maxRecDepth 8192 in
theorem root89Candidates8_checked : candidates edge 4 479 1187 root89Tail9 winner.bound = root89Candidates8 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root89Child8 : search edge 3 [479, 1187] root89Candidates8 1666 winner = winner := by decide
theorem root89Step8 : scan edge (search edge 3) 4 [479] 479 root89Tail8 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail9 winner := by
  rw [root89Tail8, scan, if_neg (by decide), root89Candidates8_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail9
    (if 3 ≤ root89Candidates8.length then search edge 3 [479, 1187] root89Candidates8 1666 winner else winner) = _
  rw [if_pos (by decide), root89Child8]
def root89Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root89Candidates9_checked : candidates edge 4 479 1307 root89Tail10 winner.bound = root89Candidates9 := by
  unfold candidates
  have row : edge 1307 = (SortedNatTree.contains FastRows.row1307) := by
    change Data.row1307.contains = (SortedNatTree.contains FastRows.row1307)
    exact FastRows.row1307_checked.symm
  rw [row]
  decide
theorem root89Step9 : scan edge (search edge 3) 4 [479] 479 root89Tail9 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail10 winner := by
  rw [root89Tail9, scan, if_neg (by decide), root89Candidates9_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail10
    (if 3 ≤ root89Candidates9.length then search edge 3 [479, 1307] root89Candidates9 1786 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates10 : List Nat := [12041, 20123]
set_option maxRecDepth 8192 in
theorem root89Candidates10_checked : candidates edge 4 479 1559 root89Tail11 winner.bound = root89Candidates10 := by
  unfold candidates
  have row : edge 1559 = (SortedNatTree.contains FastRows.row1559) := by
    change Data.row1559.contains = (SortedNatTree.contains FastRows.row1559)
    exact FastRows.row1559_checked.symm
  rw [row]
  decide
theorem root89Step10 : scan edge (search edge 3) 4 [479] 479 root89Tail10 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail11 winner := by
  rw [root89Tail10, scan, if_neg (by decide), root89Candidates10_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail11
    (if 3 ≤ root89Candidates10.length then search edge 3 [479, 1559] root89Candidates10 2038 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root89Candidates11_checked : candidates edge 4 479 2297 root89Tail12 winner.bound = root89Candidates11 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
theorem root89Step11 : scan edge (search edge 3) 4 [479] 479 root89Tail11 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail12 winner := by
  rw [root89Tail11, scan, if_neg (by decide), root89Candidates11_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail12
    (if 3 ≤ root89Candidates11.length then search edge 3 [479, 2297] root89Candidates11 2776 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root89Candidates12_checked : candidates edge 4 479 3533 root89Tail13 winner.bound = root89Candidates12 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root89Step12 : scan edge (search edge 3) 4 [479] 479 root89Tail12 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail13 winner := by
  rw [root89Tail12, scan, if_neg (by decide), root89Candidates12_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail13
    (if 3 ≤ root89Candidates12.length then search edge 3 [479, 3533] root89Candidates12 4012 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root89Candidates13_checked : candidates edge 4 479 4337 root89Tail14 winner.bound = root89Candidates13 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root89Step13 : scan edge (search edge 3) 4 [479] 479 root89Tail13 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail14 winner := by
  rw [root89Tail13, scan, if_neg (by decide), root89Candidates13_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail14
    (if 3 ≤ root89Candidates13.length then search edge 3 [479, 4337] root89Candidates13 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates14 : List Nat := [6203]
set_option maxRecDepth 8192 in
theorem root89Candidates14_checked : candidates edge 4 479 4481 root89Tail15 winner.bound = root89Candidates14 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root89Step14 : scan edge (search edge 3) 4 [479] 479 root89Tail14 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail15 winner := by
  rw [root89Tail14, scan, if_neg (by decide), root89Candidates14_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail15
    (if 3 ≤ root89Candidates14.length then search edge 3 [479, 4481] root89Candidates14 4960 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates15 : List Nat := [7517]
set_option maxRecDepth 8192 in
theorem root89Candidates15_checked : candidates edge 4 479 4733 root89Tail16 winner.bound = root89Candidates15 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root89Step15 : scan edge (search edge 3) 4 [479] 479 root89Tail15 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail16 winner := by
  rw [root89Tail15, scan, if_neg (by decide), root89Candidates15_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail16
    (if 3 ≤ root89Candidates15.length then search edge 3 [479, 4733] root89Candidates15 5212 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates16 : List Nat := [14813, 17417]
set_option maxRecDepth 8192 in
theorem root89Candidates16_checked : candidates edge 4 479 4817 root89Tail17 winner.bound = root89Candidates16 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root89Step16 : scan edge (search edge 3) 4 [479] 479 root89Tail16 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail17 winner := by
  rw [root89Tail16, scan, if_neg (by decide), root89Candidates16_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail17
    (if 3 ≤ root89Candidates16.length then search edge 3 [479, 4817] root89Candidates16 5296 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates17 : List Nat := [9689]
set_option maxRecDepth 8192 in
theorem root89Candidates17_checked : candidates edge 4 479 5807 root89Tail18 winner.bound = root89Candidates17 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root89Step17 : scan edge (search edge 3) 4 [479] 479 root89Tail17 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail18 winner := by
  rw [root89Tail17, scan, if_neg (by decide), root89Candidates17_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail18
    (if 3 ≤ root89Candidates17.length then search edge 3 [479, 5807] root89Candidates17 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root89Candidates18 : List Nat := [16811]
set_option maxRecDepth 8192 in
theorem root89Candidates18_checked : candidates edge 4 479 6203 root89Tail19 winner.bound = root89Candidates18 := by
  unfold candidates
  have row : edge 6203 = (SortedNatTree.contains FastRows.row6203) := by
    change Data.row6203.contains = (SortedNatTree.contains FastRows.row6203)
    exact FastRows.row6203_checked.symm
  rw [row]
  decide
theorem root89Step18 : scan edge (search edge 3) 4 [479] 479 root89Tail18 winner =
    scan edge (search edge 3) 4 [479] 479 root89Tail19 winner := by
  rw [root89Tail18, scan, if_neg (by decide), root89Candidates18_checked]
  change scan edge (search edge 3) 4 [479] 479 root89Tail19
    (if 3 ≤ root89Candidates18.length then search edge 3 [479, 6203] root89Candidates18 6682 winner else winner) = _
  rw [if_neg (by decide)]
theorem root89Scan19 : scan edge (search edge 3) 4 [479] 479 root89Tail19 winner = winner := by rw [root89Tail19, scan, if_pos (by decide)]
theorem root89Scan18 : scan edge (search edge 3) 4 [479] 479 root89Tail18 winner = winner :=
  (root89Step18).trans root89Scan19
theorem root89Scan17 : scan edge (search edge 3) 4 [479] 479 root89Tail17 winner = winner :=
  (root89Step17).trans root89Scan18
theorem root89Scan16 : scan edge (search edge 3) 4 [479] 479 root89Tail16 winner = winner :=
  (root89Step16).trans root89Scan17
theorem root89Scan15 : scan edge (search edge 3) 4 [479] 479 root89Tail15 winner = winner :=
  (root89Step15).trans root89Scan16
theorem root89Scan14 : scan edge (search edge 3) 4 [479] 479 root89Tail14 winner = winner :=
  (root89Step14).trans root89Scan15
theorem root89Scan13 : scan edge (search edge 3) 4 [479] 479 root89Tail13 winner = winner :=
  (root89Step13).trans root89Scan14
theorem root89Scan12 : scan edge (search edge 3) 4 [479] 479 root89Tail12 winner = winner :=
  (root89Step12).trans root89Scan13
theorem root89Scan11 : scan edge (search edge 3) 4 [479] 479 root89Tail11 winner = winner :=
  (root89Step11).trans root89Scan12
theorem root89Scan10 : scan edge (search edge 3) 4 [479] 479 root89Tail10 winner = winner :=
  (root89Step10).trans root89Scan11
theorem root89Scan9 : scan edge (search edge 3) 4 [479] 479 root89Tail9 winner = winner :=
  (root89Step9).trans root89Scan10
theorem root89Scan8 : scan edge (search edge 3) 4 [479] 479 root89Tail8 winner = winner :=
  (root89Step8).trans root89Scan9
theorem root89Scan7 : scan edge (search edge 3) 4 [479] 479 root89Tail7 winner = winner :=
  (root89Step7).trans root89Scan8
theorem root89Scan6 : scan edge (search edge 3) 4 [479] 479 root89Tail6 winner = winner :=
  (root89Step6).trans root89Scan7
theorem root89Scan5 : scan edge (search edge 3) 4 [479] 479 root89Tail5 winner = winner :=
  (root89Step5).trans root89Scan6
theorem root89Scan4 : scan edge (search edge 3) 4 [479] 479 root89Tail4 winner = winner :=
  (root89Step4).trans root89Scan5
theorem root89Scan3 : scan edge (search edge 3) 4 [479] 479 root89Tail3 winner = winner :=
  (root89Step3).trans root89Scan4
theorem root89Scan2 : scan edge (search edge 3) 4 [479] 479 root89Tail2 winner = winner :=
  (root89Step2).trans root89Scan3
theorem root89Scan1 : scan edge (search edge 3) 4 [479] 479 root89Tail1 winner = winner :=
  (root89Step1).trans root89Scan2
theorem root89Scan0 : scan edge (search edge 3) 4 [479] 479 root89Tail0 winner = winner :=
  (root89Step0).trans root89Scan1
set_option maxRecDepth 4096 in
theorem root89Branch : search edge 4 [479] root89Candidates 479 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [479] 479 root89Tail0 winner = winner
  exact root89Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
