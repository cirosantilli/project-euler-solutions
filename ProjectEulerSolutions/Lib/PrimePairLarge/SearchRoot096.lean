import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root96Candidates : List Nat := Data.row523
set_option maxRecDepth 8192 in
theorem root96Candidates_checked : candidates edge 5 0 523 tail97 winner.bound = root96Candidates := by
  apply candidates_eq_row edge 5 0 523 tail97 root96Candidates winner.bound tail97_ordered
  · decide
  · decide
  · funext q
    rfl
def root96Tail61 : List Nat := []
def root96Tail60 : List Nat := 25111 :: root96Tail61
def root96Tail59 : List Nat := 24229 :: root96Tail60
def root96Tail58 : List Nat := 23911 :: root96Tail59
def root96Tail57 : List Nat := 23893 :: root96Tail58
def root96Tail56 : List Nat := 23857 :: root96Tail57
def root96Tail55 : List Nat := 23053 :: root96Tail56
def root96Tail54 : List Nat := 23041 :: root96Tail55
def root96Tail53 : List Nat := 22189 :: root96Tail54
def root96Tail52 : List Nat := 21559 :: root96Tail53
def root96Tail51 : List Nat := 21319 :: root96Tail52
def root96Tail50 : List Nat := 21001 :: root96Tail51
def root96Tail49 : List Nat := 20161 :: root96Tail50
def root96Tail48 : List Nat := 20029 :: root96Tail49
def root96Tail47 : List Nat := 19417 :: root96Tail48
def root96Tail46 : List Nat := 19249 :: root96Tail47
def root96Tail45 : List Nat := 18439 :: root96Tail46
def root96Tail44 : List Nat := 17737 :: root96Tail45
def root96Tail43 : List Nat := 17623 :: root96Tail44
def root96Tail42 : List Nat := 17497 :: root96Tail43
def root96Tail41 : List Nat := 17053 :: root96Tail42
def root96Tail40 : List Nat := 16921 :: root96Tail41
def root96Tail39 : List Nat := 16057 :: root96Tail40
def root96Tail38 : List Nat := 15787 :: root96Tail39
def root96Tail37 : List Nat := 15331 :: root96Tail38
def root96Tail36 : List Nat := 15061 :: root96Tail37
def root96Tail35 : List Nat := 14737 :: root96Tail36
def root96Tail34 : List Nat := 13411 :: root96Tail35
def root96Tail33 : List Nat := 13381 :: root96Tail34
def root96Tail32 : List Nat := 13309 :: root96Tail33
def root96Tail31 : List Nat := 12343 :: root96Tail32
def root96Tail30 : List Nat := 12049 :: root96Tail31
def root96Tail29 : List Nat := 11239 :: root96Tail30
def root96Tail28 : List Nat := 11197 :: root96Tail29
def root96Tail27 : List Nat := 10891 :: root96Tail28
def root96Tail26 : List Nat := 10837 :: root96Tail27
def root96Tail25 : List Nat := 10459 :: root96Tail26
def root96Tail24 : List Nat := 9187 :: root96Tail25
def root96Tail23 : List Nat := 8581 :: root96Tail24
def root96Tail22 : List Nat := 7753 :: root96Tail23
def root96Tail21 : List Nat := 7237 :: root96Tail22
def root96Tail20 : List Nat := 6967 :: root96Tail21
def root96Tail19 : List Nat := 6367 :: root96Tail20
def root96Tail18 : List Nat := 5851 :: root96Tail19
def root96Tail17 : List Nat := 5233 :: root96Tail18
def root96Tail16 : List Nat := 4987 :: root96Tail17
def root96Tail15 : List Nat := 4909 :: root96Tail16
def root96Tail14 : List Nat := 4591 :: root96Tail15
def root96Tail13 : List Nat := 3919 :: root96Tail14
def root96Tail12 : List Nat := 3499 :: root96Tail13
def root96Tail11 : List Nat := 3469 :: root96Tail12
def root96Tail10 : List Nat := 3307 :: root96Tail11
def root96Tail9 : List Nat := 2677 :: root96Tail10
def root96Tail8 : List Nat := 2671 :: root96Tail9
def root96Tail7 : List Nat := 2113 :: root96Tail8
def root96Tail6 : List Nat := 2089 :: root96Tail7
def root96Tail5 : List Nat := 1987 :: root96Tail6
def root96Tail4 : List Nat := 1663 :: root96Tail5
def root96Tail3 : List Nat := 1297 :: root96Tail4
def root96Tail2 : List Nat := 1129 :: root96Tail3
def root96Tail1 : List Nat := 577 :: root96Tail2
def root96Tail0 : List Nat := 541 :: root96Tail1
def root96Candidates0 : List Nat := [17053]
set_option maxRecDepth 8192 in
theorem root96Candidates0_checked : candidates edge 4 523 541 root96Tail1 winner.bound = root96Candidates0 := by
  unfold candidates
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row]
  decide
theorem root96Step0 : scan edge (search edge 3) 4 [523] 523 root96Tail0 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail1 winner := by
  rw [root96Tail0, scan, if_neg (by decide), root96Candidates0_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail1
    (if 3 ≤ root96Candidates0.length then search edge 3 [523, 541] root96Candidates0 1064 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates1 : List Nat := [3919]
set_option maxRecDepth 8192 in
theorem root96Candidates1_checked : candidates edge 4 523 577 root96Tail2 winner.bound = root96Candidates1 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
theorem root96Step1 : scan edge (search edge 3) 4 [523] 523 root96Tail1 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail2 winner := by
  rw [root96Tail1, scan, if_neg (by decide), root96Candidates1_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail2
    (if 3 ≤ root96Candidates1.length then search edge 3 [523, 577] root96Candidates1 1100 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates2 : List Nat := [13309, 22189, 23053]
set_option maxRecDepth 8192 in
theorem root96Candidates2_checked : candidates edge 4 523 1129 root96Tail3 winner.bound = root96Candidates2 := by
  unfold candidates
  have row : edge 1129 = (SortedNatTree.contains FastRows.row1129) := by
    change Data.row1129.contains = (SortedNatTree.contains FastRows.row1129)
    exact FastRows.row1129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root96Child2 : search edge 3 [523, 1129] root96Candidates2 1652 winner = winner := by decide
theorem root96Step2 : scan edge (search edge 3) 4 [523] 523 root96Tail2 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail3 winner := by
  rw [root96Tail2, scan, if_neg (by decide), root96Candidates2_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail3
    (if 3 ≤ root96Candidates2.length then search edge 3 [523, 1129] root96Candidates2 1652 winner else winner) = _
  rw [if_pos (by decide), root96Child2]
def root96Candidates3 : List Nat := [2671, 12049, 17053]
set_option maxRecDepth 8192 in
theorem root96Candidates3_checked : candidates edge 4 523 1297 root96Tail4 winner.bound = root96Candidates3 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root96Child3 : search edge 3 [523, 1297] root96Candidates3 1820 winner = winner := by decide
theorem root96Step3 : scan edge (search edge 3) 4 [523] 523 root96Tail3 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail4 winner := by
  rw [root96Tail3, scan, if_neg (by decide), root96Candidates3_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail4
    (if 3 ≤ root96Candidates3.length then search edge 3 [523, 1297] root96Candidates3 1820 winner else winner) = _
  rw [if_pos (by decide), root96Child3]
def root96Candidates4 : List Nat := [4987]
set_option maxRecDepth 8192 in
theorem root96Candidates4_checked : candidates edge 4 523 1663 root96Tail5 winner.bound = root96Candidates4 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
theorem root96Step4 : scan edge (search edge 3) 4 [523] 523 root96Tail4 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail5 winner := by
  rw [root96Tail4, scan, if_neg (by decide), root96Candidates4_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail5
    (if 3 ≤ root96Candidates4.length then search edge 3 [523, 1663] root96Candidates4 2186 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates5 : List Nat := [3499, 7237]
set_option maxRecDepth 8192 in
theorem root96Candidates5_checked : candidates edge 4 523 1987 root96Tail6 winner.bound = root96Candidates5 := by
  unfold candidates
  have row : edge 1987 = (SortedNatTree.contains FastRows.row1987) := by
    change Data.row1987.contains = (SortedNatTree.contains FastRows.row1987)
    exact FastRows.row1987_checked.symm
  rw [row]
  decide
theorem root96Step5 : scan edge (search edge 3) 4 [523] 523 root96Tail5 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail6 winner := by
  rw [root96Tail5, scan, if_neg (by decide), root96Candidates5_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail6
    (if 3 ≤ root96Candidates5.length then search edge 3 [523, 1987] root96Candidates5 2510 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates6 : List Nat := [7237, 23041]
set_option maxRecDepth 8192 in
theorem root96Candidates6_checked : candidates edge 4 523 2089 root96Tail7 winner.bound = root96Candidates6 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
theorem root96Step6 : scan edge (search edge 3) 4 [523] 523 root96Tail6 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail7 winner := by
  rw [root96Tail6, scan, if_neg (by decide), root96Candidates6_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail7
    (if 3 ≤ root96Candidates6.length then search edge 3 [523, 2089] root96Candidates6 2612 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates7 : List Nat := [11239]
set_option maxRecDepth 8192 in
theorem root96Candidates7_checked : candidates edge 4 523 2113 root96Tail8 winner.bound = root96Candidates7 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root96Step7 : scan edge (search edge 3) 4 [523] 523 root96Tail7 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail8 winner := by
  rw [root96Tail7, scan, if_neg (by decide), root96Candidates7_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail8
    (if 3 ≤ root96Candidates7.length then search edge 3 [523, 2113] root96Candidates7 2636 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates8 : List Nat := [17623, 21319]
set_option maxRecDepth 8192 in
theorem root96Candidates8_checked : candidates edge 4 523 2671 root96Tail9 winner.bound = root96Candidates8 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root96Step8 : scan edge (search edge 3) 4 [523] 523 root96Tail8 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail9 winner := by
  rw [root96Tail8, scan, if_neg (by decide), root96Candidates8_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail9
    (if 3 ≤ root96Candidates8.length then search edge 3 [523, 2671] root96Candidates8 3194 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates9 : List Nat := [3469, 5851, 17623]
set_option maxRecDepth 8192 in
theorem root96Candidates9_checked : candidates edge 4 523 2677 root96Tail10 winner.bound = root96Candidates9 := by
  unfold candidates
  have row : edge 2677 = (SortedNatTree.contains FastRows.row2677) := by
    change Data.row2677.contains = (SortedNatTree.contains FastRows.row2677)
    exact FastRows.row2677_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root96Child9 : search edge 3 [523, 2677] root96Candidates9 3200 winner = winner := by decide
theorem root96Step9 : scan edge (search edge 3) 4 [523] 523 root96Tail9 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail10 winner := by
  rw [root96Tail9, scan, if_neg (by decide), root96Candidates9_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail10
    (if 3 ≤ root96Candidates9.length then search edge 3 [523, 2677] root96Candidates9 3200 winner else winner) = _
  rw [if_pos (by decide), root96Child9]
def root96Candidates10 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root96Candidates10_checked : candidates edge 4 523 3307 root96Tail11 winner.bound = root96Candidates10 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root96Step10 : scan edge (search edge 3) 4 [523] 523 root96Tail10 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail11 winner := by
  rw [root96Tail10, scan, if_neg (by decide), root96Candidates10_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail11
    (if 3 ≤ root96Candidates10.length then search edge 3 [523, 3307] root96Candidates10 3830 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root96Candidates11_checked : candidates edge 4 523 3469 root96Tail12 winner.bound = root96Candidates11 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root96Step11 : scan edge (search edge 3) 4 [523] 523 root96Tail11 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail12 winner := by
  rw [root96Tail11, scan, if_neg (by decide), root96Candidates11_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail12
    (if 3 ≤ root96Candidates11.length then search edge 3 [523, 3469] root96Candidates11 3992 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates12 : List Nat := [15331]
set_option maxRecDepth 8192 in
theorem root96Candidates12_checked : candidates edge 4 523 3499 root96Tail13 winner.bound = root96Candidates12 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root96Step12 : scan edge (search edge 3) 4 [523] 523 root96Tail12 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail13 winner := by
  rw [root96Tail12, scan, if_neg (by decide), root96Candidates12_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail13
    (if 3 ≤ root96Candidates12.length then search edge 3 [523, 3499] root96Candidates12 4022 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates13 : List Nat := [4591]
set_option maxRecDepth 8192 in
theorem root96Candidates13_checked : candidates edge 4 523 3919 root96Tail14 winner.bound = root96Candidates13 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root96Step13 : scan edge (search edge 3) 4 [523] 523 root96Tail13 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail14 winner := by
  rw [root96Tail13, scan, if_neg (by decide), root96Candidates13_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail14
    (if 3 ≤ root96Candidates13.length then search edge 3 [523, 3919] root96Candidates13 4442 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates14 : List Nat := [11239]
set_option maxRecDepth 8192 in
theorem root96Candidates14_checked : candidates edge 4 523 4591 root96Tail15 winner.bound = root96Candidates14 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root96Step14 : scan edge (search edge 3) 4 [523] 523 root96Tail14 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail15 winner := by
  rw [root96Tail14, scan, if_neg (by decide), root96Candidates14_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail15
    (if 3 ≤ root96Candidates14.length then search edge 3 [523, 4591] root96Candidates14 5114 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates15 : List Nat := [10837]
set_option maxRecDepth 8192 in
theorem root96Candidates15_checked : candidates edge 4 523 4909 root96Tail16 winner.bound = root96Candidates15 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root96Step15 : scan edge (search edge 3) 4 [523] 523 root96Tail15 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail16 winner := by
  rw [root96Tail15, scan, if_neg (by decide), root96Candidates15_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail16
    (if 3 ≤ root96Candidates15.length then search edge 3 [523, 4909] root96Candidates15 5432 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root96Candidates16_checked : candidates edge 4 523 4987 root96Tail17 winner.bound = root96Candidates16 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root96Step16 : scan edge (search edge 3) 4 [523] 523 root96Tail16 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail17 winner := by
  rw [root96Tail16, scan, if_neg (by decide), root96Candidates16_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail17
    (if 3 ≤ root96Candidates16.length then search edge 3 [523, 4987] root96Candidates16 5510 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates17 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root96Candidates17_checked : candidates edge 4 523 5233 root96Tail18 winner.bound = root96Candidates17 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root96Step17 : scan edge (search edge 3) 4 [523] 523 root96Tail17 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail18 winner := by
  rw [root96Tail17, scan, if_neg (by decide), root96Candidates17_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail18
    (if 3 ≤ root96Candidates17.length then search edge 3 [523, 5233] root96Candidates17 5756 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates18 : List Nat := [10837]
set_option maxRecDepth 8192 in
theorem root96Candidates18_checked : candidates edge 4 523 5851 root96Tail19 winner.bound = root96Candidates18 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root96Step18 : scan edge (search edge 3) 4 [523] 523 root96Tail18 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail19 winner := by
  rw [root96Tail18, scan, if_neg (by decide), root96Candidates18_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail19
    (if 3 ≤ root96Candidates18.length then search edge 3 [523, 5851] root96Candidates18 6374 winner else winner) = _
  rw [if_neg (by decide)]
def root96Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root96Candidates19_checked : candidates edge 4 523 6367 root96Tail20 winner.bound = root96Candidates19 := by
  unfold candidates
  have row : edge 6367 = (SortedNatTree.contains FastRows.row6367) := by
    change Data.row6367.contains = (SortedNatTree.contains FastRows.row6367)
    exact FastRows.row6367_checked.symm
  rw [row]
  decide
theorem root96Step19 : scan edge (search edge 3) 4 [523] 523 root96Tail19 winner =
    scan edge (search edge 3) 4 [523] 523 root96Tail20 winner := by
  rw [root96Tail19, scan, if_neg (by decide), root96Candidates19_checked]
  change scan edge (search edge 3) 4 [523] 523 root96Tail20
    (if 3 ≤ root96Candidates19.length then search edge 3 [523, 6367] root96Candidates19 6890 winner else winner) = _
  rw [if_neg (by decide)]
theorem root96Scan20 : scan edge (search edge 3) 4 [523] 523 root96Tail20 winner = winner := by rw [root96Tail20, scan, if_pos (by decide)]
theorem root96Scan19 : scan edge (search edge 3) 4 [523] 523 root96Tail19 winner = winner :=
  (root96Step19).trans root96Scan20
theorem root96Scan18 : scan edge (search edge 3) 4 [523] 523 root96Tail18 winner = winner :=
  (root96Step18).trans root96Scan19
theorem root96Scan17 : scan edge (search edge 3) 4 [523] 523 root96Tail17 winner = winner :=
  (root96Step17).trans root96Scan18
theorem root96Scan16 : scan edge (search edge 3) 4 [523] 523 root96Tail16 winner = winner :=
  (root96Step16).trans root96Scan17
theorem root96Scan15 : scan edge (search edge 3) 4 [523] 523 root96Tail15 winner = winner :=
  (root96Step15).trans root96Scan16
theorem root96Scan14 : scan edge (search edge 3) 4 [523] 523 root96Tail14 winner = winner :=
  (root96Step14).trans root96Scan15
theorem root96Scan13 : scan edge (search edge 3) 4 [523] 523 root96Tail13 winner = winner :=
  (root96Step13).trans root96Scan14
theorem root96Scan12 : scan edge (search edge 3) 4 [523] 523 root96Tail12 winner = winner :=
  (root96Step12).trans root96Scan13
theorem root96Scan11 : scan edge (search edge 3) 4 [523] 523 root96Tail11 winner = winner :=
  (root96Step11).trans root96Scan12
theorem root96Scan10 : scan edge (search edge 3) 4 [523] 523 root96Tail10 winner = winner :=
  (root96Step10).trans root96Scan11
theorem root96Scan9 : scan edge (search edge 3) 4 [523] 523 root96Tail9 winner = winner :=
  (root96Step9).trans root96Scan10
theorem root96Scan8 : scan edge (search edge 3) 4 [523] 523 root96Tail8 winner = winner :=
  (root96Step8).trans root96Scan9
theorem root96Scan7 : scan edge (search edge 3) 4 [523] 523 root96Tail7 winner = winner :=
  (root96Step7).trans root96Scan8
theorem root96Scan6 : scan edge (search edge 3) 4 [523] 523 root96Tail6 winner = winner :=
  (root96Step6).trans root96Scan7
theorem root96Scan5 : scan edge (search edge 3) 4 [523] 523 root96Tail5 winner = winner :=
  (root96Step5).trans root96Scan6
theorem root96Scan4 : scan edge (search edge 3) 4 [523] 523 root96Tail4 winner = winner :=
  (root96Step4).trans root96Scan5
theorem root96Scan3 : scan edge (search edge 3) 4 [523] 523 root96Tail3 winner = winner :=
  (root96Step3).trans root96Scan4
theorem root96Scan2 : scan edge (search edge 3) 4 [523] 523 root96Tail2 winner = winner :=
  (root96Step2).trans root96Scan3
theorem root96Scan1 : scan edge (search edge 3) 4 [523] 523 root96Tail1 winner = winner :=
  (root96Step1).trans root96Scan2
theorem root96Scan0 : scan edge (search edge 3) 4 [523] 523 root96Tail0 winner = winner :=
  (root96Step0).trans root96Scan1
set_option maxRecDepth 4096 in
theorem root96Branch : search edge 4 [523] root96Candidates 523 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [523] 523 root96Tail0 winner = winner
  exact root96Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
