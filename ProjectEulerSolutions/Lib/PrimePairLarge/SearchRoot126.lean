import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root126Candidates : List Nat := Data.row727
set_option maxRecDepth 8192 in
theorem root126Candidates_checked : candidates edge 5 0 727 tail127 winner.bound = root126Candidates := by
  apply candidates_eq_row edge 5 0 727 tail127 root126Candidates winner.bound tail127_ordered
  · decide
  · decide
  · funext q
    rfl
def root126Tail54 : List Nat := []
def root126Tail53 : List Nat := 25183 :: root126Tail54
def root126Tail52 : List Nat := 25117 :: root126Tail53
def root126Tail51 : List Nat := 25033 :: root126Tail52
def root126Tail50 : List Nat := 20161 :: root126Tail51
def root126Tail49 : List Nat := 20143 :: root126Tail50
def root126Tail48 : List Nat := 19267 :: root126Tail49
def root126Tail47 : List Nat := 18553 :: root126Tail48
def root126Tail46 : List Nat := 18211 :: root126Tail47
def root126Tail45 : List Nat := 17851 :: root126Tail46
def root126Tail44 : List Nat := 16747 :: root126Tail45
def root126Tail43 : List Nat := 15493 :: root126Tail44
def root126Tail42 : List Nat := 14737 :: root126Tail43
def root126Tail41 : List Nat := 14347 :: root126Tail42
def root126Tail40 : List Nat := 13681 :: root126Tail41
def root126Tail39 : List Nat := 13477 :: root126Tail40
def root126Tail38 : List Nat := 13339 :: root126Tail39
def root126Tail37 : List Nat := 13093 :: root126Tail38
def root126Tail36 : List Nat := 12799 :: root126Tail37
def root126Tail35 : List Nat := 12421 :: root126Tail36
def root126Tail34 : List Nat := 10837 :: root126Tail35
def root126Tail33 : List Nat := 10369 :: root126Tail34
def root126Tail32 : List Nat := 10333 :: root126Tail33
def root126Tail31 : List Nat := 9817 :: root126Tail32
def root126Tail30 : List Nat := 9199 :: root126Tail31
def root126Tail29 : List Nat := 9187 :: root126Tail30
def root126Tail28 : List Nat := 9157 :: root126Tail29
def root126Tail27 : List Nat := 8563 :: root126Tail28
def root126Tail26 : List Nat := 8389 :: root126Tail27
def root126Tail25 : List Nat := 7639 :: root126Tail26
def root126Tail24 : List Nat := 6637 :: root126Tail25
def root126Tail23 : List Nat := 6427 :: root126Tail24
def root126Tail22 : List Nat := 6277 :: root126Tail23
def root126Tail21 : List Nat := 6121 :: root126Tail22
def root126Tail20 : List Nat := 5923 :: root126Tail21
def root126Tail19 : List Nat := 5857 :: root126Tail20
def root126Tail18 : List Nat := 5623 :: root126Tail19
def root126Tail17 : List Nat := 5503 :: root126Tail18
def root126Tail16 : List Nat := 5227 :: root126Tail17
def root126Tail15 : List Nat := 5209 :: root126Tail16
def root126Tail14 : List Nat := 5197 :: root126Tail15
def root126Tail13 : List Nat := 4327 :: root126Tail14
def root126Tail12 : List Nat := 3463 :: root126Tail13
def root126Tail11 : List Nat := 3001 :: root126Tail12
def root126Tail10 : List Nat := 2803 :: root126Tail11
def root126Tail9 : List Nat := 2689 :: root126Tail10
def root126Tail8 : List Nat := 2137 :: root126Tail9
def root126Tail7 : List Nat := 1867 :: root126Tail8
def root126Tail6 : List Nat := 1723 :: root126Tail7
def root126Tail5 : List Nat := 1669 :: root126Tail6
def root126Tail4 : List Nat := 1279 :: root126Tail5
def root126Tail3 : List Nat := 1237 :: root126Tail4
def root126Tail2 : List Nat := 1009 :: root126Tail3
def root126Tail1 : List Nat := 997 :: root126Tail2
def root126Tail0 : List Nat := 823 :: root126Tail1
def root126Candidates0 : List Nat := [5209, 9199]
set_option maxRecDepth 8192 in
theorem root126Candidates0_checked : candidates edge 4 727 823 root126Tail1 winner.bound = root126Candidates0 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
theorem root126Step0 : scan edge (search edge 3) 4 [727] 727 root126Tail0 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail1 winner := by
  rw [root126Tail0, scan, if_neg (by decide), root126Candidates0_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail1
    (if 3 ≤ root126Candidates0.length then search edge 3 [727, 823] root126Candidates0 1550 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates1 : List Nat := [10333, 15493]
set_option maxRecDepth 8192 in
theorem root126Candidates1_checked : candidates edge 4 727 997 root126Tail2 winner.bound = root126Candidates1 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
theorem root126Step1 : scan edge (search edge 3) 4 [727] 727 root126Tail1 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail2 winner := by
  rw [root126Tail1, scan, if_neg (by decide), root126Candidates1_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail2
    (if 3 ≤ root126Candidates1.length then search edge 3 [727, 997] root126Candidates1 1724 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates2 : List Nat := [20161]
set_option maxRecDepth 8192 in
theorem root126Candidates2_checked : candidates edge 4 727 1009 root126Tail3 winner.bound = root126Candidates2 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
theorem root126Step2 : scan edge (search edge 3) 4 [727] 727 root126Tail2 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail3 winner := by
  rw [root126Tail2, scan, if_neg (by decide), root126Candidates2_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail3
    (if 3 ≤ root126Candidates2.length then search edge 3 [727, 1009] root126Candidates2 1736 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates3 : List Nat := [9817]
set_option maxRecDepth 8192 in
theorem root126Candidates3_checked : candidates edge 4 727 1237 root126Tail4 winner.bound = root126Candidates3 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
theorem root126Step3 : scan edge (search edge 3) 4 [727] 727 root126Tail3 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail4 winner := by
  rw [root126Tail3, scan, if_neg (by decide), root126Candidates3_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail4
    (if 3 ≤ root126Candidates3.length then search edge 3 [727, 1237] root126Candidates3 1964 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates4 : List Nat := [2803, 3463]
set_option maxRecDepth 8192 in
theorem root126Candidates4_checked : candidates edge 4 727 1279 root126Tail5 winner.bound = root126Candidates4 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
theorem root126Step4 : scan edge (search edge 3) 4 [727] 727 root126Tail4 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail5 winner := by
  rw [root126Tail4, scan, if_neg (by decide), root126Candidates4_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail5
    (if 3 ≤ root126Candidates4.length then search edge 3 [727, 1279] root126Candidates4 2006 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates5_checked : candidates edge 4 727 1669 root126Tail6 winner.bound = root126Candidates5 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
theorem root126Step5 : scan edge (search edge 3) 4 [727] 727 root126Tail5 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail6 winner := by
  rw [root126Tail5, scan, if_neg (by decide), root126Candidates5_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail6
    (if 3 ≤ root126Candidates5.length then search edge 3 [727, 1669] root126Candidates5 2396 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates6 : List Nat := [8563]
set_option maxRecDepth 8192 in
theorem root126Candidates6_checked : candidates edge 4 727 1723 root126Tail7 winner.bound = root126Candidates6 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
theorem root126Step6 : scan edge (search edge 3) 4 [727] 727 root126Tail6 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail7 winner := by
  rw [root126Tail6, scan, if_neg (by decide), root126Candidates6_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail7
    (if 3 ≤ root126Candidates6.length then search edge 3 [727, 1723] root126Candidates6 2450 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates7 : List Nat := [6121, 7639]
set_option maxRecDepth 8192 in
theorem root126Candidates7_checked : candidates edge 4 727 1867 root126Tail8 winner.bound = root126Candidates7 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
theorem root126Step7 : scan edge (search edge 3) 4 [727] 727 root126Tail7 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail8 winner := by
  rw [root126Tail7, scan, if_neg (by decide), root126Candidates7_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail8
    (if 3 ≤ root126Candidates7.length then search edge 3 [727, 1867] root126Candidates7 2594 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates8 : List Nat := [5503, 9157]
set_option maxRecDepth 8192 in
theorem root126Candidates8_checked : candidates edge 4 727 2137 root126Tail9 winner.bound = root126Candidates8 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
theorem root126Step8 : scan edge (search edge 3) 4 [727] 727 root126Tail8 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail9 winner := by
  rw [root126Tail8, scan, if_neg (by decide), root126Candidates8_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail9
    (if 3 ≤ root126Candidates8.length then search edge 3 [727, 2137] root126Candidates8 2864 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates9_checked : candidates edge 4 727 2689 root126Tail10 winner.bound = root126Candidates9 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root126Step9 : scan edge (search edge 3) 4 [727] 727 root126Tail9 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail10 winner := by
  rw [root126Tail9, scan, if_neg (by decide), root126Candidates9_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail10
    (if 3 ≤ root126Candidates9.length then search edge 3 [727, 2689] root126Candidates9 3416 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates10_checked : candidates edge 4 727 2803 root126Tail11 winner.bound = root126Candidates10 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root126Step10 : scan edge (search edge 3) 4 [727] 727 root126Tail10 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail11 winner := by
  rw [root126Tail10, scan, if_neg (by decide), root126Candidates10_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail11
    (if 3 ≤ root126Candidates10.length then search edge 3 [727, 2803] root126Candidates10 3530 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates11 : List Nat := [6121]
set_option maxRecDepth 8192 in
theorem root126Candidates11_checked : candidates edge 4 727 3001 root126Tail12 winner.bound = root126Candidates11 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
theorem root126Step11 : scan edge (search edge 3) 4 [727] 727 root126Tail11 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail12 winner := by
  rw [root126Tail11, scan, if_neg (by decide), root126Candidates11_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail12
    (if 3 ≤ root126Candidates11.length then search edge 3 [727, 3001] root126Candidates11 3728 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates12_checked : candidates edge 4 727 3463 root126Tail13 winner.bound = root126Candidates12 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root126Step12 : scan edge (search edge 3) 4 [727] 727 root126Tail12 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail13 winner := by
  rw [root126Tail12, scan, if_neg (by decide), root126Candidates12_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail13
    (if 3 ≤ root126Candidates12.length then search edge 3 [727, 3463] root126Candidates12 4190 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates13 : List Nat := [5503, 7639]
set_option maxRecDepth 8192 in
theorem root126Candidates13_checked : candidates edge 4 727 4327 root126Tail14 winner.bound = root126Candidates13 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root126Step13 : scan edge (search edge 3) 4 [727] 727 root126Tail13 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail14 winner := by
  rw [root126Tail13, scan, if_neg (by decide), root126Candidates13_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail14
    (if 3 ≤ root126Candidates13.length then search edge 3 [727, 4327] root126Candidates13 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates14 : List Nat := [8389, 12421, 18553]
set_option maxRecDepth 8192 in
theorem root126Candidates14_checked : candidates edge 4 727 5197 root126Tail15 winner.bound = root126Candidates14 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root126Child14 : search edge 3 [727, 5197] root126Candidates14 5924 winner = winner := by decide
theorem root126Step14 : scan edge (search edge 3) 4 [727] 727 root126Tail14 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail15 winner := by
  rw [root126Tail14, scan, if_neg (by decide), root126Candidates14_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail15
    (if 3 ≤ root126Candidates14.length then search edge 3 [727, 5197] root126Candidates14 5924 winner else winner) = _
  rw [if_pos (by decide), root126Child14]
def root126Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates15_checked : candidates edge 4 727 5209 root126Tail16 winner.bound = root126Candidates15 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root126Step15 : scan edge (search edge 3) 4 [727] 727 root126Tail15 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail16 winner := by
  rw [root126Tail15, scan, if_neg (by decide), root126Candidates15_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail16
    (if 3 ≤ root126Candidates15.length then search edge 3 [727, 5209] root126Candidates15 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates16 : List Nat := [9817]
set_option maxRecDepth 8192 in
theorem root126Candidates16_checked : candidates edge 4 727 5227 root126Tail17 winner.bound = root126Candidates16 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root126Step16 : scan edge (search edge 3) 4 [727] 727 root126Tail16 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail17 winner := by
  rw [root126Tail16, scan, if_neg (by decide), root126Candidates16_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail17
    (if 3 ≤ root126Candidates16.length then search edge 3 [727, 5227] root126Candidates16 5954 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates17_checked : candidates edge 4 727 5503 root126Tail18 winner.bound = root126Candidates17 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root126Step17 : scan edge (search edge 3) 4 [727] 727 root126Tail17 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail18 winner := by
  rw [root126Tail17, scan, if_neg (by decide), root126Candidates17_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail18
    (if 3 ≤ root126Candidates17.length then search edge 3 [727, 5503] root126Candidates17 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates18 : List Nat := [7639, 10369]
set_option maxRecDepth 8192 in
theorem root126Candidates18_checked : candidates edge 4 727 5623 root126Tail19 winner.bound = root126Candidates18 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root126Step18 : scan edge (search edge 3) 4 [727] 727 root126Tail18 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail19 winner := by
  rw [root126Tail18, scan, if_neg (by decide), root126Candidates18_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail19
    (if 3 ≤ root126Candidates18.length then search edge 3 [727, 5623] root126Candidates18 6350 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root126Candidates19_checked : candidates edge 4 727 5857 root126Tail20 winner.bound = root126Candidates19 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root126Step19 : scan edge (search edge 3) 4 [727] 727 root126Tail19 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail20 winner := by
  rw [root126Tail19, scan, if_neg (by decide), root126Candidates19_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail20
    (if 3 ≤ root126Candidates19.length then search edge 3 [727, 5857] root126Candidates19 6584 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates20 : List Nat := [6427, 18553]
set_option maxRecDepth 8192 in
theorem root126Candidates20_checked : candidates edge 4 727 5923 root126Tail21 winner.bound = root126Candidates20 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
theorem root126Step20 : scan edge (search edge 3) 4 [727] 727 root126Tail20 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail21 winner := by
  rw [root126Tail20, scan, if_neg (by decide), root126Candidates20_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail21
    (if 3 ≤ root126Candidates20.length then search edge 3 [727, 5923] root126Candidates20 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates21 : List Nat := [8389]
set_option maxRecDepth 8192 in
theorem root126Candidates21_checked : candidates edge 4 727 6121 root126Tail22 winner.bound = root126Candidates21 := by
  unfold candidates
  have row : edge 6121 = (SortedNatTree.contains FastRows.row6121) := by
    change Data.row6121.contains = (SortedNatTree.contains FastRows.row6121)
    exact FastRows.row6121_checked.symm
  rw [row]
  decide
theorem root126Step21 : scan edge (search edge 3) 4 [727] 727 root126Tail21 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail22 winner := by
  rw [root126Tail21, scan, if_neg (by decide), root126Candidates21_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail22
    (if 3 ≤ root126Candidates21.length then search edge 3 [727, 6121] root126Candidates21 6848 winner else winner) = _
  rw [if_neg (by decide)]
def root126Candidates22 : List Nat := [8563, 13477, 15493]
set_option maxRecDepth 8192 in
theorem root126Candidates22_checked : candidates edge 4 727 6277 root126Tail23 winner.bound = root126Candidates22 := by
  unfold candidates
  have row : edge 6277 = (SortedNatTree.contains FastRows.row6277) := by
    change Data.row6277.contains = (SortedNatTree.contains FastRows.row6277)
    exact FastRows.row6277_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root126Child22 : search edge 3 [727, 6277] root126Candidates22 7004 winner = winner := by decide
theorem root126Step22 : scan edge (search edge 3) 4 [727] 727 root126Tail22 winner =
    scan edge (search edge 3) 4 [727] 727 root126Tail23 winner := by
  rw [root126Tail22, scan, if_neg (by decide), root126Candidates22_checked]
  change scan edge (search edge 3) 4 [727] 727 root126Tail23
    (if 3 ≤ root126Candidates22.length then search edge 3 [727, 6277] root126Candidates22 7004 winner else winner) = _
  rw [if_pos (by decide), root126Child22]
theorem root126Scan23 : scan edge (search edge 3) 4 [727] 727 root126Tail23 winner = winner := by rw [root126Tail23, scan, if_pos (by decide)]
theorem root126Scan22 : scan edge (search edge 3) 4 [727] 727 root126Tail22 winner = winner :=
  (root126Step22).trans root126Scan23
theorem root126Scan21 : scan edge (search edge 3) 4 [727] 727 root126Tail21 winner = winner :=
  (root126Step21).trans root126Scan22
theorem root126Scan20 : scan edge (search edge 3) 4 [727] 727 root126Tail20 winner = winner :=
  (root126Step20).trans root126Scan21
theorem root126Scan19 : scan edge (search edge 3) 4 [727] 727 root126Tail19 winner = winner :=
  (root126Step19).trans root126Scan20
theorem root126Scan18 : scan edge (search edge 3) 4 [727] 727 root126Tail18 winner = winner :=
  (root126Step18).trans root126Scan19
theorem root126Scan17 : scan edge (search edge 3) 4 [727] 727 root126Tail17 winner = winner :=
  (root126Step17).trans root126Scan18
theorem root126Scan16 : scan edge (search edge 3) 4 [727] 727 root126Tail16 winner = winner :=
  (root126Step16).trans root126Scan17
theorem root126Scan15 : scan edge (search edge 3) 4 [727] 727 root126Tail15 winner = winner :=
  (root126Step15).trans root126Scan16
theorem root126Scan14 : scan edge (search edge 3) 4 [727] 727 root126Tail14 winner = winner :=
  (root126Step14).trans root126Scan15
theorem root126Scan13 : scan edge (search edge 3) 4 [727] 727 root126Tail13 winner = winner :=
  (root126Step13).trans root126Scan14
theorem root126Scan12 : scan edge (search edge 3) 4 [727] 727 root126Tail12 winner = winner :=
  (root126Step12).trans root126Scan13
theorem root126Scan11 : scan edge (search edge 3) 4 [727] 727 root126Tail11 winner = winner :=
  (root126Step11).trans root126Scan12
theorem root126Scan10 : scan edge (search edge 3) 4 [727] 727 root126Tail10 winner = winner :=
  (root126Step10).trans root126Scan11
theorem root126Scan9 : scan edge (search edge 3) 4 [727] 727 root126Tail9 winner = winner :=
  (root126Step9).trans root126Scan10
theorem root126Scan8 : scan edge (search edge 3) 4 [727] 727 root126Tail8 winner = winner :=
  (root126Step8).trans root126Scan9
theorem root126Scan7 : scan edge (search edge 3) 4 [727] 727 root126Tail7 winner = winner :=
  (root126Step7).trans root126Scan8
theorem root126Scan6 : scan edge (search edge 3) 4 [727] 727 root126Tail6 winner = winner :=
  (root126Step6).trans root126Scan7
theorem root126Scan5 : scan edge (search edge 3) 4 [727] 727 root126Tail5 winner = winner :=
  (root126Step5).trans root126Scan6
theorem root126Scan4 : scan edge (search edge 3) 4 [727] 727 root126Tail4 winner = winner :=
  (root126Step4).trans root126Scan5
theorem root126Scan3 : scan edge (search edge 3) 4 [727] 727 root126Tail3 winner = winner :=
  (root126Step3).trans root126Scan4
theorem root126Scan2 : scan edge (search edge 3) 4 [727] 727 root126Tail2 winner = winner :=
  (root126Step2).trans root126Scan3
theorem root126Scan1 : scan edge (search edge 3) 4 [727] 727 root126Tail1 winner = winner :=
  (root126Step1).trans root126Scan2
theorem root126Scan0 : scan edge (search edge 3) 4 [727] 727 root126Tail0 winner = winner :=
  (root126Step0).trans root126Scan1
set_option maxRecDepth 4096 in
theorem root126Branch : search edge 4 [727] root126Candidates 727 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [727] 727 root126Tail0 winner = winner
  exact root126Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
