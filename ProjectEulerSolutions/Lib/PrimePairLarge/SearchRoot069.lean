import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root69Candidates : List Nat := Data.row359
set_option maxRecDepth 8192 in
theorem root69Candidates_checked : candidates edge 5 0 359 tail70 winner.bound = root69Candidates := by
  apply candidates_eq_row edge 5 0 359 tail70 root69Candidates winner.bound tail70_ordered
  · decide
  · decide
  · funext q
    rfl
def root69Tail67 : List Nat := []
def root69Tail66 : List Nat := 25367 :: root69Tail67
def root69Tail65 : List Nat := 24683 :: root69Tail66
def root69Tail64 : List Nat := 24671 :: root69Tail65
def root69Tail63 : List Nat := 23981 :: root69Tail64
def root69Tail62 : List Nat := 23819 :: root69Tail63
def root69Tail61 : List Nat := 23297 :: root69Tail62
def root69Tail60 : List Nat := 23201 :: root69Tail61
def root69Tail59 : List Nat := 22391 :: root69Tail60
def root69Tail58 : List Nat := 22193 :: root69Tail59
def root69Tail57 : List Nat := 22013 :: root69Tail58
def root69Tail56 : List Nat := 21881 :: root69Tail57
def root69Tail55 : List Nat := 21503 :: root69Tail56
def root69Tail54 : List Nat := 21269 :: root69Tail55
def root69Tail53 : List Nat := 20663 :: root69Tail54
def root69Tail52 : List Nat := 20231 :: root69Tail53
def root69Tail51 : List Nat := 19379 :: root69Tail52
def root69Tail50 : List Nat := 18947 :: root69Tail51
def root69Tail49 : List Nat := 18719 :: root69Tail50
def root69Tail48 : List Nat := 18401 :: root69Tail49
def root69Tail47 : List Nat := 18131 :: root69Tail48
def root69Tail46 : List Nat := 17627 :: root69Tail47
def root69Tail45 : List Nat := 16889 :: root69Tail46
def root69Tail44 : List Nat := 16763 :: root69Tail45
def root69Tail43 : List Nat := 16649 :: root69Tail44
def root69Tail42 : List Nat := 16607 :: root69Tail43
def root69Tail41 : List Nat := 16493 :: root69Tail42
def root69Tail40 : List Nat := 16253 :: root69Tail41
def root69Tail39 : List Nat := 15923 :: root69Tail40
def root69Tail38 : List Nat := 15749 :: root69Tail39
def root69Tail37 : List Nat := 14621 :: root69Tail38
def root69Tail36 : List Nat := 14327 :: root69Tail37
def root69Tail35 : List Nat := 13763 :: root69Tail36
def root69Tail34 : List Nat := 12743 :: root69Tail35
def root69Tail33 : List Nat := 12263 :: root69Tail34
def root69Tail32 : List Nat := 12113 :: root69Tail33
def root69Tail31 : List Nat := 11927 :: root69Tail32
def root69Tail30 : List Nat := 11789 :: root69Tail31
def root69Tail29 : List Nat := 11177 :: root69Tail30
def root69Tail28 : List Nat := 11171 :: root69Tail29
def root69Tail27 : List Nat := 10601 :: root69Tail28
def root69Tail26 : List Nat := 10211 :: root69Tail27
def root69Tail25 : List Nat := 10169 :: root69Tail26
def root69Tail24 : List Nat := 9887 :: root69Tail25
def root69Tail23 : List Nat := 9833 :: root69Tail24
def root69Tail22 : List Nat := 8039 :: root69Tail23
def root69Tail21 : List Nat := 7907 :: root69Tail22
def root69Tail20 : List Nat := 7001 :: root69Tail21
def root69Tail19 : List Nat := 6689 :: root69Tail20
def root69Tail18 : List Nat := 6143 :: root69Tail19
def root69Tail17 : List Nat := 5591 :: root69Tail18
def root69Tail16 : List Nat := 3881 :: root69Tail17
def root69Tail15 : List Nat := 3833 :: root69Tail16
def root69Tail14 : List Nat := 3539 :: root69Tail15
def root69Tail13 : List Nat := 3203 :: root69Tail14
def root69Tail12 : List Nat := 2969 :: root69Tail13
def root69Tail11 : List Nat := 2447 :: root69Tail12
def root69Tail10 : List Nat := 2399 :: root69Tail11
def root69Tail9 : List Nat := 2243 :: root69Tail10
def root69Tail8 : List Nat := 1949 :: root69Tail9
def root69Tail7 : List Nat := 1667 :: root69Tail8
def root69Tail6 : List Nat := 1499 :: root69Tail7
def root69Tail5 : List Nat := 1487 :: root69Tail6
def root69Tail4 : List Nat := 911 :: root69Tail5
def root69Tail3 : List Nat := 701 :: root69Tail4
def root69Tail2 : List Nat := 599 :: root69Tail3
def root69Tail1 : List Nat := 563 :: root69Tail2
def root69Tail0 : List Nat := 509 :: root69Tail1
def root69Candidates0 : List Nat := [20663]
set_option maxRecDepth 8192 in
theorem root69Candidates0_checked : candidates edge 4 359 509 root69Tail1 winner.bound = root69Candidates0 := by
  unfold candidates
  have row : edge 509 = (SortedNatTree.contains FastRows.row509) := by
    change Data.row509.contains = (SortedNatTree.contains FastRows.row509)
    exact FastRows.row509_checked.symm
  rw [row]
  decide
theorem root69Step0 : scan edge (search edge 3) 4 [359] 359 root69Tail0 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail1 winner := by
  rw [root69Tail0, scan, if_neg (by decide), root69Candidates0_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail1
    (if 3 ≤ root69Candidates0.length then search edge 3 [359, 509] root69Candidates0 868 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root69Candidates1_checked : candidates edge 4 359 563 root69Tail2 winner.bound = root69Candidates1 := by
  unfold candidates
  have row : edge 563 = (SortedNatTree.contains FastRows.row563) := by
    change Data.row563.contains = (SortedNatTree.contains FastRows.row563)
    exact FastRows.row563_checked.symm
  rw [row]
  decide
theorem root69Step1 : scan edge (search edge 3) 4 [359] 359 root69Tail1 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail2 winner := by
  rw [root69Tail1, scan, if_neg (by decide), root69Candidates1_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail2
    (if 3 ≤ root69Candidates1.length then search edge 3 [359, 563] root69Candidates1 922 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates2 : List Nat := [8039, 15923, 16889]
set_option maxRecDepth 8192 in
theorem root69Candidates2_checked : candidates edge 4 359 599 root69Tail3 winner.bound = root69Candidates2 := by
  unfold candidates
  have row : edge 599 = (SortedNatTree.contains FastRows.row599) := by
    change Data.row599.contains = (SortedNatTree.contains FastRows.row599)
    exact FastRows.row599_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root69Child2 : search edge 3 [359, 599] root69Candidates2 958 winner = winner := by decide
theorem root69Step2 : scan edge (search edge 3) 4 [359] 359 root69Tail2 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail3 winner := by
  rw [root69Tail2, scan, if_neg (by decide), root69Candidates2_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail3
    (if 3 ≤ root69Candidates2.length then search edge 3 [359, 599] root69Candidates2 958 winner else winner) = _
  rw [if_pos (by decide), root69Child2]
def root69Candidates3 : List Nat := [21881]
set_option maxRecDepth 8192 in
theorem root69Candidates3_checked : candidates edge 4 359 701 root69Tail4 winner.bound = root69Candidates3 := by
  unfold candidates
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row]
  decide
theorem root69Step3 : scan edge (search edge 3) 4 [359] 359 root69Tail3 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail4 winner := by
  rw [root69Tail3, scan, if_neg (by decide), root69Candidates3_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail4
    (if 3 ≤ root69Candidates3.length then search edge 3 [359, 701] root69Candidates3 1060 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates4 : List Nat := [2399, 22391]
set_option maxRecDepth 8192 in
theorem root69Candidates4_checked : candidates edge 4 359 911 root69Tail5 winner.bound = root69Candidates4 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root69Step4 : scan edge (search edge 3) 4 [359] 359 root69Tail4 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail5 winner := by
  rw [root69Tail4, scan, if_neg (by decide), root69Candidates4_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail5
    (if 3 ≤ root69Candidates4.length then search edge 3 [359, 911] root69Candidates4 1270 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates5 : List Nat := [16649, 19379, 22193]
set_option maxRecDepth 8192 in
theorem root69Candidates5_checked : candidates edge 4 359 1487 root69Tail6 winner.bound = root69Candidates5 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root69Child5 : search edge 3 [359, 1487] root69Candidates5 1846 winner = winner := by decide
theorem root69Step5 : scan edge (search edge 3) 4 [359] 359 root69Tail5 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail6 winner := by
  rw [root69Tail5, scan, if_neg (by decide), root69Candidates5_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail6
    (if 3 ≤ root69Candidates5.length then search edge 3 [359, 1487] root69Candidates5 1846 winner else winner) = _
  rw [if_pos (by decide), root69Child5]
def root69Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root69Candidates6_checked : candidates edge 4 359 1499 root69Tail7 winner.bound = root69Candidates6 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
theorem root69Step6 : scan edge (search edge 3) 4 [359] 359 root69Tail6 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail7 winner := by
  rw [root69Tail6, scan, if_neg (by decide), root69Candidates6_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail7
    (if 3 ≤ root69Candidates6.length then search edge 3 [359, 1499] root69Candidates6 1858 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates7 : List Nat := [12113, 21881]
set_option maxRecDepth 8192 in
theorem root69Candidates7_checked : candidates edge 4 359 1667 root69Tail8 winner.bound = root69Candidates7 := by
  unfold candidates
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row]
  decide
theorem root69Step7 : scan edge (search edge 3) 4 [359] 359 root69Tail7 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail8 winner := by
  rw [root69Tail7, scan, if_neg (by decide), root69Candidates7_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail8
    (if 3 ≤ root69Candidates7.length then search edge 3 [359, 1667] root69Candidates7 2026 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root69Candidates8_checked : candidates edge 4 359 1949 root69Tail9 winner.bound = root69Candidates8 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root69Step8 : scan edge (search edge 3) 4 [359] 359 root69Tail8 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail9 winner := by
  rw [root69Tail8, scan, if_neg (by decide), root69Candidates8_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail9
    (if 3 ≤ root69Candidates8.length then search edge 3 [359, 1949] root69Candidates8 2308 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates9 : List Nat := [3539, 12743]
set_option maxRecDepth 8192 in
theorem root69Candidates9_checked : candidates edge 4 359 2243 root69Tail10 winner.bound = root69Candidates9 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
theorem root69Step9 : scan edge (search edge 3) 4 [359] 359 root69Tail9 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail10 winner := by
  rw [root69Tail9, scan, if_neg (by decide), root69Candidates9_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail10
    (if 3 ≤ root69Candidates9.length then search edge 3 [359, 2243] root69Candidates9 2602 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root69Candidates10_checked : candidates edge 4 359 2399 root69Tail11 winner.bound = root69Candidates10 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
theorem root69Step10 : scan edge (search edge 3) 4 [359] 359 root69Tail10 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail11 winner := by
  rw [root69Tail10, scan, if_neg (by decide), root69Candidates10_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail11
    (if 3 ≤ root69Candidates10.length then search edge 3 [359, 2399] root69Candidates10 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates11 : List Nat := [7001, 7907, 10211]
set_option maxRecDepth 8192 in
theorem root69Candidates11_checked : candidates edge 4 359 2447 root69Tail12 winner.bound = root69Candidates11 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root69Child11 : search edge 3 [359, 2447] root69Candidates11 2806 winner = winner := by decide
theorem root69Step11 : scan edge (search edge 3) 4 [359] 359 root69Tail11 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail12 winner := by
  rw [root69Tail11, scan, if_neg (by decide), root69Candidates11_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail12
    (if 3 ≤ root69Candidates11.length then search edge 3 [359, 2447] root69Candidates11 2806 winner else winner) = _
  rw [if_pos (by decide), root69Child11]
def root69Candidates12 : List Nat := [3539, 16493]
set_option maxRecDepth 8192 in
theorem root69Candidates12_checked : candidates edge 4 359 2969 root69Tail13 winner.bound = root69Candidates12 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root69Step12 : scan edge (search edge 3) 4 [359] 359 root69Tail12 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail13 winner := by
  rw [root69Tail12, scan, if_neg (by decide), root69Candidates12_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail13
    (if 3 ≤ root69Candidates12.length then search edge 3 [359, 2969] root69Candidates12 3328 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root69Candidates13_checked : candidates edge 4 359 3203 root69Tail14 winner.bound = root69Candidates13 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
theorem root69Step13 : scan edge (search edge 3) 4 [359] 359 root69Tail13 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail14 winner := by
  rw [root69Tail13, scan, if_neg (by decide), root69Candidates13_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail14
    (if 3 ≤ root69Candidates13.length then search edge 3 [359, 3203] root69Candidates13 3562 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates14 : List Nat := [7001, 16889]
set_option maxRecDepth 8192 in
theorem root69Candidates14_checked : candidates edge 4 359 3539 root69Tail15 winner.bound = root69Candidates14 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root69Step14 : scan edge (search edge 3) 4 [359] 359 root69Tail14 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail15 winner := by
  rw [root69Tail14, scan, if_neg (by decide), root69Candidates14_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail15
    (if 3 ≤ root69Candidates14.length then search edge 3 [359, 3539] root69Candidates14 3898 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates15 : List Nat := [11177, 14621]
set_option maxRecDepth 8192 in
theorem root69Candidates15_checked : candidates edge 4 359 3833 root69Tail16 winner.bound = root69Candidates15 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root69Step15 : scan edge (search edge 3) 4 [359] 359 root69Tail15 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail16 winner := by
  rw [root69Tail15, scan, if_neg (by decide), root69Candidates15_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail16
    (if 3 ≤ root69Candidates15.length then search edge 3 [359, 3833] root69Candidates15 4192 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root69Candidates16_checked : candidates edge 4 359 3881 root69Tail17 winner.bound = root69Candidates16 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root69Step16 : scan edge (search edge 3) 4 [359] 359 root69Tail16 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail17 winner := by
  rw [root69Tail16, scan, if_neg (by decide), root69Candidates16_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail17
    (if 3 ≤ root69Candidates16.length then search edge 3 [359, 3881] root69Candidates16 4240 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates17 : List Nat := [9887, 18401]
set_option maxRecDepth 8192 in
theorem root69Candidates17_checked : candidates edge 4 359 5591 root69Tail18 winner.bound = root69Candidates17 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root69Step17 : scan edge (search edge 3) 4 [359] 359 root69Tail17 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail18 winner := by
  rw [root69Tail17, scan, if_neg (by decide), root69Candidates17_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail18
    (if 3 ≤ root69Candidates17.length then search edge 3 [359, 5591] root69Candidates17 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root69Candidates18 : List Nat := [6689, 16763]
set_option maxRecDepth 8192 in
theorem root69Candidates18_checked : candidates edge 4 359 6143 root69Tail19 winner.bound = root69Candidates18 := by
  unfold candidates
  have row : edge 6143 = (SortedNatTree.contains FastRows.row6143) := by
    change Data.row6143.contains = (SortedNatTree.contains FastRows.row6143)
    exact FastRows.row6143_checked.symm
  rw [row]
  decide
theorem root69Step18 : scan edge (search edge 3) 4 [359] 359 root69Tail18 winner =
    scan edge (search edge 3) 4 [359] 359 root69Tail19 winner := by
  rw [root69Tail18, scan, if_neg (by decide), root69Candidates18_checked]
  change scan edge (search edge 3) 4 [359] 359 root69Tail19
    (if 3 ≤ root69Candidates18.length then search edge 3 [359, 6143] root69Candidates18 6502 winner else winner) = _
  rw [if_neg (by decide)]
theorem root69Scan19 : scan edge (search edge 3) 4 [359] 359 root69Tail19 winner = winner := by rw [root69Tail19, scan, if_pos (by decide)]
theorem root69Scan18 : scan edge (search edge 3) 4 [359] 359 root69Tail18 winner = winner :=
  (root69Step18).trans root69Scan19
theorem root69Scan17 : scan edge (search edge 3) 4 [359] 359 root69Tail17 winner = winner :=
  (root69Step17).trans root69Scan18
theorem root69Scan16 : scan edge (search edge 3) 4 [359] 359 root69Tail16 winner = winner :=
  (root69Step16).trans root69Scan17
theorem root69Scan15 : scan edge (search edge 3) 4 [359] 359 root69Tail15 winner = winner :=
  (root69Step15).trans root69Scan16
theorem root69Scan14 : scan edge (search edge 3) 4 [359] 359 root69Tail14 winner = winner :=
  (root69Step14).trans root69Scan15
theorem root69Scan13 : scan edge (search edge 3) 4 [359] 359 root69Tail13 winner = winner :=
  (root69Step13).trans root69Scan14
theorem root69Scan12 : scan edge (search edge 3) 4 [359] 359 root69Tail12 winner = winner :=
  (root69Step12).trans root69Scan13
theorem root69Scan11 : scan edge (search edge 3) 4 [359] 359 root69Tail11 winner = winner :=
  (root69Step11).trans root69Scan12
theorem root69Scan10 : scan edge (search edge 3) 4 [359] 359 root69Tail10 winner = winner :=
  (root69Step10).trans root69Scan11
theorem root69Scan9 : scan edge (search edge 3) 4 [359] 359 root69Tail9 winner = winner :=
  (root69Step9).trans root69Scan10
theorem root69Scan8 : scan edge (search edge 3) 4 [359] 359 root69Tail8 winner = winner :=
  (root69Step8).trans root69Scan9
theorem root69Scan7 : scan edge (search edge 3) 4 [359] 359 root69Tail7 winner = winner :=
  (root69Step7).trans root69Scan8
theorem root69Scan6 : scan edge (search edge 3) 4 [359] 359 root69Tail6 winner = winner :=
  (root69Step6).trans root69Scan7
theorem root69Scan5 : scan edge (search edge 3) 4 [359] 359 root69Tail5 winner = winner :=
  (root69Step5).trans root69Scan6
theorem root69Scan4 : scan edge (search edge 3) 4 [359] 359 root69Tail4 winner = winner :=
  (root69Step4).trans root69Scan5
theorem root69Scan3 : scan edge (search edge 3) 4 [359] 359 root69Tail3 winner = winner :=
  (root69Step3).trans root69Scan4
theorem root69Scan2 : scan edge (search edge 3) 4 [359] 359 root69Tail2 winner = winner :=
  (root69Step2).trans root69Scan3
theorem root69Scan1 : scan edge (search edge 3) 4 [359] 359 root69Tail1 winner = winner :=
  (root69Step1).trans root69Scan2
theorem root69Scan0 : scan edge (search edge 3) 4 [359] 359 root69Tail0 winner = winner :=
  (root69Step0).trans root69Scan1
set_option maxRecDepth 4096 in
theorem root69Branch : search edge 4 [359] root69Candidates 359 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [359] 359 root69Tail0 winner = winner
  exact root69Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
