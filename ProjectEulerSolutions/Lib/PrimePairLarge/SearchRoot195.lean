import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root195Candidates : List Nat := Data.row1213
set_option maxRecDepth 8192 in
theorem root195Candidates_checked : candidates edge 5 0 1213 tail196 winner.bound = root195Candidates := by
  apply candidates_eq_row edge 5 0 1213 tail196 root195Candidates winner.bound tail196_ordered
  · decide
  · decide
  · funext q
    rfl
def root195Tail57 : List Nat := []
def root195Tail56 : List Nat := 23833 :: root195Tail57
def root195Tail55 : List Nat := 23743 :: root195Tail56
def root195Tail54 : List Nat := 23029 :: root195Tail55
def root195Tail53 : List Nat := 22483 :: root195Tail54
def root195Tail52 : List Nat := 22381 :: root195Tail53
def root195Tail51 : List Nat := 22273 :: root195Tail52
def root195Tail50 : List Nat := 21649 :: root195Tail51
def root195Tail49 : List Nat := 21379 :: root195Tail50
def root195Tail48 : List Nat := 20707 :: root195Tail49
def root195Tail47 : List Nat := 20389 :: root195Tail48
def root195Tail46 : List Nat := 19801 :: root195Tail47
def root195Tail45 : List Nat := 17827 :: root195Tail46
def root195Tail44 : List Nat := 17569 :: root195Tail45
def root195Tail43 : List Nat := 17257 :: root195Tail44
def root195Tail42 : List Nat := 16477 :: root195Tail43
def root195Tail41 : List Nat := 16183 :: root195Tail42
def root195Tail40 : List Nat := 14461 :: root195Tail41
def root195Tail39 : List Nat := 14071 :: root195Tail40
def root195Tail38 : List Nat := 13999 :: root195Tail39
def root195Tail37 : List Nat := 12577 :: root195Tail38
def root195Tail36 : List Nat := 12109 :: root195Tail37
def root195Tail35 : List Nat := 12007 :: root195Tail36
def root195Tail34 : List Nat := 11731 :: root195Tail35
def root195Tail33 : List Nat := 11551 :: root195Tail34
def root195Tail32 : List Nat := 11149 :: root195Tail33
def root195Tail31 : List Nat := 10831 :: root195Tail32
def root195Tail30 : List Nat := 10477 :: root195Tail31
def root195Tail29 : List Nat := 10141 :: root195Tail30
def root195Tail28 : List Nat := 9349 :: root195Tail29
def root195Tail27 : List Nat := 9013 :: root195Tail28
def root195Tail26 : List Nat := 8803 :: root195Tail27
def root195Tail25 : List Nat := 8761 :: root195Tail26
def root195Tail24 : List Nat := 8689 :: root195Tail25
def root195Tail23 : List Nat := 8647 :: root195Tail24
def root195Tail22 : List Nat := 8521 :: root195Tail23
def root195Tail21 : List Nat := 8353 :: root195Tail22
def root195Tail20 : List Nat := 8209 :: root195Tail21
def root195Tail19 : List Nat := 7243 :: root195Tail20
def root195Tail18 : List Nat := 6547 :: root195Tail19
def root195Tail17 : List Nat := 6211 :: root195Tail18
def root195Tail16 : List Nat := 6151 :: root195Tail17
def root195Tail15 : List Nat := 6073 :: root195Tail16
def root195Tail14 : List Nat := 6043 :: root195Tail15
def root195Tail13 : List Nat := 5479 :: root195Tail14
def root195Tail12 : List Nat := 5443 :: root195Tail13
def root195Tail11 : List Nat := 4201 :: root195Tail12
def root195Tail10 : List Nat := 3463 :: root195Tail11
def root195Tail9 : List Nat := 3301 :: root195Tail10
def root195Tail8 : List Nat := 3061 :: root195Tail9
def root195Tail7 : List Nat := 2851 :: root195Tail8
def root195Tail6 : List Nat := 2797 :: root195Tail7
def root195Tail5 : List Nat := 2371 :: root195Tail6
def root195Tail4 : List Nat := 2053 :: root195Tail5
def root195Tail3 : List Nat := 2011 :: root195Tail4
def root195Tail2 : List Nat := 1789 :: root195Tail3
def root195Tail1 : List Nat := 1621 :: root195Tail2
def root195Tail0 : List Nat := 1297 :: root195Tail1
def root195Candidates0 : List Nat := [12007, 12577, 23029]
set_option maxRecDepth 8192 in
theorem root195Candidates0_checked : candidates edge 4 1213 1297 root195Tail1 winner.bound = root195Candidates0 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root195Child0 : search edge 3 [1213, 1297] root195Candidates0 2510 winner = winner := by decide
theorem root195Step0 : scan edge (search edge 3) 4 [1213] 1213 root195Tail0 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail1 winner := by
  rw [root195Tail0, scan, if_neg (by decide), root195Candidates0_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail1
    (if 3 ≤ root195Candidates0.length then search edge 3 [1213, 1297] root195Candidates0 2510 winner else winner) = _
  rw [if_pos (by decide), root195Child0]
def root195Candidates1 : List Nat := [2851, 5443, 6073, 11149, 21649]
set_option maxRecDepth 8192 in
theorem root195Candidates1_checked : candidates edge 4 1213 1621 root195Tail2 winner.bound = root195Candidates1 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root195Child1 : search edge 3 [1213, 1621] root195Candidates1 2834 winner = winner := by decide
theorem root195Step1 : scan edge (search edge 3) 4 [1213] 1213 root195Tail1 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail2 winner := by
  rw [root195Tail1, scan, if_neg (by decide), root195Candidates1_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail2
    (if 3 ≤ root195Candidates1.length then search edge 3 [1213, 1621] root195Candidates1 2834 winner else winner) = _
  rw [if_pos (by decide), root195Child1]
def root195Candidates2 : List Nat := [6547]
set_option maxRecDepth 8192 in
theorem root195Candidates2_checked : candidates edge 4 1213 1789 root195Tail3 winner.bound = root195Candidates2 := by
  unfold candidates
  have row : edge 1789 = (SortedNatTree.contains FastRows.row1789) := by
    change Data.row1789.contains = (SortedNatTree.contains FastRows.row1789)
    exact FastRows.row1789_checked.symm
  rw [row]
  decide
theorem root195Step2 : scan edge (search edge 3) 4 [1213] 1213 root195Tail2 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail3 winner := by
  rw [root195Tail2, scan, if_neg (by decide), root195Candidates2_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail3
    (if 3 ≤ root195Candidates2.length then search edge 3 [1213, 1789] root195Candidates2 3002 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates3 : List Nat := [10477]
set_option maxRecDepth 8192 in
theorem root195Candidates3_checked : candidates edge 4 1213 2011 root195Tail4 winner.bound = root195Candidates3 := by
  unfold candidates
  have row : edge 2011 = (SortedNatTree.contains FastRows.row2011) := by
    change Data.row2011.contains = (SortedNatTree.contains FastRows.row2011)
    exact FastRows.row2011_checked.symm
  rw [row]
  decide
theorem root195Step3 : scan edge (search edge 3) 4 [1213] 1213 root195Tail3 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail4 winner := by
  rw [root195Tail3, scan, if_neg (by decide), root195Candidates3_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail4
    (if 3 ≤ root195Candidates3.length then search edge 3 [1213, 2011] root195Candidates3 3224 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates4 : List Nat := [2797, 8521, 17257]
set_option maxRecDepth 8192 in
theorem root195Candidates4_checked : candidates edge 4 1213 2053 root195Tail5 winner.bound = root195Candidates4 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root195Child4 : search edge 3 [1213, 2053] root195Candidates4 3266 winner = winner := by decide
theorem root195Step4 : scan edge (search edge 3) 4 [1213] 1213 root195Tail4 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail5 winner := by
  rw [root195Tail4, scan, if_neg (by decide), root195Candidates4_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail5
    (if 3 ≤ root195Candidates4.length then search edge 3 [1213, 2053] root195Candidates4 3266 winner else winner) = _
  rw [if_pos (by decide), root195Child4]
def root195Candidates5 : List Nat := [10141, 13999, 17827]
set_option maxRecDepth 8192 in
theorem root195Candidates5_checked : candidates edge 4 1213 2371 root195Tail6 winner.bound = root195Candidates5 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root195Child5 : search edge 3 [1213, 2371] root195Candidates5 3584 winner = winner := by decide
theorem root195Step5 : scan edge (search edge 3) 4 [1213] 1213 root195Tail5 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail6 winner := by
  rw [root195Tail5, scan, if_neg (by decide), root195Candidates5_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail6
    (if 3 ≤ root195Candidates5.length then search edge 3 [1213, 2371] root195Candidates5 3584 winner else winner) = _
  rw [if_pos (by decide), root195Child5]
def root195Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root195Candidates6_checked : candidates edge 4 1213 2797 root195Tail7 winner.bound = root195Candidates6 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root195Step6 : scan edge (search edge 3) 4 [1213] 1213 root195Tail6 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail7 winner := by
  rw [root195Tail6, scan, if_neg (by decide), root195Candidates6_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail7
    (if 3 ≤ root195Candidates6.length then search edge 3 [1213, 2797] root195Candidates6 4010 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates7 : List Nat := [8647, 10831, 12007]
set_option maxRecDepth 8192 in
theorem root195Candidates7_checked : candidates edge 4 1213 2851 root195Tail8 winner.bound = root195Candidates7 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root195Child7 : search edge 3 [1213, 2851] root195Candidates7 4064 winner = winner := by decide
theorem root195Step7 : scan edge (search edge 3) 4 [1213] 1213 root195Tail7 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail8 winner := by
  rw [root195Tail7, scan, if_neg (by decide), root195Candidates7_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail8
    (if 3 ≤ root195Candidates7.length then search edge 3 [1213, 2851] root195Candidates7 4064 winner else winner) = _
  rw [if_pos (by decide), root195Child7]
def root195Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root195Candidates8_checked : candidates edge 4 1213 3061 root195Tail9 winner.bound = root195Candidates8 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root195Step8 : scan edge (search edge 3) 4 [1213] 1213 root195Tail8 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail9 winner := by
  rw [root195Tail8, scan, if_neg (by decide), root195Candidates8_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail9
    (if 3 ≤ root195Candidates8.length then search edge 3 [1213, 3061] root195Candidates8 4274 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root195Candidates9_checked : candidates edge 4 1213 3301 root195Tail10 winner.bound = root195Candidates9 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root195Step9 : scan edge (search edge 3) 4 [1213] 1213 root195Tail9 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail10 winner := by
  rw [root195Tail9, scan, if_neg (by decide), root195Candidates9_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail10
    (if 3 ≤ root195Candidates9.length then search edge 3 [1213, 3301] root195Candidates9 4514 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates10 : List Nat := [5479, 6211, 20389]
set_option maxRecDepth 8192 in
theorem root195Candidates10_checked : candidates edge 4 1213 3463 root195Tail11 winner.bound = root195Candidates10 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root195Child10 : search edge 3 [1213, 3463] root195Candidates10 4676 winner = winner := by decide
theorem root195Step10 : scan edge (search edge 3) 4 [1213] 1213 root195Tail10 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail11 winner := by
  rw [root195Tail10, scan, if_neg (by decide), root195Candidates10_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail11
    (if 3 ≤ root195Candidates10.length then search edge 3 [1213, 3463] root195Candidates10 4676 winner else winner) = _
  rw [if_pos (by decide), root195Child10]
def root195Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root195Candidates11_checked : candidates edge 4 1213 4201 root195Tail12 winner.bound = root195Candidates11 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root195Step11 : scan edge (search edge 3) 4 [1213] 1213 root195Tail11 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail12 winner := by
  rw [root195Tail11, scan, if_neg (by decide), root195Candidates11_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail12
    (if 3 ≤ root195Candidates11.length then search edge 3 [1213, 4201] root195Candidates11 5414 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root195Candidates12_checked : candidates edge 4 1213 5443 root195Tail13 winner.bound = root195Candidates12 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root195Step12 : scan edge (search edge 3) 4 [1213] 1213 root195Tail12 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail13 winner := by
  rw [root195Tail12, scan, if_neg (by decide), root195Candidates12_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail13
    (if 3 ≤ root195Candidates12.length then search edge 3 [1213, 5443] root195Candidates12 6656 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates13 : List Nat := [8209]
set_option maxRecDepth 8192 in
theorem root195Candidates13_checked : candidates edge 4 1213 5479 root195Tail14 winner.bound = root195Candidates13 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root195Step13 : scan edge (search edge 3) 4 [1213] 1213 root195Tail13 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail14 winner := by
  rw [root195Tail13, scan, if_neg (by decide), root195Candidates13_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail14
    (if 3 ≤ root195Candidates13.length then search edge 3 [1213, 5479] root195Candidates13 6692 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates14 : List Nat := [16183]
set_option maxRecDepth 8192 in
theorem root195Candidates14_checked : candidates edge 4 1213 6043 root195Tail15 winner.bound = root195Candidates14 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
theorem root195Step14 : scan edge (search edge 3) 4 [1213] 1213 root195Tail14 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail15 winner := by
  rw [root195Tail14, scan, if_neg (by decide), root195Candidates14_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail15
    (if 3 ≤ root195Candidates14.length then search edge 3 [1213, 6043] root195Candidates14 7256 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates15 : List Nat := [8647, 8761]
set_option maxRecDepth 8192 in
theorem root195Candidates15_checked : candidates edge 4 1213 6073 root195Tail16 winner.bound = root195Candidates15 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root195Step15 : scan edge (search edge 3) 4 [1213] 1213 root195Tail15 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail16 winner := by
  rw [root195Tail15, scan, if_neg (by decide), root195Candidates15_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail16
    (if 3 ≤ root195Candidates15.length then search edge 3 [1213, 6073] root195Candidates15 7286 winner else winner) = _
  rw [if_neg (by decide)]
def root195Candidates16 : List Nat := [8647]
set_option maxRecDepth 8192 in
theorem root195Candidates16_checked : candidates edge 4 1213 6151 root195Tail17 winner.bound = root195Candidates16 := by
  unfold candidates
  have row : edge 6151 = (SortedNatTree.contains FastRows.row6151) := by
    change Data.row6151.contains = (SortedNatTree.contains FastRows.row6151)
    exact FastRows.row6151_checked.symm
  rw [row]
  decide
theorem root195Step16 : scan edge (search edge 3) 4 [1213] 1213 root195Tail16 winner =
    scan edge (search edge 3) 4 [1213] 1213 root195Tail17 winner := by
  rw [root195Tail16, scan, if_neg (by decide), root195Candidates16_checked]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail17
    (if 3 ≤ root195Candidates16.length then search edge 3 [1213, 6151] root195Candidates16 7364 winner else winner) = _
  rw [if_neg (by decide)]
theorem root195Scan17 : scan edge (search edge 3) 4 [1213] 1213 root195Tail17 winner = winner := by rw [root195Tail17, scan, if_pos (by decide)]
theorem root195Scan16 : scan edge (search edge 3) 4 [1213] 1213 root195Tail16 winner = winner :=
  (root195Step16).trans root195Scan17
theorem root195Scan15 : scan edge (search edge 3) 4 [1213] 1213 root195Tail15 winner = winner :=
  (root195Step15).trans root195Scan16
theorem root195Scan14 : scan edge (search edge 3) 4 [1213] 1213 root195Tail14 winner = winner :=
  (root195Step14).trans root195Scan15
theorem root195Scan13 : scan edge (search edge 3) 4 [1213] 1213 root195Tail13 winner = winner :=
  (root195Step13).trans root195Scan14
theorem root195Scan12 : scan edge (search edge 3) 4 [1213] 1213 root195Tail12 winner = winner :=
  (root195Step12).trans root195Scan13
theorem root195Scan11 : scan edge (search edge 3) 4 [1213] 1213 root195Tail11 winner = winner :=
  (root195Step11).trans root195Scan12
theorem root195Scan10 : scan edge (search edge 3) 4 [1213] 1213 root195Tail10 winner = winner :=
  (root195Step10).trans root195Scan11
theorem root195Scan9 : scan edge (search edge 3) 4 [1213] 1213 root195Tail9 winner = winner :=
  (root195Step9).trans root195Scan10
theorem root195Scan8 : scan edge (search edge 3) 4 [1213] 1213 root195Tail8 winner = winner :=
  (root195Step8).trans root195Scan9
theorem root195Scan7 : scan edge (search edge 3) 4 [1213] 1213 root195Tail7 winner = winner :=
  (root195Step7).trans root195Scan8
theorem root195Scan6 : scan edge (search edge 3) 4 [1213] 1213 root195Tail6 winner = winner :=
  (root195Step6).trans root195Scan7
theorem root195Scan5 : scan edge (search edge 3) 4 [1213] 1213 root195Tail5 winner = winner :=
  (root195Step5).trans root195Scan6
theorem root195Scan4 : scan edge (search edge 3) 4 [1213] 1213 root195Tail4 winner = winner :=
  (root195Step4).trans root195Scan5
theorem root195Scan3 : scan edge (search edge 3) 4 [1213] 1213 root195Tail3 winner = winner :=
  (root195Step3).trans root195Scan4
theorem root195Scan2 : scan edge (search edge 3) 4 [1213] 1213 root195Tail2 winner = winner :=
  (root195Step2).trans root195Scan3
theorem root195Scan1 : scan edge (search edge 3) 4 [1213] 1213 root195Tail1 winner = winner :=
  (root195Step1).trans root195Scan2
theorem root195Scan0 : scan edge (search edge 3) 4 [1213] 1213 root195Tail0 winner = winner :=
  (root195Step0).trans root195Scan1
set_option maxRecDepth 4096 in
theorem root195Branch : search edge 4 [1213] root195Candidates 1213 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1213] 1213 root195Tail0 winner = winner
  exact root195Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
