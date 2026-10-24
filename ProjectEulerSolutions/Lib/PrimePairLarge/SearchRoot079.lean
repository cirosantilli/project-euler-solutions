import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root79Candidates : List Nat := Data.row421
set_option maxRecDepth 8192 in
theorem root79Candidates_checked : candidates edge 5 0 421 tail80 winner.bound = root79Candidates := by
  apply candidates_eq_row edge 5 0 421 tail80 root79Candidates winner.bound tail80_ordered
  · decide
  · decide
  · funext q
    rfl
def root79Tail55 : List Nat := []
def root79Tail54 : List Nat := 24091 :: root79Tail55
def root79Tail53 : List Nat := 21943 :: root79Tail54
def root79Tail52 : List Nat := 21661 :: root79Tail53
def root79Tail51 : List Nat := 20809 :: root79Tail52
def root79Tail50 : List Nat := 19963 :: root79Tail51
def root79Tail49 : List Nat := 18541 :: root79Tail50
def root79Tail48 : List Nat := 17539 :: root79Tail49
def root79Tail47 : List Nat := 17377 :: root79Tail48
def root79Tail46 : List Nat := 17239 :: root79Tail47
def root79Tail45 : List Nat := 15937 :: root79Tail46
def root79Tail44 : List Nat := 15649 :: root79Tail45
def root79Tail43 : List Nat := 15643 :: root79Tail44
def root79Tail42 : List Nat := 15607 :: root79Tail43
def root79Tail41 : List Nat := 15349 :: root79Tail42
def root79Tail40 : List Nat := 14557 :: root79Tail41
def root79Tail39 : List Nat := 14551 :: root79Tail40
def root79Tail38 : List Nat := 13411 :: root79Tail39
def root79Tail37 : List Nat := 12907 :: root79Tail38
def root79Tail36 : List Nat := 12409 :: root79Tail37
def root79Tail35 : List Nat := 12211 :: root79Tail36
def root79Tail34 : List Nat := 11071 :: root79Tail35
def root79Tail33 : List Nat := 10993 :: root79Tail34
def root79Tail32 : List Nat := 10891 :: root79Tail33
def root79Tail31 : List Nat := 10273 :: root79Tail32
def root79Tail30 : List Nat := 9967 :: root79Tail31
def root79Tail29 : List Nat := 9949 :: root79Tail30
def root79Tail28 : List Nat := 8971 :: root79Tail29
def root79Tail27 : List Nat := 8713 :: root79Tail28
def root79Tail26 : List Nat := 8707 :: root79Tail27
def root79Tail25 : List Nat := 8563 :: root79Tail26
def root79Tail24 : List Nat := 8287 :: root79Tail25
def root79Tail23 : List Nat := 7669 :: root79Tail24
def root79Tail22 : List Nat := 7459 :: root79Tail23
def root79Tail21 : List Nat := 7237 :: root79Tail22
def root79Tail20 : List Nat := 7159 :: root79Tail21
def root79Tail19 : List Nat := 6871 :: root79Tail20
def root79Tail18 : List Nat := 6211 :: root79Tail19
def root79Tail17 : List Nat := 6073 :: root79Tail18
def root79Tail16 : List Nat := 5749 :: root79Tail17
def root79Tail15 : List Nat := 5683 :: root79Tail16
def root79Tail14 : List Nat := 5641 :: root79Tail15
def root79Tail13 : List Nat := 5347 :: root79Tail14
def root79Tail12 : List Nat := 4933 :: root79Tail13
def root79Tail11 : List Nat := 4519 :: root79Tail12
def root79Tail10 : List Nat := 3217 :: root79Tail11
def root79Tail9 : List Nat := 2287 :: root79Tail10
def root79Tail8 : List Nat := 2113 :: root79Tail9
def root79Tail7 : List Nat := 1657 :: root79Tail8
def root79Tail6 : List Nat := 1579 :: root79Tail7
def root79Tail5 : List Nat := 1321 :: root79Tail6
def root79Tail4 : List Nat := 709 :: root79Tail5
def root79Tail3 : List Nat := 661 :: root79Tail4
def root79Tail2 : List Nat := 643 :: root79Tail3
def root79Tail1 : List Nat := 607 :: root79Tail2
def root79Tail0 : List Nat := 433 :: root79Tail1
def root79Candidates0 : List Nat := [17539]
set_option maxRecDepth 8192 in
theorem root79Candidates0_checked : candidates edge 4 421 433 root79Tail1 winner.bound = root79Candidates0 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
theorem root79Step0 : scan edge (search edge 3) 4 [421] 421 root79Tail0 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail1 winner := by
  rw [root79Tail0, scan, if_neg (by decide), root79Candidates0_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail1
    (if 3 ≤ root79Candidates0.length then search edge 3 [421, 433] root79Candidates0 854 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates1 : List Nat := [8287]
set_option maxRecDepth 8192 in
theorem root79Candidates1_checked : candidates edge 4 421 607 root79Tail2 winner.bound = root79Candidates1 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
theorem root79Step1 : scan edge (search edge 3) 4 [421] 421 root79Tail1 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail2 winner := by
  rw [root79Tail1, scan, if_neg (by decide), root79Candidates1_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail2
    (if 3 ≤ root79Candidates1.length then search edge 3 [421, 607] root79Candidates1 1028 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root79Candidates2_checked : candidates edge 4 421 643 root79Tail3 winner.bound = root79Candidates2 := by
  unfold candidates
  have row : edge 643 = (SortedNatTree.contains FastRows.row643) := by
    change Data.row643.contains = (SortedNatTree.contains FastRows.row643)
    exact FastRows.row643_checked.symm
  rw [row]
  decide
theorem root79Step2 : scan edge (search edge 3) 4 [421] 421 root79Tail2 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail3 winner := by
  rw [root79Tail2, scan, if_neg (by decide), root79Candidates2_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail3
    (if 3 ≤ root79Candidates2.length then search edge 3 [421, 643] root79Candidates2 1064 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates3 : List Nat := [5641, 6871, 7159, 17239]
set_option maxRecDepth 8192 in
theorem root79Candidates3_checked : candidates edge 4 421 661 root79Tail4 winner.bound = root79Candidates3 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root79Child3 : search edge 3 [421, 661] root79Candidates3 1082 winner = winner := by decide
theorem root79Step3 : scan edge (search edge 3) 4 [421] 421 root79Tail3 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail4 winner := by
  rw [root79Tail3, scan, if_neg (by decide), root79Candidates3_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail4
    (if 3 ≤ root79Candidates3.length then search edge 3 [421, 661] root79Candidates3 1082 winner else winner) = _
  rw [if_pos (by decide), root79Child3]
def root79Candidates4 : List Nat := [4519, 7459, 8713, 10891]
set_option maxRecDepth 8192 in
theorem root79Candidates4_checked : candidates edge 4 421 709 root79Tail5 winner.bound = root79Candidates4 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root79Child4 : search edge 3 [421, 709] root79Candidates4 1130 winner = winner := by decide
theorem root79Step4 : scan edge (search edge 3) 4 [421] 421 root79Tail4 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail5 winner := by
  rw [root79Tail4, scan, if_neg (by decide), root79Candidates4_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail5
    (if 3 ≤ root79Candidates4.length then search edge 3 [421, 709] root79Candidates4 1130 winner else winner) = _
  rw [if_pos (by decide), root79Child4]
def root79Candidates5 : List Nat := [14551, 17239]
set_option maxRecDepth 8192 in
theorem root79Candidates5_checked : candidates edge 4 421 1321 root79Tail6 winner.bound = root79Candidates5 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
theorem root79Step5 : scan edge (search edge 3) 4 [421] 421 root79Tail5 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail6 winner := by
  rw [root79Tail5, scan, if_neg (by decide), root79Candidates5_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail6
    (if 3 ≤ root79Candidates5.length then search edge 3 [421, 1321] root79Candidates5 1742 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates6 : List Nat := [6073]
set_option maxRecDepth 8192 in
theorem root79Candidates6_checked : candidates edge 4 421 1579 root79Tail7 winner.bound = root79Candidates6 := by
  unfold candidates
  have row : edge 1579 = (SortedNatTree.contains FastRows.row1579) := by
    change Data.row1579.contains = (SortedNatTree.contains FastRows.row1579)
    exact FastRows.row1579_checked.symm
  rw [row]
  decide
theorem root79Step6 : scan edge (search edge 3) 4 [421] 421 root79Tail6 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail7 winner := by
  rw [root79Tail6, scan, if_neg (by decide), root79Candidates6_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail7
    (if 3 ≤ root79Candidates6.length then search edge 3 [421, 1579] root79Candidates6 2000 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates7 : List Nat := [7237, 17239, 20809]
set_option maxRecDepth 8192 in
theorem root79Candidates7_checked : candidates edge 4 421 1657 root79Tail8 winner.bound = root79Candidates7 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root79Child7 : search edge 3 [421, 1657] root79Candidates7 2078 winner = winner := by decide
theorem root79Step7 : scan edge (search edge 3) 4 [421] 421 root79Tail7 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail8 winner := by
  rw [root79Tail7, scan, if_neg (by decide), root79Candidates7_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail8
    (if 3 ≤ root79Candidates7.length then search edge 3 [421, 1657] root79Candidates7 2078 winner else winner) = _
  rw [if_pos (by decide), root79Child7]
def root79Candidates8 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root79Candidates8_checked : candidates edge 4 421 2113 root79Tail9 winner.bound = root79Candidates8 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root79Step8 : scan edge (search edge 3) 4 [421] 421 root79Tail8 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail9 winner := by
  rw [root79Tail8, scan, if_neg (by decide), root79Candidates8_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail9
    (if 3 ≤ root79Candidates8.length then search edge 3 [421, 2113] root79Candidates8 2534 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates9 : List Nat := [4933]
set_option maxRecDepth 8192 in
theorem root79Candidates9_checked : candidates edge 4 421 2287 root79Tail10 winner.bound = root79Candidates9 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
theorem root79Step9 : scan edge (search edge 3) 4 [421] 421 root79Tail9 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail10 winner := by
  rw [root79Tail9, scan, if_neg (by decide), root79Candidates9_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail10
    (if 3 ≤ root79Candidates9.length then search edge 3 [421, 2287] root79Candidates9 2708 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates10 : List Nat := [6073, 6211, 10273, 15649, 17377]
set_option maxRecDepth 8192 in
theorem root79Candidates10_checked : candidates edge 4 421 3217 root79Tail11 winner.bound = root79Candidates10 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root79Child10 : search edge 3 [421, 3217] root79Candidates10 3638 winner = winner := by decide
theorem root79Step10 : scan edge (search edge 3) 4 [421] 421 root79Tail10 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail11 winner := by
  rw [root79Tail10, scan, if_neg (by decide), root79Candidates10_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail11
    (if 3 ≤ root79Candidates10.length then search edge 3 [421, 3217] root79Candidates10 3638 winner else winner) = _
  rw [if_pos (by decide), root79Child10]
def root79Candidates11 : List Nat := [15649]
set_option maxRecDepth 8192 in
theorem root79Candidates11_checked : candidates edge 4 421 4519 root79Tail12 winner.bound = root79Candidates11 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root79Step11 : scan edge (search edge 3) 4 [421] 421 root79Tail11 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail12 winner := by
  rw [root79Tail11, scan, if_neg (by decide), root79Candidates11_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail12
    (if 3 ≤ root79Candidates11.length then search edge 3 [421, 4519] root79Candidates11 4940 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates12 : List Nat := [15349, 15643, 19963]
set_option maxRecDepth 8192 in
theorem root79Candidates12_checked : candidates edge 4 421 4933 root79Tail13 winner.bound = root79Candidates12 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root79Child12 : search edge 3 [421, 4933] root79Candidates12 5354 winner = winner := by decide
theorem root79Step12 : scan edge (search edge 3) 4 [421] 421 root79Tail12 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail13 winner := by
  rw [root79Tail12, scan, if_neg (by decide), root79Candidates12_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail13
    (if 3 ≤ root79Candidates12.length then search edge 3 [421, 4933] root79Candidates12 5354 winner else winner) = _
  rw [if_pos (by decide), root79Child12]
def root79Candidates13 : List Nat := [8287, 15607]
set_option maxRecDepth 8192 in
theorem root79Candidates13_checked : candidates edge 4 421 5347 root79Tail14 winner.bound = root79Candidates13 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root79Step13 : scan edge (search edge 3) 4 [421] 421 root79Tail13 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail14 winner := by
  rw [root79Tail13, scan, if_neg (by decide), root79Candidates13_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail14
    (if 3 ≤ root79Candidates13.length then search edge 3 [421, 5347] root79Candidates13 5768 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root79Candidates14_checked : candidates edge 4 421 5641 root79Tail15 winner.bound = root79Candidates14 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root79Step14 : scan edge (search edge 3) 4 [421] 421 root79Tail14 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail15 winner := by
  rw [root79Tail14, scan, if_neg (by decide), root79Candidates14_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail15
    (if 3 ≤ root79Candidates14.length then search edge 3 [421, 5641] root79Candidates14 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates15 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root79Candidates15_checked : candidates edge 4 421 5683 root79Tail16 winner.bound = root79Candidates15 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root79Step15 : scan edge (search edge 3) 4 [421] 421 root79Tail15 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail16 winner := by
  rw [root79Tail15, scan, if_neg (by decide), root79Candidates15_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail16
    (if 3 ≤ root79Candidates15.length then search edge 3 [421, 5683] root79Candidates15 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates16 : List Nat := [6211]
set_option maxRecDepth 8192 in
theorem root79Candidates16_checked : candidates edge 4 421 5749 root79Tail17 winner.bound = root79Candidates16 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root79Step16 : scan edge (search edge 3) 4 [421] 421 root79Tail16 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail17 winner := by
  rw [root79Tail16, scan, if_neg (by decide), root79Candidates16_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail17
    (if 3 ≤ root79Candidates16.length then search edge 3 [421, 5749] root79Candidates16 6170 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates17 : List Nat := [8563, 15349]
set_option maxRecDepth 8192 in
theorem root79Candidates17_checked : candidates edge 4 421 6073 root79Tail18 winner.bound = root79Candidates17 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root79Step17 : scan edge (search edge 3) 4 [421] 421 root79Tail17 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail18 winner := by
  rw [root79Tail17, scan, if_neg (by decide), root79Candidates17_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail18
    (if 3 ≤ root79Candidates17.length then search edge 3 [421, 6073] root79Candidates17 6494 winner else winner) = _
  rw [if_neg (by decide)]
def root79Candidates18 : List Nat := [10891, 14551]
set_option maxRecDepth 8192 in
theorem root79Candidates18_checked : candidates edge 4 421 6211 root79Tail19 winner.bound = root79Candidates18 := by
  unfold candidates
  have row : edge 6211 = (SortedNatTree.contains FastRows.row6211) := by
    change Data.row6211.contains = (SortedNatTree.contains FastRows.row6211)
    exact FastRows.row6211_checked.symm
  rw [row]
  decide
theorem root79Step18 : scan edge (search edge 3) 4 [421] 421 root79Tail18 winner =
    scan edge (search edge 3) 4 [421] 421 root79Tail19 winner := by
  rw [root79Tail18, scan, if_neg (by decide), root79Candidates18_checked]
  change scan edge (search edge 3) 4 [421] 421 root79Tail19
    (if 3 ≤ root79Candidates18.length then search edge 3 [421, 6211] root79Candidates18 6632 winner else winner) = _
  rw [if_neg (by decide)]
theorem root79Scan19 : scan edge (search edge 3) 4 [421] 421 root79Tail19 winner = winner := by rw [root79Tail19, scan, if_pos (by decide)]
theorem root79Scan18 : scan edge (search edge 3) 4 [421] 421 root79Tail18 winner = winner :=
  (root79Step18).trans root79Scan19
theorem root79Scan17 : scan edge (search edge 3) 4 [421] 421 root79Tail17 winner = winner :=
  (root79Step17).trans root79Scan18
theorem root79Scan16 : scan edge (search edge 3) 4 [421] 421 root79Tail16 winner = winner :=
  (root79Step16).trans root79Scan17
theorem root79Scan15 : scan edge (search edge 3) 4 [421] 421 root79Tail15 winner = winner :=
  (root79Step15).trans root79Scan16
theorem root79Scan14 : scan edge (search edge 3) 4 [421] 421 root79Tail14 winner = winner :=
  (root79Step14).trans root79Scan15
theorem root79Scan13 : scan edge (search edge 3) 4 [421] 421 root79Tail13 winner = winner :=
  (root79Step13).trans root79Scan14
theorem root79Scan12 : scan edge (search edge 3) 4 [421] 421 root79Tail12 winner = winner :=
  (root79Step12).trans root79Scan13
theorem root79Scan11 : scan edge (search edge 3) 4 [421] 421 root79Tail11 winner = winner :=
  (root79Step11).trans root79Scan12
theorem root79Scan10 : scan edge (search edge 3) 4 [421] 421 root79Tail10 winner = winner :=
  (root79Step10).trans root79Scan11
theorem root79Scan9 : scan edge (search edge 3) 4 [421] 421 root79Tail9 winner = winner :=
  (root79Step9).trans root79Scan10
theorem root79Scan8 : scan edge (search edge 3) 4 [421] 421 root79Tail8 winner = winner :=
  (root79Step8).trans root79Scan9
theorem root79Scan7 : scan edge (search edge 3) 4 [421] 421 root79Tail7 winner = winner :=
  (root79Step7).trans root79Scan8
theorem root79Scan6 : scan edge (search edge 3) 4 [421] 421 root79Tail6 winner = winner :=
  (root79Step6).trans root79Scan7
theorem root79Scan5 : scan edge (search edge 3) 4 [421] 421 root79Tail5 winner = winner :=
  (root79Step5).trans root79Scan6
theorem root79Scan4 : scan edge (search edge 3) 4 [421] 421 root79Tail4 winner = winner :=
  (root79Step4).trans root79Scan5
theorem root79Scan3 : scan edge (search edge 3) 4 [421] 421 root79Tail3 winner = winner :=
  (root79Step3).trans root79Scan4
theorem root79Scan2 : scan edge (search edge 3) 4 [421] 421 root79Tail2 winner = winner :=
  (root79Step2).trans root79Scan3
theorem root79Scan1 : scan edge (search edge 3) 4 [421] 421 root79Tail1 winner = winner :=
  (root79Step1).trans root79Scan2
theorem root79Scan0 : scan edge (search edge 3) 4 [421] 421 root79Tail0 winner = winner :=
  (root79Step0).trans root79Scan1
set_option maxRecDepth 4096 in
theorem root79Branch : search edge 4 [421] root79Candidates 421 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [421] 421 root79Tail0 winner = winner
  exact root79Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
