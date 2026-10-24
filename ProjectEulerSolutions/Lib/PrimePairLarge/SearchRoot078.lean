import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root78Candidates : List Nat := Data.row419
set_option maxRecDepth 8192 in
theorem root78Candidates_checked : candidates edge 5 0 419 tail79 winner.bound = root78Candidates := by
  apply candidates_eq_row edge 5 0 419 tail79 root78Candidates winner.bound tail79_ordered
  · decide
  · decide
  · funext q
    rfl
def root78Tail58 : List Nat := []
def root78Tail57 : List Nat := 25523 :: root78Tail58
def root78Tail56 : List Nat := 25307 :: root78Tail57
def root78Tail55 : List Nat := 24671 :: root78Tail56
def root78Tail54 : List Nat := 23993 :: root78Tail55
def root78Tail53 : List Nat := 23789 :: root78Tail54
def root78Tail52 : List Nat := 22871 :: root78Tail53
def root78Tail51 : List Nat := 22691 :: root78Tail52
def root78Tail50 : List Nat := 22613 :: root78Tail51
def root78Tail49 : List Nat := 21023 :: root78Tail50
def root78Tail48 : List Nat := 21017 :: root78Tail49
def root78Tail47 : List Nat := 20717 :: root78Tail48
def root78Tail46 : List Nat := 20261 :: root78Tail47
def root78Tail45 : List Nat := 19469 :: root78Tail46
def root78Tail44 : List Nat := 19121 :: root78Tail45
def root78Tail43 : List Nat := 17783 :: root78Tail44
def root78Tail42 : List Nat := 17231 :: root78Tail43
def root78Tail41 : List Nat := 17159 :: root78Tail42
def root78Tail40 : List Nat := 17027 :: root78Tail41
def root78Tail39 : List Nat := 16937 :: root78Tail40
def root78Tail38 : List Nat := 16703 :: root78Tail39
def root78Tail37 : List Nat := 16661 :: root78Tail38
def root78Tail36 : List Nat := 16493 :: root78Tail37
def root78Tail35 : List Nat := 16349 :: root78Tail36
def root78Tail34 : List Nat := 15227 :: root78Tail35
def root78Tail33 : List Nat := 14813 :: root78Tail34
def root78Tail32 : List Nat := 14633 :: root78Tail33
def root78Tail31 : List Nat := 14519 :: root78Tail32
def root78Tail30 : List Nat := 13841 :: root78Tail31
def root78Tail29 : List Nat := 13709 :: root78Tail30
def root78Tail28 : List Nat := 13451 :: root78Tail29
def root78Tail27 : List Nat := 12953 :: root78Tail28
def root78Tail26 : List Nat := 12239 :: root78Tail27
def root78Tail25 : List Nat := 11789 :: root78Tail26
def root78Tail24 : List Nat := 9851 :: root78Tail25
def root78Tail23 : List Nat := 9257 :: root78Tail24
def root78Tail22 : List Nat := 8837 :: root78Tail23
def root78Tail21 : List Nat := 8807 :: root78Tail22
def root78Tail20 : List Nat := 8111 :: root78Tail21
def root78Tail19 : List Nat := 7883 :: root78Tail20
def root78Tail18 : List Nat := 7757 :: root78Tail19
def root78Tail17 : List Nat := 7253 :: root78Tail18
def root78Tail16 : List Nat := 5441 :: root78Tail17
def root78Tail15 : List Nat := 5297 :: root78Tail16
def root78Tail14 : List Nat := 5273 :: root78Tail15
def root78Tail13 : List Nat := 4583 :: root78Tail14
def root78Tail12 : List Nat := 3803 :: root78Tail13
def root78Tail11 : List Nat := 3221 :: root78Tail12
def root78Tail10 : List Nat := 3209 :: root78Tail11
def root78Tail9 : List Nat := 2129 :: root78Tail10
def root78Tail8 : List Nat := 2087 :: root78Tail9
def root78Tail7 : List Nat := 1373 :: root78Tail8
def root78Tail6 : List Nat := 1097 :: root78Tail7
def root78Tail5 : List Nat := 929 :: root78Tail6
def root78Tail4 : List Nat := 701 :: root78Tail5
def root78Tail3 : List Nat := 599 :: root78Tail4
def root78Tail2 : List Nat := 563 :: root78Tail3
def root78Tail1 : List Nat := 449 :: root78Tail2
def root78Tail0 : List Nat := 443 :: root78Tail1
def root78Candidates0 : List Nat := [701, 7757, 7883]
set_option maxRecDepth 8192 in
theorem root78Candidates0_checked : candidates edge 4 419 443 root78Tail1 winner.bound = root78Candidates0 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root78Child0 : search edge 3 [419, 443] root78Candidates0 862 winner = winner := by decide
theorem root78Step0 : scan edge (search edge 3) 4 [419] 419 root78Tail0 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail1 winner := by
  rw [root78Tail0, scan, if_neg (by decide), root78Candidates0_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail1
    (if 3 ≤ root78Candidates0.length then search edge 3 [419, 443] root78Candidates0 862 winner else winner) = _
  rw [if_pos (by decide), root78Child0]
def root78Candidates1 : List Nat := [563, 5297]
set_option maxRecDepth 8192 in
theorem root78Candidates1_checked : candidates edge 4 419 449 root78Tail2 winner.bound = root78Candidates1 := by
  unfold candidates
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row]
  decide
theorem root78Step1 : scan edge (search edge 3) 4 [419] 419 root78Tail1 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail2 winner := by
  rw [root78Tail1, scan, if_neg (by decide), root78Candidates1_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail2
    (if 3 ≤ root78Candidates1.length then search edge 3 [419, 449] root78Candidates1 868 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates2 : List Nat := [5297, 13841, 16661]
set_option maxRecDepth 8192 in
theorem root78Candidates2_checked : candidates edge 4 419 563 root78Tail3 winner.bound = root78Candidates2 := by
  unfold candidates
  have row : edge 563 = (SortedNatTree.contains FastRows.row563) := by
    change Data.row563.contains = (SortedNatTree.contains FastRows.row563)
    exact FastRows.row563_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root78Child2 : search edge 3 [419, 563] root78Candidates2 982 winner = winner := by decide
theorem root78Step2 : scan edge (search edge 3) 4 [419] 419 root78Tail2 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail3 winner := by
  rw [root78Tail2, scan, if_neg (by decide), root78Candidates2_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail3
    (if 3 ≤ root78Candidates2.length then search edge 3 [419, 563] root78Candidates2 982 winner else winner) = _
  rw [if_pos (by decide), root78Child2]
def root78Candidates3 : List Nat := [7883, 19469, 20261, 21023, 22871]
set_option maxRecDepth 8192 in
theorem root78Candidates3_checked : candidates edge 4 419 599 root78Tail4 winner.bound = root78Candidates3 := by
  unfold candidates
  have row : edge 599 = (SortedNatTree.contains FastRows.row599) := by
    change Data.row599.contains = (SortedNatTree.contains FastRows.row599)
    exact FastRows.row599_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root78Child3 : search edge 3 [419, 599] root78Candidates3 1018 winner = winner := by decide
theorem root78Step3 : scan edge (search edge 3) 4 [419] 419 root78Tail3 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail4 winner := by
  rw [root78Tail3, scan, if_neg (by decide), root78Candidates3_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail4
    (if 3 ≤ root78Candidates3.length then search edge 3 [419, 599] root78Candidates3 1018 winner else winner) = _
  rw [if_pos (by decide), root78Child3]
def root78Candidates4 : List Nat := [4583, 9851, 12239, 12953, 21017]
set_option maxRecDepth 8192 in
theorem root78Candidates4_checked : candidates edge 4 419 701 root78Tail5 winner.bound = root78Candidates4 := by
  unfold candidates
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root78Child4 : search edge 3 [419, 701] root78Candidates4 1120 winner = winner := by decide
theorem root78Step4 : scan edge (search edge 3) 4 [419] 419 root78Tail4 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail5 winner := by
  rw [root78Tail4, scan, if_neg (by decide), root78Candidates4_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail5
    (if 3 ≤ root78Candidates4.length then search edge 3 [419, 701] root78Candidates4 1120 winner else winner) = _
  rw [if_pos (by decide), root78Child4]
def root78Candidates5 : List Nat := [3209, 14519]
set_option maxRecDepth 8192 in
theorem root78Candidates5_checked : candidates edge 4 419 929 root78Tail6 winner.bound = root78Candidates5 := by
  unfold candidates
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row]
  decide
theorem root78Step5 : scan edge (search edge 3) 4 [419] 419 root78Tail5 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail6 winner := by
  rw [root78Tail5, scan, if_neg (by decide), root78Candidates5_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail6
    (if 3 ≤ root78Candidates5.length then search edge 3 [419, 929] root78Candidates5 1348 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root78Candidates6_checked : candidates edge 4 419 1097 root78Tail7 winner.bound = root78Candidates6 := by
  unfold candidates
  have row : edge 1097 = (SortedNatTree.contains FastRows.row1097) := by
    change Data.row1097.contains = (SortedNatTree.contains FastRows.row1097)
    exact FastRows.row1097_checked.symm
  rw [row]
  decide
theorem root78Step6 : scan edge (search edge 3) 4 [419] 419 root78Tail6 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail7 winner := by
  rw [root78Tail6, scan, if_neg (by decide), root78Candidates6_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail7
    (if 3 ≤ root78Candidates6.length then search edge 3 [419, 1097] root78Candidates6 1516 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root78Candidates7_checked : candidates edge 4 419 1373 root78Tail8 winner.bound = root78Candidates7 := by
  unfold candidates
  have row : edge 1373 = (SortedNatTree.contains FastRows.row1373) := by
    change Data.row1373.contains = (SortedNatTree.contains FastRows.row1373)
    exact FastRows.row1373_checked.symm
  rw [row]
  decide
theorem root78Step7 : scan edge (search edge 3) 4 [419] 419 root78Tail7 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail8 winner := by
  rw [root78Tail7, scan, if_neg (by decide), root78Candidates7_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail8
    (if 3 ≤ root78Candidates7.length then search edge 3 [419, 1373] root78Candidates7 1792 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates8 : List Nat := [17231]
set_option maxRecDepth 8192 in
theorem root78Candidates8_checked : candidates edge 4 419 2087 root78Tail9 winner.bound = root78Candidates8 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root78Step8 : scan edge (search edge 3) 4 [419] 419 root78Tail8 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail9 winner := by
  rw [root78Tail8, scan, if_neg (by decide), root78Candidates8_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail9
    (if 3 ≤ root78Candidates8.length then search edge 3 [419, 2087] root78Candidates8 2506 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root78Candidates9_checked : candidates edge 4 419 2129 root78Tail10 winner.bound = root78Candidates9 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root78Step9 : scan edge (search edge 3) 4 [419] 419 root78Tail9 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail10 winner := by
  rw [root78Tail9, scan, if_neg (by decide), root78Candidates9_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail10
    (if 3 ≤ root78Candidates9.length then search edge 3 [419, 2129] root78Candidates9 2548 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates10 : List Nat := [9257, 17159]
set_option maxRecDepth 8192 in
theorem root78Candidates10_checked : candidates edge 4 419 3209 root78Tail11 winner.bound = root78Candidates10 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root78Step10 : scan edge (search edge 3) 4 [419] 419 root78Tail10 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail11 winner := by
  rw [root78Tail10, scan, if_neg (by decide), root78Candidates10_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail11
    (if 3 ≤ root78Candidates10.length then search edge 3 [419, 3209] root78Candidates10 3628 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates11 : List Nat := [12239]
set_option maxRecDepth 8192 in
theorem root78Candidates11_checked : candidates edge 4 419 3221 root78Tail12 winner.bound = root78Candidates11 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root78Step11 : scan edge (search edge 3) 4 [419] 419 root78Tail11 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail12 winner := by
  rw [root78Tail11, scan, if_neg (by decide), root78Candidates11_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail12
    (if 3 ≤ root78Candidates11.length then search edge 3 [419, 3221] root78Candidates11 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates12 : List Nat := [7253, 17159]
set_option maxRecDepth 8192 in
theorem root78Candidates12_checked : candidates edge 4 419 3803 root78Tail13 winner.bound = root78Candidates12 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root78Step12 : scan edge (search edge 3) 4 [419] 419 root78Tail12 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail13 winner := by
  rw [root78Tail12, scan, if_neg (by decide), root78Candidates12_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail13
    (if 3 ≤ root78Candidates12.length then search edge 3 [419, 3803] root78Candidates12 4222 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root78Candidates13_checked : candidates edge 4 419 4583 root78Tail14 winner.bound = root78Candidates13 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root78Step13 : scan edge (search edge 3) 4 [419] 419 root78Tail13 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail14 winner := by
  rw [root78Tail13, scan, if_neg (by decide), root78Candidates13_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail14
    (if 3 ≤ root78Candidates13.length then search edge 3 [419, 4583] root78Candidates13 5002 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates14 : List Nat := [19469]
set_option maxRecDepth 8192 in
theorem root78Candidates14_checked : candidates edge 4 419 5273 root78Tail15 winner.bound = root78Candidates14 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root78Step14 : scan edge (search edge 3) 4 [419] 419 root78Tail14 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail15 winner := by
  rw [root78Tail14, scan, if_neg (by decide), root78Candidates14_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail15
    (if 3 ≤ root78Candidates14.length then search edge 3 [419, 5273] root78Candidates14 5692 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root78Candidates15_checked : candidates edge 4 419 5297 root78Tail16 winner.bound = root78Candidates15 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root78Step15 : scan edge (search edge 3) 4 [419] 419 root78Tail15 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail16 winner := by
  rw [root78Tail15, scan, if_neg (by decide), root78Candidates15_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail16
    (if 3 ≤ root78Candidates15.length then search edge 3 [419, 5297] root78Candidates15 5716 winner else winner) = _
  rw [if_neg (by decide)]
def root78Candidates16 : List Nat := [13841]
set_option maxRecDepth 8192 in
theorem root78Candidates16_checked : candidates edge 4 419 5441 root78Tail17 winner.bound = root78Candidates16 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root78Step16 : scan edge (search edge 3) 4 [419] 419 root78Tail16 winner =
    scan edge (search edge 3) 4 [419] 419 root78Tail17 winner := by
  rw [root78Tail16, scan, if_neg (by decide), root78Candidates16_checked]
  change scan edge (search edge 3) 4 [419] 419 root78Tail17
    (if 3 ≤ root78Candidates16.length then search edge 3 [419, 5441] root78Candidates16 5860 winner else winner) = _
  rw [if_neg (by decide)]
theorem root78Scan17 : scan edge (search edge 3) 4 [419] 419 root78Tail17 winner = winner := by rw [root78Tail17, scan, if_pos (by decide)]
theorem root78Scan16 : scan edge (search edge 3) 4 [419] 419 root78Tail16 winner = winner :=
  (root78Step16).trans root78Scan17
theorem root78Scan15 : scan edge (search edge 3) 4 [419] 419 root78Tail15 winner = winner :=
  (root78Step15).trans root78Scan16
theorem root78Scan14 : scan edge (search edge 3) 4 [419] 419 root78Tail14 winner = winner :=
  (root78Step14).trans root78Scan15
theorem root78Scan13 : scan edge (search edge 3) 4 [419] 419 root78Tail13 winner = winner :=
  (root78Step13).trans root78Scan14
theorem root78Scan12 : scan edge (search edge 3) 4 [419] 419 root78Tail12 winner = winner :=
  (root78Step12).trans root78Scan13
theorem root78Scan11 : scan edge (search edge 3) 4 [419] 419 root78Tail11 winner = winner :=
  (root78Step11).trans root78Scan12
theorem root78Scan10 : scan edge (search edge 3) 4 [419] 419 root78Tail10 winner = winner :=
  (root78Step10).trans root78Scan11
theorem root78Scan9 : scan edge (search edge 3) 4 [419] 419 root78Tail9 winner = winner :=
  (root78Step9).trans root78Scan10
theorem root78Scan8 : scan edge (search edge 3) 4 [419] 419 root78Tail8 winner = winner :=
  (root78Step8).trans root78Scan9
theorem root78Scan7 : scan edge (search edge 3) 4 [419] 419 root78Tail7 winner = winner :=
  (root78Step7).trans root78Scan8
theorem root78Scan6 : scan edge (search edge 3) 4 [419] 419 root78Tail6 winner = winner :=
  (root78Step6).trans root78Scan7
theorem root78Scan5 : scan edge (search edge 3) 4 [419] 419 root78Tail5 winner = winner :=
  (root78Step5).trans root78Scan6
theorem root78Scan4 : scan edge (search edge 3) 4 [419] 419 root78Tail4 winner = winner :=
  (root78Step4).trans root78Scan5
theorem root78Scan3 : scan edge (search edge 3) 4 [419] 419 root78Tail3 winner = winner :=
  (root78Step3).trans root78Scan4
theorem root78Scan2 : scan edge (search edge 3) 4 [419] 419 root78Tail2 winner = winner :=
  (root78Step2).trans root78Scan3
theorem root78Scan1 : scan edge (search edge 3) 4 [419] 419 root78Tail1 winner = winner :=
  (root78Step1).trans root78Scan2
theorem root78Scan0 : scan edge (search edge 3) 4 [419] 419 root78Tail0 winner = winner :=
  (root78Step0).trans root78Scan1
set_option maxRecDepth 4096 in
theorem root78Branch : search edge 4 [419] root78Candidates 419 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [419] 419 root78Tail0 winner = winner
  exact root78Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
