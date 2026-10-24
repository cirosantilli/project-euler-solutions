import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root185Candidates : List Nat := Data.row1123
set_option maxRecDepth 8192 in
theorem root185Candidates_checked : candidates edge 5 0 1123 tail186 winner.bound = root185Candidates := by
  apply candidates_eq_row edge 5 0 1123 tail186 root185Candidates winner.bound tail186_ordered
  · decide
  · decide
  · funext q
    rfl
def root185Tail49 : List Nat := []
def root185Tail48 : List Nat := 24169 :: root185Tail49
def root185Tail47 : List Nat := 24103 :: root185Tail48
def root185Tail46 : List Nat := 23833 :: root185Tail47
def root185Tail45 : List Nat := 23599 :: root185Tail46
def root185Tail44 : List Nat := 22381 :: root185Tail45
def root185Tail43 : List Nat := 21487 :: root185Tail44
def root185Tail42 : List Nat := 20113 :: root185Tail43
def root185Tail41 : List Nat := 19927 :: root185Tail42
def root185Tail40 : List Nat := 19891 :: root185Tail41
def root185Tail39 : List Nat := 18793 :: root185Tail40
def root185Tail38 : List Nat := 17761 :: root185Tail39
def root185Tail37 : List Nat := 17749 :: root185Tail38
def root185Tail36 : List Nat := 17659 :: root185Tail37
def root185Tail35 : List Nat := 17449 :: root185Tail36
def root185Tail34 : List Nat := 17077 :: root185Tail35
def root185Tail33 : List Nat := 16879 :: root185Tail34
def root185Tail32 : List Nat := 16447 :: root185Tail33
def root185Tail31 : List Nat := 16033 :: root185Tail32
def root185Tail30 : List Nat := 15937 :: root185Tail31
def root185Tail29 : List Nat := 15733 :: root185Tail30
def root185Tail28 : List Nat := 15271 :: root185Tail29
def root185Tail27 : List Nat := 14551 :: root185Tail28
def root185Tail26 : List Nat := 14479 :: root185Tail27
def root185Tail25 : List Nat := 13297 :: root185Tail26
def root185Tail24 : List Nat := 11839 :: root185Tail25
def root185Tail23 : List Nat := 11821 :: root185Tail24
def root185Tail22 : List Nat := 10477 :: root185Tail23
def root185Tail21 : List Nat := 9769 :: root185Tail22
def root185Tail20 : List Nat := 9103 :: root185Tail21
def root185Tail19 : List Nat := 8221 :: root185Tail20
def root185Tail18 : List Nat := 7489 :: root185Tail19
def root185Tail17 : List Nat := 6037 :: root185Tail18
def root185Tail16 : List Nat := 5737 :: root185Tail17
def root185Tail15 : List Nat := 5641 :: root185Tail16
def root185Tail14 : List Nat := 5527 :: root185Tail15
def root185Tail13 : List Nat := 5347 :: root185Tail14
def root185Tail12 : List Nat := 5281 :: root185Tail13
def root185Tail11 : List Nat := 4423 :: root185Tail12
def root185Tail10 : List Nat := 3919 :: root185Tail11
def root185Tail9 : List Nat := 3433 :: root185Tail10
def root185Tail8 : List Nat := 3373 :: root185Tail9
def root185Tail7 : List Nat := 2281 :: root185Tail8
def root185Tail6 : List Nat := 2203 :: root185Tail7
def root185Tail5 : List Nat := 2029 :: root185Tail6
def root185Tail4 : List Nat := 1783 :: root185Tail5
def root185Tail3 : List Nat := 1657 :: root185Tail4
def root185Tail2 : List Nat := 1453 :: root185Tail3
def root185Tail1 : List Nat := 1399 :: root185Tail2
def root185Tail0 : List Nat := 1201 :: root185Tail1
def root185Candidates0 : List Nat := [5281, 11821, 17077]
set_option maxRecDepth 8192 in
theorem root185Candidates0_checked : candidates edge 4 1123 1201 root185Tail1 winner.bound = root185Candidates0 := by
  unfold candidates
  have row : edge 1201 = (SortedNatTree.contains FastRows.row1201) := by
    change Data.row1201.contains = (SortedNatTree.contains FastRows.row1201)
    exact FastRows.row1201_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root185Child0 : search edge 3 [1123, 1201] root185Candidates0 2324 winner = winner := by decide
theorem root185Step0 : scan edge (search edge 3) 4 [1123] 1123 root185Tail0 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail1 winner := by
  rw [root185Tail0, scan, if_neg (by decide), root185Candidates0_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail1
    (if 3 ≤ root185Candidates0.length then search edge 3 [1123, 1201] root185Candidates0 2324 winner else winner) = _
  rw [if_pos (by decide), root185Child0]
def root185Candidates1 : List Nat := [3433, 15733]
set_option maxRecDepth 8192 in
theorem root185Candidates1_checked : candidates edge 4 1123 1399 root185Tail2 winner.bound = root185Candidates1 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
theorem root185Step1 : scan edge (search edge 3) 4 [1123] 1123 root185Tail1 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail2 winner := by
  rw [root185Tail1, scan, if_neg (by decide), root185Candidates1_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail2
    (if 3 ≤ root185Candidates1.length then search edge 3 [1123, 1399] root185Candidates1 2522 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates2 : List Nat := [5281, 9103, 14551]
set_option maxRecDepth 8192 in
theorem root185Candidates2_checked : candidates edge 4 1123 1453 root185Tail3 winner.bound = root185Candidates2 := by
  unfold candidates
  have row : edge 1453 = (SortedNatTree.contains FastRows.row1453) := by
    change Data.row1453.contains = (SortedNatTree.contains FastRows.row1453)
    exact FastRows.row1453_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root185Child2 : search edge 3 [1123, 1453] root185Candidates2 2576 winner = winner := by decide
theorem root185Step2 : scan edge (search edge 3) 4 [1123] 1123 root185Tail2 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail3 winner := by
  rw [root185Tail2, scan, if_neg (by decide), root185Candidates2_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail3
    (if 3 ≤ root185Candidates2.length then search edge 3 [1123, 1453] root185Candidates2 2576 winner else winner) = _
  rw [if_pos (by decide), root185Child2]
def root185Candidates3 : List Nat := [2203, 18793]
set_option maxRecDepth 8192 in
theorem root185Candidates3_checked : candidates edge 4 1123 1657 root185Tail4 winner.bound = root185Candidates3 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
theorem root185Step3 : scan edge (search edge 3) 4 [1123] 1123 root185Tail3 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail4 winner := by
  rw [root185Tail3, scan, if_neg (by decide), root185Candidates3_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail4
    (if 3 ≤ root185Candidates3.length then search edge 3 [1123, 1657] root185Candidates3 2780 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates4 : List Nat := [14551]
set_option maxRecDepth 8192 in
theorem root185Candidates4_checked : candidates edge 4 1123 1783 root185Tail5 winner.bound = root185Candidates4 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
theorem root185Step4 : scan edge (search edge 3) 4 [1123] 1123 root185Tail4 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail5 winner := by
  rw [root185Tail4, scan, if_neg (by decide), root185Candidates4_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail5
    (if 3 ≤ root185Candidates4.length then search edge 3 [1123, 1783] root185Candidates4 2906 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates5 : List Nat := [2281, 5281, 13297]
set_option maxRecDepth 8192 in
theorem root185Candidates5_checked : candidates edge 4 1123 2029 root185Tail6 winner.bound = root185Candidates5 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root185Child5 : search edge 3 [1123, 2029] root185Candidates5 3152 winner = winner := by decide
theorem root185Step5 : scan edge (search edge 3) 4 [1123] 1123 root185Tail5 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail6 winner := by
  rw [root185Tail5, scan, if_neg (by decide), root185Candidates5_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail6
    (if 3 ≤ root185Candidates5.length then search edge 3 [1123, 2029] root185Candidates5 3152 winner else winner) = _
  rw [if_pos (by decide), root185Child5]
def root185Candidates6 : List Nat := [6037]
set_option maxRecDepth 8192 in
theorem root185Candidates6_checked : candidates edge 4 1123 2203 root185Tail7 winner.bound = root185Candidates6 := by
  unfold candidates
  have row : edge 2203 = (SortedNatTree.contains FastRows.row2203) := by
    change Data.row2203.contains = (SortedNatTree.contains FastRows.row2203)
    exact FastRows.row2203_checked.symm
  rw [row]
  decide
theorem root185Step6 : scan edge (search edge 3) 4 [1123] 1123 root185Tail6 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail7 winner := by
  rw [root185Tail6, scan, if_neg (by decide), root185Candidates6_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail7
    (if 3 ≤ root185Candidates6.length then search edge 3 [1123, 2203] root185Candidates6 3326 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root185Candidates7_checked : candidates edge 4 1123 2281 root185Tail8 winner.bound = root185Candidates7 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root185Step7 : scan edge (search edge 3) 4 [1123] 1123 root185Tail7 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail8 winner := by
  rw [root185Tail7, scan, if_neg (by decide), root185Candidates7_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail8
    (if 3 ≤ root185Candidates7.length then search edge 3 [1123, 2281] root185Candidates7 3404 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates8 : List Nat := [9103]
set_option maxRecDepth 8192 in
theorem root185Candidates8_checked : candidates edge 4 1123 3373 root185Tail9 winner.bound = root185Candidates8 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root185Step8 : scan edge (search edge 3) 4 [1123] 1123 root185Tail8 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail9 winner := by
  rw [root185Tail8, scan, if_neg (by decide), root185Candidates8_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail9
    (if 3 ≤ root185Candidates8.length then search edge 3 [1123, 3373] root185Candidates8 4496 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates9 : List Nat := [8221, 19927]
set_option maxRecDepth 8192 in
theorem root185Candidates9_checked : candidates edge 4 1123 3433 root185Tail10 winner.bound = root185Candidates9 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root185Step9 : scan edge (search edge 3) 4 [1123] 1123 root185Tail9 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail10 winner := by
  rw [root185Tail9, scan, if_neg (by decide), root185Candidates9_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail10
    (if 3 ≤ root185Candidates9.length then search edge 3 [1123, 3433] root185Candidates9 4556 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates10 : List Nat := [5281, 7489, 9103, 17749]
set_option maxRecDepth 8192 in
theorem root185Candidates10_checked : candidates edge 4 1123 3919 root185Tail11 winner.bound = root185Candidates10 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root185Child10 : search edge 3 [1123, 3919] root185Candidates10 5042 winner = winner := by decide
theorem root185Step10 : scan edge (search edge 3) 4 [1123] 1123 root185Tail10 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail11 winner := by
  rw [root185Tail10, scan, if_neg (by decide), root185Candidates10_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail11
    (if 3 ≤ root185Candidates10.length then search edge 3 [1123, 3919] root185Candidates10 5042 winner else winner) = _
  rw [if_pos (by decide), root185Child10]
def root185Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root185Candidates11_checked : candidates edge 4 1123 4423 root185Tail12 winner.bound = root185Candidates11 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root185Step11 : scan edge (search edge 3) 4 [1123] 1123 root185Tail11 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail12 winner := by
  rw [root185Tail11, scan, if_neg (by decide), root185Candidates11_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail12
    (if 3 ≤ root185Candidates11.length then search edge 3 [1123, 4423] root185Candidates11 5546 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates12 : List Nat := [7489]
set_option maxRecDepth 8192 in
theorem root185Candidates12_checked : candidates edge 4 1123 5281 root185Tail13 winner.bound = root185Candidates12 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root185Step12 : scan edge (search edge 3) 4 [1123] 1123 root185Tail12 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail13 winner := by
  rw [root185Tail12, scan, if_neg (by decide), root185Candidates12_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail13
    (if 3 ≤ root185Candidates12.length then search edge 3 [1123, 5281] root185Candidates12 6404 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates13 : List Nat := [8221]
set_option maxRecDepth 8192 in
theorem root185Candidates13_checked : candidates edge 4 1123 5347 root185Tail14 winner.bound = root185Candidates13 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root185Step13 : scan edge (search edge 3) 4 [1123] 1123 root185Tail13 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail14 winner := by
  rw [root185Tail13, scan, if_neg (by decide), root185Candidates13_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail14
    (if 3 ≤ root185Candidates13.length then search edge 3 [1123, 5347] root185Candidates13 6470 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates14 : List Nat := [11839]
set_option maxRecDepth 8192 in
theorem root185Candidates14_checked : candidates edge 4 1123 5527 root185Tail15 winner.bound = root185Candidates14 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root185Step14 : scan edge (search edge 3) 4 [1123] 1123 root185Tail14 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail15 winner := by
  rw [root185Tail14, scan, if_neg (by decide), root185Candidates14_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail15
    (if 3 ≤ root185Candidates14.length then search edge 3 [1123, 5527] root185Candidates14 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root185Candidates15_checked : candidates edge 4 1123 5641 root185Tail16 winner.bound = root185Candidates15 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root185Step15 : scan edge (search edge 3) 4 [1123] 1123 root185Tail15 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail16 winner := by
  rw [root185Tail15, scan, if_neg (by decide), root185Candidates15_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail16
    (if 3 ≤ root185Candidates15.length then search edge 3 [1123, 5641] root185Candidates15 6764 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates16 : List Nat := [9769]
set_option maxRecDepth 8192 in
theorem root185Candidates16_checked : candidates edge 4 1123 5737 root185Tail17 winner.bound = root185Candidates16 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root185Step16 : scan edge (search edge 3) 4 [1123] 1123 root185Tail16 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail17 winner := by
  rw [root185Tail16, scan, if_neg (by decide), root185Candidates16_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail17
    (if 3 ≤ root185Candidates16.length then search edge 3 [1123, 5737] root185Candidates16 6860 winner else winner) = _
  rw [if_neg (by decide)]
def root185Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root185Candidates17_checked : candidates edge 4 1123 6037 root185Tail18 winner.bound = root185Candidates17 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root185Step17 : scan edge (search edge 3) 4 [1123] 1123 root185Tail17 winner =
    scan edge (search edge 3) 4 [1123] 1123 root185Tail18 winner := by
  rw [root185Tail17, scan, if_neg (by decide), root185Candidates17_checked]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail18
    (if 3 ≤ root185Candidates17.length then search edge 3 [1123, 6037] root185Candidates17 7160 winner else winner) = _
  rw [if_neg (by decide)]
theorem root185Scan18 : scan edge (search edge 3) 4 [1123] 1123 root185Tail18 winner = winner := by rw [root185Tail18, scan, if_pos (by decide)]
theorem root185Scan17 : scan edge (search edge 3) 4 [1123] 1123 root185Tail17 winner = winner :=
  (root185Step17).trans root185Scan18
theorem root185Scan16 : scan edge (search edge 3) 4 [1123] 1123 root185Tail16 winner = winner :=
  (root185Step16).trans root185Scan17
theorem root185Scan15 : scan edge (search edge 3) 4 [1123] 1123 root185Tail15 winner = winner :=
  (root185Step15).trans root185Scan16
theorem root185Scan14 : scan edge (search edge 3) 4 [1123] 1123 root185Tail14 winner = winner :=
  (root185Step14).trans root185Scan15
theorem root185Scan13 : scan edge (search edge 3) 4 [1123] 1123 root185Tail13 winner = winner :=
  (root185Step13).trans root185Scan14
theorem root185Scan12 : scan edge (search edge 3) 4 [1123] 1123 root185Tail12 winner = winner :=
  (root185Step12).trans root185Scan13
theorem root185Scan11 : scan edge (search edge 3) 4 [1123] 1123 root185Tail11 winner = winner :=
  (root185Step11).trans root185Scan12
theorem root185Scan10 : scan edge (search edge 3) 4 [1123] 1123 root185Tail10 winner = winner :=
  (root185Step10).trans root185Scan11
theorem root185Scan9 : scan edge (search edge 3) 4 [1123] 1123 root185Tail9 winner = winner :=
  (root185Step9).trans root185Scan10
theorem root185Scan8 : scan edge (search edge 3) 4 [1123] 1123 root185Tail8 winner = winner :=
  (root185Step8).trans root185Scan9
theorem root185Scan7 : scan edge (search edge 3) 4 [1123] 1123 root185Tail7 winner = winner :=
  (root185Step7).trans root185Scan8
theorem root185Scan6 : scan edge (search edge 3) 4 [1123] 1123 root185Tail6 winner = winner :=
  (root185Step6).trans root185Scan7
theorem root185Scan5 : scan edge (search edge 3) 4 [1123] 1123 root185Tail5 winner = winner :=
  (root185Step5).trans root185Scan6
theorem root185Scan4 : scan edge (search edge 3) 4 [1123] 1123 root185Tail4 winner = winner :=
  (root185Step4).trans root185Scan5
theorem root185Scan3 : scan edge (search edge 3) 4 [1123] 1123 root185Tail3 winner = winner :=
  (root185Step3).trans root185Scan4
theorem root185Scan2 : scan edge (search edge 3) 4 [1123] 1123 root185Tail2 winner = winner :=
  (root185Step2).trans root185Scan3
theorem root185Scan1 : scan edge (search edge 3) 4 [1123] 1123 root185Tail1 winner = winner :=
  (root185Step1).trans root185Scan2
theorem root185Scan0 : scan edge (search edge 3) 4 [1123] 1123 root185Tail0 winner = winner :=
  (root185Step0).trans root185Scan1
set_option maxRecDepth 4096 in
theorem root185Branch : search edge 4 [1123] root185Candidates 1123 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1123] 1123 root185Tail0 winner = winner
  exact root185Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
