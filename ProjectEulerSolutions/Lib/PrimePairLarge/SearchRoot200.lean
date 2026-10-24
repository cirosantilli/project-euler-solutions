import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root200Candidates : List Nat := Data.row1237
set_option maxRecDepth 8192 in
theorem root200Candidates_checked : candidates edge 5 0 1237 tail201 winner.bound = root200Candidates := by
  apply candidates_eq_row edge 5 0 1237 tail201 root200Candidates winner.bound tail201_ordered
  · decide
  · decide
  · funext q
    rfl
def root200Tail57 : List Nat := []
def root200Tail56 : List Nat := 23911 :: root200Tail57
def root200Tail55 : List Nat := 23143 :: root200Tail56
def root200Tail54 : List Nat := 23011 :: root200Tail55
def root200Tail53 : List Nat := 22993 :: root200Tail54
def root200Tail52 : List Nat := 22639 :: root200Tail53
def root200Tail51 : List Nat := 22279 :: root200Tail52
def root200Tail50 : List Nat := 20959 :: root200Tail51
def root200Tail49 : List Nat := 20347 :: root200Tail50
def root200Tail48 : List Nat := 20029 :: root200Tail49
def root200Tail47 : List Nat := 20011 :: root200Tail48
def root200Tail46 : List Nat := 19867 :: root200Tail47
def root200Tail45 : List Nat := 19819 :: root200Tail46
def root200Tail44 : List Nat := 19801 :: root200Tail45
def root200Tail43 : List Nat := 18433 :: root200Tail44
def root200Tail42 : List Nat := 18229 :: root200Tail43
def root200Tail41 : List Nat := 18097 :: root200Tail42
def root200Tail40 : List Nat := 17599 :: root200Tail41
def root200Tail39 : List Nat := 16981 :: root200Tail40
def root200Tail38 : List Nat := 16603 :: root200Tail39
def root200Tail37 : List Nat := 16519 :: root200Tail38
def root200Tail36 : List Nat := 16249 :: root200Tail37
def root200Tail35 : List Nat := 16183 :: root200Tail36
def root200Tail34 : List Nat := 15877 :: root200Tail35
def root200Tail33 : List Nat := 15391 :: root200Tail34
def root200Tail32 : List Nat := 14887 :: root200Tail33
def root200Tail31 : List Nat := 14821 :: root200Tail32
def root200Tail30 : List Nat := 14653 :: root200Tail31
def root200Tail29 : List Nat := 13999 :: root200Tail30
def root200Tail28 : List Nat := 13399 :: root200Tail29
def root200Tail27 : List Nat := 12409 :: root200Tail28
def root200Tail26 : List Nat := 10831 :: root200Tail27
def root200Tail25 : List Nat := 9973 :: root200Tail26
def root200Tail24 : List Nat := 9931 :: root200Tail25
def root200Tail23 : List Nat := 9883 :: root200Tail24
def root200Tail22 : List Nat := 9817 :: root200Tail23
def root200Tail21 : List Nat := 9739 :: root200Tail22
def root200Tail20 : List Nat := 9343 :: root200Tail21
def root200Tail19 : List Nat := 9067 :: root200Tail20
def root200Tail18 : List Nat := 8713 :: root200Tail19
def root200Tail17 : List Nat := 7789 :: root200Tail18
def root200Tail16 : List Nat := 6673 :: root200Tail17
def root200Tail15 : List Nat := 6571 :: root200Tail16
def root200Tail14 : List Nat := 6271 :: root200Tail15
def root200Tail13 : List Nat := 5743 :: root200Tail14
def root200Tail12 : List Nat := 5557 :: root200Tail13
def root200Tail11 : List Nat := 4993 :: root200Tail12
def root200Tail10 : List Nat := 4027 :: root200Tail11
def root200Tail9 : List Nat := 3307 :: root200Tail10
def root200Tail8 : List Nat := 3253 :: root200Tail9
def root200Tail7 : List Nat := 3187 :: root200Tail8
def root200Tail6 : List Nat := 3019 :: root200Tail7
def root200Tail5 : List Nat := 2341 :: root200Tail6
def root200Tail4 : List Nat := 1753 :: root200Tail5
def root200Tail3 : List Nat := 1741 :: root200Tail4
def root200Tail2 : List Nat := 1657 :: root200Tail3
def root200Tail1 : List Nat := 1549 :: root200Tail2
def root200Tail0 : List Nat := 1543 :: root200Tail1
def root200Candidates0 : List Nat := [6571, 9883, 14653]
set_option maxRecDepth 8192 in
theorem root200Candidates0_checked : candidates edge 4 1237 1543 root200Tail1 winner.bound = root200Candidates0 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root200Child0 : search edge 3 [1237, 1543] root200Candidates0 2780 winner = winner := by decide
theorem root200Step0 : scan edge (search edge 3) 4 [1237] 1237 root200Tail0 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail1 winner := by
  rw [root200Tail0, scan, if_neg (by decide), root200Candidates0_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail1
    (if 3 ≤ root200Candidates0.length then search edge 3 [1237, 1543] root200Candidates0 2780 winner else winner) = _
  rw [if_pos (by decide), root200Child0]
def root200Candidates1 : List Nat := [3019, 3253, 19819]
set_option maxRecDepth 8192 in
theorem root200Candidates1_checked : candidates edge 4 1237 1549 root200Tail2 winner.bound = root200Candidates1 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root200Child1 : search edge 3 [1237, 1549] root200Candidates1 2786 winner = winner := by decide
theorem root200Step1 : scan edge (search edge 3) 4 [1237] 1237 root200Tail1 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail2 winner := by
  rw [root200Tail1, scan, if_neg (by decide), root200Candidates1_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail2
    (if 3 ≤ root200Candidates1.length then search edge 3 [1237, 1549] root200Candidates1 2786 winner else winner) = _
  rw [if_pos (by decide), root200Child1]
def root200Candidates2 : List Nat := [9817, 9931]
set_option maxRecDepth 8192 in
theorem root200Candidates2_checked : candidates edge 4 1237 1657 root200Tail3 winner.bound = root200Candidates2 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
theorem root200Step2 : scan edge (search edge 3) 4 [1237] 1237 root200Tail2 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail3 winner := by
  rw [root200Tail2, scan, if_neg (by decide), root200Candidates2_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail3
    (if 3 ≤ root200Candidates2.length then search edge 3 [1237, 1657] root200Candidates2 2894 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates3 : List Nat := [1753, 3307, 5557, 14653, 14887]
set_option maxRecDepth 8192 in
theorem root200Candidates3_checked : candidates edge 4 1237 1741 root200Tail4 winner.bound = root200Candidates3 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root200Child3 : search edge 3 [1237, 1741] root200Candidates3 2978 winner = winner := by decide
theorem root200Step3 : scan edge (search edge 3) 4 [1237] 1237 root200Tail3 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail4 winner := by
  rw [root200Tail3, scan, if_neg (by decide), root200Candidates3_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail4
    (if 3 ≤ root200Candidates3.length then search edge 3 [1237, 1741] root200Candidates3 2978 winner else winner) = _
  rw [if_pos (by decide), root200Child3]
def root200Candidates4 : List Nat := [7789]
set_option maxRecDepth 8192 in
theorem root200Candidates4_checked : candidates edge 4 1237 1753 root200Tail5 winner.bound = root200Candidates4 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
theorem root200Step4 : scan edge (search edge 3) 4 [1237] 1237 root200Tail4 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail5 winner := by
  rw [root200Tail4, scan, if_neg (by decide), root200Candidates4_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail5
    (if 3 ≤ root200Candidates4.length then search edge 3 [1237, 1753] root200Candidates4 2990 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates5 : List Nat := [6571, 12409, 18433]
set_option maxRecDepth 8192 in
theorem root200Candidates5_checked : candidates edge 4 1237 2341 root200Tail6 winner.bound = root200Candidates5 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root200Child5 : search edge 3 [1237, 2341] root200Candidates5 3578 winner = winner := by decide
theorem root200Step5 : scan edge (search edge 3) 4 [1237] 1237 root200Tail5 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail6 winner := by
  rw [root200Tail5, scan, if_neg (by decide), root200Candidates5_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail6
    (if 3 ≤ root200Candidates5.length then search edge 3 [1237, 2341] root200Candidates5 3578 winner else winner) = _
  rw [if_pos (by decide), root200Child5]
def root200Candidates6 : List Nat := [6571, 13399]
set_option maxRecDepth 8192 in
theorem root200Candidates6_checked : candidates edge 4 1237 3019 root200Tail7 winner.bound = root200Candidates6 := by
  unfold candidates
  have row : edge 3019 = (SortedNatTree.contains FastRows.row3019) := by
    change Data.row3019.contains = (SortedNatTree.contains FastRows.row3019)
    exact FastRows.row3019_checked.symm
  rw [row]
  decide
theorem root200Step6 : scan edge (search edge 3) 4 [1237] 1237 root200Tail6 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail7 winner := by
  rw [root200Tail6, scan, if_neg (by decide), root200Candidates6_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail7
    (if 3 ≤ root200Candidates6.length then search edge 3 [1237, 3019] root200Candidates6 4256 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root200Candidates7_checked : candidates edge 4 1237 3187 root200Tail8 winner.bound = root200Candidates7 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root200Step7 : scan edge (search edge 3) 4 [1237] 1237 root200Tail7 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail8 winner := by
  rw [root200Tail7, scan, if_neg (by decide), root200Candidates7_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail8
    (if 3 ≤ root200Candidates7.length then search edge 3 [1237, 3187] root200Candidates7 4424 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates8 : List Nat := [5557, 6571, 9817, 9973, 16603]
set_option maxRecDepth 8192 in
theorem root200Candidates8_checked : candidates edge 4 1237 3253 root200Tail9 winner.bound = root200Candidates8 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root200Child8 : search edge 3 [1237, 3253] root200Candidates8 4490 winner = winner := by decide
theorem root200Step8 : scan edge (search edge 3) 4 [1237] 1237 root200Tail8 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail9 winner := by
  rw [root200Tail8, scan, if_neg (by decide), root200Candidates8_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail9
    (if 3 ≤ root200Candidates8.length then search edge 3 [1237, 3253] root200Candidates8 4490 winner else winner) = _
  rw [if_pos (by decide), root200Child8]
def root200Candidates9 : List Nat := [8713, 9973]
set_option maxRecDepth 8192 in
theorem root200Candidates9_checked : candidates edge 4 1237 3307 root200Tail10 winner.bound = root200Candidates9 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root200Step9 : scan edge (search edge 3) 4 [1237] 1237 root200Tail9 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail10 winner := by
  rw [root200Tail9, scan, if_neg (by decide), root200Candidates9_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail10
    (if 3 ≤ root200Candidates9.length then search edge 3 [1237, 3307] root200Candidates9 4544 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root200Candidates10_checked : candidates edge 4 1237 4027 root200Tail11 winner.bound = root200Candidates10 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
theorem root200Step10 : scan edge (search edge 3) 4 [1237] 1237 root200Tail10 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail11 winner := by
  rw [root200Tail10, scan, if_neg (by decide), root200Candidates10_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail11
    (if 3 ≤ root200Candidates10.length then search edge 3 [1237, 4027] root200Candidates10 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates11 : List Nat := [6571]
set_option maxRecDepth 8192 in
theorem root200Candidates11_checked : candidates edge 4 1237 4993 root200Tail12 winner.bound = root200Candidates11 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root200Step11 : scan edge (search edge 3) 4 [1237] 1237 root200Tail11 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail12 winner := by
  rw [root200Tail11, scan, if_neg (by decide), root200Candidates11_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail12
    (if 3 ≤ root200Candidates11.length then search edge 3 [1237, 4993] root200Candidates11 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates12 : List Nat := [7789]
set_option maxRecDepth 8192 in
theorem root200Candidates12_checked : candidates edge 4 1237 5557 root200Tail13 winner.bound = root200Candidates12 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root200Step12 : scan edge (search edge 3) 4 [1237] 1237 root200Tail12 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail13 winner := by
  rw [root200Tail12, scan, if_neg (by decide), root200Candidates12_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail13
    (if 3 ≤ root200Candidates12.length then search edge 3 [1237, 5557] root200Candidates12 6794 winner else winner) = _
  rw [if_neg (by decide)]
def root200Candidates13 : List Nat := [6271, 9883]
set_option maxRecDepth 8192 in
theorem root200Candidates13_checked : candidates edge 4 1237 5743 root200Tail14 winner.bound = root200Candidates13 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root200Step13 : scan edge (search edge 3) 4 [1237] 1237 root200Tail13 winner =
    scan edge (search edge 3) 4 [1237] 1237 root200Tail14 winner := by
  rw [root200Tail13, scan, if_neg (by decide), root200Candidates13_checked]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail14
    (if 3 ≤ root200Candidates13.length then search edge 3 [1237, 5743] root200Candidates13 6980 winner else winner) = _
  rw [if_neg (by decide)]
theorem root200Scan14 : scan edge (search edge 3) 4 [1237] 1237 root200Tail14 winner = winner := by rw [root200Tail14, scan, if_pos (by decide)]
theorem root200Scan13 : scan edge (search edge 3) 4 [1237] 1237 root200Tail13 winner = winner :=
  (root200Step13).trans root200Scan14
theorem root200Scan12 : scan edge (search edge 3) 4 [1237] 1237 root200Tail12 winner = winner :=
  (root200Step12).trans root200Scan13
theorem root200Scan11 : scan edge (search edge 3) 4 [1237] 1237 root200Tail11 winner = winner :=
  (root200Step11).trans root200Scan12
theorem root200Scan10 : scan edge (search edge 3) 4 [1237] 1237 root200Tail10 winner = winner :=
  (root200Step10).trans root200Scan11
theorem root200Scan9 : scan edge (search edge 3) 4 [1237] 1237 root200Tail9 winner = winner :=
  (root200Step9).trans root200Scan10
theorem root200Scan8 : scan edge (search edge 3) 4 [1237] 1237 root200Tail8 winner = winner :=
  (root200Step8).trans root200Scan9
theorem root200Scan7 : scan edge (search edge 3) 4 [1237] 1237 root200Tail7 winner = winner :=
  (root200Step7).trans root200Scan8
theorem root200Scan6 : scan edge (search edge 3) 4 [1237] 1237 root200Tail6 winner = winner :=
  (root200Step6).trans root200Scan7
theorem root200Scan5 : scan edge (search edge 3) 4 [1237] 1237 root200Tail5 winner = winner :=
  (root200Step5).trans root200Scan6
theorem root200Scan4 : scan edge (search edge 3) 4 [1237] 1237 root200Tail4 winner = winner :=
  (root200Step4).trans root200Scan5
theorem root200Scan3 : scan edge (search edge 3) 4 [1237] 1237 root200Tail3 winner = winner :=
  (root200Step3).trans root200Scan4
theorem root200Scan2 : scan edge (search edge 3) 4 [1237] 1237 root200Tail2 winner = winner :=
  (root200Step2).trans root200Scan3
theorem root200Scan1 : scan edge (search edge 3) 4 [1237] 1237 root200Tail1 winner = winner :=
  (root200Step1).trans root200Scan2
theorem root200Scan0 : scan edge (search edge 3) 4 [1237] 1237 root200Tail0 winner = winner :=
  (root200Step0).trans root200Scan1
set_option maxRecDepth 4096 in
theorem root200Branch : search edge 4 [1237] root200Candidates 1237 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1237] 1237 root200Tail0 winner = winner
  exact root200Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
