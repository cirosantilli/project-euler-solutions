import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root204Candidates : List Nat := Data.row1279
set_option maxRecDepth 8192 in
theorem root204Candidates_checked : candidates edge 5 0 1279 tail205 winner.bound = root204Candidates := by
  apply candidates_eq_row edge 5 0 1279 tail205 root204Candidates winner.bound tail205_ordered
  · decide
  · decide
  · funext q
    rfl
def root204Tail44 : List Nat := []
def root204Tail43 : List Nat := 24019 :: root204Tail44
def root204Tail42 : List Nat := 23197 :: root204Tail43
def root204Tail41 : List Nat := 22279 :: root204Tail42
def root204Tail40 : List Nat := 22273 :: root204Tail41
def root204Tail39 : List Nat := 20071 :: root204Tail40
def root204Tail38 : List Nat := 18457 :: root204Tail39
def root204Tail37 : List Nat := 18229 :: root204Tail38
def root204Tail36 : List Nat := 18199 :: root204Tail37
def root204Tail35 : List Nat := 17599 :: root204Tail36
def root204Tail34 : List Nat := 17431 :: root204Tail35
def root204Tail33 : List Nat := 15913 :: root204Tail34
def root204Tail32 : List Nat := 14797 :: root204Tail33
def root204Tail31 : List Nat := 14533 :: root204Tail32
def root204Tail30 : List Nat := 14107 :: root204Tail31
def root204Tail29 : List Nat := 13411 :: root204Tail30
def root204Tail28 : List Nat := 13183 :: root204Tail29
def root204Tail27 : List Nat := 12979 :: root204Tail28
def root204Tail26 : List Nat := 12973 :: root204Tail27
def root204Tail25 : List Nat := 12451 :: root204Tail26
def root204Tail24 : List Nat := 12073 :: root204Tail25
def root204Tail23 : List Nat := 12049 :: root204Tail24
def root204Tail22 : List Nat := 10009 :: root204Tail23
def root204Tail21 : List Nat := 9781 :: root204Tail22
def root204Tail20 : List Nat := 8641 :: root204Tail21
def root204Tail19 : List Nat := 8293 :: root204Tail20
def root204Tail18 : List Nat := 8101 :: root204Tail19
def root204Tail17 : List Nat := 7069 :: root204Tail18
def root204Tail16 : List Nat := 6781 :: root204Tail17
def root204Tail15 : List Nat := 6607 :: root204Tail16
def root204Tail14 : List Nat := 6271 :: root204Tail15
def root204Tail13 : List Nat := 6037 :: root204Tail14
def root204Tail12 : List Nat := 5479 :: root204Tail13
def root204Tail11 : List Nat := 5437 :: root204Tail12
def root204Tail10 : List Nat := 5179 :: root204Tail11
def root204Tail9 : List Nat := 5059 :: root204Tail10
def root204Tail8 : List Nat := 3691 :: root204Tail9
def root204Tail7 : List Nat := 3559 :: root204Tail8
def root204Tail6 : List Nat := 3463 :: root204Tail7
def root204Tail5 : List Nat := 2803 :: root204Tail6
def root204Tail4 : List Nat := 2389 :: root204Tail5
def root204Tail3 : List Nat := 2179 :: root204Tail4
def root204Tail2 : List Nat := 1459 :: root204Tail3
def root204Tail1 : List Nat := 1399 :: root204Tail2
def root204Tail0 : List Nat := 1297 :: root204Tail1
def root204Candidates0 : List Nat := [6781, 12049]
set_option maxRecDepth 8192 in
theorem root204Candidates0_checked : candidates edge 4 1279 1297 root204Tail1 winner.bound = root204Candidates0 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
theorem root204Step0 : scan edge (search edge 3) 4 [1279] 1279 root204Tail0 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail1 winner := by
  rw [root204Tail0, scan, if_neg (by decide), root204Candidates0_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail1
    (if 3 ≤ root204Candidates0.length then search edge 3 [1279, 1297] root204Candidates0 2576 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root204Candidates1_checked : candidates edge 4 1279 1399 root204Tail2 winner.bound = root204Candidates1 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
theorem root204Step1 : scan edge (search edge 3) 4 [1279] 1279 root204Tail1 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail2 winner := by
  rw [root204Tail1, scan, if_neg (by decide), root204Candidates1_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail2
    (if 3 ≤ root204Candidates1.length then search edge 3 [1279, 1399] root204Candidates1 2678 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates2 : List Nat := [5437, 6271, 14107]
set_option maxRecDepth 8192 in
theorem root204Candidates2_checked : candidates edge 4 1279 1459 root204Tail3 winner.bound = root204Candidates2 := by
  unfold candidates
  have row : edge 1459 = (SortedNatTree.contains FastRows.row1459) := by
    change Data.row1459.contains = (SortedNatTree.contains FastRows.row1459)
    exact FastRows.row1459_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root204Child2 : search edge 3 [1279, 1459] root204Candidates2 2738 winner = winner := by decide
theorem root204Step2 : scan edge (search edge 3) 4 [1279] 1279 root204Tail2 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail3 winner := by
  rw [root204Tail2, scan, if_neg (by decide), root204Candidates2_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail3
    (if 3 ≤ root204Candidates2.length then search edge 3 [1279, 1459] root204Candidates2 2738 winner else winner) = _
  rw [if_pos (by decide), root204Child2]
def root204Candidates3 : List Nat := [9781]
set_option maxRecDepth 8192 in
theorem root204Candidates3_checked : candidates edge 4 1279 2179 root204Tail4 winner.bound = root204Candidates3 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
theorem root204Step3 : scan edge (search edge 3) 4 [1279] 1279 root204Tail3 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail4 winner := by
  rw [root204Tail3, scan, if_neg (by decide), root204Candidates3_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail4
    (if 3 ≤ root204Candidates3.length then search edge 3 [1279, 2179] root204Candidates3 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates4 : List Nat := [5479]
set_option maxRecDepth 8192 in
theorem root204Candidates4_checked : candidates edge 4 1279 2389 root204Tail5 winner.bound = root204Candidates4 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
theorem root204Step4 : scan edge (search edge 3) 4 [1279] 1279 root204Tail4 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail5 winner := by
  rw [root204Tail4, scan, if_neg (by decide), root204Candidates4_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail5
    (if 3 ≤ root204Candidates4.length then search edge 3 [1279, 2389] root204Candidates4 3668 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root204Candidates5_checked : candidates edge 4 1279 2803 root204Tail6 winner.bound = root204Candidates5 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root204Step5 : scan edge (search edge 3) 4 [1279] 1279 root204Tail5 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail6 winner := by
  rw [root204Tail5, scan, if_neg (by decide), root204Candidates5_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail6
    (if 3 ≤ root204Candidates5.length then search edge 3 [1279, 2803] root204Candidates5 4082 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates6 : List Nat := [5479, 6037, 15913, 17599]
set_option maxRecDepth 8192 in
theorem root204Candidates6_checked : candidates edge 4 1279 3463 root204Tail7 winner.bound = root204Candidates6 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root204Child6 : search edge 3 [1279, 3463] root204Candidates6 4742 winner = winner := by decide
theorem root204Step6 : scan edge (search edge 3) 4 [1279] 1279 root204Tail6 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail7 winner := by
  rw [root204Tail6, scan, if_neg (by decide), root204Candidates6_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail7
    (if 3 ≤ root204Candidates6.length then search edge 3 [1279, 3463] root204Candidates6 4742 winner else winner) = _
  rw [if_pos (by decide), root204Child6]
def root204Candidates7 : List Nat := [6037, 6607]
set_option maxRecDepth 8192 in
theorem root204Candidates7_checked : candidates edge 4 1279 3559 root204Tail8 winner.bound = root204Candidates7 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
theorem root204Step7 : scan edge (search edge 3) 4 [1279] 1279 root204Tail7 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail8 winner := by
  rw [root204Tail7, scan, if_neg (by decide), root204Candidates7_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail8
    (if 3 ≤ root204Candidates7.length then search edge 3 [1279, 3559] root204Candidates7 4838 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root204Candidates8_checked : candidates edge 4 1279 3691 root204Tail9 winner.bound = root204Candidates8 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
theorem root204Step8 : scan edge (search edge 3) 4 [1279] 1279 root204Tail8 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail9 winner := by
  rw [root204Tail8, scan, if_neg (by decide), root204Candidates8_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail9
    (if 3 ≤ root204Candidates8.length then search edge 3 [1279, 3691] root204Candidates8 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates9 : List Nat := [12073]
set_option maxRecDepth 8192 in
theorem root204Candidates9_checked : candidates edge 4 1279 5059 root204Tail10 winner.bound = root204Candidates9 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root204Step9 : scan edge (search edge 3) 4 [1279] 1279 root204Tail9 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail10 winner := by
  rw [root204Tail9, scan, if_neg (by decide), root204Candidates9_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail10
    (if 3 ≤ root204Candidates9.length then search edge 3 [1279, 5059] root204Candidates9 6338 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root204Candidates10_checked : candidates edge 4 1279 5179 root204Tail11 winner.bound = root204Candidates10 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root204Step10 : scan edge (search edge 3) 4 [1279] 1279 root204Tail10 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail11 winner := by
  rw [root204Tail10, scan, if_neg (by decide), root204Candidates10_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail11
    (if 3 ≤ root204Candidates10.length then search edge 3 [1279, 5179] root204Candidates10 6458 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root204Candidates11_checked : candidates edge 4 1279 5437 root204Tail12 winner.bound = root204Candidates11 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root204Step11 : scan edge (search edge 3) 4 [1279] 1279 root204Tail11 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail12 winner := by
  rw [root204Tail11, scan, if_neg (by decide), root204Candidates11_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail12
    (if 3 ≤ root204Candidates11.length then search edge 3 [1279, 5437] root204Candidates11 6716 winner else winner) = _
  rw [if_neg (by decide)]
def root204Candidates12 : List Nat := [6271, 6607, 14797]
set_option maxRecDepth 8192 in
theorem root204Candidates12_checked : candidates edge 4 1279 5479 root204Tail13 winner.bound = root204Candidates12 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root204Child12 : search edge 3 [1279, 5479] root204Candidates12 6758 winner = winner := by decide
theorem root204Step12 : scan edge (search edge 3) 4 [1279] 1279 root204Tail12 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail13 winner := by
  rw [root204Tail12, scan, if_neg (by decide), root204Candidates12_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail13
    (if 3 ≤ root204Candidates12.length then search edge 3 [1279, 5479] root204Candidates12 6758 winner else winner) = _
  rw [if_pos (by decide), root204Child12]
def root204Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root204Candidates13_checked : candidates edge 4 1279 6037 root204Tail14 winner.bound = root204Candidates13 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root204Step13 : scan edge (search edge 3) 4 [1279] 1279 root204Tail13 winner =
    scan edge (search edge 3) 4 [1279] 1279 root204Tail14 winner := by
  rw [root204Tail13, scan, if_neg (by decide), root204Candidates13_checked]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail14
    (if 3 ≤ root204Candidates13.length then search edge 3 [1279, 6037] root204Candidates13 7316 winner else winner) = _
  rw [if_neg (by decide)]
theorem root204Scan14 : scan edge (search edge 3) 4 [1279] 1279 root204Tail14 winner = winner := by rw [root204Tail14, scan, if_pos (by decide)]
theorem root204Scan13 : scan edge (search edge 3) 4 [1279] 1279 root204Tail13 winner = winner :=
  (root204Step13).trans root204Scan14
theorem root204Scan12 : scan edge (search edge 3) 4 [1279] 1279 root204Tail12 winner = winner :=
  (root204Step12).trans root204Scan13
theorem root204Scan11 : scan edge (search edge 3) 4 [1279] 1279 root204Tail11 winner = winner :=
  (root204Step11).trans root204Scan12
theorem root204Scan10 : scan edge (search edge 3) 4 [1279] 1279 root204Tail10 winner = winner :=
  (root204Step10).trans root204Scan11
theorem root204Scan9 : scan edge (search edge 3) 4 [1279] 1279 root204Tail9 winner = winner :=
  (root204Step9).trans root204Scan10
theorem root204Scan8 : scan edge (search edge 3) 4 [1279] 1279 root204Tail8 winner = winner :=
  (root204Step8).trans root204Scan9
theorem root204Scan7 : scan edge (search edge 3) 4 [1279] 1279 root204Tail7 winner = winner :=
  (root204Step7).trans root204Scan8
theorem root204Scan6 : scan edge (search edge 3) 4 [1279] 1279 root204Tail6 winner = winner :=
  (root204Step6).trans root204Scan7
theorem root204Scan5 : scan edge (search edge 3) 4 [1279] 1279 root204Tail5 winner = winner :=
  (root204Step5).trans root204Scan6
theorem root204Scan4 : scan edge (search edge 3) 4 [1279] 1279 root204Tail4 winner = winner :=
  (root204Step4).trans root204Scan5
theorem root204Scan3 : scan edge (search edge 3) 4 [1279] 1279 root204Tail3 winner = winner :=
  (root204Step3).trans root204Scan4
theorem root204Scan2 : scan edge (search edge 3) 4 [1279] 1279 root204Tail2 winner = winner :=
  (root204Step2).trans root204Scan3
theorem root204Scan1 : scan edge (search edge 3) 4 [1279] 1279 root204Tail1 winner = winner :=
  (root204Step1).trans root204Scan2
theorem root204Scan0 : scan edge (search edge 3) 4 [1279] 1279 root204Tail0 winner = winner :=
  (root204Step0).trans root204Scan1
set_option maxRecDepth 4096 in
theorem root204Branch : search edge 4 [1279] root204Candidates 1279 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1279] 1279 root204Tail0 winner = winner
  exact root204Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
