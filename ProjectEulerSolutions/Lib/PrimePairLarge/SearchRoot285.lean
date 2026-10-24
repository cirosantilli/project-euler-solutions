import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root285Candidates : List Nat := Data.row1877
set_option maxRecDepth 8192 in
theorem root285Candidates_checked : candidates edge 5 0 1877 tail286 winner.bound = root285Candidates := by
  apply candidates_eq_row edge 5 0 1877 tail286 root285Candidates winner.bound tail286_ordered
  · decide
  · decide
  · funext q
    rfl
def root285Tail56 : List Nat := []
def root285Tail55 : List Nat := 24107 :: root285Tail56
def root285Tail54 : List Nat := 24077 :: root285Tail55
def root285Tail53 : List Nat := 23561 :: root285Tail54
def root285Tail52 : List Nat := 23201 :: root285Tail53
def root285Tail51 : List Nat := 23087 :: root285Tail52
def root285Tail50 : List Nat := 23057 :: root285Tail51
def root285Tail49 : List Nat := 22751 :: root285Tail50
def root285Tail48 : List Nat := 22397 :: root285Tail49
def root285Tail47 : List Nat := 22229 :: root285Tail48
def root285Tail46 : List Nat := 21851 :: root285Tail47
def root285Tail45 : List Nat := 21323 :: root285Tail46
def root285Tail44 : List Nat := 20441 :: root285Tail45
def root285Tail43 : List Nat := 20177 :: root285Tail44
def root285Tail42 : List Nat := 19751 :: root285Tail43
def root285Tail41 : List Nat := 19211 :: root285Tail42
def root285Tail40 : List Nat := 18731 :: root285Tail41
def root285Tail39 : List Nat := 17333 :: root285Tail40
def root285Tail38 : List Nat := 16649 :: root285Tail39
def root285Tail37 : List Nat := 16223 :: root285Tail38
def root285Tail36 : List Nat := 15263 :: root285Tail37
def root285Tail35 : List Nat := 14717 :: root285Tail36
def root285Tail34 : List Nat := 14627 :: root285Tail35
def root285Tail33 : List Nat := 14591 :: root285Tail34
def root285Tail32 : List Nat := 13931 :: root285Tail33
def root285Tail31 : List Nat := 13499 :: root285Tail32
def root285Tail30 : List Nat := 13037 :: root285Tail31
def root285Tail29 : List Nat := 12809 :: root285Tail30
def root285Tail28 : List Nat := 12197 :: root285Tail29
def root285Tail27 : List Nat := 11087 :: root285Tail28
def root285Tail26 : List Nat := 11003 :: root285Tail27
def root285Tail25 : List Nat := 10739 :: root285Tail26
def root285Tail24 : List Nat := 10499 :: root285Tail25
def root285Tail23 : List Nat := 9851 :: root285Tail24
def root285Tail22 : List Nat := 9533 :: root285Tail23
def root285Tail21 : List Nat := 9209 :: root285Tail22
def root285Tail20 : List Nat := 8597 :: root285Tail21
def root285Tail19 : List Nat := 8087 :: root285Tail20
def root285Tail18 : List Nat := 7823 :: root285Tail19
def root285Tail17 : List Nat := 6983 :: root285Tail18
def root285Tail16 : List Nat := 6971 :: root285Tail17
def root285Tail15 : List Nat := 6491 :: root285Tail16
def root285Tail14 : List Nat := 6287 :: root285Tail15
def root285Tail13 : List Nat := 6113 :: root285Tail14
def root285Tail12 : List Nat := 5711 :: root285Tail13
def root285Tail11 : List Nat := 5399 :: root285Tail12
def root285Tail10 : List Nat := 5261 :: root285Tail11
def root285Tail9 : List Nat := 4733 :: root285Tail10
def root285Tail8 : List Nat := 4421 :: root285Tail9
def root285Tail7 : List Nat := 4391 :: root285Tail8
def root285Tail6 : List Nat := 4211 :: root285Tail7
def root285Tail5 : List Nat := 3917 :: root285Tail6
def root285Tail4 : List Nat := 3119 :: root285Tail5
def root285Tail3 : List Nat := 2909 :: root285Tail4
def root285Tail2 : List Nat := 2549 :: root285Tail3
def root285Tail1 : List Nat := 2447 :: root285Tail2
def root285Tail0 : List Nat := 1889 :: root285Tail1
def root285Candidates0 : List Nat := [9533, 13499]
set_option maxRecDepth 8192 in
theorem root285Candidates0_checked : candidates edge 4 1877 1889 root285Tail1 winner.bound = root285Candidates0 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root285Step0 : scan edge (search edge 3) 4 [1877] 1877 root285Tail0 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail1 winner := by
  rw [root285Tail0, scan, if_neg (by decide), root285Candidates0_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail1
    (if 3 ≤ root285Candidates0.length then search edge 3 [1877, 1889] root285Candidates0 3766 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates1 : List Nat := [6113, 12197]
set_option maxRecDepth 8192 in
theorem root285Candidates1_checked : candidates edge 4 1877 2447 root285Tail2 winner.bound = root285Candidates1 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root285Step1 : scan edge (search edge 3) 4 [1877] 1877 root285Tail1 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail2 winner := by
  rw [root285Tail1, scan, if_neg (by decide), root285Candidates1_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail2
    (if 3 ≤ root285Candidates1.length then search edge 3 [1877, 2447] root285Candidates1 4324 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates2 : List Nat := [9209, 14717]
set_option maxRecDepth 8192 in
theorem root285Candidates2_checked : candidates edge 4 1877 2549 root285Tail3 winner.bound = root285Candidates2 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root285Step2 : scan edge (search edge 3) 4 [1877] 1877 root285Tail2 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail3 winner := by
  rw [root285Tail2, scan, if_neg (by decide), root285Candidates2_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail3
    (if 3 ≤ root285Candidates2.length then search edge 3 [1877, 2549] root285Candidates2 4426 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates3 : List Nat := [6983]
set_option maxRecDepth 8192 in
theorem root285Candidates3_checked : candidates edge 4 1877 2909 root285Tail4 winner.bound = root285Candidates3 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
theorem root285Step3 : scan edge (search edge 3) 4 [1877] 1877 root285Tail3 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail4 winner := by
  rw [root285Tail3, scan, if_neg (by decide), root285Candidates3_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail4
    (if 3 ≤ root285Candidates3.length then search edge 3 [1877, 2909] root285Candidates3 4786 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates4 : List Nat := [4391, 6491, 16223]
set_option maxRecDepth 8192 in
theorem root285Candidates4_checked : candidates edge 4 1877 3119 root285Tail5 winner.bound = root285Candidates4 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root285Child4 : search edge 3 [1877, 3119] root285Candidates4 4996 winner = winner := by decide
theorem root285Step4 : scan edge (search edge 3) 4 [1877] 1877 root285Tail4 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail5 winner := by
  rw [root285Tail4, scan, if_neg (by decide), root285Candidates4_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail5
    (if 3 ≤ root285Candidates4.length then search edge 3 [1877, 3119] root285Candidates4 4996 winner else winner) = _
  rw [if_pos (by decide), root285Child4]
def root285Candidates5 : List Nat := [4211, 5399]
set_option maxRecDepth 8192 in
theorem root285Candidates5_checked : candidates edge 4 1877 3917 root285Tail6 winner.bound = root285Candidates5 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root285Step5 : scan edge (search edge 3) 4 [1877] 1877 root285Tail5 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail6 winner := by
  rw [root285Tail5, scan, if_neg (by decide), root285Candidates5_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail6
    (if 3 ≤ root285Candidates5.length then search edge 3 [1877, 3917] root285Candidates5 5794 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates6 : List Nat := [14717]
set_option maxRecDepth 8192 in
theorem root285Candidates6_checked : candidates edge 4 1877 4211 root285Tail7 winner.bound = root285Candidates6 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root285Step6 : scan edge (search edge 3) 4 [1877] 1877 root285Tail6 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail7 winner := by
  rw [root285Tail6, scan, if_neg (by decide), root285Candidates6_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail7
    (if 3 ≤ root285Candidates6.length then search edge 3 [1877, 4211] root285Candidates6 6088 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates7 : List Nat := [10499]
set_option maxRecDepth 8192 in
theorem root285Candidates7_checked : candidates edge 4 1877 4391 root285Tail8 winner.bound = root285Candidates7 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root285Step7 : scan edge (search edge 3) 4 [1877] 1877 root285Tail7 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail8 winner := by
  rw [root285Tail7, scan, if_neg (by decide), root285Candidates7_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail8
    (if 3 ≤ root285Candidates7.length then search edge 3 [1877, 4391] root285Candidates7 6268 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates8 : List Nat := [6113]
set_option maxRecDepth 8192 in
theorem root285Candidates8_checked : candidates edge 4 1877 4421 root285Tail9 winner.bound = root285Candidates8 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root285Step8 : scan edge (search edge 3) 4 [1877] 1877 root285Tail8 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail9 winner := by
  rw [root285Tail8, scan, if_neg (by decide), root285Candidates8_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail9
    (if 3 ≤ root285Candidates8.length then search edge 3 [1877, 4421] root285Candidates8 6298 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates9 : List Nat := [16223]
set_option maxRecDepth 8192 in
theorem root285Candidates9_checked : candidates edge 4 1877 4733 root285Tail10 winner.bound = root285Candidates9 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root285Step9 : scan edge (search edge 3) 4 [1877] 1877 root285Tail9 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail10 winner := by
  rw [root285Tail9, scan, if_neg (by decide), root285Candidates9_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail10
    (if 3 ≤ root285Candidates9.length then search edge 3 [1877, 4733] root285Candidates9 6610 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates10 : List Nat := [6983]
set_option maxRecDepth 8192 in
theorem root285Candidates10_checked : candidates edge 4 1877 5261 root285Tail11 winner.bound = root285Candidates10 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root285Step10 : scan edge (search edge 3) 4 [1877] 1877 root285Tail10 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail11 winner := by
  rw [root285Tail10, scan, if_neg (by decide), root285Candidates10_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail11
    (if 3 ≤ root285Candidates10.length then search edge 3 [1877, 5261] root285Candidates10 7138 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root285Candidates11_checked : candidates edge 4 1877 5399 root285Tail12 winner.bound = root285Candidates11 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root285Step11 : scan edge (search edge 3) 4 [1877] 1877 root285Tail11 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail12 winner := by
  rw [root285Tail11, scan, if_neg (by decide), root285Candidates11_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail12
    (if 3 ≤ root285Candidates11.length then search edge 3 [1877, 5399] root285Candidates11 7276 winner else winner) = _
  rw [if_neg (by decide)]
def root285Candidates12 : List Nat := [11003, 14717]
set_option maxRecDepth 8192 in
theorem root285Candidates12_checked : candidates edge 4 1877 5711 root285Tail13 winner.bound = root285Candidates12 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root285Step12 : scan edge (search edge 3) 4 [1877] 1877 root285Tail12 winner =
    scan edge (search edge 3) 4 [1877] 1877 root285Tail13 winner := by
  rw [root285Tail12, scan, if_neg (by decide), root285Candidates12_checked]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail13
    (if 3 ≤ root285Candidates12.length then search edge 3 [1877, 5711] root285Candidates12 7588 winner else winner) = _
  rw [if_neg (by decide)]
theorem root285Scan13 : scan edge (search edge 3) 4 [1877] 1877 root285Tail13 winner = winner := by rw [root285Tail13, scan, if_pos (by decide)]
theorem root285Scan12 : scan edge (search edge 3) 4 [1877] 1877 root285Tail12 winner = winner :=
  (root285Step12).trans root285Scan13
theorem root285Scan11 : scan edge (search edge 3) 4 [1877] 1877 root285Tail11 winner = winner :=
  (root285Step11).trans root285Scan12
theorem root285Scan10 : scan edge (search edge 3) 4 [1877] 1877 root285Tail10 winner = winner :=
  (root285Step10).trans root285Scan11
theorem root285Scan9 : scan edge (search edge 3) 4 [1877] 1877 root285Tail9 winner = winner :=
  (root285Step9).trans root285Scan10
theorem root285Scan8 : scan edge (search edge 3) 4 [1877] 1877 root285Tail8 winner = winner :=
  (root285Step8).trans root285Scan9
theorem root285Scan7 : scan edge (search edge 3) 4 [1877] 1877 root285Tail7 winner = winner :=
  (root285Step7).trans root285Scan8
theorem root285Scan6 : scan edge (search edge 3) 4 [1877] 1877 root285Tail6 winner = winner :=
  (root285Step6).trans root285Scan7
theorem root285Scan5 : scan edge (search edge 3) 4 [1877] 1877 root285Tail5 winner = winner :=
  (root285Step5).trans root285Scan6
theorem root285Scan4 : scan edge (search edge 3) 4 [1877] 1877 root285Tail4 winner = winner :=
  (root285Step4).trans root285Scan5
theorem root285Scan3 : scan edge (search edge 3) 4 [1877] 1877 root285Tail3 winner = winner :=
  (root285Step3).trans root285Scan4
theorem root285Scan2 : scan edge (search edge 3) 4 [1877] 1877 root285Tail2 winner = winner :=
  (root285Step2).trans root285Scan3
theorem root285Scan1 : scan edge (search edge 3) 4 [1877] 1877 root285Tail1 winner = winner :=
  (root285Step1).trans root285Scan2
theorem root285Scan0 : scan edge (search edge 3) 4 [1877] 1877 root285Tail0 winner = winner :=
  (root285Step0).trans root285Scan1
set_option maxRecDepth 4096 in
theorem root285Branch : search edge 4 [1877] root285Candidates 1877 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1877] 1877 root285Tail0 winner = winner
  exact root285Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
