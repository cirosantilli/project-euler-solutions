import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root239Candidates : List Nat := Data.row1531
set_option maxRecDepth 8192 in
theorem root239Candidates_checked : candidates edge 5 0 1531 tail240 winner.bound = root239Candidates := by
  apply candidates_eq_row edge 5 0 1531 tail240 root239Candidates winner.bound tail240_ordered
  · decide
  · decide
  · funext q
    rfl
def root239Tail52 : List Nat := []
def root239Tail51 : List Nat := 23977 :: root239Tail52
def root239Tail50 : List Nat := 23917 :: root239Tail51
def root239Tail49 : List Nat := 23767 :: root239Tail50
def root239Tail48 : List Nat := 23143 :: root239Tail49
def root239Tail47 : List Nat := 22807 :: root239Tail48
def root239Tail46 : List Nat := 22549 :: root239Tail47
def root239Tail45 : List Nat := 22447 :: root239Tail46
def root239Tail44 : List Nat := 21517 :: root239Tail45
def root239Tail43 : List Nat := 20533 :: root239Tail44
def root239Tail42 : List Nat := 20323 :: root239Tail43
def root239Tail41 : List Nat := 20089 :: root239Tail42
def root239Tail40 : List Nat := 19777 :: root239Tail41
def root239Tail39 : List Nat := 19753 :: root239Tail40
def root239Tail38 : List Nat := 18859 :: root239Tail39
def root239Tail37 : List Nat := 17467 :: root239Tail38
def root239Tail36 : List Nat := 17443 :: root239Tail37
def root239Tail35 : List Nat := 17383 :: root239Tail36
def root239Tail34 : List Nat := 14737 :: root239Tail35
def root239Tail33 : List Nat := 14593 :: root239Tail34
def root239Tail32 : List Nat := 13537 :: root239Tail33
def root239Tail31 : List Nat := 12979 :: root239Tail32
def root239Tail30 : List Nat := 12577 :: root239Tail31
def root239Tail29 : List Nat := 11731 :: root239Tail30
def root239Tail28 : List Nat := 11551 :: root239Tail29
def root239Tail27 : List Nat := 11353 :: root239Tail28
def root239Tail26 : List Nat := 11083 :: root239Tail27
def root239Tail25 : List Nat := 11047 :: root239Tail26
def root239Tail24 : List Nat := 10267 :: root239Tail25
def root239Tail23 : List Nat := 9547 :: root239Tail24
def root239Tail22 : List Nat := 9463 :: root239Tail23
def root239Tail21 : List Nat := 9151 :: root239Tail22
def root239Tail20 : List Nat := 9043 :: root239Tail21
def root239Tail19 : List Nat := 8311 :: root239Tail20
def root239Tail18 : List Nat := 7927 :: root239Tail19
def root239Tail17 : List Nat := 7789 :: root239Tail18
def root239Tail16 : List Nat := 7717 :: root239Tail17
def root239Tail15 : List Nat := 7681 :: root239Tail16
def root239Tail14 : List Nat := 7537 :: root239Tail15
def root239Tail13 : List Nat := 7213 :: root239Tail14
def root239Tail12 : List Nat := 6421 :: root239Tail13
def root239Tail11 : List Nat := 6361 :: root239Tail12
def root239Tail10 : List Nat := 6211 :: root239Tail11
def root239Tail9 : List Nat := 5647 :: root239Tail10
def root239Tail8 : List Nat := 5281 :: root239Tail9
def root239Tail7 : List Nat := 5119 :: root239Tail8
def root239Tail6 : List Nat := 5059 :: root239Tail7
def root239Tail5 : List Nat := 4177 :: root239Tail6
def root239Tail4 : List Nat := 4153 :: root239Tail5
def root239Tail3 : List Nat := 3169 :: root239Tail4
def root239Tail2 : List Nat := 2677 :: root239Tail3
def root239Tail1 : List Nat := 2467 :: root239Tail2
def root239Tail0 : List Nat := 1987 :: root239Tail1
def root239Candidates0 : List Nat := [2467, 11353]
set_option maxRecDepth 8192 in
theorem root239Candidates0_checked : candidates edge 4 1531 1987 root239Tail1 winner.bound = root239Candidates0 := by
  unfold candidates
  have row : edge 1987 = (SortedNatTree.contains FastRows.row1987) := by
    change Data.row1987.contains = (SortedNatTree.contains FastRows.row1987)
    exact FastRows.row1987_checked.symm
  rw [row]
  decide
theorem root239Step0 : scan edge (search edge 3) 4 [1531] 1531 root239Tail0 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail1 winner := by
  rw [root239Tail0, scan, if_neg (by decide), root239Candidates0_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail1
    (if 3 ≤ root239Candidates0.length then search edge 3 [1531, 1987] root239Candidates0 3518 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root239Candidates1_checked : candidates edge 4 1531 2467 root239Tail2 winner.bound = root239Candidates1 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root239Step1 : scan edge (search edge 3) 4 [1531] 1531 root239Tail1 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail2 winner := by
  rw [root239Tail1, scan, if_neg (by decide), root239Candidates1_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail2
    (if 3 ≤ root239Candidates1.length then search edge 3 [1531, 2467] root239Candidates1 3998 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates2 : List Nat := [9043]
set_option maxRecDepth 8192 in
theorem root239Candidates2_checked : candidates edge 4 1531 2677 root239Tail3 winner.bound = root239Candidates2 := by
  unfold candidates
  have row : edge 2677 = (SortedNatTree.contains FastRows.row2677) := by
    change Data.row2677.contains = (SortedNatTree.contains FastRows.row2677)
    exact FastRows.row2677_checked.symm
  rw [row]
  decide
theorem root239Step2 : scan edge (search edge 3) 4 [1531] 1531 root239Tail2 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail3 winner := by
  rw [root239Tail2, scan, if_neg (by decide), root239Candidates2_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail3
    (if 3 ≤ root239Candidates2.length then search edge 3 [1531, 2677] root239Candidates2 4208 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates3 : List Nat := [5119, 6361, 7213]
set_option maxRecDepth 8192 in
theorem root239Candidates3_checked : candidates edge 4 1531 3169 root239Tail4 winner.bound = root239Candidates3 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root239Child3 : search edge 3 [1531, 3169] root239Candidates3 4700 winner = winner := by decide
theorem root239Step3 : scan edge (search edge 3) 4 [1531] 1531 root239Tail3 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail4 winner := by
  rw [root239Tail3, scan, if_neg (by decide), root239Candidates3_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail4
    (if 3 ≤ root239Candidates3.length then search edge 3 [1531, 3169] root239Candidates3 4700 winner else winner) = _
  rw [if_pos (by decide), root239Child3]
def root239Candidates4 : List Nat := [8311, 11353, 19753]
set_option maxRecDepth 8192 in
theorem root239Candidates4_checked : candidates edge 4 1531 4153 root239Tail5 winner.bound = root239Candidates4 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root239Child4 : search edge 3 [1531, 4153] root239Candidates4 5684 winner = winner := by decide
theorem root239Step4 : scan edge (search edge 3) 4 [1531] 1531 root239Tail4 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail5 winner := by
  rw [root239Tail4, scan, if_neg (by decide), root239Candidates4_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail5
    (if 3 ≤ root239Candidates4.length then search edge 3 [1531, 4153] root239Candidates4 5684 winner else winner) = _
  rw [if_pos (by decide), root239Child4]
def root239Candidates5 : List Nat := [7537, 8311]
set_option maxRecDepth 8192 in
theorem root239Candidates5_checked : candidates edge 4 1531 4177 root239Tail6 winner.bound = root239Candidates5 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root239Step5 : scan edge (search edge 3) 4 [1531] 1531 root239Tail5 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail6 winner := by
  rw [root239Tail5, scan, if_neg (by decide), root239Candidates5_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail6
    (if 3 ≤ root239Candidates5.length then search edge 3 [1531, 4177] root239Candidates5 5708 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates6 : List Nat := [7213]
set_option maxRecDepth 8192 in
theorem root239Candidates6_checked : candidates edge 4 1531 5059 root239Tail7 winner.bound = root239Candidates6 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root239Step6 : scan edge (search edge 3) 4 [1531] 1531 root239Tail6 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail7 winner := by
  rw [root239Tail6, scan, if_neg (by decide), root239Candidates6_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail7
    (if 3 ≤ root239Candidates6.length then search edge 3 [1531, 5059] root239Candidates6 6590 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates7 : List Nat := [7537, 11731]
set_option maxRecDepth 8192 in
theorem root239Candidates7_checked : candidates edge 4 1531 5119 root239Tail8 winner.bound = root239Candidates7 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root239Step7 : scan edge (search edge 3) 4 [1531] 1531 root239Tail7 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail8 winner := by
  rw [root239Tail7, scan, if_neg (by decide), root239Candidates7_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail8
    (if 3 ≤ root239Candidates7.length then search edge 3 [1531, 5119] root239Candidates7 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root239Candidates8_checked : candidates edge 4 1531 5281 root239Tail9 winner.bound = root239Candidates8 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root239Step8 : scan edge (search edge 3) 4 [1531] 1531 root239Tail8 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail9 winner := by
  rw [root239Tail8, scan, if_neg (by decide), root239Candidates8_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail9
    (if 3 ≤ root239Candidates8.length then search edge 3 [1531, 5281] root239Candidates8 6812 winner else winner) = _
  rw [if_neg (by decide)]
def root239Candidates9 : List Nat := [9043, 17467]
set_option maxRecDepth 8192 in
theorem root239Candidates9_checked : candidates edge 4 1531 5647 root239Tail10 winner.bound = root239Candidates9 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root239Step9 : scan edge (search edge 3) 4 [1531] 1531 root239Tail9 winner =
    scan edge (search edge 3) 4 [1531] 1531 root239Tail10 winner := by
  rw [root239Tail9, scan, if_neg (by decide), root239Candidates9_checked]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail10
    (if 3 ≤ root239Candidates9.length then search edge 3 [1531, 5647] root239Candidates9 7178 winner else winner) = _
  rw [if_neg (by decide)]
theorem root239Scan10 : scan edge (search edge 3) 4 [1531] 1531 root239Tail10 winner = winner := by rw [root239Tail10, scan, if_pos (by decide)]
theorem root239Scan9 : scan edge (search edge 3) 4 [1531] 1531 root239Tail9 winner = winner :=
  (root239Step9).trans root239Scan10
theorem root239Scan8 : scan edge (search edge 3) 4 [1531] 1531 root239Tail8 winner = winner :=
  (root239Step8).trans root239Scan9
theorem root239Scan7 : scan edge (search edge 3) 4 [1531] 1531 root239Tail7 winner = winner :=
  (root239Step7).trans root239Scan8
theorem root239Scan6 : scan edge (search edge 3) 4 [1531] 1531 root239Tail6 winner = winner :=
  (root239Step6).trans root239Scan7
theorem root239Scan5 : scan edge (search edge 3) 4 [1531] 1531 root239Tail5 winner = winner :=
  (root239Step5).trans root239Scan6
theorem root239Scan4 : scan edge (search edge 3) 4 [1531] 1531 root239Tail4 winner = winner :=
  (root239Step4).trans root239Scan5
theorem root239Scan3 : scan edge (search edge 3) 4 [1531] 1531 root239Tail3 winner = winner :=
  (root239Step3).trans root239Scan4
theorem root239Scan2 : scan edge (search edge 3) 4 [1531] 1531 root239Tail2 winner = winner :=
  (root239Step2).trans root239Scan3
theorem root239Scan1 : scan edge (search edge 3) 4 [1531] 1531 root239Tail1 winner = winner :=
  (root239Step1).trans root239Scan2
theorem root239Scan0 : scan edge (search edge 3) 4 [1531] 1531 root239Tail0 winner = winner :=
  (root239Step0).trans root239Scan1
set_option maxRecDepth 4096 in
theorem root239Branch : search edge 4 [1531] root239Candidates 1531 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1531] 1531 root239Tail0 winner = winner
  exact root239Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
