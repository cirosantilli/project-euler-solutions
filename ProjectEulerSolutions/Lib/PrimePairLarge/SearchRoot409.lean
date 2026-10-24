import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root409Candidates : List Nat := Data.row2837
set_option maxRecDepth 8192 in
theorem root409Candidates_checked : candidates edge 5 0 2837 tail410 winner.bound = root409Candidates := by
  apply candidates_eq_row edge 5 0 2837 tail410 root409Candidates winner.bound tail410_ordered
  · decide
  · decide
  · funext q
    rfl
def root409Tail44 : List Nat := []
def root409Tail43 : List Nat := 22787 :: root409Tail44
def root409Tail42 : List Nat := 21929 :: root409Tail43
def root409Tail41 : List Nat := 21149 :: root409Tail42
def root409Tail40 : List Nat := 20771 :: root409Tail41
def root409Tail39 : List Nat := 20231 :: root409Tail40
def root409Tail38 : List Nat := 20117 :: root409Tail39
def root409Tail37 : List Nat := 17033 :: root409Tail38
def root409Tail36 : List Nat := 15791 :: root409Tail37
def root409Tail35 : List Nat := 15749 :: root409Tail36
def root409Tail34 : List Nat := 15647 :: root409Tail35
def root409Tail33 : List Nat := 15569 :: root409Tail34
def root409Tail32 : List Nat := 15461 :: root409Tail33
def root409Tail31 : List Nat := 14891 :: root409Tail32
def root409Tail30 : List Nat := 14747 :: root409Tail31
def root409Tail29 : List Nat := 13841 :: root409Tail30
def root409Tail28 : List Nat := 13721 :: root409Tail29
def root409Tail27 : List Nat := 12329 :: root409Tail28
def root409Tail26 : List Nat := 12203 :: root409Tail27
def root409Tail25 : List Nat := 11579 :: root409Tail26
def root409Tail24 : List Nat := 11423 :: root409Tail25
def root409Tail23 : List Nat := 11027 :: root409Tail24
def root409Tail22 : List Nat := 10859 :: root409Tail23
def root409Tail21 : List Nat := 10733 :: root409Tail22
def root409Tail20 : List Nat := 10463 :: root409Tail21
def root409Tail19 : List Nat := 10289 :: root409Tail20
def root409Tail18 : List Nat := 9431 :: root409Tail19
def root409Tail17 : List Nat := 9419 :: root409Tail18
def root409Tail16 : List Nat := 9311 :: root409Tail17
def root409Tail15 : List Nat := 9209 :: root409Tail16
def root409Tail14 : List Nat := 9161 :: root409Tail15
def root409Tail13 : List Nat := 7793 :: root409Tail14
def root409Tail12 : List Nat := 7757 :: root409Tail13
def root409Tail11 : List Nat := 7691 :: root409Tail12
def root409Tail10 : List Nat := 7001 :: root409Tail11
def root409Tail9 : List Nat := 6449 :: root409Tail10
def root409Tail8 : List Nat := 5297 :: root409Tail9
def root409Tail7 : List Nat := 5153 :: root409Tail8
def root409Tail6 : List Nat := 5003 :: root409Tail7
def root409Tail5 : List Nat := 4817 :: root409Tail6
def root409Tail4 : List Nat := 3767 :: root409Tail5
def root409Tail3 : List Nat := 3617 :: root409Tail4
def root409Tail2 : List Nat := 3407 :: root409Tail3
def root409Tail1 : List Nat := 3299 :: root409Tail2
def root409Tail0 : List Nat := 3089 :: root409Tail1
def root409Candidates0 : List Nat := [3299, 11579]
set_option maxRecDepth 8192 in
theorem root409Candidates0_checked : candidates edge 4 2837 3089 root409Tail1 winner.bound = root409Candidates0 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root409Step0 : scan edge (search edge 3) 4 [2837] 2837 root409Tail0 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail1 winner := by
  rw [root409Tail0, scan, if_neg (by decide), root409Candidates0_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail1
    (if 3 ≤ root409Candidates0.length then search edge 3 [2837, 3089] root409Candidates0 5926 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates1 : List Nat := [3767, 4817]
set_option maxRecDepth 8192 in
theorem root409Candidates1_checked : candidates edge 4 2837 3299 root409Tail2 winner.bound = root409Candidates1 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root409Step1 : scan edge (search edge 3) 4 [2837] 2837 root409Tail1 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail2 winner := by
  rw [root409Tail1, scan, if_neg (by decide), root409Candidates1_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail2
    (if 3 ≤ root409Candidates1.length then search edge 3 [2837, 3299] root409Candidates1 6136 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates2 : List Nat := [7691, 14747]
set_option maxRecDepth 8192 in
theorem root409Candidates2_checked : candidates edge 4 2837 3407 root409Tail3 winner.bound = root409Candidates2 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root409Step2 : scan edge (search edge 3) 4 [2837] 2837 root409Tail2 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail3 winner := by
  rw [root409Tail2, scan, if_neg (by decide), root409Candidates2_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail3
    (if 3 ≤ root409Candidates2.length then search edge 3 [2837, 3407] root409Candidates2 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates3 : List Nat := [5003, 11579, 15461]
set_option maxRecDepth 8192 in
theorem root409Candidates3_checked : candidates edge 4 2837 3617 root409Tail4 winner.bound = root409Candidates3 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root409Child3 : search edge 3 [2837, 3617] root409Candidates3 6454 winner = winner := by decide
theorem root409Step3 : scan edge (search edge 3) 4 [2837] 2837 root409Tail3 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail4 winner := by
  rw [root409Tail3, scan, if_neg (by decide), root409Candidates3_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail4
    (if 3 ≤ root409Candidates3.length then search edge 3 [2837, 3617] root409Candidates3 6454 winner else winner) = _
  rw [if_pos (by decide), root409Child3]
def root409Candidates4 : List Nat := [7001, 7793]
set_option maxRecDepth 8192 in
theorem root409Candidates4_checked : candidates edge 4 2837 3767 root409Tail5 winner.bound = root409Candidates4 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root409Step4 : scan edge (search edge 3) 4 [2837] 2837 root409Tail4 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail5 winner := by
  rw [root409Tail4, scan, if_neg (by decide), root409Candidates4_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail5
    (if 3 ≤ root409Candidates4.length then search edge 3 [2837, 3767] root409Candidates4 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root409Candidates5_checked : candidates edge 4 2837 4817 root409Tail6 winner.bound = root409Candidates5 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root409Step5 : scan edge (search edge 3) 4 [2837] 2837 root409Tail5 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail6 winner := by
  rw [root409Tail5, scan, if_neg (by decide), root409Candidates5_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail6
    (if 3 ≤ root409Candidates5.length then search edge 3 [2837, 4817] root409Candidates5 7654 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates6 : List Nat := [11423]
set_option maxRecDepth 8192 in
theorem root409Candidates6_checked : candidates edge 4 2837 5003 root409Tail7 winner.bound = root409Candidates6 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root409Step6 : scan edge (search edge 3) 4 [2837] 2837 root409Tail6 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail7 winner := by
  rw [root409Tail6, scan, if_neg (by decide), root409Candidates6_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail7
    (if 3 ≤ root409Candidates6.length then search edge 3 [2837, 5003] root409Candidates6 7840 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates7 : List Nat := [9311, 11579]
set_option maxRecDepth 8192 in
theorem root409Candidates7_checked : candidates edge 4 2837 5153 root409Tail8 winner.bound = root409Candidates7 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root409Step7 : scan edge (search edge 3) 4 [2837] 2837 root409Tail7 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail8 winner := by
  rw [root409Tail7, scan, if_neg (by decide), root409Candidates7_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail8
    (if 3 ≤ root409Candidates7.length then search edge 3 [2837, 5153] root409Candidates7 7990 winner else winner) = _
  rw [if_neg (by decide)]
def root409Candidates8 : List Nat := [15749]
set_option maxRecDepth 8192 in
theorem root409Candidates8_checked : candidates edge 4 2837 5297 root409Tail9 winner.bound = root409Candidates8 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root409Step8 : scan edge (search edge 3) 4 [2837] 2837 root409Tail8 winner =
    scan edge (search edge 3) 4 [2837] 2837 root409Tail9 winner := by
  rw [root409Tail8, scan, if_neg (by decide), root409Candidates8_checked]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail9
    (if 3 ≤ root409Candidates8.length then search edge 3 [2837, 5297] root409Candidates8 8134 winner else winner) = _
  rw [if_neg (by decide)]
theorem root409Scan9 : scan edge (search edge 3) 4 [2837] 2837 root409Tail9 winner = winner := by rw [root409Tail9, scan, if_pos (by decide)]
theorem root409Scan8 : scan edge (search edge 3) 4 [2837] 2837 root409Tail8 winner = winner :=
  (root409Step8).trans root409Scan9
theorem root409Scan7 : scan edge (search edge 3) 4 [2837] 2837 root409Tail7 winner = winner :=
  (root409Step7).trans root409Scan8
theorem root409Scan6 : scan edge (search edge 3) 4 [2837] 2837 root409Tail6 winner = winner :=
  (root409Step6).trans root409Scan7
theorem root409Scan5 : scan edge (search edge 3) 4 [2837] 2837 root409Tail5 winner = winner :=
  (root409Step5).trans root409Scan6
theorem root409Scan4 : scan edge (search edge 3) 4 [2837] 2837 root409Tail4 winner = winner :=
  (root409Step4).trans root409Scan5
theorem root409Scan3 : scan edge (search edge 3) 4 [2837] 2837 root409Tail3 winner = winner :=
  (root409Step3).trans root409Scan4
theorem root409Scan2 : scan edge (search edge 3) 4 [2837] 2837 root409Tail2 winner = winner :=
  (root409Step2).trans root409Scan3
theorem root409Scan1 : scan edge (search edge 3) 4 [2837] 2837 root409Tail1 winner = winner :=
  (root409Step1).trans root409Scan2
theorem root409Scan0 : scan edge (search edge 3) 4 [2837] 2837 root409Tail0 winner = winner :=
  (root409Step0).trans root409Scan1
set_option maxRecDepth 4096 in
theorem root409Branch : search edge 4 [2837] root409Candidates 2837 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2837] 2837 root409Tail0 winner = winner
  exact root409Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
