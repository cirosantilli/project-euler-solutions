import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root406Candidates : List Nat := Data.row2803
set_option maxRecDepth 8192 in
theorem root406Candidates_checked : candidates edge 5 0 2803 tail407 winner.bound = root406Candidates := by
  apply candidates_eq_row edge 5 0 2803 tail407 root406Candidates winner.bound tail407_ordered
  · decide
  · decide
  · funext q
    rfl
def root406Tail45 : List Nat := []
def root406Tail44 : List Nat := 20809 :: root406Tail45
def root406Tail43 : List Nat := 19387 :: root406Tail44
def root406Tail42 : List Nat := 19381 :: root406Tail43
def root406Tail41 : List Nat := 18217 :: root406Tail42
def root406Tail40 : List Nat := 17929 :: root406Tail41
def root406Tail39 : List Nat := 17839 :: root406Tail40
def root406Tail38 : List Nat := 17449 :: root406Tail39
def root406Tail37 : List Nat := 16903 :: root406Tail38
def root406Tail36 : List Nat := 16657 :: root406Tail37
def root406Tail35 : List Nat := 16573 :: root406Tail36
def root406Tail34 : List Nat := 15859 :: root406Tail35
def root406Tail33 : List Nat := 15727 :: root406Tail34
def root406Tail32 : List Nat := 15031 :: root406Tail33
def root406Tail31 : List Nat := 14341 :: root406Tail32
def root406Tail30 : List Nat := 14143 :: root406Tail31
def root406Tail29 : List Nat := 13381 :: root406Tail30
def root406Tail28 : List Nat := 13291 :: root406Tail29
def root406Tail27 : List Nat := 12763 :: root406Tail28
def root406Tail26 : List Nat := 11443 :: root406Tail27
def root406Tail25 : List Nat := 10567 :: root406Tail26
def root406Tail24 : List Nat := 9883 :: root406Tail25
def root406Tail23 : List Nat := 9127 :: root406Tail24
def root406Tail22 : List Nat := 9091 :: root406Tail23
def root406Tail21 : List Nat := 9049 :: root406Tail22
def root406Tail20 : List Nat := 8971 :: root406Tail21
def root406Tail19 : List Nat := 8377 :: root406Tail20
def root406Tail18 : List Nat := 7993 :: root406Tail19
def root406Tail17 : List Nat := 7951 :: root406Tail18
def root406Tail16 : List Nat := 7717 :: root406Tail17
def root406Tail15 : List Nat := 7549 :: root406Tail16
def root406Tail14 : List Nat := 6733 :: root406Tail15
def root406Tail13 : List Nat := 6361 :: root406Tail14
def root406Tail12 : List Nat := 6199 :: root406Tail13
def root406Tail11 : List Nat := 5851 :: root406Tail12
def root406Tail10 : List Nat := 5701 :: root406Tail11
def root406Tail9 : List Nat := 5107 :: root406Tail10
def root406Tail8 : List Nat := 4969 :: root406Tail9
def root406Tail7 : List Nat := 4813 :: root406Tail8
def root406Tail6 : List Nat := 4549 :: root406Tail7
def root406Tail5 : List Nat := 4507 :: root406Tail6
def root406Tail4 : List Nat := 4423 :: root406Tail5
def root406Tail3 : List Nat := 4051 :: root406Tail4
def root406Tail2 : List Nat := 3769 :: root406Tail3
def root406Tail1 : List Nat := 3631 :: root406Tail2
def root406Tail0 : List Nat := 3079 :: root406Tail1
def root406Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root406Candidates0_checked : candidates edge 4 2803 3079 root406Tail1 winner.bound = root406Candidates0 := by
  unfold candidates
  have row : edge 3079 = (SortedNatTree.contains FastRows.row3079) := by
    change Data.row3079.contains = (SortedNatTree.contains FastRows.row3079)
    exact FastRows.row3079_checked.symm
  rw [row]
  decide
theorem root406Step0 : scan edge (search edge 3) 4 [2803] 2803 root406Tail0 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail1 winner := by
  rw [root406Tail0, scan, if_neg (by decide), root406Candidates0_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail1
    (if 3 ≤ root406Candidates0.length then search edge 3 [2803, 3079] root406Candidates0 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates1 : List Nat := [8377]
set_option maxRecDepth 8192 in
theorem root406Candidates1_checked : candidates edge 4 2803 3631 root406Tail2 winner.bound = root406Candidates1 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root406Step1 : scan edge (search edge 3) 4 [2803] 2803 root406Tail1 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail2 winner := by
  rw [root406Tail1, scan, if_neg (by decide), root406Candidates1_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail2
    (if 3 ≤ root406Candidates1.length then search edge 3 [2803, 3631] root406Candidates1 6434 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates2 : List Nat := [4051, 8971, 9127]
set_option maxRecDepth 8192 in
theorem root406Candidates2_checked : candidates edge 4 2803 3769 root406Tail3 winner.bound = root406Candidates2 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root406Child2 : search edge 3 [2803, 3769] root406Candidates2 6572 winner = winner := by decide
theorem root406Step2 : scan edge (search edge 3) 4 [2803] 2803 root406Tail2 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail3 winner := by
  rw [root406Tail2, scan, if_neg (by decide), root406Candidates2_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail3
    (if 3 ≤ root406Candidates2.length then search edge 3 [2803, 3769] root406Candidates2 6572 winner else winner) = _
  rw [if_pos (by decide), root406Child2]
def root406Candidates3 : List Nat := [5701, 7549, 7951]
set_option maxRecDepth 8192 in
theorem root406Candidates3_checked : candidates edge 4 2803 4051 root406Tail4 winner.bound = root406Candidates3 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root406Child3 : search edge 3 [2803, 4051] root406Candidates3 6854 winner = winner := by decide
theorem root406Step3 : scan edge (search edge 3) 4 [2803] 2803 root406Tail3 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail4 winner := by
  rw [root406Tail3, scan, if_neg (by decide), root406Candidates3_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail4
    (if 3 ≤ root406Candidates3.length then search edge 3 [2803, 4051] root406Candidates3 6854 winner else winner) = _
  rw [if_pos (by decide), root406Child3]
def root406Candidates4 : List Nat := [10567]
set_option maxRecDepth 8192 in
theorem root406Candidates4_checked : candidates edge 4 2803 4423 root406Tail5 winner.bound = root406Candidates4 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root406Step4 : scan edge (search edge 3) 4 [2803] 2803 root406Tail4 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail5 winner := by
  rw [root406Tail4, scan, if_neg (by decide), root406Candidates4_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail5
    (if 3 ≤ root406Candidates4.length then search edge 3 [2803, 4423] root406Candidates4 7226 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates5 : List Nat := [7993, 8377]
set_option maxRecDepth 8192 in
theorem root406Candidates5_checked : candidates edge 4 2803 4507 root406Tail6 winner.bound = root406Candidates5 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root406Step5 : scan edge (search edge 3) 4 [2803] 2803 root406Tail5 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail6 winner := by
  rw [root406Tail5, scan, if_neg (by decide), root406Candidates5_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail6
    (if 3 ≤ root406Candidates5.length then search edge 3 [2803, 4507] root406Candidates5 7310 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates6 : List Nat := [5851, 6361, 7717, 9127, 15859]
set_option maxRecDepth 8192 in
theorem root406Candidates6_checked : candidates edge 4 2803 4549 root406Tail7 winner.bound = root406Candidates6 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root406Child6 : search edge 3 [2803, 4549] root406Candidates6 7352 winner = winner := by decide
theorem root406Step6 : scan edge (search edge 3) 4 [2803] 2803 root406Tail6 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail7 winner := by
  rw [root406Tail6, scan, if_neg (by decide), root406Candidates6_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail7
    (if 3 ≤ root406Candidates6.length then search edge 3 [2803, 4549] root406Candidates6 7352 winner else winner) = _
  rw [if_pos (by decide), root406Child6]
def root406Candidates7 : List Nat := [5701, 16573]
set_option maxRecDepth 8192 in
theorem root406Candidates7_checked : candidates edge 4 2803 4813 root406Tail8 winner.bound = root406Candidates7 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root406Step7 : scan edge (search edge 3) 4 [2803] 2803 root406Tail7 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail8 winner := by
  rw [root406Tail7, scan, if_neg (by decide), root406Candidates7_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail8
    (if 3 ≤ root406Candidates7.length then search edge 3 [2803, 4813] root406Candidates7 7616 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates8 : List Nat := [9049]
set_option maxRecDepth 8192 in
theorem root406Candidates8_checked : candidates edge 4 2803 4969 root406Tail9 winner.bound = root406Candidates8 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root406Step8 : scan edge (search edge 3) 4 [2803] 2803 root406Tail8 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail9 winner := by
  rw [root406Tail8, scan, if_neg (by decide), root406Candidates8_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail9
    (if 3 ≤ root406Candidates8.length then search edge 3 [2803, 4969] root406Candidates8 7772 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates9 : List Nat := [7717]
set_option maxRecDepth 8192 in
theorem root406Candidates9_checked : candidates edge 4 2803 5107 root406Tail10 winner.bound = root406Candidates9 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root406Step9 : scan edge (search edge 3) 4 [2803] 2803 root406Tail9 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail10 winner := by
  rw [root406Tail9, scan, if_neg (by decide), root406Candidates9_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail10
    (if 3 ≤ root406Candidates9.length then search edge 3 [2803, 5107] root406Candidates9 7910 winner else winner) = _
  rw [if_neg (by decide)]
def root406Candidates10 : List Nat := [6733, 16903]
set_option maxRecDepth 8192 in
theorem root406Candidates10_checked : candidates edge 4 2803 5701 root406Tail11 winner.bound = root406Candidates10 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root406Step10 : scan edge (search edge 3) 4 [2803] 2803 root406Tail10 winner =
    scan edge (search edge 3) 4 [2803] 2803 root406Tail11 winner := by
  rw [root406Tail10, scan, if_neg (by decide), root406Candidates10_checked]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail11
    (if 3 ≤ root406Candidates10.length then search edge 3 [2803, 5701] root406Candidates10 8504 winner else winner) = _
  rw [if_neg (by decide)]
theorem root406Scan11 : scan edge (search edge 3) 4 [2803] 2803 root406Tail11 winner = winner := by rw [root406Tail11, scan, if_pos (by decide)]
theorem root406Scan10 : scan edge (search edge 3) 4 [2803] 2803 root406Tail10 winner = winner :=
  (root406Step10).trans root406Scan11
theorem root406Scan9 : scan edge (search edge 3) 4 [2803] 2803 root406Tail9 winner = winner :=
  (root406Step9).trans root406Scan10
theorem root406Scan8 : scan edge (search edge 3) 4 [2803] 2803 root406Tail8 winner = winner :=
  (root406Step8).trans root406Scan9
theorem root406Scan7 : scan edge (search edge 3) 4 [2803] 2803 root406Tail7 winner = winner :=
  (root406Step7).trans root406Scan8
theorem root406Scan6 : scan edge (search edge 3) 4 [2803] 2803 root406Tail6 winner = winner :=
  (root406Step6).trans root406Scan7
theorem root406Scan5 : scan edge (search edge 3) 4 [2803] 2803 root406Tail5 winner = winner :=
  (root406Step5).trans root406Scan6
theorem root406Scan4 : scan edge (search edge 3) 4 [2803] 2803 root406Tail4 winner = winner :=
  (root406Step4).trans root406Scan5
theorem root406Scan3 : scan edge (search edge 3) 4 [2803] 2803 root406Tail3 winner = winner :=
  (root406Step3).trans root406Scan4
theorem root406Scan2 : scan edge (search edge 3) 4 [2803] 2803 root406Tail2 winner = winner :=
  (root406Step2).trans root406Scan3
theorem root406Scan1 : scan edge (search edge 3) 4 [2803] 2803 root406Tail1 winner = winner :=
  (root406Step1).trans root406Scan2
theorem root406Scan0 : scan edge (search edge 3) 4 [2803] 2803 root406Tail0 winner = winner :=
  (root406Step0).trans root406Scan1
set_option maxRecDepth 4096 in
theorem root406Branch : search edge 4 [2803] root406Candidates 2803 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2803] 2803 root406Tail0 winner = winner
  exact root406Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
