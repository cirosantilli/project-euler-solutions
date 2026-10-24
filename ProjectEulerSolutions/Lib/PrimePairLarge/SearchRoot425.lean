import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root425Candidates : List Nat := Data.row2969
set_option maxRecDepth 8192 in
theorem root425Candidates_checked : candidates edge 5 0 2969 tail426 winner.bound = root425Candidates := by
  apply candidates_eq_row edge 5 0 2969 tail426 root425Candidates winner.bound tail426_ordered
  · decide
  · decide
  · funext q
    rfl
def root425Tail32 : List Nat := []
def root425Tail31 : List Nat := 22037 :: root425Tail32
def root425Tail30 : List Nat := 21377 :: root425Tail31
def root425Tail29 : List Nat := 19937 :: root425Tail30
def root425Tail28 : List Nat := 17987 :: root425Tail29
def root425Tail27 : List Nat := 16661 :: root425Tail28
def root425Tail26 : List Nat := 16493 :: root425Tail27
def root425Tail25 : List Nat := 15383 :: root425Tail26
def root425Tail24 : List Nat := 15173 :: root425Tail25
def root425Tail23 : List Nat := 15107 :: root425Tail24
def root425Tail22 : List Nat := 14561 :: root425Tail23
def root425Tail21 : List Nat := 13451 :: root425Tail22
def root425Tail20 : List Nat := 12953 :: root425Tail21
def root425Tail19 : List Nat := 12809 :: root425Tail20
def root425Tail18 : List Nat := 12197 :: root425Tail19
def root425Tail17 : List Nat := 10067 :: root425Tail18
def root425Tail16 : List Nat := 9281 :: root425Tail17
def root425Tail15 : List Nat := 9041 :: root425Tail16
def root425Tail14 : List Nat := 9011 :: root425Tail15
def root425Tail13 : List Nat := 8861 :: root425Tail14
def root425Tail12 : List Nat := 8807 :: root425Tail13
def root425Tail11 : List Nat := 8597 :: root425Tail12
def root425Tail10 : List Nat := 7841 :: root425Tail11
def root425Tail9 : List Nat := 7823 :: root425Tail10
def root425Tail8 : List Nat := 7673 :: root425Tail9
def root425Tail7 : List Nat := 6803 :: root425Tail8
def root425Tail6 : List Nat := 6701 :: root425Tail7
def root425Tail5 : List Nat := 5153 :: root425Tail6
def root425Tail4 : List Nat := 5009 :: root425Tail5
def root425Tail3 : List Nat := 4703 :: root425Tail4
def root425Tail2 : List Nat := 4547 :: root425Tail3
def root425Tail1 : List Nat := 3557 :: root425Tail2
def root425Tail0 : List Nat := 3539 :: root425Tail1
def root425Candidates0 : List Nat := [15383]
set_option maxRecDepth 8192 in
theorem root425Candidates0_checked : candidates edge 4 2969 3539 root425Tail1 winner.bound = root425Candidates0 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root425Step0 : scan edge (search edge 3) 4 [2969] 2969 root425Tail0 winner =
    scan edge (search edge 3) 4 [2969] 2969 root425Tail1 winner := by
  rw [root425Tail0, scan, if_neg (by decide), root425Candidates0_checked]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail1
    (if 3 ≤ root425Candidates0.length then search edge 3 [2969, 3539] root425Candidates0 6508 winner else winner) = _
  rw [if_neg (by decide)]
def root425Candidates1 : List Nat := [17987]
set_option maxRecDepth 8192 in
theorem root425Candidates1_checked : candidates edge 4 2969 3557 root425Tail2 winner.bound = root425Candidates1 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root425Step1 : scan edge (search edge 3) 4 [2969] 2969 root425Tail1 winner =
    scan edge (search edge 3) 4 [2969] 2969 root425Tail2 winner := by
  rw [root425Tail1, scan, if_neg (by decide), root425Candidates1_checked]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail2
    (if 3 ≤ root425Candidates1.length then search edge 3 [2969, 3557] root425Candidates1 6526 winner else winner) = _
  rw [if_neg (by decide)]
def root425Candidates2 : List Nat := [8597]
set_option maxRecDepth 8192 in
theorem root425Candidates2_checked : candidates edge 4 2969 4547 root425Tail3 winner.bound = root425Candidates2 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root425Step2 : scan edge (search edge 3) 4 [2969] 2969 root425Tail2 winner =
    scan edge (search edge 3) 4 [2969] 2969 root425Tail3 winner := by
  rw [root425Tail2, scan, if_neg (by decide), root425Candidates2_checked]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail3
    (if 3 ≤ root425Candidates2.length then search edge 3 [2969, 4547] root425Candidates2 7516 winner else winner) = _
  rw [if_neg (by decide)]
def root425Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root425Candidates3_checked : candidates edge 4 2969 4703 root425Tail4 winner.bound = root425Candidates3 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root425Step3 : scan edge (search edge 3) 4 [2969] 2969 root425Tail3 winner =
    scan edge (search edge 3) 4 [2969] 2969 root425Tail4 winner := by
  rw [root425Tail3, scan, if_neg (by decide), root425Candidates3_checked]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail4
    (if 3 ≤ root425Candidates3.length then search edge 3 [2969, 4703] root425Candidates3 7672 winner else winner) = _
  rw [if_neg (by decide)]
def root425Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root425Candidates4_checked : candidates edge 4 2969 5009 root425Tail5 winner.bound = root425Candidates4 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root425Step4 : scan edge (search edge 3) 4 [2969] 2969 root425Tail4 winner =
    scan edge (search edge 3) 4 [2969] 2969 root425Tail5 winner := by
  rw [root425Tail4, scan, if_neg (by decide), root425Candidates4_checked]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail5
    (if 3 ≤ root425Candidates4.length then search edge 3 [2969, 5009] root425Candidates4 7978 winner else winner) = _
  rw [if_neg (by decide)]
def root425Candidates5 : List Nat := [9041, 15107, 16493]
set_option maxRecDepth 8192 in
theorem root425Candidates5_checked : candidates edge 4 2969 5153 root425Tail6 winner.bound = root425Candidates5 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root425Child5 : search edge 3 [2969, 5153] root425Candidates5 8122 winner = winner := by decide
theorem root425Step5 : scan edge (search edge 3) 4 [2969] 2969 root425Tail5 winner =
    scan edge (search edge 3) 4 [2969] 2969 root425Tail6 winner := by
  rw [root425Tail5, scan, if_neg (by decide), root425Candidates5_checked]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail6
    (if 3 ≤ root425Candidates5.length then search edge 3 [2969, 5153] root425Candidates5 8122 winner else winner) = _
  rw [if_pos (by decide), root425Child5]
theorem root425Scan6 : scan edge (search edge 3) 4 [2969] 2969 root425Tail6 winner = winner := by rw [root425Tail6, scan, if_pos (by decide)]
theorem root425Scan5 : scan edge (search edge 3) 4 [2969] 2969 root425Tail5 winner = winner :=
  (root425Step5).trans root425Scan6
theorem root425Scan4 : scan edge (search edge 3) 4 [2969] 2969 root425Tail4 winner = winner :=
  (root425Step4).trans root425Scan5
theorem root425Scan3 : scan edge (search edge 3) 4 [2969] 2969 root425Tail3 winner = winner :=
  (root425Step3).trans root425Scan4
theorem root425Scan2 : scan edge (search edge 3) 4 [2969] 2969 root425Tail2 winner = winner :=
  (root425Step2).trans root425Scan3
theorem root425Scan1 : scan edge (search edge 3) 4 [2969] 2969 root425Tail1 winner = winner :=
  (root425Step1).trans root425Scan2
theorem root425Scan0 : scan edge (search edge 3) 4 [2969] 2969 root425Tail0 winner = winner :=
  (root425Step0).trans root425Scan1
set_option maxRecDepth 4096 in
theorem root425Branch : search edge 4 [2969] root425Candidates 2969 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2969] 2969 root425Tail0 winner = winner
  exact root425Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
