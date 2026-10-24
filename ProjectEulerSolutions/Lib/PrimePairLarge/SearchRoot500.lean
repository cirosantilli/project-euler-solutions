import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root500Candidates : List Nat := Data.row3593
set_option maxRecDepth 8192 in
theorem root500Candidates_checked : candidates edge 5 0 3593 tail501 winner.bound = root500Candidates := by
  apply candidates_eq_row edge 5 0 3593 tail501 root500Candidates winner.bound tail501_ordered
  · decide
  · decide
  · funext q
    rfl
def root500Tail41 : List Nat := []
def root500Tail40 : List Nat := 20879 :: root500Tail41
def root500Tail39 : List Nat := 20249 :: root500Tail40
def root500Tail38 : List Nat := 20147 :: root500Tail39
def root500Tail37 : List Nat := 19391 :: root500Tail38
def root500Tail36 : List Nat := 18461 :: root500Tail37
def root500Tail35 : List Nat := 18131 :: root500Tail36
def root500Tail34 : List Nat := 17291 :: root500Tail35
def root500Tail33 : List Nat := 17207 :: root500Tail34
def root500Tail32 : List Nat := 17099 :: root500Tail33
def root500Tail31 : List Nat := 16553 :: root500Tail32
def root500Tail30 : List Nat := 16007 :: root500Tail31
def root500Tail29 : List Nat := 15773 :: root500Tail30
def root500Tail28 : List Nat := 15629 :: root500Tail29
def root500Tail27 : List Nat := 15083 :: root500Tail28
def root500Tail26 : List Nat := 14621 :: root500Tail27
def root500Tail25 : List Nat := 14519 :: root500Tail26
def root500Tail24 : List Nat := 13679 :: root500Tail25
def root500Tail23 : List Nat := 13469 :: root500Tail24
def root500Tail22 : List Nat := 13049 :: root500Tail23
def root500Tail21 : List Nat := 13001 :: root500Tail22
def root500Tail20 : List Nat := 11027 :: root500Tail21
def root500Tail19 : List Nat := 10607 :: root500Tail20
def root500Tail18 : List Nat := 10529 :: root500Tail19
def root500Tail17 : List Nat := 10067 :: root500Tail18
def root500Tail16 : List Nat := 9473 :: root500Tail17
def root500Tail15 : List Nat := 8831 :: root500Tail16
def root500Tail14 : List Nat := 8807 :: root500Tail15
def root500Tail13 : List Nat := 8681 :: root500Tail14
def root500Tail12 : List Nat := 8297 :: root500Tail13
def root500Tail11 : List Nat := 7907 :: root500Tail12
def root500Tail10 : List Nat := 7877 :: root500Tail11
def root500Tail9 : List Nat := 7529 :: root500Tail10
def root500Tail8 : List Nat := 6311 :: root500Tail9
def root500Tail7 : List Nat := 5903 :: root500Tail8
def root500Tail6 : List Nat := 5693 :: root500Tail7
def root500Tail5 : List Nat := 5279 :: root500Tail6
def root500Tail4 : List Nat := 5087 :: root500Tail5
def root500Tail3 : List Nat := 5021 :: root500Tail4
def root500Tail2 : List Nat := 4517 :: root500Tail3
def root500Tail1 : List Nat := 4001 :: root500Tail2
def root500Tail0 : List Nat := 3719 :: root500Tail1
def root500Candidates0 : List Nat := [17207]
set_option maxRecDepth 8192 in
theorem root500Candidates0_checked : candidates edge 4 3593 3719 root500Tail1 winner.bound = root500Candidates0 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root500Step0 : scan edge (search edge 3) 4 [3593] 3593 root500Tail0 winner =
    scan edge (search edge 3) 4 [3593] 3593 root500Tail1 winner := by
  rw [root500Tail0, scan, if_neg (by decide), root500Candidates0_checked]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail1
    (if 3 ≤ root500Candidates0.length then search edge 3 [3593, 3719] root500Candidates0 7312 winner else winner) = _
  rw [if_neg (by decide)]
def root500Candidates1 : List Nat := [7877, 15629, 15773]
set_option maxRecDepth 8192 in
theorem root500Candidates1_checked : candidates edge 4 3593 4001 root500Tail2 winner.bound = root500Candidates1 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root500Child1 : search edge 3 [3593, 4001] root500Candidates1 7594 winner = winner := by decide
theorem root500Step1 : scan edge (search edge 3) 4 [3593] 3593 root500Tail1 winner =
    scan edge (search edge 3) 4 [3593] 3593 root500Tail2 winner := by
  rw [root500Tail1, scan, if_neg (by decide), root500Candidates1_checked]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail2
    (if 3 ≤ root500Candidates1.length then search edge 3 [3593, 4001] root500Candidates1 7594 winner else winner) = _
  rw [if_pos (by decide), root500Child1]
def root500Candidates2 : List Nat := [5087, 5279, 11027, 13469]
set_option maxRecDepth 8192 in
theorem root500Candidates2_checked : candidates edge 4 3593 4517 root500Tail3 winner.bound = root500Candidates2 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root500Child2 : search edge 3 [3593, 4517] root500Candidates2 8110 winner = winner := by decide
theorem root500Step2 : scan edge (search edge 3) 4 [3593] 3593 root500Tail2 winner =
    scan edge (search edge 3) 4 [3593] 3593 root500Tail3 winner := by
  rw [root500Tail2, scan, if_neg (by decide), root500Candidates2_checked]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail3
    (if 3 ≤ root500Candidates2.length then search edge 3 [3593, 4517] root500Candidates2 8110 winner else winner) = _
  rw [if_pos (by decide), root500Child2]
def root500Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root500Candidates3_checked : candidates edge 4 3593 5021 root500Tail4 winner.bound = root500Candidates3 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root500Step3 : scan edge (search edge 3) 4 [3593] 3593 root500Tail3 winner =
    scan edge (search edge 3) 4 [3593] 3593 root500Tail4 winner := by
  rw [root500Tail3, scan, if_neg (by decide), root500Candidates3_checked]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail4
    (if 3 ≤ root500Candidates3.length then search edge 3 [3593, 5021] root500Candidates3 8614 winner else winner) = _
  rw [if_neg (by decide)]
def root500Candidates4 : List Nat := [13001, 17207]
set_option maxRecDepth 8192 in
theorem root500Candidates4_checked : candidates edge 4 3593 5087 root500Tail5 winner.bound = root500Candidates4 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root500Step4 : scan edge (search edge 3) 4 [3593] 3593 root500Tail4 winner =
    scan edge (search edge 3) 4 [3593] 3593 root500Tail5 winner := by
  rw [root500Tail4, scan, if_neg (by decide), root500Candidates4_checked]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail5
    (if 3 ≤ root500Candidates4.length then search edge 3 [3593, 5087] root500Candidates4 8680 winner else winner) = _
  rw [if_neg (by decide)]
def root500Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root500Candidates5_checked : candidates edge 4 3593 5279 root500Tail6 winner.bound = root500Candidates5 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root500Step5 : scan edge (search edge 3) 4 [3593] 3593 root500Tail5 winner =
    scan edge (search edge 3) 4 [3593] 3593 root500Tail6 winner := by
  rw [root500Tail5, scan, if_neg (by decide), root500Candidates5_checked]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail6
    (if 3 ≤ root500Candidates5.length then search edge 3 [3593, 5279] root500Candidates5 8872 winner else winner) = _
  rw [if_neg (by decide)]
theorem root500Scan6 : scan edge (search edge 3) 4 [3593] 3593 root500Tail6 winner = winner := by rw [root500Tail6, scan, if_pos (by decide)]
theorem root500Scan5 : scan edge (search edge 3) 4 [3593] 3593 root500Tail5 winner = winner :=
  (root500Step5).trans root500Scan6
theorem root500Scan4 : scan edge (search edge 3) 4 [3593] 3593 root500Tail4 winner = winner :=
  (root500Step4).trans root500Scan5
theorem root500Scan3 : scan edge (search edge 3) 4 [3593] 3593 root500Tail3 winner = winner :=
  (root500Step3).trans root500Scan4
theorem root500Scan2 : scan edge (search edge 3) 4 [3593] 3593 root500Tail2 winner = winner :=
  (root500Step2).trans root500Scan3
theorem root500Scan1 : scan edge (search edge 3) 4 [3593] 3593 root500Tail1 winner = winner :=
  (root500Step1).trans root500Scan2
theorem root500Scan0 : scan edge (search edge 3) 4 [3593] 3593 root500Tail0 winner = winner :=
  (root500Step0).trans root500Scan1
set_option maxRecDepth 4096 in
theorem root500Branch : search edge 4 [3593] root500Candidates 3593 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3593] 3593 root500Tail0 winner = winner
  exact root500Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
