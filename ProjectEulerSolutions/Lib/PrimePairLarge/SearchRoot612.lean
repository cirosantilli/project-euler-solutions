import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root612Candidates : List Nat := Data.row4523
set_option maxRecDepth 8192 in
theorem root612Candidates_checked : candidates edge 5 0 4523 tail613 winner.bound = root612Candidates := by
  apply candidates_eq_row edge 5 0 4523 tail613 root612Candidates winner.bound tail613_ordered
  · decide
  · decide
  · funext q
    rfl
def root612Tail40 : List Nat := []
def root612Tail39 : List Nat := 21179 :: root612Tail40
def root612Tail38 : List Nat := 19739 :: root612Tail39
def root612Tail37 : List Nat := 19433 :: root612Tail38
def root612Tail36 : List Nat := 19037 :: root612Tail37
def root612Tail35 : List Nat := 18701 :: root612Tail36
def root612Tail34 : List Nat := 18593 :: root612Tail35
def root612Tail33 : List Nat := 18353 :: root612Tail34
def root612Tail32 : List Nat := 17891 :: root612Tail33
def root612Tail31 : List Nat := 17573 :: root612Tail32
def root612Tail30 : List Nat := 17093 :: root612Tail31
def root612Tail29 : List Nat := 15077 :: root612Tail30
def root612Tail28 : List Nat := 14867 :: root612Tail29
def root612Tail27 : List Nat := 14843 :: root612Tail28
def root612Tail26 : List Nat := 14717 :: root612Tail27
def root612Tail25 : List Nat := 14249 :: root612Tail26
def root612Tail24 : List Nat := 14081 :: root612Tail25
def root612Tail23 : List Nat := 13709 :: root612Tail24
def root612Tail22 : List Nat := 12713 :: root612Tail23
def root612Tail21 : List Nat := 12611 :: root612Tail22
def root612Tail20 : List Nat := 12569 :: root612Tail21
def root612Tail19 : List Nat := 10457 :: root612Tail20
def root612Tail18 : List Nat := 10091 :: root612Tail19
def root612Tail17 : List Nat := 9743 :: root612Tail18
def root612Tail16 : List Nat := 9461 :: root612Tail17
def root612Tail15 : List Nat := 9221 :: root612Tail16
def root612Tail14 : List Nat := 8093 :: root612Tail15
def root612Tail13 : List Nat := 7937 :: root612Tail14
def root612Tail12 : List Nat := 7583 :: root612Tail13
def root612Tail11 : List Nat := 7457 :: root612Tail12
def root612Tail10 : List Nat := 6911 :: root612Tail11
def root612Tail9 : List Nat := 6761 :: root612Tail10
def root612Tail8 : List Nat := 6551 :: root612Tail9
def root612Tail7 : List Nat := 6329 :: root612Tail8
def root612Tail6 : List Nat := 6173 :: root612Tail7
def root612Tail5 : List Nat := 5639 :: root612Tail6
def root612Tail4 : List Nat := 5153 :: root612Tail5
def root612Tail3 : List Nat := 5081 :: root612Tail4
def root612Tail2 : List Nat := 5039 :: root612Tail3
def root612Tail1 : List Nat := 4967 :: root612Tail2
def root612Tail0 : List Nat := 4817 :: root612Tail1
def root612Candidates0 : List Nat := [6173]
set_option maxRecDepth 8192 in
theorem root612Candidates0_checked : candidates edge 4 4523 4817 root612Tail1 winner.bound = root612Candidates0 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root612Step0 : scan edge (search edge 3) 4 [4523] 4523 root612Tail0 winner =
    scan edge (search edge 3) 4 [4523] 4523 root612Tail1 winner := by
  rw [root612Tail0, scan, if_neg (by decide), root612Candidates0_checked]
  change scan edge (search edge 3) 4 [4523] 4523 root612Tail1
    (if 3 ≤ root612Candidates0.length then search edge 3 [4523, 4817] root612Candidates0 9340 winner else winner) = _
  rw [if_neg (by decide)]
def root612Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root612Candidates1_checked : candidates edge 4 4523 4967 root612Tail2 winner.bound = root612Candidates1 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root612Step1 : scan edge (search edge 3) 4 [4523] 4523 root612Tail1 winner =
    scan edge (search edge 3) 4 [4523] 4523 root612Tail2 winner := by
  rw [root612Tail1, scan, if_neg (by decide), root612Candidates1_checked]
  change scan edge (search edge 3) 4 [4523] 4523 root612Tail2
    (if 3 ≤ root612Candidates1.length then search edge 3 [4523, 4967] root612Candidates1 9490 winner else winner) = _
  rw [if_neg (by decide)]
def root612Candidates2 : List Nat := [5153, 6329, 8093]
set_option maxRecDepth 8192 in
theorem root612Candidates2_checked : candidates edge 4 4523 5039 root612Tail3 winner.bound = root612Candidates2 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root612Child2 : search edge 3 [4523, 5039] root612Candidates2 9562 winner = winner := by decide
theorem root612Step2 : scan edge (search edge 3) 4 [4523] 4523 root612Tail2 winner =
    scan edge (search edge 3) 4 [4523] 4523 root612Tail3 winner := by
  rw [root612Tail2, scan, if_neg (by decide), root612Candidates2_checked]
  change scan edge (search edge 3) 4 [4523] 4523 root612Tail3
    (if 3 ≤ root612Candidates2.length then search edge 3 [4523, 5039] root612Candidates2 9562 winner else winner) = _
  rw [if_pos (by decide), root612Child2]
def root612Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root612Candidates3_checked : candidates edge 4 4523 5081 root612Tail4 winner.bound = root612Candidates3 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root612Step3 : scan edge (search edge 3) 4 [4523] 4523 root612Tail3 winner =
    scan edge (search edge 3) 4 [4523] 4523 root612Tail4 winner := by
  rw [root612Tail3, scan, if_neg (by decide), root612Candidates3_checked]
  change scan edge (search edge 3) 4 [4523] 4523 root612Tail4
    (if 3 ≤ root612Candidates3.length then search edge 3 [4523, 5081] root612Candidates3 9604 winner else winner) = _
  rw [if_neg (by decide)]
def root612Candidates4 : List Nat := [14843]
set_option maxRecDepth 8192 in
theorem root612Candidates4_checked : candidates edge 4 4523 5153 root612Tail5 winner.bound = root612Candidates4 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root612Step4 : scan edge (search edge 3) 4 [4523] 4523 root612Tail4 winner =
    scan edge (search edge 3) 4 [4523] 4523 root612Tail5 winner := by
  rw [root612Tail4, scan, if_neg (by decide), root612Candidates4_checked]
  change scan edge (search edge 3) 4 [4523] 4523 root612Tail5
    (if 3 ≤ root612Candidates4.length then search edge 3 [4523, 5153] root612Candidates4 9676 winner else winner) = _
  rw [if_neg (by decide)]
theorem root612Scan5 : scan edge (search edge 3) 4 [4523] 4523 root612Tail5 winner = winner := by rw [root612Tail5, scan, if_pos (by decide)]
theorem root612Scan4 : scan edge (search edge 3) 4 [4523] 4523 root612Tail4 winner = winner :=
  (root612Step4).trans root612Scan5
theorem root612Scan3 : scan edge (search edge 3) 4 [4523] 4523 root612Tail3 winner = winner :=
  (root612Step3).trans root612Scan4
theorem root612Scan2 : scan edge (search edge 3) 4 [4523] 4523 root612Tail2 winner = winner :=
  (root612Step2).trans root612Scan3
theorem root612Scan1 : scan edge (search edge 3) 4 [4523] 4523 root612Tail1 winner = winner :=
  (root612Step1).trans root612Scan2
theorem root612Scan0 : scan edge (search edge 3) 4 [4523] 4523 root612Tail0 winner = winner :=
  (root612Step0).trans root612Scan1
set_option maxRecDepth 4096 in
theorem root612Branch : search edge 4 [4523] root612Candidates 4523 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4523] 4523 root612Tail0 winner = winner
  exact root612Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
