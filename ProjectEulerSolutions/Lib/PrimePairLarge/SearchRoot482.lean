import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root482Candidates : List Nat := Data.row3463
set_option maxRecDepth 8192 in
theorem root482Candidates_checked : candidates edge 5 0 3463 tail483 winner.bound = root482Candidates := by
  apply candidates_eq_row edge 5 0 3463 tail483 root482Candidates winner.bound tail483_ordered
  · decide
  · decide
  · funext q
    rfl
def root482Tail44 : List Nat := []
def root482Tail43 : List Nat := 21601 :: root482Tail44
def root482Tail42 : List Nat := 21169 :: root482Tail43
def root482Tail41 : List Nat := 21061 :: root482Tail42
def root482Tail40 : List Nat := 20743 :: root482Tail41
def root482Tail39 : List Nat := 20389 :: root482Tail40
def root482Tail38 : List Nat := 19069 :: root482Tail39
def root482Tail37 : List Nat := 18979 :: root482Tail38
def root482Tail36 : List Nat := 18919 :: root482Tail37
def root482Tail35 : List Nat := 17977 :: root482Tail36
def root482Tail34 : List Nat := 17683 :: root482Tail35
def root482Tail33 : List Nat := 17599 :: root482Tail34
def root482Tail32 : List Nat := 17467 :: root482Tail33
def root482Tail31 : List Nat := 17053 :: root482Tail32
def root482Tail30 : List Nat := 17029 :: root482Tail31
def root482Tail29 : List Nat := 16561 :: root482Tail30
def root482Tail28 : List Nat := 15913 :: root482Tail29
def root482Tail27 : List Nat := 15583 :: root482Tail28
def root482Tail26 : List Nat := 14713 :: root482Tail27
def root482Tail25 : List Nat := 14503 :: root482Tail26
def root482Tail24 : List Nat := 13567 :: root482Tail25
def root482Tail23 : List Nat := 11443 :: root482Tail24
def root482Tail22 : List Nat := 10687 :: root482Tail23
def root482Tail21 : List Nat := 10267 :: root482Tail22
def root482Tail20 : List Nat := 10099 :: root482Tail21
def root482Tail19 : List Nat := 9883 :: root482Tail20
def root482Tail18 : List Nat := 9697 :: root482Tail19
def root482Tail17 : List Nat := 9049 :: root482Tail18
def root482Tail16 : List Nat := 9043 :: root482Tail17
def root482Tail15 : List Nat := 8629 :: root482Tail16
def root482Tail14 : List Nat := 8059 :: root482Tail15
def root482Tail13 : List Nat := 8053 :: root482Tail14
def root482Tail12 : List Nat := 8011 :: root482Tail13
def root482Tail11 : List Nat := 7411 :: root482Tail12
def root482Tail10 : List Nat := 6883 :: root482Tail11
def root482Tail9 : List Nat := 6337 :: root482Tail10
def root482Tail8 : List Nat := 6211 :: root482Tail9
def root482Tail7 : List Nat := 6079 :: root482Tail8
def root482Tail6 : List Nat := 6037 :: root482Tail7
def root482Tail5 : List Nat := 5869 :: root482Tail6
def root482Tail4 : List Nat := 5479 :: root482Tail5
def root482Tail3 : List Nat := 4969 :: root482Tail4
def root482Tail2 : List Nat := 4801 :: root482Tail3
def root482Tail1 : List Nat := 4447 :: root482Tail2
def root482Tail0 : List Nat := 4051 :: root482Tail1
def root482Candidates0 : List Nat := [6211, 15913]
set_option maxRecDepth 8192 in
theorem root482Candidates0_checked : candidates edge 4 3463 4051 root482Tail1 winner.bound = root482Candidates0 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root482Step0 : scan edge (search edge 3) 4 [3463] 3463 root482Tail0 winner =
    scan edge (search edge 3) 4 [3463] 3463 root482Tail1 winner := by
  rw [root482Tail0, scan, if_neg (by decide), root482Candidates0_checked]
  change scan edge (search edge 3) 4 [3463] 3463 root482Tail1
    (if 3 ≤ root482Candidates0.length then search edge 3 [3463, 4051] root482Candidates0 7514 winner else winner) = _
  rw [if_neg (by decide)]
def root482Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root482Candidates1_checked : candidates edge 4 3463 4447 root482Tail2 winner.bound = root482Candidates1 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root482Step1 : scan edge (search edge 3) 4 [3463] 3463 root482Tail1 winner =
    scan edge (search edge 3) 4 [3463] 3463 root482Tail2 winner := by
  rw [root482Tail1, scan, if_neg (by decide), root482Candidates1_checked]
  change scan edge (search edge 3) 4 [3463] 3463 root482Tail2
    (if 3 ≤ root482Candidates1.length then search edge 3 [3463, 4447] root482Candidates1 7910 winner else winner) = _
  rw [if_neg (by decide)]
def root482Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root482Candidates2_checked : candidates edge 4 3463 4801 root482Tail3 winner.bound = root482Candidates2 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root482Step2 : scan edge (search edge 3) 4 [3463] 3463 root482Tail2 winner =
    scan edge (search edge 3) 4 [3463] 3463 root482Tail3 winner := by
  rw [root482Tail2, scan, if_neg (by decide), root482Candidates2_checked]
  change scan edge (search edge 3) 4 [3463] 3463 root482Tail3
    (if 3 ≤ root482Candidates2.length then search edge 3 [3463, 4801] root482Candidates2 8264 winner else winner) = _
  rw [if_neg (by decide)]
def root482Candidates3 : List Nat := [9049]
set_option maxRecDepth 8192 in
theorem root482Candidates3_checked : candidates edge 4 3463 4969 root482Tail4 winner.bound = root482Candidates3 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root482Step3 : scan edge (search edge 3) 4 [3463] 3463 root482Tail3 winner =
    scan edge (search edge 3) 4 [3463] 3463 root482Tail4 winner := by
  rw [root482Tail3, scan, if_neg (by decide), root482Candidates3_checked]
  change scan edge (search edge 3) 4 [3463] 3463 root482Tail4
    (if 3 ≤ root482Candidates3.length then search edge 3 [3463, 4969] root482Candidates3 8432 winner else winner) = _
  rw [if_neg (by decide)]
def root482Candidates4 : List Nat := [5869]
set_option maxRecDepth 8192 in
theorem root482Candidates4_checked : candidates edge 4 3463 5479 root482Tail5 winner.bound = root482Candidates4 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root482Step4 : scan edge (search edge 3) 4 [3463] 3463 root482Tail4 winner =
    scan edge (search edge 3) 4 [3463] 3463 root482Tail5 winner := by
  rw [root482Tail4, scan, if_neg (by decide), root482Candidates4_checked]
  change scan edge (search edge 3) 4 [3463] 3463 root482Tail5
    (if 3 ≤ root482Candidates4.length then search edge 3 [3463, 5479] root482Candidates4 8942 winner else winner) = _
  rw [if_neg (by decide)]
theorem root482Scan5 : scan edge (search edge 3) 4 [3463] 3463 root482Tail5 winner = winner := by rw [root482Tail5, scan, if_pos (by decide)]
theorem root482Scan4 : scan edge (search edge 3) 4 [3463] 3463 root482Tail4 winner = winner :=
  (root482Step4).trans root482Scan5
theorem root482Scan3 : scan edge (search edge 3) 4 [3463] 3463 root482Tail3 winner = winner :=
  (root482Step3).trans root482Scan4
theorem root482Scan2 : scan edge (search edge 3) 4 [3463] 3463 root482Tail2 winner = winner :=
  (root482Step2).trans root482Scan3
theorem root482Scan1 : scan edge (search edge 3) 4 [3463] 3463 root482Tail1 winner = winner :=
  (root482Step1).trans root482Scan2
theorem root482Scan0 : scan edge (search edge 3) 4 [3463] 3463 root482Tail0 winner = winner :=
  (root482Step0).trans root482Scan1
set_option maxRecDepth 4096 in
theorem root482Branch : search edge 4 [3463] root482Candidates 3463 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3463] 3463 root482Tail0 winner = winner
  exact root482Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
