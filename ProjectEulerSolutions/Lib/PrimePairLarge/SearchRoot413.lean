import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root413Candidates : List Nat := Data.row2861
set_option maxRecDepth 8192 in
theorem root413Candidates_checked : candidates edge 5 0 2861 tail414 winner.bound = root413Candidates := by
  apply candidates_eq_row edge 5 0 2861 tail414 root413Candidates winner.bound tail414_ordered
  · decide
  · decide
  · funext q
    rfl
def root413Tail35 : List Nat := []
def root413Tail34 : List Nat := 23081 :: root413Tail35
def root413Tail33 : List Nat := 22751 :: root413Tail34
def root413Tail32 : List Nat := 22679 :: root413Tail33
def root413Tail31 : List Nat := 22619 :: root413Tail32
def root413Tail30 : List Nat := 20903 :: root413Tail31
def root413Tail29 : List Nat := 20543 :: root413Tail30
def root413Tail28 : List Nat := 20357 :: root413Tail29
def root413Tail27 : List Nat := 20147 :: root413Tail28
def root413Tail26 : List Nat := 18719 :: root413Tail27
def root413Tail25 : List Nat := 18047 :: root413Tail26
def root413Tail24 : List Nat := 17729 :: root413Tail25
def root413Tail23 : List Nat := 17519 :: root413Tail24
def root413Tail22 : List Nat := 16451 :: root413Tail23
def root413Tail21 : List Nat := 12689 :: root413Tail22
def root413Tail20 : List Nat := 12161 :: root413Tail21
def root413Tail19 : List Nat := 12119 :: root413Tail20
def root413Tail18 : List Nat := 11831 :: root413Tail19
def root413Tail17 : List Nat := 11783 :: root413Tail18
def root413Tail16 : List Nat := 11321 :: root413Tail17
def root413Tail15 : List Nat := 11213 :: root413Tail16
def root413Tail14 : List Nat := 11159 :: root413Tail15
def root413Tail13 : List Nat := 9743 :: root413Tail14
def root413Tail12 : List Nat := 9281 :: root413Tail13
def root413Tail11 : List Nat := 9137 :: root413Tail12
def root413Tail10 : List Nat := 8171 :: root413Tail11
def root413Tail9 : List Nat := 7649 :: root413Tail10
def root413Tail8 : List Nat := 7499 :: root413Tail9
def root413Tail7 : List Nat := 7013 :: root413Tail8
def root413Tail6 : List Nat := 6719 :: root413Tail7
def root413Tail5 : List Nat := 5861 :: root413Tail6
def root413Tail4 : List Nat := 5693 :: root413Tail5
def root413Tail3 : List Nat := 4919 :: root413Tail4
def root413Tail2 : List Nat := 3911 :: root413Tail3
def root413Tail1 : List Nat := 3671 :: root413Tail2
def root413Tail0 : List Nat := 3257 :: root413Tail1
def root413Candidates0 : List Nat := [17519]
set_option maxRecDepth 8192 in
theorem root413Candidates0_checked : candidates edge 4 2861 3257 root413Tail1 winner.bound = root413Candidates0 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root413Step0 : scan edge (search edge 3) 4 [2861] 2861 root413Tail0 winner =
    scan edge (search edge 3) 4 [2861] 2861 root413Tail1 winner := by
  rw [root413Tail0, scan, if_neg (by decide), root413Candidates0_checked]
  change scan edge (search edge 3) 4 [2861] 2861 root413Tail1
    (if 3 ≤ root413Candidates0.length then search edge 3 [2861, 3257] root413Candidates0 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root413Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root413Candidates1_checked : candidates edge 4 2861 3671 root413Tail2 winner.bound = root413Candidates1 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root413Step1 : scan edge (search edge 3) 4 [2861] 2861 root413Tail1 winner =
    scan edge (search edge 3) 4 [2861] 2861 root413Tail2 winner := by
  rw [root413Tail1, scan, if_neg (by decide), root413Candidates1_checked]
  change scan edge (search edge 3) 4 [2861] 2861 root413Tail2
    (if 3 ≤ root413Candidates1.length then search edge 3 [2861, 3671] root413Candidates1 6532 winner else winner) = _
  rw [if_neg (by decide)]
def root413Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root413Candidates2_checked : candidates edge 4 2861 3911 root413Tail3 winner.bound = root413Candidates2 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root413Step2 : scan edge (search edge 3) 4 [2861] 2861 root413Tail2 winner =
    scan edge (search edge 3) 4 [2861] 2861 root413Tail3 winner := by
  rw [root413Tail2, scan, if_neg (by decide), root413Candidates2_checked]
  change scan edge (search edge 3) 4 [2861] 2861 root413Tail3
    (if 3 ≤ root413Candidates2.length then search edge 3 [2861, 3911] root413Candidates2 6772 winner else winner) = _
  rw [if_neg (by decide)]
def root413Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root413Candidates3_checked : candidates edge 4 2861 4919 root413Tail4 winner.bound = root413Candidates3 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root413Step3 : scan edge (search edge 3) 4 [2861] 2861 root413Tail3 winner =
    scan edge (search edge 3) 4 [2861] 2861 root413Tail4 winner := by
  rw [root413Tail3, scan, if_neg (by decide), root413Candidates3_checked]
  change scan edge (search edge 3) 4 [2861] 2861 root413Tail4
    (if 3 ≤ root413Candidates3.length then search edge 3 [2861, 4919] root413Candidates3 7780 winner else winner) = _
  rw [if_neg (by decide)]
def root413Candidates4 : List Nat := [8171, 12119]
set_option maxRecDepth 8192 in
theorem root413Candidates4_checked : candidates edge 4 2861 5693 root413Tail5 winner.bound = root413Candidates4 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root413Step4 : scan edge (search edge 3) 4 [2861] 2861 root413Tail4 winner =
    scan edge (search edge 3) 4 [2861] 2861 root413Tail5 winner := by
  rw [root413Tail4, scan, if_neg (by decide), root413Candidates4_checked]
  change scan edge (search edge 3) 4 [2861] 2861 root413Tail5
    (if 3 ≤ root413Candidates4.length then search edge 3 [2861, 5693] root413Candidates4 8554 winner else winner) = _
  rw [if_neg (by decide)]
theorem root413Scan5 : scan edge (search edge 3) 4 [2861] 2861 root413Tail5 winner = winner := by rw [root413Tail5, scan, if_pos (by decide)]
theorem root413Scan4 : scan edge (search edge 3) 4 [2861] 2861 root413Tail4 winner = winner :=
  (root413Step4).trans root413Scan5
theorem root413Scan3 : scan edge (search edge 3) 4 [2861] 2861 root413Tail3 winner = winner :=
  (root413Step3).trans root413Scan4
theorem root413Scan2 : scan edge (search edge 3) 4 [2861] 2861 root413Tail2 winner = winner :=
  (root413Step2).trans root413Scan3
theorem root413Scan1 : scan edge (search edge 3) 4 [2861] 2861 root413Tail1 winner = winner :=
  (root413Step1).trans root413Scan2
theorem root413Scan0 : scan edge (search edge 3) 4 [2861] 2861 root413Tail0 winner = winner :=
  (root413Step0).trans root413Scan1
set_option maxRecDepth 4096 in
theorem root413Branch : search edge 4 [2861] root413Candidates 2861 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2861] 2861 root413Tail0 winner = winner
  exact root413Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
