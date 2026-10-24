import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root449Candidates : List Nat := Data.row3191
set_option maxRecDepth 8192 in
theorem root449Candidates_checked : candidates edge 5 0 3191 tail450 winner.bound = root449Candidates := by
  apply candidates_eq_row edge 5 0 3191 tail450 root449Candidates winner.bound tail450_ordered
  · decide
  · decide
  · funext q
    rfl
def root449Tail31 : List Nat := []
def root449Tail30 : List Nat := 20411 :: root449Tail31
def root449Tail29 : List Nat := 20297 :: root449Tail30
def root449Tail28 : List Nat := 18749 :: root449Tail29
def root449Tail27 : List Nat := 17891 :: root449Tail28
def root449Tail26 : List Nat := 17333 :: root449Tail27
def root449Tail25 : List Nat := 15413 :: root449Tail26
def root449Tail24 : List Nat := 15287 :: root449Tail25
def root449Tail23 : List Nat := 14423 :: root449Tail24
def root449Tail22 : List Nat := 13931 :: root449Tail23
def root449Tail21 : List Nat := 13499 :: root449Tail22
def root449Tail20 : List Nat := 13217 :: root449Tail21
def root449Tail19 : List Nat := 11927 :: root449Tail20
def root449Tail18 : List Nat := 11831 :: root449Tail19
def root449Tail17 : List Nat := 10499 :: root449Tail18
def root449Tail16 : List Nat := 9539 :: root449Tail17
def root449Tail15 : List Nat := 9161 :: root449Tail16
def root449Tail14 : List Nat := 8597 :: root449Tail15
def root449Tail13 : List Nat := 8111 :: root449Tail14
def root449Tail12 : List Nat := 8069 :: root449Tail13
def root449Tail11 : List Nat := 7937 :: root449Tail12
def root449Tail10 : List Nat := 7883 :: root449Tail11
def root449Tail9 : List Nat := 7703 :: root449Tail10
def root449Tail8 : List Nat := 7547 :: root449Tail9
def root449Tail7 : List Nat := 7451 :: root449Tail8
def root449Tail6 : List Nat := 7043 :: root449Tail7
def root449Tail5 : List Nat := 6779 :: root449Tail6
def root449Tail4 : List Nat := 5531 :: root449Tail5
def root449Tail3 : List Nat := 4583 :: root449Tail4
def root449Tail2 : List Nat := 4397 :: root449Tail3
def root449Tail1 : List Nat := 3821 :: root449Tail2
def root449Tail0 : List Nat := 3221 :: root449Tail1
def root449Candidates0 : List Nat := [13931]
set_option maxRecDepth 8192 in
theorem root449Candidates0_checked : candidates edge 4 3191 3221 root449Tail1 winner.bound = root449Candidates0 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root449Step0 : scan edge (search edge 3) 4 [3191] 3191 root449Tail0 winner =
    scan edge (search edge 3) 4 [3191] 3191 root449Tail1 winner := by
  rw [root449Tail0, scan, if_neg (by decide), root449Candidates0_checked]
  change scan edge (search edge 3) 4 [3191] 3191 root449Tail1
    (if 3 ≤ root449Candidates0.length then search edge 3 [3191, 3221] root449Candidates0 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root449Candidates1 : List Nat := [9539]
set_option maxRecDepth 8192 in
theorem root449Candidates1_checked : candidates edge 4 3191 3821 root449Tail2 winner.bound = root449Candidates1 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root449Step1 : scan edge (search edge 3) 4 [3191] 3191 root449Tail1 winner =
    scan edge (search edge 3) 4 [3191] 3191 root449Tail2 winner := by
  rw [root449Tail1, scan, if_neg (by decide), root449Candidates1_checked]
  change scan edge (search edge 3) 4 [3191] 3191 root449Tail2
    (if 3 ≤ root449Candidates1.length then search edge 3 [3191, 3821] root449Candidates1 7012 winner else winner) = _
  rw [if_neg (by decide)]
def root449Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root449Candidates2_checked : candidates edge 4 3191 4397 root449Tail3 winner.bound = root449Candidates2 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root449Step2 : scan edge (search edge 3) 4 [3191] 3191 root449Tail2 winner =
    scan edge (search edge 3) 4 [3191] 3191 root449Tail3 winner := by
  rw [root449Tail2, scan, if_neg (by decide), root449Candidates2_checked]
  change scan edge (search edge 3) 4 [3191] 3191 root449Tail3
    (if 3 ≤ root449Candidates2.length then search edge 3 [3191, 4397] root449Candidates2 7588 winner else winner) = _
  rw [if_neg (by decide)]
def root449Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root449Candidates3_checked : candidates edge 4 3191 4583 root449Tail4 winner.bound = root449Candidates3 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root449Step3 : scan edge (search edge 3) 4 [3191] 3191 root449Tail3 winner =
    scan edge (search edge 3) 4 [3191] 3191 root449Tail4 winner := by
  rw [root449Tail3, scan, if_neg (by decide), root449Candidates3_checked]
  change scan edge (search edge 3) 4 [3191] 3191 root449Tail4
    (if 3 ≤ root449Candidates3.length then search edge 3 [3191, 4583] root449Candidates3 7774 winner else winner) = _
  rw [if_neg (by decide)]
def root449Candidates4 : List Nat := [7883]
set_option maxRecDepth 8192 in
theorem root449Candidates4_checked : candidates edge 4 3191 5531 root449Tail5 winner.bound = root449Candidates4 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root449Step4 : scan edge (search edge 3) 4 [3191] 3191 root449Tail4 winner =
    scan edge (search edge 3) 4 [3191] 3191 root449Tail5 winner := by
  rw [root449Tail4, scan, if_neg (by decide), root449Candidates4_checked]
  change scan edge (search edge 3) 4 [3191] 3191 root449Tail5
    (if 3 ≤ root449Candidates4.length then search edge 3 [3191, 5531] root449Candidates4 8722 winner else winner) = _
  rw [if_neg (by decide)]
theorem root449Scan5 : scan edge (search edge 3) 4 [3191] 3191 root449Tail5 winner = winner := by rw [root449Tail5, scan, if_pos (by decide)]
theorem root449Scan4 : scan edge (search edge 3) 4 [3191] 3191 root449Tail4 winner = winner :=
  (root449Step4).trans root449Scan5
theorem root449Scan3 : scan edge (search edge 3) 4 [3191] 3191 root449Tail3 winner = winner :=
  (root449Step3).trans root449Scan4
theorem root449Scan2 : scan edge (search edge 3) 4 [3191] 3191 root449Tail2 winner = winner :=
  (root449Step2).trans root449Scan3
theorem root449Scan1 : scan edge (search edge 3) 4 [3191] 3191 root449Tail1 winner = winner :=
  (root449Step1).trans root449Scan2
theorem root449Scan0 : scan edge (search edge 3) 4 [3191] 3191 root449Tail0 winner = winner :=
  (root449Step0).trans root449Scan1
set_option maxRecDepth 4096 in
theorem root449Branch : search edge 4 [3191] root449Candidates 3191 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3191] 3191 root449Tail0 winner = winner
  exact root449Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
