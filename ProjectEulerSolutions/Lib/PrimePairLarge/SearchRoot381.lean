import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root381Candidates : List Nat := Data.row2657
set_option maxRecDepth 8192 in
theorem root381Candidates_checked : candidates edge 5 0 2657 tail382 winner.bound = root381Candidates := by
  apply candidates_eq_row edge 5 0 2657 tail382 root381Candidates winner.bound tail382_ordered
  · decide
  · decide
  · funext q
    rfl
def root381Tail32 : List Nat := []
def root381Tail31 : List Nat := 23333 :: root381Tail32
def root381Tail30 : List Nat := 22901 :: root381Tail31
def root381Tail29 : List Nat := 21143 :: root381Tail30
def root381Tail28 : List Nat := 20921 :: root381Tail29
def root381Tail27 : List Nat := 20639 :: root381Tail28
def root381Tail26 : List Nat := 19121 :: root381Tail27
def root381Tail25 : List Nat := 19031 :: root381Tail26
def root381Tail24 : List Nat := 18917 :: root381Tail25
def root381Tail23 : List Nat := 18539 :: root381Tail24
def root381Tail22 : List Nat := 18443 :: root381Tail23
def root381Tail21 : List Nat := 18191 :: root381Tail22
def root381Tail20 : List Nat := 17477 :: root381Tail21
def root381Tail19 : List Nat := 16481 :: root381Tail20
def root381Tail18 : List Nat := 15773 :: root381Tail19
def root381Tail17 : List Nat := 14633 :: root381Tail18
def root381Tail16 : List Nat := 13619 :: root381Tail17
def root381Tail15 : List Nat := 12653 :: root381Tail16
def root381Tail14 : List Nat := 12269 :: root381Tail15
def root381Tail13 : List Nat := 11867 :: root381Tail14
def root381Tail12 : List Nat := 9587 :: root381Tail13
def root381Tail11 : List Nat := 9221 :: root381Tail12
def root381Tail10 : List Nat := 8963 :: root381Tail11
def root381Tail9 : List Nat := 8243 :: root381Tail10
def root381Tail8 : List Nat := 7247 :: root381Tail9
def root381Tail7 : List Nat := 6833 :: root381Tail8
def root381Tail6 : List Nat := 6449 :: root381Tail7
def root381Tail5 : List Nat := 5279 :: root381Tail6
def root381Tail4 : List Nat := 5099 :: root381Tail5
def root381Tail3 : List Nat := 4517 :: root381Tail4
def root381Tail2 : List Nat := 4391 :: root381Tail3
def root381Tail1 : List Nat := 3923 :: root381Tail2
def root381Tail0 : List Nat := 3719 :: root381Tail1
def root381Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root381Candidates0_checked : candidates edge 4 2657 3719 root381Tail1 winner.bound = root381Candidates0 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root381Step0 : scan edge (search edge 3) 4 [2657] 2657 root381Tail0 winner =
    scan edge (search edge 3) 4 [2657] 2657 root381Tail1 winner := by
  rw [root381Tail0, scan, if_neg (by decide), root381Candidates0_checked]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail1
    (if 3 ≤ root381Candidates0.length then search edge 3 [2657, 3719] root381Candidates0 6376 winner else winner) = _
  rw [if_neg (by decide)]
def root381Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root381Candidates1_checked : candidates edge 4 2657 3923 root381Tail2 winner.bound = root381Candidates1 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root381Step1 : scan edge (search edge 3) 4 [2657] 2657 root381Tail1 winner =
    scan edge (search edge 3) 4 [2657] 2657 root381Tail2 winner := by
  rw [root381Tail1, scan, if_neg (by decide), root381Candidates1_checked]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail2
    (if 3 ≤ root381Candidates1.length then search edge 3 [2657, 3923] root381Candidates1 6580 winner else winner) = _
  rw [if_neg (by decide)]
def root381Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root381Candidates2_checked : candidates edge 4 2657 4391 root381Tail3 winner.bound = root381Candidates2 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root381Step2 : scan edge (search edge 3) 4 [2657] 2657 root381Tail2 winner =
    scan edge (search edge 3) 4 [2657] 2657 root381Tail3 winner := by
  rw [root381Tail2, scan, if_neg (by decide), root381Candidates2_checked]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail3
    (if 3 ≤ root381Candidates2.length then search edge 3 [2657, 4391] root381Candidates2 7048 winner else winner) = _
  rw [if_neg (by decide)]
def root381Candidates3 : List Nat := [5279]
set_option maxRecDepth 8192 in
theorem root381Candidates3_checked : candidates edge 4 2657 4517 root381Tail4 winner.bound = root381Candidates3 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root381Step3 : scan edge (search edge 3) 4 [2657] 2657 root381Tail3 winner =
    scan edge (search edge 3) 4 [2657] 2657 root381Tail4 winner := by
  rw [root381Tail3, scan, if_neg (by decide), root381Candidates3_checked]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail4
    (if 3 ≤ root381Candidates3.length then search edge 3 [2657, 4517] root381Candidates3 7174 winner else winner) = _
  rw [if_neg (by decide)]
def root381Candidates4 : List Nat := [9587, 13619, 17477]
set_option maxRecDepth 8192 in
theorem root381Candidates4_checked : candidates edge 4 2657 5099 root381Tail5 winner.bound = root381Candidates4 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root381Child4 : search edge 3 [2657, 5099] root381Candidates4 7756 winner = winner := by decide
theorem root381Step4 : scan edge (search edge 3) 4 [2657] 2657 root381Tail4 winner =
    scan edge (search edge 3) 4 [2657] 2657 root381Tail5 winner := by
  rw [root381Tail4, scan, if_neg (by decide), root381Candidates4_checked]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail5
    (if 3 ≤ root381Candidates4.length then search edge 3 [2657, 5099] root381Candidates4 7756 winner else winner) = _
  rw [if_pos (by decide), root381Child4]
def root381Candidates5 : List Nat := [6449]
set_option maxRecDepth 8192 in
theorem root381Candidates5_checked : candidates edge 4 2657 5279 root381Tail6 winner.bound = root381Candidates5 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root381Step5 : scan edge (search edge 3) 4 [2657] 2657 root381Tail5 winner =
    scan edge (search edge 3) 4 [2657] 2657 root381Tail6 winner := by
  rw [root381Tail5, scan, if_neg (by decide), root381Candidates5_checked]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail6
    (if 3 ≤ root381Candidates5.length then search edge 3 [2657, 5279] root381Candidates5 7936 winner else winner) = _
  rw [if_neg (by decide)]
theorem root381Scan6 : scan edge (search edge 3) 4 [2657] 2657 root381Tail6 winner = winner := by rw [root381Tail6, scan, if_pos (by decide)]
theorem root381Scan5 : scan edge (search edge 3) 4 [2657] 2657 root381Tail5 winner = winner :=
  (root381Step5).trans root381Scan6
theorem root381Scan4 : scan edge (search edge 3) 4 [2657] 2657 root381Tail4 winner = winner :=
  (root381Step4).trans root381Scan5
theorem root381Scan3 : scan edge (search edge 3) 4 [2657] 2657 root381Tail3 winner = winner :=
  (root381Step3).trans root381Scan4
theorem root381Scan2 : scan edge (search edge 3) 4 [2657] 2657 root381Tail2 winner = winner :=
  (root381Step2).trans root381Scan3
theorem root381Scan1 : scan edge (search edge 3) 4 [2657] 2657 root381Tail1 winner = winner :=
  (root381Step1).trans root381Scan2
theorem root381Scan0 : scan edge (search edge 3) 4 [2657] 2657 root381Tail0 winner = winner :=
  (root381Step0).trans root381Scan1
set_option maxRecDepth 4096 in
theorem root381Branch : search edge 4 [2657] root381Candidates 2657 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2657] 2657 root381Tail0 winner = winner
  exact root381Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
