import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root504Candidates : List Nat := Data.row3623
set_option maxRecDepth 8192 in
theorem root504Candidates_checked : candidates edge 5 0 3623 tail505 winner.bound = root504Candidates := by
  apply candidates_eq_row edge 5 0 3623 tail505 root504Candidates winner.bound tail505_ordered
  · decide
  · decide
  · funext q
    rfl
def root504Tail31 : List Nat := []
def root504Tail30 : List Nat := 21977 :: root504Tail31
def root504Tail29 : List Nat := 21821 :: root504Tail30
def root504Tail28 : List Nat := 21467 :: root504Tail29
def root504Tail27 : List Nat := 21107 :: root504Tail28
def root504Tail26 : List Nat := 20921 :: root504Tail27
def root504Tail25 : List Nat := 19973 :: root504Tail26
def root504Tail24 : List Nat := 19469 :: root504Tail25
def root504Tail23 : List Nat := 18947 :: root504Tail24
def root504Tail22 : List Nat := 18413 :: root504Tail23
def root504Tail21 : List Nat := 18401 :: root504Tail22
def root504Tail20 : List Nat := 18251 :: root504Tail21
def root504Tail19 : List Nat := 17393 :: root504Tail20
def root504Tail18 : List Nat := 16901 :: root504Tail19
def root504Tail17 : List Nat := 16193 :: root504Tail18
def root504Tail16 : List Nat := 15263 :: root504Tail17
def root504Tail15 : List Nat := 14369 :: root504Tail16
def root504Tail14 : List Nat := 12227 :: root504Tail15
def root504Tail13 : List Nat := 11867 :: root504Tail14
def root504Tail12 : List Nat := 9719 :: root504Tail13
def root504Tail11 : List Nat := 9173 :: root504Tail12
def root504Tail10 : List Nat := 9059 :: root504Tail11
def root504Tail9 : List Nat := 8963 :: root504Tail10
def root504Tail8 : List Nat := 8123 :: root504Tail9
def root504Tail7 : List Nat := 6521 :: root504Tail8
def root504Tail6 : List Nat := 6449 :: root504Tail7
def root504Tail5 : List Nat := 5981 :: root504Tail6
def root504Tail4 : List Nat := 4967 :: root504Tail5
def root504Tail3 : List Nat := 4271 :: root504Tail4
def root504Tail2 : List Nat := 4127 :: root504Tail3
def root504Tail1 : List Nat := 4091 :: root504Tail2
def root504Tail0 : List Nat := 4049 :: root504Tail1
def root504Candidates0 : List Nat := [17393]
set_option maxRecDepth 8192 in
theorem root504Candidates0_checked : candidates edge 4 3623 4049 root504Tail1 winner.bound = root504Candidates0 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root504Step0 : scan edge (search edge 3) 4 [3623] 3623 root504Tail0 winner =
    scan edge (search edge 3) 4 [3623] 3623 root504Tail1 winner := by
  rw [root504Tail0, scan, if_neg (by decide), root504Candidates0_checked]
  change scan edge (search edge 3) 4 [3623] 3623 root504Tail1
    (if 3 ≤ root504Candidates0.length then search edge 3 [3623, 4049] root504Candidates0 7672 winner else winner) = _
  rw [if_neg (by decide)]
def root504Candidates1 : List Nat := [14369]
set_option maxRecDepth 8192 in
theorem root504Candidates1_checked : candidates edge 4 3623 4091 root504Tail2 winner.bound = root504Candidates1 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root504Step1 : scan edge (search edge 3) 4 [3623] 3623 root504Tail1 winner =
    scan edge (search edge 3) 4 [3623] 3623 root504Tail2 winner := by
  rw [root504Tail1, scan, if_neg (by decide), root504Candidates1_checked]
  change scan edge (search edge 3) 4 [3623] 3623 root504Tail2
    (if 3 ≤ root504Candidates1.length then search edge 3 [3623, 4091] root504Candidates1 7714 winner else winner) = _
  rw [if_neg (by decide)]
def root504Candidates2 : List Nat := [4967]
set_option maxRecDepth 8192 in
theorem root504Candidates2_checked : candidates edge 4 3623 4127 root504Tail3 winner.bound = root504Candidates2 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root504Step2 : scan edge (search edge 3) 4 [3623] 3623 root504Tail2 winner =
    scan edge (search edge 3) 4 [3623] 3623 root504Tail3 winner := by
  rw [root504Tail2, scan, if_neg (by decide), root504Candidates2_checked]
  change scan edge (search edge 3) 4 [3623] 3623 root504Tail3
    (if 3 ≤ root504Candidates2.length then search edge 3 [3623, 4127] root504Candidates2 7750 winner else winner) = _
  rw [if_neg (by decide)]
def root504Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root504Candidates3_checked : candidates edge 4 3623 4271 root504Tail4 winner.bound = root504Candidates3 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root504Step3 : scan edge (search edge 3) 4 [3623] 3623 root504Tail3 winner =
    scan edge (search edge 3) 4 [3623] 3623 root504Tail4 winner := by
  rw [root504Tail3, scan, if_neg (by decide), root504Candidates3_checked]
  change scan edge (search edge 3) 4 [3623] 3623 root504Tail4
    (if 3 ≤ root504Candidates3.length then search edge 3 [3623, 4271] root504Candidates3 7894 winner else winner) = _
  rw [if_neg (by decide)]
def root504Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root504Candidates4_checked : candidates edge 4 3623 4967 root504Tail5 winner.bound = root504Candidates4 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root504Step4 : scan edge (search edge 3) 4 [3623] 3623 root504Tail4 winner =
    scan edge (search edge 3) 4 [3623] 3623 root504Tail5 winner := by
  rw [root504Tail4, scan, if_neg (by decide), root504Candidates4_checked]
  change scan edge (search edge 3) 4 [3623] 3623 root504Tail5
    (if 3 ≤ root504Candidates4.length then search edge 3 [3623, 4967] root504Candidates4 8590 winner else winner) = _
  rw [if_neg (by decide)]
theorem root504Scan5 : scan edge (search edge 3) 4 [3623] 3623 root504Tail5 winner = winner := by rw [root504Tail5, scan, if_pos (by decide)]
theorem root504Scan4 : scan edge (search edge 3) 4 [3623] 3623 root504Tail4 winner = winner :=
  (root504Step4).trans root504Scan5
theorem root504Scan3 : scan edge (search edge 3) 4 [3623] 3623 root504Tail3 winner = winner :=
  (root504Step3).trans root504Scan4
theorem root504Scan2 : scan edge (search edge 3) 4 [3623] 3623 root504Tail2 winner = winner :=
  (root504Step2).trans root504Scan3
theorem root504Scan1 : scan edge (search edge 3) 4 [3623] 3623 root504Tail1 winner = winner :=
  (root504Step1).trans root504Scan2
theorem root504Scan0 : scan edge (search edge 3) 4 [3623] 3623 root504Tail0 winner = winner :=
  (root504Step0).trans root504Scan1
set_option maxRecDepth 4096 in
theorem root504Branch : search edge 4 [3623] root504Candidates 3623 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3623] 3623 root504Tail0 winner = winner
  exact root504Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
