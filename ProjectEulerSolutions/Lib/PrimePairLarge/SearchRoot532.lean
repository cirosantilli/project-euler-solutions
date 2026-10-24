import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root532Candidates : List Nat := Data.row3853
set_option maxRecDepth 8192 in
theorem root532Candidates_checked : candidates edge 5 0 3853 tail533 winner.bound = root532Candidates := by
  apply candidates_eq_row edge 5 0 3853 tail533 root532Candidates winner.bound tail533_ordered
  · decide
  · decide
  · funext q
    rfl
def root532Tail35 : List Nat := []
def root532Tail34 : List Nat := 21499 :: root532Tail35
def root532Tail33 : List Nat := 21487 :: root532Tail34
def root532Tail32 : List Nat := 21061 :: root532Tail33
def root532Tail31 : List Nat := 19219 :: root532Tail32
def root532Tail30 : List Nat := 18979 :: root532Tail31
def root532Tail29 : List Nat := 18457 :: root532Tail30
def root532Tail28 : List Nat := 16273 :: root532Tail29
def root532Tail27 : List Nat := 15787 :: root532Tail28
def root532Tail26 : List Nat := 14107 :: root532Tail27
def root532Tail25 : List Nat := 13297 :: root532Tail26
def root532Tail24 : List Nat := 11119 :: root532Tail25
def root532Tail23 : List Nat := 10399 :: root532Tail24
def root532Tail22 : List Nat := 10273 :: root532Tail23
def root532Tail21 : List Nat := 10111 :: root532Tail22
def root532Tail20 : List Nat := 9433 :: root532Tail21
def root532Tail19 : List Nat := 9349 :: root532Tail20
def root532Tail18 : List Nat := 9103 :: root532Tail19
def root532Tail17 : List Nat := 8923 :: root532Tail18
def root532Tail16 : List Nat := 8887 :: root532Tail17
def root532Tail15 : List Nat := 8389 :: root532Tail16
def root532Tail14 : List Nat := 8293 :: root532Tail15
def root532Tail13 : List Nat := 7699 :: root532Tail14
def root532Tail12 : List Nat := 7591 :: root532Tail13
def root532Tail11 : List Nat := 7297 :: root532Tail12
def root532Tail10 : List Nat := 7243 :: root532Tail11
def root532Tail9 : List Nat := 7237 :: root532Tail10
def root532Tail8 : List Nat := 6991 :: root532Tail9
def root532Tail7 : List Nat := 6709 :: root532Tail8
def root532Tail6 : List Nat := 6607 :: root532Tail7
def root532Tail5 : List Nat := 6367 :: root532Tail6
def root532Tail4 : List Nat := 6361 :: root532Tail5
def root532Tail3 : List Nat := 5683 :: root532Tail4
def root532Tail2 : List Nat := 5521 :: root532Tail3
def root532Tail1 : List Nat := 4759 :: root532Tail2
def root532Tail0 : List Nat := 4099 :: root532Tail1
def root532Candidates0 : List Nat := [15787]
set_option maxRecDepth 8192 in
theorem root532Candidates0_checked : candidates edge 4 3853 4099 root532Tail1 winner.bound = root532Candidates0 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root532Step0 : scan edge (search edge 3) 4 [3853] 3853 root532Tail0 winner =
    scan edge (search edge 3) 4 [3853] 3853 root532Tail1 winner := by
  rw [root532Tail0, scan, if_neg (by decide), root532Candidates0_checked]
  change scan edge (search edge 3) 4 [3853] 3853 root532Tail1
    (if 3 ≤ root532Candidates0.length then search edge 3 [3853, 4099] root532Candidates0 7952 winner else winner) = _
  rw [if_neg (by decide)]
def root532Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root532Candidates1_checked : candidates edge 4 3853 4759 root532Tail2 winner.bound = root532Candidates1 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root532Step1 : scan edge (search edge 3) 4 [3853] 3853 root532Tail1 winner =
    scan edge (search edge 3) 4 [3853] 3853 root532Tail2 winner := by
  rw [root532Tail1, scan, if_neg (by decide), root532Candidates1_checked]
  change scan edge (search edge 3) 4 [3853] 3853 root532Tail2
    (if 3 ≤ root532Candidates1.length then search edge 3 [3853, 4759] root532Candidates1 8612 winner else winner) = _
  rw [if_neg (by decide)]
def root532Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root532Candidates2_checked : candidates edge 4 3853 5521 root532Tail3 winner.bound = root532Candidates2 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root532Step2 : scan edge (search edge 3) 4 [3853] 3853 root532Tail2 winner =
    scan edge (search edge 3) 4 [3853] 3853 root532Tail3 winner := by
  rw [root532Tail2, scan, if_neg (by decide), root532Candidates2_checked]
  change scan edge (search edge 3) 4 [3853] 3853 root532Tail3
    (if 3 ≤ root532Candidates2.length then search edge 3 [3853, 5521] root532Candidates2 9374 winner else winner) = _
  rw [if_neg (by decide)]
theorem root532Scan3 : scan edge (search edge 3) 4 [3853] 3853 root532Tail3 winner = winner := by rw [root532Tail3, scan, if_pos (by decide)]
theorem root532Scan2 : scan edge (search edge 3) 4 [3853] 3853 root532Tail2 winner = winner :=
  (root532Step2).trans root532Scan3
theorem root532Scan1 : scan edge (search edge 3) 4 [3853] 3853 root532Tail1 winner = winner :=
  (root532Step1).trans root532Scan2
theorem root532Scan0 : scan edge (search edge 3) 4 [3853] 3853 root532Tail0 winner = winner :=
  (root532Step0).trans root532Scan1
set_option maxRecDepth 4096 in
theorem root532Branch : search edge 4 [3853] root532Candidates 3853 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3853] 3853 root532Tail0 winner = winner
  exact root532Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
