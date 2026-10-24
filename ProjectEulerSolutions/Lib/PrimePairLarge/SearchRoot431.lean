import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root431Candidates : List Nat := Data.row3023
set_option maxRecDepth 8192 in
theorem root431Candidates_checked : candidates edge 5 0 3023 tail432 winner.bound = root431Candidates := by
  apply candidates_eq_row edge 5 0 3023 tail432 root431Candidates winner.bound tail432_ordered
  · decide
  · decide
  · funext q
    rfl
def root431Tail34 : List Nat := []
def root431Tail33 : List Nat := 21893 :: root431Tail34
def root431Tail32 : List Nat := 20177 :: root431Tail33
def root431Tail31 : List Nat := 19961 :: root431Tail32
def root431Tail30 : List Nat := 19001 :: root431Tail31
def root431Tail29 : List Nat := 18131 :: root431Tail30
def root431Tail28 : List Nat := 17909 :: root431Tail29
def root431Tail27 : List Nat := 17627 :: root431Tail28
def root431Tail26 : List Nat := 16673 :: root431Tail27
def root431Tail25 : List Nat := 15683 :: root431Tail26
def root431Tail24 : List Nat := 14543 :: root431Tail25
def root431Tail23 : List Nat := 14447 :: root431Tail24
def root431Tail22 : List Nat := 14057 :: root431Tail23
def root431Tail21 : List Nat := 11393 :: root431Tail22
def root431Tail20 : List Nat := 11369 :: root431Tail21
def root431Tail19 : List Nat := 10223 :: root431Tail20
def root431Tail18 : List Nat := 10139 :: root431Tail19
def root431Tail17 : List Nat := 9497 :: root431Tail18
def root431Tail16 : List Nat := 9437 :: root431Tail17
def root431Tail15 : List Nat := 9059 :: root431Tail16
def root431Tail14 : List Nat := 8783 :: root431Tail15
def root431Tail13 : List Nat := 8669 :: root431Tail14
def root431Tail12 : List Nat := 8069 :: root431Tail13
def root431Tail11 : List Nat := 7589 :: root431Tail12
def root431Tail10 : List Nat := 6959 :: root431Tail11
def root431Tail9 : List Nat := 6917 :: root431Tail10
def root431Tail8 : List Nat := 6551 :: root431Tail9
def root431Tail7 : List Nat := 5981 :: root431Tail8
def root431Tail6 : List Nat := 5879 :: root431Tail7
def root431Tail5 : List Nat := 4649 :: root431Tail6
def root431Tail4 : List Nat := 4637 :: root431Tail5
def root431Tail3 : List Nat := 4229 :: root431Tail4
def root431Tail2 : List Nat := 3989 :: root431Tail3
def root431Tail1 : List Nat := 3767 :: root431Tail2
def root431Tail0 : List Nat := 3527 :: root431Tail1
def root431Candidates0 : List Nat := [4649, 6551, 8669]
set_option maxRecDepth 8192 in
theorem root431Candidates0_checked : candidates edge 4 3023 3527 root431Tail1 winner.bound = root431Candidates0 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root431Child0 : search edge 3 [3023, 3527] root431Candidates0 6550 winner = winner := by decide
theorem root431Step0 : scan edge (search edge 3) 4 [3023] 3023 root431Tail0 winner =
    scan edge (search edge 3) 4 [3023] 3023 root431Tail1 winner := by
  rw [root431Tail0, scan, if_neg (by decide), root431Candidates0_checked]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail1
    (if 3 ≤ root431Candidates0.length then search edge 3 [3023, 3527] root431Candidates0 6550 winner else winner) = _
  rw [if_pos (by decide), root431Child0]
def root431Candidates1 : List Nat := [9437]
set_option maxRecDepth 8192 in
theorem root431Candidates1_checked : candidates edge 4 3023 3767 root431Tail2 winner.bound = root431Candidates1 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root431Step1 : scan edge (search edge 3) 4 [3023] 3023 root431Tail1 winner =
    scan edge (search edge 3) 4 [3023] 3023 root431Tail2 winner := by
  rw [root431Tail1, scan, if_neg (by decide), root431Candidates1_checked]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail2
    (if 3 ≤ root431Candidates1.length then search edge 3 [3023, 3767] root431Candidates1 6790 winner else winner) = _
  rw [if_neg (by decide)]
def root431Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root431Candidates2_checked : candidates edge 4 3023 3989 root431Tail3 winner.bound = root431Candidates2 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root431Step2 : scan edge (search edge 3) 4 [3023] 3023 root431Tail2 winner =
    scan edge (search edge 3) 4 [3023] 3023 root431Tail3 winner := by
  rw [root431Tail2, scan, if_neg (by decide), root431Candidates2_checked]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail3
    (if 3 ≤ root431Candidates2.length then search edge 3 [3023, 3989] root431Candidates2 7012 winner else winner) = _
  rw [if_neg (by decide)]
def root431Candidates3 : List Nat := [7589, 15683]
set_option maxRecDepth 8192 in
theorem root431Candidates3_checked : candidates edge 4 3023 4229 root431Tail4 winner.bound = root431Candidates3 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root431Step3 : scan edge (search edge 3) 4 [3023] 3023 root431Tail3 winner =
    scan edge (search edge 3) 4 [3023] 3023 root431Tail4 winner := by
  rw [root431Tail3, scan, if_neg (by decide), root431Candidates3_checked]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail4
    (if 3 ≤ root431Candidates3.length then search edge 3 [3023, 4229] root431Candidates3 7252 winner else winner) = _
  rw [if_neg (by decide)]
def root431Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root431Candidates4_checked : candidates edge 4 3023 4637 root431Tail5 winner.bound = root431Candidates4 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root431Step4 : scan edge (search edge 3) 4 [3023] 3023 root431Tail4 winner =
    scan edge (search edge 3) 4 [3023] 3023 root431Tail5 winner := by
  rw [root431Tail4, scan, if_neg (by decide), root431Candidates4_checked]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail5
    (if 3 ≤ root431Candidates4.length then search edge 3 [3023, 4637] root431Candidates4 7660 winner else winner) = _
  rw [if_neg (by decide)]
def root431Candidates5 : List Nat := [15683]
set_option maxRecDepth 8192 in
theorem root431Candidates5_checked : candidates edge 4 3023 4649 root431Tail6 winner.bound = root431Candidates5 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root431Step5 : scan edge (search edge 3) 4 [3023] 3023 root431Tail5 winner =
    scan edge (search edge 3) 4 [3023] 3023 root431Tail6 winner := by
  rw [root431Tail5, scan, if_neg (by decide), root431Candidates5_checked]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail6
    (if 3 ≤ root431Candidates5.length then search edge 3 [3023, 4649] root431Candidates5 7672 winner else winner) = _
  rw [if_neg (by decide)]
theorem root431Scan6 : scan edge (search edge 3) 4 [3023] 3023 root431Tail6 winner = winner := by rw [root431Tail6, scan, if_pos (by decide)]
theorem root431Scan5 : scan edge (search edge 3) 4 [3023] 3023 root431Tail5 winner = winner :=
  (root431Step5).trans root431Scan6
theorem root431Scan4 : scan edge (search edge 3) 4 [3023] 3023 root431Tail4 winner = winner :=
  (root431Step4).trans root431Scan5
theorem root431Scan3 : scan edge (search edge 3) 4 [3023] 3023 root431Tail3 winner = winner :=
  (root431Step3).trans root431Scan4
theorem root431Scan2 : scan edge (search edge 3) 4 [3023] 3023 root431Tail2 winner = winner :=
  (root431Step2).trans root431Scan3
theorem root431Scan1 : scan edge (search edge 3) 4 [3023] 3023 root431Tail1 winner = winner :=
  (root431Step1).trans root431Scan2
theorem root431Scan0 : scan edge (search edge 3) 4 [3023] 3023 root431Tail0 winner = winner :=
  (root431Step0).trans root431Scan1
set_option maxRecDepth 4096 in
theorem root431Branch : search edge 4 [3023] root431Candidates 3023 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3023] 3023 root431Tail0 winner = winner
  exact root431Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
