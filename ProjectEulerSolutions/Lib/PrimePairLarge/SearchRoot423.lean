import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root423Candidates : List Nat := Data.row2957
set_option maxRecDepth 8192 in
theorem root423Candidates_checked : candidates edge 5 0 2957 tail424 winner.bound = root423Candidates := by
  apply candidates_eq_row edge 5 0 2957 tail424 root423Candidates winner.bound tail424_ordered
  · decide
  · decide
  · funext q
    rfl
def root423Tail37 : List Nat := []
def root423Tail36 : List Nat := 23039 :: root423Tail37
def root423Tail35 : List Nat := 22853 :: root423Tail36
def root423Tail34 : List Nat := 21347 :: root423Tail35
def root423Tail33 : List Nat := 19961 :: root423Tail34
def root423Tail32 : List Nat := 19841 :: root423Tail33
def root423Tail31 : List Nat := 19031 :: root423Tail32
def root423Tail30 : List Nat := 18749 :: root423Tail31
def root423Tail29 : List Nat := 18617 :: root423Tail30
def root423Tail28 : List Nat := 18461 :: root423Tail29
def root423Tail27 : List Nat := 17189 :: root423Tail28
def root423Tail26 : List Nat := 16943 :: root423Tail27
def root423Tail25 : List Nat := 15467 :: root423Tail26
def root423Tail24 : List Nat := 15131 :: root423Tail25
def root423Tail23 : List Nat := 14759 :: root423Tail24
def root423Tail22 : List Nat := 14411 :: root423Tail23
def root423Tail21 : List Nat := 14087 :: root423Tail22
def root423Tail20 : List Nat := 13799 :: root423Tail21
def root423Tail19 : List Nat := 13109 :: root423Tail20
def root423Tail18 : List Nat := 12611 :: root423Tail19
def root423Tail17 : List Nat := 12479 :: root423Tail18
def root423Tail16 : List Nat := 9791 :: root423Tail17
def root423Tail15 : List Nat := 9551 :: root423Tail16
def root423Tail14 : List Nat := 9377 :: root423Tail15
def root423Tail13 : List Nat := 9323 :: root423Tail14
def root423Tail12 : List Nat := 9281 :: root423Tail13
def root423Tail11 : List Nat := 8831 :: root423Tail12
def root423Tail10 : List Nat := 8543 :: root423Tail11
def root423Tail9 : List Nat := 8291 :: root423Tail10
def root423Tail8 : List Nat := 7451 :: root423Tail9
def root423Tail7 : List Nat := 7307 :: root423Tail8
def root423Tail6 : List Nat := 6803 :: root423Tail7
def root423Tail5 : List Nat := 6221 :: root423Tail6
def root423Tail4 : List Nat := 6101 :: root423Tail5
def root423Tail3 : List Nat := 4973 :: root423Tail4
def root423Tail2 : List Nat := 4451 :: root423Tail3
def root423Tail1 : List Nat := 3449 :: root423Tail2
def root423Tail0 : List Nat := 3041 :: root423Tail1
def root423Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root423Candidates0_checked : candidates edge 4 2957 3041 root423Tail1 winner.bound = root423Candidates0 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root423Step0 : scan edge (search edge 3) 4 [2957] 2957 root423Tail0 winner =
    scan edge (search edge 3) 4 [2957] 2957 root423Tail1 winner := by
  rw [root423Tail0, scan, if_neg (by decide), root423Candidates0_checked]
  change scan edge (search edge 3) 4 [2957] 2957 root423Tail1
    (if 3 ≤ root423Candidates0.length then search edge 3 [2957, 3041] root423Candidates0 5998 winner else winner) = _
  rw [if_neg (by decide)]
def root423Candidates1 : List Nat := [4451]
set_option maxRecDepth 8192 in
theorem root423Candidates1_checked : candidates edge 4 2957 3449 root423Tail2 winner.bound = root423Candidates1 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root423Step1 : scan edge (search edge 3) 4 [2957] 2957 root423Tail1 winner =
    scan edge (search edge 3) 4 [2957] 2957 root423Tail2 winner := by
  rw [root423Tail1, scan, if_neg (by decide), root423Candidates1_checked]
  change scan edge (search edge 3) 4 [2957] 2957 root423Tail2
    (if 3 ≤ root423Candidates1.length then search edge 3 [2957, 3449] root423Candidates1 6406 winner else winner) = _
  rw [if_neg (by decide)]
def root423Candidates2 : List Nat := [6803, 7307, 9791, 16943]
set_option maxRecDepth 8192 in
theorem root423Candidates2_checked : candidates edge 4 2957 4451 root423Tail3 winner.bound = root423Candidates2 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root423Child2 : search edge 3 [2957, 4451] root423Candidates2 7408 winner = winner := by decide
theorem root423Step2 : scan edge (search edge 3) 4 [2957] 2957 root423Tail2 winner =
    scan edge (search edge 3) 4 [2957] 2957 root423Tail3 winner := by
  rw [root423Tail2, scan, if_neg (by decide), root423Candidates2_checked]
  change scan edge (search edge 3) 4 [2957] 2957 root423Tail3
    (if 3 ≤ root423Candidates2.length then search edge 3 [2957, 4451] root423Candidates2 7408 winner else winner) = _
  rw [if_pos (by decide), root423Child2]
def root423Candidates3 : List Nat := [6221, 14087]
set_option maxRecDepth 8192 in
theorem root423Candidates3_checked : candidates edge 4 2957 4973 root423Tail4 winner.bound = root423Candidates3 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root423Step3 : scan edge (search edge 3) 4 [2957] 2957 root423Tail3 winner =
    scan edge (search edge 3) 4 [2957] 2957 root423Tail4 winner := by
  rw [root423Tail3, scan, if_neg (by decide), root423Candidates3_checked]
  change scan edge (search edge 3) 4 [2957] 2957 root423Tail4
    (if 3 ≤ root423Candidates3.length then search edge 3 [2957, 4973] root423Candidates3 7930 winner else winner) = _
  rw [if_neg (by decide)]
theorem root423Scan4 : scan edge (search edge 3) 4 [2957] 2957 root423Tail4 winner = winner := by rw [root423Tail4, scan, if_pos (by decide)]
theorem root423Scan3 : scan edge (search edge 3) 4 [2957] 2957 root423Tail3 winner = winner :=
  (root423Step3).trans root423Scan4
theorem root423Scan2 : scan edge (search edge 3) 4 [2957] 2957 root423Tail2 winner = winner :=
  (root423Step2).trans root423Scan3
theorem root423Scan1 : scan edge (search edge 3) 4 [2957] 2957 root423Tail1 winner = winner :=
  (root423Step1).trans root423Scan2
theorem root423Scan0 : scan edge (search edge 3) 4 [2957] 2957 root423Tail0 winner = winner :=
  (root423Step0).trans root423Scan1
set_option maxRecDepth 4096 in
theorem root423Branch : search edge 4 [2957] root423Candidates 2957 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2957] 2957 root423Tail0 winner = winner
  exact root423Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
