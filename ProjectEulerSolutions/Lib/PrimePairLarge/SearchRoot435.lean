import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root435Candidates : List Nat := Data.row3061
set_option maxRecDepth 8192 in
theorem root435Candidates_checked : candidates edge 5 0 3061 tail436 winner.bound = root435Candidates := by
  apply candidates_eq_row edge 5 0 3061 tail436 root435Candidates winner.bound tail436_ordered
  · decide
  · decide
  · funext q
    rfl
def root435Tail34 : List Nat := []
def root435Tail33 : List Nat := 22741 :: root435Tail34
def root435Tail32 : List Nat := 22717 :: root435Tail33
def root435Tail31 : List Nat := 22093 :: root435Tail32
def root435Tail30 : List Nat := 21433 :: root435Tail31
def root435Tail29 : List Nat := 21391 :: root435Tail30
def root435Tail28 : List Nat := 20011 :: root435Tail29
def root435Tail27 : List Nat := 17491 :: root435Tail28
def root435Tail26 : List Nat := 16843 :: root435Tail27
def root435Tail25 : List Nat := 16567 :: root435Tail26
def root435Tail24 : List Nat := 16333 :: root435Tail25
def root435Tail23 : List Nat := 15919 :: root435Tail24
def root435Tail22 : List Nat := 15877 :: root435Tail23
def root435Tail21 : List Nat := 15493 :: root435Tail22
def root435Tail20 : List Nat := 15259 :: root435Tail21
def root435Tail19 : List Nat := 13963 :: root435Tail20
def root435Tail18 : List Nat := 13627 :: root435Tail19
def root435Tail17 : List Nat := 12853 :: root435Tail18
def root435Tail16 : List Nat := 12721 :: root435Tail17
def root435Tail15 : List Nat := 12619 :: root435Tail16
def root435Tail14 : List Nat := 12511 :: root435Tail15
def root435Tail13 : List Nat := 8719 :: root435Tail14
def root435Tail12 : List Nat := 8563 :: root435Tail13
def root435Tail11 : List Nat := 8293 :: root435Tail12
def root435Tail10 : List Nat := 8167 :: root435Tail11
def root435Tail9 : List Nat := 6907 :: root435Tail10
def root435Tail8 : List Nat := 6829 :: root435Tail9
def root435Tail7 : List Nat := 6673 :: root435Tail8
def root435Tail6 : List Nat := 6343 :: root435Tail7
def root435Tail5 : List Nat := 5857 :: root435Tail6
def root435Tail4 : List Nat := 5437 :: root435Tail5
def root435Tail3 : List Nat := 4729 :: root435Tail4
def root435Tail2 : List Nat := 3613 :: root435Tail3
def root435Tail1 : List Nat := 3607 :: root435Tail2
def root435Tail0 : List Nat := 3571 :: root435Tail1
def root435Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root435Candidates0_checked : candidates edge 4 3061 3571 root435Tail1 winner.bound = root435Candidates0 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root435Step0 : scan edge (search edge 3) 4 [3061] 3061 root435Tail0 winner =
    scan edge (search edge 3) 4 [3061] 3061 root435Tail1 winner := by
  rw [root435Tail0, scan, if_neg (by decide), root435Candidates0_checked]
  change scan edge (search edge 3) 4 [3061] 3061 root435Tail1
    (if 3 ≤ root435Candidates0.length then search edge 3 [3061, 3571] root435Candidates0 6632 winner else winner) = _
  rw [if_neg (by decide)]
def root435Candidates1 : List Nat := [15259]
set_option maxRecDepth 8192 in
theorem root435Candidates1_checked : candidates edge 4 3061 3607 root435Tail2 winner.bound = root435Candidates1 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root435Step1 : scan edge (search edge 3) 4 [3061] 3061 root435Tail1 winner =
    scan edge (search edge 3) 4 [3061] 3061 root435Tail2 winner := by
  rw [root435Tail1, scan, if_neg (by decide), root435Candidates1_checked]
  change scan edge (search edge 3) 4 [3061] 3061 root435Tail2
    (if 3 ≤ root435Candidates1.length then search edge 3 [3061, 3607] root435Candidates1 6668 winner else winner) = _
  rw [if_neg (by decide)]
def root435Candidates2 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root435Candidates2_checked : candidates edge 4 3061 3613 root435Tail3 winner.bound = root435Candidates2 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root435Step2 : scan edge (search edge 3) 4 [3061] 3061 root435Tail2 winner =
    scan edge (search edge 3) 4 [3061] 3061 root435Tail3 winner := by
  rw [root435Tail2, scan, if_neg (by decide), root435Candidates2_checked]
  change scan edge (search edge 3) 4 [3061] 3061 root435Tail3
    (if 3 ≤ root435Candidates2.length then search edge 3 [3061, 3613] root435Candidates2 6674 winner else winner) = _
  rw [if_neg (by decide)]
def root435Candidates3 : List Nat := [12619]
set_option maxRecDepth 8192 in
theorem root435Candidates3_checked : candidates edge 4 3061 4729 root435Tail4 winner.bound = root435Candidates3 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
theorem root435Step3 : scan edge (search edge 3) 4 [3061] 3061 root435Tail3 winner =
    scan edge (search edge 3) 4 [3061] 3061 root435Tail4 winner := by
  rw [root435Tail3, scan, if_neg (by decide), root435Candidates3_checked]
  change scan edge (search edge 3) 4 [3061] 3061 root435Tail4
    (if 3 ≤ root435Candidates3.length then search edge 3 [3061, 4729] root435Candidates3 7790 winner else winner) = _
  rw [if_neg (by decide)]
def root435Candidates4 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root435Candidates4_checked : candidates edge 4 3061 5437 root435Tail5 winner.bound = root435Candidates4 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root435Step4 : scan edge (search edge 3) 4 [3061] 3061 root435Tail4 winner =
    scan edge (search edge 3) 4 [3061] 3061 root435Tail5 winner := by
  rw [root435Tail4, scan, if_neg (by decide), root435Candidates4_checked]
  change scan edge (search edge 3) 4 [3061] 3061 root435Tail5
    (if 3 ≤ root435Candidates4.length then search edge 3 [3061, 5437] root435Candidates4 8498 winner else winner) = _
  rw [if_neg (by decide)]
theorem root435Scan5 : scan edge (search edge 3) 4 [3061] 3061 root435Tail5 winner = winner := by rw [root435Tail5, scan, if_pos (by decide)]
theorem root435Scan4 : scan edge (search edge 3) 4 [3061] 3061 root435Tail4 winner = winner :=
  (root435Step4).trans root435Scan5
theorem root435Scan3 : scan edge (search edge 3) 4 [3061] 3061 root435Tail3 winner = winner :=
  (root435Step3).trans root435Scan4
theorem root435Scan2 : scan edge (search edge 3) 4 [3061] 3061 root435Tail2 winner = winner :=
  (root435Step2).trans root435Scan3
theorem root435Scan1 : scan edge (search edge 3) 4 [3061] 3061 root435Tail1 winner = winner :=
  (root435Step1).trans root435Scan2
theorem root435Scan0 : scan edge (search edge 3) 4 [3061] 3061 root435Tail0 winner = winner :=
  (root435Step0).trans root435Scan1
set_option maxRecDepth 4096 in
theorem root435Branch : search edge 4 [3061] root435Candidates 3061 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3061] 3061 root435Tail0 winner = winner
  exact root435Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
