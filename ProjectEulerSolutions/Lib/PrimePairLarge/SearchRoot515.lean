import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root515Candidates : List Nat := Data.row3709
set_option maxRecDepth 8192 in
theorem root515Candidates_checked : candidates edge 5 0 3709 tail516 winner.bound = root515Candidates := by
  apply candidates_eq_row edge 5 0 3709 tail516 root515Candidates winner.bound tail516_ordered
  · decide
  · decide
  · funext q
    rfl
def root515Tail27 : List Nat := []
def root515Tail26 : List Nat := 21319 :: root515Tail27
def root515Tail25 : List Nat := 20593 :: root515Tail26
def root515Tail24 : List Nat := 20233 :: root515Tail25
def root515Tail23 : List Nat := 20047 :: root515Tail24
def root515Tail22 : List Nat := 19687 :: root515Tail23
def root515Tail21 : List Nat := 19603 :: root515Tail22
def root515Tail20 : List Nat := 18481 :: root515Tail21
def root515Tail19 : List Nat := 18211 :: root515Tail20
def root515Tail18 : List Nat := 17377 :: root515Tail19
def root515Tail17 : List Nat := 15817 :: root515Tail18
def root515Tail16 : List Nat := 15679 :: root515Tail17
def root515Tail15 : List Nat := 14503 :: root515Tail16
def root515Tail14 : List Nat := 13831 :: root515Tail15
def root515Tail13 : List Nat := 13339 :: root515Tail14
def root515Tail12 : List Nat := 12601 :: root515Tail13
def root515Tail11 : List Nat := 11743 :: root515Tail12
def root515Tail10 : List Nat := 11257 :: root515Tail11
def root515Tail9 : List Nat := 10711 :: root515Tail10
def root515Tail8 : List Nat := 10597 :: root515Tail9
def root515Tail7 : List Nat := 8779 :: root515Tail8
def root515Tail6 : List Nat := 8389 :: root515Tail7
def root515Tail5 : List Nat := 8377 :: root515Tail6
def root515Tail4 : List Nat := 7369 :: root515Tail5
def root515Tail3 : List Nat := 6991 :: root515Tail4
def root515Tail2 : List Nat := 4621 :: root515Tail3
def root515Tail1 : List Nat := 4261 :: root515Tail2
def root515Tail0 : List Nat := 4231 :: root515Tail1
def root515Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root515Candidates0_checked : candidates edge 4 3709 4231 root515Tail1 winner.bound = root515Candidates0 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root515Step0 : scan edge (search edge 3) 4 [3709] 3709 root515Tail0 winner =
    scan edge (search edge 3) 4 [3709] 3709 root515Tail1 winner := by
  rw [root515Tail0, scan, if_neg (by decide), root515Candidates0_checked]
  change scan edge (search edge 3) 4 [3709] 3709 root515Tail1
    (if 3 ≤ root515Candidates0.length then search edge 3 [3709, 4231] root515Candidates0 7940 winner else winner) = _
  rw [if_neg (by decide)]
def root515Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root515Candidates1_checked : candidates edge 4 3709 4261 root515Tail2 winner.bound = root515Candidates1 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root515Step1 : scan edge (search edge 3) 4 [3709] 3709 root515Tail1 winner =
    scan edge (search edge 3) 4 [3709] 3709 root515Tail2 winner := by
  rw [root515Tail1, scan, if_neg (by decide), root515Candidates1_checked]
  change scan edge (search edge 3) 4 [3709] 3709 root515Tail2
    (if 3 ≤ root515Candidates1.length then search edge 3 [3709, 4261] root515Candidates1 7970 winner else winner) = _
  rw [if_neg (by decide)]
def root515Candidates2 : List Nat := [11257, 17377]
set_option maxRecDepth 8192 in
theorem root515Candidates2_checked : candidates edge 4 3709 4621 root515Tail3 winner.bound = root515Candidates2 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root515Step2 : scan edge (search edge 3) 4 [3709] 3709 root515Tail2 winner =
    scan edge (search edge 3) 4 [3709] 3709 root515Tail3 winner := by
  rw [root515Tail2, scan, if_neg (by decide), root515Candidates2_checked]
  change scan edge (search edge 3) 4 [3709] 3709 root515Tail3
    (if 3 ≤ root515Candidates2.length then search edge 3 [3709, 4621] root515Candidates2 8330 winner else winner) = _
  rw [if_neg (by decide)]
theorem root515Scan3 : scan edge (search edge 3) 4 [3709] 3709 root515Tail3 winner = winner := by rw [root515Tail3, scan, if_pos (by decide)]
theorem root515Scan2 : scan edge (search edge 3) 4 [3709] 3709 root515Tail2 winner = winner :=
  (root515Step2).trans root515Scan3
theorem root515Scan1 : scan edge (search edge 3) 4 [3709] 3709 root515Tail1 winner = winner :=
  (root515Step1).trans root515Scan2
theorem root515Scan0 : scan edge (search edge 3) 4 [3709] 3709 root515Tail0 winner = winner :=
  (root515Step0).trans root515Scan1
set_option maxRecDepth 4096 in
theorem root515Branch : search edge 4 [3709] root515Candidates 3709 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3709] 3709 root515Tail0 winner = winner
  exact root515Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
