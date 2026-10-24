import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root588Candidates : List Nat := Data.row4327
set_option maxRecDepth 8192 in
theorem root588Candidates_checked : candidates edge 5 0 4327 tail589 winner.bound = root588Candidates := by
  apply candidates_eq_row edge 5 0 4327 tail589 root588Candidates winner.bound tail589_ordered
  · decide
  · decide
  · funext q
    rfl
def root588Tail38 : List Nat := []
def root588Tail37 : List Nat := 21277 :: root588Tail38
def root588Tail36 : List Nat := 20593 :: root588Tail37
def root588Tail35 : List Nat := 19387 :: root588Tail36
def root588Tail34 : List Nat := 19051 :: root588Tail35
def root588Tail33 : List Nat := 17749 :: root588Tail34
def root588Tail32 : List Nat := 17539 :: root588Tail33
def root588Tail31 : List Nat := 17137 :: root588Tail32
def root588Tail30 : List Nat := 17107 :: root588Tail31
def root588Tail29 : List Nat := 17011 :: root588Tail30
def root588Tail28 : List Nat := 16069 :: root588Tail29
def root588Tail27 : List Nat := 15919 :: root588Tail28
def root588Tail26 : List Nat := 15193 :: root588Tail27
def root588Tail25 : List Nat := 14389 :: root588Tail26
def root588Tail24 : List Nat := 14251 :: root588Tail25
def root588Tail23 : List Nat := 13063 :: root588Tail24
def root588Tail22 : List Nat := 12967 :: root588Tail23
def root588Tail21 : List Nat := 12109 :: root588Tail22
def root588Tail20 : List Nat := 12097 :: root588Tail21
def root588Tail19 : List Nat := 11551 :: root588Tail20
def root588Tail18 : List Nat := 10357 :: root588Tail19
def root588Tail17 : List Nat := 9883 :: root588Tail18
def root588Tail16 : List Nat := 9871 :: root588Tail17
def root588Tail15 : List Nat := 9697 :: root588Tail16
def root588Tail14 : List Nat := 9619 :: root588Tail15
def root588Tail13 : List Nat := 9337 :: root588Tail14
def root588Tail12 : List Nat := 9241 :: root588Tail13
def root588Tail11 : List Nat := 8233 :: root588Tail12
def root588Tail10 : List Nat := 7963 :: root588Tail11
def root588Tail9 : List Nat := 7639 :: root588Tail10
def root588Tail8 : List Nat := 6703 :: root588Tail9
def root588Tail7 : List Nat := 6553 :: root588Tail8
def root588Tail6 : List Nat := 6481 :: root588Tail7
def root588Tail5 : List Nat := 5737 :: root588Tail6
def root588Tail4 : List Nat := 5503 :: root588Tail5
def root588Tail3 : List Nat := 5107 :: root588Tail4
def root588Tail2 : List Nat := 4801 :: root588Tail3
def root588Tail1 : List Nat := 4591 :: root588Tail2
def root588Tail0 : List Nat := 4519 :: root588Tail1
def root588Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root588Candidates0_checked : candidates edge 4 4327 4519 root588Tail1 winner.bound = root588Candidates0 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root588Step0 : scan edge (search edge 3) 4 [4327] 4327 root588Tail0 winner =
    scan edge (search edge 3) 4 [4327] 4327 root588Tail1 winner := by
  rw [root588Tail0, scan, if_neg (by decide), root588Candidates0_checked]
  change scan edge (search edge 3) 4 [4327] 4327 root588Tail1
    (if 3 ≤ root588Candidates0.length then search edge 3 [4327, 4519] root588Candidates0 8846 winner else winner) = _
  rw [if_neg (by decide)]
def root588Candidates1 : List Nat := [9883, 12109]
set_option maxRecDepth 8192 in
theorem root588Candidates1_checked : candidates edge 4 4327 4591 root588Tail2 winner.bound = root588Candidates1 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root588Step1 : scan edge (search edge 3) 4 [4327] 4327 root588Tail1 winner =
    scan edge (search edge 3) 4 [4327] 4327 root588Tail2 winner := by
  rw [root588Tail1, scan, if_neg (by decide), root588Candidates1_checked]
  change scan edge (search edge 3) 4 [4327] 4327 root588Tail2
    (if 3 ≤ root588Candidates1.length then search edge 3 [4327, 4591] root588Candidates1 8918 winner else winner) = _
  rw [if_neg (by decide)]
def root588Candidates2 : List Nat := [12109, 15919]
set_option maxRecDepth 8192 in
theorem root588Candidates2_checked : candidates edge 4 4327 4801 root588Tail3 winner.bound = root588Candidates2 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root588Step2 : scan edge (search edge 3) 4 [4327] 4327 root588Tail2 winner =
    scan edge (search edge 3) 4 [4327] 4327 root588Tail3 winner := by
  rw [root588Tail2, scan, if_neg (by decide), root588Candidates2_checked]
  change scan edge (search edge 3) 4 [4327] 4327 root588Tail3
    (if 3 ≤ root588Candidates2.length then search edge 3 [4327, 4801] root588Candidates2 9128 winner else winner) = _
  rw [if_neg (by decide)]
def root588Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root588Candidates3_checked : candidates edge 4 4327 5107 root588Tail4 winner.bound = root588Candidates3 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root588Step3 : scan edge (search edge 3) 4 [4327] 4327 root588Tail3 winner =
    scan edge (search edge 3) 4 [4327] 4327 root588Tail4 winner := by
  rw [root588Tail3, scan, if_neg (by decide), root588Candidates3_checked]
  change scan edge (search edge 3) 4 [4327] 4327 root588Tail4
    (if 3 ≤ root588Candidates3.length then search edge 3 [4327, 5107] root588Candidates3 9434 winner else winner) = _
  rw [if_neg (by decide)]
theorem root588Scan4 : scan edge (search edge 3) 4 [4327] 4327 root588Tail4 winner = winner := by rw [root588Tail4, scan, if_pos (by decide)]
theorem root588Scan3 : scan edge (search edge 3) 4 [4327] 4327 root588Tail3 winner = winner :=
  (root588Step3).trans root588Scan4
theorem root588Scan2 : scan edge (search edge 3) 4 [4327] 4327 root588Tail2 winner = winner :=
  (root588Step2).trans root588Scan3
theorem root588Scan1 : scan edge (search edge 3) 4 [4327] 4327 root588Tail1 winner = winner :=
  (root588Step1).trans root588Scan2
theorem root588Scan0 : scan edge (search edge 3) 4 [4327] 4327 root588Tail0 winner = winner :=
  (root588Step0).trans root588Scan1
set_option maxRecDepth 4096 in
theorem root588Branch : search edge 4 [4327] root588Candidates 4327 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4327] 4327 root588Tail0 winner = winner
  exact root588Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
