import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root576Candidates : List Nat := Data.row4229
set_option maxRecDepth 8192 in
theorem root576Candidates_checked : candidates edge 5 0 4229 tail577 winner.bound = root576Candidates := by
  apply candidates_eq_row edge 5 0 4229 tail577 root576Candidates winner.bound tail577_ordered
  · decide
  · decide
  · funext q
    rfl
def root576Tail26 : List Nat := []
def root576Tail25 : List Nat := 21221 :: root576Tail26
def root576Tail24 : List Nat := 20483 :: root576Tail25
def root576Tail23 : List Nat := 20219 :: root576Tail24
def root576Tail22 : List Nat := 18719 :: root576Tail23
def root576Tail21 : List Nat := 18371 :: root576Tail22
def root576Tail20 : List Nat := 17669 :: root576Tail21
def root576Tail19 : List Nat := 17597 :: root576Tail20
def root576Tail18 : List Nat := 17021 :: root576Tail19
def root576Tail17 : List Nat := 16883 :: root576Tail18
def root576Tail16 : List Nat := 16619 :: root576Tail17
def root576Tail15 : List Nat := 16607 :: root576Tail16
def root576Tail14 : List Nat := 16349 :: root576Tail15
def root576Tail13 : List Nat := 16073 :: root576Tail14
def root576Tail12 : List Nat := 15683 :: root576Tail13
def root576Tail11 : List Nat := 11549 :: root576Tail12
def root576Tail10 : List Nat := 11399 :: root576Tail11
def root576Tail9 : List Nat := 11213 :: root576Tail10
def root576Tail8 : List Nat := 10457 :: root576Tail9
def root576Tail7 : List Nat := 9941 :: root576Tail8
def root576Tail6 : List Nat := 8831 :: root576Tail7
def root576Tail5 : List Nat := 7643 :: root576Tail6
def root576Tail4 : List Nat := 7589 :: root576Tail5
def root576Tail3 : List Nat := 6689 :: root576Tail4
def root576Tail2 : List Nat := 5399 :: root576Tail3
def root576Tail1 : List Nat := 5039 :: root576Tail2
def root576Tail0 : List Nat := 4391 :: root576Tail1
def root576Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root576Candidates0_checked : candidates edge 4 4229 4391 root576Tail1 winner.bound = root576Candidates0 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root576Step0 : scan edge (search edge 3) 4 [4229] 4229 root576Tail0 winner =
    scan edge (search edge 3) 4 [4229] 4229 root576Tail1 winner := by
  rw [root576Tail0, scan, if_neg (by decide), root576Candidates0_checked]
  change scan edge (search edge 3) 4 [4229] 4229 root576Tail1
    (if 3 ≤ root576Candidates0.length then search edge 3 [4229, 4391] root576Candidates0 8620 winner else winner) = _
  rw [if_neg (by decide)]
def root576Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root576Candidates1_checked : candidates edge 4 4229 5039 root576Tail2 winner.bound = root576Candidates1 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root576Step1 : scan edge (search edge 3) 4 [4229] 4229 root576Tail1 winner =
    scan edge (search edge 3) 4 [4229] 4229 root576Tail2 winner := by
  rw [root576Tail1, scan, if_neg (by decide), root576Candidates1_checked]
  change scan edge (search edge 3) 4 [4229] 4229 root576Tail2
    (if 3 ≤ root576Candidates1.length then search edge 3 [4229, 5039] root576Candidates1 9268 winner else winner) = _
  rw [if_neg (by decide)]
def root576Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root576Candidates2_checked : candidates edge 4 4229 5399 root576Tail3 winner.bound = root576Candidates2 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root576Step2 : scan edge (search edge 3) 4 [4229] 4229 root576Tail2 winner =
    scan edge (search edge 3) 4 [4229] 4229 root576Tail3 winner := by
  rw [root576Tail2, scan, if_neg (by decide), root576Candidates2_checked]
  change scan edge (search edge 3) 4 [4229] 4229 root576Tail3
    (if 3 ≤ root576Candidates2.length then search edge 3 [4229, 5399] root576Candidates2 9628 winner else winner) = _
  rw [if_neg (by decide)]
theorem root576Scan3 : scan edge (search edge 3) 4 [4229] 4229 root576Tail3 winner = winner := by rw [root576Tail3, scan, if_pos (by decide)]
theorem root576Scan2 : scan edge (search edge 3) 4 [4229] 4229 root576Tail2 winner = winner :=
  (root576Step2).trans root576Scan3
theorem root576Scan1 : scan edge (search edge 3) 4 [4229] 4229 root576Tail1 winner = winner :=
  (root576Step1).trans root576Scan2
theorem root576Scan0 : scan edge (search edge 3) 4 [4229] 4229 root576Tail0 winner = winner :=
  (root576Step0).trans root576Scan1
set_option maxRecDepth 4096 in
theorem root576Branch : search edge 4 [4229] root576Candidates 4229 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4229] 4229 root576Tail0 winner = winner
  exact root576Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
