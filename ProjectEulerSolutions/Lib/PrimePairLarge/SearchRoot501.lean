import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root501Candidates : List Nat := Data.row3607
set_option maxRecDepth 8192 in
theorem root501Candidates_checked : candidates edge 5 0 3607 tail502 winner.bound = root501Candidates := by
  apply candidates_eq_row edge 5 0 3607 tail502 root501Candidates winner.bound tail502_ordered
  · decide
  · decide
  · funext q
    rfl
def root501Tail31 : List Nat := []
def root501Tail30 : List Nat := 22381 :: root501Tail31
def root501Tail29 : List Nat := 22279 :: root501Tail30
def root501Tail28 : List Nat := 21379 :: root501Tail29
def root501Tail27 : List Nat := 20509 :: root501Tail28
def root501Tail26 : List Nat := 19717 :: root501Tail27
def root501Tail25 : List Nat := 19231 :: root501Tail26
def root501Tail24 : List Nat := 18859 :: root501Tail25
def root501Tail23 : List Nat := 18517 :: root501Tail24
def root501Tail22 : List Nat := 18493 :: root501Tail23
def root501Tail21 : List Nat := 17011 :: root501Tail22
def root501Tail20 : List Nat := 16381 :: root501Tail21
def root501Tail19 : List Nat := 16063 :: root501Tail20
def root501Tail18 : List Nat := 15307 :: root501Tail19
def root501Tail17 : List Nat := 15259 :: root501Tail18
def root501Tail16 : List Nat := 14797 :: root501Tail17
def root501Tail15 : List Nat := 14737 :: root501Tail16
def root501Tail14 : List Nat := 13033 :: root501Tail15
def root501Tail13 : List Nat := 8941 :: root501Tail14
def root501Tail12 : List Nat := 8803 :: root501Tail13
def root501Tail11 : List Nat := 8521 :: root501Tail12
def root501Tail10 : List Nat := 8101 :: root501Tail11
def root501Tail9 : List Nat := 8089 :: root501Tail10
def root501Tail8 : List Nat := 7669 :: root501Tail9
def root501Tail7 : List Nat := 7351 :: root501Tail8
def root501Tail6 : List Nat := 6577 :: root501Tail7
def root501Tail5 : List Nat := 6571 :: root501Tail6
def root501Tail4 : List Nat := 6379 :: root501Tail5
def root501Tail3 : List Nat := 6367 :: root501Tail4
def root501Tail2 : List Nat := 5077 :: root501Tail3
def root501Tail1 : List Nat := 5023 :: root501Tail2
def root501Tail0 : List Nat := 4099 :: root501Tail1
def root501Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root501Candidates0_checked : candidates edge 4 3607 4099 root501Tail1 winner.bound = root501Candidates0 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root501Step0 : scan edge (search edge 3) 4 [3607] 3607 root501Tail0 winner =
    scan edge (search edge 3) 4 [3607] 3607 root501Tail1 winner := by
  rw [root501Tail0, scan, if_neg (by decide), root501Candidates0_checked]
  change scan edge (search edge 3) 4 [3607] 3607 root501Tail1
    (if 3 ≤ root501Candidates0.length then search edge 3 [3607, 4099] root501Candidates0 7706 winner else winner) = _
  rw [if_neg (by decide)]
def root501Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root501Candidates1_checked : candidates edge 4 3607 5023 root501Tail2 winner.bound = root501Candidates1 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root501Step1 : scan edge (search edge 3) 4 [3607] 3607 root501Tail1 winner =
    scan edge (search edge 3) 4 [3607] 3607 root501Tail2 winner := by
  rw [root501Tail1, scan, if_neg (by decide), root501Candidates1_checked]
  change scan edge (search edge 3) 4 [3607] 3607 root501Tail2
    (if 3 ≤ root501Candidates1.length then search edge 3 [3607, 5023] root501Candidates1 8630 winner else winner) = _
  rw [if_neg (by decide)]
def root501Candidates2 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root501Candidates2_checked : candidates edge 4 3607 5077 root501Tail3 winner.bound = root501Candidates2 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root501Step2 : scan edge (search edge 3) 4 [3607] 3607 root501Tail2 winner =
    scan edge (search edge 3) 4 [3607] 3607 root501Tail3 winner := by
  rw [root501Tail2, scan, if_neg (by decide), root501Candidates2_checked]
  change scan edge (search edge 3) 4 [3607] 3607 root501Tail3
    (if 3 ≤ root501Candidates2.length then search edge 3 [3607, 5077] root501Candidates2 8684 winner else winner) = _
  rw [if_neg (by decide)]
theorem root501Scan3 : scan edge (search edge 3) 4 [3607] 3607 root501Tail3 winner = winner := by rw [root501Tail3, scan, if_pos (by decide)]
theorem root501Scan2 : scan edge (search edge 3) 4 [3607] 3607 root501Tail2 winner = winner :=
  (root501Step2).trans root501Scan3
theorem root501Scan1 : scan edge (search edge 3) 4 [3607] 3607 root501Tail1 winner = winner :=
  (root501Step1).trans root501Scan2
theorem root501Scan0 : scan edge (search edge 3) 4 [3607] 3607 root501Tail0 winner = winner :=
  (root501Step0).trans root501Scan1
set_option maxRecDepth 4096 in
theorem root501Branch : search edge 4 [3607] root501Candidates 3607 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3607] 3607 root501Tail0 winner = winner
  exact root501Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
