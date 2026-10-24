import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root647Candidates : List Nat := Data.row4831
set_option maxRecDepth 8192 in
theorem root647Candidates_checked : candidates edge 5 0 4831 tail648 winner.bound = root647Candidates := by
  apply candidates_eq_row edge 5 0 4831 tail648 root647Candidates winner.bound tail648_ordered
  · decide
  · decide
  · funext q
    rfl
def root647Tail26 : List Nat := []
def root647Tail25 : List Nat := 21031 :: root647Tail26
def root647Tail24 : List Nat := 20593 :: root647Tail25
def root647Tail23 : List Nat := 20563 :: root647Tail24
def root647Tail22 : List Nat := 19993 :: root647Tail23
def root647Tail21 : List Nat := 17623 :: root647Tail22
def root647Tail20 : List Nat := 17497 :: root647Tail21
def root647Tail19 : List Nat := 17359 :: root647Tail20
def root647Tail18 : List Nat := 16561 :: root647Tail19
def root647Tail17 : List Nat := 14713 :: root647Tail18
def root647Tail16 : List Nat := 13567 :: root647Tail17
def root647Tail15 : List Nat := 12841 :: root647Tail16
def root647Tail14 : List Nat := 12823 :: root647Tail15
def root647Tail13 : List Nat := 11677 :: root647Tail14
def root647Tail12 : List Nat := 9631 :: root647Tail13
def root647Tail11 : List Nat := 9181 :: root647Tail12
def root647Tail10 : List Nat := 8737 :: root647Tail11
def root647Tail9 : List Nat := 8191 :: root647Tail10
def root647Tail8 : List Nat := 7699 :: root647Tail9
def root647Tail7 : List Nat := 7069 :: root647Tail8
def root647Tail6 : List Nat := 6967 :: root647Tail7
def root647Tail5 : List Nat := 6691 :: root647Tail6
def root647Tail4 : List Nat := 6553 :: root647Tail5
def root647Tail3 : List Nat := 6451 :: root647Tail4
def root647Tail2 : List Nat := 5623 :: root647Tail3
def root647Tail1 : List Nat := 5209 :: root647Tail2
def root647Tail0 : List Nat := 4951 :: root647Tail1
def root647Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root647Candidates0_checked : candidates edge 4 4831 4951 root647Tail1 winner.bound = root647Candidates0 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root647Step0 : scan edge (search edge 3) 4 [4831] 4831 root647Tail0 winner =
    scan edge (search edge 3) 4 [4831] 4831 root647Tail1 winner := by
  rw [root647Tail0, scan, if_neg (by decide), root647Candidates0_checked]
  change scan edge (search edge 3) 4 [4831] 4831 root647Tail1
    (if 3 ≤ root647Candidates0.length then search edge 3 [4831, 4951] root647Candidates0 9782 winner else winner) = _
  rw [if_neg (by decide)]
def root647Candidates1 : List Nat := [8191, 12823]
set_option maxRecDepth 8192 in
theorem root647Candidates1_checked : candidates edge 4 4831 5209 root647Tail2 winner.bound = root647Candidates1 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root647Step1 : scan edge (search edge 3) 4 [4831] 4831 root647Tail1 winner =
    scan edge (search edge 3) 4 [4831] 4831 root647Tail2 winner := by
  rw [root647Tail1, scan, if_neg (by decide), root647Candidates1_checked]
  change scan edge (search edge 3) 4 [4831] 4831 root647Tail2
    (if 3 ≤ root647Candidates1.length then search edge 3 [4831, 5209] root647Candidates1 10040 winner else winner) = _
  rw [if_neg (by decide)]
theorem root647Scan2 : scan edge (search edge 3) 4 [4831] 4831 root647Tail2 winner = winner := by rw [root647Tail2, scan, if_pos (by decide)]
theorem root647Scan1 : scan edge (search edge 3) 4 [4831] 4831 root647Tail1 winner = winner :=
  (root647Step1).trans root647Scan2
theorem root647Scan0 : scan edge (search edge 3) 4 [4831] 4831 root647Tail0 winner = winner :=
  (root647Step0).trans root647Scan1
set_option maxRecDepth 4096 in
theorem root647Branch : search edge 4 [4831] root647Candidates 4831 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4831] 4831 root647Tail0 winner = winner
  exact root647Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
