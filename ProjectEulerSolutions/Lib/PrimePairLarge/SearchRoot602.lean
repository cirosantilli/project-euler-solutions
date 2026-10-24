import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root602Candidates : List Nat := Data.row4451
set_option maxRecDepth 8192 in
theorem root602Candidates_checked : candidates edge 5 0 4451 tail603 winner.bound = root602Candidates := by
  apply candidates_eq_row edge 5 0 4451 tail603 root602Candidates winner.bound tail603_ordered
  · decide
  · decide
  · funext q
    rfl
def root602Tail32 : List Nat := []
def root602Tail31 : List Nat := 21503 :: root602Tail32
def root602Tail30 : List Nat := 20873 :: root602Tail31
def root602Tail29 : List Nat := 20369 :: root602Tail30
def root602Tail28 : List Nat := 19937 :: root602Tail29
def root602Tail27 : List Nat := 18911 :: root602Tail28
def root602Tail26 : List Nat := 17909 :: root602Tail27
def root602Tail25 : List Nat := 17471 :: root602Tail26
def root602Tail24 : List Nat := 16943 :: root602Tail25
def root602Tail23 : List Nat := 16103 :: root602Tail24
def root602Tail22 : List Nat := 15791 :: root602Tail23
def root602Tail21 : List Nat := 15053 :: root602Tail22
def root602Tail20 : List Nat := 14753 :: root602Tail21
def root602Tail19 : List Nat := 14669 :: root602Tail20
def root602Tail18 : List Nat := 14057 :: root602Tail19
def root602Tail17 : List Nat := 13103 :: root602Tail18
def root602Tail16 : List Nat := 11789 :: root602Tail17
def root602Tail15 : List Nat := 11483 :: root602Tail16
def root602Tail14 : List Nat := 9791 :: root602Tail15
def root602Tail13 : List Nat := 9533 :: root602Tail14
def root602Tail12 : List Nat := 9521 :: root602Tail13
def root602Tail11 : List Nat := 8963 :: root602Tail12
def root602Tail10 : List Nat := 8861 :: root602Tail11
def root602Tail9 : List Nat := 8783 :: root602Tail10
def root602Tail8 : List Nat := 8681 :: root602Tail9
def root602Tail7 : List Nat := 7853 :: root602Tail8
def root602Tail6 : List Nat := 7307 :: root602Tail7
def root602Tail5 : List Nat := 6833 :: root602Tail6
def root602Tail4 : List Nat := 6803 :: root602Tail5
def root602Tail3 : List Nat := 6599 :: root602Tail4
def root602Tail2 : List Nat := 6257 :: root602Tail3
def root602Tail1 : List Nat := 6089 :: root602Tail2
def root602Tail0 : List Nat := 5279 :: root602Tail1
def root602Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root602Candidates0_checked : candidates edge 4 4451 5279 root602Tail1 winner.bound = root602Candidates0 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root602Step0 : scan edge (search edge 3) 4 [4451] 4451 root602Tail0 winner =
    scan edge (search edge 3) 4 [4451] 4451 root602Tail1 winner := by
  rw [root602Tail0, scan, if_neg (by decide), root602Candidates0_checked]
  change scan edge (search edge 3) 4 [4451] 4451 root602Tail1
    (if 3 ≤ root602Candidates0.length then search edge 3 [4451, 5279] root602Candidates0 9730 winner else winner) = _
  rw [if_neg (by decide)]
theorem root602Scan1 : scan edge (search edge 3) 4 [4451] 4451 root602Tail1 winner = winner := by rw [root602Tail1, scan, if_pos (by decide)]
theorem root602Scan0 : scan edge (search edge 3) 4 [4451] 4451 root602Tail0 winner = winner :=
  (root602Step0).trans root602Scan1
set_option maxRecDepth 4096 in
theorem root602Branch : search edge 4 [4451] root602Candidates 4451 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4451] 4451 root602Tail0 winner = winner
  exact root602Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
