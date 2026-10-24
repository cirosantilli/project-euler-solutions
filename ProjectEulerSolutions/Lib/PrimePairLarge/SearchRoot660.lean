import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root660Candidates : List Nat := Data.row4957
set_option maxRecDepth 8192 in
theorem root660Candidates_checked : candidates edge 5 0 4957 tail661 winner.bound = root660Candidates := by
  apply candidates_eq_row edge 5 0 4957 tail661 root660Candidates winner.bound tail661_ordered
  · decide
  · decide
  · funext q
    rfl
def root660Tail36 : List Nat := []
def root660Tail35 : List Nat := 20551 :: root660Tail36
def root660Tail34 : List Nat := 20161 :: root660Tail35
def root660Tail33 : List Nat := 20047 :: root660Tail34
def root660Tail32 : List Nat := 19891 :: root660Tail33
def root660Tail31 : List Nat := 19141 :: root660Tail32
def root660Tail30 : List Nat := 18523 :: root660Tail31
def root660Tail29 : List Nat := 18397 :: root660Tail30
def root660Tail28 : List Nat := 18313 :: root660Tail29
def root660Tail27 : List Nat := 17419 :: root660Tail28
def root660Tail26 : List Nat := 17203 :: root660Tail27
def root660Tail25 : List Nat := 17041 :: root660Tail26
def root660Tail24 : List Nat := 16267 :: root660Tail25
def root660Tail23 : List Nat := 16069 :: root660Tail24
def root660Tail22 : List Nat := 15919 :: root660Tail23
def root660Tail21 : List Nat := 15607 :: root660Tail22
def root660Tail20 : List Nat := 15361 :: root660Tail21
def root660Tail19 : List Nat := 15091 :: root660Tail20
def root660Tail18 : List Nat := 13723 :: root660Tail19
def root660Tail17 : List Nat := 13441 :: root660Tail18
def root660Tail16 : List Nat := 12697 :: root660Tail17
def root660Tail15 : List Nat := 12601 :: root660Tail16
def root660Tail14 : List Nat := 11941 :: root660Tail15
def root660Tail13 : List Nat := 10723 :: root660Tail14
def root660Tail12 : List Nat := 9787 :: root660Tail13
def root660Tail11 : List Nat := 9613 :: root660Tail12
def root660Tail10 : List Nat := 8929 :: root660Tail11
def root660Tail9 : List Nat := 8623 :: root660Tail10
def root660Tail8 : List Nat := 8233 :: root660Tail9
def root660Tail7 : List Nat := 8059 :: root660Tail8
def root660Tail6 : List Nat := 8017 :: root660Tail7
def root660Tail5 : List Nat := 7603 :: root660Tail6
def root660Tail4 : List Nat := 6577 :: root660Tail5
def root660Tail3 : List Nat := 5839 :: root660Tail4
def root660Tail2 : List Nat := 5623 :: root660Tail3
def root660Tail1 : List Nat := 5569 :: root660Tail2
def root660Tail0 : List Nat := 5023 :: root660Tail1
def root660Candidates0 : List Nat := [8059]
set_option maxRecDepth 8192 in
theorem root660Candidates0_checked : candidates edge 4 4957 5023 root660Tail1 winner.bound = root660Candidates0 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root660Step0 : scan edge (search edge 3) 4 [4957] 4957 root660Tail0 winner =
    scan edge (search edge 3) 4 [4957] 4957 root660Tail1 winner := by
  rw [root660Tail0, scan, if_neg (by decide), root660Candidates0_checked]
  change scan edge (search edge 3) 4 [4957] 4957 root660Tail1
    (if 3 ≤ root660Candidates0.length then search edge 3 [4957, 5023] root660Candidates0 9980 winner else winner) = _
  rw [if_neg (by decide)]
theorem root660Scan1 : scan edge (search edge 3) 4 [4957] 4957 root660Tail1 winner = winner := by rw [root660Tail1, scan, if_pos (by decide)]
theorem root660Scan0 : scan edge (search edge 3) 4 [4957] 4957 root660Tail0 winner = winner :=
  (root660Step0).trans root660Scan1
set_option maxRecDepth 4096 in
theorem root660Branch : search edge 4 [4957] root660Candidates 4957 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4957] 4957 root660Tail0 winner = winner
  exact root660Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
