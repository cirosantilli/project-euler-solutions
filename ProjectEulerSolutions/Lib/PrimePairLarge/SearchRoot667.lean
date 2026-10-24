import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root667Candidates : List Nat := Data.row5003
set_option maxRecDepth 8192 in
theorem root667Candidates_checked : candidates edge 5 0 5003 tail668 winner.bound = root667Candidates := by
  apply candidates_eq_row edge 5 0 5003 tail668 root667Candidates winner.bound tail668_ordered
  · decide
  · decide
  · funext q
    rfl
def root667Tail28 : List Nat := []
def root667Tail27 : List Nat := 18593 :: root667Tail28
def root667Tail26 : List Nat := 18233 :: root667Tail27
def root667Tail25 : List Nat := 18047 :: root667Tail26
def root667Tail24 : List Nat := 17627 :: root667Tail25
def root667Tail23 : List Nat := 17387 :: root667Tail24
def root667Tail22 : List Nat := 14939 :: root667Tail23
def root667Tail21 : List Nat := 14723 :: root667Tail22
def root667Tail20 : List Nat := 14519 :: root667Tail21
def root667Tail19 : List Nat := 14447 :: root667Tail20
def root667Tail18 : List Nat := 14051 :: root667Tail19
def root667Tail17 : List Nat := 13523 :: root667Tail18
def root667Tail16 : List Nat := 13049 :: root667Tail17
def root667Tail15 : List Nat := 12983 :: root667Tail16
def root667Tail14 : List Nat := 12809 :: root667Tail15
def root667Tail13 : List Nat := 11423 :: root667Tail14
def root667Tail12 : List Nat := 10589 :: root667Tail13
def root667Tail11 : List Nat := 9029 :: root667Tail12
def root667Tail10 : List Nat := 8423 :: root667Tail11
def root667Tail9 : List Nat := 8147 :: root667Tail10
def root667Tail8 : List Nat := 7703 :: root667Tail9
def root667Tail7 : List Nat := 7541 :: root667Tail8
def root667Tail6 : List Nat := 7349 :: root667Tail7
def root667Tail5 : List Nat := 6911 :: root667Tail6
def root667Tail4 : List Nat := 6551 :: root667Tail5
def root667Tail3 : List Nat := 6197 :: root667Tail4
def root667Tail2 : List Nat := 6131 :: root667Tail3
def root667Tail1 : List Nat := 5939 :: root667Tail2
def root667Tail0 : List Nat := 5087 :: root667Tail1
def root667Candidates0 : List Nat := [8147]
set_option maxRecDepth 8192 in
theorem root667Candidates0_checked : candidates edge 4 5003 5087 root667Tail1 winner.bound = root667Candidates0 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root667Step0 : scan edge (search edge 3) 4 [5003] 5003 root667Tail0 winner =
    scan edge (search edge 3) 4 [5003] 5003 root667Tail1 winner := by
  rw [root667Tail0, scan, if_neg (by decide), root667Candidates0_checked]
  change scan edge (search edge 3) 4 [5003] 5003 root667Tail1
    (if 3 ≤ root667Candidates0.length then search edge 3 [5003, 5087] root667Candidates0 10090 winner else winner) = _
  rw [if_neg (by decide)]
theorem root667Scan1 : scan edge (search edge 3) 4 [5003] 5003 root667Tail1 winner = winner := by rw [root667Tail1, scan, if_pos (by decide)]
theorem root667Scan0 : scan edge (search edge 3) 4 [5003] 5003 root667Tail0 winner = winner :=
  (root667Step0).trans root667Scan1
set_option maxRecDepth 4096 in
theorem root667Branch : search edge 4 [5003] root667Candidates 5003 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5003] 5003 root667Tail0 winner = winner
  exact root667Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
