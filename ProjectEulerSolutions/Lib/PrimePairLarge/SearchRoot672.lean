import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root672Candidates : List Nat := Data.row5039
set_option maxRecDepth 8192 in
theorem root672Candidates_checked : candidates edge 5 0 5039 tail673 winner.bound = root672Candidates := by
  apply candidates_eq_row edge 5 0 5039 tail673 root672Candidates winner.bound tail673_ordered
  · decide
  · decide
  · funext q
    rfl
def root672Tail25 : List Nat := []
def root672Tail24 : List Nat := 20201 :: root672Tail25
def root672Tail23 : List Nat := 20177 :: root672Tail24
def root672Tail22 : List Nat := 19421 :: root672Tail23
def root672Tail21 : List Nat := 19037 :: root672Tail22
def root672Tail20 : List Nat := 18521 :: root672Tail21
def root672Tail19 : List Nat := 18341 :: root672Tail20
def root672Tail18 : List Nat := 17627 :: root672Tail19
def root672Tail17 : List Nat := 16661 :: root672Tail18
def root672Tail16 : List Nat := 16001 :: root672Tail17
def root672Tail15 : List Nat := 15761 :: root672Tail16
def root672Tail14 : List Nat := 15233 :: root672Tail15
def root672Tail13 : List Nat := 14897 :: root672Tail14
def root672Tail12 : List Nat := 12107 :: root672Tail13
def root672Tail11 : List Nat := 9539 :: root672Tail12
def root672Tail10 : List Nat := 8867 :: root672Tail11
def root672Tail9 : List Nat := 8111 :: root672Tail10
def root672Tail8 : List Nat := 8093 :: root672Tail9
def root672Tail7 : List Nat := 7727 :: root672Tail8
def root672Tail6 : List Nat := 7547 :: root672Tail7
def root672Tail5 : List Nat := 7001 :: root672Tail6
def root672Tail4 : List Nat := 6329 :: root672Tail5
def root672Tail3 : List Nat := 6317 :: root672Tail4
def root672Tail2 : List Nat := 6299 :: root672Tail3
def root672Tail1 : List Nat := 5669 :: root672Tail2
def root672Tail0 : List Nat := 5153 :: root672Tail1
def root672Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root672Candidates0_checked : candidates edge 4 5039 5153 root672Tail1 winner.bound = root672Candidates0 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root672Step0 : scan edge (search edge 3) 4 [5039] 5039 root672Tail0 winner =
    scan edge (search edge 3) 4 [5039] 5039 root672Tail1 winner := by
  rw [root672Tail0, scan, if_neg (by decide), root672Candidates0_checked]
  change scan edge (search edge 3) 4 [5039] 5039 root672Tail1
    (if 3 ≤ root672Candidates0.length then search edge 3 [5039, 5153] root672Candidates0 10192 winner else winner) = _
  rw [if_neg (by decide)]
theorem root672Scan1 : scan edge (search edge 3) 4 [5039] 5039 root672Tail1 winner = winner := by rw [root672Tail1, scan, if_pos (by decide)]
theorem root672Scan0 : scan edge (search edge 3) 4 [5039] 5039 root672Tail0 winner = winner :=
  (root672Step0).trans root672Scan1
set_option maxRecDepth 4096 in
theorem root672Branch : search edge 4 [5039] root672Candidates 5039 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5039] 5039 root672Tail0 winner = winner
  exact root672Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
