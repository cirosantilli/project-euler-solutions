import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root659Candidates : List Nat := Data.row4951
set_option maxRecDepth 8192 in
theorem root659Candidates_checked : candidates edge 5 0 4951 tail660 winner.bound = root659Candidates := by
  apply candidates_eq_row edge 5 0 4951 tail660 root659Candidates winner.bound tail660_ordered
  · decide
  · decide
  · funext q
    rfl
def root659Tail25 : List Nat := []
def root659Tail24 : List Nat := 18451 :: root659Tail25
def root659Tail23 : List Nat := 18229 :: root659Tail24
def root659Tail22 : List Nat := 16273 :: root659Tail23
def root659Tail21 : List Nat := 16267 :: root659Tail22
def root659Tail20 : List Nat := 15349 :: root659Tail21
def root659Tail19 : List Nat := 15307 :: root659Tail20
def root659Tail18 : List Nat := 15031 :: root659Tail19
def root659Tail17 : List Nat := 14029 :: root659Tail18
def root659Tail16 : List Nat := 13963 :: root659Tail17
def root659Tail15 : List Nat := 11779 :: root659Tail16
def root659Tail14 : List Nat := 11311 :: root659Tail15
def root659Tail13 : List Nat := 10723 :: root659Tail14
def root659Tail12 : List Nat := 10453 :: root659Tail13
def root659Tail11 : List Nat := 9697 :: root659Tail12
def root659Tail10 : List Nat := 9601 :: root659Tail11
def root659Tail9 : List Nat := 9241 :: root659Tail10
def root659Tail8 : List Nat := 9049 :: root659Tail9
def root659Tail7 : List Nat := 8779 :: root659Tail8
def root659Tail6 : List Nat := 8719 :: root659Tail7
def root659Tail5 : List Nat := 7537 :: root659Tail6
def root659Tail4 : List Nat := 6781 :: root659Tail5
def root659Tail3 : List Nat := 6343 :: root659Tail4
def root659Tail2 : List Nat := 5821 :: root659Tail3
def root659Tail1 : List Nat := 5563 :: root659Tail2
def root659Tail0 : List Nat := 5107 :: root659Tail1
def root659Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root659Candidates0_checked : candidates edge 4 4951 5107 root659Tail1 winner.bound = root659Candidates0 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root659Step0 : scan edge (search edge 3) 4 [4951] 4951 root659Tail0 winner =
    scan edge (search edge 3) 4 [4951] 4951 root659Tail1 winner := by
  rw [root659Tail0, scan, if_neg (by decide), root659Candidates0_checked]
  change scan edge (search edge 3) 4 [4951] 4951 root659Tail1
    (if 3 ≤ root659Candidates0.length then search edge 3 [4951, 5107] root659Candidates0 10058 winner else winner) = _
  rw [if_neg (by decide)]
theorem root659Scan1 : scan edge (search edge 3) 4 [4951] 4951 root659Tail1 winner = winner := by rw [root659Tail1, scan, if_pos (by decide)]
theorem root659Scan0 : scan edge (search edge 3) 4 [4951] 4951 root659Tail0 winner = winner :=
  (root659Step0).trans root659Scan1
set_option maxRecDepth 4096 in
theorem root659Branch : search edge 4 [4951] root659Candidates 4951 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4951] 4951 root659Tail0 winner = winner
  exact root659Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
