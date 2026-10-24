import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root567Candidates : List Nat := Data.row4139
set_option maxRecDepth 8192 in
theorem root567Candidates_checked : candidates edge 5 0 4139 tail568 winner.bound = root567Candidates := by
  apply candidates_eq_row edge 5 0 4139 tail568 root567Candidates winner.bound tail568_ordered
  · decide
  · decide
  · funext q
    rfl
def root567Tail32 : List Nat := []
def root567Tail31 : List Nat := 21713 :: root567Tail32
def root567Tail30 : List Nat := 20693 :: root567Tail31
def root567Tail29 : List Nat := 18899 :: root567Tail30
def root567Tail28 : List Nat := 17393 :: root567Tail29
def root567Tail27 : List Nat := 16253 :: root567Tail28
def root567Tail26 : List Nat := 16001 :: root567Tail27
def root567Tail25 : List Nat := 15959 :: root567Tail26
def root567Tail24 : List Nat := 15443 :: root567Tail25
def root567Tail23 : List Nat := 15233 :: root567Tail24
def root567Tail22 : List Nat := 13691 :: root567Tail23
def root567Tail21 : List Nat := 13103 :: root567Tail22
def root567Tail20 : List Nat := 12641 :: root567Tail21
def root567Tail19 : List Nat := 11423 :: root567Tail20
def root567Tail18 : List Nat := 11177 :: root567Tail19
def root567Tail17 : List Nat := 11159 :: root567Tail18
def root567Tail16 : List Nat := 10499 :: root567Tail17
def root567Tail15 : List Nat := 9749 :: root567Tail16
def root567Tail14 : List Nat := 8969 :: root567Tail15
def root567Tail13 : List Nat := 8867 :: root567Tail14
def root567Tail12 : List Nat := 8783 :: root567Tail13
def root567Tail11 : List Nat := 8387 :: root567Tail12
def root567Tail10 : List Nat := 8369 :: root567Tail11
def root567Tail9 : List Nat := 7673 :: root567Tail10
def root567Tail8 : List Nat := 7607 :: root567Tail9
def root567Tail7 : List Nat := 7541 :: root567Tail8
def root567Tail6 : List Nat := 7457 :: root567Tail7
def root567Tail5 : List Nat := 7253 :: root567Tail6
def root567Tail4 : List Nat := 6959 :: root567Tail5
def root567Tail3 : List Nat := 6911 :: root567Tail4
def root567Tail2 : List Nat := 6263 :: root567Tail3
def root567Tail1 : List Nat := 5657 :: root567Tail2
def root567Tail0 : List Nat := 4937 :: root567Tail1
def root567Candidates0 : List Nat := [13691]
set_option maxRecDepth 8192 in
theorem root567Candidates0_checked : candidates edge 4 4139 4937 root567Tail1 winner.bound = root567Candidates0 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root567Step0 : scan edge (search edge 3) 4 [4139] 4139 root567Tail0 winner =
    scan edge (search edge 3) 4 [4139] 4139 root567Tail1 winner := by
  rw [root567Tail0, scan, if_neg (by decide), root567Candidates0_checked]
  change scan edge (search edge 3) 4 [4139] 4139 root567Tail1
    (if 3 ≤ root567Candidates0.length then search edge 3 [4139, 4937] root567Candidates0 9076 winner else winner) = _
  rw [if_neg (by decide)]
theorem root567Scan1 : scan edge (search edge 3) 4 [4139] 4139 root567Tail1 winner = winner := by rw [root567Tail1, scan, if_pos (by decide)]
theorem root567Scan0 : scan edge (search edge 3) 4 [4139] 4139 root567Tail0 winner = winner :=
  (root567Step0).trans root567Scan1
set_option maxRecDepth 4096 in
theorem root567Branch : search edge 4 [4139] root567Candidates 4139 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4139] 4139 root567Tail0 winner = winner
  exact root567Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
