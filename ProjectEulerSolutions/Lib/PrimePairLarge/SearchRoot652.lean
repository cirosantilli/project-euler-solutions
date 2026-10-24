import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root652Candidates : List Nat := Data.row4903
set_option maxRecDepth 8192 in
theorem root652Candidates_checked : candidates edge 5 0 4903 tail653 winner.bound = root652Candidates := by
  apply candidates_eq_row edge 5 0 4903 tail653 root652Candidates winner.bound tail653_ordered
  · decide
  · decide
  · funext q
    rfl
def root652Tail29 : List Nat := []
def root652Tail28 : List Nat := 20731 :: root652Tail29
def root652Tail27 : List Nat := 20359 :: root652Tail28
def root652Tail26 : List Nat := 17341 :: root652Tail27
def root652Tail25 : List Nat := 16741 :: root652Tail26
def root652Tail24 : List Nat := 16249 :: root652Tail25
def root652Tail23 : List Nat := 15493 :: root652Tail24
def root652Tail22 : List Nat := 15391 :: root652Tail23
def root652Tail21 : List Nat := 14887 :: root652Tail22
def root652Tail20 : List Nat := 13627 :: root652Tail21
def root652Tail19 : List Nat := 13597 :: root652Tail20
def root652Tail18 : List Nat := 12379 :: root652Tail19
def root652Tail17 : List Nat := 12211 :: root652Tail18
def root652Tail16 : List Nat := 12073 :: root652Tail17
def root652Tail15 : List Nat := 11287 :: root652Tail16
def root652Tail14 : List Nat := 10531 :: root652Tail15
def root652Tail13 : List Nat := 9811 :: root652Tail14
def root652Tail12 : List Nat := 9643 :: root652Tail13
def root652Tail11 : List Nat := 9349 :: root652Tail12
def root652Tail10 : List Nat := 9337 :: root652Tail11
def root652Tail9 : List Nat := 9007 :: root652Tail10
def root652Tail8 : List Nat := 8893 :: root652Tail9
def root652Tail7 : List Nat := 8839 :: root652Tail8
def root652Tail6 : List Nat := 8389 :: root652Tail7
def root652Tail5 : List Nat := 7789 :: root652Tail6
def root652Tail4 : List Nat := 6991 :: root652Tail5
def root652Tail3 : List Nat := 6271 :: root652Tail4
def root652Tail2 : List Nat := 5683 :: root652Tail3
def root652Tail1 : List Nat := 5659 :: root652Tail2
def root652Tail0 : List Nat := 4933 :: root652Tail1
def root652Candidates0 : List Nat := [6271, 8893]
set_option maxRecDepth 8192 in
theorem root652Candidates0_checked : candidates edge 4 4903 4933 root652Tail1 winner.bound = root652Candidates0 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root652Step0 : scan edge (search edge 3) 4 [4903] 4903 root652Tail0 winner =
    scan edge (search edge 3) 4 [4903] 4903 root652Tail1 winner := by
  rw [root652Tail0, scan, if_neg (by decide), root652Candidates0_checked]
  change scan edge (search edge 3) 4 [4903] 4903 root652Tail1
    (if 3 ≤ root652Candidates0.length then search edge 3 [4903, 4933] root652Candidates0 9836 winner else winner) = _
  rw [if_neg (by decide)]
theorem root652Scan1 : scan edge (search edge 3) 4 [4903] 4903 root652Tail1 winner = winner := by rw [root652Tail1, scan, if_pos (by decide)]
theorem root652Scan0 : scan edge (search edge 3) 4 [4903] 4903 root652Tail0 winner = winner :=
  (root652Step0).trans root652Scan1
set_option maxRecDepth 4096 in
theorem root652Branch : search edge 4 [4903] root652Candidates 4903 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4903] 4903 root652Tail0 winner = winner
  exact root652Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
