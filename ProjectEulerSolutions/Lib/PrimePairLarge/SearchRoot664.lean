import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root664Candidates : List Nat := Data.row4987
set_option maxRecDepth 8192 in
theorem root664Candidates_checked : candidates edge 5 0 4987 tail665 winner.bound = root664Candidates := by
  apply candidates_eq_row edge 5 0 4987 tail665 root664Candidates winner.bound tail665_ordered
  · decide
  · decide
  · funext q
    rfl
def root664Tail20 : List Nat := []
def root664Tail19 : List Nat := 20899 :: root664Tail20
def root664Tail18 : List Nat := 18541 :: root664Tail19
def root664Tail17 : List Nat := 18217 :: root664Tail18
def root664Tail16 : List Nat := 18097 :: root664Tail17
def root664Tail15 : List Nat := 18043 :: root664Tail16
def root664Tail14 : List Nat := 16453 :: root664Tail15
def root664Tail13 : List Nat := 16033 :: root664Tail14
def root664Tail12 : List Nat := 15991 :: root664Tail13
def root664Tail11 : List Nat := 14947 :: root664Tail12
def root664Tail10 : List Nat := 14149 :: root664Tail11
def root664Tail9 : List Nat := 13759 :: root664Tail10
def root664Tail8 : List Nat := 12829 :: root664Tail9
def root664Tail7 : List Nat := 12763 :: root664Tail8
def root664Tail6 : List Nat := 12619 :: root664Tail7
def root664Tail5 : List Nat := 12583 :: root664Tail6
def root664Tail4 : List Nat := 11497 :: root664Tail5
def root664Tail3 : List Nat := 8641 :: root664Tail4
def root664Tail2 : List Nat := 8461 :: root664Tail3
def root664Tail1 : List Nat := 6991 :: root664Tail2
def root664Tail0 : List Nat := 5197 :: root664Tail1
def root664Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root664Candidates0_checked : candidates edge 4 4987 5197 root664Tail1 winner.bound = root664Candidates0 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root664Step0 : scan edge (search edge 3) 4 [4987] 4987 root664Tail0 winner =
    scan edge (search edge 3) 4 [4987] 4987 root664Tail1 winner := by
  rw [root664Tail0, scan, if_neg (by decide), root664Candidates0_checked]
  change scan edge (search edge 3) 4 [4987] 4987 root664Tail1
    (if 3 ≤ root664Candidates0.length then search edge 3 [4987, 5197] root664Candidates0 10184 winner else winner) = _
  rw [if_neg (by decide)]
theorem root664Scan1 : scan edge (search edge 3) 4 [4987] 4987 root664Tail1 winner = winner := by rw [root664Tail1, scan, if_pos (by decide)]
theorem root664Scan0 : scan edge (search edge 3) 4 [4987] 4987 root664Tail0 winner = winner :=
  (root664Step0).trans root664Scan1
set_option maxRecDepth 4096 in
theorem root664Branch : search edge 4 [4987] root664Candidates 4987 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4987] 4987 root664Tail0 winner = winner
  exact root664Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
