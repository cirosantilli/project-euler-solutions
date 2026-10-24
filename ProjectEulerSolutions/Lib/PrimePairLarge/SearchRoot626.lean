import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root626Candidates : List Nat := Data.row4651
set_option maxRecDepth 8192 in
theorem root626Candidates_checked : candidates edge 5 0 4651 tail627 winner.bound = root626Candidates := by
  apply candidates_eq_row edge 5 0 4651 tail627 root626Candidates winner.bound tail627_ordered
  · decide
  · decide
  · funext q
    rfl
def root626Tail22 : List Nat := []
def root626Tail21 : List Nat := 20611 :: root626Tail22
def root626Tail20 : List Nat := 20521 :: root626Tail21
def root626Tail19 : List Nat := 20431 :: root626Tail20
def root626Tail18 : List Nat := 20023 :: root626Tail19
def root626Tail17 : List Nat := 18541 :: root626Tail18
def root626Tail16 : List Nat := 17761 :: root626Tail17
def root626Tail15 : List Nat := 17209 :: root626Tail16
def root626Tail14 : List Nat := 16063 :: root626Tail15
def root626Tail13 : List Nat := 15643 :: root626Tail14
def root626Tail12 : List Nat := 14437 :: root626Tail13
def root626Tail11 : List Nat := 14347 :: root626Tail12
def root626Tail10 : List Nat := 13921 :: root626Tail11
def root626Tail9 : List Nat := 12763 :: root626Tail10
def root626Tail8 : List Nat := 11527 :: root626Tail9
def root626Tail7 : List Nat := 10663 :: root626Tail8
def root626Tail6 : List Nat := 9631 :: root626Tail7
def root626Tail5 : List Nat := 9349 :: root626Tail6
def root626Tail4 : List Nat := 9091 :: root626Tail5
def root626Tail3 : List Nat := 8293 :: root626Tail4
def root626Tail2 : List Nat := 7243 :: root626Tail3
def root626Tail1 : List Nat := 6991 :: root626Tail2
def root626Tail0 : List Nat := 4723 :: root626Tail1
def root626Candidates0 : List Nat := [7243]
set_option maxRecDepth 8192 in
theorem root626Candidates0_checked : candidates edge 4 4651 4723 root626Tail1 winner.bound = root626Candidates0 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root626Step0 : scan edge (search edge 3) 4 [4651] 4651 root626Tail0 winner =
    scan edge (search edge 3) 4 [4651] 4651 root626Tail1 winner := by
  rw [root626Tail0, scan, if_neg (by decide), root626Candidates0_checked]
  change scan edge (search edge 3) 4 [4651] 4651 root626Tail1
    (if 3 ≤ root626Candidates0.length then search edge 3 [4651, 4723] root626Candidates0 9374 winner else winner) = _
  rw [if_neg (by decide)]
theorem root626Scan1 : scan edge (search edge 3) 4 [4651] 4651 root626Tail1 winner = winner := by rw [root626Tail1, scan, if_pos (by decide)]
theorem root626Scan0 : scan edge (search edge 3) 4 [4651] 4651 root626Tail0 winner = winner :=
  (root626Step0).trans root626Scan1
set_option maxRecDepth 4096 in
theorem root626Branch : search edge 4 [4651] root626Candidates 4651 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4651] 4651 root626Tail0 winner = winner
  exact root626Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
