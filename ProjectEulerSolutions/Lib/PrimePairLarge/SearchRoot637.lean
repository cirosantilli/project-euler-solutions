import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root637Candidates : List Nat := Data.row4751
set_option maxRecDepth 8192 in
theorem root637Candidates_checked : candidates edge 5 0 4751 tail638 winner.bound = root637Candidates := by
  apply candidates_eq_row edge 5 0 4751 tail638 root637Candidates winner.bound tail638_ordered
  · decide
  · decide
  · funext q
    rfl
def root637Tail29 : List Nat := []
def root637Tail28 : List Nat := 20939 :: root637Tail29
def root637Tail27 : List Nat := 18593 :: root637Tail28
def root637Tail26 : List Nat := 18257 :: root637Tail27
def root637Tail25 : List Nat := 18233 :: root637Tail26
def root637Tail24 : List Nat := 17789 :: root637Tail25
def root637Tail23 : List Nat := 17519 :: root637Tail24
def root637Tail22 : List Nat := 15737 :: root637Tail23
def root637Tail21 : List Nat := 15581 :: root637Tail22
def root637Tail20 : List Nat := 14951 :: root637Tail21
def root637Tail19 : List Nat := 14033 :: root637Tail20
def root637Tail18 : List Nat := 13901 :: root637Tail19
def root637Tail17 : List Nat := 13763 :: root637Tail18
def root637Tail16 : List Nat := 13217 :: root637Tail17
def root637Tail15 : List Nat := 13007 :: root637Tail16
def root637Tail14 : List Nat := 10631 :: root637Tail15
def root637Tail13 : List Nat := 9929 :: root637Tail14
def root637Tail12 : List Nat := 8963 :: root637Tail13
def root637Tail11 : List Nat := 8837 :: root637Tail12
def root637Tail10 : List Nat := 8609 :: root637Tail11
def root637Tail9 : List Nat := 7649 :: root637Tail10
def root637Tail8 : List Nat := 7607 :: root637Tail9
def root637Tail7 : List Nat := 7331 :: root637Tail8
def root637Tail6 : List Nat := 6659 :: root637Tail7
def root637Tail5 : List Nat := 6317 :: root637Tail6
def root637Tail4 : List Nat := 6113 :: root637Tail5
def root637Tail3 : List Nat := 5879 :: root637Tail4
def root637Tail2 : List Nat := 5711 :: root637Tail3
def root637Tail1 : List Nat := 5297 :: root637Tail2
def root637Tail0 : List Nat := 5099 :: root637Tail1
def root637Candidates0 : List Nat := [7649]
set_option maxRecDepth 8192 in
theorem root637Candidates0_checked : candidates edge 4 4751 5099 root637Tail1 winner.bound = root637Candidates0 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root637Step0 : scan edge (search edge 3) 4 [4751] 4751 root637Tail0 winner =
    scan edge (search edge 3) 4 [4751] 4751 root637Tail1 winner := by
  rw [root637Tail0, scan, if_neg (by decide), root637Candidates0_checked]
  change scan edge (search edge 3) 4 [4751] 4751 root637Tail1
    (if 3 ≤ root637Candidates0.length then search edge 3 [4751, 5099] root637Candidates0 9850 winner else winner) = _
  rw [if_neg (by decide)]
def root637Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root637Candidates1_checked : candidates edge 4 4751 5297 root637Tail2 winner.bound = root637Candidates1 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root637Step1 : scan edge (search edge 3) 4 [4751] 4751 root637Tail1 winner =
    scan edge (search edge 3) 4 [4751] 4751 root637Tail2 winner := by
  rw [root637Tail1, scan, if_neg (by decide), root637Candidates1_checked]
  change scan edge (search edge 3) 4 [4751] 4751 root637Tail2
    (if 3 ≤ root637Candidates1.length then search edge 3 [4751, 5297] root637Candidates1 10048 winner else winner) = _
  rw [if_neg (by decide)]
theorem root637Scan2 : scan edge (search edge 3) 4 [4751] 4751 root637Tail2 winner = winner := by rw [root637Tail2, scan, if_pos (by decide)]
theorem root637Scan1 : scan edge (search edge 3) 4 [4751] 4751 root637Tail1 winner = winner :=
  (root637Step1).trans root637Scan2
theorem root637Scan0 : scan edge (search edge 3) 4 [4751] 4751 root637Tail0 winner = winner :=
  (root637Step0).trans root637Scan1
set_option maxRecDepth 4096 in
theorem root637Branch : search edge 4 [4751] root637Candidates 4751 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4751] 4751 root637Tail0 winner = winner
  exact root637Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
