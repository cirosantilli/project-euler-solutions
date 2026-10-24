import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root623Candidates : List Nat := Data.row4639
set_option maxRecDepth 8192 in
theorem root623Candidates_checked : candidates edge 5 0 4639 tail624 winner.bound = root623Candidates := by
  apply candidates_eq_row edge 5 0 4639 tail624 root623Candidates winner.bound tail624_ordered
  · decide
  · decide
  · funext q
    rfl
def root623Tail31 : List Nat := []
def root623Tail30 : List Nat := 20161 :: root623Tail31
def root623Tail29 : List Nat := 20113 :: root623Tail30
def root623Tail28 : List Nat := 20029 :: root623Tail29
def root623Tail27 : List Nat := 19861 :: root623Tail28
def root623Tail26 : List Nat := 19501 :: root623Tail27
def root623Tail25 : List Nat := 19483 :: root623Tail26
def root623Tail24 : List Nat := 19381 :: root623Tail25
def root623Tail23 : List Nat := 18919 :: root623Tail24
def root623Tail22 : List Nat := 17599 :: root623Tail23
def root623Tail21 : List Nat := 17551 :: root623Tail22
def root623Tail20 : List Nat := 17431 :: root623Tail21
def root623Tail19 : List Nat := 17257 :: root623Tail20
def root623Tail18 : List Nat := 16477 :: root623Tail19
def root623Tail17 : List Nat := 16183 :: root623Tail18
def root623Tail16 : List Nat := 15679 :: root623Tail17
def root623Tail15 : List Nat := 14797 :: root623Tail16
def root623Tail14 : List Nat := 13999 :: root623Tail15
def root623Tail13 : List Nat := 11593 :: root623Tail14
def root623Tail12 : List Nat := 9973 :: root623Tail13
def root623Tail11 : List Nat := 9679 :: root623Tail12
def root623Tail10 : List Nat := 9277 :: root623Tail11
def root623Tail9 : List Nat := 8563 :: root623Tail10
def root623Tail8 : List Nat := 7237 :: root623Tail9
def root623Tail7 : List Nat := 7177 :: root623Tail8
def root623Tail6 : List Nat := 6709 :: root623Tail7
def root623Tail5 : List Nat := 6373 :: root623Tail6
def root623Tail4 : List Nat := 6211 :: root623Tail5
def root623Tail3 : List Nat := 6121 :: root623Tail4
def root623Tail2 : List Nat := 5827 :: root623Tail3
def root623Tail1 : List Nat := 5749 :: root623Tail2
def root623Tail0 : List Nat := 4951 :: root623Tail1
def root623Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root623Candidates0_checked : candidates edge 4 4639 4951 root623Tail1 winner.bound = root623Candidates0 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root623Step0 : scan edge (search edge 3) 4 [4639] 4639 root623Tail0 winner =
    scan edge (search edge 3) 4 [4639] 4639 root623Tail1 winner := by
  rw [root623Tail0, scan, if_neg (by decide), root623Candidates0_checked]
  change scan edge (search edge 3) 4 [4639] 4639 root623Tail1
    (if 3 ≤ root623Candidates0.length then search edge 3 [4639, 4951] root623Candidates0 9590 winner else winner) = _
  rw [if_neg (by decide)]
theorem root623Scan1 : scan edge (search edge 3) 4 [4639] 4639 root623Tail1 winner = winner := by rw [root623Tail1, scan, if_pos (by decide)]
theorem root623Scan0 : scan edge (search edge 3) 4 [4639] 4639 root623Tail0 winner = winner :=
  (root623Step0).trans root623Scan1
set_option maxRecDepth 4096 in
theorem root623Branch : search edge 4 [4639] root623Candidates 4639 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4639] 4639 root623Tail0 winner = winner
  exact root623Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
