import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root536Candidates : List Nat := Data.row3889
set_option maxRecDepth 8192 in
theorem root536Candidates_checked : candidates edge 5 0 3889 tail537 winner.bound = root536Candidates := by
  apply candidates_eq_row edge 5 0 3889 tail537 root536Candidates winner.bound tail537_ordered
  · decide
  · decide
  · funext q
    rfl
def root536Tail35 : List Nat := []
def root536Tail34 : List Nat := 21751 :: root536Tail35
def root536Tail33 : List Nat := 20011 :: root536Tail34
def root536Tail32 : List Nat := 19819 :: root536Tail33
def root536Tail31 : List Nat := 19813 :: root536Tail32
def root536Tail30 : List Nat := 19603 :: root536Tail31
def root536Tail29 : List Nat := 19309 :: root536Tail30
def root536Tail28 : List Nat := 18553 :: root536Tail29
def root536Tail27 : List Nat := 18253 :: root536Tail28
def root536Tail26 : List Nat := 17827 :: root536Tail27
def root536Tail25 : List Nat := 17377 :: root536Tail26
def root536Tail24 : List Nat := 17257 :: root536Tail25
def root536Tail23 : List Nat := 17047 :: root536Tail24
def root536Tail22 : List Nat := 16069 :: root536Tail23
def root536Tail21 : List Nat := 15859 :: root536Tail22
def root536Tail20 : List Nat := 14947 :: root536Tail21
def root536Tail19 : List Nat := 14887 :: root536Tail20
def root536Tail18 : List Nat := 13807 :: root536Tail19
def root536Tail17 : List Nat := 13729 :: root536Tail18
def root536Tail16 : List Nat := 13597 :: root536Tail17
def root536Tail15 : List Nat := 12973 :: root536Tail16
def root536Tail14 : List Nat := 12889 :: root536Tail15
def root536Tail13 : List Nat := 11503 :: root536Tail14
def root536Tail12 : List Nat := 9781 :: root536Tail13
def root536Tail11 : List Nat := 9439 :: root536Tail12
def root536Tail10 : List Nat := 9103 :: root536Tail11
def root536Tail9 : List Nat := 8677 :: root536Tail10
def root536Tail8 : List Nat := 8647 :: root536Tail9
def root536Tail7 : List Nat := 7927 :: root536Tail8
def root536Tail6 : List Nat := 7477 :: root536Tail7
def root536Tail5 : List Nat := 6763 :: root536Tail6
def root536Tail4 : List Nat := 6619 :: root536Tail5
def root536Tail3 : List Nat := 6007 :: root536Tail4
def root536Tail2 : List Nat := 5821 :: root536Tail3
def root536Tail1 : List Nat := 5749 :: root536Tail2
def root536Tail0 : List Nat := 4021 :: root536Tail1
def root536Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root536Candidates0_checked : candidates edge 4 3889 4021 root536Tail1 winner.bound = root536Candidates0 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root536Step0 : scan edge (search edge 3) 4 [3889] 3889 root536Tail0 winner =
    scan edge (search edge 3) 4 [3889] 3889 root536Tail1 winner := by
  rw [root536Tail0, scan, if_neg (by decide), root536Candidates0_checked]
  change scan edge (search edge 3) 4 [3889] 3889 root536Tail1
    (if 3 ≤ root536Candidates0.length then search edge 3 [3889, 4021] root536Candidates0 7910 winner else winner) = _
  rw [if_neg (by decide)]
theorem root536Scan1 : scan edge (search edge 3) 4 [3889] 3889 root536Tail1 winner = winner := by rw [root536Tail1, scan, if_pos (by decide)]
theorem root536Scan0 : scan edge (search edge 3) 4 [3889] 3889 root536Tail0 winner = winner :=
  (root536Step0).trans root536Scan1
set_option maxRecDepth 4096 in
theorem root536Branch : search edge 4 [3889] root536Candidates 3889 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3889] 3889 root536Tail0 winner = winner
  exact root536Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
