import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root644Candidates : List Nat := Data.row4801
set_option maxRecDepth 8192 in
theorem root644Candidates_checked : candidates edge 5 0 4801 tail645 winner.bound = root644Candidates := by
  apply candidates_eq_row edge 5 0 4801 tail645 root644Candidates winner.bound tail645_ordered
  · decide
  · decide
  · funext q
    rfl
def root644Tail29 : List Nat := []
def root644Tail28 : List Nat := 19501 :: root644Tail29
def root644Tail27 : List Nat := 19249 :: root644Tail28
def root644Tail26 : List Nat := 18481 :: root644Tail27
def root644Tail25 : List Nat := 17431 :: root644Tail26
def root644Tail24 : List Nat := 16381 :: root644Tail25
def root644Tail23 : List Nat := 15919 :: root644Tail24
def root644Tail22 : List Nat := 15061 :: root644Tail23
def root644Tail21 : List Nat := 14419 :: root644Tail22
def root644Tail20 : List Nat := 12739 :: root644Tail21
def root644Tail19 : List Nat := 12697 :: root644Tail20
def root644Tail18 : List Nat := 12211 :: root644Tail19
def root644Tail17 : List Nat := 12163 :: root644Tail18
def root644Tail16 : List Nat := 12109 :: root644Tail17
def root644Tail15 : List Nat := 11953 :: root644Tail16
def root644Tail14 : List Nat := 9931 :: root644Tail15
def root644Tail13 : List Nat := 9397 :: root644Tail14
def root644Tail12 : List Nat := 9133 :: root644Tail13
def root644Tail11 : List Nat := 8923 :: root644Tail12
def root644Tail10 : List Nat := 8317 :: root644Tail11
def root644Tail9 : List Nat := 7993 :: root644Tail10
def root644Tail8 : List Nat := 7927 :: root644Tail9
def root644Tail7 : List Nat := 7159 :: root644Tail8
def root644Tail6 : List Nat := 6961 :: root644Tail7
def root644Tail5 : List Nat := 6733 :: root644Tail6
def root644Tail4 : List Nat := 6361 :: root644Tail5
def root644Tail3 : List Nat := 6229 :: root644Tail4
def root644Tail2 : List Nat := 5839 :: root644Tail3
def root644Tail1 : List Nat := 5419 :: root644Tail2
def root644Tail0 : List Nat := 5197 :: root644Tail1
def root644Candidates0 : List Nat := [6733]
set_option maxRecDepth 8192 in
theorem root644Candidates0_checked : candidates edge 4 4801 5197 root644Tail1 winner.bound = root644Candidates0 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root644Step0 : scan edge (search edge 3) 4 [4801] 4801 root644Tail0 winner =
    scan edge (search edge 3) 4 [4801] 4801 root644Tail1 winner := by
  rw [root644Tail0, scan, if_neg (by decide), root644Candidates0_checked]
  change scan edge (search edge 3) 4 [4801] 4801 root644Tail1
    (if 3 ≤ root644Candidates0.length then search edge 3 [4801, 5197] root644Candidates0 9998 winner else winner) = _
  rw [if_neg (by decide)]
theorem root644Scan1 : scan edge (search edge 3) 4 [4801] 4801 root644Tail1 winner = winner := by rw [root644Tail1, scan, if_pos (by decide)]
theorem root644Scan0 : scan edge (search edge 3) 4 [4801] 4801 root644Tail0 winner = winner :=
  (root644Step0).trans root644Scan1
set_option maxRecDepth 4096 in
theorem root644Branch : search edge 4 [4801] root644Candidates 4801 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4801] 4801 root644Tail0 winner = winner
  exact root644Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
