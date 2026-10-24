import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root621Candidates : List Nat := Data.row4621
set_option maxRecDepth 8192 in
theorem root621Candidates_checked : candidates edge 5 0 4621 tail622 winner.bound = root621Candidates := by
  apply candidates_eq_row edge 5 0 4621 tail622 root621Candidates winner.bound tail622_ordered
  · decide
  · decide
  · funext q
    rfl
def root621Tail29 : List Nat := []
def root621Tail28 : List Nat := 21169 :: root621Tail29
def root621Tail27 : List Nat := 20773 :: root621Tail28
def root621Tail26 : List Nat := 20359 :: root621Tail27
def root621Tail25 : List Nat := 20287 :: root621Tail26
def root621Tail24 : List Nat := 20089 :: root621Tail25
def root621Tail23 : List Nat := 19531 :: root621Tail24
def root621Tail22 : List Nat := 18661 :: root621Tail23
def root621Tail21 : List Nat := 17989 :: root621Tail22
def root621Tail20 : List Nat := 17419 :: root621Tail21
def root621Tail19 : List Nat := 17377 :: root621Tail20
def root621Tail18 : List Nat := 16033 :: root621Tail19
def root621Tail17 : List Nat := 15889 :: root621Tail18
def root621Tail16 : List Nat := 15733 :: root621Tail17
def root621Tail15 : List Nat := 15583 :: root621Tail16
def root621Tail14 : List Nat := 14197 :: root621Tail15
def root621Tail13 : List Nat := 12421 :: root621Tail14
def root621Tail12 : List Nat := 11257 :: root621Tail13
def root621Tail11 : List Nat := 11161 :: root621Tail12
def root621Tail10 : List Nat := 10651 :: root621Tail11
def root621Tail9 : List Nat := 9403 :: root621Tail10
def root621Tail8 : List Nat := 9337 :: root621Tail9
def root621Tail7 : List Nat := 8287 :: root621Tail8
def root621Tail6 : List Nat := 7573 :: root621Tail7
def root621Tail5 : List Nat := 7489 :: root621Tail6
def root621Tail4 : List Nat := 6679 :: root621Tail5
def root621Tail3 : List Nat := 6673 :: root621Tail4
def root621Tail2 : List Nat := 5779 :: root621Tail3
def root621Tail1 : List Nat := 5641 :: root621Tail2
def root621Tail0 : List Nat := 5347 :: root621Tail1
def root621Candidates0 : List Nat := [8287]
set_option maxRecDepth 8192 in
theorem root621Candidates0_checked : candidates edge 4 4621 5347 root621Tail1 winner.bound = root621Candidates0 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root621Step0 : scan edge (search edge 3) 4 [4621] 4621 root621Tail0 winner =
    scan edge (search edge 3) 4 [4621] 4621 root621Tail1 winner := by
  rw [root621Tail0, scan, if_neg (by decide), root621Candidates0_checked]
  change scan edge (search edge 3) 4 [4621] 4621 root621Tail1
    (if 3 ≤ root621Candidates0.length then search edge 3 [4621, 5347] root621Candidates0 9968 winner else winner) = _
  rw [if_neg (by decide)]
theorem root621Scan1 : scan edge (search edge 3) 4 [4621] 4621 root621Tail1 winner = winner := by rw [root621Tail1, scan, if_pos (by decide)]
theorem root621Scan0 : scan edge (search edge 3) 4 [4621] 4621 root621Tail0 winner = winner :=
  (root621Step0).trans root621Scan1
set_option maxRecDepth 4096 in
theorem root621Branch : search edge 4 [4621] root621Candidates 4621 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4621] 4621 root621Tail0 winner = winner
  exact root621Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
