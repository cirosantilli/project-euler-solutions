import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root493Candidates : List Nat := Data.row3541
set_option maxRecDepth 8192 in
theorem root493Candidates_checked : candidates edge 5 0 3541 tail494 winner.bound = root493Candidates := by
  apply candidates_eq_row edge 5 0 3541 tail494 root493Candidates winner.bound tail494_ordered
  · decide
  · decide
  · funext q
    rfl
def root493Tail31 : List Nat := []
def root493Tail30 : List Nat := 22123 :: root493Tail31
def root493Tail29 : List Nat := 21787 :: root493Tail30
def root493Tail28 : List Nat := 20749 :: root493Tail29
def root493Tail27 : List Nat := 20563 :: root493Tail28
def root493Tail26 : List Nat := 20089 :: root493Tail27
def root493Tail25 : List Nat := 20023 :: root493Tail26
def root493Tail24 : List Nat := 19687 :: root493Tail25
def root493Tail23 : List Nat := 18691 :: root493Tail24
def root493Tail22 : List Nat := 18169 :: root493Tail23
def root493Tail21 : List Nat := 17209 :: root493Tail22
def root493Tail20 : List Nat := 14557 :: root493Tail21
def root493Tail19 : List Nat := 13789 :: root493Tail20
def root493Tail18 : List Nat := 13537 :: root493Tail19
def root493Tail17 : List Nat := 13063 :: root493Tail18
def root493Tail16 : List Nat := 12919 :: root493Tail17
def root493Tail15 : List Nat := 12757 :: root493Tail16
def root493Tail14 : List Nat := 11701 :: root493Tail15
def root493Tail13 : List Nat := 10651 :: root493Tail14
def root493Tail12 : List Nat := 10597 :: root493Tail13
def root493Tail11 : List Nat := 10567 :: root493Tail12
def root493Tail10 : List Nat := 10321 :: root493Tail11
def root493Tail9 : List Nat := 10177 :: root493Tail10
def root493Tail8 : List Nat := 9187 :: root493Tail9
def root493Tail7 : List Nat := 9001 :: root493Tail8
def root493Tail6 : List Nat := 8263 :: root493Tail7
def root493Tail5 : List Nat := 8011 :: root493Tail6
def root493Tail4 : List Nat := 7951 :: root493Tail5
def root493Tail3 : List Nat := 7489 :: root493Tail4
def root493Tail2 : List Nat := 7351 :: root493Tail3
def root493Tail1 : List Nat := 7243 :: root493Tail2
def root493Tail0 : List Nat := 4567 :: root493Tail1
def root493Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root493Candidates0_checked : candidates edge 4 3541 4567 root493Tail1 winner.bound = root493Candidates0 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root493Step0 : scan edge (search edge 3) 4 [3541] 3541 root493Tail0 winner =
    scan edge (search edge 3) 4 [3541] 3541 root493Tail1 winner := by
  rw [root493Tail0, scan, if_neg (by decide), root493Candidates0_checked]
  change scan edge (search edge 3) 4 [3541] 3541 root493Tail1
    (if 3 ≤ root493Candidates0.length then search edge 3 [3541, 4567] root493Candidates0 8108 winner else winner) = _
  rw [if_neg (by decide)]
theorem root493Scan1 : scan edge (search edge 3) 4 [3541] 3541 root493Tail1 winner = winner := by rw [root493Tail1, scan, if_pos (by decide)]
theorem root493Scan0 : scan edge (search edge 3) 4 [3541] 3541 root493Tail0 winner = winner :=
  (root493Step0).trans root493Scan1
set_option maxRecDepth 4096 in
theorem root493Branch : search edge 4 [3541] root493Candidates 3541 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3541] 3541 root493Tail0 winner = winner
  exact root493Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
