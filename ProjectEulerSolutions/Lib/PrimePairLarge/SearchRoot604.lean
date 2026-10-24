import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root604Candidates : List Nat := Data.row4463
set_option maxRecDepth 8192 in
theorem root604Candidates_checked : candidates edge 5 0 4463 tail605 winner.bound = root604Candidates := by
  apply candidates_eq_row edge 5 0 4463 tail605 root604Candidates winner.bound tail605_ordered
  · decide
  · decide
  · funext q
    rfl
def root604Tail34 : List Nat := []
def root604Tail33 : List Nat := 20477 :: root604Tail34
def root604Tail32 : List Nat := 19157 :: root604Tail33
def root604Tail31 : List Nat := 18947 :: root604Tail32
def root604Tail30 : List Nat := 18839 :: root604Tail31
def root604Tail29 : List Nat := 18701 :: root604Tail30
def root604Tail28 : List Nat := 18077 :: root604Tail29
def root604Tail27 : List Nat := 17957 :: root604Tail28
def root604Tail26 : List Nat := 17483 :: root604Tail27
def root604Tail25 : List Nat := 16319 :: root604Tail26
def root604Tail24 : List Nat := 15383 :: root604Tail25
def root604Tail23 : List Nat := 14411 :: root604Tail24
def root604Tail22 : List Nat := 13613 :: root604Tail23
def root604Tail21 : List Nat := 13421 :: root604Tail22
def root604Tail20 : List Nat := 12491 :: root604Tail21
def root604Tail19 : List Nat := 12281 :: root604Tail20
def root604Tail18 : List Nat := 12143 :: root604Tail19
def root604Tail17 : List Nat := 12107 :: root604Tail18
def root604Tail16 : List Nat := 11933 :: root604Tail17
def root604Tail15 : List Nat := 10859 :: root604Tail16
def root604Tail14 : List Nat := 10301 :: root604Tail15
def root604Tail13 : List Nat := 9851 :: root604Tail14
def root604Tail12 : List Nat := 9587 :: root604Tail13
def root604Tail11 : List Nat := 8963 :: root604Tail12
def root604Tail10 : List Nat := 8387 :: root604Tail11
def root604Tail9 : List Nat := 8369 :: root604Tail10
def root604Tail8 : List Nat := 7877 :: root604Tail9
def root604Tail7 : List Nat := 7079 :: root604Tail8
def root604Tail6 : List Nat := 6791 :: root604Tail7
def root604Tail5 : List Nat := 6521 :: root604Tail6
def root604Tail4 : List Nat := 5861 :: root604Tail5
def root604Tail3 : List Nat := 5813 :: root604Tail4
def root604Tail2 : List Nat := 5441 :: root604Tail3
def root604Tail1 : List Nat := 5399 :: root604Tail2
def root604Tail0 : List Nat := 5273 :: root604Tail1
def root604Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root604Candidates0_checked : candidates edge 4 4463 5273 root604Tail1 winner.bound = root604Candidates0 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root604Step0 : scan edge (search edge 3) 4 [4463] 4463 root604Tail0 winner =
    scan edge (search edge 3) 4 [4463] 4463 root604Tail1 winner := by
  rw [root604Tail0, scan, if_neg (by decide), root604Candidates0_checked]
  change scan edge (search edge 3) 4 [4463] 4463 root604Tail1
    (if 3 ≤ root604Candidates0.length then search edge 3 [4463, 5273] root604Candidates0 9736 winner else winner) = _
  rw [if_neg (by decide)]
theorem root604Scan1 : scan edge (search edge 3) 4 [4463] 4463 root604Tail1 winner = winner := by rw [root604Tail1, scan, if_pos (by decide)]
theorem root604Scan0 : scan edge (search edge 3) 4 [4463] 4463 root604Tail0 winner = winner :=
  (root604Step0).trans root604Scan1
set_option maxRecDepth 4096 in
theorem root604Branch : search edge 4 [4463] root604Candidates 4463 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4463] 4463 root604Tail0 winner = winner
  exact root604Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
