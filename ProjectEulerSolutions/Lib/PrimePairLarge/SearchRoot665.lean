import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root665Candidates : List Nat := Data.row4993
set_option maxRecDepth 8192 in
theorem root665Candidates_checked : candidates edge 5 0 4993 tail666 winner.bound = root665Candidates := by
  apply candidates_eq_row edge 5 0 4993 tail666 root665Candidates winner.bound tail666_ordered
  · decide
  · decide
  · funext q
    rfl
def root665Tail31 : List Nat := []
def root665Tail30 : List Nat := 20407 :: root665Tail31
def root665Tail29 : List Nat := 19609 :: root665Tail30
def root665Tail28 : List Nat := 18307 :: root665Tail29
def root665Tail27 : List Nat := 18223 :: root665Tail28
def root665Tail26 : List Nat := 18181 :: root665Tail27
def root665Tail25 : List Nat := 17551 :: root665Tail26
def root665Tail24 : List Nat := 17383 :: root665Tail25
def root665Tail23 : List Nat := 16927 :: root665Tail24
def root665Tail22 : List Nat := 15373 :: root665Tail23
def root665Tail21 : List Nat := 15199 :: root665Tail22
def root665Tail20 : List Nat := 14461 :: root665Tail21
def root665Tail19 : List Nat := 13669 :: root665Tail20
def root665Tail18 : List Nat := 13567 :: root665Tail19
def root665Tail17 : List Nat := 13183 :: root665Tail18
def root665Tail16 : List Nat := 12613 :: root665Tail17
def root665Tail15 : List Nat := 12487 :: root665Tail16
def root665Tail14 : List Nat := 10369 :: root665Tail15
def root665Tail13 : List Nat := 9661 :: root665Tail14
def root665Tail12 : List Nat := 9511 :: root665Tail13
def root665Tail11 : List Nat := 9181 :: root665Tail12
def root665Tail10 : List Nat := 9133 :: root665Tail11
def root665Tail9 : List Nat := 8971 :: root665Tail10
def root665Tail8 : List Nat := 8887 :: root665Tail9
def root665Tail7 : List Nat := 8563 :: root665Tail8
def root665Tail6 : List Nat := 8521 :: root665Tail7
def root665Tail5 : List Nat := 8269 :: root665Tail6
def root665Tail4 : List Nat := 7561 :: root665Tail5
def root665Tail3 : List Nat := 6571 :: root665Tail4
def root665Tail2 : List Nat := 5779 :: root665Tail3
def root665Tail1 : List Nat := 5527 :: root665Tail2
def root665Tail0 : List Nat := 4999 :: root665Tail1
def root665Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root665Candidates0_checked : candidates edge 4 4993 4999 root665Tail1 winner.bound = root665Candidates0 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root665Step0 : scan edge (search edge 3) 4 [4993] 4993 root665Tail0 winner =
    scan edge (search edge 3) 4 [4993] 4993 root665Tail1 winner := by
  rw [root665Tail0, scan, if_neg (by decide), root665Candidates0_checked]
  change scan edge (search edge 3) 4 [4993] 4993 root665Tail1
    (if 3 ≤ root665Candidates0.length then search edge 3 [4993, 4999] root665Candidates0 9992 winner else winner) = _
  rw [if_neg (by decide)]
theorem root665Scan1 : scan edge (search edge 3) 4 [4993] 4993 root665Tail1 winner = winner := by rw [root665Tail1, scan, if_pos (by decide)]
theorem root665Scan0 : scan edge (search edge 3) 4 [4993] 4993 root665Tail0 winner = winner :=
  (root665Step0).trans root665Scan1
set_option maxRecDepth 4096 in
theorem root665Branch : search edge 4 [4993] root665Candidates 4993 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4993] 4993 root665Tail0 winner = winner
  exact root665Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
