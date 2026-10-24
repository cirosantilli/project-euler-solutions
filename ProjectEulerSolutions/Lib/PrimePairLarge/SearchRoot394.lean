import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root394Candidates : List Nat := Data.row2719
set_option maxRecDepth 8192 in
theorem root394Candidates_checked : candidates edge 5 0 2719 tail395 winner.bound = root394Candidates := by
  apply candidates_eq_row edge 5 0 2719 tail395 root394Candidates winner.bound tail395_ordered
  · decide
  · decide
  · funext q
    rfl
def root394Tail31 : List Nat := []
def root394Tail30 : List Nat := 23203 :: root394Tail31
def root394Tail29 : List Nat := 22741 :: root394Tail30
def root394Tail28 : List Nat := 21961 :: root394Tail29
def root394Tail27 : List Nat := 19603 :: root394Tail28
def root394Tail26 : List Nat := 18979 :: root394Tail27
def root394Tail25 : List Nat := 17749 :: root394Tail26
def root394Tail24 : List Nat := 17713 :: root394Tail25
def root394Tail23 : List Nat := 17299 :: root394Tail24
def root394Tail22 : List Nat := 16987 :: root394Tail23
def root394Tail21 : List Nat := 16369 :: root394Tail22
def root394Tail20 : List Nat := 16273 :: root394Tail21
def root394Tail19 : List Nat := 15349 :: root394Tail20
def root394Tail18 : List Nat := 15199 :: root394Tail19
def root394Tail17 : List Nat := 15031 :: root394Tail18
def root394Tail16 : List Nat := 14563 :: root394Tail17
def root394Tail15 : List Nat := 14479 :: root394Tail16
def root394Tail14 : List Nat := 14011 :: root394Tail15
def root394Tail13 : List Nat := 13729 :: root394Tail14
def root394Tail12 : List Nat := 12379 :: root394Tail13
def root394Tail11 : List Nat := 12073 :: root394Tail12
def root394Tail10 : List Nat := 10687 :: root394Tail11
def root394Tail9 : List Nat := 10567 :: root394Tail10
def root394Tail8 : List Nat := 9817 :: root394Tail9
def root394Tail7 : List Nat := 9631 :: root394Tail8
def root394Tail6 : List Nat := 8713 :: root394Tail7
def root394Tail5 : List Nat := 8599 :: root394Tail6
def root394Tail4 : List Nat := 7669 :: root394Tail5
def root394Tail3 : List Nat := 7129 :: root394Tail4
def root394Tail2 : List Nat := 6121 :: root394Tail3
def root394Tail1 : List Nat := 4549 :: root394Tail2
def root394Tail0 : List Nat := 3391 :: root394Tail1
def root394Candidates0 : List Nat := [9817]
set_option maxRecDepth 8192 in
theorem root394Candidates0_checked : candidates edge 4 2719 3391 root394Tail1 winner.bound = root394Candidates0 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
theorem root394Step0 : scan edge (search edge 3) 4 [2719] 2719 root394Tail0 winner =
    scan edge (search edge 3) 4 [2719] 2719 root394Tail1 winner := by
  rw [root394Tail0, scan, if_neg (by decide), root394Candidates0_checked]
  change scan edge (search edge 3) 4 [2719] 2719 root394Tail1
    (if 3 ≤ root394Candidates0.length then search edge 3 [2719, 3391] root394Candidates0 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root394Candidates1 : List Nat := [9631, 14011]
set_option maxRecDepth 8192 in
theorem root394Candidates1_checked : candidates edge 4 2719 4549 root394Tail2 winner.bound = root394Candidates1 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root394Step1 : scan edge (search edge 3) 4 [2719] 2719 root394Tail1 winner =
    scan edge (search edge 3) 4 [2719] 2719 root394Tail2 winner := by
  rw [root394Tail1, scan, if_neg (by decide), root394Candidates1_checked]
  change scan edge (search edge 3) 4 [2719] 2719 root394Tail2
    (if 3 ≤ root394Candidates1.length then search edge 3 [2719, 4549] root394Candidates1 7268 winner else winner) = _
  rw [if_neg (by decide)]
theorem root394Scan2 : scan edge (search edge 3) 4 [2719] 2719 root394Tail2 winner = winner := by rw [root394Tail2, scan, if_pos (by decide)]
theorem root394Scan1 : scan edge (search edge 3) 4 [2719] 2719 root394Tail1 winner = winner :=
  (root394Step1).trans root394Scan2
theorem root394Scan0 : scan edge (search edge 3) 4 [2719] 2719 root394Tail0 winner = winner :=
  (root394Step0).trans root394Scan1
set_option maxRecDepth 4096 in
theorem root394Branch : search edge 4 [2719] root394Candidates 2719 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2719] 2719 root394Tail0 winner = winner
  exact root394Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
