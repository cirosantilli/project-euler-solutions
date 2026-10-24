import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root629Candidates : List Nat := Data.row4673
set_option maxRecDepth 8192 in
theorem root629Candidates_checked : candidates edge 5 0 4673 tail630 winner.bound = root629Candidates := by
  apply candidates_eq_row edge 5 0 4673 tail630 root629Candidates winner.bound tail630_ordered
  · decide
  · decide
  · funext q
    rfl
def root629Tail27 : List Nat := []
def root629Tail26 : List Nat := 21059 :: root629Tail27
def root629Tail25 : List Nat := 20639 :: root629Tail26
def root629Tail24 : List Nat := 20393 :: root629Tail25
def root629Tail23 : List Nat := 19913 :: root629Tail24
def root629Tail22 : List Nat := 16901 :: root629Tail23
def root629Tail21 : List Nat := 16481 :: root629Tail22
def root629Tail20 : List Nat := 15461 :: root629Tail21
def root629Tail19 : List Nat := 15137 :: root629Tail20
def root629Tail18 : List Nat := 14957 :: root629Tail19
def root629Tail17 : List Nat := 14723 :: root629Tail18
def root629Tail16 : List Nat := 13967 :: root629Tail17
def root629Tail15 : List Nat := 13883 :: root629Tail16
def root629Tail14 : List Nat := 10937 :: root629Tail15
def root629Tail13 : List Nat := 10601 :: root629Tail14
def root629Tail12 : List Nat := 10391 :: root629Tail13
def root629Tail11 : List Nat := 10313 :: root629Tail12
def root629Tail10 : List Nat := 9551 :: root629Tail11
def root629Tail9 : List Nat := 9437 :: root629Tail10
def root629Tail8 : List Nat := 8009 :: root629Tail9
def root629Tail7 : List Nat := 7451 :: root629Tail8
def root629Tail6 : List Nat := 6947 :: root629Tail7
def root629Tail5 : List Nat := 6779 :: root629Tail6
def root629Tail4 : List Nat := 6737 :: root629Tail5
def root629Tail3 : List Nat := 6359 :: root629Tail4
def root629Tail2 : List Nat := 5519 :: root629Tail3
def root629Tail1 : List Nat := 5309 :: root629Tail2
def root629Tail0 : List Nat := 4799 :: root629Tail1
def root629Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root629Candidates0_checked : candidates edge 4 4673 4799 root629Tail1 winner.bound = root629Candidates0 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root629Step0 : scan edge (search edge 3) 4 [4673] 4673 root629Tail0 winner =
    scan edge (search edge 3) 4 [4673] 4673 root629Tail1 winner := by
  rw [root629Tail0, scan, if_neg (by decide), root629Candidates0_checked]
  change scan edge (search edge 3) 4 [4673] 4673 root629Tail1
    (if 3 ≤ root629Candidates0.length then search edge 3 [4673, 4799] root629Candidates0 9472 winner else winner) = _
  rw [if_neg (by decide)]
def root629Candidates1 : List Nat := [6359]
set_option maxRecDepth 8192 in
theorem root629Candidates1_checked : candidates edge 4 4673 5309 root629Tail2 winner.bound = root629Candidates1 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root629Step1 : scan edge (search edge 3) 4 [4673] 4673 root629Tail1 winner =
    scan edge (search edge 3) 4 [4673] 4673 root629Tail2 winner := by
  rw [root629Tail1, scan, if_neg (by decide), root629Candidates1_checked]
  change scan edge (search edge 3) 4 [4673] 4673 root629Tail2
    (if 3 ≤ root629Candidates1.length then search edge 3 [4673, 5309] root629Candidates1 9982 winner else winner) = _
  rw [if_neg (by decide)]
theorem root629Scan2 : scan edge (search edge 3) 4 [4673] 4673 root629Tail2 winner = winner := by rw [root629Tail2, scan, if_pos (by decide)]
theorem root629Scan1 : scan edge (search edge 3) 4 [4673] 4673 root629Tail1 winner = winner :=
  (root629Step1).trans root629Scan2
theorem root629Scan0 : scan edge (search edge 3) 4 [4673] 4673 root629Tail0 winner = winner :=
  (root629Step0).trans root629Scan1
set_option maxRecDepth 4096 in
theorem root629Branch : search edge 4 [4673] root629Candidates 4673 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4673] 4673 root629Tail0 winner = winner
  exact root629Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
