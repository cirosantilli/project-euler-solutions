import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root643Candidates : List Nat := Data.row4799
set_option maxRecDepth 8192 in
theorem root643Candidates_checked : candidates edge 5 0 4799 tail644 winner.bound = root643Candidates := by
  apply candidates_eq_row edge 5 0 4799 tail644 root643Candidates winner.bound tail644_ordered
  · decide
  · decide
  · funext q
    rfl
def root643Tail30 : List Nat := []
def root643Tail29 : List Nat := 19421 :: root643Tail30
def root643Tail28 : List Nat := 18617 :: root643Tail29
def root643Tail27 : List Nat := 18287 :: root643Tail28
def root643Tail26 : List Nat := 17489 :: root643Tail27
def root643Tail25 : List Nat := 17189 :: root643Tail26
def root643Tail24 : List Nat := 16979 :: root643Tail25
def root643Tail23 : List Nat := 15377 :: root643Tail24
def root643Tail22 : List Nat := 14843 :: root643Tail23
def root643Tail21 : List Nat := 14537 :: root643Tail22
def root643Tail20 : List Nat := 13613 :: root643Tail21
def root643Tail19 : List Nat := 13367 :: root643Tail20
def root643Tail18 : List Nat := 12569 :: root643Tail19
def root643Tail17 : List Nat := 12113 :: root643Tail18
def root643Tail16 : List Nat := 11489 :: root643Tail17
def root643Tail15 : List Nat := 11483 :: root643Tail16
def root643Tail14 : List Nat := 11447 :: root643Tail15
def root643Tail13 : List Nat := 11321 :: root643Tail14
def root643Tail12 : List Nat := 10253 :: root643Tail13
def root643Tail11 : List Nat := 10181 :: root643Tail12
def root643Tail10 : List Nat := 9521 :: root643Tail11
def root643Tail9 : List Nat := 9461 :: root643Tail10
def root643Tail8 : List Nat := 8597 :: root643Tail9
def root643Tail7 : List Nat := 7583 :: root643Tail8
def root643Tail6 : List Nat := 7487 :: root643Tail7
def root643Tail5 : List Nat := 6329 :: root643Tail6
def root643Tail4 : List Nat := 5867 :: root643Tail5
def root643Tail3 : List Nat := 5639 :: root643Tail4
def root643Tail2 : List Nat := 5531 :: root643Tail3
def root643Tail1 : List Nat := 5231 :: root643Tail2
def root643Tail0 : List Nat := 4889 :: root643Tail1
def root643Candidates0 : List Nat := [14843]
set_option maxRecDepth 8192 in
theorem root643Candidates0_checked : candidates edge 4 4799 4889 root643Tail1 winner.bound = root643Candidates0 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root643Step0 : scan edge (search edge 3) 4 [4799] 4799 root643Tail0 winner =
    scan edge (search edge 3) 4 [4799] 4799 root643Tail1 winner := by
  rw [root643Tail0, scan, if_neg (by decide), root643Candidates0_checked]
  change scan edge (search edge 3) 4 [4799] 4799 root643Tail1
    (if 3 ≤ root643Candidates0.length then search edge 3 [4799, 4889] root643Candidates0 9688 winner else winner) = _
  rw [if_neg (by decide)]
def root643Candidates1 : List Nat := [15377]
set_option maxRecDepth 8192 in
theorem root643Candidates1_checked : candidates edge 4 4799 5231 root643Tail2 winner.bound = root643Candidates1 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root643Step1 : scan edge (search edge 3) 4 [4799] 4799 root643Tail1 winner =
    scan edge (search edge 3) 4 [4799] 4799 root643Tail2 winner := by
  rw [root643Tail1, scan, if_neg (by decide), root643Candidates1_checked]
  change scan edge (search edge 3) 4 [4799] 4799 root643Tail2
    (if 3 ≤ root643Candidates1.length then search edge 3 [4799, 5231] root643Candidates1 10030 winner else winner) = _
  rw [if_neg (by decide)]
theorem root643Scan2 : scan edge (search edge 3) 4 [4799] 4799 root643Tail2 winner = winner := by rw [root643Tail2, scan, if_pos (by decide)]
theorem root643Scan1 : scan edge (search edge 3) 4 [4799] 4799 root643Tail1 winner = winner :=
  (root643Step1).trans root643Scan2
theorem root643Scan0 : scan edge (search edge 3) 4 [4799] 4799 root643Tail0 winner = winner :=
  (root643Step0).trans root643Scan1
set_option maxRecDepth 4096 in
theorem root643Branch : search edge 4 [4799] root643Candidates 4799 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4799] 4799 root643Tail0 winner = winner
  exact root643Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
