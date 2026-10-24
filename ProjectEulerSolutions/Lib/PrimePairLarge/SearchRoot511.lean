import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root511Candidates : List Nat := Data.row3677
set_option maxRecDepth 8192 in
theorem root511Candidates_checked : candidates edge 5 0 3677 tail512 winner.bound = root511Candidates := by
  apply candidates_eq_row edge 5 0 3677 tail512 root511Candidates winner.bound tail512_ordered
  · decide
  · decide
  · funext q
    rfl
def root511Tail27 : List Nat := []
def root511Tail26 : List Nat := 22259 :: root511Tail27
def root511Tail25 : List Nat := 21617 :: root511Tail26
def root511Tail24 : List Nat := 19139 :: root511Tail25
def root511Tail23 : List Nat := 18731 :: root511Tail24
def root511Tail22 : List Nat := 18401 :: root511Tail23
def root511Tail21 : List Nat := 17417 :: root511Tail22
def root511Tail20 : List Nat := 17033 :: root511Tail21
def root511Tail19 : List Nat := 16763 :: root511Tail20
def root511Tail18 : List Nat := 14813 :: root511Tail19
def root511Tail17 : List Nat := 14411 :: root511Tail18
def root511Tail16 : List Nat := 14327 :: root511Tail17
def root511Tail15 : List Nat := 13679 :: root511Tail16
def root511Tail14 : List Nat := 13613 :: root511Tail15
def root511Tail13 : List Nat := 12269 :: root511Tail14
def root511Tail12 : List Nat := 11969 :: root511Tail13
def root511Tail11 : List Nat := 11321 :: root511Tail12
def root511Tail10 : List Nat := 10151 :: root511Tail11
def root511Tail9 : List Nat := 9629 :: root511Tail10
def root511Tail8 : List Nat := 9377 :: root511Tail9
def root511Tail7 : List Nat := 8069 :: root511Tail8
def root511Tail6 : List Nat := 7583 :: root511Tail7
def root511Tail5 : List Nat := 7229 :: root511Tail6
def root511Tail4 : List Nat := 6659 :: root511Tail5
def root511Tail3 : List Nat := 5903 :: root511Tail4
def root511Tail2 : List Nat := 5807 :: root511Tail3
def root511Tail1 : List Nat := 4733 :: root511Tail2
def root511Tail0 : List Nat := 4211 :: root511Tail1
def root511Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root511Candidates0_checked : candidates edge 4 3677 4211 root511Tail1 winner.bound = root511Candidates0 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root511Step0 : scan edge (search edge 3) 4 [3677] 3677 root511Tail0 winner =
    scan edge (search edge 3) 4 [3677] 3677 root511Tail1 winner := by
  rw [root511Tail0, scan, if_neg (by decide), root511Candidates0_checked]
  change scan edge (search edge 3) 4 [3677] 3677 root511Tail1
    (if 3 ≤ root511Candidates0.length then search edge 3 [3677, 4211] root511Candidates0 7888 winner else winner) = _
  rw [if_neg (by decide)]
def root511Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root511Candidates1_checked : candidates edge 4 3677 4733 root511Tail2 winner.bound = root511Candidates1 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root511Step1 : scan edge (search edge 3) 4 [3677] 3677 root511Tail1 winner =
    scan edge (search edge 3) 4 [3677] 3677 root511Tail2 winner := by
  rw [root511Tail1, scan, if_neg (by decide), root511Candidates1_checked]
  change scan edge (search edge 3) 4 [3677] 3677 root511Tail2
    (if 3 ≤ root511Candidates1.length then search edge 3 [3677, 4733] root511Candidates1 8410 winner else winner) = _
  rw [if_neg (by decide)]
theorem root511Scan2 : scan edge (search edge 3) 4 [3677] 3677 root511Tail2 winner = winner := by rw [root511Tail2, scan, if_pos (by decide)]
theorem root511Scan1 : scan edge (search edge 3) 4 [3677] 3677 root511Tail1 winner = winner :=
  (root511Step1).trans root511Scan2
theorem root511Scan0 : scan edge (search edge 3) 4 [3677] 3677 root511Tail0 winner = winner :=
  (root511Step0).trans root511Scan1
set_option maxRecDepth 4096 in
theorem root511Branch : search edge 4 [3677] root511Candidates 3677 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3677] 3677 root511Tail0 winner = winner
  exact root511Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
