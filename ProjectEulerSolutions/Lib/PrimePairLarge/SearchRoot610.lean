import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root610Candidates : List Nat := Data.row4517
set_option maxRecDepth 8192 in
theorem root610Candidates_checked : candidates edge 5 0 4517 tail611 winner.bound = root610Candidates := by
  apply candidates_eq_row edge 5 0 4517 tail611 root610Candidates winner.bound tail611_ordered
  · decide
  · decide
  · funext q
    rfl
def root610Tail29 : List Nat := []
def root610Tail28 : List Nat := 21503 :: root610Tail29
def root610Tail27 : List Nat := 20663 :: root610Tail28
def root610Tail26 : List Nat := 20147 :: root610Tail27
def root610Tail25 : List Nat := 19739 :: root610Tail26
def root610Tail24 : List Nat := 17729 :: root610Tail25
def root610Tail23 : List Nat := 16217 :: root610Tail24
def root610Tail22 : List Nat := 15359 :: root610Tail23
def root610Tail21 : List Nat := 14657 :: root610Tail22
def root610Tail20 : List Nat := 13763 :: root610Tail21
def root610Tail19 : List Nat := 13469 :: root610Tail20
def root610Tail18 : List Nat := 13109 :: root610Tail19
def root610Tail17 : List Nat := 12941 :: root610Tail18
def root610Tail16 : List Nat := 12473 :: root610Tail17
def root610Tail15 : List Nat := 12227 :: root610Tail16
def root610Tail14 : List Nat := 11027 :: root610Tail15
def root610Tail13 : List Nat := 10799 :: root610Tail14
def root610Tail12 : List Nat := 9689 :: root610Tail13
def root610Tail11 : List Nat := 9377 :: root610Tail12
def root610Tail10 : List Nat := 8387 :: root610Tail11
def root610Tail9 : List Nat := 8087 :: root610Tail10
def root610Tail8 : List Nat := 7523 :: root610Tail9
def root610Tail7 : List Nat := 6959 :: root610Tail8
def root610Tail6 : List Nat := 6869 :: root610Tail7
def root610Tail5 : List Nat := 6689 :: root610Tail6
def root610Tail4 : List Nat := 6581 :: root610Tail5
def root610Tail3 : List Nat := 5849 :: root610Tail4
def root610Tail2 : List Nat := 5387 :: root610Tail3
def root610Tail1 : List Nat := 5279 :: root610Tail2
def root610Tail0 : List Nat := 5087 :: root610Tail1
def root610Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root610Candidates0_checked : candidates edge 4 4517 5087 root610Tail1 winner.bound = root610Candidates0 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root610Step0 : scan edge (search edge 3) 4 [4517] 4517 root610Tail0 winner =
    scan edge (search edge 3) 4 [4517] 4517 root610Tail1 winner := by
  rw [root610Tail0, scan, if_neg (by decide), root610Candidates0_checked]
  change scan edge (search edge 3) 4 [4517] 4517 root610Tail1
    (if 3 ≤ root610Candidates0.length then search edge 3 [4517, 5087] root610Candidates0 9604 winner else winner) = _
  rw [if_neg (by decide)]
def root610Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root610Candidates1_checked : candidates edge 4 4517 5279 root610Tail2 winner.bound = root610Candidates1 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root610Step1 : scan edge (search edge 3) 4 [4517] 4517 root610Tail1 winner =
    scan edge (search edge 3) 4 [4517] 4517 root610Tail2 winner := by
  rw [root610Tail1, scan, if_neg (by decide), root610Candidates1_checked]
  change scan edge (search edge 3) 4 [4517] 4517 root610Tail2
    (if 3 ≤ root610Candidates1.length then search edge 3 [4517, 5279] root610Candidates1 9796 winner else winner) = _
  rw [if_neg (by decide)]
theorem root610Scan2 : scan edge (search edge 3) 4 [4517] 4517 root610Tail2 winner = winner := by rw [root610Tail2, scan, if_pos (by decide)]
theorem root610Scan1 : scan edge (search edge 3) 4 [4517] 4517 root610Tail1 winner = winner :=
  (root610Step1).trans root610Scan2
theorem root610Scan0 : scan edge (search edge 3) 4 [4517] 4517 root610Tail0 winner = winner :=
  (root610Step0).trans root610Scan1
set_option maxRecDepth 4096 in
theorem root610Branch : search edge 4 [4517] root610Candidates 4517 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4517] 4517 root610Tail0 winner = winner
  exact root610Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
