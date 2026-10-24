import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root572Candidates : List Nat := Data.row4201
set_option maxRecDepth 8192 in
theorem root572Candidates_checked : candidates edge 5 0 4201 tail573 winner.bound = root572Candidates := by
  apply candidates_eq_row edge 5 0 4201 tail573 root572Candidates winner.bound tail573_ordered
  · decide
  · decide
  · funext q
    rfl
def root572Tail29 : List Nat := []
def root572Tail28 : List Nat := 21529 :: root572Tail29
def root572Tail27 : List Nat := 21379 :: root572Tail28
def root572Tail26 : List Nat := 19237 :: root572Tail27
def root572Tail25 : List Nat := 19069 :: root572Tail26
def root572Tail24 : List Nat := 17041 :: root572Tail25
def root572Tail23 : List Nat := 16729 :: root572Tail24
def root572Tail22 : List Nat := 16339 :: root572Tail23
def root572Tail21 : List Nat := 15991 :: root572Tail22
def root572Tail20 : List Nat := 15877 :: root572Tail21
def root572Tail19 : List Nat := 14797 :: root572Tail20
def root572Tail18 : List Nat := 14281 :: root572Tail19
def root572Tail17 : List Nat := 13219 :: root572Tail18
def root572Tail16 : List Nat := 12379 :: root572Tail17
def root572Tail15 : List Nat := 11437 :: root572Tail16
def root572Tail14 : List Nat := 10753 :: root572Tail15
def root572Tail13 : List Nat := 10597 :: root572Tail14
def root572Tail12 : List Nat := 9421 :: root572Tail13
def root572Tail11 : List Nat := 9403 :: root572Tail12
def root572Tail10 : List Nat := 9067 :: root572Tail11
def root572Tail9 : List Nat := 8971 :: root572Tail10
def root572Tail8 : List Nat := 8863 :: root572Tail9
def root572Tail7 : List Nat := 8161 :: root572Tail8
def root572Tail6 : List Nat := 6421 :: root572Tail7
def root572Tail5 : List Nat := 6217 :: root572Tail6
def root572Tail4 : List Nat := 6199 :: root572Tail5
def root572Tail3 : List Nat := 5749 :: root572Tail4
def root572Tail2 : List Nat := 5527 :: root572Tail3
def root572Tail1 : List Nat := 5419 :: root572Tail2
def root572Tail0 : List Nat := 4759 :: root572Tail1
def root572Candidates0 : List Nat := [14797, 16729]
set_option maxRecDepth 8192 in
theorem root572Candidates0_checked : candidates edge 4 4201 4759 root572Tail1 winner.bound = root572Candidates0 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root572Step0 : scan edge (search edge 3) 4 [4201] 4201 root572Tail0 winner =
    scan edge (search edge 3) 4 [4201] 4201 root572Tail1 winner := by
  rw [root572Tail0, scan, if_neg (by decide), root572Candidates0_checked]
  change scan edge (search edge 3) 4 [4201] 4201 root572Tail1
    (if 3 ≤ root572Candidates0.length then search edge 3 [4201, 4759] root572Candidates0 8960 winner else winner) = _
  rw [if_neg (by decide)]
def root572Candidates1 : List Nat := [9421]
set_option maxRecDepth 8192 in
theorem root572Candidates1_checked : candidates edge 4 4201 5419 root572Tail2 winner.bound = root572Candidates1 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root572Step1 : scan edge (search edge 3) 4 [4201] 4201 root572Tail1 winner =
    scan edge (search edge 3) 4 [4201] 4201 root572Tail2 winner := by
  rw [root572Tail1, scan, if_neg (by decide), root572Candidates1_checked]
  change scan edge (search edge 3) 4 [4201] 4201 root572Tail2
    (if 3 ≤ root572Candidates1.length then search edge 3 [4201, 5419] root572Candidates1 9620 winner else winner) = _
  rw [if_neg (by decide)]
theorem root572Scan2 : scan edge (search edge 3) 4 [4201] 4201 root572Tail2 winner = winner := by rw [root572Tail2, scan, if_pos (by decide)]
theorem root572Scan1 : scan edge (search edge 3) 4 [4201] 4201 root572Tail1 winner = winner :=
  (root572Step1).trans root572Scan2
theorem root572Scan0 : scan edge (search edge 3) 4 [4201] 4201 root572Tail0 winner = winner :=
  (root572Step0).trans root572Scan1
set_option maxRecDepth 4096 in
theorem root572Branch : search edge 4 [4201] root572Candidates 4201 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4201] 4201 root572Tail0 winner = winner
  exact root572Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
