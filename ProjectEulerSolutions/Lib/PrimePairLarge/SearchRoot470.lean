import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root470Candidates : List Nat := Data.row3359
set_option maxRecDepth 8192 in
theorem root470Candidates_checked : candidates edge 5 0 3359 tail471 winner.bound = root470Candidates := by
  apply candidates_eq_row edge 5 0 3359 tail471 root470Candidates winner.bound tail471_ordered
  · decide
  · decide
  · funext q
    rfl
def root470Tail34 : List Nat := []
def root470Tail33 : List Nat := 22637 :: root470Tail34
def root470Tail32 : List Nat := 21143 :: root470Tail33
def root470Tail31 : List Nat := 20903 :: root470Tail32
def root470Tail30 : List Nat := 20789 :: root470Tail31
def root470Tail29 : List Nat := 20297 :: root470Tail30
def root470Tail28 : List Nat := 20177 :: root470Tail29
def root470Tail27 : List Nat := 19079 :: root470Tail28
def root470Tail26 : List Nat := 18287 :: root470Tail27
def root470Tail25 : List Nat := 17837 :: root470Tail26
def root470Tail24 : List Nat := 16901 :: root470Tail25
def root470Tail23 : List Nat := 16871 :: root470Tail24
def root470Tail22 : List Nat := 16631 :: root470Tail23
def root470Tail21 : List Nat := 15791 :: root470Tail22
def root470Tail20 : List Nat := 15683 :: root470Tail21
def root470Tail19 : List Nat := 12923 :: root470Tail20
def root470Tail18 : List Nat := 12659 :: root470Tail19
def root470Tail17 : List Nat := 11087 :: root470Tail18
def root470Tail16 : List Nat := 11003 :: root470Tail17
def root470Tail15 : List Nat := 10247 :: root470Tail16
def root470Tail14 : List Nat := 9677 :: root470Tail15
def root470Tail13 : List Nat := 9521 :: root470Tail14
def root470Tail12 : List Nat := 9281 :: root470Tail13
def root470Tail11 : List Nat := 8753 :: root470Tail12
def root470Tail10 : List Nat := 8597 :: root470Tail11
def root470Tail9 : List Nat := 8447 :: root470Tail10
def root470Tail8 : List Nat := 7853 :: root470Tail9
def root470Tail7 : List Nat := 7649 :: root470Tail8
def root470Tail6 : List Nat := 6779 :: root470Tail7
def root470Tail5 : List Nat := 6737 :: root470Tail6
def root470Tail4 : List Nat := 6599 :: root470Tail5
def root470Tail3 : List Nat := 6257 :: root470Tail4
def root470Tail2 : List Nat := 5981 :: root470Tail3
def root470Tail1 : List Nat := 4229 :: root470Tail2
def root470Tail0 : List Nat := 4217 :: root470Tail1
def root470Candidates0 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root470Candidates0_checked : candidates edge 4 3359 4217 root470Tail1 winner.bound = root470Candidates0 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root470Step0 : scan edge (search edge 3) 4 [3359] 3359 root470Tail0 winner =
    scan edge (search edge 3) 4 [3359] 3359 root470Tail1 winner := by
  rw [root470Tail0, scan, if_neg (by decide), root470Candidates0_checked]
  change scan edge (search edge 3) 4 [3359] 3359 root470Tail1
    (if 3 ≤ root470Candidates0.length then search edge 3 [3359, 4217] root470Candidates0 7576 winner else winner) = _
  rw [if_neg (by decide)]
def root470Candidates1 : List Nat := [15683]
set_option maxRecDepth 8192 in
theorem root470Candidates1_checked : candidates edge 4 3359 4229 root470Tail2 winner.bound = root470Candidates1 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root470Step1 : scan edge (search edge 3) 4 [3359] 3359 root470Tail1 winner =
    scan edge (search edge 3) 4 [3359] 3359 root470Tail2 winner := by
  rw [root470Tail1, scan, if_neg (by decide), root470Candidates1_checked]
  change scan edge (search edge 3) 4 [3359] 3359 root470Tail2
    (if 3 ≤ root470Candidates1.length then search edge 3 [3359, 4229] root470Candidates1 7588 winner else winner) = _
  rw [if_neg (by decide)]
theorem root470Scan2 : scan edge (search edge 3) 4 [3359] 3359 root470Tail2 winner = winner := by rw [root470Tail2, scan, if_pos (by decide)]
theorem root470Scan1 : scan edge (search edge 3) 4 [3359] 3359 root470Tail1 winner = winner :=
  (root470Step1).trans root470Scan2
theorem root470Scan0 : scan edge (search edge 3) 4 [3359] 3359 root470Tail0 winner = winner :=
  (root470Step0).trans root470Scan1
set_option maxRecDepth 4096 in
theorem root470Branch : search edge 4 [3359] root470Candidates 3359 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3359] 3359 root470Tail0 winner = winner
  exact root470Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
