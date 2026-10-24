import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root476Candidates : List Nat := Data.row3407
set_option maxRecDepth 8192 in
theorem root476Candidates_checked : candidates edge 5 0 3407 tail477 winner.bound = root476Candidates := by
  apply candidates_eq_row edge 5 0 3407 tail477 root476Candidates winner.bound tail477_ordered
  · decide
  · decide
  · funext q
    rfl
def root476Tail40 : List Nat := []
def root476Tail39 : List Nat := 21503 :: root476Tail40
def root476Tail38 : List Nat := 20981 :: root476Tail39
def root476Tail37 : List Nat := 20879 :: root476Tail38
def root476Tail36 : List Nat := 20399 :: root476Tail37
def root476Tail35 : List Nat := 20147 :: root476Tail36
def root476Tail34 : List Nat := 19433 :: root476Tail35
def root476Tail33 : List Nat := 19259 :: root476Tail34
def root476Tail32 : List Nat := 17921 :: root476Tail33
def root476Tail31 : List Nat := 16619 :: root476Tail32
def root476Tail30 : List Nat := 16427 :: root476Tail31
def root476Tail29 : List Nat := 15527 :: root476Tail30
def root476Tail28 : List Nat := 15497 :: root476Tail29
def root476Tail27 : List Nat := 15329 :: root476Tail28
def root476Tail26 : List Nat := 14747 :: root476Tail27
def root476Tail25 : List Nat := 14657 :: root476Tail26
def root476Tail24 : List Nat := 13691 :: root476Tail25
def root476Tail23 : List Nat := 13523 :: root476Tail24
def root476Tail22 : List Nat := 12689 :: root476Tail23
def root476Tail21 : List Nat := 12101 :: root476Tail22
def root476Tail20 : List Nat := 11621 :: root476Tail21
def root476Tail19 : List Nat := 11471 :: root476Tail20
def root476Tail18 : List Nat := 11411 :: root476Tail19
def root476Tail17 : List Nat := 10529 :: root476Tail18
def root476Tail16 : List Nat := 10151 :: root476Tail17
def root476Tail15 : List Nat := 9743 :: root476Tail16
def root476Tail14 : List Nat := 9467 :: root476Tail15
def root476Tail13 : List Nat := 9239 :: root476Tail14
def root476Tail12 : List Nat := 9029 :: root476Tail13
def root476Tail11 : List Nat := 8819 :: root476Tail12
def root476Tail10 : List Nat := 8699 :: root476Tail11
def root476Tail9 : List Nat := 8669 :: root476Tail10
def root476Tail8 : List Nat := 7919 :: root476Tail9
def root476Tail7 : List Nat := 7691 :: root476Tail8
def root476Tail6 : List Nat := 7079 :: root476Tail7
def root476Tail5 : List Nat := 6599 :: root476Tail6
def root476Tail4 : List Nat := 6389 :: root476Tail5
def root476Tail3 : List Nat := 6299 :: root476Tail4
def root476Tail2 : List Nat := 4889 :: root476Tail3
def root476Tail1 : List Nat := 4751 :: root476Tail2
def root476Tail0 : List Nat := 3671 :: root476Tail1
def root476Candidates0 : List Nat := [7691, 9467]
set_option maxRecDepth 8192 in
theorem root476Candidates0_checked : candidates edge 4 3407 3671 root476Tail1 winner.bound = root476Candidates0 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root476Step0 : scan edge (search edge 3) 4 [3407] 3407 root476Tail0 winner =
    scan edge (search edge 3) 4 [3407] 3407 root476Tail1 winner := by
  rw [root476Tail0, scan, if_neg (by decide), root476Candidates0_checked]
  change scan edge (search edge 3) 4 [3407] 3407 root476Tail1
    (if 3 ≤ root476Candidates0.length then search edge 3 [3407, 3671] root476Candidates0 7078 winner else winner) = _
  rw [if_neg (by decide)]
def root476Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root476Candidates1_checked : candidates edge 4 3407 4751 root476Tail2 winner.bound = root476Candidates1 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root476Step1 : scan edge (search edge 3) 4 [3407] 3407 root476Tail1 winner =
    scan edge (search edge 3) 4 [3407] 3407 root476Tail2 winner := by
  rw [root476Tail1, scan, if_neg (by decide), root476Candidates1_checked]
  change scan edge (search edge 3) 4 [3407] 3407 root476Tail2
    (if 3 ≤ root476Candidates1.length then search edge 3 [3407, 4751] root476Candidates1 8158 winner else winner) = _
  rw [if_neg (by decide)]
def root476Candidates2 : List Nat := [9743, 12101]
set_option maxRecDepth 8192 in
theorem root476Candidates2_checked : candidates edge 4 3407 4889 root476Tail3 winner.bound = root476Candidates2 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root476Step2 : scan edge (search edge 3) 4 [3407] 3407 root476Tail2 winner =
    scan edge (search edge 3) 4 [3407] 3407 root476Tail3 winner := by
  rw [root476Tail2, scan, if_neg (by decide), root476Candidates2_checked]
  change scan edge (search edge 3) 4 [3407] 3407 root476Tail3
    (if 3 ≤ root476Candidates2.length then search edge 3 [3407, 4889] root476Candidates2 8296 winner else winner) = _
  rw [if_neg (by decide)]
theorem root476Scan3 : scan edge (search edge 3) 4 [3407] 3407 root476Tail3 winner = winner := by rw [root476Tail3, scan, if_pos (by decide)]
theorem root476Scan2 : scan edge (search edge 3) 4 [3407] 3407 root476Tail2 winner = winner :=
  (root476Step2).trans root476Scan3
theorem root476Scan1 : scan edge (search edge 3) 4 [3407] 3407 root476Tail1 winner = winner :=
  (root476Step1).trans root476Scan2
theorem root476Scan0 : scan edge (search edge 3) 4 [3407] 3407 root476Tail0 winner = winner :=
  (root476Step0).trans root476Scan1
set_option maxRecDepth 4096 in
theorem root476Branch : search edge 4 [3407] root476Candidates 3407 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3407] 3407 root476Tail0 winner = winner
  exact root476Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
