import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root630Candidates : List Nat := Data.row4679
set_option maxRecDepth 8192 in
theorem root630Candidates_checked : candidates edge 5 0 4679 tail631 winner.bound = root630Candidates := by
  apply candidates_eq_row edge 5 0 4679 tail631 root630Candidates winner.bound tail631_ordered
  · decide
  · decide
  · funext q
    rfl
def root630Tail29 : List Nat := []
def root630Tail28 : List Nat := 20879 :: root630Tail29
def root630Tail27 : List Nat := 20183 :: root630Tail28
def root630Tail26 : List Nat := 18413 :: root630Tail27
def root630Tail25 : List Nat := 18401 :: root630Tail26
def root630Tail24 : List Nat := 16607 :: root630Tail25
def root630Tail23 : List Nat := 16091 :: root630Tail24
def root630Tail22 : List Nat := 16061 :: root630Tail23
def root630Tail21 : List Nat := 15647 :: root630Tail22
def root630Tail20 : List Nat := 15131 :: root630Tail21
def root630Tail19 : List Nat := 14633 :: root630Tail20
def root630Tail18 : List Nat := 14627 :: root630Tail19
def root630Tail17 : List Nat := 14033 :: root630Tail18
def root630Tail16 : List Nat := 11597 :: root630Tail17
def root630Tail15 : List Nat := 11177 :: root630Tail16
def root630Tail14 : List Nat := 10691 :: root630Tail15
def root630Tail13 : List Nat := 10343 :: root630Tail14
def root630Tail12 : List Nat := 10337 :: root630Tail13
def root630Tail11 : List Nat := 9887 :: root630Tail12
def root630Tail10 : List Nat := 8867 :: root630Tail11
def root630Tail9 : List Nat := 8819 :: root630Tail10
def root630Tail8 : List Nat := 8783 :: root630Tail9
def root630Tail7 : List Nat := 6653 :: root630Tail8
def root630Tail6 : List Nat := 6569 :: root630Tail7
def root630Tail5 : List Nat := 6521 :: root630Tail6
def root630Tail4 : List Nat := 5927 :: root630Tail5
def root630Tail3 : List Nat := 5717 :: root630Tail4
def root630Tail2 : List Nat := 5393 :: root630Tail3
def root630Tail1 : List Nat := 5003 :: root630Tail2
def root630Tail0 : List Nat := 4889 :: root630Tail1
def root630Candidates0 : List Nat := [11597]
set_option maxRecDepth 8192 in
theorem root630Candidates0_checked : candidates edge 4 4679 4889 root630Tail1 winner.bound = root630Candidates0 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root630Step0 : scan edge (search edge 3) 4 [4679] 4679 root630Tail0 winner =
    scan edge (search edge 3) 4 [4679] 4679 root630Tail1 winner := by
  rw [root630Tail0, scan, if_neg (by decide), root630Candidates0_checked]
  change scan edge (search edge 3) 4 [4679] 4679 root630Tail1
    (if 3 ≤ root630Candidates0.length then search edge 3 [4679, 4889] root630Candidates0 9568 winner else winner) = _
  rw [if_neg (by decide)]
def root630Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root630Candidates1_checked : candidates edge 4 4679 5003 root630Tail2 winner.bound = root630Candidates1 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root630Step1 : scan edge (search edge 3) 4 [4679] 4679 root630Tail1 winner =
    scan edge (search edge 3) 4 [4679] 4679 root630Tail2 winner := by
  rw [root630Tail1, scan, if_neg (by decide), root630Candidates1_checked]
  change scan edge (search edge 3) 4 [4679] 4679 root630Tail2
    (if 3 ≤ root630Candidates1.length then search edge 3 [4679, 5003] root630Candidates1 9682 winner else winner) = _
  rw [if_neg (by decide)]
theorem root630Scan2 : scan edge (search edge 3) 4 [4679] 4679 root630Tail2 winner = winner := by rw [root630Tail2, scan, if_pos (by decide)]
theorem root630Scan1 : scan edge (search edge 3) 4 [4679] 4679 root630Tail1 winner = winner :=
  (root630Step1).trans root630Scan2
theorem root630Scan0 : scan edge (search edge 3) 4 [4679] 4679 root630Tail0 winner = winner :=
  (root630Step0).trans root630Scan1
set_option maxRecDepth 4096 in
theorem root630Branch : search edge 4 [4679] root630Candidates 4679 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4679] 4679 root630Tail0 winner = winner
  exact root630Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
