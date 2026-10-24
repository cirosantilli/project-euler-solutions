import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root607Candidates : List Nat := Data.row4493
set_option maxRecDepth 8192 in
theorem root607Candidates_checked : candidates edge 5 0 4493 tail608 winner.bound = root607Candidates := by
  apply candidates_eq_row edge 5 0 4493 tail608 root607Candidates winner.bound tail608_ordered
  · decide
  · decide
  · funext q
    rfl
def root607Tail30 : List Nat := []
def root607Tail29 : List Nat := 21221 :: root607Tail30
def root607Tail28 : List Nat := 21143 :: root607Tail29
def root607Tail27 : List Nat := 20903 :: root607Tail28
def root607Tail26 : List Nat := 20297 :: root607Tail27
def root607Tail25 : List Nat := 19991 :: root607Tail26
def root607Tail24 : List Nat := 18773 :: root607Tail25
def root607Tail23 : List Nat := 17669 :: root607Tail24
def root607Tail22 : List Nat := 17597 :: root607Tail23
def root607Tail21 : List Nat := 17027 :: root607Tail22
def root607Tail20 : List Nat := 15641 :: root607Tail21
def root607Tail19 : List Nat := 15497 :: root607Tail20
def root607Tail18 : List Nat := 14843 :: root607Tail19
def root607Tail17 : List Nat := 14741 :: root607Tail18
def root607Tail16 : List Nat := 13499 :: root607Tail17
def root607Tail15 : List Nat := 11777 :: root607Tail16
def root607Tail14 : List Nat := 11243 :: root607Tail15
def root607Tail13 : List Nat := 10889 :: root607Tail14
def root607Tail12 : List Nat := 10529 :: root607Tail13
def root607Tail11 : List Nat := 10151 :: root607Tail12
def root607Tail10 : List Nat := 10091 :: root607Tail11
def root607Tail9 : List Nat := 8693 :: root607Tail10
def root607Tail8 : List Nat := 7349 :: root607Tail9
def root607Tail7 : List Nat := 7187 :: root607Tail8
def root607Tail6 : List Nat := 6173 :: root607Tail7
def root607Tail5 : List Nat := 6053 :: root607Tail6
def root607Tail4 : List Nat := 5717 :: root607Tail5
def root607Tail3 : List Nat := 5519 :: root607Tail4
def root607Tail2 : List Nat := 5507 :: root607Tail3
def root607Tail1 : List Nat := 5381 :: root607Tail2
def root607Tail0 : List Nat := 4793 :: root607Tail1
def root607Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root607Candidates0_checked : candidates edge 4 4493 4793 root607Tail1 winner.bound = root607Candidates0 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root607Step0 : scan edge (search edge 3) 4 [4493] 4493 root607Tail0 winner =
    scan edge (search edge 3) 4 [4493] 4493 root607Tail1 winner := by
  rw [root607Tail0, scan, if_neg (by decide), root607Candidates0_checked]
  change scan edge (search edge 3) 4 [4493] 4493 root607Tail1
    (if 3 ≤ root607Candidates0.length then search edge 3 [4493, 4793] root607Candidates0 9286 winner else winner) = _
  rw [if_neg (by decide)]
def root607Candidates1 : List Nat := [5507]
set_option maxRecDepth 8192 in
theorem root607Candidates1_checked : candidates edge 4 4493 5381 root607Tail2 winner.bound = root607Candidates1 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root607Step1 : scan edge (search edge 3) 4 [4493] 4493 root607Tail1 winner =
    scan edge (search edge 3) 4 [4493] 4493 root607Tail2 winner := by
  rw [root607Tail1, scan, if_neg (by decide), root607Candidates1_checked]
  change scan edge (search edge 3) 4 [4493] 4493 root607Tail2
    (if 3 ≤ root607Candidates1.length then search edge 3 [4493, 5381] root607Candidates1 9874 winner else winner) = _
  rw [if_neg (by decide)]
theorem root607Scan2 : scan edge (search edge 3) 4 [4493] 4493 root607Tail2 winner = winner := by rw [root607Tail2, scan, if_pos (by decide)]
theorem root607Scan1 : scan edge (search edge 3) 4 [4493] 4493 root607Tail1 winner = winner :=
  (root607Step1).trans root607Scan2
theorem root607Scan0 : scan edge (search edge 3) 4 [4493] 4493 root607Tail0 winner = winner :=
  (root607Step0).trans root607Scan1
set_option maxRecDepth 4096 in
theorem root607Branch : search edge 4 [4493] root607Candidates 4493 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4493] 4493 root607Tail0 winner = winner
  exact root607Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
