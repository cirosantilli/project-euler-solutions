import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root638Candidates : List Nat := Data.row4759
set_option maxRecDepth 8192 in
theorem root638Candidates_checked : candidates edge 5 0 4759 tail639 winner.bound = root638Candidates := by
  apply candidates_eq_row edge 5 0 4759 tail639 root638Candidates winner.bound tail639_ordered
  · decide
  · decide
  · funext q
    rfl
def root638Tail25 : List Nat := []
def root638Tail24 : List Nat := 21211 :: root638Tail25
def root638Tail23 : List Nat := 21169 :: root638Tail24
def root638Tail22 : List Nat := 20983 :: root638Tail23
def root638Tail21 : List Nat := 20047 :: root638Tail22
def root638Tail20 : List Nat := 19333 :: root638Tail21
def root638Tail19 : List Nat := 19237 :: root638Tail20
def root638Tail18 : List Nat := 18787 :: root638Tail19
def root638Tail17 : List Nat := 17539 :: root638Tail18
def root638Tail16 : List Nat := 17191 :: root638Tail17
def root638Tail15 : List Nat := 16729 :: root638Tail16
def root638Tail14 : List Nat := 16087 :: root638Tail15
def root638Tail13 : List Nat := 14797 :: root638Tail14
def root638Tail12 : List Nat := 14479 :: root638Tail13
def root638Tail11 : List Nat := 13597 :: root638Tail12
def root638Tail10 : List Nat := 12919 :: root638Tail11
def root638Tail9 : List Nat := 12097 :: root638Tail10
def root638Tail8 : List Nat := 11827 :: root638Tail9
def root638Tail7 : List Nat := 11551 :: root638Tail8
def root638Tail6 : List Nat := 9661 :: root638Tail7
def root638Tail5 : List Nat := 9511 :: root638Tail6
def root638Tail4 : List Nat := 6823 :: root638Tail5
def root638Tail3 : List Nat := 6151 :: root638Tail4
def root638Tail2 : List Nat := 5701 :: root638Tail3
def root638Tail1 : List Nat := 5227 :: root638Tail2
def root638Tail0 : List Nat := 4903 :: root638Tail1
def root638Candidates0 : List Nat := [13597]
set_option maxRecDepth 8192 in
theorem root638Candidates0_checked : candidates edge 4 4759 4903 root638Tail1 winner.bound = root638Candidates0 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root638Step0 : scan edge (search edge 3) 4 [4759] 4759 root638Tail0 winner =
    scan edge (search edge 3) 4 [4759] 4759 root638Tail1 winner := by
  rw [root638Tail0, scan, if_neg (by decide), root638Candidates0_checked]
  change scan edge (search edge 3) 4 [4759] 4759 root638Tail1
    (if 3 ≤ root638Candidates0.length then search edge 3 [4759, 4903] root638Candidates0 9662 winner else winner) = _
  rw [if_neg (by decide)]
def root638Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root638Candidates1_checked : candidates edge 4 4759 5227 root638Tail2 winner.bound = root638Candidates1 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root638Step1 : scan edge (search edge 3) 4 [4759] 4759 root638Tail1 winner =
    scan edge (search edge 3) 4 [4759] 4759 root638Tail2 winner := by
  rw [root638Tail1, scan, if_neg (by decide), root638Candidates1_checked]
  change scan edge (search edge 3) 4 [4759] 4759 root638Tail2
    (if 3 ≤ root638Candidates1.length then search edge 3 [4759, 5227] root638Candidates1 9986 winner else winner) = _
  rw [if_neg (by decide)]
theorem root638Scan2 : scan edge (search edge 3) 4 [4759] 4759 root638Tail2 winner = winner := by rw [root638Tail2, scan, if_pos (by decide)]
theorem root638Scan1 : scan edge (search edge 3) 4 [4759] 4759 root638Tail1 winner = winner :=
  (root638Step1).trans root638Scan2
theorem root638Scan0 : scan edge (search edge 3) 4 [4759] 4759 root638Tail0 winner = winner :=
  (root638Step0).trans root638Scan1
set_option maxRecDepth 4096 in
theorem root638Branch : search edge 4 [4759] root638Candidates 4759 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4759] 4759 root638Tail0 winner = winner
  exact root638Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
