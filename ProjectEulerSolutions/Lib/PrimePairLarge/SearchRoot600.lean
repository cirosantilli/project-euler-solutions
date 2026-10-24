import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root600Candidates : List Nat := Data.row4441
set_option maxRecDepth 8192 in
theorem root600Candidates_checked : candidates edge 5 0 4441 tail601 winner.bound = root600Candidates := by
  apply candidates_eq_row edge 5 0 4441 tail601 root600Candidates winner.bound tail601_ordered
  · decide
  · decide
  · funext q
    rfl
def root600Tail21 : List Nat := []
def root600Tail20 : List Nat := 21577 :: root600Tail21
def root600Tail19 : List Nat := 21319 :: root600Tail20
def root600Tail18 : List Nat := 21247 :: root600Tail19
def root600Tail17 : List Nat := 17497 :: root600Tail18
def root600Tail16 : List Nat := 12007 :: root600Tail17
def root600Tail15 : List Nat := 10477 :: root600Tail16
def root600Tail14 : List Nat := 9547 :: root600Tail15
def root600Tail13 : List Nat := 9127 :: root600Tail14
def root600Tail12 : List Nat := 8761 :: root600Tail13
def root600Tail11 : List Nat := 7873 :: root600Tail12
def root600Tail10 : List Nat := 7759 :: root600Tail11
def root600Tail9 : List Nat := 6949 :: root600Tail10
def root600Tail8 : List Nat := 6481 :: root600Tail9
def root600Tail7 : List Nat := 6373 :: root600Tail8
def root600Tail6 : List Nat := 6367 :: root600Tail7
def root600Tail5 : List Nat := 6271 :: root600Tail6
def root600Tail4 : List Nat := 6247 :: root600Tail5
def root600Tail3 : List Nat := 5953 :: root600Tail4
def root600Tail2 : List Nat := 5743 :: root600Tail3
def root600Tail1 : List Nat := 5347 :: root600Tail2
def root600Tail0 : List Nat := 5227 :: root600Tail1
def root600Candidates0 : List Nat := [6481]
set_option maxRecDepth 8192 in
theorem root600Candidates0_checked : candidates edge 4 4441 5227 root600Tail1 winner.bound = root600Candidates0 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root600Step0 : scan edge (search edge 3) 4 [4441] 4441 root600Tail0 winner =
    scan edge (search edge 3) 4 [4441] 4441 root600Tail1 winner := by
  rw [root600Tail0, scan, if_neg (by decide), root600Candidates0_checked]
  change scan edge (search edge 3) 4 [4441] 4441 root600Tail1
    (if 3 ≤ root600Candidates0.length then search edge 3 [4441, 5227] root600Candidates0 9668 winner else winner) = _
  rw [if_neg (by decide)]
def root600Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root600Candidates1_checked : candidates edge 4 4441 5347 root600Tail2 winner.bound = root600Candidates1 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root600Step1 : scan edge (search edge 3) 4 [4441] 4441 root600Tail1 winner =
    scan edge (search edge 3) 4 [4441] 4441 root600Tail2 winner := by
  rw [root600Tail1, scan, if_neg (by decide), root600Candidates1_checked]
  change scan edge (search edge 3) 4 [4441] 4441 root600Tail2
    (if 3 ≤ root600Candidates1.length then search edge 3 [4441, 5347] root600Candidates1 9788 winner else winner) = _
  rw [if_neg (by decide)]
theorem root600Scan2 : scan edge (search edge 3) 4 [4441] 4441 root600Tail2 winner = winner := by rw [root600Tail2, scan, if_pos (by decide)]
theorem root600Scan1 : scan edge (search edge 3) 4 [4441] 4441 root600Tail1 winner = winner :=
  (root600Step1).trans root600Scan2
theorem root600Scan0 : scan edge (search edge 3) 4 [4441] 4441 root600Tail0 winner = winner :=
  (root600Step0).trans root600Scan1
set_option maxRecDepth 4096 in
theorem root600Branch : search edge 4 [4441] root600Candidates 4441 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4441] 4441 root600Tail0 winner = winner
  exact root600Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
