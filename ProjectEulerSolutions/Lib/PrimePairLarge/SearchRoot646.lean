import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root646Candidates : List Nat := Data.row4817
set_option maxRecDepth 8192 in
theorem root646Candidates_checked : candidates edge 5 0 4817 tail647 winner.bound = root646Candidates := by
  apply candidates_eq_row edge 5 0 4817 tail647 root646Candidates winner.bound tail647_ordered
  · decide
  · decide
  · funext q
    rfl
def root646Tail23 : List Nat := []
def root646Tail22 : List Nat := 20639 :: root646Tail23
def root646Tail21 : List Nat := 18959 :: root646Tail22
def root646Tail20 : List Nat := 18521 :: root646Tail21
def root646Tail19 : List Nat := 18257 :: root646Tail20
def root646Tail18 : List Nat := 17417 :: root646Tail19
def root646Tail17 : List Nat := 17333 :: root646Tail18
def root646Tail16 : List Nat := 17207 :: root646Tail17
def root646Tail15 : List Nat := 16691 :: root646Tail16
def root646Tail14 : List Nat := 16619 :: root646Tail15
def root646Tail13 : List Nat := 15413 :: root646Tail14
def root646Tail12 : List Nat := 15263 :: root646Tail13
def root646Tail11 : List Nat := 14813 :: root646Tail12
def root646Tail10 : List Nat := 14633 :: root646Tail11
def root646Tail9 : List Nat := 13901 :: root646Tail10
def root646Tail8 : List Nat := 13367 :: root646Tail9
def root646Tail7 : List Nat := 12251 :: root646Tail8
def root646Tail6 : List Nat := 7127 :: root646Tail7
def root646Tail5 : List Nat := 6197 :: root646Tail6
def root646Tail4 : List Nat := 6173 :: root646Tail5
def root646Tail3 : List Nat := 5903 :: root646Tail4
def root646Tail2 : List Nat := 5783 :: root646Tail3
def root646Tail1 : List Nat := 5741 :: root646Tail2
def root646Tail0 : List Nat := 5021 :: root646Tail1
def root646Candidates0 : List Nat := [5741, 7127]
set_option maxRecDepth 8192 in
theorem root646Candidates0_checked : candidates edge 4 4817 5021 root646Tail1 winner.bound = root646Candidates0 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root646Step0 : scan edge (search edge 3) 4 [4817] 4817 root646Tail0 winner =
    scan edge (search edge 3) 4 [4817] 4817 root646Tail1 winner := by
  rw [root646Tail0, scan, if_neg (by decide), root646Candidates0_checked]
  change scan edge (search edge 3) 4 [4817] 4817 root646Tail1
    (if 3 ≤ root646Candidates0.length then search edge 3 [4817, 5021] root646Candidates0 9838 winner else winner) = _
  rw [if_neg (by decide)]
theorem root646Scan1 : scan edge (search edge 3) 4 [4817] 4817 root646Tail1 winner = winner := by rw [root646Tail1, scan, if_pos (by decide)]
theorem root646Scan0 : scan edge (search edge 3) 4 [4817] 4817 root646Tail0 winner = winner :=
  (root646Step0).trans root646Scan1
set_option maxRecDepth 4096 in
theorem root646Branch : search edge 4 [4817] root646Candidates 4817 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4817] 4817 root646Tail0 winner = winner
  exact root646Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
