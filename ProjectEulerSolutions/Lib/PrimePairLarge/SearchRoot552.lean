import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root552Candidates : List Nat := Data.row4019
set_option maxRecDepth 8192 in
theorem root552Candidates_checked : candidates edge 5 0 4019 tail553 winner.bound = root552Candidates := by
  apply candidates_eq_row edge 5 0 4019 tail553 root552Candidates winner.bound tail553_ordered
  · decide
  · decide
  · funext q
    rfl
def root552Tail21 : List Nat := []
def root552Tail20 : List Nat := 20903 :: root552Tail21
def root552Tail19 : List Nat := 20231 :: root552Tail20
def root552Tail18 : List Nat := 20063 :: root552Tail19
def root552Tail17 : List Nat := 19571 :: root552Tail18
def root552Tail16 : List Nat := 18719 :: root552Tail17
def root552Tail15 : List Nat := 17903 :: root552Tail16
def root552Tail14 : List Nat := 17489 :: root552Tail15
def root552Tail13 : List Nat := 15161 :: root552Tail14
def root552Tail12 : List Nat := 13901 :: root552Tail13
def root552Tail11 : List Nat := 11987 :: root552Tail12
def root552Tail10 : List Nat := 10433 :: root552Tail11
def root552Tail9 : List Nat := 10181 :: root552Tail10
def root552Tail8 : List Nat := 10139 :: root552Tail9
def root552Tail7 : List Nat := 9941 :: root552Tail8
def root552Tail6 : List Nat := 9479 :: root552Tail7
def root552Tail5 : List Nat := 7541 :: root552Tail6
def root552Tail4 : List Nat := 7211 :: root552Tail5
def root552Tail3 : List Nat := 7121 :: root552Tail4
def root552Tail2 : List Nat := 6791 :: root552Tail3
def root552Tail1 : List Nat := 5471 :: root552Tail2
def root552Tail0 : List Nat := 4211 :: root552Tail1
def root552Candidates0 : List Nat := [10181]
set_option maxRecDepth 8192 in
theorem root552Candidates0_checked : candidates edge 4 4019 4211 root552Tail1 winner.bound = root552Candidates0 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root552Step0 : scan edge (search edge 3) 4 [4019] 4019 root552Tail0 winner =
    scan edge (search edge 3) 4 [4019] 4019 root552Tail1 winner := by
  rw [root552Tail0, scan, if_neg (by decide), root552Candidates0_checked]
  change scan edge (search edge 3) 4 [4019] 4019 root552Tail1
    (if 3 ≤ root552Candidates0.length then search edge 3 [4019, 4211] root552Candidates0 8230 winner else winner) = _
  rw [if_neg (by decide)]
def root552Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root552Candidates1_checked : candidates edge 4 4019 5471 root552Tail2 winner.bound = root552Candidates1 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root552Step1 : scan edge (search edge 3) 4 [4019] 4019 root552Tail1 winner =
    scan edge (search edge 3) 4 [4019] 4019 root552Tail2 winner := by
  rw [root552Tail1, scan, if_neg (by decide), root552Candidates1_checked]
  change scan edge (search edge 3) 4 [4019] 4019 root552Tail2
    (if 3 ≤ root552Candidates1.length then search edge 3 [4019, 5471] root552Candidates1 9490 winner else winner) = _
  rw [if_neg (by decide)]
theorem root552Scan2 : scan edge (search edge 3) 4 [4019] 4019 root552Tail2 winner = winner := by rw [root552Tail2, scan, if_pos (by decide)]
theorem root552Scan1 : scan edge (search edge 3) 4 [4019] 4019 root552Tail1 winner = winner :=
  (root552Step1).trans root552Scan2
theorem root552Scan0 : scan edge (search edge 3) 4 [4019] 4019 root552Tail0 winner = winner :=
  (root552Step0).trans root552Scan1
set_option maxRecDepth 4096 in
theorem root552Branch : search edge 4 [4019] root552Candidates 4019 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4019] 4019 root552Tail0 winner = winner
  exact root552Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
