import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root606Candidates : List Nat := Data.row4483
set_option maxRecDepth 8192 in
theorem root606Candidates_checked : candidates edge 5 0 4483 tail607 winner.bound = root606Candidates := by
  apply candidates_eq_row edge 5 0 4483 tail607 root606Candidates winner.bound tail607_ordered
  · decide
  · decide
  · funext q
    rfl
def root606Tail21 : List Nat := []
def root606Tail20 : List Nat := 21493 :: root606Tail21
def root606Tail19 : List Nat := 21319 :: root606Tail20
def root606Tail18 : List Nat := 20407 :: root606Tail19
def root606Tail17 : List Nat := 20011 :: root606Tail18
def root606Tail16 : List Nat := 17959 :: root606Tail17
def root606Tail15 : List Nat := 14221 :: root606Tail16
def root606Tail14 : List Nat := 13963 :: root606Tail15
def root606Tail13 : List Nat := 12289 :: root606Tail14
def root606Tail12 : List Nat := 11923 :: root606Tail13
def root606Tail11 : List Nat := 9463 :: root606Tail12
def root606Tail10 : List Nat := 9049 :: root606Tail11
def root606Tail9 : List Nat := 7759 :: root606Tail10
def root606Tail8 : List Nat := 7717 :: root606Tail9
def root606Tail7 : List Nat := 7537 :: root606Tail8
def root606Tail6 : List Nat := 7489 :: root606Tail7
def root606Tail5 : List Nat := 6793 :: root606Tail6
def root606Tail4 : List Nat := 6607 :: root606Tail5
def root606Tail3 : List Nat := 6577 :: root606Tail4
def root606Tail2 : List Nat := 5737 :: root606Tail3
def root606Tail1 : List Nat := 5347 :: root606Tail2
def root606Tail0 : List Nat := 5113 :: root606Tail1
def root606Candidates0 : List Nat := [6793, 13963]
set_option maxRecDepth 8192 in
theorem root606Candidates0_checked : candidates edge 4 4483 5113 root606Tail1 winner.bound = root606Candidates0 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root606Step0 : scan edge (search edge 3) 4 [4483] 4483 root606Tail0 winner =
    scan edge (search edge 3) 4 [4483] 4483 root606Tail1 winner := by
  rw [root606Tail0, scan, if_neg (by decide), root606Candidates0_checked]
  change scan edge (search edge 3) 4 [4483] 4483 root606Tail1
    (if 3 ≤ root606Candidates0.length then search edge 3 [4483, 5113] root606Candidates0 9596 winner else winner) = _
  rw [if_neg (by decide)]
def root606Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root606Candidates1_checked : candidates edge 4 4483 5347 root606Tail2 winner.bound = root606Candidates1 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root606Step1 : scan edge (search edge 3) 4 [4483] 4483 root606Tail1 winner =
    scan edge (search edge 3) 4 [4483] 4483 root606Tail2 winner := by
  rw [root606Tail1, scan, if_neg (by decide), root606Candidates1_checked]
  change scan edge (search edge 3) 4 [4483] 4483 root606Tail2
    (if 3 ≤ root606Candidates1.length then search edge 3 [4483, 5347] root606Candidates1 9830 winner else winner) = _
  rw [if_neg (by decide)]
theorem root606Scan2 : scan edge (search edge 3) 4 [4483] 4483 root606Tail2 winner = winner := by rw [root606Tail2, scan, if_pos (by decide)]
theorem root606Scan1 : scan edge (search edge 3) 4 [4483] 4483 root606Tail1 winner = winner :=
  (root606Step1).trans root606Scan2
theorem root606Scan0 : scan edge (search edge 3) 4 [4483] 4483 root606Tail0 winner = winner :=
  (root606Step0).trans root606Scan1
set_option maxRecDepth 4096 in
theorem root606Branch : search edge 4 [4483] root606Candidates 4483 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4483] 4483 root606Tail0 winner = winner
  exact root606Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
