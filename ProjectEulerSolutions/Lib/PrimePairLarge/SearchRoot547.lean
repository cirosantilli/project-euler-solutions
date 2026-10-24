import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root547Candidates : List Nat := Data.row3989
set_option maxRecDepth 8192 in
theorem root547Candidates_checked : candidates edge 5 0 3989 tail548 winner.bound = root547Candidates := by
  apply candidates_eq_row edge 5 0 3989 tail548 root547Candidates winner.bound tail548_ordered
  · decide
  · decide
  · funext q
    rfl
def root547Tail35 : List Nat := []
def root547Tail34 : List Nat := 21491 :: root547Tail35
def root547Tail33 : List Nat := 20411 :: root547Tail34
def root547Tail32 : List Nat := 20261 :: root547Tail33
def root547Tail31 : List Nat := 20231 :: root547Tail32
def root547Tail30 : List Nat := 20177 :: root547Tail31
def root547Tail29 : List Nat := 19403 :: root547Tail30
def root547Tail28 : List Nat := 19289 :: root547Tail29
def root547Tail27 : List Nat := 17921 :: root547Tail28
def root547Tail26 : List Nat := 17573 :: root547Tail27
def root547Tail25 : List Nat := 15791 :: root547Tail26
def root547Tail24 : List Nat := 14759 :: root547Tail25
def root547Tail23 : List Nat := 14207 :: root547Tail24
def root547Tail22 : List Nat := 13901 :: root547Tail23
def root547Tail21 : List Nat := 13469 :: root547Tail22
def root547Tail20 : List Nat := 11633 :: root547Tail21
def root547Tail19 : List Nat := 10133 :: root547Tail20
def root547Tail18 : List Nat := 10037 :: root547Tail19
def root547Tail17 : List Nat := 9623 :: root547Tail18
def root547Tail16 : List Nat := 9491 :: root547Tail17
def root547Tail15 : List Nat := 9311 :: root547Tail16
def root547Tail14 : List Nat := 9239 :: root547Tail15
def root547Tail13 : List Nat := 8933 :: root547Tail14
def root547Tail12 : List Nat := 8861 :: root547Tail13
def root547Tail11 : List Nat := 8597 :: root547Tail12
def root547Tail10 : List Nat := 8513 :: root547Tail11
def root547Tail9 : List Nat := 8291 :: root547Tail10
def root547Tail8 : List Nat := 8093 :: root547Tail9
def root547Tail7 : List Nat := 7703 :: root547Tail8
def root547Tail6 : List Nat := 7043 :: root547Tail7
def root547Tail5 : List Nat := 6719 :: root547Tail6
def root547Tail4 : List Nat := 6581 :: root547Tail5
def root547Tail3 : List Nat := 5927 :: root547Tail4
def root547Tail2 : List Nat := 5783 :: root547Tail3
def root547Tail1 : List Nat := 4931 :: root547Tail2
def root547Tail0 : List Nat := 4157 :: root547Tail1
def root547Candidates0 : List Nat := [7043]
set_option maxRecDepth 8192 in
theorem root547Candidates0_checked : candidates edge 4 3989 4157 root547Tail1 winner.bound = root547Candidates0 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root547Step0 : scan edge (search edge 3) 4 [3989] 3989 root547Tail0 winner =
    scan edge (search edge 3) 4 [3989] 3989 root547Tail1 winner := by
  rw [root547Tail0, scan, if_neg (by decide), root547Candidates0_checked]
  change scan edge (search edge 3) 4 [3989] 3989 root547Tail1
    (if 3 ≤ root547Candidates0.length then search edge 3 [3989, 4157] root547Candidates0 8146 winner else winner) = _
  rw [if_neg (by decide)]
def root547Candidates1 : List Nat := [7703]
set_option maxRecDepth 8192 in
theorem root547Candidates1_checked : candidates edge 4 3989 4931 root547Tail2 winner.bound = root547Candidates1 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root547Step1 : scan edge (search edge 3) 4 [3989] 3989 root547Tail1 winner =
    scan edge (search edge 3) 4 [3989] 3989 root547Tail2 winner := by
  rw [root547Tail1, scan, if_neg (by decide), root547Candidates1_checked]
  change scan edge (search edge 3) 4 [3989] 3989 root547Tail2
    (if 3 ≤ root547Candidates1.length then search edge 3 [3989, 4931] root547Candidates1 8920 winner else winner) = _
  rw [if_neg (by decide)]
theorem root547Scan2 : scan edge (search edge 3) 4 [3989] 3989 root547Tail2 winner = winner := by rw [root547Tail2, scan, if_pos (by decide)]
theorem root547Scan1 : scan edge (search edge 3) 4 [3989] 3989 root547Tail1 winner = winner :=
  (root547Step1).trans root547Scan2
theorem root547Scan0 : scan edge (search edge 3) 4 [3989] 3989 root547Tail0 winner = winner :=
  (root547Step0).trans root547Scan1
set_option maxRecDepth 4096 in
theorem root547Branch : search edge 4 [3989] root547Candidates 3989 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3989] 3989 root547Tail0 winner = winner
  exact root547Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
