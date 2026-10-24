import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root585Candidates : List Nat := Data.row4283
set_option maxRecDepth 8192 in
theorem root585Candidates_checked : candidates edge 5 0 4283 tail586 winner.bound = root585Candidates := by
  apply candidates_eq_row edge 5 0 4283 tail586 root585Candidates winner.bound tail586_ordered
  · decide
  · decide
  · funext q
    rfl
def root585Tail24 : List Nat := []
def root585Tail23 : List Nat := 21701 :: root585Tail24
def root585Tail22 : List Nat := 20483 :: root585Tail23
def root585Tail21 : List Nat := 19421 :: root585Tail22
def root585Tail20 : List Nat := 19391 :: root585Tail21
def root585Tail19 : List Nat := 19379 :: root585Tail20
def root585Tail18 : List Nat := 18287 :: root585Tail19
def root585Tail17 : List Nat := 18089 :: root585Tail18
def root585Tail16 : List Nat := 17471 :: root585Tail17
def root585Tail15 : List Nat := 16901 :: root585Tail16
def root585Tail14 : List Nat := 15359 :: root585Tail15
def root585Tail13 : List Nat := 14657 :: root585Tail14
def root585Tail12 : List Nat := 14423 :: root585Tail13
def root585Tail11 : List Nat := 13241 :: root585Tail12
def root585Tail10 : List Nat := 10613 :: root585Tail11
def root585Tail9 : List Nat := 10343 :: root585Tail10
def root585Tail8 : List Nat := 9833 :: root585Tail9
def root585Tail7 : List Nat := 8447 :: root585Tail8
def root585Tail6 : List Nat := 8273 :: root585Tail7
def root585Tail5 : List Nat := 8093 :: root585Tail6
def root585Tail4 : List Nat := 8069 :: root585Tail5
def root585Tail3 : List Nat := 7103 :: root585Tail4
def root585Tail2 : List Nat := 5669 :: root585Tail3
def root585Tail1 : List Nat := 4523 :: root585Tail2
def root585Tail0 : List Nat := 4289 :: root585Tail1
def root585Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root585Candidates0_checked : candidates edge 4 4283 4289 root585Tail1 winner.bound = root585Candidates0 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root585Step0 : scan edge (search edge 3) 4 [4283] 4283 root585Tail0 winner =
    scan edge (search edge 3) 4 [4283] 4283 root585Tail1 winner := by
  rw [root585Tail0, scan, if_neg (by decide), root585Candidates0_checked]
  change scan edge (search edge 3) 4 [4283] 4283 root585Tail1
    (if 3 ≤ root585Candidates0.length then search edge 3 [4283, 4289] root585Candidates0 8572 winner else winner) = _
  rw [if_neg (by decide)]
def root585Candidates1 : List Nat := [8093]
set_option maxRecDepth 8192 in
theorem root585Candidates1_checked : candidates edge 4 4283 4523 root585Tail2 winner.bound = root585Candidates1 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root585Step1 : scan edge (search edge 3) 4 [4283] 4283 root585Tail1 winner =
    scan edge (search edge 3) 4 [4283] 4283 root585Tail2 winner := by
  rw [root585Tail1, scan, if_neg (by decide), root585Candidates1_checked]
  change scan edge (search edge 3) 4 [4283] 4283 root585Tail2
    (if 3 ≤ root585Candidates1.length then search edge 3 [4283, 4523] root585Candidates1 8806 winner else winner) = _
  rw [if_neg (by decide)]
theorem root585Scan2 : scan edge (search edge 3) 4 [4283] 4283 root585Tail2 winner = winner := by rw [root585Tail2, scan, if_pos (by decide)]
theorem root585Scan1 : scan edge (search edge 3) 4 [4283] 4283 root585Tail1 winner = winner :=
  (root585Step1).trans root585Scan2
theorem root585Scan0 : scan edge (search edge 3) 4 [4283] 4283 root585Tail0 winner = winner :=
  (root585Step0).trans root585Scan1
set_option maxRecDepth 4096 in
theorem root585Branch : search edge 4 [4283] root585Candidates 4283 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4283] 4283 root585Tail0 winner = winner
  exact root585Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
