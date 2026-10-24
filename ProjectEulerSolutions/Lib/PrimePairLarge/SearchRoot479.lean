import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root479Candidates : List Nat := Data.row3449
set_option maxRecDepth 8192 in
theorem root479Candidates_checked : candidates edge 5 0 3449 tail480 winner.bound = root479Candidates := by
  apply candidates_eq_row edge 5 0 3449 tail480 root479Candidates winner.bound tail480_ordered
  · decide
  · decide
  · funext q
    rfl
def root479Tail30 : List Nat := []
def root479Tail29 : List Nat := 22157 :: root479Tail30
def root479Tail28 : List Nat := 21107 :: root479Tail29
def root479Tail27 : List Nat := 20963 :: root479Tail28
def root479Tail26 : List Nat := 20063 :: root479Tail27
def root479Tail25 : List Nat := 19403 :: root479Tail26
def root479Tail24 : List Nat := 18713 :: root479Tail25
def root479Tail23 : List Nat := 18443 :: root479Tail24
def root479Tail22 : List Nat := 16787 :: root479Tail23
def root479Tail21 : List Nat := 16493 :: root479Tail22
def root479Tail20 : List Nat := 16361 :: root479Tail21
def root479Tail19 : List Nat := 15359 :: root479Tail20
def root479Tail18 : List Nat := 14699 :: root479Tail19
def root479Tail17 : List Nat := 14639 :: root479Tail18
def root479Tail16 : List Nat := 14369 :: root479Tail17
def root479Tail15 : List Nat := 13649 :: root479Tail16
def root479Tail14 : List Nat := 13229 :: root479Tail15
def root479Tail13 : List Nat := 11117 :: root479Tail14
def root479Tail12 : List Nat := 10799 :: root479Tail13
def root479Tail11 : List Nat := 10193 :: root479Tail12
def root479Tail10 : List Nat := 8969 :: root479Tail11
def root479Tail9 : List Nat := 8861 :: root479Tail10
def root479Tail8 : List Nat := 8111 :: root479Tail9
def root479Tail7 : List Nat := 7817 :: root479Tail8
def root479Tail6 : List Nat := 7457 :: root479Tail7
def root479Tail5 : List Nat := 7433 :: root479Tail6
def root479Tail4 : List Nat := 6359 :: root479Tail5
def root479Tail3 : List Nat := 5693 :: root479Tail4
def root479Tail2 : List Nat := 5501 :: root479Tail3
def root479Tail1 : List Nat := 4451 :: root479Tail2
def root479Tail0 : List Nat := 4013 :: root479Tail1
def root479Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root479Candidates0_checked : candidates edge 4 3449 4013 root479Tail1 winner.bound = root479Candidates0 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root479Step0 : scan edge (search edge 3) 4 [3449] 3449 root479Tail0 winner =
    scan edge (search edge 3) 4 [3449] 3449 root479Tail1 winner := by
  rw [root479Tail0, scan, if_neg (by decide), root479Candidates0_checked]
  change scan edge (search edge 3) 4 [3449] 3449 root479Tail1
    (if 3 ≤ root479Candidates0.length then search edge 3 [3449, 4013] root479Candidates0 7462 winner else winner) = _
  rw [if_neg (by decide)]
def root479Candidates1 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root479Candidates1_checked : candidates edge 4 3449 4451 root479Tail2 winner.bound = root479Candidates1 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
theorem root479Step1 : scan edge (search edge 3) 4 [3449] 3449 root479Tail1 winner =
    scan edge (search edge 3) 4 [3449] 3449 root479Tail2 winner := by
  rw [root479Tail1, scan, if_neg (by decide), root479Candidates1_checked]
  change scan edge (search edge 3) 4 [3449] 3449 root479Tail2
    (if 3 ≤ root479Candidates1.length then search edge 3 [3449, 4451] root479Candidates1 7900 winner else winner) = _
  rw [if_neg (by decide)]
def root479Candidates2 : List Nat := [6359, 11117]
set_option maxRecDepth 8192 in
theorem root479Candidates2_checked : candidates edge 4 3449 5501 root479Tail3 winner.bound = root479Candidates2 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root479Step2 : scan edge (search edge 3) 4 [3449] 3449 root479Tail2 winner =
    scan edge (search edge 3) 4 [3449] 3449 root479Tail3 winner := by
  rw [root479Tail2, scan, if_neg (by decide), root479Candidates2_checked]
  change scan edge (search edge 3) 4 [3449] 3449 root479Tail3
    (if 3 ≤ root479Candidates2.length then search edge 3 [3449, 5501] root479Candidates2 8950 winner else winner) = _
  rw [if_neg (by decide)]
theorem root479Scan3 : scan edge (search edge 3) 4 [3449] 3449 root479Tail3 winner = winner := by rw [root479Tail3, scan, if_pos (by decide)]
theorem root479Scan2 : scan edge (search edge 3) 4 [3449] 3449 root479Tail2 winner = winner :=
  (root479Step2).trans root479Scan3
theorem root479Scan1 : scan edge (search edge 3) 4 [3449] 3449 root479Tail1 winner = winner :=
  (root479Step1).trans root479Scan2
theorem root479Scan0 : scan edge (search edge 3) 4 [3449] 3449 root479Tail0 winner = winner :=
  (root479Step0).trans root479Scan1
set_option maxRecDepth 4096 in
theorem root479Branch : search edge 4 [3449] root479Candidates 3449 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3449] 3449 root479Tail0 winner = winner
  exact root479Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
