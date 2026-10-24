import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root477Candidates : List Nat := Data.row3413
set_option maxRecDepth 8192 in
theorem root477Candidates_checked : candidates edge 5 0 3413 tail478 winner.bound = root477Candidates := by
  apply candidates_eq_row edge 5 0 3413 tail478 root477Candidates winner.bound tail478_ordered
  · decide
  · decide
  · funext q
    rfl
def root477Tail33 : List Nat := []
def root477Tail32 : List Nat := 19997 :: root477Tail33
def root477Tail31 : List Nat := 18077 :: root477Tail32
def root477Tail30 : List Nat := 17939 :: root477Tail31
def root477Tail29 : List Nat := 17351 :: root477Tail30
def root477Tail28 : List Nat := 16193 :: root477Tail29
def root477Tail27 : List Nat := 16007 :: root477Tail28
def root477Tail26 : List Nat := 15773 :: root477Tail27
def root477Tail25 : List Nat := 15731 :: root477Tail26
def root477Tail24 : List Nat := 14387 :: root477Tail25
def root477Tail23 : List Nat := 13781 :: root477Tail24
def root477Tail22 : List Nat := 13331 :: root477Tail23
def root477Tail21 : List Nat := 12401 :: root477Tail22
def root477Tail20 : List Nat := 12269 :: root477Tail21
def root477Tail19 : List Nat := 12239 :: root477Tail20
def root477Tail18 : List Nat := 11483 :: root477Tail19
def root477Tail17 : List Nat := 11027 :: root477Tail18
def root477Tail16 : List Nat := 10559 :: root477Tail17
def root477Tail15 : List Nat := 9461 :: root477Tail16
def root477Tail14 : List Nat := 9173 :: root477Tail15
def root477Tail13 : List Nat := 8513 :: root477Tail14
def root477Tail12 : List Nat := 7949 :: root477Tail13
def root477Tail11 : List Nat := 7211 :: root477Tail12
def root477Tail10 : List Nat := 7043 :: root477Tail11
def root477Tail9 : List Nat := 7001 :: root477Tail10
def root477Tail8 : List Nat := 6689 :: root477Tail9
def root477Tail7 : List Nat := 6653 :: root477Tail8
def root477Tail6 : List Nat := 6449 :: root477Tail7
def root477Tail5 : List Nat := 6101 :: root477Tail6
def root477Tail4 : List Nat := 5939 :: root477Tail5
def root477Tail3 : List Nat := 5009 :: root477Tail4
def root477Tail2 : List Nat := 4649 :: root477Tail3
def root477Tail1 : List Nat := 3929 :: root477Tail2
def root477Tail0 : List Nat := 3761 :: root477Tail1
def root477Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root477Candidates0_checked : candidates edge 4 3413 3761 root477Tail1 winner.bound = root477Candidates0 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root477Step0 : scan edge (search edge 3) 4 [3413] 3413 root477Tail0 winner =
    scan edge (search edge 3) 4 [3413] 3413 root477Tail1 winner := by
  rw [root477Tail0, scan, if_neg (by decide), root477Candidates0_checked]
  change scan edge (search edge 3) 4 [3413] 3413 root477Tail1
    (if 3 ≤ root477Candidates0.length then search edge 3 [3413, 3761] root477Candidates0 7174 winner else winner) = _
  rw [if_neg (by decide)]
def root477Candidates1 : List Nat := [7043]
set_option maxRecDepth 8192 in
theorem root477Candidates1_checked : candidates edge 4 3413 3929 root477Tail2 winner.bound = root477Candidates1 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root477Step1 : scan edge (search edge 3) 4 [3413] 3413 root477Tail1 winner =
    scan edge (search edge 3) 4 [3413] 3413 root477Tail2 winner := by
  rw [root477Tail1, scan, if_neg (by decide), root477Candidates1_checked]
  change scan edge (search edge 3) 4 [3413] 3413 root477Tail2
    (if 3 ≤ root477Candidates1.length then search edge 3 [3413, 3929] root477Candidates1 7342 winner else winner) = _
  rw [if_neg (by decide)]
def root477Candidates2 : List Nat := [9461]
set_option maxRecDepth 8192 in
theorem root477Candidates2_checked : candidates edge 4 3413 4649 root477Tail3 winner.bound = root477Candidates2 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root477Step2 : scan edge (search edge 3) 4 [3413] 3413 root477Tail2 winner =
    scan edge (search edge 3) 4 [3413] 3413 root477Tail3 winner := by
  rw [root477Tail2, scan, if_neg (by decide), root477Candidates2_checked]
  change scan edge (search edge 3) 4 [3413] 3413 root477Tail3
    (if 3 ≤ root477Candidates2.length then search edge 3 [3413, 4649] root477Candidates2 8062 winner else winner) = _
  rw [if_neg (by decide)]
def root477Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root477Candidates3_checked : candidates edge 4 3413 5009 root477Tail4 winner.bound = root477Candidates3 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root477Step3 : scan edge (search edge 3) 4 [3413] 3413 root477Tail3 winner =
    scan edge (search edge 3) 4 [3413] 3413 root477Tail4 winner := by
  rw [root477Tail3, scan, if_neg (by decide), root477Candidates3_checked]
  change scan edge (search edge 3) 4 [3413] 3413 root477Tail4
    (if 3 ≤ root477Candidates3.length then search edge 3 [3413, 5009] root477Candidates3 8422 winner else winner) = _
  rw [if_neg (by decide)]
theorem root477Scan4 : scan edge (search edge 3) 4 [3413] 3413 root477Tail4 winner = winner := by rw [root477Tail4, scan, if_pos (by decide)]
theorem root477Scan3 : scan edge (search edge 3) 4 [3413] 3413 root477Tail3 winner = winner :=
  (root477Step3).trans root477Scan4
theorem root477Scan2 : scan edge (search edge 3) 4 [3413] 3413 root477Tail2 winner = winner :=
  (root477Step2).trans root477Scan3
theorem root477Scan1 : scan edge (search edge 3) 4 [3413] 3413 root477Tail1 winner = winner :=
  (root477Step1).trans root477Scan2
theorem root477Scan0 : scan edge (search edge 3) 4 [3413] 3413 root477Tail0 winner = winner :=
  (root477Step0).trans root477Scan1
set_option maxRecDepth 4096 in
theorem root477Branch : search edge 4 [3413] root477Candidates 3413 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3413] 3413 root477Tail0 winner = winner
  exact root477Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
