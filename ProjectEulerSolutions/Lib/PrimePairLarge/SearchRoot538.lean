import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root538Candidates : List Nat := Data.row3911
set_option maxRecDepth 8192 in
theorem root538Candidates_checked : candidates edge 5 0 3911 tail539 winner.bound = root538Candidates := by
  apply candidates_eq_row edge 5 0 3911 tail539 root538Candidates winner.bound tail539_ordered
  · decide
  · decide
  · funext q
    rfl
def root538Tail33 : List Nat := []
def root538Tail32 : List Nat := 21491 :: root538Tail33
def root538Tail31 : List Nat := 20903 :: root538Tail32
def root538Tail30 : List Nat := 18773 :: root538Tail31
def root538Tail29 : List Nat := 18461 :: root538Tail30
def root538Tail28 : List Nat := 17807 :: root538Tail29
def root538Tail27 : List Nat := 16889 :: root538Tail28
def root538Tail26 : List Nat := 15233 :: root538Tail27
def root538Tail25 : List Nat := 14957 :: root538Tail26
def root538Tail24 : List Nat := 14741 :: root538Tail25
def root538Tail23 : List Nat := 14519 :: root538Tail24
def root538Tail22 : List Nat := 14303 :: root538Tail23
def root538Tail21 : List Nat := 13523 :: root538Tail22
def root538Tail20 : List Nat := 13127 :: root538Tail21
def root538Tail19 : List Nat := 12983 :: root538Tail20
def root538Tail18 : List Nat := 12269 :: root538Tail19
def root538Tail17 : List Nat := 11579 :: root538Tail18
def root538Tail16 : List Nat := 10739 :: root538Tail17
def root538Tail15 : List Nat := 10463 :: root538Tail16
def root538Tail14 : List Nat := 9851 :: root538Tail15
def root538Tail13 : List Nat := 9791 :: root538Tail14
def root538Tail12 : List Nat := 9539 :: root538Tail13
def root538Tail11 : List Nat := 9497 :: root538Tail12
def root538Tail10 : List Nat := 9257 :: root538Tail11
def root538Tail9 : List Nat := 8867 :: root538Tail10
def root538Tail8 : List Nat := 7937 :: root538Tail9
def root538Tail7 : List Nat := 7919 :: root538Tail8
def root538Tail6 : List Nat := 7349 :: root538Tail7
def root538Tail5 : List Nat := 6257 :: root538Tail6
def root538Tail4 : List Nat := 5717 :: root538Tail5
def root538Tail3 : List Nat := 5477 :: root538Tail4
def root538Tail2 : List Nat := 4457 :: root538Tail3
def root538Tail1 : List Nat := 4259 :: root538Tail2
def root538Tail0 : List Nat := 4079 :: root538Tail1
def root538Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root538Candidates0_checked : candidates edge 4 3911 4079 root538Tail1 winner.bound = root538Candidates0 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
theorem root538Step0 : scan edge (search edge 3) 4 [3911] 3911 root538Tail0 winner =
    scan edge (search edge 3) 4 [3911] 3911 root538Tail1 winner := by
  rw [root538Tail0, scan, if_neg (by decide), root538Candidates0_checked]
  change scan edge (search edge 3) 4 [3911] 3911 root538Tail1
    (if 3 ≤ root538Candidates0.length then search edge 3 [3911, 4079] root538Candidates0 7990 winner else winner) = _
  rw [if_neg (by decide)]
def root538Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root538Candidates1_checked : candidates edge 4 3911 4259 root538Tail2 winner.bound = root538Candidates1 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root538Step1 : scan edge (search edge 3) 4 [3911] 3911 root538Tail1 winner =
    scan edge (search edge 3) 4 [3911] 3911 root538Tail2 winner := by
  rw [root538Tail1, scan, if_neg (by decide), root538Candidates1_checked]
  change scan edge (search edge 3) 4 [3911] 3911 root538Tail2
    (if 3 ≤ root538Candidates1.length then search edge 3 [3911, 4259] root538Candidates1 8170 winner else winner) = _
  rw [if_neg (by decide)]
def root538Candidates2 : List Nat := [7919, 9791, 11579]
set_option maxRecDepth 8192 in
theorem root538Candidates2_checked : candidates edge 4 3911 4457 root538Tail3 winner.bound = root538Candidates2 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root538Child2 : search edge 3 [3911, 4457] root538Candidates2 8368 winner = winner := by decide
theorem root538Step2 : scan edge (search edge 3) 4 [3911] 3911 root538Tail2 winner =
    scan edge (search edge 3) 4 [3911] 3911 root538Tail3 winner := by
  rw [root538Tail2, scan, if_neg (by decide), root538Candidates2_checked]
  change scan edge (search edge 3) 4 [3911] 3911 root538Tail3
    (if 3 ≤ root538Candidates2.length then search edge 3 [3911, 4457] root538Candidates2 8368 winner else winner) = _
  rw [if_pos (by decide), root538Child2]
def root538Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root538Candidates3_checked : candidates edge 4 3911 5477 root538Tail4 winner.bound = root538Candidates3 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root538Step3 : scan edge (search edge 3) 4 [3911] 3911 root538Tail3 winner =
    scan edge (search edge 3) 4 [3911] 3911 root538Tail4 winner := by
  rw [root538Tail3, scan, if_neg (by decide), root538Candidates3_checked]
  change scan edge (search edge 3) 4 [3911] 3911 root538Tail4
    (if 3 ≤ root538Candidates3.length then search edge 3 [3911, 5477] root538Candidates3 9388 winner else winner) = _
  rw [if_neg (by decide)]
theorem root538Scan4 : scan edge (search edge 3) 4 [3911] 3911 root538Tail4 winner = winner := by rw [root538Tail4, scan, if_pos (by decide)]
theorem root538Scan3 : scan edge (search edge 3) 4 [3911] 3911 root538Tail3 winner = winner :=
  (root538Step3).trans root538Scan4
theorem root538Scan2 : scan edge (search edge 3) 4 [3911] 3911 root538Tail2 winner = winner :=
  (root538Step2).trans root538Scan3
theorem root538Scan1 : scan edge (search edge 3) 4 [3911] 3911 root538Tail1 winner = winner :=
  (root538Step1).trans root538Scan2
theorem root538Scan0 : scan edge (search edge 3) 4 [3911] 3911 root538Tail0 winner = winner :=
  (root538Step0).trans root538Scan1
set_option maxRecDepth 4096 in
theorem root538Branch : search edge 4 [3911] root538Candidates 3911 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3911] 3911 root538Tail0 winner = winner
  exact root538Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
