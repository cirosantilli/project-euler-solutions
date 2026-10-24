import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root590Candidates : List Nat := Data.row4339
set_option maxRecDepth 8192 in
theorem root590Candidates_checked : candidates edge 5 0 4339 tail591 winner.bound = root590Candidates := by
  apply candidates_eq_row edge 5 0 4339 tail591 root590Candidates winner.bound tail591_ordered
  · decide
  · decide
  · funext q
    rfl
def root590Tail40 : List Nat := []
def root590Tail39 : List Nat := 21613 :: root590Tail40
def root590Tail38 : List Nat := 21559 :: root590Tail39
def root590Tail37 : List Nat := 21193 :: root590Tail38
def root590Tail36 : List Nat := 20551 :: root590Tail37
def root590Tail35 : List Nat := 20101 :: root590Tail36
def root590Tail34 : List Nat := 19417 :: root590Tail35
def root590Tail33 : List Nat := 19309 :: root590Tail34
def root590Tail32 : List Nat := 18787 :: root590Tail33
def root590Tail31 : List Nat := 17431 :: root590Tail32
def root590Tail30 : List Nat := 16657 :: root590Tail31
def root590Tail29 : List Nat := 16369 :: root590Tail30
def root590Tail28 : List Nat := 14563 :: root590Tail29
def root590Tail27 : List Nat := 14479 :: root590Tail28
def root590Tail26 : List Nat := 14083 :: root590Tail27
def root590Tail25 : List Nat := 13681 :: root590Tail26
def root590Tail24 : List Nat := 13567 :: root590Tail25
def root590Tail23 : List Nat := 12907 :: root590Tail24
def root590Tail22 : List Nat := 12421 :: root590Tail23
def root590Tail21 : List Nat := 12379 :: root590Tail22
def root590Tail20 : List Nat := 11329 :: root590Tail21
def root590Tail19 : List Nat := 11131 :: root590Tail20
def root590Tail18 : List Nat := 11047 :: root590Tail19
def root590Tail17 : List Nat := 10903 :: root590Tail18
def root590Tail16 : List Nat := 10651 :: root590Tail17
def root590Tail15 : List Nat := 10303 :: root590Tail16
def root590Tail14 : List Nat := 9931 :: root590Tail15
def root590Tail13 : List Nat := 9679 :: root590Tail14
def root590Tail12 : List Nat := 9661 :: root590Tail13
def root590Tail11 : List Nat := 9241 :: root590Tail12
def root590Tail10 : List Nat := 8461 :: root590Tail11
def root590Tail9 : List Nat := 8317 :: root590Tail10
def root590Tail8 : List Nat := 8167 :: root590Tail9
def root590Tail7 : List Nat := 7927 :: root590Tail8
def root590Tail6 : List Nat := 7603 :: root590Tail7
def root590Tail5 : List Nat := 6961 :: root590Tail6
def root590Tail4 : List Nat := 5701 :: root590Tail5
def root590Tail3 : List Nat := 4903 :: root590Tail4
def root590Tail2 : List Nat := 4831 :: root590Tail3
def root590Tail1 : List Nat := 4639 :: root590Tail2
def root590Tail0 : List Nat := 4567 :: root590Tail1
def root590Candidates0 : List Nat := [16369]
set_option maxRecDepth 8192 in
theorem root590Candidates0_checked : candidates edge 4 4339 4567 root590Tail1 winner.bound = root590Candidates0 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root590Step0 : scan edge (search edge 3) 4 [4339] 4339 root590Tail0 winner =
    scan edge (search edge 3) 4 [4339] 4339 root590Tail1 winner := by
  rw [root590Tail0, scan, if_neg (by decide), root590Candidates0_checked]
  change scan edge (search edge 3) 4 [4339] 4339 root590Tail1
    (if 3 ≤ root590Candidates0.length then search edge 3 [4339, 4567] root590Candidates0 8906 winner else winner) = _
  rw [if_neg (by decide)]
def root590Candidates1 : List Nat := [9679]
set_option maxRecDepth 8192 in
theorem root590Candidates1_checked : candidates edge 4 4339 4639 root590Tail2 winner.bound = root590Candidates1 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root590Step1 : scan edge (search edge 3) 4 [4339] 4339 root590Tail1 winner =
    scan edge (search edge 3) 4 [4339] 4339 root590Tail2 winner := by
  rw [root590Tail1, scan, if_neg (by decide), root590Candidates1_checked]
  change scan edge (search edge 3) 4 [4339] 4339 root590Tail2
    (if 3 ≤ root590Candidates1.length then search edge 3 [4339, 4639] root590Candidates1 8978 winner else winner) = _
  rw [if_neg (by decide)]
def root590Candidates2 : List Nat := [13567]
set_option maxRecDepth 8192 in
theorem root590Candidates2_checked : candidates edge 4 4339 4831 root590Tail3 winner.bound = root590Candidates2 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root590Step2 : scan edge (search edge 3) 4 [4339] 4339 root590Tail2 winner =
    scan edge (search edge 3) 4 [4339] 4339 root590Tail3 winner := by
  rw [root590Tail2, scan, if_neg (by decide), root590Candidates2_checked]
  change scan edge (search edge 3) 4 [4339] 4339 root590Tail3
    (if 3 ≤ root590Candidates2.length then search edge 3 [4339, 4831] root590Candidates2 9170 winner else winner) = _
  rw [if_neg (by decide)]
def root590Candidates3 : List Nat := [12379]
set_option maxRecDepth 8192 in
theorem root590Candidates3_checked : candidates edge 4 4339 4903 root590Tail4 winner.bound = root590Candidates3 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root590Step3 : scan edge (search edge 3) 4 [4339] 4339 root590Tail3 winner =
    scan edge (search edge 3) 4 [4339] 4339 root590Tail4 winner := by
  rw [root590Tail3, scan, if_neg (by decide), root590Candidates3_checked]
  change scan edge (search edge 3) 4 [4339] 4339 root590Tail4
    (if 3 ≤ root590Candidates3.length then search edge 3 [4339, 4903] root590Candidates3 9242 winner else winner) = _
  rw [if_neg (by decide)]
theorem root590Scan4 : scan edge (search edge 3) 4 [4339] 4339 root590Tail4 winner = winner := by rw [root590Tail4, scan, if_pos (by decide)]
theorem root590Scan3 : scan edge (search edge 3) 4 [4339] 4339 root590Tail3 winner = winner :=
  (root590Step3).trans root590Scan4
theorem root590Scan2 : scan edge (search edge 3) 4 [4339] 4339 root590Tail2 winner = winner :=
  (root590Step2).trans root590Scan3
theorem root590Scan1 : scan edge (search edge 3) 4 [4339] 4339 root590Tail1 winner = winner :=
  (root590Step1).trans root590Scan2
theorem root590Scan0 : scan edge (search edge 3) 4 [4339] 4339 root590Tail0 winner = winner :=
  (root590Step0).trans root590Scan1
set_option maxRecDepth 4096 in
theorem root590Branch : search edge 4 [4339] root590Candidates 4339 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4339] 4339 root590Tail0 winner = winner
  exact root590Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
