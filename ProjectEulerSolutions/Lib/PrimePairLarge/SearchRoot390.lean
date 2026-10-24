import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root390Candidates : List Nat := Data.row2699
set_option maxRecDepth 8192 in
theorem root390Candidates_checked : candidates edge 5 0 2699 tail391 winner.bound = root390Candidates := by
  apply candidates_eq_row edge 5 0 2699 tail391 root390Candidates winner.bound tail391_ordered
  · decide
  · decide
  · funext q
    rfl
def root390Tail36 : List Nat := []
def root390Tail35 : List Nat := 22943 :: root390Tail36
def root390Tail34 : List Nat := 22193 :: root390Tail35
def root390Tail33 : List Nat := 21143 :: root390Tail34
def root390Tail32 : List Nat := 19427 :: root390Tail33
def root390Tail31 : List Nat := 18749 :: root390Tail32
def root390Tail30 : List Nat := 17327 :: root390Tail31
def root390Tail29 : List Nat := 16943 :: root390Tail30
def root390Tail28 : List Nat := 16481 :: root390Tail29
def root390Tail27 : List Nat := 16319 :: root390Tail28
def root390Tail26 : List Nat := 15923 :: root390Tail27
def root390Tail25 : List Nat := 15797 :: root390Tail26
def root390Tail24 : List Nat := 15377 :: root390Tail25
def root390Tail23 : List Nat := 15101 :: root390Tail24
def root390Tail22 : List Nat := 13883 :: root390Tail23
def root390Tail21 : List Nat := 13697 :: root390Tail22
def root390Tail20 : List Nat := 13367 :: root390Tail21
def root390Tail19 : List Nat := 12959 :: root390Tail20
def root390Tail18 : List Nat := 12611 :: root390Tail19
def root390Tail17 : List Nat := 11027 :: root390Tail18
def root390Tail16 : List Nat := 10691 :: root390Tail17
def root390Tail15 : List Nat := 9839 :: root390Tail16
def root390Tail14 : List Nat := 8627 :: root390Tail15
def root390Tail13 : List Nat := 8039 :: root390Tail14
def root390Tail12 : List Nat := 7589 :: root390Tail13
def root390Tail11 : List Nat := 7577 :: root390Tail12
def root390Tail10 : List Nat := 7211 :: root390Tail11
def root390Tail9 : List Nat := 7121 :: root390Tail10
def root390Tail8 : List Nat := 6353 :: root390Tail9
def root390Tail7 : List Nat := 6329 :: root390Tail8
def root390Tail6 : List Nat := 5939 :: root390Tail7
def root390Tail5 : List Nat := 4337 :: root390Tail6
def root390Tail4 : List Nat := 4133 :: root390Tail5
def root390Tail3 : List Nat := 3761 :: root390Tail4
def root390Tail2 : List Nat := 3413 :: root390Tail3
def root390Tail1 : List Nat := 3209 :: root390Tail2
def root390Tail0 : List Nat := 3041 :: root390Tail1
def root390Candidates0 : List Nat := [3209, 4133, 4337, 9839, 16481]
set_option maxRecDepth 8192 in
theorem root390Candidates0_checked : candidates edge 4 2699 3041 root390Tail1 winner.bound = root390Candidates0 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root390Child0 : search edge 3 [2699, 3041] root390Candidates0 5740 winner = winner := by decide
theorem root390Step0 : scan edge (search edge 3) 4 [2699] 2699 root390Tail0 winner =
    scan edge (search edge 3) 4 [2699] 2699 root390Tail1 winner := by
  rw [root390Tail0, scan, if_neg (by decide), root390Candidates0_checked]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail1
    (if 3 ≤ root390Candidates0.length then search edge 3 [2699, 3041] root390Candidates0 5740 winner else winner) = _
  rw [if_pos (by decide), root390Child0]
def root390Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root390Candidates1_checked : candidates edge 4 2699 3209 root390Tail2 winner.bound = root390Candidates1 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root390Step1 : scan edge (search edge 3) 4 [2699] 2699 root390Tail1 winner =
    scan edge (search edge 3) 4 [2699] 2699 root390Tail2 winner := by
  rw [root390Tail1, scan, if_neg (by decide), root390Candidates1_checked]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail2
    (if 3 ≤ root390Candidates1.length then search edge 3 [2699, 3209] root390Candidates1 5908 winner else winner) = _
  rw [if_neg (by decide)]
def root390Candidates2 : List Nat := [3761, 5939, 7211, 11027]
set_option maxRecDepth 8192 in
theorem root390Candidates2_checked : candidates edge 4 2699 3413 root390Tail3 winner.bound = root390Candidates2 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root390Child2 : search edge 3 [2699, 3413] root390Candidates2 6112 winner = winner := by decide
theorem root390Step2 : scan edge (search edge 3) 4 [2699] 2699 root390Tail2 winner =
    scan edge (search edge 3) 4 [2699] 2699 root390Tail3 winner := by
  rw [root390Tail2, scan, if_neg (by decide), root390Candidates2_checked]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail3
    (if 3 ≤ root390Candidates2.length then search edge 3 [2699, 3413] root390Candidates2 6112 winner else winner) = _
  rw [if_pos (by decide), root390Child2]
def root390Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root390Candidates3_checked : candidates edge 4 2699 3761 root390Tail4 winner.bound = root390Candidates3 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root390Step3 : scan edge (search edge 3) 4 [2699] 2699 root390Tail3 winner =
    scan edge (search edge 3) 4 [2699] 2699 root390Tail4 winner := by
  rw [root390Tail3, scan, if_neg (by decide), root390Candidates3_checked]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail4
    (if 3 ≤ root390Candidates3.length then search edge 3 [2699, 3761] root390Candidates3 6460 winner else winner) = _
  rw [if_neg (by decide)]
def root390Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root390Candidates4_checked : candidates edge 4 2699 4133 root390Tail5 winner.bound = root390Candidates4 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root390Step4 : scan edge (search edge 3) 4 [2699] 2699 root390Tail4 winner =
    scan edge (search edge 3) 4 [2699] 2699 root390Tail5 winner := by
  rw [root390Tail4, scan, if_neg (by decide), root390Candidates4_checked]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail5
    (if 3 ≤ root390Candidates4.length then search edge 3 [2699, 4133] root390Candidates4 6832 winner else winner) = _
  rw [if_neg (by decide)]
def root390Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root390Candidates5_checked : candidates edge 4 2699 4337 root390Tail6 winner.bound = root390Candidates5 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root390Step5 : scan edge (search edge 3) 4 [2699] 2699 root390Tail5 winner =
    scan edge (search edge 3) 4 [2699] 2699 root390Tail6 winner := by
  rw [root390Tail5, scan, if_neg (by decide), root390Candidates5_checked]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail6
    (if 3 ≤ root390Candidates5.length then search edge 3 [2699, 4337] root390Candidates5 7036 winner else winner) = _
  rw [if_neg (by decide)]
theorem root390Scan6 : scan edge (search edge 3) 4 [2699] 2699 root390Tail6 winner = winner := by rw [root390Tail6, scan, if_pos (by decide)]
theorem root390Scan5 : scan edge (search edge 3) 4 [2699] 2699 root390Tail5 winner = winner :=
  (root390Step5).trans root390Scan6
theorem root390Scan4 : scan edge (search edge 3) 4 [2699] 2699 root390Tail4 winner = winner :=
  (root390Step4).trans root390Scan5
theorem root390Scan3 : scan edge (search edge 3) 4 [2699] 2699 root390Tail3 winner = winner :=
  (root390Step3).trans root390Scan4
theorem root390Scan2 : scan edge (search edge 3) 4 [2699] 2699 root390Tail2 winner = winner :=
  (root390Step2).trans root390Scan3
theorem root390Scan1 : scan edge (search edge 3) 4 [2699] 2699 root390Tail1 winner = winner :=
  (root390Step1).trans root390Scan2
theorem root390Scan0 : scan edge (search edge 3) 4 [2699] 2699 root390Tail0 winner = winner :=
  (root390Step0).trans root390Scan1
set_option maxRecDepth 4096 in
theorem root390Branch : search edge 4 [2699] root390Candidates 2699 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2699] 2699 root390Tail0 winner = winner
  exact root390Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
