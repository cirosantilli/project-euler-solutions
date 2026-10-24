import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root473Candidates : List Nat := Data.row3373
set_option maxRecDepth 8192 in
theorem root473Candidates_checked : candidates edge 5 0 3373 tail474 winner.bound = root473Candidates := by
  apply candidates_eq_row edge 5 0 3373 tail474 root473Candidates winner.bound tail474_ordered
  · decide
  · decide
  · funext q
    rfl
def root473Tail34 : List Nat := []
def root473Tail33 : List Nat := 21727 :: root473Tail34
def root473Tail32 : List Nat := 20023 :: root473Tail33
def root473Tail31 : List Nat := 19603 :: root473Tail32
def root473Tail30 : List Nat := 19471 :: root473Tail31
def root473Tail29 : List Nat := 19009 :: root473Tail30
def root473Tail28 : List Nat := 18493 :: root473Tail29
def root473Tail27 : List Nat := 18481 :: root473Tail28
def root473Tail26 : List Nat := 18313 :: root473Tail27
def root473Tail25 : List Nat := 16069 :: root473Tail26
def root473Tail24 : List Nat := 14083 :: root473Tail25
def root473Tail23 : List Nat := 13807 :: root473Tail24
def root473Tail22 : List Nat := 13681 :: root473Tail23
def root473Tail21 : List Nat := 13381 :: root473Tail22
def root473Tail20 : List Nat := 12739 :: root473Tail21
def root473Tail19 : List Nat := 12697 :: root473Tail20
def root473Tail18 : List Nat := 12583 :: root473Tail19
def root473Tail17 : List Nat := 12211 :: root473Tail18
def root473Tail16 : List Nat := 11287 :: root473Tail17
def root473Tail15 : List Nat := 10657 :: root473Tail16
def root473Tail14 : List Nat := 9103 :: root473Tail15
def root473Tail13 : List Nat := 8923 :: root473Tail14
def root473Tail12 : List Nat := 8233 :: root473Tail13
def root473Tail11 : List Nat := 8161 :: root473Tail12
def root473Tail10 : List Nat := 7621 :: root473Tail11
def root473Tail9 : List Nat := 6907 :: root473Tail10
def root473Tail8 : List Nat := 6793 :: root473Tail9
def root473Tail7 : List Nat := 6247 :: root473Tail8
def root473Tail6 : List Nat := 6067 :: root473Tail7
def root473Tail5 : List Nat := 5881 :: root473Tail6
def root473Tail4 : List Nat := 5557 :: root473Tail5
def root473Tail3 : List Nat := 4957 :: root473Tail4
def root473Tail2 : List Nat := 4621 :: root473Tail3
def root473Tail1 : List Nat := 4339 :: root473Tail2
def root473Tail0 : List Nat := 3631 :: root473Tail1
def root473Candidates0 : List Nat := [8923, 14083]
set_option maxRecDepth 8192 in
theorem root473Candidates0_checked : candidates edge 4 3373 3631 root473Tail1 winner.bound = root473Candidates0 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root473Step0 : scan edge (search edge 3) 4 [3373] 3373 root473Tail0 winner =
    scan edge (search edge 3) 4 [3373] 3373 root473Tail1 winner := by
  rw [root473Tail0, scan, if_neg (by decide), root473Candidates0_checked]
  change scan edge (search edge 3) 4 [3373] 3373 root473Tail1
    (if 3 ≤ root473Candidates0.length then search edge 3 [3373, 3631] root473Candidates0 7004 winner else winner) = _
  rw [if_neg (by decide)]
def root473Candidates1 : List Nat := [13681, 14083]
set_option maxRecDepth 8192 in
theorem root473Candidates1_checked : candidates edge 4 3373 4339 root473Tail2 winner.bound = root473Candidates1 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root473Step1 : scan edge (search edge 3) 4 [3373] 3373 root473Tail1 winner =
    scan edge (search edge 3) 4 [3373] 3373 root473Tail2 winner := by
  rw [root473Tail1, scan, if_neg (by decide), root473Candidates1_checked]
  change scan edge (search edge 3) 4 [3373] 3373 root473Tail2
    (if 3 ≤ root473Candidates1.length then search edge 3 [3373, 4339] root473Candidates1 7712 winner else winner) = _
  rw [if_neg (by decide)]
def root473Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root473Candidates2_checked : candidates edge 4 3373 4621 root473Tail3 winner.bound = root473Candidates2 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root473Step2 : scan edge (search edge 3) 4 [3373] 3373 root473Tail2 winner =
    scan edge (search edge 3) 4 [3373] 3373 root473Tail3 winner := by
  rw [root473Tail2, scan, if_neg (by decide), root473Candidates2_checked]
  change scan edge (search edge 3) 4 [3373] 3373 root473Tail3
    (if 3 ≤ root473Candidates2.length then search edge 3 [3373, 4621] root473Candidates2 7994 winner else winner) = _
  rw [if_neg (by decide)]
def root473Candidates3 : List Nat := [8233, 12697, 16069]
set_option maxRecDepth 8192 in
theorem root473Candidates3_checked : candidates edge 4 3373 4957 root473Tail4 winner.bound = root473Candidates3 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root473Child3 : search edge 3 [3373, 4957] root473Candidates3 8330 winner = winner := by decide
theorem root473Step3 : scan edge (search edge 3) 4 [3373] 3373 root473Tail3 winner =
    scan edge (search edge 3) 4 [3373] 3373 root473Tail4 winner := by
  rw [root473Tail3, scan, if_neg (by decide), root473Candidates3_checked]
  change scan edge (search edge 3) 4 [3373] 3373 root473Tail4
    (if 3 ≤ root473Candidates3.length then search edge 3 [3373, 4957] root473Candidates3 8330 winner else winner) = _
  rw [if_pos (by decide), root473Child3]
def root473Candidates4 : List Nat := [6067, 8233]
set_option maxRecDepth 8192 in
theorem root473Candidates4_checked : candidates edge 4 3373 5557 root473Tail5 winner.bound = root473Candidates4 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root473Step4 : scan edge (search edge 3) 4 [3373] 3373 root473Tail4 winner =
    scan edge (search edge 3) 4 [3373] 3373 root473Tail5 winner := by
  rw [root473Tail4, scan, if_neg (by decide), root473Candidates4_checked]
  change scan edge (search edge 3) 4 [3373] 3373 root473Tail5
    (if 3 ≤ root473Candidates4.length then search edge 3 [3373, 5557] root473Candidates4 8930 winner else winner) = _
  rw [if_neg (by decide)]
theorem root473Scan5 : scan edge (search edge 3) 4 [3373] 3373 root473Tail5 winner = winner := by rw [root473Tail5, scan, if_pos (by decide)]
theorem root473Scan4 : scan edge (search edge 3) 4 [3373] 3373 root473Tail4 winner = winner :=
  (root473Step4).trans root473Scan5
theorem root473Scan3 : scan edge (search edge 3) 4 [3373] 3373 root473Tail3 winner = winner :=
  (root473Step3).trans root473Scan4
theorem root473Scan2 : scan edge (search edge 3) 4 [3373] 3373 root473Tail2 winner = winner :=
  (root473Step2).trans root473Scan3
theorem root473Scan1 : scan edge (search edge 3) 4 [3373] 3373 root473Tail1 winner = winner :=
  (root473Step1).trans root473Scan2
theorem root473Scan0 : scan edge (search edge 3) 4 [3373] 3373 root473Tail0 winner = winner :=
  (root473Step0).trans root473Scan1
set_option maxRecDepth 4096 in
theorem root473Branch : search edge 4 [3373] root473Candidates 3373 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3373] 3373 root473Tail0 winner = winner
  exact root473Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
