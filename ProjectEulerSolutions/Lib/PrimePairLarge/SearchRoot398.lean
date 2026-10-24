import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root398Candidates : List Nat := Data.row2749
set_option maxRecDepth 8192 in
theorem root398Candidates_checked : candidates edge 5 0 2749 tail399 winner.bound = root398Candidates := by
  apply candidates_eq_row edge 5 0 2749 tail399 root398Candidates winner.bound tail399_ordered
  · decide
  · decide
  · funext q
    rfl
def root398Tail37 : List Nat := []
def root398Tail36 : List Nat := 23197 :: root398Tail37
def root398Tail35 : List Nat := 22993 :: root398Tail36
def root398Tail34 : List Nat := 22741 :: root398Tail35
def root398Tail33 : List Nat := 22567 :: root398Tail34
def root398Tail32 : List Nat := 22051 :: root398Tail33
def root398Tail31 : List Nat := 21391 :: root398Tail32
def root398Tail30 : List Nat := 20599 :: root398Tail31
def root398Tail29 : List Nat := 19963 :: root398Tail30
def root398Tail28 : List Nat := 19231 :: root398Tail29
def root398Tail27 : List Nat := 18439 :: root398Tail28
def root398Tail26 : List Nat := 18097 :: root398Tail27
def root398Tail25 : List Nat := 17449 :: root398Tail26
def root398Tail24 : List Nat := 17191 :: root398Tail25
def root398Tail23 : List Nat := 15241 :: root398Tail24
def root398Tail22 : List Nat := 14737 :: root398Tail23
def root398Tail21 : List Nat := 14449 :: root398Tail22
def root398Tail20 : List Nat := 14281 :: root398Tail21
def root398Tail19 : List Nat := 14149 :: root398Tail20
def root398Tail18 : List Nat := 13537 :: root398Tail19
def root398Tail17 : List Nat := 13183 :: root398Tail18
def root398Tail16 : List Nat := 12073 :: root398Tail17
def root398Tail15 : List Nat := 11467 :: root398Tail16
def root398Tail14 : List Nat := 10501 :: root398Tail15
def root398Tail13 : List Nat := 10477 :: root398Tail14
def root398Tail12 : List Nat := 9781 :: root398Tail13
def root398Tail11 : List Nat := 9067 :: root398Tail12
def root398Tail10 : List Nat := 8923 :: root398Tail11
def root398Tail9 : List Nat := 8017 :: root398Tail10
def root398Tail8 : List Nat := 7741 :: root398Tail9
def root398Tail7 : List Nat := 7219 :: root398Tail8
def root398Tail6 : List Nat := 6067 :: root398Tail7
def root398Tail5 : List Nat := 4993 :: root398Tail6
def root398Tail4 : List Nat := 4297 :: root398Tail5
def root398Tail3 : List Nat := 4261 :: root398Tail4
def root398Tail2 : List Nat := 4021 :: root398Tail3
def root398Tail1 : List Nat := 3643 :: root398Tail2
def root398Tail0 : List Nat := 3541 :: root398Tail1
def root398Candidates0 : List Nat := [13537]
set_option maxRecDepth 8192 in
theorem root398Candidates0_checked : candidates edge 4 2749 3541 root398Tail1 winner.bound = root398Candidates0 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root398Step0 : scan edge (search edge 3) 4 [2749] 2749 root398Tail0 winner =
    scan edge (search edge 3) 4 [2749] 2749 root398Tail1 winner := by
  rw [root398Tail0, scan, if_neg (by decide), root398Candidates0_checked]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail1
    (if 3 ≤ root398Candidates0.length then search edge 3 [2749, 3541] root398Candidates0 6290 winner else winner) = _
  rw [if_neg (by decide)]
def root398Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root398Candidates1_checked : candidates edge 4 2749 3643 root398Tail2 winner.bound = root398Candidates1 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root398Step1 : scan edge (search edge 3) 4 [2749] 2749 root398Tail1 winner =
    scan edge (search edge 3) 4 [2749] 2749 root398Tail2 winner := by
  rw [root398Tail1, scan, if_neg (by decide), root398Candidates1_checked]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail2
    (if 3 ≤ root398Candidates1.length then search edge 3 [2749, 3643] root398Candidates1 6392 winner else winner) = _
  rw [if_neg (by decide)]
def root398Candidates2 : List Nat := [14737, 18097]
set_option maxRecDepth 8192 in
theorem root398Candidates2_checked : candidates edge 4 2749 4021 root398Tail3 winner.bound = root398Candidates2 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root398Step2 : scan edge (search edge 3) 4 [2749] 2749 root398Tail2 winner =
    scan edge (search edge 3) 4 [2749] 2749 root398Tail3 winner := by
  rw [root398Tail2, scan, if_neg (by decide), root398Candidates2_checked]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail3
    (if 3 ≤ root398Candidates2.length then search edge 3 [2749, 4021] root398Candidates2 6770 winner else winner) = _
  rw [if_neg (by decide)]
def root398Candidates3 : List Nat := [12073]
set_option maxRecDepth 8192 in
theorem root398Candidates3_checked : candidates edge 4 2749 4261 root398Tail4 winner.bound = root398Candidates3 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root398Step3 : scan edge (search edge 3) 4 [2749] 2749 root398Tail3 winner =
    scan edge (search edge 3) 4 [2749] 2749 root398Tail4 winner := by
  rw [root398Tail3, scan, if_neg (by decide), root398Candidates3_checked]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail4
    (if 3 ≤ root398Candidates3.length then search edge 3 [2749, 4261] root398Candidates3 7010 winner else winner) = _
  rw [if_neg (by decide)]
def root398Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root398Candidates4_checked : candidates edge 4 2749 4297 root398Tail5 winner.bound = root398Candidates4 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root398Step4 : scan edge (search edge 3) 4 [2749] 2749 root398Tail4 winner =
    scan edge (search edge 3) 4 [2749] 2749 root398Tail5 winner := by
  rw [root398Tail4, scan, if_neg (by decide), root398Candidates4_checked]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail5
    (if 3 ≤ root398Candidates4.length then search edge 3 [2749, 4297] root398Candidates4 7046 winner else winner) = _
  rw [if_neg (by decide)]
def root398Candidates5 : List Nat := [13183]
set_option maxRecDepth 8192 in
theorem root398Candidates5_checked : candidates edge 4 2749 4993 root398Tail6 winner.bound = root398Candidates5 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root398Step5 : scan edge (search edge 3) 4 [2749] 2749 root398Tail5 winner =
    scan edge (search edge 3) 4 [2749] 2749 root398Tail6 winner := by
  rw [root398Tail5, scan, if_neg (by decide), root398Candidates5_checked]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail6
    (if 3 ≤ root398Candidates5.length then search edge 3 [2749, 4993] root398Candidates5 7742 winner else winner) = _
  rw [if_neg (by decide)]
theorem root398Scan6 : scan edge (search edge 3) 4 [2749] 2749 root398Tail6 winner = winner := by rw [root398Tail6, scan, if_pos (by decide)]
theorem root398Scan5 : scan edge (search edge 3) 4 [2749] 2749 root398Tail5 winner = winner :=
  (root398Step5).trans root398Scan6
theorem root398Scan4 : scan edge (search edge 3) 4 [2749] 2749 root398Tail4 winner = winner :=
  (root398Step4).trans root398Scan5
theorem root398Scan3 : scan edge (search edge 3) 4 [2749] 2749 root398Tail3 winner = winner :=
  (root398Step3).trans root398Scan4
theorem root398Scan2 : scan edge (search edge 3) 4 [2749] 2749 root398Tail2 winner = winner :=
  (root398Step2).trans root398Scan3
theorem root398Scan1 : scan edge (search edge 3) 4 [2749] 2749 root398Tail1 winner = winner :=
  (root398Step1).trans root398Scan2
theorem root398Scan0 : scan edge (search edge 3) 4 [2749] 2749 root398Tail0 winner = winner :=
  (root398Step0).trans root398Scan1
set_option maxRecDepth 4096 in
theorem root398Branch : search edge 4 [2749] root398Candidates 2749 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2749] 2749 root398Tail0 winner = winner
  exact root398Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
