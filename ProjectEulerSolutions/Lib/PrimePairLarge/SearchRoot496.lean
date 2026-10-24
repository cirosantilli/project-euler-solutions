import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root496Candidates : List Nat := Data.row3559
set_option maxRecDepth 8192 in
theorem root496Candidates_checked : candidates edge 5 0 3559 tail497 winner.bound = root496Candidates := by
  apply candidates_eq_row edge 5 0 3559 tail497 root496Candidates winner.bound tail497_ordered
  · decide
  · decide
  · funext q
    rfl
def root496Tail41 : List Nat := []
def root496Tail40 : List Nat := 21859 :: root496Tail41
def root496Tail39 : List Nat := 20743 :: root496Tail40
def root496Tail38 : List Nat := 19843 :: root496Tail39
def root496Tail37 : List Nat := 19507 :: root496Tail38
def root496Tail36 : List Nat := 19441 :: root496Tail37
def root496Tail35 : List Nat := 18679 :: root496Tail36
def root496Tail34 : List Nat := 18637 :: root496Tail35
def root496Tail33 : List Nat := 18217 :: root496Tail34
def root496Tail32 : List Nat := 16657 :: root496Tail33
def root496Tail31 : List Nat := 16411 :: root496Tail32
def root496Tail30 : List Nat := 15727 :: root496Tail31
def root496Tail29 : List Nat := 15649 :: root496Tail30
def root496Tail28 : List Nat := 15139 :: root496Tail29
def root496Tail27 : List Nat := 14341 :: root496Tail28
def root496Tail26 : List Nat := 13627 :: root496Tail27
def root496Tail25 : List Nat := 13219 :: root496Tail26
def root496Tail24 : List Nat := 13177 :: root496Tail25
def root496Tail23 : List Nat := 11161 :: root496Tail24
def root496Tail22 : List Nat := 10987 :: root496Tail23
def root496Tail21 : List Nat := 10771 :: root496Tail22
def root496Tail20 : List Nat := 10099 :: root496Tail21
def root496Tail19 : List Nat := 9973 :: root496Tail20
def root496Tail18 : List Nat := 9721 :: root496Tail19
def root496Tail17 : List Nat := 9511 :: root496Tail18
def root496Tail16 : List Nat := 9397 :: root496Tail17
def root496Tail15 : List Nat := 9337 :: root496Tail16
def root496Tail14 : List Nat := 8707 :: root496Tail15
def root496Tail13 : List Nat := 8053 :: root496Tail14
def root496Tail12 : List Nat := 7417 :: root496Tail13
def root496Tail11 : List Nat := 7369 :: root496Tail12
def root496Tail10 : List Nat := 6961 :: root496Tail11
def root496Tail9 : List Nat := 6619 :: root496Tail10
def root496Tail8 : List Nat := 6607 :: root496Tail9
def root496Tail7 : List Nat := 6037 :: root496Tail8
def root496Tail6 : List Nat := 5869 :: root496Tail7
def root496Tail5 : List Nat := 5449 :: root496Tail6
def root496Tail4 : List Nat := 4021 :: root496Tail5
def root496Tail3 : List Nat := 3931 :: root496Tail4
def root496Tail2 : List Nat := 3769 :: root496Tail3
def root496Tail1 : List Nat := 3709 :: root496Tail2
def root496Tail0 : List Nat := 3583 :: root496Tail1
def root496Candidates0 : List Nat := [15649]
set_option maxRecDepth 8192 in
theorem root496Candidates0_checked : candidates edge 4 3559 3583 root496Tail1 winner.bound = root496Candidates0 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root496Step0 : scan edge (search edge 3) 4 [3559] 3559 root496Tail0 winner =
    scan edge (search edge 3) 4 [3559] 3559 root496Tail1 winner := by
  rw [root496Tail0, scan, if_neg (by decide), root496Candidates0_checked]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail1
    (if 3 ≤ root496Candidates0.length then search edge 3 [3559, 3583] root496Candidates0 7142 winner else winner) = _
  rw [if_neg (by decide)]
def root496Candidates1 : List Nat := [7369]
set_option maxRecDepth 8192 in
theorem root496Candidates1_checked : candidates edge 4 3559 3709 root496Tail2 winner.bound = root496Candidates1 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root496Step1 : scan edge (search edge 3) 4 [3559] 3559 root496Tail1 winner =
    scan edge (search edge 3) 4 [3559] 3559 root496Tail2 winner := by
  rw [root496Tail1, scan, if_neg (by decide), root496Candidates1_checked]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail2
    (if 3 ≤ root496Candidates1.length then search edge 3 [3559, 3709] root496Candidates1 7268 winner else winner) = _
  rw [if_neg (by decide)]
def root496Candidates2 : List Nat := [6619, 13219, 15139]
set_option maxRecDepth 8192 in
theorem root496Candidates2_checked : candidates edge 4 3559 3769 root496Tail3 winner.bound = root496Candidates2 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root496Child2 : search edge 3 [3559, 3769] root496Candidates2 7328 winner = winner := by decide
theorem root496Step2 : scan edge (search edge 3) 4 [3559] 3559 root496Tail2 winner =
    scan edge (search edge 3) 4 [3559] 3559 root496Tail3 winner := by
  rw [root496Tail2, scan, if_neg (by decide), root496Candidates2_checked]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail3
    (if 3 ≤ root496Candidates2.length then search edge 3 [3559, 3769] root496Candidates2 7328 winner else winner) = _
  rw [if_pos (by decide), root496Child2]
def root496Candidates3 : List Nat := [5449]
set_option maxRecDepth 8192 in
theorem root496Candidates3_checked : candidates edge 4 3559 3931 root496Tail4 winner.bound = root496Candidates3 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root496Step3 : scan edge (search edge 3) 4 [3559] 3559 root496Tail3 winner =
    scan edge (search edge 3) 4 [3559] 3559 root496Tail4 winner := by
  rw [root496Tail3, scan, if_neg (by decide), root496Candidates3_checked]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail4
    (if 3 ≤ root496Candidates3.length then search edge 3 [3559, 3931] root496Candidates3 7490 winner else winner) = _
  rw [if_neg (by decide)]
def root496Candidates4 : List Nat := [10987]
set_option maxRecDepth 8192 in
theorem root496Candidates4_checked : candidates edge 4 3559 4021 root496Tail5 winner.bound = root496Candidates4 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root496Step4 : scan edge (search edge 3) 4 [3559] 3559 root496Tail4 winner =
    scan edge (search edge 3) 4 [3559] 3559 root496Tail5 winner := by
  rw [root496Tail4, scan, if_neg (by decide), root496Candidates4_checked]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail5
    (if 3 ≤ root496Candidates4.length then search edge 3 [3559, 4021] root496Candidates4 7580 winner else winner) = _
  rw [if_neg (by decide)]
def root496Candidates5 : List Nat := [13627, 15139]
set_option maxRecDepth 8192 in
theorem root496Candidates5_checked : candidates edge 4 3559 5449 root496Tail6 winner.bound = root496Candidates5 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root496Step5 : scan edge (search edge 3) 4 [3559] 3559 root496Tail5 winner =
    scan edge (search edge 3) 4 [3559] 3559 root496Tail6 winner := by
  rw [root496Tail5, scan, if_neg (by decide), root496Candidates5_checked]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail6
    (if 3 ≤ root496Candidates5.length then search edge 3 [3559, 5449] root496Candidates5 9008 winner else winner) = _
  rw [if_neg (by decide)]
theorem root496Scan6 : scan edge (search edge 3) 4 [3559] 3559 root496Tail6 winner = winner := by rw [root496Tail6, scan, if_pos (by decide)]
theorem root496Scan5 : scan edge (search edge 3) 4 [3559] 3559 root496Tail5 winner = winner :=
  (root496Step5).trans root496Scan6
theorem root496Scan4 : scan edge (search edge 3) 4 [3559] 3559 root496Tail4 winner = winner :=
  (root496Step4).trans root496Scan5
theorem root496Scan3 : scan edge (search edge 3) 4 [3559] 3559 root496Tail3 winner = winner :=
  (root496Step3).trans root496Scan4
theorem root496Scan2 : scan edge (search edge 3) 4 [3559] 3559 root496Tail2 winner = winner :=
  (root496Step2).trans root496Scan3
theorem root496Scan1 : scan edge (search edge 3) 4 [3559] 3559 root496Tail1 winner = winner :=
  (root496Step1).trans root496Scan2
theorem root496Scan0 : scan edge (search edge 3) 4 [3559] 3559 root496Tail0 winner = winner :=
  (root496Step0).trans root496Scan1
set_option maxRecDepth 4096 in
theorem root496Branch : search edge 4 [3559] root496Candidates 3559 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3559] 3559 root496Tail0 winner = winner
  exact root496Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
