import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root292Candidates : List Nat := Data.row1933
set_option maxRecDepth 8192 in
theorem root292Candidates_checked : candidates edge 5 0 1933 tail293 winner.bound = root292Candidates := by
  apply candidates_eq_row edge 5 0 1933 tail293 root292Candidates winner.bound tail293_ordered
  · decide
  · decide
  · funext q
    rfl
def root292Tail41 : List Nat := []
def root292Tail40 : List Nat := 23857 :: root292Tail41
def root292Tail39 : List Nat := 23311 :: root292Tail40
def root292Tail38 : List Nat := 23197 :: root292Tail39
def root292Tail37 : List Nat := 23071 :: root292Tail38
def root292Tail36 : List Nat := 21727 :: root292Tail37
def root292Tail35 : List Nat := 20887 :: root292Tail36
def root292Tail34 : List Nat := 20593 :: root292Tail35
def root292Tail33 : List Nat := 19993 :: root292Tail34
def root292Tail32 : List Nat := 19267 :: root292Tail33
def root292Tail31 : List Nat := 18859 :: root292Tail32
def root292Tail30 : List Nat := 18523 :: root292Tail31
def root292Tail29 : List Nat := 18481 :: root292Tail30
def root292Tail28 : List Nat := 17569 :: root292Tail29
def root292Tail27 : List Nat := 17401 :: root292Tail28
def root292Tail26 : List Nat := 17107 :: root292Tail27
def root292Tail25 : List Nat := 16369 :: root292Tail26
def root292Tail24 : List Nat := 15583 :: root292Tail25
def root292Tail23 : List Nat := 14533 :: root292Tail24
def root292Tail22 : List Nat := 13411 :: root292Tail23
def root292Tail21 : List Nat := 13297 :: root292Tail22
def root292Tail20 : List Nat := 12799 :: root292Tail21
def root292Tail19 : List Nat := 11383 :: root292Tail20
def root292Tail18 : List Nat := 10627 :: root292Tail19
def root292Tail17 : List Nat := 10597 :: root292Tail18
def root292Tail16 : List Nat := 9907 :: root292Tail17
def root292Tail15 : List Nat := 9391 :: root292Tail16
def root292Tail14 : List Nat := 8677 :: root292Tail15
def root292Tail13 : List Nat := 8521 :: root292Tail14
def root292Tail12 : List Nat := 8287 :: root292Tail13
def root292Tail11 : List Nat := 7933 :: root292Tail12
def root292Tail10 : List Nat := 7639 :: root292Tail11
def root292Tail9 : List Nat := 7333 :: root292Tail10
def root292Tail8 : List Nat := 6841 :: root292Tail9
def root292Tail7 : List Nat := 6703 :: root292Tail8
def root292Tail6 : List Nat := 5791 :: root292Tail7
def root292Tail5 : List Nat := 5023 :: root292Tail6
def root292Tail4 : List Nat := 5011 :: root292Tail5
def root292Tail3 : List Nat := 4201 :: root292Tail4
def root292Tail2 : List Nat := 3733 :: root292Tail3
def root292Tail1 : List Nat := 3313 :: root292Tail2
def root292Tail0 : List Nat := 2293 :: root292Tail1
def root292Candidates0 : List Nat := [7333]
set_option maxRecDepth 8192 in
theorem root292Candidates0_checked : candidates edge 4 1933 2293 root292Tail1 winner.bound = root292Candidates0 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root292Step0 : scan edge (search edge 3) 4 [1933] 1933 root292Tail0 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail1 winner := by
  rw [root292Tail0, scan, if_neg (by decide), root292Candidates0_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail1
    (if 3 ≤ root292Candidates0.length then search edge 3 [1933, 2293] root292Candidates0 4226 winner else winner) = _
  rw [if_neg (by decide)]
def root292Candidates1 : List Nat := [3733]
set_option maxRecDepth 8192 in
theorem root292Candidates1_checked : candidates edge 4 1933 3313 root292Tail2 winner.bound = root292Candidates1 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root292Step1 : scan edge (search edge 3) 4 [1933] 1933 root292Tail1 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail2 winner := by
  rw [root292Tail1, scan, if_neg (by decide), root292Candidates1_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail2
    (if 3 ≤ root292Candidates1.length then search edge 3 [1933, 3313] root292Candidates1 5246 winner else winner) = _
  rw [if_neg (by decide)]
def root292Candidates2 : List Nat := [10627]
set_option maxRecDepth 8192 in
theorem root292Candidates2_checked : candidates edge 4 1933 3733 root292Tail3 winner.bound = root292Candidates2 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root292Step2 : scan edge (search edge 3) 4 [1933] 1933 root292Tail2 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail3 winner := by
  rw [root292Tail2, scan, if_neg (by decide), root292Candidates2_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail3
    (if 3 ≤ root292Candidates2.length then search edge 3 [1933, 3733] root292Candidates2 5666 winner else winner) = _
  rw [if_neg (by decide)]
def root292Candidates3 : List Nat := [10597]
set_option maxRecDepth 8192 in
theorem root292Candidates3_checked : candidates edge 4 1933 4201 root292Tail4 winner.bound = root292Candidates3 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root292Step3 : scan edge (search edge 3) 4 [1933] 1933 root292Tail3 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail4 winner := by
  rw [root292Tail3, scan, if_neg (by decide), root292Candidates3_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail4
    (if 3 ≤ root292Candidates3.length then search edge 3 [1933, 4201] root292Candidates3 6134 winner else winner) = _
  rw [if_neg (by decide)]
def root292Candidates4 : List Nat := [11383, 14533, 17107]
set_option maxRecDepth 8192 in
theorem root292Candidates4_checked : candidates edge 4 1933 5011 root292Tail5 winner.bound = root292Candidates4 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root292Child4 : search edge 3 [1933, 5011] root292Candidates4 6944 winner = winner := by decide
theorem root292Step4 : scan edge (search edge 3) 4 [1933] 1933 root292Tail4 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail5 winner := by
  rw [root292Tail4, scan, if_neg (by decide), root292Candidates4_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail5
    (if 3 ≤ root292Candidates4.length then search edge 3 [1933, 5011] root292Candidates4 6944 winner else winner) = _
  rw [if_pos (by decide), root292Child4]
def root292Candidates5 : List Nat := [6841, 9907]
set_option maxRecDepth 8192 in
theorem root292Candidates5_checked : candidates edge 4 1933 5023 root292Tail6 winner.bound = root292Candidates5 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root292Step5 : scan edge (search edge 3) 4 [1933] 1933 root292Tail5 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail6 winner := by
  rw [root292Tail5, scan, if_neg (by decide), root292Candidates5_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail6
    (if 3 ≤ root292Candidates5.length then search edge 3 [1933, 5023] root292Candidates5 6956 winner else winner) = _
  rw [if_neg (by decide)]
def root292Candidates6 : List Nat := [7333, 17569]
set_option maxRecDepth 8192 in
theorem root292Candidates6_checked : candidates edge 4 1933 5791 root292Tail7 winner.bound = root292Candidates6 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root292Step6 : scan edge (search edge 3) 4 [1933] 1933 root292Tail6 winner =
    scan edge (search edge 3) 4 [1933] 1933 root292Tail7 winner := by
  rw [root292Tail6, scan, if_neg (by decide), root292Candidates6_checked]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail7
    (if 3 ≤ root292Candidates6.length then search edge 3 [1933, 5791] root292Candidates6 7724 winner else winner) = _
  rw [if_neg (by decide)]
theorem root292Scan7 : scan edge (search edge 3) 4 [1933] 1933 root292Tail7 winner = winner := by rw [root292Tail7, scan, if_pos (by decide)]
theorem root292Scan6 : scan edge (search edge 3) 4 [1933] 1933 root292Tail6 winner = winner :=
  (root292Step6).trans root292Scan7
theorem root292Scan5 : scan edge (search edge 3) 4 [1933] 1933 root292Tail5 winner = winner :=
  (root292Step5).trans root292Scan6
theorem root292Scan4 : scan edge (search edge 3) 4 [1933] 1933 root292Tail4 winner = winner :=
  (root292Step4).trans root292Scan5
theorem root292Scan3 : scan edge (search edge 3) 4 [1933] 1933 root292Tail3 winner = winner :=
  (root292Step3).trans root292Scan4
theorem root292Scan2 : scan edge (search edge 3) 4 [1933] 1933 root292Tail2 winner = winner :=
  (root292Step2).trans root292Scan3
theorem root292Scan1 : scan edge (search edge 3) 4 [1933] 1933 root292Tail1 winner = winner :=
  (root292Step1).trans root292Scan2
theorem root292Scan0 : scan edge (search edge 3) 4 [1933] 1933 root292Tail0 winner = winner :=
  (root292Step0).trans root292Scan1
set_option maxRecDepth 4096 in
theorem root292Branch : search edge 4 [1933] root292Candidates 1933 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1933] 1933 root292Tail0 winner = winner
  exact root292Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
