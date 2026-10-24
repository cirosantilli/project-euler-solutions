import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root348Candidates : List Nat := Data.row2371
set_option maxRecDepth 8192 in
theorem root348Candidates_checked : candidates edge 5 0 2371 tail349 winner.bound = root348Candidates := by
  apply candidates_eq_row edge 5 0 2371 tail349 root348Candidates winner.bound tail349_ordered
  · decide
  · decide
  · funext q
    rfl
def root348Tail34 : List Nat := []
def root348Tail33 : List Nat := 23473 :: root348Tail34
def root348Tail32 : List Nat := 23197 :: root348Tail33
def root348Tail31 : List Nat := 22153 :: root348Tail32
def root348Tail30 : List Nat := 21817 :: root348Tail31
def root348Tail29 : List Nat := 21019 :: root348Tail30
def root348Tail28 : List Nat := 20749 :: root348Tail29
def root348Tail27 : List Nat := 19777 :: root348Tail28
def root348Tail26 : List Nat := 19609 :: root348Tail27
def root348Tail25 : List Nat := 18289 :: root348Tail26
def root348Tail24 : List Nat := 17911 :: root348Tail25
def root348Tail23 : List Nat := 17827 :: root348Tail24
def root348Tail22 : List Nat := 16567 :: root348Tail23
def root348Tail21 : List Nat := 16189 :: root348Tail22
def root348Tail20 : List Nat := 15121 :: root348Tail21
def root348Tail19 : List Nat := 13999 :: root348Tail20
def root348Tail18 : List Nat := 13441 :: root348Tail19
def root348Tail17 : List Nat := 13099 :: root348Tail18
def root348Tail16 : List Nat := 12703 :: root348Tail17
def root348Tail15 : List Nat := 12637 :: root348Tail16
def root348Tail14 : List Nat := 11821 :: root348Tail15
def root348Tail13 : List Nat := 10141 :: root348Tail14
def root348Tail12 : List Nat := 10099 :: root348Tail13
def root348Tail11 : List Nat := 9967 :: root348Tail12
def root348Tail10 : List Nat := 9901 :: root348Tail11
def root348Tail9 : List Nat := 8641 :: root348Tail10
def root348Tail8 : List Nat := 8287 :: root348Tail9
def root348Tail7 : List Nat := 7219 :: root348Tail8
def root348Tail6 : List Nat := 4423 :: root348Tail7
def root348Tail5 : List Nat := 4021 :: root348Tail6
def root348Tail4 : List Nat := 3643 :: root348Tail5
def root348Tail3 : List Nat := 3607 :: root348Tail4
def root348Tail2 : List Nat := 3229 :: root348Tail3
def root348Tail1 : List Nat := 3169 :: root348Tail2
def root348Tail0 : List Nat := 2917 :: root348Tail1
def root348Candidates0 : List Nat := [3169, 3643, 10099]
set_option maxRecDepth 8192 in
theorem root348Candidates0_checked : candidates edge 4 2371 2917 root348Tail1 winner.bound = root348Candidates0 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root348Child0 : search edge 3 [2371, 2917] root348Candidates0 5288 winner = winner := by decide
theorem root348Step0 : scan edge (search edge 3) 4 [2371] 2371 root348Tail0 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail1 winner := by
  rw [root348Tail0, scan, if_neg (by decide), root348Candidates0_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail1
    (if 3 ≤ root348Candidates0.length then search edge 3 [2371, 2917] root348Candidates0 5288 winner else winner) = _
  rw [if_pos (by decide), root348Child0]
def root348Candidates1 : List Nat := [3229]
set_option maxRecDepth 8192 in
theorem root348Candidates1_checked : candidates edge 4 2371 3169 root348Tail2 winner.bound = root348Candidates1 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
theorem root348Step1 : scan edge (search edge 3) 4 [2371] 2371 root348Tail1 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail2 winner := by
  rw [root348Tail1, scan, if_neg (by decide), root348Candidates1_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail2
    (if 3 ≤ root348Candidates1.length then search edge 3 [2371, 3169] root348Candidates1 5540 winner else winner) = _
  rw [if_neg (by decide)]
def root348Candidates2 : List Nat := [17911]
set_option maxRecDepth 8192 in
theorem root348Candidates2_checked : candidates edge 4 2371 3229 root348Tail3 winner.bound = root348Candidates2 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root348Step2 : scan edge (search edge 3) 4 [2371] 2371 root348Tail2 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail3 winner := by
  rw [root348Tail2, scan, if_neg (by decide), root348Candidates2_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail3
    (if 3 ≤ root348Candidates2.length then search edge 3 [2371, 3229] root348Candidates2 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root348Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root348Candidates3_checked : candidates edge 4 2371 3607 root348Tail4 winner.bound = root348Candidates3 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root348Step3 : scan edge (search edge 3) 4 [2371] 2371 root348Tail3 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail4 winner := by
  rw [root348Tail3, scan, if_neg (by decide), root348Candidates3_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail4
    (if 3 ≤ root348Candidates3.length then search edge 3 [2371, 3607] root348Candidates3 5978 winner else winner) = _
  rw [if_neg (by decide)]
def root348Candidates4 : List Nat := [18289]
set_option maxRecDepth 8192 in
theorem root348Candidates4_checked : candidates edge 4 2371 3643 root348Tail5 winner.bound = root348Candidates4 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root348Step4 : scan edge (search edge 3) 4 [2371] 2371 root348Tail4 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail5 winner := by
  rw [root348Tail4, scan, if_neg (by decide), root348Candidates4_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail5
    (if 3 ≤ root348Candidates4.length then search edge 3 [2371, 3643] root348Candidates4 6014 winner else winner) = _
  rw [if_neg (by decide)]
def root348Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root348Candidates5_checked : candidates edge 4 2371 4021 root348Tail6 winner.bound = root348Candidates5 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root348Step5 : scan edge (search edge 3) 4 [2371] 2371 root348Tail5 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail6 winner := by
  rw [root348Tail5, scan, if_neg (by decide), root348Candidates5_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail6
    (if 3 ≤ root348Candidates5.length then search edge 3 [2371, 4021] root348Candidates5 6392 winner else winner) = _
  rw [if_neg (by decide)]
def root348Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root348Candidates6_checked : candidates edge 4 2371 4423 root348Tail7 winner.bound = root348Candidates6 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root348Step6 : scan edge (search edge 3) 4 [2371] 2371 root348Tail6 winner =
    scan edge (search edge 3) 4 [2371] 2371 root348Tail7 winner := by
  rw [root348Tail6, scan, if_neg (by decide), root348Candidates6_checked]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail7
    (if 3 ≤ root348Candidates6.length then search edge 3 [2371, 4423] root348Candidates6 6794 winner else winner) = _
  rw [if_neg (by decide)]
theorem root348Scan7 : scan edge (search edge 3) 4 [2371] 2371 root348Tail7 winner = winner := by rw [root348Tail7, scan, if_pos (by decide)]
theorem root348Scan6 : scan edge (search edge 3) 4 [2371] 2371 root348Tail6 winner = winner :=
  (root348Step6).trans root348Scan7
theorem root348Scan5 : scan edge (search edge 3) 4 [2371] 2371 root348Tail5 winner = winner :=
  (root348Step5).trans root348Scan6
theorem root348Scan4 : scan edge (search edge 3) 4 [2371] 2371 root348Tail4 winner = winner :=
  (root348Step4).trans root348Scan5
theorem root348Scan3 : scan edge (search edge 3) 4 [2371] 2371 root348Tail3 winner = winner :=
  (root348Step3).trans root348Scan4
theorem root348Scan2 : scan edge (search edge 3) 4 [2371] 2371 root348Tail2 winner = winner :=
  (root348Step2).trans root348Scan3
theorem root348Scan1 : scan edge (search edge 3) 4 [2371] 2371 root348Tail1 winner = winner :=
  (root348Step1).trans root348Scan2
theorem root348Scan0 : scan edge (search edge 3) 4 [2371] 2371 root348Tail0 winner = winner :=
  (root348Step0).trans root348Scan1
set_option maxRecDepth 4096 in
theorem root348Branch : search edge 4 [2371] root348Candidates 2371 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2371] 2371 root348Tail0 winner = winner
  exact root348Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
