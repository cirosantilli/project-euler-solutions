import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root336Candidates : List Nat := Data.row2281
set_option maxRecDepth 8192 in
theorem root336Candidates_checked : candidates edge 5 0 2281 tail337 winner.bound = root336Candidates := by
  apply candidates_eq_row edge 5 0 2281 tail337 root336Candidates winner.bound tail337_ordered
  · decide
  · decide
  · funext q
    rfl
def root336Tail37 : List Nat := []
def root336Tail36 : List Nat := 23209 :: root336Tail37
def root336Tail35 : List Nat := 21871 :: root336Tail36
def root336Tail34 : List Nat := 21319 :: root336Tail35
def root336Tail33 : List Nat := 21031 :: root336Tail34
def root336Tail32 : List Nat := 20983 :: root336Tail33
def root336Tail31 : List Nat := 20359 :: root336Tail32
def root336Tail30 : List Nat := 20173 :: root336Tail31
def root336Tail29 : List Nat := 19699 :: root336Tail30
def root336Tail28 : List Nat := 18973 :: root336Tail29
def root336Tail27 : List Nat := 18691 :: root336Tail28
def root336Tail26 : List Nat := 18169 :: root336Tail27
def root336Tail25 : List Nat := 17581 :: root336Tail26
def root336Tail24 : List Nat := 17443 :: root336Tail25
def root336Tail23 : List Nat := 14629 :: root336Tail24
def root336Tail22 : List Nat := 12853 :: root336Tail23
def root336Tail21 : List Nat := 12373 :: root336Tail22
def root336Tail20 : List Nat := 11197 :: root336Tail21
def root336Tail19 : List Nat := 10627 :: root336Tail20
def root336Tail18 : List Nat := 10459 :: root336Tail19
def root336Tail17 : List Nat := 9871 :: root336Tail18
def root336Tail16 : List Nat := 9649 :: root336Tail17
def root336Tail15 : List Nat := 9127 :: root336Tail16
def root336Tail14 : List Nat := 9091 :: root336Tail15
def root336Tail13 : List Nat := 8863 :: root336Tail14
def root336Tail12 : List Nat := 8179 :: root336Tail13
def root336Tail11 : List Nat := 7351 :: root336Tail12
def root336Tail10 : List Nat := 6793 :: root336Tail11
def root336Tail9 : List Nat := 6481 :: root336Tail10
def root336Tail8 : List Nat := 6373 :: root336Tail9
def root336Tail7 : List Nat := 6361 :: root336Tail8
def root336Tail6 : List Nat := 5857 :: root336Tail7
def root336Tail5 : List Nat := 4591 :: root336Tail6
def root336Tail4 : List Nat := 4273 :: root336Tail5
def root336Tail3 : List Nat := 4201 :: root336Tail4
def root336Tail2 : List Nat := 2887 :: root336Tail3
def root336Tail1 : List Nat := 2731 :: root336Tail2
def root336Tail0 : List Nat := 2551 :: root336Tail1
def root336Candidates0 : List Nat := [2887, 4591, 6793]
set_option maxRecDepth 8192 in
theorem root336Candidates0_checked : candidates edge 4 2281 2551 root336Tail1 winner.bound = root336Candidates0 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root336Child0 : search edge 3 [2281, 2551] root336Candidates0 4832 winner = winner := by decide
theorem root336Step0 : scan edge (search edge 3) 4 [2281] 2281 root336Tail0 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail1 winner := by
  rw [root336Tail0, scan, if_neg (by decide), root336Candidates0_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail1
    (if 3 ≤ root336Candidates0.length then search edge 3 [2281, 2551] root336Candidates0 4832 winner else winner) = _
  rw [if_pos (by decide), root336Child0]
def root336Candidates1 : List Nat := [18691]
set_option maxRecDepth 8192 in
theorem root336Candidates1_checked : candidates edge 4 2281 2731 root336Tail2 winner.bound = root336Candidates1 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
theorem root336Step1 : scan edge (search edge 3) 4 [2281] 2281 root336Tail1 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail2 winner := by
  rw [root336Tail1, scan, if_neg (by decide), root336Candidates1_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail2
    (if 3 ≤ root336Candidates1.length then search edge 3 [2281, 2731] root336Candidates1 5012 winner else winner) = _
  rw [if_neg (by decide)]
def root336Candidates2 : List Nat := [4273, 20359]
set_option maxRecDepth 8192 in
theorem root336Candidates2_checked : candidates edge 4 2281 2887 root336Tail3 winner.bound = root336Candidates2 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root336Step2 : scan edge (search edge 3) 4 [2281] 2281 root336Tail2 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail3 winner := by
  rw [root336Tail2, scan, if_neg (by decide), root336Candidates2_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail3
    (if 3 ≤ root336Candidates2.length then search edge 3 [2281, 2887] root336Candidates2 5168 winner else winner) = _
  rw [if_neg (by decide)]
def root336Candidates3 : List Nat := [8863]
set_option maxRecDepth 8192 in
theorem root336Candidates3_checked : candidates edge 4 2281 4201 root336Tail4 winner.bound = root336Candidates3 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root336Step3 : scan edge (search edge 3) 4 [2281] 2281 root336Tail3 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail4 winner := by
  rw [root336Tail3, scan, if_neg (by decide), root336Candidates3_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail4
    (if 3 ≤ root336Candidates3.length then search edge 3 [2281, 4201] root336Candidates3 6482 winner else winner) = _
  rw [if_neg (by decide)]
def root336Candidates4 : List Nat := [6373]
set_option maxRecDepth 8192 in
theorem root336Candidates4_checked : candidates edge 4 2281 4273 root336Tail5 winner.bound = root336Candidates4 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root336Step4 : scan edge (search edge 3) 4 [2281] 2281 root336Tail4 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail5 winner := by
  rw [root336Tail4, scan, if_neg (by decide), root336Candidates4_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail5
    (if 3 ≤ root336Candidates4.length then search edge 3 [2281, 4273] root336Candidates4 6554 winner else winner) = _
  rw [if_neg (by decide)]
def root336Candidates5 : List Nat := [10627, 17443]
set_option maxRecDepth 8192 in
theorem root336Candidates5_checked : candidates edge 4 2281 4591 root336Tail6 winner.bound = root336Candidates5 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root336Step5 : scan edge (search edge 3) 4 [2281] 2281 root336Tail5 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail6 winner := by
  rw [root336Tail5, scan, if_neg (by decide), root336Candidates5_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail6
    (if 3 ≤ root336Candidates5.length then search edge 3 [2281, 4591] root336Candidates5 6872 winner else winner) = _
  rw [if_neg (by decide)]
def root336Candidates6 : List Nat := [6481]
set_option maxRecDepth 8192 in
theorem root336Candidates6_checked : candidates edge 4 2281 5857 root336Tail7 winner.bound = root336Candidates6 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root336Step6 : scan edge (search edge 3) 4 [2281] 2281 root336Tail6 winner =
    scan edge (search edge 3) 4 [2281] 2281 root336Tail7 winner := by
  rw [root336Tail6, scan, if_neg (by decide), root336Candidates6_checked]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail7
    (if 3 ≤ root336Candidates6.length then search edge 3 [2281, 5857] root336Candidates6 8138 winner else winner) = _
  rw [if_neg (by decide)]
theorem root336Scan7 : scan edge (search edge 3) 4 [2281] 2281 root336Tail7 winner = winner := by rw [root336Tail7, scan, if_pos (by decide)]
theorem root336Scan6 : scan edge (search edge 3) 4 [2281] 2281 root336Tail6 winner = winner :=
  (root336Step6).trans root336Scan7
theorem root336Scan5 : scan edge (search edge 3) 4 [2281] 2281 root336Tail5 winner = winner :=
  (root336Step5).trans root336Scan6
theorem root336Scan4 : scan edge (search edge 3) 4 [2281] 2281 root336Tail4 winner = winner :=
  (root336Step4).trans root336Scan5
theorem root336Scan3 : scan edge (search edge 3) 4 [2281] 2281 root336Tail3 winner = winner :=
  (root336Step3).trans root336Scan4
theorem root336Scan2 : scan edge (search edge 3) 4 [2281] 2281 root336Tail2 winner = winner :=
  (root336Step2).trans root336Scan3
theorem root336Scan1 : scan edge (search edge 3) 4 [2281] 2281 root336Tail1 winner = winner :=
  (root336Step1).trans root336Scan2
theorem root336Scan0 : scan edge (search edge 3) 4 [2281] 2281 root336Tail0 winner = winner :=
  (root336Step0).trans root336Scan1
set_option maxRecDepth 4096 in
theorem root336Branch : search edge 4 [2281] root336Candidates 2281 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2281] 2281 root336Tail0 winner = winner
  exact root336Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
