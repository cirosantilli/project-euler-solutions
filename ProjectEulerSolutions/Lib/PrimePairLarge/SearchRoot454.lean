import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root454Candidates : List Nat := Data.row3229
set_option maxRecDepth 8192 in
theorem root454Candidates_checked : candidates edge 5 0 3229 tail455 winner.bound = root454Candidates := by
  apply candidates_eq_row edge 5 0 3229 tail455 root454Candidates winner.bound tail455_ordered
  · decide
  · decide
  · funext q
    rfl
def root454Tail43 : List Nat := []
def root454Tail42 : List Nat := 21727 :: root454Tail43
def root454Tail41 : List Nat := 21649 :: root454Tail42
def root454Tail40 : List Nat := 21601 :: root454Tail41
def root454Tail39 : List Nat := 21019 :: root454Tail40
def root454Tail38 : List Nat := 20929 :: root454Tail39
def root454Tail37 : List Nat := 20743 :: root454Tail38
def root454Tail36 : List Nat := 20089 :: root454Tail37
def root454Tail35 : List Nat := 19417 :: root454Tail36
def root454Tail34 : List Nat := 18517 :: root454Tail35
def root454Tail33 : List Nat := 18307 :: root454Tail34
def root454Tail32 : List Nat := 17911 :: root454Tail33
def root454Tail31 : List Nat := 17509 :: root454Tail32
def root454Tail30 : List Nat := 17467 :: root454Tail31
def root454Tail29 : List Nat := 17317 :: root454Tail30
def root454Tail28 : List Nat := 16927 :: root454Tail29
def root454Tail27 : List Nat := 14929 :: root454Tail28
def root454Tail26 : List Nat := 14923 :: root454Tail27
def root454Tail25 : List Nat := 12829 :: root454Tail26
def root454Tail24 : List Nat := 12451 :: root454Tail25
def root454Tail23 : List Nat := 9721 :: root454Tail24
def root454Tail22 : List Nat := 9511 :: root454Tail23
def root454Tail21 : List Nat := 9277 :: root454Tail22
def root454Tail20 : List Nat := 9151 :: root454Tail21
def root454Tail19 : List Nat := 9109 :: root454Tail20
def root454Tail18 : List Nat := 8527 :: root454Tail19
def root454Tail17 : List Nat := 8293 :: root454Tail18
def root454Tail16 : List Nat := 8191 :: root454Tail17
def root454Tail15 : List Nat := 6997 :: root454Tail16
def root454Tail14 : List Nat := 6883 :: root454Tail15
def root454Tail13 : List Nat := 6841 :: root454Tail14
def root454Tail12 : List Nat := 6703 :: root454Tail13
def root454Tail11 : List Nat := 6673 :: root454Tail12
def root454Tail10 : List Nat := 6553 :: root454Tail11
def root454Tail9 : List Nat := 6007 :: root454Tail10
def root454Tail8 : List Nat := 5581 :: root454Tail9
def root454Tail7 : List Nat := 5023 :: root454Tail8
def root454Tail6 : List Nat := 4993 :: root454Tail7
def root454Tail5 : List Nat := 4789 :: root454Tail6
def root454Tail4 : List Nat := 4621 :: root454Tail5
def root454Tail3 : List Nat := 4153 :: root454Tail4
def root454Tail2 : List Nat := 4057 :: root454Tail3
def root454Tail1 : List Nat := 3529 :: root454Tail2
def root454Tail0 : List Nat := 3463 :: root454Tail1
def root454Candidates0 : List Nat := [6883, 17467]
set_option maxRecDepth 8192 in
theorem root454Candidates0_checked : candidates edge 4 3229 3463 root454Tail1 winner.bound = root454Candidates0 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root454Step0 : scan edge (search edge 3) 4 [3229] 3229 root454Tail0 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail1 winner := by
  rw [root454Tail0, scan, if_neg (by decide), root454Candidates0_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail1
    (if 3 ≤ root454Candidates0.length then search edge 3 [3229, 3463] root454Candidates0 6692 winner else winner) = _
  rw [if_neg (by decide)]
def root454Candidates1 : List Nat := [6553]
set_option maxRecDepth 8192 in
theorem root454Candidates1_checked : candidates edge 4 3229 3529 root454Tail2 winner.bound = root454Candidates1 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root454Step1 : scan edge (search edge 3) 4 [3229] 3229 root454Tail1 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail2 winner := by
  rw [root454Tail1, scan, if_neg (by decide), root454Candidates1_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail2
    (if 3 ≤ root454Candidates1.length then search edge 3 [3229, 3529] root454Candidates1 6758 winner else winner) = _
  rw [if_neg (by decide)]
def root454Candidates2 : List Nat := [6007, 9109]
set_option maxRecDepth 8192 in
theorem root454Candidates2_checked : candidates edge 4 3229 4057 root454Tail3 winner.bound = root454Candidates2 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root454Step2 : scan edge (search edge 3) 4 [3229] 3229 root454Tail2 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail3 winner := by
  rw [root454Tail2, scan, if_neg (by decide), root454Candidates2_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail3
    (if 3 ≤ root454Candidates2.length then search edge 3 [3229, 4057] root454Candidates2 7286 winner else winner) = _
  rw [if_neg (by decide)]
def root454Candidates3 : List Nat := [4789, 5023, 6997]
set_option maxRecDepth 8192 in
theorem root454Candidates3_checked : candidates edge 4 3229 4153 root454Tail4 winner.bound = root454Candidates3 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root454Child3 : search edge 3 [3229, 4153] root454Candidates3 7382 winner = winner := by decide
theorem root454Step3 : scan edge (search edge 3) 4 [3229] 3229 root454Tail3 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail4 winner := by
  rw [root454Tail3, scan, if_neg (by decide), root454Candidates3_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail4
    (if 3 ≤ root454Candidates3.length then search edge 3 [3229, 4153] root454Candidates3 7382 winner else winner) = _
  rw [if_pos (by decide), root454Child3]
def root454Candidates4 : List Nat := [6673]
set_option maxRecDepth 8192 in
theorem root454Candidates4_checked : candidates edge 4 3229 4621 root454Tail5 winner.bound = root454Candidates4 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root454Step4 : scan edge (search edge 3) 4 [3229] 3229 root454Tail4 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail5 winner := by
  rw [root454Tail4, scan, if_neg (by decide), root454Candidates4_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail5
    (if 3 ≤ root454Candidates4.length then search edge 3 [3229, 4621] root454Candidates4 7850 winner else winner) = _
  rw [if_neg (by decide)]
def root454Candidates5 : List Nat := [6553, 9277, 17317]
set_option maxRecDepth 8192 in
theorem root454Candidates5_checked : candidates edge 4 3229 4789 root454Tail6 winner.bound = root454Candidates5 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root454Child5 : search edge 3 [3229, 4789] root454Candidates5 8018 winner = winner := by decide
theorem root454Step5 : scan edge (search edge 3) 4 [3229] 3229 root454Tail5 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail6 winner := by
  rw [root454Tail5, scan, if_neg (by decide), root454Candidates5_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail6
    (if 3 ≤ root454Candidates5.length then search edge 3 [3229, 4789] root454Candidates5 8018 winner else winner) = _
  rw [if_pos (by decide), root454Child5]
def root454Candidates6 : List Nat := [9511, 16927]
set_option maxRecDepth 8192 in
theorem root454Candidates6_checked : candidates edge 4 3229 4993 root454Tail7 winner.bound = root454Candidates6 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root454Step6 : scan edge (search edge 3) 4 [3229] 3229 root454Tail6 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail7 winner := by
  rw [root454Tail6, scan, if_neg (by decide), root454Candidates6_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail7
    (if 3 ≤ root454Candidates6.length then search edge 3 [3229, 4993] root454Candidates6 8222 winner else winner) = _
  rw [if_neg (by decide)]
def root454Candidates7 : List Nat := [6841, 17509]
set_option maxRecDepth 8192 in
theorem root454Candidates7_checked : candidates edge 4 3229 5023 root454Tail8 winner.bound = root454Candidates7 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root454Step7 : scan edge (search edge 3) 4 [3229] 3229 root454Tail7 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail8 winner := by
  rw [root454Tail7, scan, if_neg (by decide), root454Candidates7_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail8
    (if 3 ≤ root454Candidates7.length then search edge 3 [3229, 5023] root454Candidates7 8252 winner else winner) = _
  rw [if_neg (by decide)]
def root454Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root454Candidates8_checked : candidates edge 4 3229 5581 root454Tail9 winner.bound = root454Candidates8 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root454Step8 : scan edge (search edge 3) 4 [3229] 3229 root454Tail8 winner =
    scan edge (search edge 3) 4 [3229] 3229 root454Tail9 winner := by
  rw [root454Tail8, scan, if_neg (by decide), root454Candidates8_checked]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail9
    (if 3 ≤ root454Candidates8.length then search edge 3 [3229, 5581] root454Candidates8 8810 winner else winner) = _
  rw [if_neg (by decide)]
theorem root454Scan9 : scan edge (search edge 3) 4 [3229] 3229 root454Tail9 winner = winner := by rw [root454Tail9, scan, if_pos (by decide)]
theorem root454Scan8 : scan edge (search edge 3) 4 [3229] 3229 root454Tail8 winner = winner :=
  (root454Step8).trans root454Scan9
theorem root454Scan7 : scan edge (search edge 3) 4 [3229] 3229 root454Tail7 winner = winner :=
  (root454Step7).trans root454Scan8
theorem root454Scan6 : scan edge (search edge 3) 4 [3229] 3229 root454Tail6 winner = winner :=
  (root454Step6).trans root454Scan7
theorem root454Scan5 : scan edge (search edge 3) 4 [3229] 3229 root454Tail5 winner = winner :=
  (root454Step5).trans root454Scan6
theorem root454Scan4 : scan edge (search edge 3) 4 [3229] 3229 root454Tail4 winner = winner :=
  (root454Step4).trans root454Scan5
theorem root454Scan3 : scan edge (search edge 3) 4 [3229] 3229 root454Tail3 winner = winner :=
  (root454Step3).trans root454Scan4
theorem root454Scan2 : scan edge (search edge 3) 4 [3229] 3229 root454Tail2 winner = winner :=
  (root454Step2).trans root454Scan3
theorem root454Scan1 : scan edge (search edge 3) 4 [3229] 3229 root454Tail1 winner = winner :=
  (root454Step1).trans root454Scan2
theorem root454Scan0 : scan edge (search edge 3) 4 [3229] 3229 root454Tail0 winner = winner :=
  (root454Step0).trans root454Scan1
set_option maxRecDepth 4096 in
theorem root454Branch : search edge 4 [3229] root454Candidates 3229 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3229] 3229 root454Tail0 winner = winner
  exact root454Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
