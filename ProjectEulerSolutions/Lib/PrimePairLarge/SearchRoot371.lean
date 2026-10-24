import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root371Candidates : List Nat := Data.row2551
set_option maxRecDepth 8192 in
theorem root371Candidates_checked : candidates edge 5 0 2551 tail372 winner.bound = root371Candidates := by
  apply candidates_eq_row edge 5 0 2551 tail372 root371Candidates winner.bound tail372_ordered
  · decide
  · decide
  · funext q
    rfl
def root371Tail40 : List Nat := []
def root371Tail39 : List Nat := 22447 :: root371Tail40
def root371Tail38 : List Nat := 22171 :: root371Tail39
def root371Tail37 : List Nat := 21991 :: root371Tail38
def root371Tail36 : List Nat := 20113 :: root371Tail37
def root371Tail35 : List Nat := 20101 :: root371Tail36
def root371Tail34 : List Nat := 19219 :: root371Tail35
def root371Tail33 : List Nat := 19087 :: root371Tail34
def root371Tail32 : List Nat := 18289 :: root371Tail33
def root371Tail31 : List Nat := 18049 :: root371Tail32
def root371Tail30 : List Nat := 17377 :: root371Tail31
def root371Tail29 : List Nat := 16921 :: root371Tail30
def root371Tail28 : List Nat := 16333 :: root371Tail29
def root371Tail27 : List Nat := 16231 :: root371Tail28
def root371Tail26 : List Nat := 15901 :: root371Tail27
def root371Tail25 : List Nat := 15307 :: root371Tail26
def root371Tail24 : List Nat := 15031 :: root371Tail25
def root371Tail23 : List Nat := 13681 :: root371Tail24
def root371Tail22 : List Nat := 12799 :: root371Tail23
def root371Tail21 : List Nat := 12763 :: root371Tail22
def root371Tail20 : List Nat := 12379 :: root371Tail21
def root371Tail19 : List Nat := 12157 :: root371Tail20
def root371Tail18 : List Nat := 12007 :: root371Tail19
def root371Tail17 : List Nat := 11953 :: root371Tail18
def root371Tail16 : List Nat := 11743 :: root371Tail17
def root371Tail15 : List Nat := 9859 :: root371Tail16
def root371Tail14 : List Nat := 9631 :: root371Tail15
def root371Tail13 : List Nat := 9181 :: root371Tail14
def root371Tail12 : List Nat := 8719 :: root371Tail13
def root371Tail11 : List Nat := 7537 :: root371Tail12
def root371Tail10 : List Nat := 6793 :: root371Tail11
def root371Tail9 : List Nat := 6163 :: root371Tail10
def root371Tail8 : List Nat := 6037 :: root371Tail9
def root371Tail7 : List Nat := 5227 :: root371Tail8
def root371Tail6 : List Nat := 5059 :: root371Tail7
def root371Tail5 : List Nat := 4969 :: root371Tail6
def root371Tail4 : List Nat := 4813 :: root371Tail5
def root371Tail3 : List Nat := 4591 :: root371Tail4
def root371Tail2 : List Nat := 4219 :: root371Tail3
def root371Tail1 : List Nat := 3331 :: root371Tail2
def root371Tail0 : List Nat := 2887 :: root371Tail1
def root371Candidates0 : List Nat := [12763, 20101]
set_option maxRecDepth 8192 in
theorem root371Candidates0_checked : candidates edge 4 2551 2887 root371Tail1 winner.bound = root371Candidates0 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root371Step0 : scan edge (search edge 3) 4 [2551] 2551 root371Tail0 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail1 winner := by
  rw [root371Tail0, scan, if_neg (by decide), root371Candidates0_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail1
    (if 3 ≤ root371Candidates0.length then search edge 3 [2551, 2887] root371Candidates0 5438 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root371Candidates1_checked : candidates edge 4 2551 3331 root371Tail2 winner.bound = root371Candidates1 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root371Step1 : scan edge (search edge 3) 4 [2551] 2551 root371Tail1 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail2 winner := by
  rw [root371Tail1, scan, if_neg (by decide), root371Candidates1_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail2
    (if 3 ≤ root371Candidates1.length then search edge 3 [2551, 3331] root371Candidates1 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates2 : List Nat := [5227, 6793]
set_option maxRecDepth 8192 in
theorem root371Candidates2_checked : candidates edge 4 2551 4219 root371Tail3 winner.bound = root371Candidates2 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
theorem root371Step2 : scan edge (search edge 3) 4 [2551] 2551 root371Tail2 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail3 winner := by
  rw [root371Tail2, scan, if_neg (by decide), root371Candidates2_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail3
    (if 3 ≤ root371Candidates2.length then search edge 3 [2551, 4219] root371Candidates2 6770 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root371Candidates3_checked : candidates edge 4 2551 4591 root371Tail4 winner.bound = root371Candidates3 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root371Step3 : scan edge (search edge 3) 4 [2551] 2551 root371Tail3 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail4 winner := by
  rw [root371Tail3, scan, if_neg (by decide), root371Candidates3_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail4
    (if 3 ≤ root371Candidates3.length then search edge 3 [2551, 4591] root371Candidates3 7142 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates4 : List Nat := [12379]
set_option maxRecDepth 8192 in
theorem root371Candidates4_checked : candidates edge 4 2551 4813 root371Tail5 winner.bound = root371Candidates4 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root371Step4 : scan edge (search edge 3) 4 [2551] 2551 root371Tail4 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail5 winner := by
  rw [root371Tail4, scan, if_neg (by decide), root371Candidates4_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail5
    (if 3 ≤ root371Candidates4.length then search edge 3 [2551, 4813] root371Candidates4 7364 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates5 : List Nat := [18049]
set_option maxRecDepth 8192 in
theorem root371Candidates5_checked : candidates edge 4 2551 4969 root371Tail6 winner.bound = root371Candidates5 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root371Step5 : scan edge (search edge 3) 4 [2551] 2551 root371Tail5 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail6 winner := by
  rw [root371Tail5, scan, if_neg (by decide), root371Candidates5_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail6
    (if 3 ≤ root371Candidates5.length then search edge 3 [2551, 4969] root371Candidates5 7520 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root371Candidates6_checked : candidates edge 4 2551 5059 root371Tail7 winner.bound = root371Candidates6 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root371Step6 : scan edge (search edge 3) 4 [2551] 2551 root371Tail6 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail7 winner := by
  rw [root371Tail6, scan, if_neg (by decide), root371Candidates6_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail7
    (if 3 ≤ root371Candidates6.length then search edge 3 [2551, 5059] root371Candidates6 7610 winner else winner) = _
  rw [if_neg (by decide)]
def root371Candidates7 : List Nat := [16921]
set_option maxRecDepth 8192 in
theorem root371Candidates7_checked : candidates edge 4 2551 5227 root371Tail8 winner.bound = root371Candidates7 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root371Step7 : scan edge (search edge 3) 4 [2551] 2551 root371Tail7 winner =
    scan edge (search edge 3) 4 [2551] 2551 root371Tail8 winner := by
  rw [root371Tail7, scan, if_neg (by decide), root371Candidates7_checked]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail8
    (if 3 ≤ root371Candidates7.length then search edge 3 [2551, 5227] root371Candidates7 7778 winner else winner) = _
  rw [if_neg (by decide)]
theorem root371Scan8 : scan edge (search edge 3) 4 [2551] 2551 root371Tail8 winner = winner := by rw [root371Tail8, scan, if_pos (by decide)]
theorem root371Scan7 : scan edge (search edge 3) 4 [2551] 2551 root371Tail7 winner = winner :=
  (root371Step7).trans root371Scan8
theorem root371Scan6 : scan edge (search edge 3) 4 [2551] 2551 root371Tail6 winner = winner :=
  (root371Step6).trans root371Scan7
theorem root371Scan5 : scan edge (search edge 3) 4 [2551] 2551 root371Tail5 winner = winner :=
  (root371Step5).trans root371Scan6
theorem root371Scan4 : scan edge (search edge 3) 4 [2551] 2551 root371Tail4 winner = winner :=
  (root371Step4).trans root371Scan5
theorem root371Scan3 : scan edge (search edge 3) 4 [2551] 2551 root371Tail3 winner = winner :=
  (root371Step3).trans root371Scan4
theorem root371Scan2 : scan edge (search edge 3) 4 [2551] 2551 root371Tail2 winner = winner :=
  (root371Step2).trans root371Scan3
theorem root371Scan1 : scan edge (search edge 3) 4 [2551] 2551 root371Tail1 winner = winner :=
  (root371Step1).trans root371Scan2
theorem root371Scan0 : scan edge (search edge 3) 4 [2551] 2551 root371Tail0 winner = winner :=
  (root371Step0).trans root371Scan1
set_option maxRecDepth 4096 in
theorem root371Branch : search edge 4 [2551] root371Candidates 2551 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2551] 2551 root371Tail0 winner = winner
  exact root371Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
