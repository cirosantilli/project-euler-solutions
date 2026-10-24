import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root299Candidates : List Nat := Data.row1997
set_option maxRecDepth 8192 in
theorem root299Candidates_checked : candidates edge 5 0 1997 tail300 winner.bound = root299Candidates := by
  apply candidates_eq_row edge 5 0 1997 tail300 root299Candidates winner.bound tail300_ordered
  · decide
  · decide
  · funext q
    rfl
def root299Tail41 : List Nat := []
def root299Tail40 : List Nat := 23417 :: root299Tail41
def root299Tail39 : List Nat := 23399 :: root299Tail40
def root299Tail38 : List Nat := 23057 :: root299Tail39
def root299Tail37 : List Nat := 21383 :: root299Tail38
def root299Tail36 : List Nat := 21107 :: root299Tail37
def root299Tail35 : List Nat := 20441 :: root299Tail36
def root299Tail34 : List Nat := 19853 :: root299Tail35
def root299Tail33 : List Nat := 19301 :: root299Tail34
def root299Tail32 : List Nat := 19073 :: root299Tail33
def root299Tail31 : List Nat := 18149 :: root299Tail32
def root299Tail30 : List Nat := 18143 :: root299Tail31
def root299Tail29 : List Nat := 17681 :: root299Tail30
def root299Tail28 : List Nat := 16001 :: root299Tail29
def root299Tail27 : List Nat := 15959 :: root299Tail28
def root299Tail26 : List Nat := 15773 :: root299Tail27
def root299Tail25 : List Nat := 14951 :: root299Tail26
def root299Tail24 : List Nat := 14153 :: root299Tail25
def root299Tail23 : List Nat := 13109 :: root299Tail24
def root299Tail22 : List Nat := 12713 :: root299Tail23
def root299Tail21 : List Nat := 12671 :: root299Tail22
def root299Tail20 : List Nat := 12437 :: root299Tail21
def root299Tail19 : List Nat := 12377 :: root299Tail20
def root299Tail18 : List Nat := 11549 :: root299Tail19
def root299Tail17 : List Nat := 11117 :: root299Tail18
def root299Tail16 : List Nat := 10163 :: root299Tail17
def root299Tail15 : List Nat := 9941 :: root299Tail16
def root299Tail14 : List Nat := 9473 :: root299Tail15
def root299Tail13 : List Nat := 9413 :: root299Tail14
def root299Tail12 : List Nat := 8831 :: root299Tail13
def root299Tail11 : List Nat := 7583 :: root299Tail12
def root299Tail10 : List Nat := 7043 :: root299Tail11
def root299Tail9 : List Nat := 6053 :: root299Tail10
def root299Tail8 : List Nat := 4793 :: root299Tail9
def root299Tail7 : List Nat := 4583 :: root299Tail8
def root299Tail6 : List Nat := 4397 :: root299Tail7
def root299Tail5 : List Nat := 3677 :: root299Tail6
def root299Tail4 : List Nat := 3323 :: root299Tail5
def root299Tail3 : List Nat := 3119 :: root299Tail4
def root299Tail2 : List Nat := 2903 :: root299Tail3
def root299Tail1 : List Nat := 2423 :: root299Tail2
def root299Tail0 : List Nat := 2003 :: root299Tail1
def root299Candidates0 : List Nat := [12713, 15959]
set_option maxRecDepth 8192 in
theorem root299Candidates0_checked : candidates edge 4 1997 2003 root299Tail1 winner.bound = root299Candidates0 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
theorem root299Step0 : scan edge (search edge 3) 4 [1997] 1997 root299Tail0 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail1 winner := by
  rw [root299Tail0, scan, if_neg (by decide), root299Candidates0_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail1
    (if 3 ≤ root299Candidates0.length then search edge 3 [1997, 2003] root299Candidates0 4000 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates1 : List Nat := [11117, 14153]
set_option maxRecDepth 8192 in
theorem root299Candidates1_checked : candidates edge 4 1997 2423 root299Tail2 winner.bound = root299Candidates1 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root299Step1 : scan edge (search edge 3) 4 [1997] 1997 root299Tail1 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail2 winner := by
  rw [root299Tail1, scan, if_neg (by decide), root299Candidates1_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail2
    (if 3 ≤ root299Candidates1.length then search edge 3 [1997, 2423] root299Candidates1 4420 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates2 : List Nat := [16001, 19853]
set_option maxRecDepth 8192 in
theorem root299Candidates2_checked : candidates edge 4 1997 2903 root299Tail3 winner.bound = root299Candidates2 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root299Step2 : scan edge (search edge 3) 4 [1997] 1997 root299Tail2 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail3 winner := by
  rw [root299Tail2, scan, if_neg (by decide), root299Candidates2_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail3
    (if 3 ≤ root299Candidates2.length then search edge 3 [1997, 2903] root299Candidates2 4900 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates3 : List Nat := [3677, 8831]
set_option maxRecDepth 8192 in
theorem root299Candidates3_checked : candidates edge 4 1997 3119 root299Tail4 winner.bound = root299Candidates3 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
theorem root299Step3 : scan edge (search edge 3) 4 [1997] 1997 root299Tail3 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail4 winner := by
  rw [root299Tail3, scan, if_neg (by decide), root299Candidates3_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail4
    (if 3 ≤ root299Candidates3.length then search edge 3 [1997, 3119] root299Candidates3 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates4 : List Nat := [4583, 12713]
set_option maxRecDepth 8192 in
theorem root299Candidates4_checked : candidates edge 4 1997 3323 root299Tail5 winner.bound = root299Candidates4 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
theorem root299Step4 : scan edge (search edge 3) 4 [1997] 1997 root299Tail4 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail5 winner := by
  rw [root299Tail4, scan, if_neg (by decide), root299Candidates4_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail5
    (if 3 ≤ root299Candidates4.length then search edge 3 [1997, 3323] root299Candidates4 5320 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates5 : List Nat := [7583]
set_option maxRecDepth 8192 in
theorem root299Candidates5_checked : candidates edge 4 1997 3677 root299Tail6 winner.bound = root299Candidates5 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root299Step5 : scan edge (search edge 3) 4 [1997] 1997 root299Tail5 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail6 winner := by
  rw [root299Tail5, scan, if_neg (by decide), root299Candidates5_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail6
    (if 3 ≤ root299Candidates5.length then search edge 3 [1997, 3677] root299Candidates5 5674 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates6 : List Nat := [11549]
set_option maxRecDepth 8192 in
theorem root299Candidates6_checked : candidates edge 4 1997 4397 root299Tail7 winner.bound = root299Candidates6 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root299Step6 : scan edge (search edge 3) 4 [1997] 1997 root299Tail6 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail7 winner := by
  rw [root299Tail6, scan, if_neg (by decide), root299Candidates6_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail7
    (if 3 ≤ root299Candidates6.length then search edge 3 [1997, 4397] root299Candidates6 6394 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates7 : List Nat := [15773, 15959]
set_option maxRecDepth 8192 in
theorem root299Candidates7_checked : candidates edge 4 1997 4583 root299Tail8 winner.bound = root299Candidates7 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root299Step7 : scan edge (search edge 3) 4 [1997] 1997 root299Tail7 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail8 winner := by
  rw [root299Tail7, scan, if_neg (by decide), root299Candidates7_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail8
    (if 3 ≤ root299Candidates7.length then search edge 3 [1997, 4583] root299Candidates7 6580 winner else winner) = _
  rw [if_neg (by decide)]
def root299Candidates8 : List Nat := [19073]
set_option maxRecDepth 8192 in
theorem root299Candidates8_checked : candidates edge 4 1997 4793 root299Tail9 winner.bound = root299Candidates8 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root299Step8 : scan edge (search edge 3) 4 [1997] 1997 root299Tail8 winner =
    scan edge (search edge 3) 4 [1997] 1997 root299Tail9 winner := by
  rw [root299Tail8, scan, if_neg (by decide), root299Candidates8_checked]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail9
    (if 3 ≤ root299Candidates8.length then search edge 3 [1997, 4793] root299Candidates8 6790 winner else winner) = _
  rw [if_neg (by decide)]
theorem root299Scan9 : scan edge (search edge 3) 4 [1997] 1997 root299Tail9 winner = winner := by rw [root299Tail9, scan, if_pos (by decide)]
theorem root299Scan8 : scan edge (search edge 3) 4 [1997] 1997 root299Tail8 winner = winner :=
  (root299Step8).trans root299Scan9
theorem root299Scan7 : scan edge (search edge 3) 4 [1997] 1997 root299Tail7 winner = winner :=
  (root299Step7).trans root299Scan8
theorem root299Scan6 : scan edge (search edge 3) 4 [1997] 1997 root299Tail6 winner = winner :=
  (root299Step6).trans root299Scan7
theorem root299Scan5 : scan edge (search edge 3) 4 [1997] 1997 root299Tail5 winner = winner :=
  (root299Step5).trans root299Scan6
theorem root299Scan4 : scan edge (search edge 3) 4 [1997] 1997 root299Tail4 winner = winner :=
  (root299Step4).trans root299Scan5
theorem root299Scan3 : scan edge (search edge 3) 4 [1997] 1997 root299Tail3 winner = winner :=
  (root299Step3).trans root299Scan4
theorem root299Scan2 : scan edge (search edge 3) 4 [1997] 1997 root299Tail2 winner = winner :=
  (root299Step2).trans root299Scan3
theorem root299Scan1 : scan edge (search edge 3) 4 [1997] 1997 root299Tail1 winner = winner :=
  (root299Step1).trans root299Scan2
theorem root299Scan0 : scan edge (search edge 3) 4 [1997] 1997 root299Tail0 winner = winner :=
  (root299Step0).trans root299Scan1
set_option maxRecDepth 4096 in
theorem root299Branch : search edge 4 [1997] root299Candidates 1997 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1997] 1997 root299Tail0 winner = winner
  exact root299Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
