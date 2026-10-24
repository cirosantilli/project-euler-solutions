import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root218Candidates : List Nat := Data.row1381
set_option maxRecDepth 8192 in
theorem root218Candidates_checked : candidates edge 5 0 1381 tail219 winner.bound = root218Candidates := by
  apply candidates_eq_row edge 5 0 1381 tail219 root218Candidates winner.bound tail219_ordered
  · decide
  · decide
  · funext q
    rfl
def root218Tail37 : List Nat := []
def root218Tail36 : List Nat := 24247 :: root218Tail37
def root218Tail35 : List Nat := 23203 :: root218Tail36
def root218Tail34 : List Nat := 22993 :: root218Tail35
def root218Tail33 : List Nat := 22129 :: root218Tail34
def root218Tail32 : List Nat := 20341 :: root218Tail33
def root218Tail31 : List Nat := 18181 :: root218Tail32
def root218Tail30 : List Nat := 17467 :: root218Tail31
def root218Tail29 : List Nat := 17191 :: root218Tail30
def root218Tail28 : List Nat := 16729 :: root218Tail29
def root218Tail27 : List Nat := 16477 :: root218Tail28
def root218Tail26 : List Nat := 16333 :: root218Tail27
def root218Tail25 : List Nat := 15727 :: root218Tail26
def root218Tail24 : List Nat := 15331 :: root218Tail25
def root218Tail23 : List Nat := 15121 :: root218Tail24
def root218Tail22 : List Nat := 12979 :: root218Tail23
def root218Tail21 : List Nat := 11299 :: root218Tail22
def root218Tail20 : List Nat := 10729 :: root218Tail21
def root218Tail19 : List Nat := 10627 :: root218Tail20
def root218Tail18 : List Nat := 10597 :: root218Tail19
def root218Tail17 : List Nat := 9811 :: root218Tail18
def root218Tail16 : List Nat := 9679 :: root218Tail17
def root218Tail15 : List Nat := 9199 :: root218Tail16
def root218Tail14 : List Nat := 8887 :: root218Tail15
def root218Tail13 : List Nat := 8821 :: root218Tail14
def root218Tail12 : List Nat := 8521 :: root218Tail13
def root218Tail11 : List Nat := 8017 :: root218Tail12
def root218Tail10 : List Nat := 7933 :: root218Tail11
def root218Tail9 : List Nat := 6823 :: root218Tail10
def root218Tail8 : List Nat := 6571 :: root218Tail9
def root218Tail7 : List Nat := 5233 :: root218Tail8
def root218Tail6 : List Nat := 4111 :: root218Tail7
def root218Tail5 : List Nat := 4027 :: root218Tail6
def root218Tail4 : List Nat := 3169 :: root218Tail5
def root218Tail3 : List Nat := 2767 :: root218Tail4
def root218Tail2 : List Nat := 2683 :: root218Tail3
def root218Tail1 : List Nat := 1789 :: root218Tail2
def root218Tail0 : List Nat := 1657 :: root218Tail1
def root218Candidates0 : List Nat := [2683, 4111, 8017, 10627]
set_option maxRecDepth 8192 in
theorem root218Candidates0_checked : candidates edge 4 1381 1657 root218Tail1 winner.bound = root218Candidates0 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root218Child0 : search edge 3 [1381, 1657] root218Candidates0 3038 winner = winner := by decide
theorem root218Step0 : scan edge (search edge 3) 4 [1381] 1381 root218Tail0 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail1 winner := by
  rw [root218Tail0, scan, if_neg (by decide), root218Candidates0_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail1
    (if 3 ≤ root218Candidates0.length then search edge 3 [1381, 1657] root218Candidates0 3038 winner else winner) = _
  rw [if_pos (by decide), root218Child0]
def root218Candidates1 : List Nat := [2683, 4111, 9811]
set_option maxRecDepth 8192 in
theorem root218Candidates1_checked : candidates edge 4 1381 1789 root218Tail2 winner.bound = root218Candidates1 := by
  unfold candidates
  have row : edge 1789 = (SortedNatTree.contains FastRows.row1789) := by
    change Data.row1789.contains = (SortedNatTree.contains FastRows.row1789)
    exact FastRows.row1789_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root218Child1 : search edge 3 [1381, 1789] root218Candidates1 3170 winner = winner := by decide
theorem root218Step1 : scan edge (search edge 3) 4 [1381] 1381 root218Tail1 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail2 winner := by
  rw [root218Tail1, scan, if_neg (by decide), root218Candidates1_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail2
    (if 3 ≤ root218Candidates1.length then search edge 3 [1381, 1789] root218Candidates1 3170 winner else winner) = _
  rw [if_pos (by decide), root218Child1]
def root218Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root218Candidates2_checked : candidates edge 4 1381 2683 root218Tail3 winner.bound = root218Candidates2 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root218Step2 : scan edge (search edge 3) 4 [1381] 1381 root218Tail2 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail3 winner := by
  rw [root218Tail2, scan, if_neg (by decide), root218Candidates2_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail3
    (if 3 ≤ root218Candidates2.length then search edge 3 [1381, 2683] root218Candidates2 4064 winner else winner) = _
  rw [if_neg (by decide)]
def root218Candidates3 : List Nat := [7933, 10627, 15121, 16477]
set_option maxRecDepth 8192 in
theorem root218Candidates3_checked : candidates edge 4 1381 2767 root218Tail4 winner.bound = root218Candidates3 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root218Child3 : search edge 3 [1381, 2767] root218Candidates3 4148 winner = winner := by decide
theorem root218Step3 : scan edge (search edge 3) 4 [1381] 1381 root218Tail3 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail4 winner := by
  rw [root218Tail3, scan, if_neg (by decide), root218Candidates3_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail4
    (if 3 ≤ root218Candidates3.length then search edge 3 [1381, 2767] root218Candidates3 4148 winner else winner) = _
  rw [if_pos (by decide), root218Child3]
def root218Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root218Candidates4_checked : candidates edge 4 1381 3169 root218Tail5 winner.bound = root218Candidates4 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
theorem root218Step4 : scan edge (search edge 3) 4 [1381] 1381 root218Tail4 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail5 winner := by
  rw [root218Tail4, scan, if_neg (by decide), root218Candidates4_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail5
    (if 3 ≤ root218Candidates4.length then search edge 3 [1381, 3169] root218Candidates4 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root218Candidates5 : List Nat := [5233, 15727, 18181]
set_option maxRecDepth 8192 in
theorem root218Candidates5_checked : candidates edge 4 1381 4027 root218Tail6 winner.bound = root218Candidates5 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root218Child5 : search edge 3 [1381, 4027] root218Candidates5 5408 winner = winner := by decide
theorem root218Step5 : scan edge (search edge 3) 4 [1381] 1381 root218Tail5 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail6 winner := by
  rw [root218Tail5, scan, if_neg (by decide), root218Candidates5_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail6
    (if 3 ≤ root218Candidates5.length then search edge 3 [1381, 4027] root218Candidates5 5408 winner else winner) = _
  rw [if_pos (by decide), root218Child5]
def root218Candidates6 : List Nat := [9811, 10597, 10627, 17467, 20341]
set_option maxRecDepth 8192 in
theorem root218Candidates6_checked : candidates edge 4 1381 4111 root218Tail7 winner.bound = root218Candidates6 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root218Child6 : search edge 3 [1381, 4111] root218Candidates6 5492 winner = winner := by decide
theorem root218Step6 : scan edge (search edge 3) 4 [1381] 1381 root218Tail6 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail7 winner := by
  rw [root218Tail6, scan, if_neg (by decide), root218Candidates6_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail7
    (if 3 ≤ root218Candidates6.length then search edge 3 [1381, 4111] root218Candidates6 5492 winner else winner) = _
  rw [if_pos (by decide), root218Child6]
def root218Candidates7 : List Nat := [10729]
set_option maxRecDepth 8192 in
theorem root218Candidates7_checked : candidates edge 4 1381 5233 root218Tail8 winner.bound = root218Candidates7 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root218Step7 : scan edge (search edge 3) 4 [1381] 1381 root218Tail7 winner =
    scan edge (search edge 3) 4 [1381] 1381 root218Tail8 winner := by
  rw [root218Tail7, scan, if_neg (by decide), root218Candidates7_checked]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail8
    (if 3 ≤ root218Candidates7.length then search edge 3 [1381, 5233] root218Candidates7 6614 winner else winner) = _
  rw [if_neg (by decide)]
theorem root218Scan8 : scan edge (search edge 3) 4 [1381] 1381 root218Tail8 winner = winner := by rw [root218Tail8, scan, if_pos (by decide)]
theorem root218Scan7 : scan edge (search edge 3) 4 [1381] 1381 root218Tail7 winner = winner :=
  (root218Step7).trans root218Scan8
theorem root218Scan6 : scan edge (search edge 3) 4 [1381] 1381 root218Tail6 winner = winner :=
  (root218Step6).trans root218Scan7
theorem root218Scan5 : scan edge (search edge 3) 4 [1381] 1381 root218Tail5 winner = winner :=
  (root218Step5).trans root218Scan6
theorem root218Scan4 : scan edge (search edge 3) 4 [1381] 1381 root218Tail4 winner = winner :=
  (root218Step4).trans root218Scan5
theorem root218Scan3 : scan edge (search edge 3) 4 [1381] 1381 root218Tail3 winner = winner :=
  (root218Step3).trans root218Scan4
theorem root218Scan2 : scan edge (search edge 3) 4 [1381] 1381 root218Tail2 winner = winner :=
  (root218Step2).trans root218Scan3
theorem root218Scan1 : scan edge (search edge 3) 4 [1381] 1381 root218Tail1 winner = winner :=
  (root218Step1).trans root218Scan2
theorem root218Scan0 : scan edge (search edge 3) 4 [1381] 1381 root218Tail0 winner = winner :=
  (root218Step0).trans root218Scan1
set_option maxRecDepth 4096 in
theorem root218Branch : search edge 4 [1381] root218Candidates 1381 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1381] 1381 root218Tail0 winner = winner
  exact root218Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
