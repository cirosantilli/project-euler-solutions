import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root414Candidates : List Nat := Data.row2879
set_option maxRecDepth 8192 in
theorem root414Candidates_checked : candidates edge 5 0 2879 tail415 winner.bound = root414Candidates := by
  apply candidates_eq_row edge 5 0 2879 tail415 root414Candidates winner.bound tail415_ordered
  · decide
  · decide
  · funext q
    rfl
def root414Tail40 : List Nat := []
def root414Tail39 : List Nat := 23087 :: root414Tail40
def root414Tail38 : List Nat := 23057 :: root414Tail39
def root414Tail37 : List Nat := 22511 :: root414Tail38
def root414Tail36 : List Nat := 22307 :: root414Tail37
def root414Tail35 : List Nat := 21419 :: root414Tail36
def root414Tail34 : List Nat := 21089 :: root414Tail35
def root414Tail33 : List Nat := 20231 :: root414Tail34
def root414Tail32 : List Nat := 20021 :: root414Tail33
def root414Tail31 : List Nat := 19403 :: root414Tail32
def root414Tail30 : List Nat := 16787 :: root414Tail31
def root414Tail29 : List Nat := 16529 :: root414Tail30
def root414Tail28 : List Nat := 15773 :: root414Tail29
def root414Tail27 : List Nat := 14447 :: root414Tail28
def root414Tail26 : List Nat := 13763 :: root414Tail27
def root414Tail25 : List Nat := 12809 :: root414Tail26
def root414Tail24 : List Nat := 12161 :: root414Tail25
def root414Tail23 : List Nat := 10631 :: root414Tail24
def root414Tail22 : List Nat := 10313 :: root414Tail23
def root414Tail21 : List Nat := 10253 :: root414Tail22
def root414Tail20 : List Nat := 10193 :: root414Tail21
def root414Tail19 : List Nat := 9941 :: root414Tail20
def root414Tail18 : List Nat := 8741 :: root414Tail19
def root414Tail17 : List Nat := 7883 :: root414Tail18
def root414Tail16 : List Nat := 7151 :: root414Tail17
def root414Tail15 : List Nat := 6959 :: root414Tail16
def root414Tail14 : List Nat := 6899 :: root414Tail15
def root414Tail13 : List Nat := 6701 :: root414Tail14
def root414Tail12 : List Nat := 6221 :: root414Tail13
def root414Tail11 : List Nat := 6143 :: root414Tail12
def root414Tail10 : List Nat := 5867 :: root414Tail11
def root414Tail9 : List Nat := 5693 :: root414Tail10
def root414Tail8 : List Nat := 5573 :: root414Tail9
def root414Tail7 : List Nat := 5531 :: root414Tail8
def root414Tail6 : List Nat := 5387 :: root414Tail7
def root414Tail5 : List Nat := 5303 :: root414Tail6
def root414Tail4 : List Nat := 5171 :: root414Tail5
def root414Tail3 : List Nat := 4673 :: root414Tail4
def root414Tail2 : List Nat := 4397 :: root414Tail3
def root414Tail1 : List Nat := 3581 :: root414Tail2
def root414Tail0 : List Nat := 3449 :: root414Tail1
def root414Candidates0 : List Nat := [5693, 10193, 16787, 19403]
set_option maxRecDepth 8192 in
theorem root414Candidates0_checked : candidates edge 4 2879 3449 root414Tail1 winner.bound = root414Candidates0 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root414Child0 : search edge 3 [2879, 3449] root414Candidates0 6328 winner = winner := by decide
theorem root414Step0 : scan edge (search edge 3) 4 [2879] 2879 root414Tail0 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail1 winner := by
  rw [root414Tail0, scan, if_neg (by decide), root414Candidates0_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail1
    (if 3 ≤ root414Candidates0.length then search edge 3 [2879, 3449] root414Candidates0 6328 winner else winner) = _
  rw [if_pos (by decide), root414Child0]
def root414Candidates1 : List Nat := [5303, 5387, 6701, 9941]
set_option maxRecDepth 8192 in
theorem root414Candidates1_checked : candidates edge 4 2879 3581 root414Tail2 winner.bound = root414Candidates1 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root414Child1 : search edge 3 [2879, 3581] root414Candidates1 6460 winner = winner := by decide
theorem root414Step1 : scan edge (search edge 3) 4 [2879] 2879 root414Tail1 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail2 winner := by
  rw [root414Tail1, scan, if_neg (by decide), root414Candidates1_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail2
    (if 3 ≤ root414Candidates1.length then search edge 3 [2879, 3581] root414Candidates1 6460 winner else winner) = _
  rw [if_pos (by decide), root414Child1]
def root414Candidates2 : List Nat := [6701, 12809]
set_option maxRecDepth 8192 in
theorem root414Candidates2_checked : candidates edge 4 2879 4397 root414Tail3 winner.bound = root414Candidates2 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root414Step2 : scan edge (search edge 3) 4 [2879] 2879 root414Tail2 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail3 winner := by
  rw [root414Tail2, scan, if_neg (by decide), root414Candidates2_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail3
    (if 3 ≤ root414Candidates2.length then search edge 3 [2879, 4397] root414Candidates2 7276 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates3 : List Nat := [10313]
set_option maxRecDepth 8192 in
theorem root414Candidates3_checked : candidates edge 4 2879 4673 root414Tail4 winner.bound = root414Candidates3 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root414Step3 : scan edge (search edge 3) 4 [2879] 2879 root414Tail3 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail4 winner := by
  rw [root414Tail3, scan, if_neg (by decide), root414Candidates3_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail4
    (if 3 ≤ root414Candidates3.length then search edge 3 [2879, 4673] root414Candidates3 7552 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates4 : List Nat := [14447]
set_option maxRecDepth 8192 in
theorem root414Candidates4_checked : candidates edge 4 2879 5171 root414Tail5 winner.bound = root414Candidates4 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root414Step4 : scan edge (search edge 3) 4 [2879] 2879 root414Tail4 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail5 winner := by
  rw [root414Tail4, scan, if_neg (by decide), root414Candidates4_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail5
    (if 3 ≤ root414Candidates4.length then search edge 3 [2879, 5171] root414Candidates4 8050 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root414Candidates5_checked : candidates edge 4 2879 5303 root414Tail6 winner.bound = root414Candidates5 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root414Step5 : scan edge (search edge 3) 4 [2879] 2879 root414Tail5 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail6 winner := by
  rw [root414Tail5, scan, if_neg (by decide), root414Candidates5_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail6
    (if 3 ≤ root414Candidates5.length then search edge 3 [2879, 5303] root414Candidates5 8182 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates6 : List Nat := [6959]
set_option maxRecDepth 8192 in
theorem root414Candidates6_checked : candidates edge 4 2879 5387 root414Tail7 winner.bound = root414Candidates6 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root414Step6 : scan edge (search edge 3) 4 [2879] 2879 root414Tail6 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail7 winner := by
  rw [root414Tail6, scan, if_neg (by decide), root414Candidates6_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail7
    (if 3 ≤ root414Candidates6.length then search edge 3 [2879, 5387] root414Candidates6 8266 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates7 : List Nat := [7883, 10193]
set_option maxRecDepth 8192 in
theorem root414Candidates7_checked : candidates edge 4 2879 5531 root414Tail8 winner.bound = root414Candidates7 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root414Step7 : scan edge (search edge 3) 4 [2879] 2879 root414Tail7 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail8 winner := by
  rw [root414Tail7, scan, if_neg (by decide), root414Candidates7_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail8
    (if 3 ≤ root414Candidates7.length then search edge 3 [2879, 5531] root414Candidates7 8410 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root414Candidates8_checked : candidates edge 4 2879 5573 root414Tail9 winner.bound = root414Candidates8 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root414Step8 : scan edge (search edge 3) 4 [2879] 2879 root414Tail8 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail9 winner := by
  rw [root414Tail8, scan, if_neg (by decide), root414Candidates8_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail9
    (if 3 ≤ root414Candidates8.length then search edge 3 [2879, 5573] root414Candidates8 8452 winner else winner) = _
  rw [if_neg (by decide)]
def root414Candidates9 : List Nat := [8741]
set_option maxRecDepth 8192 in
theorem root414Candidates9_checked : candidates edge 4 2879 5693 root414Tail10 winner.bound = root414Candidates9 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root414Step9 : scan edge (search edge 3) 4 [2879] 2879 root414Tail9 winner =
    scan edge (search edge 3) 4 [2879] 2879 root414Tail10 winner := by
  rw [root414Tail9, scan, if_neg (by decide), root414Candidates9_checked]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail10
    (if 3 ≤ root414Candidates9.length then search edge 3 [2879, 5693] root414Candidates9 8572 winner else winner) = _
  rw [if_neg (by decide)]
theorem root414Scan10 : scan edge (search edge 3) 4 [2879] 2879 root414Tail10 winner = winner := by rw [root414Tail10, scan, if_pos (by decide)]
theorem root414Scan9 : scan edge (search edge 3) 4 [2879] 2879 root414Tail9 winner = winner :=
  (root414Step9).trans root414Scan10
theorem root414Scan8 : scan edge (search edge 3) 4 [2879] 2879 root414Tail8 winner = winner :=
  (root414Step8).trans root414Scan9
theorem root414Scan7 : scan edge (search edge 3) 4 [2879] 2879 root414Tail7 winner = winner :=
  (root414Step7).trans root414Scan8
theorem root414Scan6 : scan edge (search edge 3) 4 [2879] 2879 root414Tail6 winner = winner :=
  (root414Step6).trans root414Scan7
theorem root414Scan5 : scan edge (search edge 3) 4 [2879] 2879 root414Tail5 winner = winner :=
  (root414Step5).trans root414Scan6
theorem root414Scan4 : scan edge (search edge 3) 4 [2879] 2879 root414Tail4 winner = winner :=
  (root414Step4).trans root414Scan5
theorem root414Scan3 : scan edge (search edge 3) 4 [2879] 2879 root414Tail3 winner = winner :=
  (root414Step3).trans root414Scan4
theorem root414Scan2 : scan edge (search edge 3) 4 [2879] 2879 root414Tail2 winner = winner :=
  (root414Step2).trans root414Scan3
theorem root414Scan1 : scan edge (search edge 3) 4 [2879] 2879 root414Tail1 winner = winner :=
  (root414Step1).trans root414Scan2
theorem root414Scan0 : scan edge (search edge 3) 4 [2879] 2879 root414Tail0 winner = winner :=
  (root414Step0).trans root414Scan1
set_option maxRecDepth 4096 in
theorem root414Branch : search edge 4 [2879] root414Candidates 2879 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2879] 2879 root414Tail0 winner = winner
  exact root414Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
