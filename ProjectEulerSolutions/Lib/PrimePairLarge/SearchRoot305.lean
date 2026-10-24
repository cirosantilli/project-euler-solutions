import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root305Candidates : List Nat := Data.row2029
set_option maxRecDepth 8192 in
theorem root305Candidates_checked : candidates edge 5 0 2029 tail306 winner.bound = root305Candidates := by
  apply candidates_eq_row edge 5 0 2029 tail306 root305Candidates winner.bound tail306_ordered
  · decide
  · decide
  · funext q
    rfl
def root305Tail43 : List Nat := []
def root305Tail42 : List Nat := 23311 :: root305Tail43
def root305Tail41 : List Nat := 23269 :: root305Tail42
def root305Tail40 : List Nat := 23197 :: root305Tail41
def root305Tail39 : List Nat := 22861 :: root305Tail40
def root305Tail38 : List Nat := 19681 :: root305Tail39
def root305Tail37 : List Nat := 19471 :: root305Tail38
def root305Tail36 : List Nat := 19009 :: root305Tail37
def root305Tail35 : List Nat := 18253 :: root305Tail36
def root305Tail34 : List Nat := 17977 :: root305Tail35
def root305Tail33 : List Nat := 17581 :: root305Tail34
def root305Tail32 : List Nat := 15907 :: root305Tail33
def root305Tail31 : List Nat := 14323 :: root305Tail32
def root305Tail30 : List Nat := 14197 :: root305Tail31
def root305Tail29 : List Nat := 13297 :: root305Tail30
def root305Tail28 : List Nat := 12907 :: root305Tail29
def root305Tail27 : List Nat := 12109 :: root305Tail28
def root305Tail26 : List Nat := 11827 :: root305Tail27
def root305Tail25 : List Nat := 11701 :: root305Tail26
def root305Tail24 : List Nat := 11299 :: root305Tail25
def root305Tail23 : List Nat := 11287 :: root305Tail24
def root305Tail22 : List Nat := 11257 :: root305Tail23
def root305Tail21 : List Nat := 11047 :: root305Tail22
def root305Tail20 : List Nat := 9931 :: root305Tail21
def root305Tail19 : List Nat := 9859 :: root305Tail20
def root305Tail18 : List Nat := 9679 :: root305Tail19
def root305Tail17 : List Nat := 9439 :: root305Tail18
def root305Tail16 : List Nat := 8293 :: root305Tail17
def root305Tail15 : List Nat := 7309 :: root305Tail16
def root305Tail14 : List Nat := 7213 :: root305Tail15
def root305Tail13 : List Nat := 7027 :: root305Tail14
def root305Tail12 : List Nat := 6247 :: root305Tail13
def root305Tail11 : List Nat := 6121 :: root305Tail12
def root305Tail10 : List Nat := 5701 :: root305Tail11
def root305Tail9 : List Nat := 5689 :: root305Tail10
def root305Tail8 : List Nat := 5419 :: root305Tail9
def root305Tail7 : List Nat := 5281 :: root305Tail8
def root305Tail6 : List Nat := 5227 :: root305Tail7
def root305Tail5 : List Nat := 5101 :: root305Tail6
def root305Tail4 : List Nat := 4507 :: root305Tail5
def root305Tail3 : List Nat := 3739 :: root305Tail4
def root305Tail2 : List Nat := 2791 :: root305Tail3
def root305Tail1 : List Nat := 2341 :: root305Tail2
def root305Tail0 : List Nat := 2281 :: root305Tail1
def root305Candidates0 : List Nat := [17581]
set_option maxRecDepth 8192 in
theorem root305Candidates0_checked : candidates edge 4 2029 2281 root305Tail1 winner.bound = root305Candidates0 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root305Step0 : scan edge (search edge 3) 4 [2029] 2029 root305Tail0 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail1 winner := by
  rw [root305Tail0, scan, if_neg (by decide), root305Candidates0_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail1
    (if 3 ≤ root305Candidates0.length then search edge 3 [2029, 2281] root305Candidates0 4310 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates1 : List Nat := [9679, 9859, 11701, 15907]
set_option maxRecDepth 8192 in
theorem root305Candidates1_checked : candidates edge 4 2029 2341 root305Tail2 winner.bound = root305Candidates1 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root305Child1 : search edge 3 [2029, 2341] root305Candidates1 4370 winner = winner := by decide
theorem root305Step1 : scan edge (search edge 3) 4 [2029] 2029 root305Tail1 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail2 winner := by
  rw [root305Tail1, scan, if_neg (by decide), root305Candidates1_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail2
    (if 3 ≤ root305Candidates1.length then search edge 3 [2029, 2341] root305Candidates1 4370 winner else winner) = _
  rw [if_pos (by decide), root305Child1]
def root305Candidates2 : List Nat := [5281]
set_option maxRecDepth 8192 in
theorem root305Candidates2_checked : candidates edge 4 2029 2791 root305Tail3 winner.bound = root305Candidates2 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root305Step2 : scan edge (search edge 3) 4 [2029] 2029 root305Tail2 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail3 winner := by
  rw [root305Tail2, scan, if_neg (by decide), root305Candidates2_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail3
    (if 3 ≤ root305Candidates2.length then search edge 3 [2029, 2791] root305Candidates2 4820 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root305Candidates3_checked : candidates edge 4 2029 3739 root305Tail4 winner.bound = root305Candidates3 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root305Step3 : scan edge (search edge 3) 4 [2029] 2029 root305Tail3 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail4 winner := by
  rw [root305Tail3, scan, if_neg (by decide), root305Candidates3_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail4
    (if 3 ≤ root305Candidates3.length then search edge 3 [2029, 3739] root305Candidates3 5768 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates4 : List Nat := [5419, 11257, 12907]
set_option maxRecDepth 8192 in
theorem root305Candidates4_checked : candidates edge 4 2029 4507 root305Tail5 winner.bound = root305Candidates4 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root305Child4 : search edge 3 [2029, 4507] root305Candidates4 6536 winner = winner := by decide
theorem root305Step4 : scan edge (search edge 3) 4 [2029] 2029 root305Tail4 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail5 winner := by
  rw [root305Tail4, scan, if_neg (by decide), root305Candidates4_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail5
    (if 3 ≤ root305Candidates4.length then search edge 3 [2029, 4507] root305Candidates4 6536 winner else winner) = _
  rw [if_pos (by decide), root305Child4]
def root305Candidates5 : List Nat := [8293, 9931]
set_option maxRecDepth 8192 in
theorem root305Candidates5_checked : candidates edge 4 2029 5101 root305Tail6 winner.bound = root305Candidates5 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root305Step5 : scan edge (search edge 3) 4 [2029] 2029 root305Tail5 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail6 winner := by
  rw [root305Tail5, scan, if_neg (by decide), root305Candidates5_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail6
    (if 3 ≤ root305Candidates5.length then search edge 3 [2029, 5101] root305Candidates5 7130 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root305Candidates6_checked : candidates edge 4 2029 5227 root305Tail7 winner.bound = root305Candidates6 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root305Step6 : scan edge (search edge 3) 4 [2029] 2029 root305Tail6 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail7 winner := by
  rw [root305Tail6, scan, if_neg (by decide), root305Candidates6_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail7
    (if 3 ≤ root305Candidates6.length then search edge 3 [2029, 5227] root305Candidates6 7256 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates7 : List Nat := [6121]
set_option maxRecDepth 8192 in
theorem root305Candidates7_checked : candidates edge 4 2029 5281 root305Tail8 winner.bound = root305Candidates7 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root305Step7 : scan edge (search edge 3) 4 [2029] 2029 root305Tail7 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail8 winner := by
  rw [root305Tail7, scan, if_neg (by decide), root305Candidates7_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail8
    (if 3 ≤ root305Candidates7.length then search edge 3 [2029, 5281] root305Candidates7 7310 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root305Candidates8_checked : candidates edge 4 2029 5419 root305Tail9 winner.bound = root305Candidates8 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root305Step8 : scan edge (search edge 3) 4 [2029] 2029 root305Tail8 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail9 winner := by
  rw [root305Tail8, scan, if_neg (by decide), root305Candidates8_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail9
    (if 3 ≤ root305Candidates8.length then search edge 3 [2029, 5419] root305Candidates8 7448 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates9 : List Nat := [9679, 11257]
set_option maxRecDepth 8192 in
theorem root305Candidates9_checked : candidates edge 4 2029 5689 root305Tail10 winner.bound = root305Candidates9 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root305Step9 : scan edge (search edge 3) 4 [2029] 2029 root305Tail9 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail10 winner := by
  rw [root305Tail9, scan, if_neg (by decide), root305Candidates9_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail10
    (if 3 ≤ root305Candidates9.length then search edge 3 [2029, 5689] root305Candidates9 7718 winner else winner) = _
  rw [if_neg (by decide)]
def root305Candidates10 : List Nat := [9439]
set_option maxRecDepth 8192 in
theorem root305Candidates10_checked : candidates edge 4 2029 5701 root305Tail11 winner.bound = root305Candidates10 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root305Step10 : scan edge (search edge 3) 4 [2029] 2029 root305Tail10 winner =
    scan edge (search edge 3) 4 [2029] 2029 root305Tail11 winner := by
  rw [root305Tail10, scan, if_neg (by decide), root305Candidates10_checked]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail11
    (if 3 ≤ root305Candidates10.length then search edge 3 [2029, 5701] root305Candidates10 7730 winner else winner) = _
  rw [if_neg (by decide)]
theorem root305Scan11 : scan edge (search edge 3) 4 [2029] 2029 root305Tail11 winner = winner := by rw [root305Tail11, scan, if_pos (by decide)]
theorem root305Scan10 : scan edge (search edge 3) 4 [2029] 2029 root305Tail10 winner = winner :=
  (root305Step10).trans root305Scan11
theorem root305Scan9 : scan edge (search edge 3) 4 [2029] 2029 root305Tail9 winner = winner :=
  (root305Step9).trans root305Scan10
theorem root305Scan8 : scan edge (search edge 3) 4 [2029] 2029 root305Tail8 winner = winner :=
  (root305Step8).trans root305Scan9
theorem root305Scan7 : scan edge (search edge 3) 4 [2029] 2029 root305Tail7 winner = winner :=
  (root305Step7).trans root305Scan8
theorem root305Scan6 : scan edge (search edge 3) 4 [2029] 2029 root305Tail6 winner = winner :=
  (root305Step6).trans root305Scan7
theorem root305Scan5 : scan edge (search edge 3) 4 [2029] 2029 root305Tail5 winner = winner :=
  (root305Step5).trans root305Scan6
theorem root305Scan4 : scan edge (search edge 3) 4 [2029] 2029 root305Tail4 winner = winner :=
  (root305Step4).trans root305Scan5
theorem root305Scan3 : scan edge (search edge 3) 4 [2029] 2029 root305Tail3 winner = winner :=
  (root305Step3).trans root305Scan4
theorem root305Scan2 : scan edge (search edge 3) 4 [2029] 2029 root305Tail2 winner = winner :=
  (root305Step2).trans root305Scan3
theorem root305Scan1 : scan edge (search edge 3) 4 [2029] 2029 root305Tail1 winner = winner :=
  (root305Step1).trans root305Scan2
theorem root305Scan0 : scan edge (search edge 3) 4 [2029] 2029 root305Tail0 winner = winner :=
  (root305Step0).trans root305Scan1
set_option maxRecDepth 4096 in
theorem root305Branch : search edge 4 [2029] root305Candidates 2029 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2029] 2029 root305Tail0 winner = winner
  exact root305Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
