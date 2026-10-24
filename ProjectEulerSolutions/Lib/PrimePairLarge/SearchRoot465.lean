import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root465Candidates : List Nat := Data.row3323
set_option maxRecDepth 8192 in
theorem root465Candidates_checked : candidates edge 5 0 3323 tail466 winner.bound = root465Candidates := by
  apply candidates_eq_row edge 5 0 3323 tail466 root465Candidates winner.bound tail466_ordered
  · decide
  · decide
  · funext q
    rfl
def root465Tail36 : List Nat := []
def root465Tail35 : List Nat := 20411 :: root465Tail36
def root465Tail34 : List Nat := 19913 :: root465Tail35
def root465Tail33 : List Nat := 19577 :: root465Tail34
def root465Tail32 : List Nat := 18731 :: root465Tail33
def root465Tail31 : List Nat := 18587 :: root465Tail32
def root465Tail30 : List Nat := 17837 :: root465Tail31
def root465Tail29 : List Nat := 17333 :: root465Tail30
def root465Tail28 : List Nat := 16253 :: root465Tail29
def root465Tail27 : List Nat := 16073 :: root465Tail28
def root465Tail26 : List Nat := 16067 :: root465Tail27
def root465Tail25 : List Nat := 15923 :: root465Tail26
def root465Tail24 : List Nat := 15413 :: root465Tail25
def root465Tail23 : List Nat := 14561 :: root465Tail24
def root465Tail22 : List Nat := 13151 :: root465Tail23
def root465Tail21 : List Nat := 12713 :: root465Tail22
def root465Tail20 : List Nat := 10733 :: root465Tail21
def root465Tail19 : List Nat := 10037 :: root465Tail20
def root465Tail18 : List Nat := 8861 :: root465Tail19
def root465Tail17 : List Nat := 8423 :: root465Tail18
def root465Tail16 : List Nat := 7901 :: root465Tail17
def root465Tail15 : List Nat := 7349 :: root465Tail16
def root465Tail14 : List Nat := 7229 :: root465Tail15
def root465Tail13 : List Nat := 6869 :: root465Tail14
def root465Tail12 : List Nat := 5507 :: root465Tail13
def root465Tail11 : List Nat := 5393 :: root465Tail12
def root465Tail10 : List Nat := 5381 :: root465Tail11
def root465Tail9 : List Nat := 4871 :: root465Tail10
def root465Tail8 : List Nat := 4787 :: root465Tail9
def root465Tail7 : List Nat := 4679 :: root465Tail8
def root465Tail6 : List Nat := 4583 :: root465Tail7
def root465Tail5 : List Nat := 4349 :: root465Tail6
def root465Tail4 : List Nat := 4013 :: root465Tail5
def root465Tail3 : List Nat := 3989 :: root465Tail4
def root465Tail2 : List Nat := 3803 :: root465Tail3
def root465Tail1 : List Nat := 3701 :: root465Tail2
def root465Tail0 : List Nat := 3359 :: root465Tail1
def root465Candidates0 : List Nat := [17837]
set_option maxRecDepth 8192 in
theorem root465Candidates0_checked : candidates edge 4 3323 3359 root465Tail1 winner.bound = root465Candidates0 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root465Step0 : scan edge (search edge 3) 4 [3323] 3323 root465Tail0 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail1 winner := by
  rw [root465Tail0, scan, if_neg (by decide), root465Candidates0_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail1
    (if 3 ≤ root465Candidates0.length then search edge 3 [3323, 3359] root465Candidates0 6682 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates1 : List Nat := [6869, 17333]
set_option maxRecDepth 8192 in
theorem root465Candidates1_checked : candidates edge 4 3323 3701 root465Tail2 winner.bound = root465Candidates1 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
theorem root465Step1 : scan edge (search edge 3) 4 [3323] 3323 root465Tail1 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail2 winner := by
  rw [root465Tail1, scan, if_neg (by decide), root465Candidates1_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail2
    (if 3 ≤ root465Candidates1.length then search edge 3 [3323, 3701] root465Candidates1 7024 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates2 : List Nat := [5507, 13151, 16073, 16253]
set_option maxRecDepth 8192 in
theorem root465Candidates2_checked : candidates edge 4 3323 3803 root465Tail3 winner.bound = root465Candidates2 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root465Child2 : search edge 3 [3323, 3803] root465Candidates2 7126 winner = winner := by decide
theorem root465Step2 : scan edge (search edge 3) 4 [3323] 3323 root465Tail2 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail3 winner := by
  rw [root465Tail2, scan, if_neg (by decide), root465Candidates2_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail3
    (if 3 ≤ root465Candidates2.length then search edge 3 [3323, 3803] root465Candidates2 7126 winner else winner) = _
  rw [if_pos (by decide), root465Child2]
def root465Candidates3 : List Nat := [8861, 10037]
set_option maxRecDepth 8192 in
theorem root465Candidates3_checked : candidates edge 4 3323 3989 root465Tail4 winner.bound = root465Candidates3 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root465Step3 : scan edge (search edge 3) 4 [3323] 3323 root465Tail3 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail4 winner := by
  rw [root465Tail3, scan, if_neg (by decide), root465Candidates3_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail4
    (if 3 ≤ root465Candidates3.length then search edge 3 [3323, 3989] root465Candidates3 7312 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates4 : List Nat := [10037]
set_option maxRecDepth 8192 in
theorem root465Candidates4_checked : candidates edge 4 3323 4013 root465Tail5 winner.bound = root465Candidates4 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root465Step4 : scan edge (search edge 3) 4 [3323] 3323 root465Tail4 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail5 winner := by
  rw [root465Tail4, scan, if_neg (by decide), root465Candidates4_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail5
    (if 3 ≤ root465Candidates4.length then search edge 3 [3323, 4013] root465Candidates4 7336 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates5 : List Nat := [12713]
set_option maxRecDepth 8192 in
theorem root465Candidates5_checked : candidates edge 4 3323 4349 root465Tail6 winner.bound = root465Candidates5 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root465Step5 : scan edge (search edge 3) 4 [3323] 3323 root465Tail5 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail6 winner := by
  rw [root465Tail5, scan, if_neg (by decide), root465Candidates5_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail6
    (if 3 ≤ root465Candidates5.length then search edge 3 [3323, 4349] root465Candidates5 7672 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates6 : List Nat := [4787, 4871]
set_option maxRecDepth 8192 in
theorem root465Candidates6_checked : candidates edge 4 3323 4583 root465Tail7 winner.bound = root465Candidates6 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root465Step6 : scan edge (search edge 3) 4 [3323] 3323 root465Tail6 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail7 winner := by
  rw [root465Tail6, scan, if_neg (by decide), root465Candidates6_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail7
    (if 3 ≤ root465Candidates6.length then search edge 3 [3323, 4583] root465Candidates6 7906 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates7 : List Nat := [5393]
set_option maxRecDepth 8192 in
theorem root465Candidates7_checked : candidates edge 4 3323 4679 root465Tail8 winner.bound = root465Candidates7 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root465Step7 : scan edge (search edge 3) 4 [3323] 3323 root465Tail7 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail8 winner := by
  rw [root465Tail7, scan, if_neg (by decide), root465Candidates7_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail8
    (if 3 ≤ root465Candidates7.length then search edge 3 [3323, 4679] root465Candidates7 8002 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates8 : List Nat := [7229, 17333]
set_option maxRecDepth 8192 in
theorem root465Candidates8_checked : candidates edge 4 3323 4787 root465Tail9 winner.bound = root465Candidates8 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root465Step8 : scan edge (search edge 3) 4 [3323] 3323 root465Tail8 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail9 winner := by
  rw [root465Tail8, scan, if_neg (by decide), root465Candidates8_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail9
    (if 3 ≤ root465Candidates8.length then search edge 3 [3323, 4787] root465Candidates8 8110 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates9 : List Nat := [8861, 17333]
set_option maxRecDepth 8192 in
theorem root465Candidates9_checked : candidates edge 4 3323 4871 root465Tail10 winner.bound = root465Candidates9 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root465Step9 : scan edge (search edge 3) 4 [3323] 3323 root465Tail9 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail10 winner := by
  rw [root465Tail9, scan, if_neg (by decide), root465Candidates9_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail10
    (if 3 ≤ root465Candidates9.length then search edge 3 [3323, 4871] root465Candidates9 8194 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates10 : List Nat := [5507]
set_option maxRecDepth 8192 in
theorem root465Candidates10_checked : candidates edge 4 3323 5381 root465Tail11 winner.bound = root465Candidates10 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root465Step10 : scan edge (search edge 3) 4 [3323] 3323 root465Tail10 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail11 winner := by
  rw [root465Tail10, scan, if_neg (by decide), root465Candidates10_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail11
    (if 3 ≤ root465Candidates10.length then search edge 3 [3323, 5381] root465Candidates10 8704 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root465Candidates11_checked : candidates edge 4 3323 5393 root465Tail12 winner.bound = root465Candidates11 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root465Step11 : scan edge (search edge 3) 4 [3323] 3323 root465Tail11 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail12 winner := by
  rw [root465Tail11, scan, if_neg (by decide), root465Candidates11_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail12
    (if 3 ≤ root465Candidates11.length then search edge 3 [3323, 5393] root465Candidates11 8716 winner else winner) = _
  rw [if_neg (by decide)]
def root465Candidates12 : List Nat := [15413]
set_option maxRecDepth 8192 in
theorem root465Candidates12_checked : candidates edge 4 3323 5507 root465Tail13 winner.bound = root465Candidates12 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root465Step12 : scan edge (search edge 3) 4 [3323] 3323 root465Tail12 winner =
    scan edge (search edge 3) 4 [3323] 3323 root465Tail13 winner := by
  rw [root465Tail12, scan, if_neg (by decide), root465Candidates12_checked]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail13
    (if 3 ≤ root465Candidates12.length then search edge 3 [3323, 5507] root465Candidates12 8830 winner else winner) = _
  rw [if_neg (by decide)]
theorem root465Scan13 : scan edge (search edge 3) 4 [3323] 3323 root465Tail13 winner = winner := by rw [root465Tail13, scan, if_pos (by decide)]
theorem root465Scan12 : scan edge (search edge 3) 4 [3323] 3323 root465Tail12 winner = winner :=
  (root465Step12).trans root465Scan13
theorem root465Scan11 : scan edge (search edge 3) 4 [3323] 3323 root465Tail11 winner = winner :=
  (root465Step11).trans root465Scan12
theorem root465Scan10 : scan edge (search edge 3) 4 [3323] 3323 root465Tail10 winner = winner :=
  (root465Step10).trans root465Scan11
theorem root465Scan9 : scan edge (search edge 3) 4 [3323] 3323 root465Tail9 winner = winner :=
  (root465Step9).trans root465Scan10
theorem root465Scan8 : scan edge (search edge 3) 4 [3323] 3323 root465Tail8 winner = winner :=
  (root465Step8).trans root465Scan9
theorem root465Scan7 : scan edge (search edge 3) 4 [3323] 3323 root465Tail7 winner = winner :=
  (root465Step7).trans root465Scan8
theorem root465Scan6 : scan edge (search edge 3) 4 [3323] 3323 root465Tail6 winner = winner :=
  (root465Step6).trans root465Scan7
theorem root465Scan5 : scan edge (search edge 3) 4 [3323] 3323 root465Tail5 winner = winner :=
  (root465Step5).trans root465Scan6
theorem root465Scan4 : scan edge (search edge 3) 4 [3323] 3323 root465Tail4 winner = winner :=
  (root465Step4).trans root465Scan5
theorem root465Scan3 : scan edge (search edge 3) 4 [3323] 3323 root465Tail3 winner = winner :=
  (root465Step3).trans root465Scan4
theorem root465Scan2 : scan edge (search edge 3) 4 [3323] 3323 root465Tail2 winner = winner :=
  (root465Step2).trans root465Scan3
theorem root465Scan1 : scan edge (search edge 3) 4 [3323] 3323 root465Tail1 winner = winner :=
  (root465Step1).trans root465Scan2
theorem root465Scan0 : scan edge (search edge 3) 4 [3323] 3323 root465Tail0 winner = winner :=
  (root465Step0).trans root465Scan1
set_option maxRecDepth 4096 in
theorem root465Branch : search edge 4 [3323] root465Candidates 3323 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3323] 3323 root465Tail0 winner = winner
  exact root465Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
