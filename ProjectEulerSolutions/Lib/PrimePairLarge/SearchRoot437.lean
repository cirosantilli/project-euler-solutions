import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root437Candidates : List Nat := Data.row3079
set_option maxRecDepth 8192 in
theorem root437Candidates_checked : candidates edge 5 0 3079 tail438 winner.bound = root437Candidates := by
  apply candidates_eq_row edge 5 0 3079 tail438 root437Candidates winner.bound tail438_ordered
  · decide
  · decide
  · funext q
    rfl
def root437Tail38 : List Nat := []
def root437Tail37 : List Nat := 21841 :: root437Tail38
def root437Tail36 : List Nat := 21379 :: root437Tail37
def root437Tail35 : List Nat := 20287 :: root437Tail36
def root437Tail34 : List Nat := 20047 :: root437Tail35
def root437Tail33 : List Nat := 18553 :: root437Tail34
def root437Tail32 : List Nat := 18493 :: root437Tail33
def root437Tail31 : List Nat := 17497 :: root437Tail32
def root437Tail30 : List Nat := 17419 :: root437Tail31
def root437Tail29 : List Nat := 16363 :: root437Tail30
def root437Tail28 : List Nat := 14419 :: root437Tail29
def root437Tail27 : List Nat := 14281 :: root437Tail28
def root437Tail26 : List Nat := 13633 :: root437Tail27
def root437Tail25 : List Nat := 12037 :: root437Tail26
def root437Tail24 : List Nat := 11689 :: root437Tail25
def root437Tail23 : List Nat := 11047 :: root437Tail24
def root437Tail22 : List Nat := 10597 :: root437Tail23
def root437Tail21 : List Nat := 10399 :: root437Tail22
def root437Tail20 : List Nat := 9973 :: root437Tail21
def root437Tail19 : List Nat := 9679 :: root437Tail20
def root437Tail18 : List Nat := 9463 :: root437Tail19
def root437Tail17 : List Nat := 8821 :: root437Tail18
def root437Tail16 : List Nat := 8329 :: root437Tail17
def root437Tail15 : List Nat := 8161 :: root437Tail16
def root437Tail14 : List Nat := 7867 :: root437Tail15
def root437Tail13 : List Nat := 7699 :: root437Tail14
def root437Tail12 : List Nat := 7561 :: root437Tail13
def root437Tail11 : List Nat := 7297 :: root437Tail12
def root437Tail10 : List Nat := 6709 :: root437Tail11
def root437Tail9 : List Nat := 6679 :: root437Tail10
def root437Tail8 : List Nat := 6301 :: root437Tail9
def root437Tail7 : List Nat := 5641 :: root437Tail8
def root437Tail6 : List Nat := 4957 :: root437Tail7
def root437Tail5 : List Nat := 4591 :: root437Tail6
def root437Tail4 : List Nat := 4273 :: root437Tail5
def root437Tail3 : List Nat := 3877 :: root437Tail4
def root437Tail2 : List Nat := 3709 :: root437Tail3
def root437Tail1 : List Nat := 3673 :: root437Tail2
def root437Tail0 : List Nat := 3181 :: root437Tail1
def root437Candidates0 : List Nat := [4273, 6709, 9973]
set_option maxRecDepth 8192 in
theorem root437Candidates0_checked : candidates edge 4 3079 3181 root437Tail1 winner.bound = root437Candidates0 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root437Child0 : search edge 3 [3079, 3181] root437Candidates0 6260 winner = winner := by decide
theorem root437Step0 : scan edge (search edge 3) 4 [3079] 3079 root437Tail0 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail1 winner := by
  rw [root437Tail0, scan, if_neg (by decide), root437Candidates0_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail1
    (if 3 ≤ root437Candidates0.length then search edge 3 [3079, 3181] root437Candidates0 6260 winner else winner) = _
  rw [if_pos (by decide), root437Child0]
def root437Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root437Candidates1_checked : candidates edge 4 3079 3673 root437Tail2 winner.bound = root437Candidates1 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root437Step1 : scan edge (search edge 3) 4 [3079] 3079 root437Tail1 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail2 winner := by
  rw [root437Tail1, scan, if_neg (by decide), root437Candidates1_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail2
    (if 3 ≤ root437Candidates1.length then search edge 3 [3079, 3673] root437Candidates1 6752 winner else winner) = _
  rw [if_neg (by decide)]
def root437Candidates2 : List Nat := [10597]
set_option maxRecDepth 8192 in
theorem root437Candidates2_checked : candidates edge 4 3079 3709 root437Tail3 winner.bound = root437Candidates2 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root437Step2 : scan edge (search edge 3) 4 [3079] 3079 root437Tail2 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail3 winner := by
  rw [root437Tail2, scan, if_neg (by decide), root437Candidates2_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail3
    (if 3 ≤ root437Candidates2.length then search edge 3 [3079, 3709] root437Candidates2 6788 winner else winner) = _
  rw [if_neg (by decide)]
def root437Candidates3 : List Nat := [18553]
set_option maxRecDepth 8192 in
theorem root437Candidates3_checked : candidates edge 4 3079 3877 root437Tail4 winner.bound = root437Candidates3 := by
  unfold candidates
  have row : edge 3877 = (SortedNatTree.contains FastRows.row3877) := by
    change Data.row3877.contains = (SortedNatTree.contains FastRows.row3877)
    exact FastRows.row3877_checked.symm
  rw [row]
  decide
theorem root437Step3 : scan edge (search edge 3) 4 [3079] 3079 root437Tail3 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail4 winner := by
  rw [root437Tail3, scan, if_neg (by decide), root437Candidates3_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail4
    (if 3 ≤ root437Candidates3.length then search edge 3 [3079, 3877] root437Candidates3 6956 winner else winner) = _
  rw [if_neg (by decide)]
def root437Candidates4 : List Nat := [7297, 17497]
set_option maxRecDepth 8192 in
theorem root437Candidates4_checked : candidates edge 4 3079 4273 root437Tail5 winner.bound = root437Candidates4 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root437Step4 : scan edge (search edge 3) 4 [3079] 3079 root437Tail4 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail5 winner := by
  rw [root437Tail4, scan, if_neg (by decide), root437Candidates4_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail5
    (if 3 ≤ root437Candidates4.length then search edge 3 [3079, 4273] root437Candidates4 7352 winner else winner) = _
  rw [if_neg (by decide)]
def root437Candidates5 : List Nat := [8329]
set_option maxRecDepth 8192 in
theorem root437Candidates5_checked : candidates edge 4 3079 4591 root437Tail6 winner.bound = root437Candidates5 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root437Step5 : scan edge (search edge 3) 4 [3079] 3079 root437Tail5 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail6 winner := by
  rw [root437Tail5, scan, if_neg (by decide), root437Candidates5_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail6
    (if 3 ≤ root437Candidates5.length then search edge 3 [3079, 4591] root437Candidates5 7670 winner else winner) = _
  rw [if_neg (by decide)]
def root437Candidates6 : List Nat := [17419]
set_option maxRecDepth 8192 in
theorem root437Candidates6_checked : candidates edge 4 3079 4957 root437Tail7 winner.bound = root437Candidates6 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root437Step6 : scan edge (search edge 3) 4 [3079] 3079 root437Tail6 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail7 winner := by
  rw [root437Tail6, scan, if_neg (by decide), root437Candidates6_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail7
    (if 3 ≤ root437Candidates6.length then search edge 3 [3079, 4957] root437Candidates6 8036 winner else winner) = _
  rw [if_neg (by decide)]
def root437Candidates7 : List Nat := [9463]
set_option maxRecDepth 8192 in
theorem root437Candidates7_checked : candidates edge 4 3079 5641 root437Tail8 winner.bound = root437Candidates7 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root437Step7 : scan edge (search edge 3) 4 [3079] 3079 root437Tail7 winner =
    scan edge (search edge 3) 4 [3079] 3079 root437Tail8 winner := by
  rw [root437Tail7, scan, if_neg (by decide), root437Candidates7_checked]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail8
    (if 3 ≤ root437Candidates7.length then search edge 3 [3079, 5641] root437Candidates7 8720 winner else winner) = _
  rw [if_neg (by decide)]
theorem root437Scan8 : scan edge (search edge 3) 4 [3079] 3079 root437Tail8 winner = winner := by rw [root437Tail8, scan, if_pos (by decide)]
theorem root437Scan7 : scan edge (search edge 3) 4 [3079] 3079 root437Tail7 winner = winner :=
  (root437Step7).trans root437Scan8
theorem root437Scan6 : scan edge (search edge 3) 4 [3079] 3079 root437Tail6 winner = winner :=
  (root437Step6).trans root437Scan7
theorem root437Scan5 : scan edge (search edge 3) 4 [3079] 3079 root437Tail5 winner = winner :=
  (root437Step5).trans root437Scan6
theorem root437Scan4 : scan edge (search edge 3) 4 [3079] 3079 root437Tail4 winner = winner :=
  (root437Step4).trans root437Scan5
theorem root437Scan3 : scan edge (search edge 3) 4 [3079] 3079 root437Tail3 winner = winner :=
  (root437Step3).trans root437Scan4
theorem root437Scan2 : scan edge (search edge 3) 4 [3079] 3079 root437Tail2 winner = winner :=
  (root437Step2).trans root437Scan3
theorem root437Scan1 : scan edge (search edge 3) 4 [3079] 3079 root437Tail1 winner = winner :=
  (root437Step1).trans root437Scan2
theorem root437Scan0 : scan edge (search edge 3) 4 [3079] 3079 root437Tail0 winner = winner :=
  (root437Step0).trans root437Scan1
set_option maxRecDepth 4096 in
theorem root437Branch : search edge 4 [3079] root437Candidates 3079 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3079] 3079 root437Tail0 winner = winner
  exact root437Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
