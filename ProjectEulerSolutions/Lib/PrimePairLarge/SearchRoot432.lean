import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root432Candidates : List Nat := Data.row3037
set_option maxRecDepth 8192 in
theorem root432Candidates_checked : candidates edge 5 0 3037 tail433 winner.bound = root432Candidates := by
  apply candidates_eq_row edge 5 0 3037 tail433 root432Candidates winner.bound tail433_ordered
  · decide
  · decide
  · funext q
    rfl
def root432Tail45 : List Nat := []
def root432Tail44 : List Nat := 22651 :: root432Tail45
def root432Tail43 : List Nat := 22039 :: root432Tail44
def root432Tail42 : List Nat := 21487 :: root432Tail43
def root432Tail41 : List Nat := 20887 :: root432Tail42
def root432Tail40 : List Nat := 20707 :: root432Tail41
def root432Tail39 : List Nat := 20089 :: root432Tail40
def root432Tail38 : List Nat := 19387 :: root432Tail39
def root432Tail37 : List Nat := 18313 :: root432Tail38
def root432Tail36 : List Nat := 18043 :: root432Tail37
def root432Tail35 : List Nat := 17539 :: root432Tail36
def root432Tail34 : List Nat := 17443 :: root432Tail35
def root432Tail33 : List Nat := 17209 :: root432Tail34
def root432Tail32 : List Nat := 16057 :: root432Tail33
def root432Tail31 : List Nat := 15901 :: root432Tail32
def root432Tail30 : List Nat := 15877 :: root432Tail31
def root432Tail29 : List Nat := 14797 :: root432Tail30
def root432Tail28 : List Nat := 14557 :: root432Tail29
def root432Tail27 : List Nat := 12163 :: root432Tail28
def root432Tail26 : List Nat := 11971 :: root432Tail27
def root432Tail25 : List Nat := 11437 :: root432Tail26
def root432Tail24 : List Nat := 11299 :: root432Tail25
def root432Tail23 : List Nat := 11119 :: root432Tail24
def root432Tail22 : List Nat := 10357 :: root432Tail23
def root432Tail21 : List Nat := 9787 :: root432Tail22
def root432Tail20 : List Nat := 9397 :: root432Tail21
def root432Tail19 : List Nat := 8713 :: root432Tail20
def root432Tail18 : List Nat := 8707 :: root432Tail19
def root432Tail17 : List Nat := 8461 :: root432Tail18
def root432Tail16 : List Nat := 8431 :: root432Tail17
def root432Tail15 : List Nat := 8329 :: root432Tail16
def root432Tail14 : List Nat := 7129 :: root432Tail15
def root432Tail13 : List Nat := 6823 :: root432Tail14
def root432Tail12 : List Nat := 6607 :: root432Tail13
def root432Tail11 : List Nat := 6301 :: root432Tail12
def root432Tail10 : List Nat := 6007 :: root432Tail11
def root432Tail9 : List Nat := 5701 :: root432Tail10
def root432Tail8 : List Nat := 5689 :: root432Tail9
def root432Tail7 : List Nat := 4987 :: root432Tail8
def root432Tail6 : List Nat := 4507 :: root432Tail7
def root432Tail5 : List Nat := 4423 :: root432Tail6
def root432Tail4 : List Nat := 3709 :: root432Tail5
def root432Tail3 : List Nat := 3517 :: root432Tail4
def root432Tail2 : List Nat := 3433 :: root432Tail3
def root432Tail1 : List Nat := 3331 :: root432Tail2
def root432Tail0 : List Nat := 3253 :: root432Tail1
def root432Candidates0 : List Nat := [3517]
set_option maxRecDepth 8192 in
theorem root432Candidates0_checked : candidates edge 4 3037 3253 root432Tail1 winner.bound = root432Candidates0 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root432Step0 : scan edge (search edge 3) 4 [3037] 3037 root432Tail0 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail1 winner := by
  rw [root432Tail0, scan, if_neg (by decide), root432Candidates0_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail1
    (if 3 ≤ root432Candidates0.length then search edge 3 [3037, 3253] root432Candidates0 6290 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates1 : List Nat := [5701, 8713]
set_option maxRecDepth 8192 in
theorem root432Candidates1_checked : candidates edge 4 3037 3331 root432Tail2 winner.bound = root432Candidates1 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root432Step1 : scan edge (search edge 3) 4 [3037] 3037 root432Tail1 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail2 winner := by
  rw [root432Tail1, scan, if_neg (by decide), root432Candidates1_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail2
    (if 3 ≤ root432Candidates1.length then search edge 3 [3037, 3331] root432Candidates1 6368 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates2 : List Nat := [4987, 6607, 7129, 8461]
set_option maxRecDepth 8192 in
theorem root432Candidates2_checked : candidates edge 4 3037 3433 root432Tail3 winner.bound = root432Candidates2 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root432Child2 : search edge 3 [3037, 3433] root432Candidates2 6470 winner = winner := by decide
theorem root432Step2 : scan edge (search edge 3) 4 [3037] 3037 root432Tail2 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail3 winner := by
  rw [root432Tail2, scan, if_neg (by decide), root432Candidates2_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail3
    (if 3 ≤ root432Candidates2.length then search edge 3 [3037, 3433] root432Candidates2 6470 winner else winner) = _
  rw [if_pos (by decide), root432Child2]
def root432Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root432Candidates3_checked : candidates edge 4 3037 3517 root432Tail4 winner.bound = root432Candidates3 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root432Step3 : scan edge (search edge 3) 4 [3037] 3037 root432Tail3 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail4 winner := by
  rw [root432Tail3, scan, if_neg (by decide), root432Candidates3_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail4
    (if 3 ≤ root432Candidates3.length then search edge 3 [3037, 3517] root432Candidates3 6554 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root432Candidates4_checked : candidates edge 4 3037 3709 root432Tail5 winner.bound = root432Candidates4 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root432Step4 : scan edge (search edge 3) 4 [3037] 3037 root432Tail4 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail5 winner := by
  rw [root432Tail4, scan, if_neg (by decide), root432Candidates4_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail5
    (if 3 ≤ root432Candidates4.length then search edge 3 [3037, 3709] root432Candidates4 6746 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates5 : List Nat := [5689, 10357, 15901]
set_option maxRecDepth 8192 in
theorem root432Candidates5_checked : candidates edge 4 3037 4423 root432Tail6 winner.bound = root432Candidates5 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root432Child5 : search edge 3 [3037, 4423] root432Candidates5 7460 winner = winner := by decide
theorem root432Step5 : scan edge (search edge 3) 4 [3037] 3037 root432Tail5 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail6 winner := by
  rw [root432Tail5, scan, if_neg (by decide), root432Candidates5_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail6
    (if 3 ≤ root432Candidates5.length then search edge 3 [3037, 4423] root432Candidates5 7460 winner else winner) = _
  rw [if_pos (by decide), root432Child5]
def root432Candidates6 : List Nat := [7129]
set_option maxRecDepth 8192 in
theorem root432Candidates6_checked : candidates edge 4 3037 4507 root432Tail7 winner.bound = root432Candidates6 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root432Step6 : scan edge (search edge 3) 4 [3037] 3037 root432Tail6 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail7 winner := by
  rw [root432Tail6, scan, if_neg (by decide), root432Candidates6_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail7
    (if 3 ≤ root432Candidates6.length then search edge 3 [3037, 4507] root432Candidates6 7544 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates7 : List Nat := [8461]
set_option maxRecDepth 8192 in
theorem root432Candidates7_checked : candidates edge 4 3037 4987 root432Tail8 winner.bound = root432Candidates7 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root432Step7 : scan edge (search edge 3) 4 [3037] 3037 root432Tail7 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail8 winner := by
  rw [root432Tail7, scan, if_neg (by decide), root432Candidates7_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail8
    (if 3 ≤ root432Candidates7.length then search edge 3 [3037, 4987] root432Candidates7 8024 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates8 : List Nat := [6823, 8329]
set_option maxRecDepth 8192 in
theorem root432Candidates8_checked : candidates edge 4 3037 5689 root432Tail9 winner.bound = root432Candidates8 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root432Step8 : scan edge (search edge 3) 4 [3037] 3037 root432Tail8 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail9 winner := by
  rw [root432Tail8, scan, if_neg (by decide), root432Candidates8_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail9
    (if 3 ≤ root432Candidates8.length then search edge 3 [3037, 5689] root432Candidates8 8726 winner else winner) = _
  rw [if_neg (by decide)]
def root432Candidates9 : List Nat := [8707]
set_option maxRecDepth 8192 in
theorem root432Candidates9_checked : candidates edge 4 3037 5701 root432Tail10 winner.bound = root432Candidates9 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root432Step9 : scan edge (search edge 3) 4 [3037] 3037 root432Tail9 winner =
    scan edge (search edge 3) 4 [3037] 3037 root432Tail10 winner := by
  rw [root432Tail9, scan, if_neg (by decide), root432Candidates9_checked]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail10
    (if 3 ≤ root432Candidates9.length then search edge 3 [3037, 5701] root432Candidates9 8738 winner else winner) = _
  rw [if_neg (by decide)]
theorem root432Scan10 : scan edge (search edge 3) 4 [3037] 3037 root432Tail10 winner = winner := by rw [root432Tail10, scan, if_pos (by decide)]
theorem root432Scan9 : scan edge (search edge 3) 4 [3037] 3037 root432Tail9 winner = winner :=
  (root432Step9).trans root432Scan10
theorem root432Scan8 : scan edge (search edge 3) 4 [3037] 3037 root432Tail8 winner = winner :=
  (root432Step8).trans root432Scan9
theorem root432Scan7 : scan edge (search edge 3) 4 [3037] 3037 root432Tail7 winner = winner :=
  (root432Step7).trans root432Scan8
theorem root432Scan6 : scan edge (search edge 3) 4 [3037] 3037 root432Tail6 winner = winner :=
  (root432Step6).trans root432Scan7
theorem root432Scan5 : scan edge (search edge 3) 4 [3037] 3037 root432Tail5 winner = winner :=
  (root432Step5).trans root432Scan6
theorem root432Scan4 : scan edge (search edge 3) 4 [3037] 3037 root432Tail4 winner = winner :=
  (root432Step4).trans root432Scan5
theorem root432Scan3 : scan edge (search edge 3) 4 [3037] 3037 root432Tail3 winner = winner :=
  (root432Step3).trans root432Scan4
theorem root432Scan2 : scan edge (search edge 3) 4 [3037] 3037 root432Tail2 winner = winner :=
  (root432Step2).trans root432Scan3
theorem root432Scan1 : scan edge (search edge 3) 4 [3037] 3037 root432Tail1 winner = winner :=
  (root432Step1).trans root432Scan2
theorem root432Scan0 : scan edge (search edge 3) 4 [3037] 3037 root432Tail0 winner = winner :=
  (root432Step0).trans root432Scan1
set_option maxRecDepth 4096 in
theorem root432Branch : search edge 4 [3037] root432Candidates 3037 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3037] 3037 root432Tail0 winner = winner
  exact root432Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
