import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root368Candidates : List Nat := Data.row2539
set_option maxRecDepth 8192 in
theorem root368Candidates_checked : candidates edge 5 0 2539 tail369 winner.bound = root368Candidates := by
  apply candidates_eq_row edge 5 0 2539 tail369 root368Candidates winner.bound tail369_ordered
  · decide
  · decide
  · funext q
    rfl
def root368Tail39 : List Nat := []
def root368Tail38 : List Nat := 23269 :: root368Tail39
def root368Tail37 : List Nat := 23227 :: root368Tail38
def root368Tail36 : List Nat := 23029 :: root368Tail37
def root368Tail35 : List Nat := 23017 :: root368Tail36
def root368Tail34 : List Nat := 22111 :: root368Tail35
def root368Tail33 : List Nat := 21433 :: root368Tail34
def root368Tail32 : List Nat := 20071 :: root368Tail33
def root368Tail31 : List Nat := 19843 :: root368Tail32
def root368Tail30 : List Nat := 19717 :: root368Tail31
def root368Tail29 : List Nat := 19069 :: root368Tail30
def root368Tail28 : List Nat := 17509 :: root368Tail29
def root368Tail27 : List Nat := 17341 :: root368Tail28
def root368Tail26 : List Nat := 17029 :: root368Tail27
def root368Tail25 : List Nat := 16603 :: root368Tail26
def root368Tail24 : List Nat := 15877 :: root368Tail25
def root368Tail23 : List Nat := 15661 :: root368Tail24
def root368Tail22 : List Nat := 15643 :: root368Tail23
def root368Tail21 : List Nat := 14629 :: root368Tail22
def root368Tail20 : List Nat := 13999 :: root368Tail21
def root368Tail19 : List Nat := 12601 :: root368Tail20
def root368Tail18 : List Nat := 12511 :: root368Tail19
def root368Tail17 : List Nat := 12097 :: root368Tail18
def root368Tail16 : List Nat := 9817 :: root368Tail17
def root368Tail15 : List Nat := 9511 :: root368Tail16
def root368Tail14 : List Nat := 8707 :: root368Tail15
def root368Tail13 : List Nat := 7951 :: root368Tail14
def root368Tail12 : List Nat := 7159 :: root368Tail13
def root368Tail11 : List Nat := 6823 :: root368Tail12
def root368Tail10 : List Nat := 6421 :: root368Tail11
def root368Tail9 : List Nat := 5449 :: root368Tail10
def root368Tail8 : List Nat := 5179 :: root368Tail9
def root368Tail7 : List Nat := 5119 :: root368Tail8
def root368Tail6 : List Nat := 4723 :: root368Tail7
def root368Tail5 : List Nat := 4441 :: root368Tail6
def root368Tail4 : List Nat := 4339 :: root368Tail5
def root368Tail3 : List Nat := 4297 :: root368Tail4
def root368Tail2 : List Nat := 3463 :: root368Tail3
def root368Tail1 : List Nat := 3229 :: root368Tail2
def root368Tail0 : List Nat := 3163 :: root368Tail1
def root368Candidates0 : List Nat := [7951, 8707]
set_option maxRecDepth 8192 in
theorem root368Candidates0_checked : candidates edge 4 2539 3163 root368Tail1 winner.bound = root368Candidates0 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root368Step0 : scan edge (search edge 3) 4 [2539] 2539 root368Tail0 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail1 winner := by
  rw [root368Tail0, scan, if_neg (by decide), root368Candidates0_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail1
    (if 3 ≤ root368Candidates0.length then search edge 3 [2539, 3163] root368Candidates0 5702 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates1 : List Nat := [3463, 9511, 17509]
set_option maxRecDepth 8192 in
theorem root368Candidates1_checked : candidates edge 4 2539 3229 root368Tail2 winner.bound = root368Candidates1 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root368Child1 : search edge 3 [2539, 3229] root368Candidates1 5768 winner = winner := by decide
theorem root368Step1 : scan edge (search edge 3) 4 [2539] 2539 root368Tail1 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail2 winner := by
  rw [root368Tail1, scan, if_neg (by decide), root368Candidates1_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail2
    (if 3 ≤ root368Candidates1.length then search edge 3 [2539, 3229] root368Candidates1 5768 winner else winner) = _
  rw [if_pos (by decide), root368Child1]
def root368Candidates2 : List Nat := [17029, 19069]
set_option maxRecDepth 8192 in
theorem root368Candidates2_checked : candidates edge 4 2539 3463 root368Tail3 winner.bound = root368Candidates2 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root368Step2 : scan edge (search edge 3) 4 [2539] 2539 root368Tail2 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail3 winner := by
  rw [root368Tail2, scan, if_neg (by decide), root368Candidates2_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail3
    (if 3 ≤ root368Candidates2.length then search edge 3 [2539, 3463] root368Candidates2 6002 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates3 : List Nat := [4441]
set_option maxRecDepth 8192 in
theorem root368Candidates3_checked : candidates edge 4 2539 4297 root368Tail4 winner.bound = root368Candidates3 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root368Step3 : scan edge (search edge 3) 4 [2539] 2539 root368Tail3 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail4 winner := by
  rw [root368Tail3, scan, if_neg (by decide), root368Candidates3_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail4
    (if 3 ≤ root368Candidates3.length then search edge 3 [2539, 4297] root368Candidates3 6836 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root368Candidates4_checked : candidates edge 4 2539 4339 root368Tail5 winner.bound = root368Candidates4 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root368Step4 : scan edge (search edge 3) 4 [2539] 2539 root368Tail4 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail5 winner := by
  rw [root368Tail4, scan, if_neg (by decide), root368Candidates4_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail5
    (if 3 ≤ root368Candidates4.length then search edge 3 [2539, 4339] root368Candidates4 6878 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root368Candidates5_checked : candidates edge 4 2539 4441 root368Tail6 winner.bound = root368Candidates5 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root368Step5 : scan edge (search edge 3) 4 [2539] 2539 root368Tail5 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail6 winner := by
  rw [root368Tail5, scan, if_neg (by decide), root368Candidates5_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail6
    (if 3 ≤ root368Candidates5.length then search edge 3 [2539, 4441] root368Candidates5 6980 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root368Candidates6_checked : candidates edge 4 2539 4723 root368Tail7 winner.bound = root368Candidates6 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root368Step6 : scan edge (search edge 3) 4 [2539] 2539 root368Tail6 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail7 winner := by
  rw [root368Tail6, scan, if_neg (by decide), root368Candidates6_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail7
    (if 3 ≤ root368Candidates6.length then search edge 3 [2539, 4723] root368Candidates6 7262 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates7 : List Nat := [12601]
set_option maxRecDepth 8192 in
theorem root368Candidates7_checked : candidates edge 4 2539 5119 root368Tail8 winner.bound = root368Candidates7 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root368Step7 : scan edge (search edge 3) 4 [2539] 2539 root368Tail7 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail8 winner := by
  rw [root368Tail7, scan, if_neg (by decide), root368Candidates7_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail8
    (if 3 ≤ root368Candidates7.length then search edge 3 [2539, 5119] root368Candidates7 7658 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root368Candidates8_checked : candidates edge 4 2539 5179 root368Tail9 winner.bound = root368Candidates8 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root368Step8 : scan edge (search edge 3) 4 [2539] 2539 root368Tail8 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail9 winner := by
  rw [root368Tail8, scan, if_neg (by decide), root368Candidates8_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail9
    (if 3 ≤ root368Candidates8.length then search edge 3 [2539, 5179] root368Candidates8 7718 winner else winner) = _
  rw [if_neg (by decide)]
def root368Candidates9 : List Nat := [9817, 15643, 17029]
set_option maxRecDepth 8192 in
theorem root368Candidates9_checked : candidates edge 4 2539 5449 root368Tail10 winner.bound = root368Candidates9 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root368Child9 : search edge 3 [2539, 5449] root368Candidates9 7988 winner = winner := by decide
theorem root368Step9 : scan edge (search edge 3) 4 [2539] 2539 root368Tail9 winner =
    scan edge (search edge 3) 4 [2539] 2539 root368Tail10 winner := by
  rw [root368Tail9, scan, if_neg (by decide), root368Candidates9_checked]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail10
    (if 3 ≤ root368Candidates9.length then search edge 3 [2539, 5449] root368Candidates9 7988 winner else winner) = _
  rw [if_pos (by decide), root368Child9]
theorem root368Scan10 : scan edge (search edge 3) 4 [2539] 2539 root368Tail10 winner = winner := by rw [root368Tail10, scan, if_pos (by decide)]
theorem root368Scan9 : scan edge (search edge 3) 4 [2539] 2539 root368Tail9 winner = winner :=
  (root368Step9).trans root368Scan10
theorem root368Scan8 : scan edge (search edge 3) 4 [2539] 2539 root368Tail8 winner = winner :=
  (root368Step8).trans root368Scan9
theorem root368Scan7 : scan edge (search edge 3) 4 [2539] 2539 root368Tail7 winner = winner :=
  (root368Step7).trans root368Scan8
theorem root368Scan6 : scan edge (search edge 3) 4 [2539] 2539 root368Tail6 winner = winner :=
  (root368Step6).trans root368Scan7
theorem root368Scan5 : scan edge (search edge 3) 4 [2539] 2539 root368Tail5 winner = winner :=
  (root368Step5).trans root368Scan6
theorem root368Scan4 : scan edge (search edge 3) 4 [2539] 2539 root368Tail4 winner = winner :=
  (root368Step4).trans root368Scan5
theorem root368Scan3 : scan edge (search edge 3) 4 [2539] 2539 root368Tail3 winner = winner :=
  (root368Step3).trans root368Scan4
theorem root368Scan2 : scan edge (search edge 3) 4 [2539] 2539 root368Tail2 winner = winner :=
  (root368Step2).trans root368Scan3
theorem root368Scan1 : scan edge (search edge 3) 4 [2539] 2539 root368Tail1 winner = winner :=
  (root368Step1).trans root368Scan2
theorem root368Scan0 : scan edge (search edge 3) 4 [2539] 2539 root368Tail0 winner = winner :=
  (root368Step0).trans root368Scan1
set_option maxRecDepth 4096 in
theorem root368Branch : search edge 4 [2539] root368Candidates 2539 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2539] 2539 root368Tail0 winner = winner
  exact root368Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
