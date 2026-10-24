import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root427Candidates : List Nat := Data.row2999
set_option maxRecDepth 8192 in
theorem root427Candidates_checked : candidates edge 5 0 2999 tail428 winner.bound = root427Candidates := by
  apply candidates_eq_row edge 5 0 2999 tail428 root427Candidates winner.bound tail428_ordered
  · decide
  · decide
  · funext q
    rfl
def root427Tail38 : List Nat := []
def root427Tail37 : List Nat := 22067 :: root427Tail38
def root427Tail36 : List Nat := 21569 :: root427Tail37
def root427Tail35 : List Nat := 19121 :: root427Tail36
def root427Tail34 : List Nat := 18713 :: root427Tail35
def root427Tail33 : List Nat := 18191 :: root427Tail34
def root427Tail32 : List Nat := 17789 :: root427Tail33
def root427Tail31 : List Nat := 17609 :: root427Tail32
def root427Tail30 : List Nat := 17117 :: root427Tail31
def root427Tail29 : List Nat := 16607 :: root427Tail30
def root427Tail28 : List Nat := 16319 :: root427Tail29
def root427Tail27 : List Nat := 16301 :: root427Tail28
def root427Tail26 : List Nat := 15299 :: root427Tail27
def root427Tail25 : List Nat := 14957 :: root427Tail26
def root427Tail24 : List Nat := 14669 :: root427Tail25
def root427Tail23 : List Nat := 14033 :: root427Tail24
def root427Tail22 : List Nat := 12653 :: root427Tail23
def root427Tail21 : List Nat := 11699 :: root427Tail22
def root427Tail20 : List Nat := 11483 :: root427Tail21
def root427Tail19 : List Nat := 11273 :: root427Tail20
def root427Tail18 : List Nat := 10979 :: root427Tail19
def root427Tail17 : List Nat := 10799 :: root427Tail18
def root427Tail16 : List Nat := 10601 :: root427Tail17
def root427Tail15 : List Nat := 10463 :: root427Tail16
def root427Tail14 : List Nat := 8951 :: root427Tail15
def root427Tail13 : List Nat := 8537 :: root427Tail14
def root427Tail12 : List Nat := 7907 :: root427Tail13
def root427Tail11 : List Nat := 7577 :: root427Tail12
def root427Tail10 : List Nat := 7481 :: root427Tail11
def root427Tail9 : List Nat := 7349 :: root427Tail10
def root427Tail8 : List Nat := 6311 :: root427Tail9
def root427Tail7 : List Nat := 5003 :: root427Tail8
def root427Tail6 : List Nat := 4871 :: root427Tail7
def root427Tail5 : List Nat := 4373 :: root427Tail6
def root427Tail4 : List Nat := 4259 :: root427Tail5
def root427Tail3 : List Nat := 4073 :: root427Tail4
def root427Tail2 : List Nat := 3881 :: root427Tail3
def root427Tail1 : List Nat := 3413 :: root427Tail2
def root427Tail0 : List Nat := 3371 :: root427Tail1
def root427Candidates0 : List Nat := [3413, 11273, 15299, 17609]
set_option maxRecDepth 8192 in
theorem root427Candidates0_checked : candidates edge 4 2999 3371 root427Tail1 winner.bound = root427Candidates0 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root427Child0 : search edge 3 [2999, 3371] root427Candidates0 6370 winner = winner := by decide
theorem root427Step0 : scan edge (search edge 3) 4 [2999] 2999 root427Tail0 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail1 winner := by
  rw [root427Tail0, scan, if_neg (by decide), root427Candidates0_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail1
    (if 3 ≤ root427Candidates0.length then search edge 3 [2999, 3371] root427Candidates0 6370 winner else winner) = _
  rw [if_pos (by decide), root427Child0]
def root427Candidates1 : List Nat := [11483]
set_option maxRecDepth 8192 in
theorem root427Candidates1_checked : candidates edge 4 2999 3413 root427Tail2 winner.bound = root427Candidates1 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root427Step1 : scan edge (search edge 3) 4 [2999] 2999 root427Tail1 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail2 winner := by
  rw [root427Tail1, scan, if_neg (by decide), root427Candidates1_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail2
    (if 3 ≤ root427Candidates1.length then search edge 3 [2999, 3413] root427Candidates1 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root427Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root427Candidates2_checked : candidates edge 4 2999 3881 root427Tail3 winner.bound = root427Candidates2 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root427Step2 : scan edge (search edge 3) 4 [2999] 2999 root427Tail2 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail3 winner := by
  rw [root427Tail2, scan, if_neg (by decide), root427Candidates2_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail3
    (if 3 ≤ root427Candidates2.length then search edge 3 [2999, 3881] root427Candidates2 6880 winner else winner) = _
  rw [if_neg (by decide)]
def root427Candidates3 : List Nat := [5003]
set_option maxRecDepth 8192 in
theorem root427Candidates3_checked : candidates edge 4 2999 4073 root427Tail4 winner.bound = root427Candidates3 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root427Step3 : scan edge (search edge 3) 4 [2999] 2999 root427Tail3 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail4 winner := by
  rw [root427Tail3, scan, if_neg (by decide), root427Candidates3_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail4
    (if 3 ≤ root427Candidates3.length then search edge 3 [2999, 4073] root427Candidates3 7072 winner else winner) = _
  rw [if_neg (by decide)]
def root427Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root427Candidates4_checked : candidates edge 4 2999 4259 root427Tail5 winner.bound = root427Candidates4 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root427Step4 : scan edge (search edge 3) 4 [2999] 2999 root427Tail4 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail5 winner := by
  rw [root427Tail4, scan, if_neg (by decide), root427Candidates4_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail5
    (if 3 ≤ root427Candidates4.length then search edge 3 [2999, 4259] root427Candidates4 7258 winner else winner) = _
  rw [if_neg (by decide)]
def root427Candidates5 : List Nat := [10463]
set_option maxRecDepth 8192 in
theorem root427Candidates5_checked : candidates edge 4 2999 4373 root427Tail6 winner.bound = root427Candidates5 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root427Step5 : scan edge (search edge 3) 4 [2999] 2999 root427Tail5 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail6 winner := by
  rw [root427Tail5, scan, if_neg (by decide), root427Candidates5_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail6
    (if 3 ≤ root427Candidates5.length then search edge 3 [2999, 4373] root427Candidates5 7372 winner else winner) = _
  rw [if_neg (by decide)]
def root427Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root427Candidates6_checked : candidates edge 4 2999 4871 root427Tail7 winner.bound = root427Candidates6 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root427Step6 : scan edge (search edge 3) 4 [2999] 2999 root427Tail6 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail7 winner := by
  rw [root427Tail6, scan, if_neg (by decide), root427Candidates6_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail7
    (if 3 ≤ root427Candidates6.length then search edge 3 [2999, 4871] root427Candidates6 7870 winner else winner) = _
  rw [if_neg (by decide)]
def root427Candidates7 : List Nat := [7349]
set_option maxRecDepth 8192 in
theorem root427Candidates7_checked : candidates edge 4 2999 5003 root427Tail8 winner.bound = root427Candidates7 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root427Step7 : scan edge (search edge 3) 4 [2999] 2999 root427Tail7 winner =
    scan edge (search edge 3) 4 [2999] 2999 root427Tail8 winner := by
  rw [root427Tail7, scan, if_neg (by decide), root427Candidates7_checked]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail8
    (if 3 ≤ root427Candidates7.length then search edge 3 [2999, 5003] root427Candidates7 8002 winner else winner) = _
  rw [if_neg (by decide)]
theorem root427Scan8 : scan edge (search edge 3) 4 [2999] 2999 root427Tail8 winner = winner := by rw [root427Tail8, scan, if_pos (by decide)]
theorem root427Scan7 : scan edge (search edge 3) 4 [2999] 2999 root427Tail7 winner = winner :=
  (root427Step7).trans root427Scan8
theorem root427Scan6 : scan edge (search edge 3) 4 [2999] 2999 root427Tail6 winner = winner :=
  (root427Step6).trans root427Scan7
theorem root427Scan5 : scan edge (search edge 3) 4 [2999] 2999 root427Tail5 winner = winner :=
  (root427Step5).trans root427Scan6
theorem root427Scan4 : scan edge (search edge 3) 4 [2999] 2999 root427Tail4 winner = winner :=
  (root427Step4).trans root427Scan5
theorem root427Scan3 : scan edge (search edge 3) 4 [2999] 2999 root427Tail3 winner = winner :=
  (root427Step3).trans root427Scan4
theorem root427Scan2 : scan edge (search edge 3) 4 [2999] 2999 root427Tail2 winner = winner :=
  (root427Step2).trans root427Scan3
theorem root427Scan1 : scan edge (search edge 3) 4 [2999] 2999 root427Tail1 winner = winner :=
  (root427Step1).trans root427Scan2
theorem root427Scan0 : scan edge (search edge 3) 4 [2999] 2999 root427Tail0 winner = winner :=
  (root427Step0).trans root427Scan1
set_option maxRecDepth 4096 in
theorem root427Branch : search edge 4 [2999] root427Candidates 2999 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2999] 2999 root427Tail0 winner = winner
  exact root427Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
