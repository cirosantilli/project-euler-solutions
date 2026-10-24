import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root366Candidates : List Nat := Data.row2521
set_option maxRecDepth 8192 in
theorem root366Candidates_checked : candidates edge 5 0 2521 tail367 winner.bound = root366Candidates := by
  apply candidates_eq_row edge 5 0 2521 tail367 root366Candidates winner.bound tail367_ordered
  · decide
  · decide
  · funext q
    rfl
def root366Tail33 : List Nat := []
def root366Tail32 : List Nat := 21757 :: root366Tail33
def root366Tail31 : List Nat := 19753 :: root366Tail32
def root366Tail30 : List Nat := 19081 :: root366Tail31
def root366Tail29 : List Nat := 18913 :: root366Tail30
def root366Tail28 : List Nat := 18787 :: root366Tail29
def root366Tail27 : List Nat := 16747 :: root366Tail28
def root366Tail26 : List Nat := 16573 :: root366Tail27
def root366Tail25 : List Nat := 15649 :: root366Tail26
def root366Tail24 : List Nat := 13999 :: root366Tail25
def root366Tail23 : List Nat := 13933 :: root366Tail24
def root366Tail22 : List Nat := 12517 :: root366Tail23
def root366Tail21 : List Nat := 12163 :: root366Tail22
def root366Tail20 : List Nat := 12097 :: root366Tail21
def root366Tail19 : List Nat := 11593 :: root366Tail20
def root366Tail18 : List Nat := 11467 :: root366Tail19
def root366Tail17 : List Nat := 11329 :: root366Tail18
def root366Tail16 : List Nat := 10597 :: root366Tail17
def root366Tail15 : List Nat := 10273 :: root366Tail16
def root366Tail14 : List Nat := 9907 :: root366Tail15
def root366Tail13 : List Nat := 9781 :: root366Tail14
def root366Tail12 : List Nat := 8929 :: root366Tail13
def root366Tail11 : List Nat := 8887 :: root366Tail12
def root366Tail10 : List Nat := 7561 :: root366Tail11
def root366Tail9 : List Nat := 6343 :: root366Tail10
def root366Tail8 : List Nat := 5557 :: root366Tail9
def root366Tail7 : List Nat := 4999 :: root366Tail8
def root366Tail6 : List Nat := 4759 :: root366Tail7
def root366Tail5 : List Nat := 4363 :: root366Tail6
def root366Tail4 : List Nat := 4201 :: root366Tail5
def root366Tail3 : List Nat := 3739 :: root366Tail4
def root366Tail2 : List Nat := 3697 :: root366Tail3
def root366Tail1 : List Nat := 3271 :: root366Tail2
def root366Tail0 : List Nat := 2797 :: root366Tail1
def root366Candidates0 : List Nat := [13933]
set_option maxRecDepth 8192 in
theorem root366Candidates0_checked : candidates edge 4 2521 2797 root366Tail1 winner.bound = root366Candidates0 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root366Step0 : scan edge (search edge 3) 4 [2521] 2521 root366Tail0 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail1 winner := by
  rw [root366Tail0, scan, if_neg (by decide), root366Candidates0_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail1
    (if 3 ≤ root366Candidates0.length then search edge 3 [2521, 2797] root366Candidates0 5318 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root366Candidates1_checked : candidates edge 4 2521 3271 root366Tail2 winner.bound = root366Candidates1 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root366Step1 : scan edge (search edge 3) 4 [2521] 2521 root366Tail1 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail2 winner := by
  rw [root366Tail1, scan, if_neg (by decide), root366Candidates1_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail2
    (if 3 ≤ root366Candidates1.length then search edge 3 [2521, 3271] root366Candidates1 5792 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root366Candidates2_checked : candidates edge 4 2521 3697 root366Tail3 winner.bound = root366Candidates2 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
theorem root366Step2 : scan edge (search edge 3) 4 [2521] 2521 root366Tail2 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail3 winner := by
  rw [root366Tail2, scan, if_neg (by decide), root366Candidates2_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail3
    (if 3 ≤ root366Candidates2.length then search edge 3 [2521, 3697] root366Candidates2 6218 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates3 : List Nat := [9781]
set_option maxRecDepth 8192 in
theorem root366Candidates3_checked : candidates edge 4 2521 3739 root366Tail4 winner.bound = root366Candidates3 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root366Step3 : scan edge (search edge 3) 4 [2521] 2521 root366Tail3 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail4 winner := by
  rw [root366Tail3, scan, if_neg (by decide), root366Candidates3_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail4
    (if 3 ≤ root366Candidates3.length then search edge 3 [2521, 3739] root366Candidates3 6260 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates4 : List Nat := [4759, 10597]
set_option maxRecDepth 8192 in
theorem root366Candidates4_checked : candidates edge 4 2521 4201 root366Tail5 winner.bound = root366Candidates4 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root366Step4 : scan edge (search edge 3) 4 [2521] 2521 root366Tail4 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail5 winner := by
  rw [root366Tail4, scan, if_neg (by decide), root366Candidates4_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail5
    (if 3 ≤ root366Candidates4.length then search edge 3 [2521, 4201] root366Candidates4 6722 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root366Candidates5_checked : candidates edge 4 2521 4363 root366Tail6 winner.bound = root366Candidates5 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root366Step5 : scan edge (search edge 3) 4 [2521] 2521 root366Tail5 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail6 winner := by
  rw [root366Tail5, scan, if_neg (by decide), root366Candidates5_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail6
    (if 3 ≤ root366Candidates5.length then search edge 3 [2521, 4363] root366Candidates5 6884 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates6 : List Nat := [12097]
set_option maxRecDepth 8192 in
theorem root366Candidates6_checked : candidates edge 4 2521 4759 root366Tail7 winner.bound = root366Candidates6 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root366Step6 : scan edge (search edge 3) 4 [2521] 2521 root366Tail6 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail7 winner := by
  rw [root366Tail6, scan, if_neg (by decide), root366Candidates6_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail7
    (if 3 ≤ root366Candidates6.length then search edge 3 [2521, 4759] root366Candidates6 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root366Candidates7_checked : candidates edge 4 2521 4999 root366Tail8 winner.bound = root366Candidates7 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root366Step7 : scan edge (search edge 3) 4 [2521] 2521 root366Tail7 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail8 winner := by
  rw [root366Tail7, scan, if_neg (by decide), root366Candidates7_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail8
    (if 3 ≤ root366Candidates7.length then search edge 3 [2521, 4999] root366Candidates7 7520 winner else winner) = _
  rw [if_neg (by decide)]
def root366Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root366Candidates8_checked : candidates edge 4 2521 5557 root366Tail9 winner.bound = root366Candidates8 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root366Step8 : scan edge (search edge 3) 4 [2521] 2521 root366Tail8 winner =
    scan edge (search edge 3) 4 [2521] 2521 root366Tail9 winner := by
  rw [root366Tail8, scan, if_neg (by decide), root366Candidates8_checked]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail9
    (if 3 ≤ root366Candidates8.length then search edge 3 [2521, 5557] root366Candidates8 8078 winner else winner) = _
  rw [if_neg (by decide)]
theorem root366Scan9 : scan edge (search edge 3) 4 [2521] 2521 root366Tail9 winner = winner := by rw [root366Tail9, scan, if_pos (by decide)]
theorem root366Scan8 : scan edge (search edge 3) 4 [2521] 2521 root366Tail8 winner = winner :=
  (root366Step8).trans root366Scan9
theorem root366Scan7 : scan edge (search edge 3) 4 [2521] 2521 root366Tail7 winner = winner :=
  (root366Step7).trans root366Scan8
theorem root366Scan6 : scan edge (search edge 3) 4 [2521] 2521 root366Tail6 winner = winner :=
  (root366Step6).trans root366Scan7
theorem root366Scan5 : scan edge (search edge 3) 4 [2521] 2521 root366Tail5 winner = winner :=
  (root366Step5).trans root366Scan6
theorem root366Scan4 : scan edge (search edge 3) 4 [2521] 2521 root366Tail4 winner = winner :=
  (root366Step4).trans root366Scan5
theorem root366Scan3 : scan edge (search edge 3) 4 [2521] 2521 root366Tail3 winner = winner :=
  (root366Step3).trans root366Scan4
theorem root366Scan2 : scan edge (search edge 3) 4 [2521] 2521 root366Tail2 winner = winner :=
  (root366Step2).trans root366Scan3
theorem root366Scan1 : scan edge (search edge 3) 4 [2521] 2521 root366Tail1 winner = winner :=
  (root366Step1).trans root366Scan2
theorem root366Scan0 : scan edge (search edge 3) 4 [2521] 2521 root366Tail0 winner = winner :=
  (root366Step0).trans root366Scan1
set_option maxRecDepth 4096 in
theorem root366Branch : search edge 4 [2521] root366Candidates 2521 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2521] 2521 root366Tail0 winner = winner
  exact root366Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
