import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root480Candidates : List Nat := Data.row3457
set_option maxRecDepth 8192 in
theorem root480Candidates_checked : candidates edge 5 0 3457 tail481 winner.bound = root480Candidates := by
  apply candidates_eq_row edge 5 0 3457 tail481 root480Candidates winner.bound tail481_ordered
  · decide
  · decide
  · funext q
    rfl
def root480Tail38 : List Nat := []
def root480Tail37 : List Nat := 22567 :: root480Tail38
def root480Tail36 : List Nat := 21661 :: root480Tail37
def root480Tail35 : List Nat := 21157 :: root480Tail36
def root480Tail34 : List Nat := 20983 :: root480Tail35
def root480Tail33 : List Nat := 19813 :: root480Tail34
def root480Tail32 : List Nat := 18481 :: root480Tail33
def root480Tail31 : List Nat := 18127 :: root480Tail32
def root480Tail30 : List Nat := 16927 :: root480Tail31
def root480Tail29 : List Nat := 16411 :: root480Tail30
def root480Tail28 : List Nat := 16183 :: root480Tail29
def root480Tail27 : List Nat := 15907 :: root480Tail28
def root480Tail26 : List Nat := 15607 :: root480Tail27
def root480Tail25 : List Nat := 14827 :: root480Tail26
def root480Tail24 : List Nat := 13033 :: root480Tail25
def root480Tail23 : List Nat := 12109 :: root480Tail24
def root480Tail22 : List Nat := 11617 :: root480Tail23
def root480Tail21 : List Nat := 11257 :: root480Tail22
def root480Tail20 : List Nat := 11071 :: root480Tail21
def root480Tail19 : List Nat := 10531 :: root480Tail20
def root480Tail18 : List Nat := 9931 :: root480Tail19
def root480Tail17 : List Nat := 8839 :: root480Tail18
def root480Tail16 : List Nat := 8377 :: root480Tail17
def root480Tail15 : List Nat := 8209 :: root480Tail16
def root480Tail14 : List Nat := 7603 :: root480Tail15
def root480Tail13 : List Nat := 6823 :: root480Tail14
def root480Tail12 : List Nat := 6679 :: root480Tail13
def root480Tail11 : List Nat := 6247 :: root480Tail12
def root480Tail10 : List Nat := 5869 :: root480Tail11
def root480Tail9 : List Nat := 5449 :: root480Tail10
def root480Tail8 : List Nat := 5419 :: root480Tail9
def root480Tail7 : List Nat := 5197 :: root480Tail8
def root480Tail6 : List Nat := 5113 :: root480Tail7
def root480Tail5 : List Nat := 4789 :: root480Tail6
def root480Tail4 : List Nat := 4759 :: root480Tail5
def root480Tail3 : List Nat := 4567 :: root480Tail4
def root480Tail2 : List Nat := 4327 :: root480Tail3
def root480Tail1 : List Nat := 4297 :: root480Tail2
def root480Tail0 : List Nat := 3499 :: root480Tail1
def root480Candidates0 : List Nat := [5449, 8377, 9931]
set_option maxRecDepth 8192 in
theorem root480Candidates0_checked : candidates edge 4 3457 3499 root480Tail1 winner.bound = root480Candidates0 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root480Child0 : search edge 3 [3457, 3499] root480Candidates0 6956 winner = winner := by decide
theorem root480Step0 : scan edge (search edge 3) 4 [3457] 3457 root480Tail0 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail1 winner := by
  rw [root480Tail0, scan, if_neg (by decide), root480Candidates0_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail1
    (if 3 ≤ root480Candidates0.length then search edge 3 [3457, 3499] root480Candidates0 6956 winner else winner) = _
  rw [if_pos (by decide), root480Child0]
def root480Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root480Candidates1_checked : candidates edge 4 3457 4297 root480Tail2 winner.bound = root480Candidates1 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root480Step1 : scan edge (search edge 3) 4 [3457] 3457 root480Tail1 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail2 winner := by
  rw [root480Tail1, scan, if_neg (by decide), root480Candidates1_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail2
    (if 3 ≤ root480Candidates1.length then search edge 3 [3457, 4297] root480Candidates1 7754 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates2 : List Nat := [12109]
set_option maxRecDepth 8192 in
theorem root480Candidates2_checked : candidates edge 4 3457 4327 root480Tail3 winner.bound = root480Candidates2 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root480Step2 : scan edge (search edge 3) 4 [3457] 3457 root480Tail2 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail3 winner := by
  rw [root480Tail2, scan, if_neg (by decide), root480Candidates2_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail3
    (if 3 ≤ root480Candidates2.length then search edge 3 [3457, 4327] root480Candidates2 7784 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root480Candidates3_checked : candidates edge 4 3457 4567 root480Tail4 winner.bound = root480Candidates3 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root480Step3 : scan edge (search edge 3) 4 [3457] 3457 root480Tail3 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail4 winner := by
  rw [root480Tail3, scan, if_neg (by decide), root480Candidates3_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail4
    (if 3 ≤ root480Candidates3.length then search edge 3 [3457, 4567] root480Candidates3 8024 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates4 : List Nat := [6823]
set_option maxRecDepth 8192 in
theorem root480Candidates4_checked : candidates edge 4 3457 4759 root480Tail5 winner.bound = root480Candidates4 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root480Step4 : scan edge (search edge 3) 4 [3457] 3457 root480Tail4 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail5 winner := by
  rw [root480Tail4, scan, if_neg (by decide), root480Candidates4_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail5
    (if 3 ≤ root480Candidates4.length then search edge 3 [3457, 4759] root480Candidates4 8216 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates5 : List Nat := [11071, 11617]
set_option maxRecDepth 8192 in
theorem root480Candidates5_checked : candidates edge 4 3457 4789 root480Tail6 winner.bound = root480Candidates5 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root480Step5 : scan edge (search edge 3) 4 [3457] 3457 root480Tail5 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail6 winner := by
  rw [root480Tail5, scan, if_neg (by decide), root480Candidates5_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail6
    (if 3 ≤ root480Candidates5.length then search edge 3 [3457, 4789] root480Candidates5 8246 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates6 : List Nat := [5449, 6823]
set_option maxRecDepth 8192 in
theorem root480Candidates6_checked : candidates edge 4 3457 5113 root480Tail7 winner.bound = root480Candidates6 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root480Step6 : scan edge (search edge 3) 4 [3457] 3457 root480Tail6 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail7 winner := by
  rw [root480Tail6, scan, if_neg (by decide), root480Candidates6_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail7
    (if 3 ≤ root480Candidates6.length then search edge 3 [3457, 5113] root480Candidates6 8570 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates7 : List Nat := [8209]
set_option maxRecDepth 8192 in
theorem root480Candidates7_checked : candidates edge 4 3457 5197 root480Tail8 winner.bound = root480Candidates7 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root480Step7 : scan edge (search edge 3) 4 [3457] 3457 root480Tail7 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail8 winner := by
  rw [root480Tail7, scan, if_neg (by decide), root480Candidates7_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail8
    (if 3 ≤ root480Candidates7.length then search edge 3 [3457, 5197] root480Candidates7 8654 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root480Candidates8_checked : candidates edge 4 3457 5419 root480Tail9 winner.bound = root480Candidates8 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root480Step8 : scan edge (search edge 3) 4 [3457] 3457 root480Tail8 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail9 winner := by
  rw [root480Tail8, scan, if_neg (by decide), root480Candidates8_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail9
    (if 3 ≤ root480Candidates8.length then search edge 3 [3457, 5419] root480Candidates8 8876 winner else winner) = _
  rw [if_neg (by decide)]
def root480Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root480Candidates9_checked : candidates edge 4 3457 5449 root480Tail10 winner.bound = root480Candidates9 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root480Step9 : scan edge (search edge 3) 4 [3457] 3457 root480Tail9 winner =
    scan edge (search edge 3) 4 [3457] 3457 root480Tail10 winner := by
  rw [root480Tail9, scan, if_neg (by decide), root480Candidates9_checked]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail10
    (if 3 ≤ root480Candidates9.length then search edge 3 [3457, 5449] root480Candidates9 8906 winner else winner) = _
  rw [if_neg (by decide)]
theorem root480Scan10 : scan edge (search edge 3) 4 [3457] 3457 root480Tail10 winner = winner := by rw [root480Tail10, scan, if_pos (by decide)]
theorem root480Scan9 : scan edge (search edge 3) 4 [3457] 3457 root480Tail9 winner = winner :=
  (root480Step9).trans root480Scan10
theorem root480Scan8 : scan edge (search edge 3) 4 [3457] 3457 root480Tail8 winner = winner :=
  (root480Step8).trans root480Scan9
theorem root480Scan7 : scan edge (search edge 3) 4 [3457] 3457 root480Tail7 winner = winner :=
  (root480Step7).trans root480Scan8
theorem root480Scan6 : scan edge (search edge 3) 4 [3457] 3457 root480Tail6 winner = winner :=
  (root480Step6).trans root480Scan7
theorem root480Scan5 : scan edge (search edge 3) 4 [3457] 3457 root480Tail5 winner = winner :=
  (root480Step5).trans root480Scan6
theorem root480Scan4 : scan edge (search edge 3) 4 [3457] 3457 root480Tail4 winner = winner :=
  (root480Step4).trans root480Scan5
theorem root480Scan3 : scan edge (search edge 3) 4 [3457] 3457 root480Tail3 winner = winner :=
  (root480Step3).trans root480Scan4
theorem root480Scan2 : scan edge (search edge 3) 4 [3457] 3457 root480Tail2 winner = winner :=
  (root480Step2).trans root480Scan3
theorem root480Scan1 : scan edge (search edge 3) 4 [3457] 3457 root480Tail1 winner = winner :=
  (root480Step1).trans root480Scan2
theorem root480Scan0 : scan edge (search edge 3) 4 [3457] 3457 root480Tail0 winner = winner :=
  (root480Step0).trans root480Scan1
set_option maxRecDepth 4096 in
theorem root480Branch : search edge 4 [3457] root480Candidates 3457 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3457] 3457 root480Tail0 winner = winner
  exact root480Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
