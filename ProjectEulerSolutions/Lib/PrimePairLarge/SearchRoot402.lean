import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root402Candidates : List Nat := Data.row2789
set_option maxRecDepth 8192 in
theorem root402Candidates_checked : candidates edge 5 0 2789 tail403 winner.bound = root402Candidates := by
  apply candidates_eq_row edge 5 0 2789 tail403 root402Candidates winner.bound tail403_ordered
  · decide
  · decide
  · funext q
    rfl
def root402Tail35 : List Nat := []
def root402Tail34 : List Nat := 22901 :: root402Tail35
def root402Tail33 : List Nat := 21107 :: root402Tail34
def root402Tail32 : List Nat := 21089 :: root402Tail33
def root402Tail31 : List Nat := 21017 :: root402Tail32
def root402Tail30 : List Nat := 18443 :: root402Tail31
def root402Tail29 : List Nat := 17477 :: root402Tail30
def root402Tail28 : List Nat := 16691 :: root402Tail29
def root402Tail27 : List Nat := 16061 :: root402Tail28
def root402Tail26 : List Nat := 15887 :: root402Tail27
def root402Tail25 : List Nat := 15671 :: root402Tail26
def root402Tail24 : List Nat := 14543 :: root402Tail25
def root402Tail23 : List Nat := 13841 :: root402Tail24
def root402Tail22 : List Nat := 12197 :: root402Tail23
def root402Tail21 : List Nat := 10601 :: root402Tail22
def root402Tail20 : List Nat := 9497 :: root402Tail21
def root402Tail19 : List Nat := 9173 :: root402Tail20
def root402Tail18 : List Nat := 8951 :: root402Tail19
def root402Tail17 : List Nat := 8831 :: root402Tail18
def root402Tail16 : List Nat := 8537 :: root402Tail17
def root402Tail15 : List Nat := 8447 :: root402Tail16
def root402Tail14 : List Nat := 8369 :: root402Tail15
def root402Tail13 : List Nat := 7691 :: root402Tail14
def root402Tail12 : List Nat := 6689 :: root402Tail13
def root402Tail11 : List Nat := 6563 :: root402Tail12
def root402Tail10 : List Nat := 6299 :: root402Tail11
def root402Tail9 : List Nat := 5639 :: root402Tail10
def root402Tail8 : List Nat := 5471 :: root402Tail9
def root402Tail7 : List Nat := 5051 :: root402Tail8
def root402Tail6 : List Nat := 4703 :: root402Tail7
def root402Tail5 : List Nat := 4673 :: root402Tail6
def root402Tail4 : List Nat := 4133 :: root402Tail5
def root402Tail3 : List Nat := 4007 :: root402Tail4
def root402Tail2 : List Nat := 3461 :: root402Tail3
def root402Tail1 : List Nat := 3323 :: root402Tail2
def root402Tail0 : List Nat := 2861 :: root402Tail1
def root402Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root402Candidates0_checked : candidates edge 4 2789 2861 root402Tail1 winner.bound = root402Candidates0 := by
  unfold candidates
  have row : edge 2861 = (SortedNatTree.contains FastRows.row2861) := by
    change Data.row2861.contains = (SortedNatTree.contains FastRows.row2861)
    exact FastRows.row2861_checked.symm
  rw [row]
  decide
theorem root402Step0 : scan edge (search edge 3) 4 [2789] 2789 root402Tail0 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail1 winner := by
  rw [root402Tail0, scan, if_neg (by decide), root402Candidates0_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail1
    (if 3 ≤ root402Candidates0.length then search edge 3 [2789, 2861] root402Candidates0 5650 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root402Candidates1_checked : candidates edge 4 2789 3323 root402Tail2 winner.bound = root402Candidates1 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
theorem root402Step1 : scan edge (search edge 3) 4 [2789] 2789 root402Tail1 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail2 winner := by
  rw [root402Tail1, scan, if_neg (by decide), root402Candidates1_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail2
    (if 3 ≤ root402Candidates1.length then search edge 3 [2789, 3323] root402Candidates1 6112 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root402Candidates2_checked : candidates edge 4 2789 3461 root402Tail3 winner.bound = root402Candidates2 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root402Step2 : scan edge (search edge 3) 4 [2789] 2789 root402Tail2 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail3 winner := by
  rw [root402Tail2, scan, if_neg (by decide), root402Candidates2_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail3
    (if 3 ≤ root402Candidates2.length then search edge 3 [2789, 3461] root402Candidates2 6250 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates3 : List Nat := [5639, 8369]
set_option maxRecDepth 8192 in
theorem root402Candidates3_checked : candidates edge 4 2789 4007 root402Tail4 winner.bound = root402Candidates3 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
theorem root402Step3 : scan edge (search edge 3) 4 [2789] 2789 root402Tail3 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail4 winner := by
  rw [root402Tail3, scan, if_neg (by decide), root402Candidates3_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail4
    (if 3 ≤ root402Candidates3.length then search edge 3 [2789, 4007] root402Candidates3 6796 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates4 : List Nat := [9173, 16061]
set_option maxRecDepth 8192 in
theorem root402Candidates4_checked : candidates edge 4 2789 4133 root402Tail5 winner.bound = root402Candidates4 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root402Step4 : scan edge (search edge 3) 4 [2789] 2789 root402Tail4 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail5 winner := by
  rw [root402Tail4, scan, if_neg (by decide), root402Candidates4_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail5
    (if 3 ≤ root402Candidates4.length then search edge 3 [2789, 4133] root402Candidates4 6922 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates5 : List Nat := [10601]
set_option maxRecDepth 8192 in
theorem root402Candidates5_checked : candidates edge 4 2789 4673 root402Tail6 winner.bound = root402Candidates5 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root402Step5 : scan edge (search edge 3) 4 [2789] 2789 root402Tail5 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail6 winner := by
  rw [root402Tail5, scan, if_neg (by decide), root402Candidates5_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail6
    (if 3 ≤ root402Candidates5.length then search edge 3 [2789, 4673] root402Candidates5 7462 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root402Candidates6_checked : candidates edge 4 2789 4703 root402Tail7 winner.bound = root402Candidates6 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root402Step6 : scan edge (search edge 3) 4 [2789] 2789 root402Tail6 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail7 winner := by
  rw [root402Tail6, scan, if_neg (by decide), root402Candidates6_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail7
    (if 3 ≤ root402Candidates6.length then search edge 3 [2789, 4703] root402Candidates6 7492 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates7 : List Nat := [9173]
set_option maxRecDepth 8192 in
theorem root402Candidates7_checked : candidates edge 4 2789 5051 root402Tail8 winner.bound = root402Candidates7 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root402Step7 : scan edge (search edge 3) 4 [2789] 2789 root402Tail7 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail8 winner := by
  rw [root402Tail7, scan, if_neg (by decide), root402Candidates7_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail8
    (if 3 ≤ root402Candidates7.length then search edge 3 [2789, 5051] root402Candidates7 7840 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates8 : List Nat := [16061]
set_option maxRecDepth 8192 in
theorem root402Candidates8_checked : candidates edge 4 2789 5471 root402Tail9 winner.bound = root402Candidates8 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root402Step8 : scan edge (search edge 3) 4 [2789] 2789 root402Tail8 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail9 winner := by
  rw [root402Tail8, scan, if_neg (by decide), root402Candidates8_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail9
    (if 3 ≤ root402Candidates8.length then search edge 3 [2789, 5471] root402Candidates8 8260 winner else winner) = _
  rw [if_neg (by decide)]
def root402Candidates9 : List Nat := [6563, 13841]
set_option maxRecDepth 8192 in
theorem root402Candidates9_checked : candidates edge 4 2789 5639 root402Tail10 winner.bound = root402Candidates9 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root402Step9 : scan edge (search edge 3) 4 [2789] 2789 root402Tail9 winner =
    scan edge (search edge 3) 4 [2789] 2789 root402Tail10 winner := by
  rw [root402Tail9, scan, if_neg (by decide), root402Candidates9_checked]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail10
    (if 3 ≤ root402Candidates9.length then search edge 3 [2789, 5639] root402Candidates9 8428 winner else winner) = _
  rw [if_neg (by decide)]
theorem root402Scan10 : scan edge (search edge 3) 4 [2789] 2789 root402Tail10 winner = winner := by rw [root402Tail10, scan, if_pos (by decide)]
theorem root402Scan9 : scan edge (search edge 3) 4 [2789] 2789 root402Tail9 winner = winner :=
  (root402Step9).trans root402Scan10
theorem root402Scan8 : scan edge (search edge 3) 4 [2789] 2789 root402Tail8 winner = winner :=
  (root402Step8).trans root402Scan9
theorem root402Scan7 : scan edge (search edge 3) 4 [2789] 2789 root402Tail7 winner = winner :=
  (root402Step7).trans root402Scan8
theorem root402Scan6 : scan edge (search edge 3) 4 [2789] 2789 root402Tail6 winner = winner :=
  (root402Step6).trans root402Scan7
theorem root402Scan5 : scan edge (search edge 3) 4 [2789] 2789 root402Tail5 winner = winner :=
  (root402Step5).trans root402Scan6
theorem root402Scan4 : scan edge (search edge 3) 4 [2789] 2789 root402Tail4 winner = winner :=
  (root402Step4).trans root402Scan5
theorem root402Scan3 : scan edge (search edge 3) 4 [2789] 2789 root402Tail3 winner = winner :=
  (root402Step3).trans root402Scan4
theorem root402Scan2 : scan edge (search edge 3) 4 [2789] 2789 root402Tail2 winner = winner :=
  (root402Step2).trans root402Scan3
theorem root402Scan1 : scan edge (search edge 3) 4 [2789] 2789 root402Tail1 winner = winner :=
  (root402Step1).trans root402Scan2
theorem root402Scan0 : scan edge (search edge 3) 4 [2789] 2789 root402Tail0 winner = winner :=
  (root402Step0).trans root402Scan1
set_option maxRecDepth 4096 in
theorem root402Branch : search edge 4 [2789] root402Candidates 2789 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2789] 2789 root402Tail0 winner = winner
  exact root402Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
