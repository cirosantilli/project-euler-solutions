import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root286Candidates : List Nat := Data.row1879
set_option maxRecDepth 8192 in
theorem root286Candidates_checked : candidates edge 5 0 1879 tail287 winner.bound = root286Candidates := by
  apply candidates_eq_row edge 5 0 1879 tail287 root286Candidates winner.bound tail287_ordered
  · decide
  · decide
  · funext q
    rfl
def root286Tail37 : List Nat := []
def root286Tail36 : List Nat := 24043 :: root286Tail37
def root286Tail35 : List Nat := 22621 :: root286Tail36
def root286Tail34 : List Nat := 21817 :: root286Tail35
def root286Tail33 : List Nat := 19267 :: root286Tail34
def root286Tail32 : List Nat := 19183 :: root286Tail33
def root286Tail31 : List Nat := 17791 :: root286Tail32
def root286Tail30 : List Nat := 17203 :: root286Tail31
def root286Tail29 : List Nat := 16903 :: root286Tail30
def root286Tail28 : List Nat := 16189 :: root286Tail29
def root286Tail27 : List Nat := 15619 :: root286Tail28
def root286Tail26 : List Nat := 15307 :: root286Tail27
def root286Tail25 : List Nat := 13513 :: root286Tail26
def root286Tail24 : List Nat := 13183 :: root286Tail25
def root286Tail23 : List Nat := 12763 :: root286Tail24
def root286Tail22 : List Nat := 12541 :: root286Tail23
def root286Tail21 : List Nat := 12457 :: root286Tail22
def root286Tail20 : List Nat := 12409 :: root286Tail21
def root286Tail19 : List Nat := 12379 :: root286Tail20
def root286Tail18 : List Nat := 12343 :: root286Tail19
def root286Tail17 : List Nat := 12301 :: root286Tail18
def root286Tail16 : List Nat := 10267 :: root286Tail17
def root286Tail15 : List Nat := 9967 :: root286Tail16
def root286Tail14 : List Nat := 9883 :: root286Tail15
def root286Tail13 : List Nat := 9103 :: root286Tail14
def root286Tail12 : List Nat := 8971 :: root286Tail13
def root286Tail11 : List Nat := 8707 :: root286Tail12
def root286Tail10 : List Nat := 7951 :: root286Tail11
def root286Tail9 : List Nat := 7411 :: root286Tail10
def root286Tail8 : List Nat := 7297 :: root286Tail9
def root286Tail7 : List Nat := 6823 :: root286Tail8
def root286Tail6 : List Nat := 4861 :: root286Tail7
def root286Tail5 : List Nat := 3331 :: root286Tail6
def root286Tail4 : List Nat := 2971 :: root286Tail5
def root286Tail3 : List Nat := 2713 :: root286Tail4
def root286Tail2 : List Nat := 2341 :: root286Tail3
def root286Tail1 : List Nat := 2281 :: root286Tail2
def root286Tail0 : List Nat := 1999 :: root286Tail1
def root286Candidates0 : List Nat := [12379, 12541]
set_option maxRecDepth 8192 in
theorem root286Candidates0_checked : candidates edge 4 1879 1999 root286Tail1 winner.bound = root286Candidates0 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root286Step0 : scan edge (search edge 3) 4 [1879] 1879 root286Tail0 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail1 winner := by
  rw [root286Tail0, scan, if_neg (by decide), root286Candidates0_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail1
    (if 3 ≤ root286Candidates0.length then search edge 3 [1879, 1999] root286Candidates0 3878 winner else winner) = _
  rw [if_neg (by decide)]
def root286Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root286Candidates1_checked : candidates edge 4 1879 2281 root286Tail2 winner.bound = root286Candidates1 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root286Step1 : scan edge (search edge 3) 4 [1879] 1879 root286Tail1 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail2 winner := by
  rw [root286Tail1, scan, if_neg (by decide), root286Candidates1_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail2
    (if 3 ≤ root286Candidates1.length then search edge 3 [1879, 2281] root286Candidates1 4160 winner else winner) = _
  rw [if_neg (by decide)]
def root286Candidates2 : List Nat := [12379, 12409]
set_option maxRecDepth 8192 in
theorem root286Candidates2_checked : candidates edge 4 1879 2341 root286Tail3 winner.bound = root286Candidates2 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root286Step2 : scan edge (search edge 3) 4 [1879] 1879 root286Tail2 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail3 winner := by
  rw [root286Tail2, scan, if_neg (by decide), root286Candidates2_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail3
    (if 3 ≤ root286Candidates2.length then search edge 3 [1879, 2341] root286Candidates2 4220 winner else winner) = _
  rw [if_neg (by decide)]
def root286Candidates3 : List Nat := [16189]
set_option maxRecDepth 8192 in
theorem root286Candidates3_checked : candidates edge 4 1879 2713 root286Tail4 winner.bound = root286Candidates3 := by
  unfold candidates
  have row : edge 2713 = (SortedNatTree.contains FastRows.row2713) := by
    change Data.row2713.contains = (SortedNatTree.contains FastRows.row2713)
    exact FastRows.row2713_checked.symm
  rw [row]
  decide
theorem root286Step3 : scan edge (search edge 3) 4 [1879] 1879 root286Tail3 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail4 winner := by
  rw [root286Tail3, scan, if_neg (by decide), root286Candidates3_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail4
    (if 3 ≤ root286Candidates3.length then search edge 3 [1879, 2713] root286Candidates3 4592 winner else winner) = _
  rw [if_neg (by decide)]
def root286Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root286Candidates4_checked : candidates edge 4 1879 2971 root286Tail5 winner.bound = root286Candidates4 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
theorem root286Step4 : scan edge (search edge 3) 4 [1879] 1879 root286Tail4 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail5 winner := by
  rw [root286Tail4, scan, if_neg (by decide), root286Candidates4_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail5
    (if 3 ≤ root286Candidates4.length then search edge 3 [1879, 2971] root286Candidates4 4850 winner else winner) = _
  rw [if_neg (by decide)]
def root286Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root286Candidates5_checked : candidates edge 4 1879 3331 root286Tail6 winner.bound = root286Candidates5 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root286Step5 : scan edge (search edge 3) 4 [1879] 1879 root286Tail5 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail6 winner := by
  rw [root286Tail5, scan, if_neg (by decide), root286Candidates5_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail6
    (if 3 ≤ root286Candidates5.length then search edge 3 [1879, 3331] root286Candidates5 5210 winner else winner) = _
  rw [if_neg (by decide)]
def root286Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root286Candidates6_checked : candidates edge 4 1879 4861 root286Tail7 winner.bound = root286Candidates6 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root286Step6 : scan edge (search edge 3) 4 [1879] 1879 root286Tail6 winner =
    scan edge (search edge 3) 4 [1879] 1879 root286Tail7 winner := by
  rw [root286Tail6, scan, if_neg (by decide), root286Candidates6_checked]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail7
    (if 3 ≤ root286Candidates6.length then search edge 3 [1879, 4861] root286Candidates6 6740 winner else winner) = _
  rw [if_neg (by decide)]
theorem root286Scan7 : scan edge (search edge 3) 4 [1879] 1879 root286Tail7 winner = winner := by rw [root286Tail7, scan, if_pos (by decide)]
theorem root286Scan6 : scan edge (search edge 3) 4 [1879] 1879 root286Tail6 winner = winner :=
  (root286Step6).trans root286Scan7
theorem root286Scan5 : scan edge (search edge 3) 4 [1879] 1879 root286Tail5 winner = winner :=
  (root286Step5).trans root286Scan6
theorem root286Scan4 : scan edge (search edge 3) 4 [1879] 1879 root286Tail4 winner = winner :=
  (root286Step4).trans root286Scan5
theorem root286Scan3 : scan edge (search edge 3) 4 [1879] 1879 root286Tail3 winner = winner :=
  (root286Step3).trans root286Scan4
theorem root286Scan2 : scan edge (search edge 3) 4 [1879] 1879 root286Tail2 winner = winner :=
  (root286Step2).trans root286Scan3
theorem root286Scan1 : scan edge (search edge 3) 4 [1879] 1879 root286Tail1 winner = winner :=
  (root286Step1).trans root286Scan2
theorem root286Scan0 : scan edge (search edge 3) 4 [1879] 1879 root286Tail0 winner = winner :=
  (root286Step0).trans root286Scan1
set_option maxRecDepth 4096 in
theorem root286Branch : search edge 4 [1879] root286Candidates 1879 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1879] 1879 root286Tail0 winner = winner
  exact root286Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
