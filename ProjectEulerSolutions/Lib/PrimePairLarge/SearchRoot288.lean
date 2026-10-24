import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root288Candidates : List Nat := Data.row1901
set_option maxRecDepth 8192 in
theorem root288Candidates_checked : candidates edge 5 0 1901 tail289 winner.bound = root288Candidates := by
  apply candidates_eq_row edge 5 0 1901 tail289 root288Candidates winner.bound tail289_ordered
  · decide
  · decide
  · funext q
    rfl
def root288Tail36 : List Nat := []
def root288Tail35 : List Nat := 22643 :: root288Tail36
def root288Tail34 : List Nat := 20939 :: root288Tail35
def root288Tail33 : List Nat := 20753 :: root288Tail34
def root288Tail32 : List Nat := 19913 :: root288Tail33
def root288Tail31 : List Nat := 19751 :: root288Tail32
def root288Tail30 : List Nat := 18947 :: root288Tail31
def root288Tail29 : List Nat := 18371 :: root288Tail30
def root288Tail28 : List Nat := 18041 :: root288Tail29
def root288Tail27 : List Nat := 17489 :: root288Tail28
def root288Tail26 : List Nat := 15971 :: root288Tail27
def root288Tail25 : List Nat := 15923 :: root288Tail26
def root288Tail24 : List Nat := 15383 :: root288Tail25
def root288Tail23 : List Nat := 14957 :: root288Tail24
def root288Tail22 : List Nat := 14891 :: root288Tail23
def root288Tail21 : List Nat := 14669 :: root288Tail22
def root288Tail20 : List Nat := 10889 :: root288Tail21
def root288Tail19 : List Nat := 10313 :: root288Tail20
def root288Tail18 : List Nat := 9851 :: root288Tail19
def root288Tail17 : List Nat := 9587 :: root288Tail18
def root288Tail16 : List Nat := 9467 :: root288Tail17
def root288Tail15 : List Nat := 9431 :: root288Tail16
def root288Tail14 : List Nat := 9227 :: root288Tail15
def root288Tail13 : List Nat := 8837 :: root288Tail14
def root288Tail12 : List Nat := 8663 :: root288Tail13
def root288Tail11 : List Nat := 8243 :: root288Tail12
def root288Tail10 : List Nat := 7949 :: root288Tail11
def root288Tail9 : List Nat := 7589 :: root288Tail10
def root288Tail8 : List Nat := 7151 :: root288Tail9
def root288Tail7 : List Nat := 5861 :: root288Tail8
def root288Tail6 : List Nat := 4817 :: root288Tail7
def root288Tail5 : List Nat := 4019 :: root288Tail6
def root288Tail4 : List Nat := 3221 :: root288Tail5
def root288Tail3 : List Nat := 2969 :: root288Tail4
def root288Tail2 : List Nat := 2801 :: root288Tail3
def root288Tail1 : List Nat := 2621 :: root288Tail2
def root288Tail0 : List Nat := 2153 :: root288Tail1
def root288Candidates0 : List Nat := [2969, 7151, 18041]
set_option maxRecDepth 8192 in
theorem root288Candidates0_checked : candidates edge 4 1901 2153 root288Tail1 winner.bound = root288Candidates0 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root288Child0 : search edge 3 [1901, 2153] root288Candidates0 4054 winner = winner := by decide
theorem root288Step0 : scan edge (search edge 3) 4 [1901] 1901 root288Tail0 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail1 winner := by
  rw [root288Tail0, scan, if_neg (by decide), root288Candidates0_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail1
    (if 3 ≤ root288Candidates0.length then search edge 3 [1901, 2153] root288Candidates0 4054 winner else winner) = _
  rw [if_pos (by decide), root288Child0]
def root288Candidates1 : List Nat := [7151, 8663]
set_option maxRecDepth 8192 in
theorem root288Candidates1_checked : candidates edge 4 1901 2621 root288Tail2 winner.bound = root288Candidates1 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root288Step1 : scan edge (search edge 3) 4 [1901] 1901 root288Tail1 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail2 winner := by
  rw [root288Tail1, scan, if_neg (by decide), root288Candidates1_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail2
    (if 3 ≤ root288Candidates1.length then search edge 3 [1901, 2621] root288Candidates1 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root288Candidates2 : List Nat := [3221]
set_option maxRecDepth 8192 in
theorem root288Candidates2_checked : candidates edge 4 1901 2801 root288Tail3 winner.bound = root288Candidates2 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
theorem root288Step2 : scan edge (search edge 3) 4 [1901] 1901 root288Tail2 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail3 winner := by
  rw [root288Tail2, scan, if_neg (by decide), root288Candidates2_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail3
    (if 3 ≤ root288Candidates2.length then search edge 3 [1901, 2801] root288Candidates2 4702 winner else winner) = _
  rw [if_neg (by decide)]
def root288Candidates3 : List Nat := [15383]
set_option maxRecDepth 8192 in
theorem root288Candidates3_checked : candidates edge 4 1901 2969 root288Tail4 winner.bound = root288Candidates3 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root288Step3 : scan edge (search edge 3) 4 [1901] 1901 root288Tail3 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail4 winner := by
  rw [root288Tail3, scan, if_neg (by decide), root288Candidates3_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail4
    (if 3 ≤ root288Candidates3.length then search edge 3 [1901, 2969] root288Candidates3 4870 winner else winner) = _
  rw [if_neg (by decide)]
def root288Candidates4 : List Nat := [4817]
set_option maxRecDepth 8192 in
theorem root288Candidates4_checked : candidates edge 4 1901 3221 root288Tail5 winner.bound = root288Candidates4 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root288Step4 : scan edge (search edge 3) 4 [1901] 1901 root288Tail4 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail5 winner := by
  rw [root288Tail4, scan, if_neg (by decide), root288Candidates4_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail5
    (if 3 ≤ root288Candidates4.length then search edge 3 [1901, 3221] root288Candidates4 5122 winner else winner) = _
  rw [if_neg (by decide)]
def root288Candidates5 : List Nat := [17489]
set_option maxRecDepth 8192 in
theorem root288Candidates5_checked : candidates edge 4 1901 4019 root288Tail6 winner.bound = root288Candidates5 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root288Step5 : scan edge (search edge 3) 4 [1901] 1901 root288Tail5 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail6 winner := by
  rw [root288Tail5, scan, if_neg (by decide), root288Candidates5_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail6
    (if 3 ≤ root288Candidates5.length then search edge 3 [1901, 4019] root288Candidates5 5920 winner else winner) = _
  rw [if_neg (by decide)]
def root288Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root288Candidates6_checked : candidates edge 4 1901 4817 root288Tail7 winner.bound = root288Candidates6 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root288Step6 : scan edge (search edge 3) 4 [1901] 1901 root288Tail6 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail7 winner := by
  rw [root288Tail6, scan, if_neg (by decide), root288Candidates6_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail7
    (if 3 ≤ root288Candidates6.length then search edge 3 [1901, 4817] root288Candidates6 6718 winner else winner) = _
  rw [if_neg (by decide)]
def root288Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root288Candidates7_checked : candidates edge 4 1901 5861 root288Tail8 winner.bound = root288Candidates7 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root288Step7 : scan edge (search edge 3) 4 [1901] 1901 root288Tail7 winner =
    scan edge (search edge 3) 4 [1901] 1901 root288Tail8 winner := by
  rw [root288Tail7, scan, if_neg (by decide), root288Candidates7_checked]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail8
    (if 3 ≤ root288Candidates7.length then search edge 3 [1901, 5861] root288Candidates7 7762 winner else winner) = _
  rw [if_neg (by decide)]
theorem root288Scan8 : scan edge (search edge 3) 4 [1901] 1901 root288Tail8 winner = winner := by rw [root288Tail8, scan, if_pos (by decide)]
theorem root288Scan7 : scan edge (search edge 3) 4 [1901] 1901 root288Tail7 winner = winner :=
  (root288Step7).trans root288Scan8
theorem root288Scan6 : scan edge (search edge 3) 4 [1901] 1901 root288Tail6 winner = winner :=
  (root288Step6).trans root288Scan7
theorem root288Scan5 : scan edge (search edge 3) 4 [1901] 1901 root288Tail5 winner = winner :=
  (root288Step5).trans root288Scan6
theorem root288Scan4 : scan edge (search edge 3) 4 [1901] 1901 root288Tail4 winner = winner :=
  (root288Step4).trans root288Scan5
theorem root288Scan3 : scan edge (search edge 3) 4 [1901] 1901 root288Tail3 winner = winner :=
  (root288Step3).trans root288Scan4
theorem root288Scan2 : scan edge (search edge 3) 4 [1901] 1901 root288Tail2 winner = winner :=
  (root288Step2).trans root288Scan3
theorem root288Scan1 : scan edge (search edge 3) 4 [1901] 1901 root288Tail1 winner = winner :=
  (root288Step1).trans root288Scan2
theorem root288Scan0 : scan edge (search edge 3) 4 [1901] 1901 root288Tail0 winner = winner :=
  (root288Step0).trans root288Scan1
set_option maxRecDepth 4096 in
theorem root288Branch : search edge 4 [1901] root288Candidates 1901 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1901] 1901 root288Tail0 winner = winner
  exact root288Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
