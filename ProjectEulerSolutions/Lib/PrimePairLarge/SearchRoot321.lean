import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root321Candidates : List Nat := Data.row2143
set_option maxRecDepth 8192 in
theorem root321Candidates_checked : candidates edge 5 0 2143 tail322 winner.bound = root321Candidates := by
  apply candidates_eq_row edge 5 0 2143 tail322 root321Candidates winner.bound tail322_ordered
  · decide
  · decide
  · funext q
    rfl
def root321Tail35 : List Nat := []
def root321Tail34 : List Nat := 23251 :: root321Tail35
def root321Tail33 : List Nat := 22861 :: root321Tail34
def root321Tail32 : List Nat := 22777 :: root321Tail33
def root321Tail31 : List Nat := 22063 :: root321Tail32
def root321Tail30 : List Nat := 22039 :: root321Tail31
def root321Tail29 : List Nat := 21937 :: root321Tail30
def root321Tail28 : List Nat := 20959 :: root321Tail29
def root321Tail27 : List Nat := 20749 :: root321Tail28
def root321Tail26 : List Nat := 20731 :: root321Tail27
def root321Tail25 : List Nat := 20521 :: root321Tail26
def root321Tail24 : List Nat := 19891 :: root321Tail25
def root321Tail23 : List Nat := 19603 :: root321Tail24
def root321Tail22 : List Nat := 18523 :: root321Tail23
def root321Tail21 : List Nat := 17989 :: root321Tail22
def root321Tail20 : List Nat := 17137 :: root321Tail21
def root321Tail19 : List Nat := 15427 :: root321Tail20
def root321Tail18 : List Nat := 14629 :: root321Tail19
def root321Tail17 : List Nat := 13999 :: root321Tail18
def root321Tail16 : List Nat := 13399 :: root321Tail17
def root321Tail15 : List Nat := 13381 :: root321Tail16
def root321Tail14 : List Nat := 12613 :: root321Tail15
def root321Tail13 : List Nat := 11941 :: root321Tail14
def root321Tail12 : List Nat := 9283 :: root321Tail13
def root321Tail11 : List Nat := 9109 :: root321Tail12
def root321Tail10 : List Nat := 8863 :: root321Tail11
def root321Tail9 : List Nat := 8089 :: root321Tail10
def root321Tail8 : List Nat := 8017 :: root321Tail9
def root321Tail7 : List Nat := 6871 :: root321Tail8
def root321Tail6 : List Nat := 6691 :: root321Tail7
def root321Tail5 : List Nat := 6301 :: root321Tail6
def root321Tail4 : List Nat := 6229 :: root321Tail5
def root321Tail3 : List Nat := 4801 :: root321Tail4
def root321Tail2 : List Nat := 4657 :: root321Tail3
def root321Tail1 : List Nat := 2347 :: root321Tail2
def root321Tail0 : List Nat := 2239 :: root321Tail1
def root321Candidates0 : List Nat := [9283, 18523]
set_option maxRecDepth 8192 in
theorem root321Candidates0_checked : candidates edge 4 2143 2239 root321Tail1 winner.bound = root321Candidates0 := by
  unfold candidates
  have row : edge 2239 = (SortedNatTree.contains FastRows.row2239) := by
    change Data.row2239.contains = (SortedNatTree.contains FastRows.row2239)
    exact FastRows.row2239_checked.symm
  rw [row]
  decide
theorem root321Step0 : scan edge (search edge 3) 4 [2143] 2143 root321Tail0 winner =
    scan edge (search edge 3) 4 [2143] 2143 root321Tail1 winner := by
  rw [root321Tail0, scan, if_neg (by decide), root321Candidates0_checked]
  change scan edge (search edge 3) 4 [2143] 2143 root321Tail1
    (if 3 ≤ root321Candidates0.length then search edge 3 [2143, 2239] root321Candidates0 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root321Candidates1 : List Nat := [20749]
set_option maxRecDepth 8192 in
theorem root321Candidates1_checked : candidates edge 4 2143 2347 root321Tail2 winner.bound = root321Candidates1 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
theorem root321Step1 : scan edge (search edge 3) 4 [2143] 2143 root321Tail1 winner =
    scan edge (search edge 3) 4 [2143] 2143 root321Tail2 winner := by
  rw [root321Tail1, scan, if_neg (by decide), root321Candidates1_checked]
  change scan edge (search edge 3) 4 [2143] 2143 root321Tail2
    (if 3 ≤ root321Candidates1.length then search edge 3 [2143, 2347] root321Candidates1 4490 winner else winner) = _
  rw [if_neg (by decide)]
def root321Candidates2 : List Nat := [9283]
set_option maxRecDepth 8192 in
theorem root321Candidates2_checked : candidates edge 4 2143 4657 root321Tail3 winner.bound = root321Candidates2 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root321Step2 : scan edge (search edge 3) 4 [2143] 2143 root321Tail2 winner =
    scan edge (search edge 3) 4 [2143] 2143 root321Tail3 winner := by
  rw [root321Tail2, scan, if_neg (by decide), root321Candidates2_checked]
  change scan edge (search edge 3) 4 [2143] 2143 root321Tail3
    (if 3 ≤ root321Candidates2.length then search edge 3 [2143, 4657] root321Candidates2 6800 winner else winner) = _
  rw [if_neg (by decide)]
def root321Candidates3 : List Nat := [6229]
set_option maxRecDepth 8192 in
theorem root321Candidates3_checked : candidates edge 4 2143 4801 root321Tail4 winner.bound = root321Candidates3 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root321Step3 : scan edge (search edge 3) 4 [2143] 2143 root321Tail3 winner =
    scan edge (search edge 3) 4 [2143] 2143 root321Tail4 winner := by
  rw [root321Tail3, scan, if_neg (by decide), root321Candidates3_checked]
  change scan edge (search edge 3) 4 [2143] 2143 root321Tail4
    (if 3 ≤ root321Candidates3.length then search edge 3 [2143, 4801] root321Candidates3 6944 winner else winner) = _
  rw [if_neg (by decide)]
theorem root321Scan4 : scan edge (search edge 3) 4 [2143] 2143 root321Tail4 winner = winner := by rw [root321Tail4, scan, if_pos (by decide)]
theorem root321Scan3 : scan edge (search edge 3) 4 [2143] 2143 root321Tail3 winner = winner :=
  (root321Step3).trans root321Scan4
theorem root321Scan2 : scan edge (search edge 3) 4 [2143] 2143 root321Tail2 winner = winner :=
  (root321Step2).trans root321Scan3
theorem root321Scan1 : scan edge (search edge 3) 4 [2143] 2143 root321Tail1 winner = winner :=
  (root321Step1).trans root321Scan2
theorem root321Scan0 : scan edge (search edge 3) 4 [2143] 2143 root321Tail0 winner = winner :=
  (root321Step0).trans root321Scan1
set_option maxRecDepth 4096 in
theorem root321Branch : search edge 4 [2143] root321Candidates 2143 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2143] 2143 root321Tail0 winner = winner
  exact root321Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
