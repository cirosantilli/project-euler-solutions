import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root360Candidates : List Nat := Data.row2447
set_option maxRecDepth 8192 in
theorem root360Candidates_checked : candidates edge 5 0 2447 tail361 winner.bound = root360Candidates := by
  apply candidates_eq_row edge 5 0 2447 tail361 root360Candidates winner.bound tail361_ordered
  · decide
  · decide
  · funext q
    rfl
def root360Tail34 : List Nat := []
def root360Tail33 : List Nat := 23537 :: root360Tail34
def root360Tail32 : List Nat := 23417 :: root360Tail33
def root360Tail31 : List Nat := 21179 :: root360Tail32
def root360Tail30 : List Nat := 21017 :: root360Tail31
def root360Tail29 : List Nat := 21011 :: root360Tail30
def root360Tail28 : List Nat := 20129 :: root360Tail29
def root360Tail27 : List Nat := 19841 :: root360Tail28
def root360Tail26 : List Nat := 18839 :: root360Tail27
def root360Tail25 : List Nat := 16553 :: root360Tail26
def root360Tail24 : List Nat := 15971 :: root360Tail25
def root360Tail23 : List Nat := 14033 :: root360Tail24
def root360Tail22 : List Nat := 13697 :: root360Tail23
def root360Tail21 : List Nat := 13463 :: root360Tail22
def root360Tail20 : List Nat := 12911 :: root360Tail21
def root360Tail19 : List Nat := 12659 :: root360Tail20
def root360Tail18 : List Nat := 12197 :: root360Tail19
def root360Tail17 : List Nat := 11351 :: root360Tail18
def root360Tail16 : List Nat := 10889 :: root360Tail17
def root360Tail15 : List Nat := 10259 :: root360Tail16
def root360Tail14 : List Nat := 10211 :: root360Tail15
def root360Tail13 : List Nat := 9293 :: root360Tail14
def root360Tail12 : List Nat := 8123 :: root360Tail13
def root360Tail11 : List Nat := 8093 :: root360Tail12
def root360Tail10 : List Nat := 7907 :: root360Tail11
def root360Tail9 : List Nat := 7283 :: root360Tail10
def root360Tail8 : List Nat := 7001 :: root360Tail9
def root360Tail7 : List Nat := 6323 :: root360Tail8
def root360Tail6 : List Nat := 6113 :: root360Tail7
def root360Tail5 : List Nat := 5939 :: root360Tail6
def root360Tail4 : List Nat := 5813 :: root360Tail5
def root360Tail3 : List Nat := 4937 :: root360Tail4
def root360Tail2 : List Nat := 4019 :: root360Tail3
def root360Tail1 : List Nat := 3797 :: root360Tail2
def root360Tail0 : List Nat := 3011 :: root360Tail1
def root360Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root360Candidates0_checked : candidates edge 4 2447 3011 root360Tail1 winner.bound = root360Candidates0 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root360Step0 : scan edge (search edge 3) 4 [2447] 2447 root360Tail0 winner =
    scan edge (search edge 3) 4 [2447] 2447 root360Tail1 winner := by
  rw [root360Tail0, scan, if_neg (by decide), root360Candidates0_checked]
  change scan edge (search edge 3) 4 [2447] 2447 root360Tail1
    (if 3 ≤ root360Candidates0.length then search edge 3 [2447, 3011] root360Candidates0 5458 winner else winner) = _
  rw [if_neg (by decide)]
def root360Candidates1 : List Nat := [10889, 14033]
set_option maxRecDepth 8192 in
theorem root360Candidates1_checked : candidates edge 4 2447 3797 root360Tail2 winner.bound = root360Candidates1 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root360Step1 : scan edge (search edge 3) 4 [2447] 2447 root360Tail1 winner =
    scan edge (search edge 3) 4 [2447] 2447 root360Tail2 winner := by
  rw [root360Tail1, scan, if_neg (by decide), root360Candidates1_checked]
  change scan edge (search edge 3) 4 [2447] 2447 root360Tail2
    (if 3 ≤ root360Candidates1.length then search edge 3 [2447, 3797] root360Candidates1 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root360Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root360Candidates2_checked : candidates edge 4 2447 4019 root360Tail3 winner.bound = root360Candidates2 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root360Step2 : scan edge (search edge 3) 4 [2447] 2447 root360Tail2 winner =
    scan edge (search edge 3) 4 [2447] 2447 root360Tail3 winner := by
  rw [root360Tail2, scan, if_neg (by decide), root360Candidates2_checked]
  change scan edge (search edge 3) 4 [2447] 2447 root360Tail3
    (if 3 ≤ root360Candidates2.length then search edge 3 [2447, 4019] root360Candidates2 6466 winner else winner) = _
  rw [if_neg (by decide)]
def root360Candidates3 : List Nat := [7001]
set_option maxRecDepth 8192 in
theorem root360Candidates3_checked : candidates edge 4 2447 4937 root360Tail4 winner.bound = root360Candidates3 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root360Step3 : scan edge (search edge 3) 4 [2447] 2447 root360Tail3 winner =
    scan edge (search edge 3) 4 [2447] 2447 root360Tail4 winner := by
  rw [root360Tail3, scan, if_neg (by decide), root360Candidates3_checked]
  change scan edge (search edge 3) 4 [2447] 2447 root360Tail4
    (if 3 ≤ root360Candidates3.length then search edge 3 [2447, 4937] root360Candidates3 7384 winner else winner) = _
  rw [if_neg (by decide)]
def root360Candidates4 : List Nat := [7907, 12659]
set_option maxRecDepth 8192 in
theorem root360Candidates4_checked : candidates edge 4 2447 5813 root360Tail5 winner.bound = root360Candidates4 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root360Step4 : scan edge (search edge 3) 4 [2447] 2447 root360Tail4 winner =
    scan edge (search edge 3) 4 [2447] 2447 root360Tail5 winner := by
  rw [root360Tail4, scan, if_neg (by decide), root360Candidates4_checked]
  change scan edge (search edge 3) 4 [2447] 2447 root360Tail5
    (if 3 ≤ root360Candidates4.length then search edge 3 [2447, 5813] root360Candidates4 8260 winner else winner) = _
  rw [if_neg (by decide)]
theorem root360Scan5 : scan edge (search edge 3) 4 [2447] 2447 root360Tail5 winner = winner := by rw [root360Tail5, scan, if_pos (by decide)]
theorem root360Scan4 : scan edge (search edge 3) 4 [2447] 2447 root360Tail4 winner = winner :=
  (root360Step4).trans root360Scan5
theorem root360Scan3 : scan edge (search edge 3) 4 [2447] 2447 root360Tail3 winner = winner :=
  (root360Step3).trans root360Scan4
theorem root360Scan2 : scan edge (search edge 3) 4 [2447] 2447 root360Tail2 winner = winner :=
  (root360Step2).trans root360Scan3
theorem root360Scan1 : scan edge (search edge 3) 4 [2447] 2447 root360Tail1 winner = winner :=
  (root360Step1).trans root360Scan2
theorem root360Scan0 : scan edge (search edge 3) 4 [2447] 2447 root360Tail0 winner = winner :=
  (root360Step0).trans root360Scan1
set_option maxRecDepth 4096 in
theorem root360Branch : search edge 4 [2447] root360Candidates 2447 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2447] 2447 root360Tail0 winner = winner
  exact root360Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
