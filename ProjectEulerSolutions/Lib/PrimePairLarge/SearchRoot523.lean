import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root523Candidates : List Nat := Data.row3779
set_option maxRecDepth 8192 in
theorem root523Candidates_checked : candidates edge 5 0 3779 tail524 winner.bound = root523Candidates := by
  apply candidates_eq_row edge 5 0 3779 tail524 root523Candidates winner.bound tail524_ordered
  · decide
  · decide
  · funext q
    rfl
def root523Tail32 : List Nat := []
def root523Tail31 : List Nat := 21893 :: root523Tail32
def root523Tail30 : List Nat := 21347 :: root523Tail31
def root523Tail29 : List Nat := 19727 :: root523Tail30
def root523Tail28 : List Nat := 18539 :: root523Tail29
def root523Tail27 : List Nat := 18059 :: root523Tail28
def root523Tail26 : List Nat := 17783 :: root523Tail27
def root523Tail25 : List Nat := 17489 :: root523Tail26
def root523Tail24 : List Nat := 14951 :: root523Tail25
def root523Tail23 : List Nat := 14897 :: root523Tail24
def root523Tail22 : List Nat := 14561 :: root523Tail23
def root523Tail21 : List Nat := 14549 :: root523Tail22
def root523Tail20 : List Nat := 14447 :: root523Tail21
def root523Tail19 : List Nat := 13259 :: root523Tail20
def root523Tail18 : List Nat := 13241 :: root523Tail19
def root523Tail17 : List Nat := 12911 :: root523Tail18
def root523Tail16 : List Nat := 12671 :: root523Tail17
def root523Tail15 : List Nat := 11927 :: root523Tail16
def root523Tail14 : List Nat := 10529 :: root523Tail15
def root523Tail13 : List Nat := 9941 :: root523Tail14
def root523Tail12 : List Nat := 8573 :: root523Tail13
def root523Tail11 : List Nat := 7829 :: root523Tail12
def root523Tail10 : List Nat := 7577 :: root523Tail11
def root523Tail9 : List Nat := 7523 :: root523Tail10
def root523Tail8 : List Nat := 6329 :: root523Tail9
def root523Tail7 : List Nat := 6299 :: root523Tail8
def root523Tail6 : List Nat := 6053 :: root523Tail7
def root523Tail5 : List Nat := 5843 :: root523Tail6
def root523Tail4 : List Nat := 5099 :: root523Tail5
def root523Tail3 : List Nat := 4931 :: root523Tail4
def root523Tail2 : List Nat := 4637 :: root523Tail3
def root523Tail1 : List Nat := 4241 :: root523Tail2
def root523Tail0 : List Nat := 3863 :: root523Tail1
def root523Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root523Candidates0_checked : candidates edge 4 3779 3863 root523Tail1 winner.bound = root523Candidates0 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root523Step0 : scan edge (search edge 3) 4 [3779] 3779 root523Tail0 winner =
    scan edge (search edge 3) 4 [3779] 3779 root523Tail1 winner := by
  rw [root523Tail0, scan, if_neg (by decide), root523Candidates0_checked]
  change scan edge (search edge 3) 4 [3779] 3779 root523Tail1
    (if 3 ≤ root523Candidates0.length then search edge 3 [3779, 3863] root523Candidates0 7642 winner else winner) = _
  rw [if_neg (by decide)]
def root523Candidates1 : List Nat := [6053]
set_option maxRecDepth 8192 in
theorem root523Candidates1_checked : candidates edge 4 3779 4241 root523Tail2 winner.bound = root523Candidates1 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root523Step1 : scan edge (search edge 3) 4 [3779] 3779 root523Tail1 winner =
    scan edge (search edge 3) 4 [3779] 3779 root523Tail2 winner := by
  rw [root523Tail1, scan, if_neg (by decide), root523Candidates1_checked]
  change scan edge (search edge 3) 4 [3779] 3779 root523Tail2
    (if 3 ≤ root523Candidates1.length then search edge 3 [3779, 4241] root523Candidates1 8020 winner else winner) = _
  rw [if_neg (by decide)]
def root523Candidates2 : List Nat := [17489]
set_option maxRecDepth 8192 in
theorem root523Candidates2_checked : candidates edge 4 3779 4637 root523Tail3 winner.bound = root523Candidates2 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root523Step2 : scan edge (search edge 3) 4 [3779] 3779 root523Tail2 winner =
    scan edge (search edge 3) 4 [3779] 3779 root523Tail3 winner := by
  rw [root523Tail2, scan, if_neg (by decide), root523Candidates2_checked]
  change scan edge (search edge 3) 4 [3779] 3779 root523Tail3
    (if 3 ≤ root523Candidates2.length then search edge 3 [3779, 4637] root523Candidates2 8416 winner else winner) = _
  rw [if_neg (by decide)]
def root523Candidates3 : List Nat := [5843]
set_option maxRecDepth 8192 in
theorem root523Candidates3_checked : candidates edge 4 3779 4931 root523Tail4 winner.bound = root523Candidates3 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root523Step3 : scan edge (search edge 3) 4 [3779] 3779 root523Tail3 winner =
    scan edge (search edge 3) 4 [3779] 3779 root523Tail4 winner := by
  rw [root523Tail3, scan, if_neg (by decide), root523Candidates3_checked]
  change scan edge (search edge 3) 4 [3779] 3779 root523Tail4
    (if 3 ≤ root523Candidates3.length then search edge 3 [3779, 4931] root523Candidates3 8710 winner else winner) = _
  rw [if_neg (by decide)]
def root523Candidates4 : List Nat := [7523]
set_option maxRecDepth 8192 in
theorem root523Candidates4_checked : candidates edge 4 3779 5099 root523Tail5 winner.bound = root523Candidates4 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root523Step4 : scan edge (search edge 3) 4 [3779] 3779 root523Tail4 winner =
    scan edge (search edge 3) 4 [3779] 3779 root523Tail5 winner := by
  rw [root523Tail4, scan, if_neg (by decide), root523Candidates4_checked]
  change scan edge (search edge 3) 4 [3779] 3779 root523Tail5
    (if 3 ≤ root523Candidates4.length then search edge 3 [3779, 5099] root523Candidates4 8878 winner else winner) = _
  rw [if_neg (by decide)]
theorem root523Scan5 : scan edge (search edge 3) 4 [3779] 3779 root523Tail5 winner = winner := by rw [root523Tail5, scan, if_pos (by decide)]
theorem root523Scan4 : scan edge (search edge 3) 4 [3779] 3779 root523Tail4 winner = winner :=
  (root523Step4).trans root523Scan5
theorem root523Scan3 : scan edge (search edge 3) 4 [3779] 3779 root523Tail3 winner = winner :=
  (root523Step3).trans root523Scan4
theorem root523Scan2 : scan edge (search edge 3) 4 [3779] 3779 root523Tail2 winner = winner :=
  (root523Step2).trans root523Scan3
theorem root523Scan1 : scan edge (search edge 3) 4 [3779] 3779 root523Tail1 winner = winner :=
  (root523Step1).trans root523Scan2
theorem root523Scan0 : scan edge (search edge 3) 4 [3779] 3779 root523Tail0 winner = winner :=
  (root523Step0).trans root523Scan1
set_option maxRecDepth 4096 in
theorem root523Branch : search edge 4 [3779] root523Candidates 3779 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3779] 3779 root523Tail0 winner = winner
  exact root523Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
