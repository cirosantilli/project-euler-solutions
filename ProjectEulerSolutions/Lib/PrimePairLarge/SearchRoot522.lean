import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root522Candidates : List Nat := Data.row3769
set_option maxRecDepth 8192 in
theorem root522Candidates_checked : candidates edge 5 0 3769 tail523 winner.bound = root522Candidates := by
  apply candidates_eq_row edge 5 0 3769 tail523 root522Candidates winner.bound tail523_ordered
  · decide
  · decide
  · funext q
    rfl
def root522Tail26 : List Nat := []
def root522Tail25 : List Nat := 22027 :: root522Tail26
def root522Tail24 : List Nat := 21577 :: root522Tail25
def root522Tail23 : List Nat := 21313 :: root522Tail24
def root522Tail22 : List Nat := 19927 :: root522Tail23
def root522Tail21 : List Nat := 19843 :: root522Tail22
def root522Tail20 : List Nat := 19687 :: root522Tail21
def root522Tail19 : List Nat := 18379 :: root522Tail20
def root522Tail18 : List Nat := 16993 :: root522Tail19
def root522Tail17 : List Nat := 16747 :: root522Tail18
def root522Tail16 : List Nat := 16063 :: root522Tail17
def root522Tail15 : List Nat := 15307 :: root522Tail16
def root522Tail14 : List Nat := 15139 :: root522Tail15
def root522Tail13 : List Nat := 14779 :: root522Tail14
def root522Tail12 : List Nat := 13219 :: root522Tail13
def root522Tail11 : List Nat := 12589 :: root522Tail12
def root522Tail10 : List Nat := 10789 :: root522Tail11
def root522Tail9 : List Nat := 9127 :: root522Tail10
def root522Tail8 : List Nat := 8971 :: root522Tail9
def root522Tail7 : List Nat := 8161 :: root522Tail8
def root522Tail6 : List Nat := 7669 :: root522Tail7
def root522Tail5 : List Nat := 6949 :: root522Tail6
def root522Tail4 : List Nat := 6619 :: root522Tail5
def root522Tail3 : List Nat := 5653 :: root522Tail4
def root522Tail2 : List Nat := 4603 :: root522Tail3
def root522Tail1 : List Nat := 4597 :: root522Tail2
def root522Tail0 : List Nat := 4051 :: root522Tail1
def root522Candidates0 : List Nat := [10789]
set_option maxRecDepth 8192 in
theorem root522Candidates0_checked : candidates edge 4 3769 4051 root522Tail1 winner.bound = root522Candidates0 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root522Step0 : scan edge (search edge 3) 4 [3769] 3769 root522Tail0 winner =
    scan edge (search edge 3) 4 [3769] 3769 root522Tail1 winner := by
  rw [root522Tail0, scan, if_neg (by decide), root522Candidates0_checked]
  change scan edge (search edge 3) 4 [3769] 3769 root522Tail1
    (if 3 ≤ root522Candidates0.length then search edge 3 [3769, 4051] root522Candidates0 7820 winner else winner) = _
  rw [if_neg (by decide)]
def root522Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root522Candidates1_checked : candidates edge 4 3769 4597 root522Tail2 winner.bound = root522Candidates1 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root522Step1 : scan edge (search edge 3) 4 [3769] 3769 root522Tail1 winner =
    scan edge (search edge 3) 4 [3769] 3769 root522Tail2 winner := by
  rw [root522Tail1, scan, if_neg (by decide), root522Candidates1_checked]
  change scan edge (search edge 3) 4 [3769] 3769 root522Tail2
    (if 3 ≤ root522Candidates1.length then search edge 3 [3769, 4597] root522Candidates1 8366 winner else winner) = _
  rw [if_neg (by decide)]
def root522Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root522Candidates2_checked : candidates edge 4 3769 4603 root522Tail3 winner.bound = root522Candidates2 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root522Step2 : scan edge (search edge 3) 4 [3769] 3769 root522Tail2 winner =
    scan edge (search edge 3) 4 [3769] 3769 root522Tail3 winner := by
  rw [root522Tail2, scan, if_neg (by decide), root522Candidates2_checked]
  change scan edge (search edge 3) 4 [3769] 3769 root522Tail3
    (if 3 ≤ root522Candidates2.length then search edge 3 [3769, 4603] root522Candidates2 8372 winner else winner) = _
  rw [if_neg (by decide)]
theorem root522Scan3 : scan edge (search edge 3) 4 [3769] 3769 root522Tail3 winner = winner := by rw [root522Tail3, scan, if_pos (by decide)]
theorem root522Scan2 : scan edge (search edge 3) 4 [3769] 3769 root522Tail2 winner = winner :=
  (root522Step2).trans root522Scan3
theorem root522Scan1 : scan edge (search edge 3) 4 [3769] 3769 root522Tail1 winner = winner :=
  (root522Step1).trans root522Scan2
theorem root522Scan0 : scan edge (search edge 3) 4 [3769] 3769 root522Tail0 winner = winner :=
  (root522Step0).trans root522Scan1
set_option maxRecDepth 4096 in
theorem root522Branch : search edge 4 [3769] root522Candidates 3769 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3769] 3769 root522Tail0 winner = winner
  exact root522Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
