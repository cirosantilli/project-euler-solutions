import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root513Candidates : List Nat := Data.row3697
set_option maxRecDepth 8192 in
theorem root513Candidates_checked : candidates edge 5 0 3697 tail514 winner.bound = root513Candidates := by
  apply candidates_eq_row edge 5 0 3697 tail514 root513Candidates winner.bound tail514_ordered
  · decide
  · decide
  · funext q
    rfl
def root513Tail34 : List Nat := []
def root513Tail33 : List Nat := 21277 :: root513Tail34
def root513Tail32 : List Nat := 20161 :: root513Tail33
def root513Tail31 : List Nat := 19219 :: root513Tail32
def root513Tail30 : List Nat := 19069 :: root513Tail31
def root513Tail29 : List Nat := 18379 :: root513Tail30
def root513Tail28 : List Nat := 18211 :: root513Tail29
def root513Tail27 : List Nat := 17443 :: root513Tail28
def root513Tail26 : List Nat := 16843 :: root513Tail27
def root513Tail25 : List Nat := 15739 :: root513Tail26
def root513Tail24 : List Nat := 14479 :: root513Tail25
def root513Tail23 : List Nat := 13873 :: root513Tail24
def root513Tail22 : List Nat := 11689 :: root513Tail23
def root513Tail21 : List Nat := 11311 :: root513Tail22
def root513Tail20 : List Nat := 11059 :: root513Tail21
def root513Tail19 : List Nat := 10651 :: root513Tail20
def root513Tail18 : List Nat := 10333 :: root513Tail19
def root513Tail17 : List Nat := 9619 :: root513Tail18
def root513Tail16 : List Nat := 9181 :: root513Tail17
def root513Tail15 : List Nat := 9043 :: root513Tail16
def root513Tail14 : List Nat := 8941 :: root513Tail15
def root513Tail13 : List Nat := 8803 :: root513Tail14
def root513Tail12 : List Nat := 8647 :: root513Tail13
def root513Tail11 : List Nat := 8563 :: root513Tail12
def root513Tail10 : List Nat := 7879 :: root513Tail11
def root513Tail9 : List Nat := 7867 :: root513Tail10
def root513Tail8 : List Nat := 7459 :: root513Tail9
def root513Tail7 : List Nat := 6673 :: root513Tail8
def root513Tail6 : List Nat := 6469 :: root513Tail7
def root513Tail5 : List Nat := 6007 :: root513Tail6
def root513Tail4 : List Nat := 5749 :: root513Tail5
def root513Tail3 : List Nat := 5431 :: root513Tail4
def root513Tail2 : List Nat := 5209 :: root513Tail3
def root513Tail1 : List Nat := 4603 :: root513Tail2
def root513Tail0 : List Nat := 3919 :: root513Tail1
def root513Candidates0 : List Nat := [6007]
set_option maxRecDepth 8192 in
theorem root513Candidates0_checked : candidates edge 4 3697 3919 root513Tail1 winner.bound = root513Candidates0 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root513Step0 : scan edge (search edge 3) 4 [3697] 3697 root513Tail0 winner =
    scan edge (search edge 3) 4 [3697] 3697 root513Tail1 winner := by
  rw [root513Tail0, scan, if_neg (by decide), root513Candidates0_checked]
  change scan edge (search edge 3) 4 [3697] 3697 root513Tail1
    (if 3 ≤ root513Candidates0.length then search edge 3 [3697, 3919] root513Candidates0 7616 winner else winner) = _
  rw [if_neg (by decide)]
def root513Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root513Candidates1_checked : candidates edge 4 3697 4603 root513Tail2 winner.bound = root513Candidates1 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root513Step1 : scan edge (search edge 3) 4 [3697] 3697 root513Tail1 winner =
    scan edge (search edge 3) 4 [3697] 3697 root513Tail2 winner := by
  rw [root513Tail1, scan, if_neg (by decide), root513Candidates1_checked]
  change scan edge (search edge 3) 4 [3697] 3697 root513Tail2
    (if 3 ≤ root513Candidates1.length then search edge 3 [3697, 4603] root513Candidates1 8300 winner else winner) = _
  rw [if_neg (by decide)]
def root513Candidates2 : List Nat := [7879, 13873]
set_option maxRecDepth 8192 in
theorem root513Candidates2_checked : candidates edge 4 3697 5209 root513Tail3 winner.bound = root513Candidates2 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root513Step2 : scan edge (search edge 3) 4 [3697] 3697 root513Tail2 winner =
    scan edge (search edge 3) 4 [3697] 3697 root513Tail3 winner := by
  rw [root513Tail2, scan, if_neg (by decide), root513Candidates2_checked]
  change scan edge (search edge 3) 4 [3697] 3697 root513Tail3
    (if 3 ≤ root513Candidates2.length then search edge 3 [3697, 5209] root513Candidates2 8906 winner else winner) = _
  rw [if_neg (by decide)]
def root513Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root513Candidates3_checked : candidates edge 4 3697 5431 root513Tail4 winner.bound = root513Candidates3 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root513Step3 : scan edge (search edge 3) 4 [3697] 3697 root513Tail3 winner =
    scan edge (search edge 3) 4 [3697] 3697 root513Tail4 winner := by
  rw [root513Tail3, scan, if_neg (by decide), root513Candidates3_checked]
  change scan edge (search edge 3) 4 [3697] 3697 root513Tail4
    (if 3 ≤ root513Candidates3.length then search edge 3 [3697, 5431] root513Candidates3 9128 winner else winner) = _
  rw [if_neg (by decide)]
theorem root513Scan4 : scan edge (search edge 3) 4 [3697] 3697 root513Tail4 winner = winner := by rw [root513Tail4, scan, if_pos (by decide)]
theorem root513Scan3 : scan edge (search edge 3) 4 [3697] 3697 root513Tail3 winner = winner :=
  (root513Step3).trans root513Scan4
theorem root513Scan2 : scan edge (search edge 3) 4 [3697] 3697 root513Tail2 winner = winner :=
  (root513Step2).trans root513Scan3
theorem root513Scan1 : scan edge (search edge 3) 4 [3697] 3697 root513Tail1 winner = winner :=
  (root513Step1).trans root513Scan2
theorem root513Scan0 : scan edge (search edge 3) 4 [3697] 3697 root513Tail0 winner = winner :=
  (root513Step0).trans root513Scan1
set_option maxRecDepth 4096 in
theorem root513Branch : search edge 4 [3697] root513Candidates 3697 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3697] 3697 root513Tail0 winner = winner
  exact root513Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
