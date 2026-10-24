import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root624Candidates : List Nat := Data.row4643
set_option maxRecDepth 8192 in
theorem root624Candidates_checked : candidates edge 5 0 4643 tail625 winner.bound = root624Candidates := by
  apply candidates_eq_row edge 5 0 4643 tail625 root624Candidates winner.bound tail625_ordered
  · decide
  · decide
  · funext q
    rfl
def root624Tail33 : List Nat := []
def root624Tail32 : List Nat := 20627 :: root624Tail33
def root624Tail31 : List Nat := 20399 :: root624Tail32
def root624Tail30 : List Nat := 20183 :: root624Tail31
def root624Tail29 : List Nat := 18947 :: root624Tail30
def root624Tail28 : List Nat := 18461 :: root624Tail29
def root624Tail27 : List Nat := 18401 :: root624Tail28
def root624Tail26 : List Nat := 17939 :: root624Tail27
def root624Tail25 : List Nat := 17837 :: root624Tail26
def root624Tail24 : List Nat := 16823 :: root624Tail25
def root624Tail23 : List Nat := 15083 :: root624Tail24
def root624Tail22 : List Nat := 14303 :: root624Tail23
def root624Tail21 : List Nat := 13049 :: root624Tail22
def root624Tail20 : List Nat := 13007 :: root624Tail21
def root624Tail19 : List Nat := 11909 :: root624Tail20
def root624Tail18 : List Nat := 11699 :: root624Tail19
def root624Tail17 : List Nat := 11579 :: root624Tail18
def root624Tail16 : List Nat := 10949 :: root624Tail17
def root624Tail15 : List Nat := 10799 :: root624Tail16
def root624Tail14 : List Nat := 10211 :: root624Tail15
def root624Tail13 : List Nat := 9203 :: root624Tail14
def root624Tail12 : List Nat := 9161 :: root624Tail13
def root624Tail11 : List Nat := 7757 :: root624Tail12
def root624Tail10 : List Nat := 6827 :: root624Tail11
def root624Tail9 : List Nat := 6473 :: root624Tail10
def root624Tail8 : List Nat := 6263 :: root624Tail9
def root624Tail7 : List Nat := 5849 :: root624Tail8
def root624Tail6 : List Nat := 5717 :: root624Tail7
def root624Tail5 : List Nat := 5693 :: root624Tail6
def root624Tail4 : List Nat := 5471 :: root624Tail5
def root624Tail3 : List Nat := 5237 :: root624Tail4
def root624Tail2 : List Nat := 5231 :: root624Tail3
def root624Tail1 : List Nat := 5153 :: root624Tail2
def root624Tail0 : List Nat := 4877 :: root624Tail1
def root624Candidates0 : List Nat := [5693, 9161]
set_option maxRecDepth 8192 in
theorem root624Candidates0_checked : candidates edge 4 4643 4877 root624Tail1 winner.bound = root624Candidates0 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root624Step0 : scan edge (search edge 3) 4 [4643] 4643 root624Tail0 winner =
    scan edge (search edge 3) 4 [4643] 4643 root624Tail1 winner := by
  rw [root624Tail0, scan, if_neg (by decide), root624Candidates0_checked]
  change scan edge (search edge 3) 4 [4643] 4643 root624Tail1
    (if 3 ≤ root624Candidates0.length then search edge 3 [4643, 4877] root624Candidates0 9520 winner else winner) = _
  rw [if_neg (by decide)]
def root624Candidates1 : List Nat := [5849, 10949, 11579]
set_option maxRecDepth 8192 in
theorem root624Candidates1_checked : candidates edge 4 4643 5153 root624Tail2 winner.bound = root624Candidates1 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root624Child1 : search edge 3 [4643, 5153] root624Candidates1 9796 winner = winner := by decide
theorem root624Step1 : scan edge (search edge 3) 4 [4643] 4643 root624Tail1 winner =
    scan edge (search edge 3) 4 [4643] 4643 root624Tail2 winner := by
  rw [root624Tail1, scan, if_neg (by decide), root624Candidates1_checked]
  change scan edge (search edge 3) 4 [4643] 4643 root624Tail2
    (if 3 ≤ root624Candidates1.length then search edge 3 [4643, 5153] root624Candidates1 9796 winner else winner) = _
  rw [if_pos (by decide), root624Child1]
def root624Candidates2 : List Nat := [10949]
set_option maxRecDepth 8192 in
theorem root624Candidates2_checked : candidates edge 4 4643 5231 root624Tail3 winner.bound = root624Candidates2 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root624Step2 : scan edge (search edge 3) 4 [4643] 4643 root624Tail2 winner =
    scan edge (search edge 3) 4 [4643] 4643 root624Tail3 winner := by
  rw [root624Tail2, scan, if_neg (by decide), root624Candidates2_checked]
  change scan edge (search edge 3) 4 [4643] 4643 root624Tail3
    (if 3 ≤ root624Candidates2.length then search edge 3 [4643, 5231] root624Candidates2 9874 winner else winner) = _
  rw [if_neg (by decide)]
def root624Candidates3 : List Nat := [5471, 5693]
set_option maxRecDepth 8192 in
theorem root624Candidates3_checked : candidates edge 4 4643 5237 root624Tail4 winner.bound = root624Candidates3 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root624Step3 : scan edge (search edge 3) 4 [4643] 4643 root624Tail3 winner =
    scan edge (search edge 3) 4 [4643] 4643 root624Tail4 winner := by
  rw [root624Tail3, scan, if_neg (by decide), root624Candidates3_checked]
  change scan edge (search edge 3) 4 [4643] 4643 root624Tail4
    (if 3 ≤ root624Candidates3.length then search edge 3 [4643, 5237] root624Candidates3 9880 winner else winner) = _
  rw [if_neg (by decide)]
theorem root624Scan4 : scan edge (search edge 3) 4 [4643] 4643 root624Tail4 winner = winner := by rw [root624Tail4, scan, if_pos (by decide)]
theorem root624Scan3 : scan edge (search edge 3) 4 [4643] 4643 root624Tail3 winner = winner :=
  (root624Step3).trans root624Scan4
theorem root624Scan2 : scan edge (search edge 3) 4 [4643] 4643 root624Tail2 winner = winner :=
  (root624Step2).trans root624Scan3
theorem root624Scan1 : scan edge (search edge 3) 4 [4643] 4643 root624Tail1 winner = winner :=
  (root624Step1).trans root624Scan2
theorem root624Scan0 : scan edge (search edge 3) 4 [4643] 4643 root624Tail0 winner = winner :=
  (root624Step0).trans root624Scan1
set_option maxRecDepth 4096 in
theorem root624Branch : search edge 4 [4643] root624Candidates 4643 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4643] 4643 root624Tail0 winner = winner
  exact root624Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
