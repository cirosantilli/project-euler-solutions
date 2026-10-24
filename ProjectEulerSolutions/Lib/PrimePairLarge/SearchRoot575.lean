import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root575Candidates : List Nat := Data.row4219
set_option maxRecDepth 8192 in
theorem root575Candidates_checked : candidates edge 5 0 4219 tail576 winner.bound = root575Candidates := by
  apply candidates_eq_row edge 5 0 4219 tail576 root575Candidates winner.bound tail576_ordered
  · decide
  · decide
  · funext q
    rfl
def root575Tail30 : List Nat := []
def root575Tail29 : List Nat := 21601 :: root575Tail30
def root575Tail28 : List Nat := 21013 :: root575Tail29
def root575Tail27 : List Nat := 20743 :: root575Tail28
def root575Tail26 : List Nat := 18583 :: root575Tail27
def root575Tail25 : List Nat := 18493 :: root575Tail26
def root575Tail24 : List Nat := 15937 :: root575Tail25
def root575Tail23 : List Nat := 15679 :: root575Tail24
def root575Tail22 : List Nat := 14797 :: root575Tail23
def root575Tail21 : List Nat := 14197 :: root575Tail22
def root575Tail20 : List Nat := 13669 :: root575Tail21
def root575Tail19 : List Nat := 12889 :: root575Tail20
def root575Tail18 : List Nat := 10771 :: root575Tail19
def root575Tail17 : List Nat := 10141 :: root575Tail18
def root575Tail16 : List Nat := 9679 :: root575Tail17
def root575Tail15 : List Nat := 9403 :: root575Tail16
def root575Tail14 : List Nat := 8707 :: root575Tail15
def root575Tail13 : List Nat := 8209 :: root575Tail14
def root575Tail12 : List Nat := 7369 :: root575Tail13
def root575Tail11 : List Nat := 7027 :: root575Tail12
def root575Tail10 : List Nat := 6883 :: root575Tail11
def root575Tail9 : List Nat := 6793 :: root575Tail10
def root575Tail8 : List Nat := 6529 :: root575Tail9
def root575Tail7 : List Nat := 6379 :: root575Tail8
def root575Tail6 : List Nat := 6337 :: root575Tail7
def root575Tail5 : List Nat := 5749 :: root575Tail6
def root575Tail4 : List Nat := 5521 :: root575Tail5
def root575Tail3 : List Nat := 5227 :: root575Tail4
def root575Tail2 : List Nat := 5119 :: root575Tail3
def root575Tail1 : List Nat := 5113 :: root575Tail2
def root575Tail0 : List Nat := 4597 :: root575Tail1
def root575Candidates0 : List Nat := [5113, 5749, 9403, 14197]
set_option maxRecDepth 8192 in
theorem root575Candidates0_checked : candidates edge 4 4219 4597 root575Tail1 winner.bound = root575Candidates0 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root575Child0 : search edge 3 [4219, 4597] root575Candidates0 8816 winner = winner := by decide
theorem root575Step0 : scan edge (search edge 3) 4 [4219] 4219 root575Tail0 winner =
    scan edge (search edge 3) 4 [4219] 4219 root575Tail1 winner := by
  rw [root575Tail0, scan, if_neg (by decide), root575Candidates0_checked]
  change scan edge (search edge 3) 4 [4219] 4219 root575Tail1
    (if 3 ≤ root575Candidates0.length then search edge 3 [4219, 4597] root575Candidates0 8816 winner else winner) = _
  rw [if_pos (by decide), root575Child0]
def root575Candidates1 : List Nat := [6793]
set_option maxRecDepth 8192 in
theorem root575Candidates1_checked : candidates edge 4 4219 5113 root575Tail2 winner.bound = root575Candidates1 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root575Step1 : scan edge (search edge 3) 4 [4219] 4219 root575Tail1 winner =
    scan edge (search edge 3) 4 [4219] 4219 root575Tail2 winner := by
  rw [root575Tail1, scan, if_neg (by decide), root575Candidates1_checked]
  change scan edge (search edge 3) 4 [4219] 4219 root575Tail2
    (if 3 ≤ root575Candidates1.length then search edge 3 [4219, 5113] root575Candidates1 9332 winner else winner) = _
  rw [if_neg (by decide)]
def root575Candidates2 : List Nat := [6337, 6529, 14797]
set_option maxRecDepth 8192 in
theorem root575Candidates2_checked : candidates edge 4 4219 5119 root575Tail3 winner.bound = root575Candidates2 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root575Child2 : search edge 3 [4219, 5119] root575Candidates2 9338 winner = winner := by decide
theorem root575Step2 : scan edge (search edge 3) 4 [4219] 4219 root575Tail2 winner =
    scan edge (search edge 3) 4 [4219] 4219 root575Tail3 winner := by
  rw [root575Tail2, scan, if_neg (by decide), root575Candidates2_checked]
  change scan edge (search edge 3) 4 [4219] 4219 root575Tail3
    (if 3 ≤ root575Candidates2.length then search edge 3 [4219, 5119] root575Candidates2 9338 winner else winner) = _
  rw [if_pos (by decide), root575Child2]
def root575Candidates3 : List Nat := [6379, 13669]
set_option maxRecDepth 8192 in
theorem root575Candidates3_checked : candidates edge 4 4219 5227 root575Tail4 winner.bound = root575Candidates3 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root575Step3 : scan edge (search edge 3) 4 [4219] 4219 root575Tail3 winner =
    scan edge (search edge 3) 4 [4219] 4219 root575Tail4 winner := by
  rw [root575Tail3, scan, if_neg (by decide), root575Candidates3_checked]
  change scan edge (search edge 3) 4 [4219] 4219 root575Tail4
    (if 3 ≤ root575Candidates3.length then search edge 3 [4219, 5227] root575Candidates3 9446 winner else winner) = _
  rw [if_neg (by decide)]
theorem root575Scan4 : scan edge (search edge 3) 4 [4219] 4219 root575Tail4 winner = winner := by rw [root575Tail4, scan, if_pos (by decide)]
theorem root575Scan3 : scan edge (search edge 3) 4 [4219] 4219 root575Tail3 winner = winner :=
  (root575Step3).trans root575Scan4
theorem root575Scan2 : scan edge (search edge 3) 4 [4219] 4219 root575Tail2 winner = winner :=
  (root575Step2).trans root575Scan3
theorem root575Scan1 : scan edge (search edge 3) 4 [4219] 4219 root575Tail1 winner = winner :=
  (root575Step1).trans root575Scan2
theorem root575Scan0 : scan edge (search edge 3) 4 [4219] 4219 root575Tail0 winner = winner :=
  (root575Step0).trans root575Scan1
set_option maxRecDepth 4096 in
theorem root575Branch : search edge 4 [4219] root575Candidates 4219 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4219] 4219 root575Tail0 winner = winner
  exact root575Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
