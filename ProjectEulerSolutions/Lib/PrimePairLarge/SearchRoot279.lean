import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root279Candidates : List Nat := Data.row1831
set_option maxRecDepth 8192 in
theorem root279Candidates_checked : candidates edge 5 0 1831 tail280 winner.bound = root279Candidates := by
  apply candidates_eq_row edge 5 0 1831 tail280 root279Candidates winner.bound tail280_ordered
  · decide
  · decide
  · funext q
    rfl
def root279Tail36 : List Nat := []
def root279Tail35 : List Nat := 22531 :: root279Tail36
def root279Tail34 : List Nat := 22447 :: root279Tail35
def root279Tail33 : List Nat := 21613 :: root279Tail34
def root279Tail32 : List Nat := 21319 :: root279Tail33
def root279Tail31 : List Nat := 20611 :: root279Tail32
def root279Tail30 : List Nat := 20149 :: root279Tail31
def root279Tail29 : List Nat := 20029 :: root279Tail30
def root279Tail28 : List Nat := 17707 :: root279Tail29
def root279Tail27 : List Nat := 16651 :: root279Tail28
def root279Tail26 : List Nat := 16447 :: root279Tail27
def root279Tail25 : List Nat := 15937 :: root279Tail26
def root279Tail24 : List Nat := 15817 :: root279Tail25
def root279Tail23 : List Nat := 15073 :: root279Tail24
def root279Tail22 : List Nat := 14389 :: root279Tail23
def root279Tail21 : List Nat := 14143 :: root279Tail22
def root279Tail20 : List Nat := 13729 :: root279Tail21
def root279Tail19 : List Nat := 13309 :: root279Tail20
def root279Tail18 : List Nat := 13003 :: root279Tail19
def root279Tail17 : List Nat := 12421 :: root279Tail18
def root279Tail16 : List Nat := 11491 :: root279Tail17
def root279Tail15 : List Nat := 11287 :: root279Tail16
def root279Tail14 : List Nat := 11239 :: root279Tail15
def root279Tail13 : List Nat := 10903 :: root279Tail14
def root279Tail12 : List Nat := 10729 :: root279Tail13
def root279Tail11 : List Nat := 10663 :: root279Tail12
def root279Tail10 : List Nat := 10651 :: root279Tail11
def root279Tail9 : List Nat := 9241 :: root279Tail10
def root279Tail8 : List Nat := 8941 :: root279Tail9
def root279Tail7 : List Nat := 8233 :: root279Tail8
def root279Tail6 : List Nat := 7927 :: root279Tail7
def root279Tail5 : List Nat := 7603 :: root279Tail6
def root279Tail4 : List Nat := 6301 :: root279Tail5
def root279Tail3 : List Nat := 5197 :: root279Tail4
def root279Tail2 : List Nat := 3769 :: root279Tail3
def root279Tail1 : List Nat := 3259 :: root279Tail2
def root279Tail0 : List Nat := 2803 :: root279Tail1
def root279Candidates0 : List Nat := [3769, 14143]
set_option maxRecDepth 8192 in
theorem root279Candidates0_checked : candidates edge 4 1831 2803 root279Tail1 winner.bound = root279Candidates0 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root279Step0 : scan edge (search edge 3) 4 [1831] 1831 root279Tail0 winner =
    scan edge (search edge 3) 4 [1831] 1831 root279Tail1 winner := by
  rw [root279Tail0, scan, if_neg (by decide), root279Candidates0_checked]
  change scan edge (search edge 3) 4 [1831] 1831 root279Tail1
    (if 3 ≤ root279Candidates0.length then search edge 3 [1831, 2803] root279Candidates0 4634 winner else winner) = _
  rw [if_neg (by decide)]
def root279Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root279Candidates1_checked : candidates edge 4 1831 3259 root279Tail2 winner.bound = root279Candidates1 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
theorem root279Step1 : scan edge (search edge 3) 4 [1831] 1831 root279Tail1 winner =
    scan edge (search edge 3) 4 [1831] 1831 root279Tail2 winner := by
  rw [root279Tail1, scan, if_neg (by decide), root279Candidates1_checked]
  change scan edge (search edge 3) 4 [1831] 1831 root279Tail2
    (if 3 ≤ root279Candidates1.length then search edge 3 [1831, 3259] root279Candidates1 5090 winner else winner) = _
  rw [if_neg (by decide)]
def root279Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root279Candidates2_checked : candidates edge 4 1831 3769 root279Tail3 winner.bound = root279Candidates2 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root279Step2 : scan edge (search edge 3) 4 [1831] 1831 root279Tail2 winner =
    scan edge (search edge 3) 4 [1831] 1831 root279Tail3 winner := by
  rw [root279Tail2, scan, if_neg (by decide), root279Candidates2_checked]
  change scan edge (search edge 3) 4 [1831] 1831 root279Tail3
    (if 3 ≤ root279Candidates2.length then search edge 3 [1831, 3769] root279Candidates2 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root279Candidates3 : List Nat := [10729, 12421]
set_option maxRecDepth 8192 in
theorem root279Candidates3_checked : candidates edge 4 1831 5197 root279Tail4 winner.bound = root279Candidates3 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root279Step3 : scan edge (search edge 3) 4 [1831] 1831 root279Tail3 winner =
    scan edge (search edge 3) 4 [1831] 1831 root279Tail4 winner := by
  rw [root279Tail3, scan, if_neg (by decide), root279Candidates3_checked]
  change scan edge (search edge 3) 4 [1831] 1831 root279Tail4
    (if 3 ≤ root279Candidates3.length then search edge 3 [1831, 5197] root279Candidates3 7028 winner else winner) = _
  rw [if_neg (by decide)]
theorem root279Scan4 : scan edge (search edge 3) 4 [1831] 1831 root279Tail4 winner = winner := by rw [root279Tail4, scan, if_pos (by decide)]
theorem root279Scan3 : scan edge (search edge 3) 4 [1831] 1831 root279Tail3 winner = winner :=
  (root279Step3).trans root279Scan4
theorem root279Scan2 : scan edge (search edge 3) 4 [1831] 1831 root279Tail2 winner = winner :=
  (root279Step2).trans root279Scan3
theorem root279Scan1 : scan edge (search edge 3) 4 [1831] 1831 root279Tail1 winner = winner :=
  (root279Step1).trans root279Scan2
theorem root279Scan0 : scan edge (search edge 3) 4 [1831] 1831 root279Tail0 winner = winner :=
  (root279Step0).trans root279Scan1
set_option maxRecDepth 4096 in
theorem root279Branch : search edge 4 [1831] root279Candidates 1831 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1831] 1831 root279Tail0 winner = winner
  exact root279Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
