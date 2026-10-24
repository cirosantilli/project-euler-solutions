import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root304Candidates : List Nat := Data.row2027
set_option maxRecDepth 8192 in
theorem root304Candidates_checked : candidates edge 5 0 2027 tail305 winner.bound = root304Candidates := by
  apply candidates_eq_row edge 5 0 2027 tail305 root304Candidates winner.bound tail305_ordered
  · decide
  · decide
  · funext q
    rfl
def root304Tail39 : List Nat := []
def root304Tail38 : List Nat := 23753 :: root304Tail39
def root304Tail37 : List Nat := 23063 :: root304Tail38
def root304Tail36 : List Nat := 22853 :: root304Tail37
def root304Tail35 : List Nat := 22067 :: root304Tail36
def root304Tail34 : List Nat := 21863 :: root304Tail35
def root304Tail33 : List Nat := 21191 :: root304Tail34
def root304Tail32 : List Nat := 20753 :: root304Tail33
def root304Tail31 : List Nat := 19793 :: root304Tail32
def root304Tail30 : List Nat := 19763 :: root304Tail31
def root304Tail29 : List Nat := 18947 :: root304Tail30
def root304Tail28 : List Nat := 18401 :: root304Tail29
def root304Tail27 : List Nat := 18077 :: root304Tail28
def root304Tail26 : List Nat := 17483 :: root304Tail27
def root304Tail25 : List Nat := 16433 :: root304Tail26
def root304Tail24 : List Nat := 15473 :: root304Tail25
def root304Tail23 : List Nat := 15269 :: root304Tail24
def root304Tail22 : List Nat := 15263 :: root304Tail23
def root304Tail21 : List Nat := 13781 :: root304Tail22
def root304Tail20 : List Nat := 12791 :: root304Tail21
def root304Tail19 : List Nat := 12611 :: root304Tail20
def root304Tail18 : List Nat := 12149 :: root304Tail19
def root304Tail17 : List Nat := 10889 :: root304Tail18
def root304Tail16 : List Nat := 10631 :: root304Tail17
def root304Tail15 : List Nat := 10463 :: root304Tail16
def root304Tail14 : List Nat := 10181 :: root304Tail15
def root304Tail13 : List Nat := 9839 :: root304Tail14
def root304Tail12 : List Nat := 9803 :: root304Tail13
def root304Tail11 : List Nat := 8837 :: root304Tail12
def root304Tail10 : List Nat := 7703 :: root304Tail11
def root304Tail9 : List Nat := 7457 :: root304Tail10
def root304Tail8 : List Nat := 7253 :: root304Tail9
def root304Tail7 : List Nat := 6911 :: root304Tail8
def root304Tail6 : List Nat := 6701 :: root304Tail7
def root304Tail5 : List Nat := 6203 :: root304Tail6
def root304Tail4 : List Nat := 5483 :: root304Tail5
def root304Tail3 : List Nat := 5153 :: root304Tail4
def root304Tail2 : List Nat := 4259 :: root304Tail3
def root304Tail1 : List Nat := 3761 :: root304Tail2
def root304Tail0 : List Nat := 3119 :: root304Tail1
def root304Candidates0 : List Nat := [16433]
set_option maxRecDepth 8192 in
theorem root304Candidates0_checked : candidates edge 4 2027 3119 root304Tail1 winner.bound = root304Candidates0 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
theorem root304Step0 : scan edge (search edge 3) 4 [2027] 2027 root304Tail0 winner =
    scan edge (search edge 3) 4 [2027] 2027 root304Tail1 winner := by
  rw [root304Tail0, scan, if_neg (by decide), root304Candidates0_checked]
  change scan edge (search edge 3) 4 [2027] 2027 root304Tail1
    (if 3 ≤ root304Candidates0.length then search edge 3 [2027, 3119] root304Candidates0 5146 winner else winner) = _
  rw [if_neg (by decide)]
def root304Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root304Candidates1_checked : candidates edge 4 2027 3761 root304Tail2 winner.bound = root304Candidates1 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root304Step1 : scan edge (search edge 3) 4 [2027] 2027 root304Tail1 winner =
    scan edge (search edge 3) 4 [2027] 2027 root304Tail2 winner := by
  rw [root304Tail1, scan, if_neg (by decide), root304Candidates1_checked]
  change scan edge (search edge 3) 4 [2027] 2027 root304Tail2
    (if 3 ≤ root304Candidates1.length then search edge 3 [2027, 3761] root304Candidates1 5788 winner else winner) = _
  rw [if_neg (by decide)]
def root304Candidates2 : List Nat := [10181]
set_option maxRecDepth 8192 in
theorem root304Candidates2_checked : candidates edge 4 2027 4259 root304Tail3 winner.bound = root304Candidates2 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root304Step2 : scan edge (search edge 3) 4 [2027] 2027 root304Tail2 winner =
    scan edge (search edge 3) 4 [2027] 2027 root304Tail3 winner := by
  rw [root304Tail2, scan, if_neg (by decide), root304Candidates2_checked]
  change scan edge (search edge 3) 4 [2027] 2027 root304Tail3
    (if 3 ≤ root304Candidates2.length then search edge 3 [2027, 4259] root304Candidates2 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root304Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root304Candidates3_checked : candidates edge 4 2027 5153 root304Tail4 winner.bound = root304Candidates3 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root304Step3 : scan edge (search edge 3) 4 [2027] 2027 root304Tail3 winner =
    scan edge (search edge 3) 4 [2027] 2027 root304Tail4 winner := by
  rw [root304Tail3, scan, if_neg (by decide), root304Candidates3_checked]
  change scan edge (search edge 3) 4 [2027] 2027 root304Tail4
    (if 3 ≤ root304Candidates3.length then search edge 3 [2027, 5153] root304Candidates3 7180 winner else winner) = _
  rw [if_neg (by decide)]
def root304Candidates4 : List Nat := [10463]
set_option maxRecDepth 8192 in
theorem root304Candidates4_checked : candidates edge 4 2027 5483 root304Tail5 winner.bound = root304Candidates4 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root304Step4 : scan edge (search edge 3) 4 [2027] 2027 root304Tail4 winner =
    scan edge (search edge 3) 4 [2027] 2027 root304Tail5 winner := by
  rw [root304Tail4, scan, if_neg (by decide), root304Candidates4_checked]
  change scan edge (search edge 3) 4 [2027] 2027 root304Tail5
    (if 3 ≤ root304Candidates4.length then search edge 3 [2027, 5483] root304Candidates4 7510 winner else winner) = _
  rw [if_neg (by decide)]
theorem root304Scan5 : scan edge (search edge 3) 4 [2027] 2027 root304Tail5 winner = winner := by rw [root304Tail5, scan, if_pos (by decide)]
theorem root304Scan4 : scan edge (search edge 3) 4 [2027] 2027 root304Tail4 winner = winner :=
  (root304Step4).trans root304Scan5
theorem root304Scan3 : scan edge (search edge 3) 4 [2027] 2027 root304Tail3 winner = winner :=
  (root304Step3).trans root304Scan4
theorem root304Scan2 : scan edge (search edge 3) 4 [2027] 2027 root304Tail2 winner = winner :=
  (root304Step2).trans root304Scan3
theorem root304Scan1 : scan edge (search edge 3) 4 [2027] 2027 root304Tail1 winner = winner :=
  (root304Step1).trans root304Scan2
theorem root304Scan0 : scan edge (search edge 3) 4 [2027] 2027 root304Tail0 winner = winner :=
  (root304Step0).trans root304Scan1
set_option maxRecDepth 4096 in
theorem root304Branch : search edge 4 [2027] root304Candidates 2027 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2027] 2027 root304Tail0 winner = winner
  exact root304Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
