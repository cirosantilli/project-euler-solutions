import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root492Candidates : List Nat := Data.row3539
set_option maxRecDepth 8192 in
theorem root492Candidates_checked : candidates edge 5 0 3539 tail493 winner.bound = root492Candidates := by
  apply candidates_eq_row edge 5 0 3539 tail493 root492Candidates winner.bound tail493_ordered
  · decide
  · decide
  · funext q
    rfl
def root492Tail35 : List Nat := []
def root492Tail34 : List Nat := 22193 :: root492Tail35
def root492Tail33 : List Nat := 21767 :: root492Tail34
def root492Tail32 : List Nat := 21023 :: root492Tail33
def root492Tail31 : List Nat := 20627 :: root492Tail32
def root492Tail30 : List Nat := 19751 :: root492Tail31
def root492Tail29 : List Nat := 19457 :: root492Tail30
def root492Tail28 : List Nat := 18839 :: root492Tail29
def root492Tail27 : List Nat := 18119 :: root492Tail28
def root492Tail26 : List Nat := 18077 :: root492Tail27
def root492Tail25 : List Nat := 17909 :: root492Tail26
def root492Tail24 : List Nat := 17903 :: root492Tail25
def root492Tail23 : List Nat := 17021 :: root492Tail24
def root492Tail22 : List Nat := 16889 :: root492Tail23
def root492Tail21 : List Nat := 16103 :: root492Tail22
def root492Tail20 : List Nat := 16007 :: root492Tail21
def root492Tail19 : List Nat := 15383 :: root492Tail20
def root492Tail18 : List Nat := 15083 :: root492Tail19
def root492Tail17 : List Nat := 13997 :: root492Tail18
def root492Tail16 : List Nat := 13331 :: root492Tail17
def root492Tail15 : List Nat := 12941 :: root492Tail16
def root492Tail14 : List Nat := 12899 :: root492Tail15
def root492Tail13 : List Nat := 11981 :: root492Tail14
def root492Tail12 : List Nat := 10181 :: root492Tail13
def root492Tail11 : List Nat := 9803 :: root492Tail12
def root492Tail10 : List Nat := 9467 :: root492Tail11
def root492Tail9 : List Nat := 9137 :: root492Tail10
def root492Tail8 : List Nat := 7829 :: root492Tail9
def root492Tail7 : List Nat := 7001 :: root492Tail8
def root492Tail6 : List Nat := 6521 :: root492Tail7
def root492Tail5 : List Nat := 5861 :: root492Tail6
def root492Tail4 : List Nat := 5471 :: root492Tail5
def root492Tail3 : List Nat := 5441 :: root492Tail4
def root492Tail2 : List Nat := 4649 :: root492Tail3
def root492Tail1 : List Nat := 4013 :: root492Tail2
def root492Tail0 : List Nat := 3671 :: root492Tail1
def root492Candidates0 : List Nat := [9467, 10181]
set_option maxRecDepth 8192 in
theorem root492Candidates0_checked : candidates edge 4 3539 3671 root492Tail1 winner.bound = root492Candidates0 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root492Step0 : scan edge (search edge 3) 4 [3539] 3539 root492Tail0 winner =
    scan edge (search edge 3) 4 [3539] 3539 root492Tail1 winner := by
  rw [root492Tail0, scan, if_neg (by decide), root492Candidates0_checked]
  change scan edge (search edge 3) 4 [3539] 3539 root492Tail1
    (if 3 ≤ root492Candidates0.length then search edge 3 [3539, 3671] root492Candidates0 7210 winner else winner) = _
  rw [if_neg (by decide)]
def root492Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root492Candidates1_checked : candidates edge 4 3539 4013 root492Tail2 winner.bound = root492Candidates1 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root492Step1 : scan edge (search edge 3) 4 [3539] 3539 root492Tail1 winner =
    scan edge (search edge 3) 4 [3539] 3539 root492Tail2 winner := by
  rw [root492Tail1, scan, if_neg (by decide), root492Candidates1_checked]
  change scan edge (search edge 3) 4 [3539] 3539 root492Tail2
    (if 3 ≤ root492Candidates1.length then search edge 3 [3539, 4013] root492Candidates1 7552 winner else winner) = _
  rw [if_neg (by decide)]
def root492Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root492Candidates2_checked : candidates edge 4 3539 4649 root492Tail3 winner.bound = root492Candidates2 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root492Step2 : scan edge (search edge 3) 4 [3539] 3539 root492Tail2 winner =
    scan edge (search edge 3) 4 [3539] 3539 root492Tail3 winner := by
  rw [root492Tail2, scan, if_neg (by decide), root492Candidates2_checked]
  change scan edge (search edge 3) 4 [3539] 3539 root492Tail3
    (if 3 ≤ root492Candidates2.length then search edge 3 [3539, 4649] root492Candidates2 8188 winner else winner) = _
  rw [if_neg (by decide)]
def root492Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root492Candidates3_checked : candidates edge 4 3539 5441 root492Tail4 winner.bound = root492Candidates3 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root492Step3 : scan edge (search edge 3) 4 [3539] 3539 root492Tail3 winner =
    scan edge (search edge 3) 4 [3539] 3539 root492Tail4 winner := by
  rw [root492Tail3, scan, if_neg (by decide), root492Candidates3_checked]
  change scan edge (search edge 3) 4 [3539] 3539 root492Tail4
    (if 3 ≤ root492Candidates3.length then search edge 3 [3539, 5441] root492Candidates3 8980 winner else winner) = _
  rw [if_neg (by decide)]
def root492Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root492Candidates4_checked : candidates edge 4 3539 5471 root492Tail5 winner.bound = root492Candidates4 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root492Step4 : scan edge (search edge 3) 4 [3539] 3539 root492Tail4 winner =
    scan edge (search edge 3) 4 [3539] 3539 root492Tail5 winner := by
  rw [root492Tail4, scan, if_neg (by decide), root492Candidates4_checked]
  change scan edge (search edge 3) 4 [3539] 3539 root492Tail5
    (if 3 ≤ root492Candidates4.length then search edge 3 [3539, 5471] root492Candidates4 9010 winner else winner) = _
  rw [if_neg (by decide)]
theorem root492Scan5 : scan edge (search edge 3) 4 [3539] 3539 root492Tail5 winner = winner := by rw [root492Tail5, scan, if_pos (by decide)]
theorem root492Scan4 : scan edge (search edge 3) 4 [3539] 3539 root492Tail4 winner = winner :=
  (root492Step4).trans root492Scan5
theorem root492Scan3 : scan edge (search edge 3) 4 [3539] 3539 root492Tail3 winner = winner :=
  (root492Step3).trans root492Scan4
theorem root492Scan2 : scan edge (search edge 3) 4 [3539] 3539 root492Tail2 winner = winner :=
  (root492Step2).trans root492Scan3
theorem root492Scan1 : scan edge (search edge 3) 4 [3539] 3539 root492Tail1 winner = winner :=
  (root492Step1).trans root492Scan2
theorem root492Scan0 : scan edge (search edge 3) 4 [3539] 3539 root492Tail0 winner = winner :=
  (root492Step0).trans root492Scan1
set_option maxRecDepth 4096 in
theorem root492Branch : search edge 4 [3539] root492Candidates 3539 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3539] 3539 root492Tail0 winner = winner
  exact root492Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
