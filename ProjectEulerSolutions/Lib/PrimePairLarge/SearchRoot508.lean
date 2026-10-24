import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root508Candidates : List Nat := Data.row3659
set_option maxRecDepth 8192 in
theorem root508Candidates_checked : candidates edge 5 0 3659 tail509 winner.bound = root508Candidates := by
  apply candidates_eq_row edge 5 0 3659 tail509 root508Candidates winner.bound tail509_ordered
  · decide
  · decide
  · funext q
    rfl
def root508Tail41 : List Nat := []
def root508Tail40 : List Nat := 22079 :: root508Tail41
def root508Tail39 : List Nat := 21407 :: root508Tail40
def root508Tail38 : List Nat := 20117 :: root508Tail39
def root508Tail37 : List Nat := 17729 :: root508Tail38
def root508Tail36 : List Nat := 15443 :: root508Tail37
def root508Tail35 : List Nat := 15287 :: root508Tail36
def root508Tail34 : List Nat := 14957 :: root508Tail35
def root508Tail33 : List Nat := 14897 :: root508Tail34
def root508Tail32 : List Nat := 14741 :: root508Tail33
def root508Tail31 : List Nat := 14639 :: root508Tail32
def root508Tail30 : List Nat := 14561 :: root508Tail31
def root508Tail29 : List Nat := 14387 :: root508Tail30
def root508Tail28 : List Nat := 14057 :: root508Tail29
def root508Tail27 : List Nat := 13901 :: root508Tail28
def root508Tail26 : List Nat := 13697 :: root508Tail27
def root508Tail25 : List Nat := 13217 :: root508Tail26
def root508Tail24 : List Nat := 12983 :: root508Tail25
def root508Tail23 : List Nat := 12941 :: root508Tail24
def root508Tail22 : List Nat := 12479 :: root508Tail23
def root508Tail21 : List Nat := 12161 :: root508Tail22
def root508Tail20 : List Nat := 11069 :: root508Tail21
def root508Tail19 : List Nat := 10163 :: root508Tail20
def root508Tail18 : List Nat := 9851 :: root508Tail19
def root508Tail17 : List Nat := 9767 :: root508Tail18
def root508Tail16 : List Nat := 9677 :: root508Tail17
def root508Tail15 : List Nat := 9473 :: root508Tail16
def root508Tail14 : List Nat := 9341 :: root508Tail15
def root508Tail13 : List Nat := 9221 :: root508Tail14
def root508Tail12 : List Nat := 8273 :: root508Tail13
def root508Tail11 : List Nat := 7829 :: root508Tail12
def root508Tail10 : List Nat := 7703 :: root508Tail11
def root508Tail9 : List Nat := 7229 :: root508Tail10
def root508Tail8 : List Nat := 6449 :: root508Tail9
def root508Tail7 : List Nat := 6317 :: root508Tail8
def root508Tail6 : List Nat := 6047 :: root508Tail7
def root508Tail5 : List Nat := 5981 :: root508Tail6
def root508Tail4 : List Nat := 4643 :: root508Tail5
def root508Tail3 : List Nat := 4373 :: root508Tail4
def root508Tail2 : List Nat := 4157 :: root508Tail3
def root508Tail1 : List Nat := 4091 :: root508Tail2
def root508Tail0 : List Nat := 3803 :: root508Tail1
def root508Candidates0 : List Nat := [15443]
set_option maxRecDepth 8192 in
theorem root508Candidates0_checked : candidates edge 4 3659 3803 root508Tail1 winner.bound = root508Candidates0 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root508Step0 : scan edge (search edge 3) 4 [3659] 3659 root508Tail0 winner =
    scan edge (search edge 3) 4 [3659] 3659 root508Tail1 winner := by
  rw [root508Tail0, scan, if_neg (by decide), root508Candidates0_checked]
  change scan edge (search edge 3) 4 [3659] 3659 root508Tail1
    (if 3 ≤ root508Candidates0.length then search edge 3 [3659, 3803] root508Candidates0 7462 winner else winner) = _
  rw [if_neg (by decide)]
def root508Candidates1 : List Nat := [12941]
set_option maxRecDepth 8192 in
theorem root508Candidates1_checked : candidates edge 4 3659 4091 root508Tail2 winner.bound = root508Candidates1 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root508Step1 : scan edge (search edge 3) 4 [3659] 3659 root508Tail1 winner =
    scan edge (search edge 3) 4 [3659] 3659 root508Tail2 winner := by
  rw [root508Tail1, scan, if_neg (by decide), root508Candidates1_checked]
  change scan edge (search edge 3) 4 [3659] 3659 root508Tail2
    (if 3 ≤ root508Candidates1.length then search edge 3 [3659, 4091] root508Candidates1 7750 winner else winner) = _
  rw [if_neg (by decide)]
def root508Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root508Candidates2_checked : candidates edge 4 3659 4157 root508Tail3 winner.bound = root508Candidates2 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root508Step2 : scan edge (search edge 3) 4 [3659] 3659 root508Tail2 winner =
    scan edge (search edge 3) 4 [3659] 3659 root508Tail3 winner := by
  rw [root508Tail2, scan, if_neg (by decide), root508Candidates2_checked]
  change scan edge (search edge 3) 4 [3659] 3659 root508Tail3
    (if 3 ≤ root508Candidates2.length then search edge 3 [3659, 4157] root508Candidates2 7816 winner else winner) = _
  rw [if_neg (by decide)]
def root508Candidates3 : List Nat := [10163, 12983, 15287]
set_option maxRecDepth 8192 in
theorem root508Candidates3_checked : candidates edge 4 3659 4373 root508Tail4 winner.bound = root508Candidates3 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root508Child3 : search edge 3 [3659, 4373] root508Candidates3 8032 winner = winner := by decide
theorem root508Step3 : scan edge (search edge 3) 4 [3659] 3659 root508Tail3 winner =
    scan edge (search edge 3) 4 [3659] 3659 root508Tail4 winner := by
  rw [root508Tail3, scan, if_neg (by decide), root508Candidates3_checked]
  change scan edge (search edge 3) 4 [3659] 3659 root508Tail4
    (if 3 ≤ root508Candidates3.length then search edge 3 [3659, 4373] root508Candidates3 8032 winner else winner) = _
  rw [if_pos (by decide), root508Child3]
def root508Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root508Candidates4_checked : candidates edge 4 3659 4643 root508Tail5 winner.bound = root508Candidates4 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root508Step4 : scan edge (search edge 3) 4 [3659] 3659 root508Tail4 winner =
    scan edge (search edge 3) 4 [3659] 3659 root508Tail5 winner := by
  rw [root508Tail4, scan, if_neg (by decide), root508Candidates4_checked]
  change scan edge (search edge 3) 4 [3659] 3659 root508Tail5
    (if 3 ≤ root508Candidates4.length then search edge 3 [3659, 4643] root508Candidates4 8302 winner else winner) = _
  rw [if_neg (by decide)]
theorem root508Scan5 : scan edge (search edge 3) 4 [3659] 3659 root508Tail5 winner = winner := by rw [root508Tail5, scan, if_pos (by decide)]
theorem root508Scan4 : scan edge (search edge 3) 4 [3659] 3659 root508Tail4 winner = winner :=
  (root508Step4).trans root508Scan5
theorem root508Scan3 : scan edge (search edge 3) 4 [3659] 3659 root508Tail3 winner = winner :=
  (root508Step3).trans root508Scan4
theorem root508Scan2 : scan edge (search edge 3) 4 [3659] 3659 root508Tail2 winner = winner :=
  (root508Step2).trans root508Scan3
theorem root508Scan1 : scan edge (search edge 3) 4 [3659] 3659 root508Tail1 winner = winner :=
  (root508Step1).trans root508Scan2
theorem root508Scan0 : scan edge (search edge 3) 4 [3659] 3659 root508Tail0 winner = winner :=
  (root508Step0).trans root508Scan1
set_option maxRecDepth 4096 in
theorem root508Branch : search edge 4 [3659] root508Candidates 3659 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3659] 3659 root508Tail0 winner = winner
  exact root508Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
