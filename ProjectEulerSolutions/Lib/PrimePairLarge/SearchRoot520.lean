import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root520Candidates : List Nat := Data.row3761
set_option maxRecDepth 8192 in
theorem root520Candidates_checked : candidates edge 5 0 3761 tail521 winner.bound = root520Candidates := by
  apply candidates_eq_row edge 5 0 3761 tail521 root520Candidates winner.bound tail521_ordered
  · decide
  · decide
  · funext q
    rfl
def root520Tail31 : List Nat := []
def root520Tail30 : List Nat := 20123 :: root520Tail31
def root520Tail29 : List Nat := 19661 :: root520Tail30
def root520Tail28 : List Nat := 19259 :: root520Tail29
def root520Tail27 : List Nat := 19157 :: root520Tail28
def root520Tail26 : List Nat := 18233 :: root520Tail27
def root520Tail25 : List Nat := 18059 :: root520Tail26
def root520Tail24 : List Nat := 17921 :: root520Tail25
def root520Tail23 : List Nat := 16553 :: root520Tail24
def root520Tail22 : List Nat := 15359 :: root520Tail23
def root520Tail21 : List Nat := 13841 :: root520Tail22
def root520Tail20 : List Nat := 13691 :: root520Tail21
def root520Tail19 : List Nat := 12413 :: root520Tail20
def root520Tail18 : List Nat := 12263 :: root520Tail19
def root520Tail17 : List Nat := 11447 :: root520Tail18
def root520Tail16 : List Nat := 11171 :: root520Tail17
def root520Tail15 : List Nat := 10037 :: root520Tail16
def root520Tail14 : List Nat := 9749 :: root520Tail15
def root520Tail13 : List Nat := 9209 :: root520Tail14
def root520Tail12 : List Nat := 8783 :: root520Tail13
def root520Tail11 : List Nat := 8573 :: root520Tail12
def root520Tail10 : List Nat := 8369 :: root520Tail11
def root520Tail9 : List Nat := 7793 :: root520Tail10
def root520Tail8 : List Nat := 7523 :: root520Tail9
def root520Tail7 : List Nat := 7499 :: root520Tail8
def root520Tail6 : List Nat := 6287 :: root520Tail7
def root520Tail5 : List Nat := 5843 :: root520Tail6
def root520Tail4 : List Nat := 5807 :: root520Tail5
def root520Tail3 : List Nat := 4799 :: root520Tail4
def root520Tail2 : List Nat := 4391 :: root520Tail3
def root520Tail1 : List Nat := 4229 :: root520Tail2
def root520Tail0 : List Nat := 3767 :: root520Tail1
def root520Candidates0 : List Nat := [7793]
set_option maxRecDepth 8192 in
theorem root520Candidates0_checked : candidates edge 4 3761 3767 root520Tail1 winner.bound = root520Candidates0 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root520Step0 : scan edge (search edge 3) 4 [3761] 3761 root520Tail0 winner =
    scan edge (search edge 3) 4 [3761] 3761 root520Tail1 winner := by
  rw [root520Tail0, scan, if_neg (by decide), root520Candidates0_checked]
  change scan edge (search edge 3) 4 [3761] 3761 root520Tail1
    (if 3 ≤ root520Candidates0.length then search edge 3 [3761, 3767] root520Candidates0 7528 winner else winner) = _
  rw [if_neg (by decide)]
def root520Candidates1 : List Nat := [4391]
set_option maxRecDepth 8192 in
theorem root520Candidates1_checked : candidates edge 4 3761 4229 root520Tail2 winner.bound = root520Candidates1 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root520Step1 : scan edge (search edge 3) 4 [3761] 3761 root520Tail1 winner =
    scan edge (search edge 3) 4 [3761] 3761 root520Tail2 winner := by
  rw [root520Tail1, scan, if_neg (by decide), root520Candidates1_checked]
  change scan edge (search edge 3) 4 [3761] 3761 root520Tail2
    (if 3 ≤ root520Candidates1.length then search edge 3 [3761, 4229] root520Candidates1 7990 winner else winner) = _
  rw [if_neg (by decide)]
def root520Candidates2 : List Nat := [8783]
set_option maxRecDepth 8192 in
theorem root520Candidates2_checked : candidates edge 4 3761 4391 root520Tail3 winner.bound = root520Candidates2 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root520Step2 : scan edge (search edge 3) 4 [3761] 3761 root520Tail2 winner =
    scan edge (search edge 3) 4 [3761] 3761 root520Tail3 winner := by
  rw [root520Tail2, scan, if_neg (by decide), root520Candidates2_checked]
  change scan edge (search edge 3) 4 [3761] 3761 root520Tail3
    (if 3 ≤ root520Candidates2.length then search edge 3 [3761, 4391] root520Candidates2 8152 winner else winner) = _
  rw [if_neg (by decide)]
def root520Candidates3 : List Nat := [11447]
set_option maxRecDepth 8192 in
theorem root520Candidates3_checked : candidates edge 4 3761 4799 root520Tail4 winner.bound = root520Candidates3 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root520Step3 : scan edge (search edge 3) 4 [3761] 3761 root520Tail3 winner =
    scan edge (search edge 3) 4 [3761] 3761 root520Tail4 winner := by
  rw [root520Tail3, scan, if_neg (by decide), root520Candidates3_checked]
  change scan edge (search edge 3) 4 [3761] 3761 root520Tail4
    (if 3 ≤ root520Candidates3.length then search edge 3 [3761, 4799] root520Candidates3 8560 winner else winner) = _
  rw [if_neg (by decide)]
theorem root520Scan4 : scan edge (search edge 3) 4 [3761] 3761 root520Tail4 winner = winner := by rw [root520Tail4, scan, if_pos (by decide)]
theorem root520Scan3 : scan edge (search edge 3) 4 [3761] 3761 root520Tail3 winner = winner :=
  (root520Step3).trans root520Scan4
theorem root520Scan2 : scan edge (search edge 3) 4 [3761] 3761 root520Tail2 winner = winner :=
  (root520Step2).trans root520Scan3
theorem root520Scan1 : scan edge (search edge 3) 4 [3761] 3761 root520Tail1 winner = winner :=
  (root520Step1).trans root520Scan2
theorem root520Scan0 : scan edge (search edge 3) 4 [3761] 3761 root520Tail0 winner = winner :=
  (root520Step0).trans root520Scan1
set_option maxRecDepth 4096 in
theorem root520Branch : search edge 4 [3761] root520Candidates 3761 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3761] 3761 root520Tail0 winner = winner
  exact root520Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
