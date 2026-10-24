import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root551Candidates : List Nat := Data.row4013
set_option maxRecDepth 8192 in
theorem root551Candidates_checked : candidates edge 5 0 4013 tail552 winner.bound = root551Candidates := by
  apply candidates_eq_row edge 5 0 4013 tail552 root551Candidates winner.bound tail552_ordered
  · decide
  · decide
  · funext q
    rfl
def root551Tail33 : List Nat := []
def root551Tail32 : List Nat := 20903 :: root551Tail33
def root551Tail31 : List Nat := 20753 :: root551Tail32
def root551Tail30 : List Nat := 20483 :: root551Tail31
def root551Tail29 : List Nat := 20393 :: root551Tail30
def root551Tail28 : List Nat := 19973 :: root551Tail29
def root551Tail27 : List Nat := 19403 :: root551Tail28
def root551Tail26 : List Nat := 17957 :: root551Tail27
def root551Tail25 : List Nat := 17627 :: root551Tail26
def root551Tail24 : List Nat := 17387 :: root551Tail25
def root551Tail23 : List Nat := 16427 :: root551Tail24
def root551Tail22 : List Nat := 16193 :: root551Tail23
def root551Tail21 : List Nat := 14723 :: root551Tail22
def root551Tail20 : List Nat := 13967 :: root551Tail21
def root551Tail19 : List Nat := 13313 :: root551Tail20
def root551Tail18 : List Nat := 12539 :: root551Tail19
def root551Tail17 : List Nat := 12479 :: root551Tail18
def root551Tail16 : List Nat := 12203 :: root551Tail17
def root551Tail15 : List Nat := 11909 :: root551Tail16
def root551Tail14 : List Nat := 11549 :: root551Tail15
def root551Tail13 : List Nat := 10037 :: root551Tail14
def root551Tail12 : List Nat := 8867 :: root551Tail13
def root551Tail11 : List Nat := 8597 :: root551Tail12
def root551Tail10 : List Nat := 8387 :: root551Tail11
def root551Tail9 : List Nat := 8069 :: root551Tail10
def root551Tail8 : List Nat := 7541 :: root551Tail9
def root551Tail7 : List Nat := 6659 :: root551Tail8
def root551Tail6 : List Nat := 5867 :: root551Tail7
def root551Tail5 : List Nat := 5741 :: root551Tail6
def root551Tail4 : List Nat := 5573 :: root551Tail5
def root551Tail3 : List Nat := 5153 :: root551Tail4
def root551Tail2 : List Nat := 4799 :: root551Tail3
def root551Tail1 : List Nat := 4643 :: root551Tail2
def root551Tail0 : List Nat := 4421 :: root551Tail1
def root551Candidates0 : List Nat := [16427]
set_option maxRecDepth 8192 in
theorem root551Candidates0_checked : candidates edge 4 4013 4421 root551Tail1 winner.bound = root551Candidates0 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root551Step0 : scan edge (search edge 3) 4 [4013] 4013 root551Tail0 winner =
    scan edge (search edge 3) 4 [4013] 4013 root551Tail1 winner := by
  rw [root551Tail0, scan, if_neg (by decide), root551Candidates0_checked]
  change scan edge (search edge 3) 4 [4013] 4013 root551Tail1
    (if 3 ≤ root551Candidates0.length then search edge 3 [4013, 4421] root551Candidates0 8434 winner else winner) = _
  rw [if_neg (by decide)]
def root551Candidates1 : List Nat := [5153, 11909]
set_option maxRecDepth 8192 in
theorem root551Candidates1_checked : candidates edge 4 4013 4643 root551Tail2 winner.bound = root551Candidates1 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root551Step1 : scan edge (search edge 3) 4 [4013] 4013 root551Tail1 winner =
    scan edge (search edge 3) 4 [4013] 4013 root551Tail2 winner := by
  rw [root551Tail1, scan, if_neg (by decide), root551Candidates1_checked]
  change scan edge (search edge 3) 4 [4013] 4013 root551Tail2
    (if 3 ≤ root551Candidates1.length then search edge 3 [4013, 4643] root551Candidates1 8656 winner else winner) = _
  rw [if_neg (by decide)]
def root551Candidates2 : List Nat := [5867, 8597]
set_option maxRecDepth 8192 in
theorem root551Candidates2_checked : candidates edge 4 4013 4799 root551Tail3 winner.bound = root551Candidates2 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root551Step2 : scan edge (search edge 3) 4 [4013] 4013 root551Tail2 winner =
    scan edge (search edge 3) 4 [4013] 4013 root551Tail3 winner := by
  rw [root551Tail2, scan, if_neg (by decide), root551Candidates2_checked]
  change scan edge (search edge 3) 4 [4013] 4013 root551Tail3
    (if 3 ≤ root551Candidates2.length then search edge 3 [4013, 4799] root551Candidates2 8812 winner else winner) = _
  rw [if_neg (by decide)]
def root551Candidates3 : List Nat := [5867]
set_option maxRecDepth 8192 in
theorem root551Candidates3_checked : candidates edge 4 4013 5153 root551Tail4 winner.bound = root551Candidates3 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root551Step3 : scan edge (search edge 3) 4 [4013] 4013 root551Tail3 winner =
    scan edge (search edge 3) 4 [4013] 4013 root551Tail4 winner := by
  rw [root551Tail3, scan, if_neg (by decide), root551Candidates3_checked]
  change scan edge (search edge 3) 4 [4013] 4013 root551Tail4
    (if 3 ≤ root551Candidates3.length then search edge 3 [4013, 5153] root551Candidates3 9166 winner else winner) = _
  rw [if_neg (by decide)]
theorem root551Scan4 : scan edge (search edge 3) 4 [4013] 4013 root551Tail4 winner = winner := by rw [root551Tail4, scan, if_pos (by decide)]
theorem root551Scan3 : scan edge (search edge 3) 4 [4013] 4013 root551Tail3 winner = winner :=
  (root551Step3).trans root551Scan4
theorem root551Scan2 : scan edge (search edge 3) 4 [4013] 4013 root551Tail2 winner = winner :=
  (root551Step2).trans root551Scan3
theorem root551Scan1 : scan edge (search edge 3) 4 [4013] 4013 root551Tail1 winner = winner :=
  (root551Step1).trans root551Scan2
theorem root551Scan0 : scan edge (search edge 3) 4 [4013] 4013 root551Tail0 winner = winner :=
  (root551Step0).trans root551Scan1
set_option maxRecDepth 4096 in
theorem root551Branch : search edge 4 [4013] root551Candidates 4013 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4013] 4013 root551Tail0 winner = winner
  exact root551Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
