import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root596Candidates : List Nat := Data.row4397
set_option maxRecDepth 8192 in
theorem root596Candidates_checked : candidates edge 5 0 4397 tail597 winner.bound = root596Candidates := by
  apply candidates_eq_row edge 5 0 4397 tail597 root596Candidates winner.bound tail597_ordered
  · decide
  · decide
  · funext q
    rfl
def root596Tail38 : List Nat := []
def root596Tail37 : List Nat := 21221 :: root596Tail38
def root596Tail36 : List Nat := 20219 :: root596Tail37
def root596Tail35 : List Nat := 19919 :: root596Tail36
def root596Tail34 : List Nat := 19571 :: root596Tail35
def root596Tail33 : List Nat := 19403 :: root596Tail34
def root596Tail32 : List Nat := 18959 :: root596Tail33
def root596Tail31 : List Nat := 17123 :: root596Tail32
def root596Tail30 : List Nat := 16451 :: root596Tail31
def root596Tail29 : List Nat := 16253 :: root596Tail30
def root596Tail28 : List Nat := 16127 :: root596Tail29
def root596Tail27 : List Nat := 16097 :: root596Tail28
def root596Tail26 : List Nat := 15791 :: root596Tail27
def root596Tail25 : List Nat := 15641 :: root596Tail26
def root596Tail24 : List Nat := 14813 :: root596Tail25
def root596Tail23 : List Nat := 13121 :: root596Tail24
def root596Tail22 : List Nat := 12809 :: root596Tail23
def root596Tail21 : List Nat := 12473 :: root596Tail22
def root596Tail20 : List Nat := 12149 :: root596Tail21
def root596Tail19 : List Nat := 11549 :: root596Tail20
def root596Tail18 : List Nat := 11411 :: root596Tail19
def root596Tail17 : List Nat := 11057 :: root596Tail18
def root596Tail16 : List Nat := 10487 :: root596Tail17
def root596Tail15 : List Nat := 9431 :: root596Tail16
def root596Tail14 : List Nat := 9227 :: root596Tail15
def root596Tail13 : List Nat := 8861 :: root596Tail14
def root596Tail12 : List Nat := 8513 :: root596Tail13
def root596Tail11 : List Nat := 8087 :: root596Tail12
def root596Tail10 : List Nat := 7841 :: root596Tail11
def root596Tail9 : List Nat := 7823 :: root596Tail10
def root596Tail8 : List Nat := 7013 :: root596Tail9
def root596Tail7 : List Nat := 6833 :: root596Tail8
def root596Tail6 : List Nat := 6701 :: root596Tail7
def root596Tail5 : List Nat := 6563 :: root596Tail6
def root596Tail4 : List Nat := 6311 :: root596Tail5
def root596Tail3 : List Nat := 4871 :: root596Tail4
def root596Tail2 : List Nat := 4703 :: root596Tail3
def root596Tail1 : List Nat := 4691 :: root596Tail2
def root596Tail0 : List Nat := 4421 :: root596Tail1
def root596Candidates0 : List Nat := [6563]
set_option maxRecDepth 8192 in
theorem root596Candidates0_checked : candidates edge 4 4397 4421 root596Tail1 winner.bound = root596Candidates0 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root596Step0 : scan edge (search edge 3) 4 [4397] 4397 root596Tail0 winner =
    scan edge (search edge 3) 4 [4397] 4397 root596Tail1 winner := by
  rw [root596Tail0, scan, if_neg (by decide), root596Candidates0_checked]
  change scan edge (search edge 3) 4 [4397] 4397 root596Tail1
    (if 3 ≤ root596Candidates0.length then search edge 3 [4397, 4421] root596Candidates0 8818 winner else winner) = _
  rw [if_neg (by decide)]
def root596Candidates1 : List Nat := [16127]
set_option maxRecDepth 8192 in
theorem root596Candidates1_checked : candidates edge 4 4397 4691 root596Tail2 winner.bound = root596Candidates1 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root596Step1 : scan edge (search edge 3) 4 [4397] 4397 root596Tail1 winner =
    scan edge (search edge 3) 4 [4397] 4397 root596Tail2 winner := by
  rw [root596Tail1, scan, if_neg (by decide), root596Candidates1_checked]
  change scan edge (search edge 3) 4 [4397] 4397 root596Tail2
    (if 3 ≤ root596Candidates1.length then search edge 3 [4397, 4691] root596Candidates1 9088 winner else winner) = _
  rw [if_neg (by decide)]
def root596Candidates2 : List Nat := [11057, 15641, 16127]
set_option maxRecDepth 8192 in
theorem root596Candidates2_checked : candidates edge 4 4397 4703 root596Tail3 winner.bound = root596Candidates2 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root596Child2 : search edge 3 [4397, 4703] root596Candidates2 9100 winner = winner := by decide
theorem root596Step2 : scan edge (search edge 3) 4 [4397] 4397 root596Tail2 winner =
    scan edge (search edge 3) 4 [4397] 4397 root596Tail3 winner := by
  rw [root596Tail2, scan, if_neg (by decide), root596Candidates2_checked]
  change scan edge (search edge 3) 4 [4397] 4397 root596Tail3
    (if 3 ≤ root596Candidates2.length then search edge 3 [4397, 4703] root596Candidates2 9100 winner else winner) = _
  rw [if_pos (by decide), root596Child2]
def root596Candidates3 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root596Candidates3_checked : candidates edge 4 4397 4871 root596Tail4 winner.bound = root596Candidates3 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root596Step3 : scan edge (search edge 3) 4 [4397] 4397 root596Tail3 winner =
    scan edge (search edge 3) 4 [4397] 4397 root596Tail4 winner := by
  rw [root596Tail3, scan, if_neg (by decide), root596Candidates3_checked]
  change scan edge (search edge 3) 4 [4397] 4397 root596Tail4
    (if 3 ≤ root596Candidates3.length then search edge 3 [4397, 4871] root596Candidates3 9268 winner else winner) = _
  rw [if_neg (by decide)]
theorem root596Scan4 : scan edge (search edge 3) 4 [4397] 4397 root596Tail4 winner = winner := by rw [root596Tail4, scan, if_pos (by decide)]
theorem root596Scan3 : scan edge (search edge 3) 4 [4397] 4397 root596Tail3 winner = winner :=
  (root596Step3).trans root596Scan4
theorem root596Scan2 : scan edge (search edge 3) 4 [4397] 4397 root596Tail2 winner = winner :=
  (root596Step2).trans root596Scan3
theorem root596Scan1 : scan edge (search edge 3) 4 [4397] 4397 root596Tail1 winner = winner :=
  (root596Step1).trans root596Scan2
theorem root596Scan0 : scan edge (search edge 3) 4 [4397] 4397 root596Tail0 winner = winner :=
  (root596Step0).trans root596Scan1
set_option maxRecDepth 4096 in
theorem root596Branch : search edge 4 [4397] root596Candidates 4397 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4397] 4397 root596Tail0 winner = winner
  exact root596Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
