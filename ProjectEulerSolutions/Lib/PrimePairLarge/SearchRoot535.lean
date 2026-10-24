import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root535Candidates : List Nat := Data.row3881
set_option maxRecDepth 8192 in
theorem root535Candidates_checked : candidates edge 5 0 3881 tail536 winner.bound = root535Candidates := by
  apply candidates_eq_row edge 5 0 3881 tail536 root535Candidates winner.bound tail536_ordered
  · decide
  · decide
  · funext q
    rfl
def root535Tail30 : List Nat := []
def root535Tail29 : List Nat := 21557 :: root535Tail30
def root535Tail28 : List Nat := 21317 :: root535Tail29
def root535Tail27 : List Nat := 20921 :: root535Tail28
def root535Tail26 : List Nat := 20549 :: root535Tail27
def root535Tail25 : List Nat := 18869 :: root535Tail26
def root535Tail24 : List Nat := 18701 :: root535Tail25
def root535Tail23 : List Nat := 15971 :: root535Tail24
def root535Tail22 : List Nat := 15263 :: root535Tail23
def root535Tail21 : List Nat := 15053 :: root535Tail22
def root535Tail20 : List Nat := 14633 :: root535Tail21
def root535Tail19 : List Nat := 14591 :: root535Tail20
def root535Tail18 : List Nat := 14537 :: root535Tail19
def root535Tail17 : List Nat := 14411 :: root535Tail18
def root535Tail16 : List Nat := 12953 :: root535Tail17
def root535Tail15 : List Nat := 12899 :: root535Tail16
def root535Tail14 : List Nat := 12401 :: root535Tail15
def root535Tail13 : List Nat := 10589 :: root535Tail14
def root535Tail12 : List Nat := 10253 :: root535Tail13
def root535Tail11 : List Nat := 9419 :: root535Tail12
def root535Tail10 : List Nat := 9281 :: root535Tail11
def root535Tail9 : List Nat := 8867 :: root535Tail10
def root535Tail8 : List Nat := 8609 :: root535Tail9
def root535Tail7 : List Nat := 8369 :: root535Tail8
def root535Tail6 : List Nat := 8363 :: root535Tail7
def root535Tail5 : List Nat := 6857 :: root535Tail6
def root535Tail4 : List Nat := 6269 :: root535Tail5
def root535Tail3 : List Nat := 5519 :: root535Tail4
def root535Tail2 : List Nat := 5261 :: root535Tail3
def root535Tail1 : List Nat := 4793 :: root535Tail2
def root535Tail0 : List Nat := 4421 :: root535Tail1
def root535Candidates0 : List Nat := [12401]
set_option maxRecDepth 8192 in
theorem root535Candidates0_checked : candidates edge 4 3881 4421 root535Tail1 winner.bound = root535Candidates0 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root535Step0 : scan edge (search edge 3) 4 [3881] 3881 root535Tail0 winner =
    scan edge (search edge 3) 4 [3881] 3881 root535Tail1 winner := by
  rw [root535Tail0, scan, if_neg (by decide), root535Candidates0_checked]
  change scan edge (search edge 3) 4 [3881] 3881 root535Tail1
    (if 3 ≤ root535Candidates0.length then search edge 3 [3881, 4421] root535Candidates0 8302 winner else winner) = _
  rw [if_neg (by decide)]
def root535Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root535Candidates1_checked : candidates edge 4 3881 4793 root535Tail2 winner.bound = root535Candidates1 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root535Step1 : scan edge (search edge 3) 4 [3881] 3881 root535Tail1 winner =
    scan edge (search edge 3) 4 [3881] 3881 root535Tail2 winner := by
  rw [root535Tail1, scan, if_neg (by decide), root535Candidates1_checked]
  change scan edge (search edge 3) 4 [3881] 3881 root535Tail2
    (if 3 ≤ root535Candidates1.length then search edge 3 [3881, 4793] root535Candidates1 8674 winner else winner) = _
  rw [if_neg (by decide)]
def root535Candidates2 : List Nat := [10253]
set_option maxRecDepth 8192 in
theorem root535Candidates2_checked : candidates edge 4 3881 5261 root535Tail3 winner.bound = root535Candidates2 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root535Step2 : scan edge (search edge 3) 4 [3881] 3881 root535Tail2 winner =
    scan edge (search edge 3) 4 [3881] 3881 root535Tail3 winner := by
  rw [root535Tail2, scan, if_neg (by decide), root535Candidates2_checked]
  change scan edge (search edge 3) 4 [3881] 3881 root535Tail3
    (if 3 ≤ root535Candidates2.length then search edge 3 [3881, 5261] root535Candidates2 9142 winner else winner) = _
  rw [if_neg (by decide)]
def root535Candidates3 : List Nat := [15263]
set_option maxRecDepth 8192 in
theorem root535Candidates3_checked : candidates edge 4 3881 5519 root535Tail4 winner.bound = root535Candidates3 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root535Step3 : scan edge (search edge 3) 4 [3881] 3881 root535Tail3 winner =
    scan edge (search edge 3) 4 [3881] 3881 root535Tail4 winner := by
  rw [root535Tail3, scan, if_neg (by decide), root535Candidates3_checked]
  change scan edge (search edge 3) 4 [3881] 3881 root535Tail4
    (if 3 ≤ root535Candidates3.length then search edge 3 [3881, 5519] root535Candidates3 9400 winner else winner) = _
  rw [if_neg (by decide)]
theorem root535Scan4 : scan edge (search edge 3) 4 [3881] 3881 root535Tail4 winner = winner := by rw [root535Tail4, scan, if_pos (by decide)]
theorem root535Scan3 : scan edge (search edge 3) 4 [3881] 3881 root535Tail3 winner = winner :=
  (root535Step3).trans root535Scan4
theorem root535Scan2 : scan edge (search edge 3) 4 [3881] 3881 root535Tail2 winner = winner :=
  (root535Step2).trans root535Scan3
theorem root535Scan1 : scan edge (search edge 3) 4 [3881] 3881 root535Tail1 winner = winner :=
  (root535Step1).trans root535Scan2
theorem root535Scan0 : scan edge (search edge 3) 4 [3881] 3881 root535Tail0 winner = winner :=
  (root535Step0).trans root535Scan1
set_option maxRecDepth 4096 in
theorem root535Branch : search edge 4 [3881] root535Candidates 3881 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3881] 3881 root535Tail0 winner = winner
  exact root535Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
