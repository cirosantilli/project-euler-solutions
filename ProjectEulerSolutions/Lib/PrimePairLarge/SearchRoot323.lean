import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root323Candidates : List Nat := Data.row2161
set_option maxRecDepth 8192 in
theorem root323Candidates_checked : candidates edge 5 0 2161 tail324 winner.bound = root323Candidates := by
  apply candidates_eq_row edge 5 0 2161 tail324 root323Candidates winner.bound tail324_ordered
  · decide
  · decide
  · funext q
    rfl
def root323Tail38 : List Nat := []
def root323Tail37 : List Nat := 22639 :: root323Tail38
def root323Tail36 : List Nat := 22441 :: root323Tail37
def root323Tail35 : List Nat := 22279 :: root323Tail36
def root323Tail34 : List Nat := 21937 :: root323Tail35
def root323Tail33 : List Nat := 21559 :: root323Tail34
def root323Tail32 : List Nat := 20551 :: root323Tail33
def root323Tail31 : List Nat := 19819 :: root323Tail32
def root323Tail30 : List Nat := 19417 :: root323Tail31
def root323Tail29 : List Nat := 18457 :: root323Tail30
def root323Tail28 : List Nat := 18289 :: root323Tail29
def root323Tail27 : List Nat := 16273 :: root323Tail28
def root323Tail26 : List Nat := 14929 :: root323Tail27
def root323Tail25 : List Nat := 14629 :: root323Tail26
def root323Tail24 : List Nat := 13147 :: root323Tail25
def root323Tail23 : List Nat := 12781 :: root323Tail24
def root323Tail22 : List Nat := 12511 :: root323Tail23
def root323Tail21 : List Nat := 11437 :: root323Tail22
def root323Tail20 : List Nat := 10729 :: root323Tail21
def root323Tail19 : List Nat := 10177 :: root323Tail20
def root323Tail18 : List Nat := 9883 :: root323Tail19
def root323Tail17 : List Nat := 9721 :: root323Tail18
def root323Tail16 : List Nat := 9127 :: root323Tail17
def root323Tail15 : List Nat := 7759 :: root323Tail16
def root323Tail14 : List Nat := 7681 :: root323Tail15
def root323Tail13 : List Nat := 7621 :: root323Tail14
def root323Tail12 : List Nat := 7411 :: root323Tail13
def root323Tail11 : List Nat := 7237 :: root323Tail12
def root323Tail10 : List Nat := 7027 :: root323Tail11
def root323Tail9 : List Nat := 6949 :: root323Tail10
def root323Tail8 : List Nat := 6841 :: root323Tail9
def root323Tail7 : List Nat := 6373 :: root323Tail8
def root323Tail6 : List Nat := 6337 :: root323Tail7
def root323Tail5 : List Nat := 6211 :: root323Tail6
def root323Tail4 : List Nat := 6079 :: root323Tail5
def root323Tail3 : List Nat := 5479 :: root323Tail4
def root323Tail2 : List Nat := 4597 :: root323Tail3
def root323Tail1 : List Nat := 4357 :: root323Tail2
def root323Tail0 : List Nat := 2887 :: root323Tail1
def root323Candidates0 : List Nat := [4597, 12511]
set_option maxRecDepth 8192 in
theorem root323Candidates0_checked : candidates edge 4 2161 2887 root323Tail1 winner.bound = root323Candidates0 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root323Step0 : scan edge (search edge 3) 4 [2161] 2161 root323Tail0 winner =
    scan edge (search edge 3) 4 [2161] 2161 root323Tail1 winner := by
  rw [root323Tail0, scan, if_neg (by decide), root323Candidates0_checked]
  change scan edge (search edge 3) 4 [2161] 2161 root323Tail1
    (if 3 ≤ root323Candidates0.length then search edge 3 [2161, 2887] root323Candidates0 5048 winner else winner) = _
  rw [if_neg (by decide)]
def root323Candidates1 : List Nat := [5479, 7411, 18289]
set_option maxRecDepth 8192 in
theorem root323Candidates1_checked : candidates edge 4 2161 4357 root323Tail2 winner.bound = root323Candidates1 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root323Child1 : search edge 3 [2161, 4357] root323Candidates1 6518 winner = winner := by decide
theorem root323Step1 : scan edge (search edge 3) 4 [2161] 2161 root323Tail1 winner =
    scan edge (search edge 3) 4 [2161] 2161 root323Tail2 winner := by
  rw [root323Tail1, scan, if_neg (by decide), root323Candidates1_checked]
  change scan edge (search edge 3) 4 [2161] 2161 root323Tail2
    (if 3 ≤ root323Candidates1.length then search edge 3 [2161, 4357] root323Candidates1 6518 winner else winner) = _
  rw [if_pos (by decide), root323Child1]
def root323Candidates2 : List Nat := [6079, 11437]
set_option maxRecDepth 8192 in
theorem root323Candidates2_checked : candidates edge 4 2161 4597 root323Tail3 winner.bound = root323Candidates2 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root323Step2 : scan edge (search edge 3) 4 [2161] 2161 root323Tail2 winner =
    scan edge (search edge 3) 4 [2161] 2161 root323Tail3 winner := by
  rw [root323Tail2, scan, if_neg (by decide), root323Candidates2_checked]
  change scan edge (search edge 3) 4 [2161] 2161 root323Tail3
    (if 3 ≤ root323Candidates2.length then search edge 3 [2161, 4597] root323Candidates2 6758 winner else winner) = _
  rw [if_neg (by decide)]
def root323Candidates3 : List Nat := [9127]
set_option maxRecDepth 8192 in
theorem root323Candidates3_checked : candidates edge 4 2161 5479 root323Tail4 winner.bound = root323Candidates3 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root323Step3 : scan edge (search edge 3) 4 [2161] 2161 root323Tail3 winner =
    scan edge (search edge 3) 4 [2161] 2161 root323Tail4 winner := by
  rw [root323Tail3, scan, if_neg (by decide), root323Candidates3_checked]
  change scan edge (search edge 3) 4 [2161] 2161 root323Tail4
    (if 3 ≤ root323Candidates3.length then search edge 3 [2161, 5479] root323Candidates3 7640 winner else winner) = _
  rw [if_neg (by decide)]
theorem root323Scan4 : scan edge (search edge 3) 4 [2161] 2161 root323Tail4 winner = winner := by rw [root323Tail4, scan, if_pos (by decide)]
theorem root323Scan3 : scan edge (search edge 3) 4 [2161] 2161 root323Tail3 winner = winner :=
  (root323Step3).trans root323Scan4
theorem root323Scan2 : scan edge (search edge 3) 4 [2161] 2161 root323Tail2 winner = winner :=
  (root323Step2).trans root323Scan3
theorem root323Scan1 : scan edge (search edge 3) 4 [2161] 2161 root323Tail1 winner = winner :=
  (root323Step1).trans root323Scan2
theorem root323Scan0 : scan edge (search edge 3) 4 [2161] 2161 root323Tail0 winner = winner :=
  (root323Step0).trans root323Scan1
set_option maxRecDepth 4096 in
theorem root323Branch : search edge 4 [2161] root323Candidates 2161 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2161] 2161 root323Tail0 winner = winner
  exact root323Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
