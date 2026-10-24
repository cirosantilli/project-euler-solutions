import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root505Candidates : List Nat := Data.row3631
set_option maxRecDepth 8192 in
theorem root505Candidates_checked : candidates edge 5 0 3631 tail506 winner.bound = root505Candidates := by
  apply candidates_eq_row edge 5 0 3631 tail506 root505Candidates winner.bound tail506_ordered
  · decide
  · decide
  · funext q
    rfl
def root505Tail33 : List Nat := []
def root505Tail32 : List Nat := 22093 :: root505Tail33
def root505Tail31 : List Nat := 21943 :: root505Tail32
def root505Tail30 : List Nat := 21937 :: root505Tail31
def root505Tail29 : List Nat := 21397 :: root505Tail30
def root505Tail28 : List Nat := 21019 :: root505Tail29
def root505Tail27 : List Nat := 19927 :: root505Tail28
def root505Tail26 : List Nat := 19417 :: root505Tail27
def root505Tail25 : List Nat := 18439 :: root505Tail26
def root505Tail24 : List Nat := 18397 :: root505Tail25
def root505Tail23 : List Nat := 17659 :: root505Tail24
def root505Tail22 : List Nat := 17389 :: root505Tail23
def root505Tail21 : List Nat := 16987 :: root505Tail22
def root505Tail20 : List Nat := 16519 :: root505Tail21
def root505Tail19 : List Nat := 16063 :: root505Tail20
def root505Tail18 : List Nat := 14197 :: root505Tail19
def root505Tail17 : List Nat := 14107 :: root505Tail18
def root505Tail16 : List Nat := 14083 :: root505Tail17
def root505Tail15 : List Nat := 13687 :: root505Tail16
def root505Tail14 : List Nat := 13537 :: root505Tail15
def root505Tail13 : List Nat := 13411 :: root505Tail14
def root505Tail12 : List Nat := 12577 :: root505Tail13
def root505Tail11 : List Nat := 11383 :: root505Tail12
def root505Tail10 : List Nat := 8923 :: root505Tail11
def root505Tail9 : List Nat := 8839 :: root505Tail10
def root505Tail8 : List Nat := 8377 :: root505Tail9
def root505Tail7 : List Nat := 8017 :: root505Tail8
def root505Tail6 : List Nat := 7789 :: root505Tail7
def root505Tail5 : List Nat := 7507 :: root505Tail6
def root505Tail4 : List Nat := 6691 :: root505Tail5
def root505Tail3 : List Nat := 6547 :: root505Tail4
def root505Tail2 : List Nat := 5347 :: root505Tail3
def root505Tail1 : List Nat := 5119 :: root505Tail2
def root505Tail0 : List Nat := 4513 :: root505Tail1
def root505Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root505Candidates0_checked : candidates edge 4 3631 4513 root505Tail1 winner.bound = root505Candidates0 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root505Step0 : scan edge (search edge 3) 4 [3631] 3631 root505Tail0 winner =
    scan edge (search edge 3) 4 [3631] 3631 root505Tail1 winner := by
  rw [root505Tail0, scan, if_neg (by decide), root505Candidates0_checked]
  change scan edge (search edge 3) 4 [3631] 3631 root505Tail1
    (if 3 ≤ root505Candidates0.length then search edge 3 [3631, 4513] root505Candidates0 8144 winner else winner) = _
  rw [if_neg (by decide)]
def root505Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root505Candidates1_checked : candidates edge 4 3631 5119 root505Tail2 winner.bound = root505Candidates1 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root505Step1 : scan edge (search edge 3) 4 [3631] 3631 root505Tail1 winner =
    scan edge (search edge 3) 4 [3631] 3631 root505Tail2 winner := by
  rw [root505Tail1, scan, if_neg (by decide), root505Candidates1_checked]
  change scan edge (search edge 3) 4 [3631] 3631 root505Tail2
    (if 3 ≤ root505Candidates1.length then search edge 3 [3631, 5119] root505Candidates1 8750 winner else winner) = _
  rw [if_neg (by decide)]
def root505Candidates2 : List Nat := [7789]
set_option maxRecDepth 8192 in
theorem root505Candidates2_checked : candidates edge 4 3631 5347 root505Tail3 winner.bound = root505Candidates2 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root505Step2 : scan edge (search edge 3) 4 [3631] 3631 root505Tail2 winner =
    scan edge (search edge 3) 4 [3631] 3631 root505Tail3 winner := by
  rw [root505Tail2, scan, if_neg (by decide), root505Candidates2_checked]
  change scan edge (search edge 3) 4 [3631] 3631 root505Tail3
    (if 3 ≤ root505Candidates2.length then search edge 3 [3631, 5347] root505Candidates2 8978 winner else winner) = _
  rw [if_neg (by decide)]
theorem root505Scan3 : scan edge (search edge 3) 4 [3631] 3631 root505Tail3 winner = winner := by rw [root505Tail3, scan, if_pos (by decide)]
theorem root505Scan2 : scan edge (search edge 3) 4 [3631] 3631 root505Tail2 winner = winner :=
  (root505Step2).trans root505Scan3
theorem root505Scan1 : scan edge (search edge 3) 4 [3631] 3631 root505Tail1 winner = winner :=
  (root505Step1).trans root505Scan2
theorem root505Scan0 : scan edge (search edge 3) 4 [3631] 3631 root505Tail0 winner = winner :=
  (root505Step0).trans root505Scan1
set_option maxRecDepth 4096 in
theorem root505Branch : search edge 4 [3631] root505Candidates 3631 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3631] 3631 root505Tail0 winner = winner
  exact root505Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
