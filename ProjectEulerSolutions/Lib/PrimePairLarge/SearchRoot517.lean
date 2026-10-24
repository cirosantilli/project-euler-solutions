import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root517Candidates : List Nat := Data.row3727
set_option maxRecDepth 8192 in
theorem root517Candidates_checked : candidates edge 5 0 3727 tail518 winner.bound = root517Candidates := by
  apply candidates_eq_row edge 5 0 3727 tail518 root517Candidates winner.bound tail518_ordered
  · decide
  · decide
  · funext q
    rfl
def root517Tail28 : List Nat := []
def root517Tail27 : List Nat := 20899 :: root517Tail28
def root517Tail26 : List Nat := 19507 :: root517Tail27
def root517Tail25 : List Nat := 19429 :: root517Tail26
def root517Tail24 : List Nat := 17959 :: root517Tail25
def root517Tail23 : List Nat := 16987 :: root517Tail24
def root517Tail22 : List Nat := 16699 :: root517Tail23
def root517Tail21 : List Nat := 16189 :: root517Tail22
def root517Tail20 : List Nat := 13963 :: root517Tail21
def root517Tail19 : List Nat := 13921 :: root517Tail20
def root517Tail18 : List Nat := 13381 :: root517Tail19
def root517Tail17 : List Nat := 13171 :: root517Tail18
def root517Tail16 : List Nat := 12073 :: root517Tail17
def root517Tail15 : List Nat := 11743 :: root517Tail16
def root517Tail14 : List Nat := 10993 :: root517Tail15
def root517Tail13 : List Nat := 9817 :: root517Tail14
def root517Tail12 : List Nat := 9049 :: root517Tail13
def root517Tail11 : List Nat := 8887 :: root517Tail12
def root517Tail10 : List Nat := 8713 :: root517Tail11
def root517Tail9 : List Nat := 8623 :: root517Tail10
def root517Tail8 : List Nat := 8419 :: root517Tail9
def root517Tail7 : List Nat := 7699 :: root517Tail8
def root517Tail6 : List Nat := 6961 :: root517Tail7
def root517Tail5 : List Nat := 6949 :: root517Tail6
def root517Tail4 : List Nat := 6037 :: root517Tail5
def root517Tail3 : List Nat := 5659 :: root517Tail4
def root517Tail2 : List Nat := 5011 :: root517Tail3
def root517Tail1 : List Nat := 4639 :: root517Tail2
def root517Tail0 : List Nat := 4129 :: root517Tail1
def root517Candidates0 : List Nat := [6949]
set_option maxRecDepth 8192 in
theorem root517Candidates0_checked : candidates edge 4 3727 4129 root517Tail1 winner.bound = root517Candidates0 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root517Step0 : scan edge (search edge 3) 4 [3727] 3727 root517Tail0 winner =
    scan edge (search edge 3) 4 [3727] 3727 root517Tail1 winner := by
  rw [root517Tail0, scan, if_neg (by decide), root517Candidates0_checked]
  change scan edge (search edge 3) 4 [3727] 3727 root517Tail1
    (if 3 ≤ root517Candidates0.length then search edge 3 [3727, 4129] root517Candidates0 7856 winner else winner) = _
  rw [if_neg (by decide)]
def root517Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root517Candidates1_checked : candidates edge 4 3727 4639 root517Tail2 winner.bound = root517Candidates1 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root517Step1 : scan edge (search edge 3) 4 [3727] 3727 root517Tail1 winner =
    scan edge (search edge 3) 4 [3727] 3727 root517Tail2 winner := by
  rw [root517Tail1, scan, if_neg (by decide), root517Candidates1_checked]
  change scan edge (search edge 3) 4 [3727] 3727 root517Tail2
    (if 3 ≤ root517Candidates1.length then search edge 3 [3727, 4639] root517Candidates1 8366 winner else winner) = _
  rw [if_neg (by decide)]
def root517Candidates2 : List Nat := [6949, 6961, 9049]
set_option maxRecDepth 8192 in
theorem root517Candidates2_checked : candidates edge 4 3727 5011 root517Tail3 winner.bound = root517Candidates2 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root517Child2 : search edge 3 [3727, 5011] root517Candidates2 8738 winner = winner := by decide
theorem root517Step2 : scan edge (search edge 3) 4 [3727] 3727 root517Tail2 winner =
    scan edge (search edge 3) 4 [3727] 3727 root517Tail3 winner := by
  rw [root517Tail2, scan, if_neg (by decide), root517Candidates2_checked]
  change scan edge (search edge 3) 4 [3727] 3727 root517Tail3
    (if 3 ≤ root517Candidates2.length then search edge 3 [3727, 5011] root517Candidates2 8738 winner else winner) = _
  rw [if_pos (by decide), root517Child2]
theorem root517Scan3 : scan edge (search edge 3) 4 [3727] 3727 root517Tail3 winner = winner := by rw [root517Tail3, scan, if_pos (by decide)]
theorem root517Scan2 : scan edge (search edge 3) 4 [3727] 3727 root517Tail2 winner = winner :=
  (root517Step2).trans root517Scan3
theorem root517Scan1 : scan edge (search edge 3) 4 [3727] 3727 root517Tail1 winner = winner :=
  (root517Step1).trans root517Scan2
theorem root517Scan0 : scan edge (search edge 3) 4 [3727] 3727 root517Tail0 winner = winner :=
  (root517Step0).trans root517Scan1
set_option maxRecDepth 4096 in
theorem root517Branch : search edge 4 [3727] root517Candidates 3727 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3727] 3727 root517Tail0 winner = winner
  exact root517Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
