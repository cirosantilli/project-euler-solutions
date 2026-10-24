import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root418Candidates : List Nat := Data.row2909
set_option maxRecDepth 8192 in
theorem root418Candidates_checked : candidates edge 5 0 2909 tail419 winner.bound = root418Candidates := by
  apply candidates_eq_row edge 5 0 2909 tail419 root418Candidates winner.bound tail419_ordered
  · decide
  · decide
  · funext q
    rfl
def root418Tail33 : List Nat := []
def root418Tail32 : List Nat := 22943 :: root418Tail33
def root418Tail31 : List Nat := 22937 :: root418Tail32
def root418Tail30 : List Nat := 22859 :: root418Tail31
def root418Tail29 : List Nat := 22013 :: root418Tail30
def root418Tail28 : List Nat := 21767 :: root418Tail29
def root418Tail27 : List Nat := 18917 :: root418Tail28
def root418Tail26 : List Nat := 16193 :: root418Tail27
def root418Tail25 : List Nat := 16103 :: root418Tail26
def root418Tail24 : List Nat := 15881 :: root418Tail25
def root418Tail23 : List Nat := 15797 :: root418Tail24
def root418Tail22 : List Nat := 12899 :: root418Tail23
def root418Tail21 : List Nat := 10889 :: root418Tail22
def root418Tail20 : List Nat := 10853 :: root418Tail21
def root418Tail19 : List Nat := 10091 :: root418Tail20
def root418Tail18 : List Nat := 9857 :: root418Tail19
def root418Tail17 : List Nat := 9689 :: root418Tail18
def root418Tail16 : List Nat := 9419 :: root418Tail17
def root418Tail15 : List Nat := 9311 :: root418Tail16
def root418Tail14 : List Nat := 9293 :: root418Tail15
def root418Tail13 : List Nat := 9227 :: root418Tail14
def root418Tail12 : List Nat := 9011 :: root418Tail13
def root418Tail11 : List Nat := 8669 :: root418Tail12
def root418Tail10 : List Nat := 7451 :: root418Tail11
def root418Tail9 : List Nat := 6983 :: root418Tail10
def root418Tail8 : List Nat := 6737 :: root418Tail9
def root418Tail7 : List Nat := 6323 :: root418Tail8
def root418Tail6 : List Nat := 6269 :: root418Tail7
def root418Tail5 : List Nat := 5813 :: root418Tail6
def root418Tail4 : List Nat := 5153 :: root418Tail5
def root418Tail3 : List Nat := 4253 :: root418Tail4
def root418Tail2 : List Nat := 4217 :: root418Tail3
def root418Tail1 : List Nat := 3557 :: root418Tail2
def root418Tail0 : List Nat := 3089 :: root418Tail1
def root418Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root418Candidates0_checked : candidates edge 4 2909 3089 root418Tail1 winner.bound = root418Candidates0 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root418Step0 : scan edge (search edge 3) 4 [2909] 2909 root418Tail0 winner =
    scan edge (search edge 3) 4 [2909] 2909 root418Tail1 winner := by
  rw [root418Tail0, scan, if_neg (by decide), root418Candidates0_checked]
  change scan edge (search edge 3) 4 [2909] 2909 root418Tail1
    (if 3 ≤ root418Candidates0.length then search edge 3 [2909, 3089] root418Candidates0 5998 winner else winner) = _
  rw [if_neg (by decide)]
def root418Candidates1 : List Nat := [9857]
set_option maxRecDepth 8192 in
theorem root418Candidates1_checked : candidates edge 4 2909 3557 root418Tail2 winner.bound = root418Candidates1 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root418Step1 : scan edge (search edge 3) 4 [2909] 2909 root418Tail1 winner =
    scan edge (search edge 3) 4 [2909] 2909 root418Tail2 winner := by
  rw [root418Tail1, scan, if_neg (by decide), root418Candidates1_checked]
  change scan edge (search edge 3) 4 [2909] 2909 root418Tail2
    (if 3 ≤ root418Candidates1.length then search edge 3 [2909, 3557] root418Candidates1 6466 winner else winner) = _
  rw [if_neg (by decide)]
def root418Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root418Candidates2_checked : candidates edge 4 2909 4217 root418Tail3 winner.bound = root418Candidates2 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root418Step2 : scan edge (search edge 3) 4 [2909] 2909 root418Tail2 winner =
    scan edge (search edge 3) 4 [2909] 2909 root418Tail3 winner := by
  rw [root418Tail2, scan, if_neg (by decide), root418Candidates2_checked]
  change scan edge (search edge 3) 4 [2909] 2909 root418Tail3
    (if 3 ≤ root418Candidates2.length then search edge 3 [2909, 4217] root418Candidates2 7126 winner else winner) = _
  rw [if_neg (by decide)]
def root418Candidates3 : List Nat := [9293]
set_option maxRecDepth 8192 in
theorem root418Candidates3_checked : candidates edge 4 2909 4253 root418Tail4 winner.bound = root418Candidates3 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root418Step3 : scan edge (search edge 3) 4 [2909] 2909 root418Tail3 winner =
    scan edge (search edge 3) 4 [2909] 2909 root418Tail4 winner := by
  rw [root418Tail3, scan, if_neg (by decide), root418Candidates3_checked]
  change scan edge (search edge 3) 4 [2909] 2909 root418Tail4
    (if 3 ≤ root418Candidates3.length then search edge 3 [2909, 4253] root418Candidates3 7162 winner else winner) = _
  rw [if_neg (by decide)]
def root418Candidates4 : List Nat := [9311]
set_option maxRecDepth 8192 in
theorem root418Candidates4_checked : candidates edge 4 2909 5153 root418Tail5 winner.bound = root418Candidates4 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root418Step4 : scan edge (search edge 3) 4 [2909] 2909 root418Tail4 winner =
    scan edge (search edge 3) 4 [2909] 2909 root418Tail5 winner := by
  rw [root418Tail4, scan, if_neg (by decide), root418Candidates4_checked]
  change scan edge (search edge 3) 4 [2909] 2909 root418Tail5
    (if 3 ≤ root418Candidates4.length then search edge 3 [2909, 5153] root418Candidates4 8062 winner else winner) = _
  rw [if_neg (by decide)]
theorem root418Scan5 : scan edge (search edge 3) 4 [2909] 2909 root418Tail5 winner = winner := by rw [root418Tail5, scan, if_pos (by decide)]
theorem root418Scan4 : scan edge (search edge 3) 4 [2909] 2909 root418Tail4 winner = winner :=
  (root418Step4).trans root418Scan5
theorem root418Scan3 : scan edge (search edge 3) 4 [2909] 2909 root418Tail3 winner = winner :=
  (root418Step3).trans root418Scan4
theorem root418Scan2 : scan edge (search edge 3) 4 [2909] 2909 root418Tail2 winner = winner :=
  (root418Step2).trans root418Scan3
theorem root418Scan1 : scan edge (search edge 3) 4 [2909] 2909 root418Tail1 winner = winner :=
  (root418Step1).trans root418Scan2
theorem root418Scan0 : scan edge (search edge 3) 4 [2909] 2909 root418Tail0 winner = winner :=
  (root418Step0).trans root418Scan1
set_option maxRecDepth 4096 in
theorem root418Branch : search edge 4 [2909] root418Candidates 2909 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2909] 2909 root418Tail0 winner = winner
  exact root418Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
