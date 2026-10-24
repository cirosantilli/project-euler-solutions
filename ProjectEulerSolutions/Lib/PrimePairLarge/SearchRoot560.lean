import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root560Candidates : List Nat := Data.row4091
set_option maxRecDepth 8192 in
theorem root560Candidates_checked : candidates edge 5 0 4091 tail561 winner.bound = root560Candidates := by
  apply candidates_eq_row edge 5 0 4091 tail561 root560Candidates winner.bound tail561_ordered
  · decide
  · decide
  · funext q
    rfl
def root560Tail28 : List Nat := []
def root560Tail27 : List Nat := 20753 :: root560Tail28
def root560Tail26 : List Nat := 19973 :: root560Tail27
def root560Tail25 : List Nat := 19793 :: root560Tail26
def root560Tail24 : List Nat := 19751 :: root560Tail25
def root560Tail23 : List Nat := 18539 :: root560Tail24
def root560Tail22 : List Nat := 16937 :: root560Tail23
def root560Tail21 : List Nat := 14549 :: root560Tail22
def root560Tail20 : List Nat := 14423 :: root560Tail21
def root560Tail19 : List Nat := 14369 :: root560Tail20
def root560Tail18 : List Nat := 13841 :: root560Tail19
def root560Tail17 : List Nat := 13241 :: root560Tail18
def root560Tail16 : List Nat := 12941 :: root560Tail17
def root560Tail15 : List Nat := 11321 :: root560Tail16
def root560Tail14 : List Nat := 10973 :: root560Tail15
def root560Tail13 : List Nat := 10313 :: root560Tail14
def root560Tail12 : List Nat := 9839 :: root560Tail13
def root560Tail11 : List Nat := 9491 :: root560Tail12
def root560Tail10 : List Nat := 9161 :: root560Tail11
def root560Tail9 : List Nat := 8237 :: root560Tail10
def root560Tail8 : List Nat := 7937 :: root560Tail9
def root560Tail7 : List Nat := 7577 :: root560Tail8
def root560Tail6 : List Nat := 7559 :: root560Tail7
def root560Tail5 : List Nat := 7103 :: root560Tail6
def root560Tail4 : List Nat := 6173 :: root560Tail5
def root560Tail3 : List Nat := 5669 :: root560Tail4
def root560Tail2 : List Nat := 4919 :: root560Tail3
def root560Tail1 : List Nat := 4283 :: root560Tail2
def root560Tail0 : List Nat := 4241 :: root560Tail1
def root560Candidates0 : List Nat := [7103, 7559, 16937]
set_option maxRecDepth 8192 in
theorem root560Candidates0_checked : candidates edge 4 4091 4241 root560Tail1 winner.bound = root560Candidates0 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root560Child0 : search edge 3 [4091, 4241] root560Candidates0 8332 winner = winner := by decide
theorem root560Step0 : scan edge (search edge 3) 4 [4091] 4091 root560Tail0 winner =
    scan edge (search edge 3) 4 [4091] 4091 root560Tail1 winner := by
  rw [root560Tail0, scan, if_neg (by decide), root560Candidates0_checked]
  change scan edge (search edge 3) 4 [4091] 4091 root560Tail1
    (if 3 ≤ root560Candidates0.length then search edge 3 [4091, 4241] root560Candidates0 8332 winner else winner) = _
  rw [if_pos (by decide), root560Child0]
def root560Candidates1 : List Nat := [5669, 7103, 13241, 14423]
set_option maxRecDepth 8192 in
theorem root560Candidates1_checked : candidates edge 4 4091 4283 root560Tail2 winner.bound = root560Candidates1 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root560Child1 : search edge 3 [4091, 4283] root560Candidates1 8374 winner = winner := by decide
theorem root560Step1 : scan edge (search edge 3) 4 [4091] 4091 root560Tail1 winner =
    scan edge (search edge 3) 4 [4091] 4091 root560Tail2 winner := by
  rw [root560Tail1, scan, if_neg (by decide), root560Candidates1_checked]
  change scan edge (search edge 3) 4 [4091] 4091 root560Tail2
    (if 3 ≤ root560Candidates1.length then search edge 3 [4091, 4283] root560Candidates1 8374 winner else winner) = _
  rw [if_pos (by decide), root560Child1]
def root560Candidates2 : List Nat := [7559, 13841]
set_option maxRecDepth 8192 in
theorem root560Candidates2_checked : candidates edge 4 4091 4919 root560Tail3 winner.bound = root560Candidates2 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root560Step2 : scan edge (search edge 3) 4 [4091] 4091 root560Tail2 winner =
    scan edge (search edge 3) 4 [4091] 4091 root560Tail3 winner := by
  rw [root560Tail2, scan, if_neg (by decide), root560Candidates2_checked]
  change scan edge (search edge 3) 4 [4091] 4091 root560Tail3
    (if 3 ≤ root560Candidates2.length then search edge 3 [4091, 4919] root560Candidates2 9010 winner else winner) = _
  rw [if_neg (by decide)]
theorem root560Scan3 : scan edge (search edge 3) 4 [4091] 4091 root560Tail3 winner = winner := by rw [root560Tail3, scan, if_pos (by decide)]
theorem root560Scan2 : scan edge (search edge 3) 4 [4091] 4091 root560Tail2 winner = winner :=
  (root560Step2).trans root560Scan3
theorem root560Scan1 : scan edge (search edge 3) 4 [4091] 4091 root560Tail1 winner = winner :=
  (root560Step1).trans root560Scan2
theorem root560Scan0 : scan edge (search edge 3) 4 [4091] 4091 root560Tail0 winner = winner :=
  (root560Step0).trans root560Scan1
set_option maxRecDepth 4096 in
theorem root560Branch : search edge 4 [4091] root560Candidates 4091 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4091] 4091 root560Tail0 winner = winner
  exact root560Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
