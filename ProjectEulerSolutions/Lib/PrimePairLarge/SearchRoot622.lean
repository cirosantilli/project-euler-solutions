import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root622Candidates : List Nat := Data.row4637
set_option maxRecDepth 8192 in
theorem root622Candidates_checked : candidates edge 5 0 4637 tail623 winner.bound = root622Candidates := by
  apply candidates_eq_row edge 5 0 4637 tail623 root622Candidates winner.bound tail623_ordered
  · decide
  · decide
  · funext q
    rfl
def root622Tail27 : List Nat := []
def root622Tail26 : List Nat := 20261 :: root622Tail27
def root622Tail25 : List Nat := 19661 :: root622Tail26
def root622Tail24 : List Nat := 18797 :: root622Tail25
def root622Tail23 : List Nat := 17489 :: root622Tail24
def root622Tail22 : List Nat := 17393 :: root622Tail23
def root622Tail21 : List Nat := 17351 :: root622Tail22
def root622Tail20 : List Nat := 16217 :: root622Tail21
def root622Tail19 : List Nat := 15971 :: root622Tail20
def root622Tail18 : List Nat := 14759 :: root622Tail19
def root622Tail17 : List Nat := 14243 :: root622Tail18
def root622Tail16 : List Nat := 13619 :: root622Tail17
def root622Tail15 : List Nat := 13043 :: root622Tail16
def root622Tail14 : List Nat := 12983 :: root622Tail15
def root622Tail13 : List Nat := 12203 :: root622Tail14
def root622Tail12 : List Nat := 11069 :: root622Tail13
def root622Tail11 : List Nat := 10733 :: root622Tail12
def root622Tail10 : List Nat := 10253 :: root622Tail11
def root622Tail9 : List Nat := 9041 :: root622Tail10
def root622Tail8 : List Nat := 7949 :: root622Tail9
def root622Tail7 : List Nat := 7517 :: root622Tail8
def root622Tail6 : List Nat := 6569 :: root622Tail7
def root622Tail5 : List Nat := 6173 :: root622Tail6
def root622Tail4 : List Nat := 5939 :: root622Tail5
def root622Tail3 : List Nat := 5471 :: root622Tail4
def root622Tail2 : List Nat := 5081 :: root622Tail3
def root622Tail1 : List Nat := 5009 :: root622Tail2
def root622Tail0 : List Nat := 4919 :: root622Tail1
def root622Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root622Candidates0_checked : candidates edge 4 4637 4919 root622Tail1 winner.bound = root622Candidates0 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root622Step0 : scan edge (search edge 3) 4 [4637] 4637 root622Tail0 winner =
    scan edge (search edge 3) 4 [4637] 4637 root622Tail1 winner := by
  rw [root622Tail0, scan, if_neg (by decide), root622Candidates0_checked]
  change scan edge (search edge 3) 4 [4637] 4637 root622Tail1
    (if 3 ≤ root622Candidates0.length then search edge 3 [4637, 4919] root622Candidates0 9556 winner else winner) = _
  rw [if_neg (by decide)]
def root622Candidates1 : List Nat := [13043]
set_option maxRecDepth 8192 in
theorem root622Candidates1_checked : candidates edge 4 4637 5009 root622Tail2 winner.bound = root622Candidates1 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root622Step1 : scan edge (search edge 3) 4 [4637] 4637 root622Tail1 winner =
    scan edge (search edge 3) 4 [4637] 4637 root622Tail2 winner := by
  rw [root622Tail1, scan, if_neg (by decide), root622Candidates1_checked]
  change scan edge (search edge 3) 4 [4637] 4637 root622Tail2
    (if 3 ≤ root622Candidates1.length then search edge 3 [4637, 5009] root622Candidates1 9646 winner else winner) = _
  rw [if_neg (by decide)]
def root622Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root622Candidates2_checked : candidates edge 4 4637 5081 root622Tail3 winner.bound = root622Candidates2 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root622Step2 : scan edge (search edge 3) 4 [4637] 4637 root622Tail2 winner =
    scan edge (search edge 3) 4 [4637] 4637 root622Tail3 winner := by
  rw [root622Tail2, scan, if_neg (by decide), root622Candidates2_checked]
  change scan edge (search edge 3) 4 [4637] 4637 root622Tail3
    (if 3 ≤ root622Candidates2.length then search edge 3 [4637, 5081] root622Candidates2 9718 winner else winner) = _
  rw [if_neg (by decide)]
theorem root622Scan3 : scan edge (search edge 3) 4 [4637] 4637 root622Tail3 winner = winner := by rw [root622Tail3, scan, if_pos (by decide)]
theorem root622Scan2 : scan edge (search edge 3) 4 [4637] 4637 root622Tail2 winner = winner :=
  (root622Step2).trans root622Scan3
theorem root622Scan1 : scan edge (search edge 3) 4 [4637] 4637 root622Tail1 winner = winner :=
  (root622Step1).trans root622Scan2
theorem root622Scan0 : scan edge (search edge 3) 4 [4637] 4637 root622Tail0 winner = winner :=
  (root622Step0).trans root622Scan1
set_option maxRecDepth 4096 in
theorem root622Branch : search edge 4 [4637] root622Candidates 4637 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4637] 4637 root622Tail0 winner = winner
  exact root622Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
