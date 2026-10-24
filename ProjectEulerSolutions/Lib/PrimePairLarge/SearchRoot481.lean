import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root481Candidates : List Nat := Data.row3461
set_option maxRecDepth 8192 in
theorem root481Candidates_checked : candidates edge 5 0 3461 tail482 winner.bound = root481Candidates := by
  apply candidates_eq_row edge 5 0 3461 tail482 root481Candidates winner.bound tail482_ordered
  · decide
  · decide
  · funext q
    rfl
def root481Tail32 : List Nat := []
def root481Tail31 : List Nat := 21767 :: root481Tail32
def root481Tail30 : List Nat := 21401 :: root481Tail31
def root481Tail29 : List Nat := 19919 :: root481Tail30
def root481Tail28 : List Nat := 19373 :: root481Tail29
def root481Tail27 : List Nat := 18917 :: root481Tail28
def root481Tail26 : List Nat := 18371 :: root481Tail27
def root481Tail25 : List Nat := 17903 :: root481Tail26
def root481Tail24 : List Nat := 17789 :: root481Tail25
def root481Tail23 : List Nat := 16091 :: root481Tail24
def root481Tail22 : List Nat := 15269 :: root481Tail23
def root481Tail21 : List Nat := 15083 :: root481Tail22
def root481Tail20 : List Nat := 14423 :: root481Tail21
def root481Tail19 : List Nat := 13967 :: root481Tail20
def root481Tail18 : List Nat := 13877 :: root481Tail19
def root481Tail17 : List Nat := 13121 :: root481Tail18
def root481Tail16 : List Nat := 13037 :: root481Tail17
def root481Tail15 : List Nat := 12953 :: root481Tail16
def root481Tail14 : List Nat := 11813 :: root481Tail15
def root481Tail13 : List Nat := 11261 :: root481Tail14
def root481Tail12 : List Nat := 11069 :: root481Tail13
def root481Tail11 : List Nat := 10559 :: root481Tail12
def root481Tail10 : List Nat := 8543 :: root481Tail11
def root481Tail9 : List Nat := 8117 :: root481Tail10
def root481Tail8 : List Nat := 7823 :: root481Tail9
def root481Tail7 : List Nat := 7523 :: root481Tail8
def root481Tail6 : List Nat := 6737 :: root481Tail7
def root481Tail5 : List Nat := 6221 :: root481Tail6
def root481Tail4 : List Nat := 6011 :: root481Tail5
def root481Tail3 : List Nat := 5333 :: root481Tail4
def root481Tail2 : List Nat := 4967 :: root481Tail3
def root481Tail1 : List Nat := 4583 :: root481Tail2
def root481Tail0 : List Nat := 3659 :: root481Tail1
def root481Candidates0 : List Nat := [11069]
set_option maxRecDepth 8192 in
theorem root481Candidates0_checked : candidates edge 4 3461 3659 root481Tail1 winner.bound = root481Candidates0 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
theorem root481Step0 : scan edge (search edge 3) 4 [3461] 3461 root481Tail0 winner =
    scan edge (search edge 3) 4 [3461] 3461 root481Tail1 winner := by
  rw [root481Tail0, scan, if_neg (by decide), root481Candidates0_checked]
  change scan edge (search edge 3) 4 [3461] 3461 root481Tail1
    (if 3 ≤ root481Candidates0.length then search edge 3 [3461, 3659] root481Candidates0 7120 winner else winner) = _
  rw [if_neg (by decide)]
def root481Candidates1 : List Nat := [7523]
set_option maxRecDepth 8192 in
theorem root481Candidates1_checked : candidates edge 4 3461 4583 root481Tail2 winner.bound = root481Candidates1 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root481Step1 : scan edge (search edge 3) 4 [3461] 3461 root481Tail1 winner =
    scan edge (search edge 3) 4 [3461] 3461 root481Tail2 winner := by
  rw [root481Tail1, scan, if_neg (by decide), root481Candidates1_checked]
  change scan edge (search edge 3) 4 [3461] 3461 root481Tail2
    (if 3 ≤ root481Candidates1.length then search edge 3 [3461, 4583] root481Candidates1 8044 winner else winner) = _
  rw [if_neg (by decide)]
def root481Candidates2 : List Nat := [8117]
set_option maxRecDepth 8192 in
theorem root481Candidates2_checked : candidates edge 4 3461 4967 root481Tail3 winner.bound = root481Candidates2 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root481Step2 : scan edge (search edge 3) 4 [3461] 3461 root481Tail2 winner =
    scan edge (search edge 3) 4 [3461] 3461 root481Tail3 winner := by
  rw [root481Tail2, scan, if_neg (by decide), root481Candidates2_checked]
  change scan edge (search edge 3) 4 [3461] 3461 root481Tail3
    (if 3 ≤ root481Candidates2.length then search edge 3 [3461, 4967] root481Candidates2 8428 winner else winner) = _
  rw [if_neg (by decide)]
def root481Candidates3 : List Nat := [8543]
set_option maxRecDepth 8192 in
theorem root481Candidates3_checked : candidates edge 4 3461 5333 root481Tail4 winner.bound = root481Candidates3 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root481Step3 : scan edge (search edge 3) 4 [3461] 3461 root481Tail3 winner =
    scan edge (search edge 3) 4 [3461] 3461 root481Tail4 winner := by
  rw [root481Tail3, scan, if_neg (by decide), root481Candidates3_checked]
  change scan edge (search edge 3) 4 [3461] 3461 root481Tail4
    (if 3 ≤ root481Candidates3.length then search edge 3 [3461, 5333] root481Candidates3 8794 winner else winner) = _
  rw [if_neg (by decide)]
theorem root481Scan4 : scan edge (search edge 3) 4 [3461] 3461 root481Tail4 winner = winner := by rw [root481Tail4, scan, if_pos (by decide)]
theorem root481Scan3 : scan edge (search edge 3) 4 [3461] 3461 root481Tail3 winner = winner :=
  (root481Step3).trans root481Scan4
theorem root481Scan2 : scan edge (search edge 3) 4 [3461] 3461 root481Tail2 winner = winner :=
  (root481Step2).trans root481Scan3
theorem root481Scan1 : scan edge (search edge 3) 4 [3461] 3461 root481Tail1 winner = winner :=
  (root481Step1).trans root481Scan2
theorem root481Scan0 : scan edge (search edge 3) 4 [3461] 3461 root481Tail0 winner = winner :=
  (root481Step0).trans root481Scan1
set_option maxRecDepth 4096 in
theorem root481Branch : search edge 4 [3461] root481Candidates 3461 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3461] 3461 root481Tail0 winner = winner
  exact root481Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
