import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root507Candidates : List Nat := Data.row3643
set_option maxRecDepth 8192 in
theorem root507Candidates_checked : candidates edge 5 0 3643 tail508 winner.bound = root507Candidates := by
  apply candidates_eq_row edge 5 0 3643 tail508 root507Candidates winner.bound tail508_ordered
  · decide
  · decide
  · funext q
    rfl
def root507Tail32 : List Nat := []
def root507Tail31 : List Nat := 21313 :: root507Tail32
def root507Tail30 : List Nat := 21187 :: root507Tail31
def root507Tail29 : List Nat := 21157 :: root507Tail30
def root507Tail28 : List Nat := 21031 :: root507Tail29
def root507Tail27 : List Nat := 20107 :: root507Tail28
def root507Tail26 : List Nat := 19819 :: root507Tail27
def root507Tail25 : List Nat := 19687 :: root507Tail26
def root507Tail24 : List Nat := 18553 :: root507Tail25
def root507Tail23 : List Nat := 18289 :: root507Tail24
def root507Tail22 : List Nat := 18127 :: root507Tail23
def root507Tail21 : List Nat := 17881 :: root507Tail22
def root507Tail20 : List Nat := 16741 :: root507Tail21
def root507Tail19 : List Nat := 16417 :: root507Tail20
def root507Tail18 : List Nat := 16231 :: root507Tail19
def root507Tail17 : List Nat := 15817 :: root507Tail18
def root507Tail16 : List Nat := 15361 :: root507Tail17
def root507Tail15 : List Nat := 13687 :: root507Tail16
def root507Tail14 : List Nat := 12721 :: root507Tail15
def root507Tail13 : List Nat := 11587 :: root507Tail14
def root507Tail12 : List Nat := 10831 :: root507Tail13
def root507Tail11 : List Nat := 9817 :: root507Tail12
def root507Tail10 : List Nat := 8971 :: root507Tail11
def root507Tail9 : List Nat := 8929 :: root507Tail10
def root507Tail8 : List Nat := 8581 :: root507Tail9
def root507Tail7 : List Nat := 6607 :: root507Tail8
def root507Tail6 : List Nat := 6199 :: root507Tail7
def root507Tail5 : List Nat := 6073 :: root507Tail6
def root507Tail4 : List Nat := 5869 :: root507Tail5
def root507Tail3 : List Nat := 5449 :: root507Tail4
def root507Tail2 : List Nat := 4987 :: root507Tail3
def root507Tail1 : List Nat := 4561 :: root507Tail2
def root507Tail0 : List Nat := 4273 :: root507Tail1
def root507Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root507Candidates0_checked : candidates edge 4 3643 4273 root507Tail1 winner.bound = root507Candidates0 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root507Step0 : scan edge (search edge 3) 4 [3643] 3643 root507Tail0 winner =
    scan edge (search edge 3) 4 [3643] 3643 root507Tail1 winner := by
  rw [root507Tail0, scan, if_neg (by decide), root507Candidates0_checked]
  change scan edge (search edge 3) 4 [3643] 3643 root507Tail1
    (if 3 ≤ root507Candidates0.length then search edge 3 [3643, 4273] root507Candidates0 7916 winner else winner) = _
  rw [if_neg (by decide)]
def root507Candidates1 : List Nat := [6073, 15817]
set_option maxRecDepth 8192 in
theorem root507Candidates1_checked : candidates edge 4 3643 4561 root507Tail2 winner.bound = root507Candidates1 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root507Step1 : scan edge (search edge 3) 4 [3643] 3643 root507Tail1 winner =
    scan edge (search edge 3) 4 [3643] 3643 root507Tail2 winner := by
  rw [root507Tail1, scan, if_neg (by decide), root507Candidates1_checked]
  change scan edge (search edge 3) 4 [3643] 3643 root507Tail2
    (if 3 ≤ root507Candidates1.length then search edge 3 [3643, 4561] root507Candidates1 8204 winner else winner) = _
  rw [if_neg (by decide)]
def root507Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root507Candidates2_checked : candidates edge 4 3643 4987 root507Tail3 winner.bound = root507Candidates2 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root507Step2 : scan edge (search edge 3) 4 [3643] 3643 root507Tail2 winner =
    scan edge (search edge 3) 4 [3643] 3643 root507Tail3 winner := by
  rw [root507Tail2, scan, if_neg (by decide), root507Candidates2_checked]
  change scan edge (search edge 3) 4 [3643] 3643 root507Tail3
    (if 3 ≤ root507Candidates2.length then search edge 3 [3643, 4987] root507Candidates2 8630 winner else winner) = _
  rw [if_neg (by decide)]
def root507Candidates3 : List Nat := [6073, 8581, 8929, 9817]
set_option maxRecDepth 8192 in
theorem root507Candidates3_checked : candidates edge 4 3643 5449 root507Tail4 winner.bound = root507Candidates3 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root507Child3 : search edge 3 [3643, 5449] root507Candidates3 9092 winner = winner := by decide
theorem root507Step3 : scan edge (search edge 3) 4 [3643] 3643 root507Tail3 winner =
    scan edge (search edge 3) 4 [3643] 3643 root507Tail4 winner := by
  rw [root507Tail3, scan, if_neg (by decide), root507Candidates3_checked]
  change scan edge (search edge 3) 4 [3643] 3643 root507Tail4
    (if 3 ≤ root507Candidates3.length then search edge 3 [3643, 5449] root507Candidates3 9092 winner else winner) = _
  rw [if_pos (by decide), root507Child3]
theorem root507Scan4 : scan edge (search edge 3) 4 [3643] 3643 root507Tail4 winner = winner := by rw [root507Tail4, scan, if_pos (by decide)]
theorem root507Scan3 : scan edge (search edge 3) 4 [3643] 3643 root507Tail3 winner = winner :=
  (root507Step3).trans root507Scan4
theorem root507Scan2 : scan edge (search edge 3) 4 [3643] 3643 root507Tail2 winner = winner :=
  (root507Step2).trans root507Scan3
theorem root507Scan1 : scan edge (search edge 3) 4 [3643] 3643 root507Tail1 winner = winner :=
  (root507Step1).trans root507Scan2
theorem root507Scan0 : scan edge (search edge 3) 4 [3643] 3643 root507Tail0 winner = winner :=
  (root507Step0).trans root507Scan1
set_option maxRecDepth 4096 in
theorem root507Branch : search edge 4 [3643] root507Candidates 3643 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3643] 3643 root507Tail0 winner = winner
  exact root507Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
