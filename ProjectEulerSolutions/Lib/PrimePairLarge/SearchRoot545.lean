import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root545Candidates : List Nat := Data.row3947
set_option maxRecDepth 8192 in
theorem root545Candidates_checked : candidates edge 5 0 3947 tail546 winner.bound = root545Candidates := by
  apply candidates_eq_row edge 5 0 3947 tail546 root545Candidates winner.bound tail546_ordered
  · decide
  · decide
  · funext q
    rfl
def root545Tail39 : List Nat := []
def root545Tail38 : List Nat := 21773 :: root545Tail39
def root545Tail37 : List Nat := 20789 :: root545Tail38
def root545Tail36 : List Nat := 20681 :: root545Tail37
def root545Tail35 : List Nat := 20441 :: root545Tail36
def root545Tail34 : List Nat := 18959 :: root545Tail35
def root545Tail33 : List Nat := 18917 :: root545Tail34
def root545Tail32 : List Nat := 17909 :: root545Tail33
def root545Tail31 : List Nat := 17627 :: root545Tail32
def root545Tail30 : List Nat := 17333 :: root545Tail31
def root545Tail29 : List Nat := 16103 :: root545Tail30
def root545Tail28 : List Nat := 15803 :: root545Tail29
def root545Tail27 : List Nat := 15137 :: root545Tail28
def root545Tail26 : List Nat := 14741 :: root545Tail27
def root545Tail25 : List Nat := 14081 :: root545Tail26
def root545Tail24 : List Nat := 14057 :: root545Tail25
def root545Tail23 : List Nat := 12743 :: root545Tail24
def root545Tail22 : List Nat := 12713 :: root545Tail23
def root545Tail21 : List Nat := 12347 :: root545Tail22
def root545Tail20 : List Nat := 11897 :: root545Tail21
def root545Tail19 : List Nat := 11813 :: root545Tail20
def root545Tail18 : List Nat := 9857 :: root545Tail19
def root545Tail17 : List Nat := 9173 :: root545Tail18
def root545Tail16 : List Nat := 8861 :: root545Tail17
def root545Tail15 : List Nat := 8849 :: root545Tail16
def root545Tail14 : List Nat := 8837 :: root545Tail15
def root545Tail13 : List Nat := 8513 :: root545Tail14
def root545Tail12 : List Nat := 8147 :: root545Tail13
def root545Tail11 : List Nat := 8093 :: root545Tail12
def root545Tail10 : List Nat := 7001 :: root545Tail11
def root545Tail9 : List Nat := 6803 :: root545Tail10
def root545Tail8 : List Nat := 6653 :: root545Tail9
def root545Tail7 : List Nat := 6263 :: root545Tail8
def root545Tail6 : List Nat := 6011 :: root545Tail7
def root545Tail5 : List Nat := 5801 :: root545Tail6
def root545Tail4 : List Nat := 5393 :: root545Tail5
def root545Tail3 : List Nat := 5171 :: root545Tail4
def root545Tail2 : List Nat := 4889 :: root545Tail3
def root545Tail1 : List Nat := 4679 :: root545Tail2
def root545Tail0 : List Nat := 4073 :: root545Tail1
def root545Candidates0 : List Nat := [5393, 8861, 17627]
set_option maxRecDepth 8192 in
theorem root545Candidates0_checked : candidates edge 4 3947 4073 root545Tail1 winner.bound = root545Candidates0 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root545Child0 : search edge 3 [3947, 4073] root545Candidates0 8020 winner = winner := by decide
theorem root545Step0 : scan edge (search edge 3) 4 [3947] 3947 root545Tail0 winner =
    scan edge (search edge 3) 4 [3947] 3947 root545Tail1 winner := by
  rw [root545Tail0, scan, if_neg (by decide), root545Candidates0_checked]
  change scan edge (search edge 3) 4 [3947] 3947 root545Tail1
    (if 3 ≤ root545Candidates0.length then search edge 3 [3947, 4073] root545Candidates0 8020 winner else winner) = _
  rw [if_pos (by decide), root545Child0]
def root545Candidates1 : List Nat := [4889, 5393, 6653]
set_option maxRecDepth 8192 in
theorem root545Candidates1_checked : candidates edge 4 3947 4679 root545Tail2 winner.bound = root545Candidates1 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root545Child1 : search edge 3 [3947, 4679] root545Candidates1 8626 winner = winner := by decide
theorem root545Step1 : scan edge (search edge 3) 4 [3947] 3947 root545Tail1 winner =
    scan edge (search edge 3) 4 [3947] 3947 root545Tail2 winner := by
  rw [root545Tail1, scan, if_neg (by decide), root545Candidates1_checked]
  change scan edge (search edge 3) 4 [3947] 3947 root545Tail2
    (if 3 ≤ root545Candidates1.length then search edge 3 [3947, 4679] root545Candidates1 8626 winner else winner) = _
  rw [if_pos (by decide), root545Child1]
def root545Candidates2 : List Nat := [15137]
set_option maxRecDepth 8192 in
theorem root545Candidates2_checked : candidates edge 4 3947 4889 root545Tail3 winner.bound = root545Candidates2 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root545Step2 : scan edge (search edge 3) 4 [3947] 3947 root545Tail2 winner =
    scan edge (search edge 3) 4 [3947] 3947 root545Tail3 winner := by
  rw [root545Tail2, scan, if_neg (by decide), root545Candidates2_checked]
  change scan edge (search edge 3) 4 [3947] 3947 root545Tail3
    (if 3 ≤ root545Candidates2.length then search edge 3 [3947, 4889] root545Candidates2 8836 winner else winner) = _
  rw [if_neg (by decide)]
def root545Candidates3 : List Nat := [5393, 8837]
set_option maxRecDepth 8192 in
theorem root545Candidates3_checked : candidates edge 4 3947 5171 root545Tail4 winner.bound = root545Candidates3 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root545Step3 : scan edge (search edge 3) 4 [3947] 3947 root545Tail3 winner =
    scan edge (search edge 3) 4 [3947] 3947 root545Tail4 winner := by
  rw [root545Tail3, scan, if_neg (by decide), root545Candidates3_checked]
  change scan edge (search edge 3) 4 [3947] 3947 root545Tail4
    (if 3 ≤ root545Candidates3.length then search edge 3 [3947, 5171] root545Candidates3 9118 winner else winner) = _
  rw [if_neg (by decide)]
def root545Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root545Candidates4_checked : candidates edge 4 3947 5393 root545Tail5 winner.bound = root545Candidates4 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root545Step4 : scan edge (search edge 3) 4 [3947] 3947 root545Tail4 winner =
    scan edge (search edge 3) 4 [3947] 3947 root545Tail5 winner := by
  rw [root545Tail4, scan, if_neg (by decide), root545Candidates4_checked]
  change scan edge (search edge 3) 4 [3947] 3947 root545Tail5
    (if 3 ≤ root545Candidates4.length then search edge 3 [3947, 5393] root545Candidates4 9340 winner else winner) = _
  rw [if_neg (by decide)]
theorem root545Scan5 : scan edge (search edge 3) 4 [3947] 3947 root545Tail5 winner = winner := by rw [root545Tail5, scan, if_pos (by decide)]
theorem root545Scan4 : scan edge (search edge 3) 4 [3947] 3947 root545Tail4 winner = winner :=
  (root545Step4).trans root545Scan5
theorem root545Scan3 : scan edge (search edge 3) 4 [3947] 3947 root545Tail3 winner = winner :=
  (root545Step3).trans root545Scan4
theorem root545Scan2 : scan edge (search edge 3) 4 [3947] 3947 root545Tail2 winner = winner :=
  (root545Step2).trans root545Scan3
theorem root545Scan1 : scan edge (search edge 3) 4 [3947] 3947 root545Tail1 winner = winner :=
  (root545Step1).trans root545Scan2
theorem root545Scan0 : scan edge (search edge 3) 4 [3947] 3947 root545Tail0 winner = winner :=
  (root545Step0).trans root545Scan1
set_option maxRecDepth 4096 in
theorem root545Branch : search edge 4 [3947] root545Candidates 3947 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3947] 3947 root545Tail0 winner = winner
  exact root545Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
