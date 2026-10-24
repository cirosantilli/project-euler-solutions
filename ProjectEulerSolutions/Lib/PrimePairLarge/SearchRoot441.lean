import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root441Candidates : List Nat := Data.row3119
set_option maxRecDepth 8192 in
theorem root441Candidates_checked : candidates edge 5 0 3119 tail442 winner.bound = root441Candidates := by
  apply candidates_eq_row edge 5 0 3119 tail442 root441Candidates winner.bound tail442_ordered
  · decide
  · decide
  · funext q
    rfl
def root441Tail44 : List Nat := []
def root441Tail43 : List Nat := 22697 :: root441Tail44
def root441Tail42 : List Nat := 22367 :: root441Tail43
def root441Tail41 : List Nat := 22283 :: root441Tail42
def root441Tail40 : List Nat := 21683 :: root441Tail41
def root441Tail39 : List Nat := 21569 :: root441Tail40
def root441Tail38 : List Nat := 21341 :: root441Tail39
def root441Tail37 : List Nat := 21227 :: root441Tail38
def root441Tail36 : List Nat := 20051 :: root441Tail37
def root441Tail35 : List Nat := 19919 :: root441Tail36
def root441Tail34 : List Nat := 19373 :: root441Tail35
def root441Tail33 : List Nat := 17747 :: root441Tail34
def root441Tail32 : List Nat := 17609 :: root441Tail33
def root441Tail31 : List Nat := 16763 :: root441Tail32
def root441Tail30 : List Nat := 16433 :: root441Tail31
def root441Tail29 : List Nat := 16223 :: root441Tail30
def root441Tail28 : List Nat := 15467 :: root441Tail29
def root441Tail27 : List Nat := 14879 :: root441Tail28
def root441Tail26 : List Nat := 14087 :: root441Tail27
def root441Tail25 : List Nat := 13913 :: root441Tail26
def root441Tail24 : List Nat := 13907 :: root441Tail25
def root441Tail23 : List Nat := 13757 :: root441Tail24
def root441Tail22 : List Nat := 12647 :: root441Tail23
def root441Tail21 : List Nat := 11681 :: root441Tail22
def root441Tail20 : List Nat := 11471 :: root441Tail21
def root441Tail19 : List Nat := 10853 :: root441Tail20
def root441Tail18 : List Nat := 10601 :: root441Tail19
def root441Tail17 : List Nat := 10139 :: root441Tail18
def root441Tail16 : List Nat := 10133 :: root441Tail17
def root441Tail15 : List Nat := 9887 :: root441Tail16
def root441Tail14 : List Nat := 9227 :: root441Tail15
def root441Tail13 : List Nat := 8831 :: root441Tail14
def root441Tail12 : List Nat := 7109 :: root441Tail13
def root441Tail11 : List Nat := 6857 :: root441Tail12
def root441Tail10 : List Nat := 6491 :: root441Tail11
def root441Tail9 : List Nat := 6359 :: root441Tail10
def root441Tail8 : List Nat := 6311 :: root441Tail9
def root441Tail7 : List Nat := 6029 :: root441Tail8
def root441Tail6 : List Nat := 5783 :: root441Tail7
def root441Tail5 : List Nat := 5741 :: root441Tail6
def root441Tail4 : List Nat := 5237 :: root441Tail5
def root441Tail3 : List Nat := 5231 :: root441Tail4
def root441Tail2 : List Nat := 4679 :: root441Tail3
def root441Tail1 : List Nat := 4391 :: root441Tail2
def root441Tail0 : List Nat := 3677 :: root441Tail1
def root441Candidates0 : List Nat := [16763]
set_option maxRecDepth 8192 in
theorem root441Candidates0_checked : candidates edge 4 3119 3677 root441Tail1 winner.bound = root441Candidates0 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root441Step0 : scan edge (search edge 3) 4 [3119] 3119 root441Tail0 winner =
    scan edge (search edge 3) 4 [3119] 3119 root441Tail1 winner := by
  rw [root441Tail0, scan, if_neg (by decide), root441Candidates0_checked]
  change scan edge (search edge 3) 4 [3119] 3119 root441Tail1
    (if 3 ≤ root441Candidates0.length then search edge 3 [3119, 3677] root441Candidates0 6796 winner else winner) = _
  rw [if_neg (by decide)]
def root441Candidates1 : List Nat := [12647]
set_option maxRecDepth 8192 in
theorem root441Candidates1_checked : candidates edge 4 3119 4391 root441Tail2 winner.bound = root441Candidates1 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root441Step1 : scan edge (search edge 3) 4 [3119] 3119 root441Tail1 winner =
    scan edge (search edge 3) 4 [3119] 3119 root441Tail2 winner := by
  rw [root441Tail1, scan, if_neg (by decide), root441Candidates1_checked]
  change scan edge (search edge 3) 4 [3119] 3119 root441Tail2
    (if 3 ≤ root441Candidates1.length then search edge 3 [3119, 4391] root441Candidates1 7510 winner else winner) = _
  rw [if_neg (by decide)]
def root441Candidates2 : List Nat := [9887]
set_option maxRecDepth 8192 in
theorem root441Candidates2_checked : candidates edge 4 3119 4679 root441Tail3 winner.bound = root441Candidates2 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root441Step2 : scan edge (search edge 3) 4 [3119] 3119 root441Tail2 winner =
    scan edge (search edge 3) 4 [3119] 3119 root441Tail3 winner := by
  rw [root441Tail2, scan, if_neg (by decide), root441Candidates2_checked]
  change scan edge (search edge 3) 4 [3119] 3119 root441Tail3
    (if 3 ≤ root441Candidates2.length then search edge 3 [3119, 4679] root441Candidates2 7798 winner else winner) = _
  rw [if_neg (by decide)]
def root441Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root441Candidates3_checked : candidates edge 4 3119 5231 root441Tail4 winner.bound = root441Candidates3 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root441Step3 : scan edge (search edge 3) 4 [3119] 3119 root441Tail3 winner =
    scan edge (search edge 3) 4 [3119] 3119 root441Tail4 winner := by
  rw [root441Tail3, scan, if_neg (by decide), root441Candidates3_checked]
  change scan edge (search edge 3) 4 [3119] 3119 root441Tail4
    (if 3 ≤ root441Candidates3.length then search edge 3 [3119, 5231] root441Candidates3 8350 winner else winner) = _
  rw [if_neg (by decide)]
def root441Candidates4 : List Nat := [5783, 10133, 17609]
set_option maxRecDepth 8192 in
theorem root441Candidates4_checked : candidates edge 4 3119 5237 root441Tail5 winner.bound = root441Candidates4 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root441Child4 : search edge 3 [3119, 5237] root441Candidates4 8356 winner = winner := by decide
theorem root441Step4 : scan edge (search edge 3) 4 [3119] 3119 root441Tail4 winner =
    scan edge (search edge 3) 4 [3119] 3119 root441Tail5 winner := by
  rw [root441Tail4, scan, if_neg (by decide), root441Candidates4_checked]
  change scan edge (search edge 3) 4 [3119] 3119 root441Tail5
    (if 3 ≤ root441Candidates4.length then search edge 3 [3119, 5237] root441Candidates4 8356 winner else winner) = _
  rw [if_pos (by decide), root441Child4]
theorem root441Scan5 : scan edge (search edge 3) 4 [3119] 3119 root441Tail5 winner = winner := by rw [root441Tail5, scan, if_pos (by decide)]
theorem root441Scan4 : scan edge (search edge 3) 4 [3119] 3119 root441Tail4 winner = winner :=
  (root441Step4).trans root441Scan5
theorem root441Scan3 : scan edge (search edge 3) 4 [3119] 3119 root441Tail3 winner = winner :=
  (root441Step3).trans root441Scan4
theorem root441Scan2 : scan edge (search edge 3) 4 [3119] 3119 root441Tail2 winner = winner :=
  (root441Step2).trans root441Scan3
theorem root441Scan1 : scan edge (search edge 3) 4 [3119] 3119 root441Tail1 winner = winner :=
  (root441Step1).trans root441Scan2
theorem root441Scan0 : scan edge (search edge 3) 4 [3119] 3119 root441Tail0 winner = winner :=
  (root441Step0).trans root441Scan1
set_option maxRecDepth 4096 in
theorem root441Branch : search edge 4 [3119] root441Candidates 3119 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3119] 3119 root441Tail0 winner = winner
  exact root441Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
