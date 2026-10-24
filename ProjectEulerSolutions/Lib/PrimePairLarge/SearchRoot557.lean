import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root557Candidates : List Nat := Data.row4057
set_option maxRecDepth 8192 in
theorem root557Candidates_checked : candidates edge 5 0 4057 tail558 winner.bound = root557Candidates := by
  apply candidates_eq_row edge 5 0 4057 tail558 root557Candidates winner.bound tail558_ordered
  · decide
  · decide
  · funext q
    rfl
def root557Tail39 : List Nat := []
def root557Tail38 : List Nat := 21751 :: root557Tail39
def root557Tail37 : List Nat := 21523 :: root557Tail38
def root557Tail36 : List Nat := 21313 :: root557Tail37
def root557Tail35 : List Nat := 21163 :: root557Tail36
def root557Tail34 : List Nat := 20857 :: root557Tail35
def root557Tail33 : List Nat := 20743 :: root557Tail34
def root557Tail32 : List Nat := 20611 :: root557Tail33
def root557Tail31 : List Nat := 20533 :: root557Tail32
def root557Tail30 : List Nat := 19927 :: root557Tail31
def root557Tail29 : List Nat := 19759 :: root557Tail30
def root557Tail28 : List Nat := 18541 :: root557Tail29
def root557Tail27 : List Nat := 17881 :: root557Tail28
def root557Tail26 : List Nat := 17497 :: root557Tail27
def root557Tail25 : List Nat := 15733 :: root557Tail26
def root557Tail24 : List Nat := 14737 :: root557Tail25
def root557Tail23 : List Nat := 14653 :: root557Tail24
def root557Tail22 : List Nat := 14347 :: root557Tail23
def root557Tail21 : List Nat := 14107 :: root557Tail22
def root557Tail20 : List Nat := 13597 :: root557Tail21
def root557Tail19 : List Nat := 12919 :: root557Tail20
def root557Tail18 : List Nat := 12889 :: root557Tail19
def root557Tail17 : List Nat := 12583 :: root557Tail18
def root557Tail16 : List Nat := 11083 :: root557Tail17
def root557Tail15 : List Nat := 9829 :: root557Tail16
def root557Tail14 : List Nat := 9109 :: root557Tail15
def root557Tail13 : List Nat := 9103 :: root557Tail14
def root557Tail12 : List Nat := 8821 :: root557Tail13
def root557Tail11 : List Nat := 7879 :: root557Tail12
def root557Tail10 : List Nat := 7753 :: root557Tail11
def root557Tail9 : List Nat := 7393 :: root557Tail10
def root557Tail8 : List Nat := 7213 :: root557Tail9
def root557Tail7 : List Nat := 6469 :: root557Tail8
def root557Tail6 : List Nat := 6421 :: root557Tail7
def root557Tail5 : List Nat := 6199 :: root557Tail6
def root557Tail4 : List Nat := 6007 :: root557Tail5
def root557Tail3 : List Nat := 5419 :: root557Tail4
def root557Tail2 : List Nat := 5077 :: root557Tail3
def root557Tail1 : List Nat := 4663 :: root557Tail2
def root557Tail0 : List Nat := 4483 :: root557Tail1
def root557Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root557Candidates0_checked : candidates edge 4 4057 4483 root557Tail1 winner.bound = root557Candidates0 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root557Step0 : scan edge (search edge 3) 4 [4057] 4057 root557Tail0 winner =
    scan edge (search edge 3) 4 [4057] 4057 root557Tail1 winner := by
  rw [root557Tail0, scan, if_neg (by decide), root557Candidates0_checked]
  change scan edge (search edge 3) 4 [4057] 4057 root557Tail1
    (if 3 ≤ root557Candidates0.length then search edge 3 [4057, 4483] root557Candidates0 8540 winner else winner) = _
  rw [if_neg (by decide)]
def root557Candidates1 : List Nat := [7753, 7879]
set_option maxRecDepth 8192 in
theorem root557Candidates1_checked : candidates edge 4 4057 4663 root557Tail2 winner.bound = root557Candidates1 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root557Step1 : scan edge (search edge 3) 4 [4057] 4057 root557Tail1 winner =
    scan edge (search edge 3) 4 [4057] 4057 root557Tail2 winner := by
  rw [root557Tail1, scan, if_neg (by decide), root557Candidates1_checked]
  change scan edge (search edge 3) 4 [4057] 4057 root557Tail2
    (if 3 ≤ root557Candidates1.length then search edge 3 [4057, 4663] root557Candidates1 8720 winner else winner) = _
  rw [if_neg (by decide)]
def root557Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root557Candidates2_checked : candidates edge 4 4057 5077 root557Tail3 winner.bound = root557Candidates2 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root557Step2 : scan edge (search edge 3) 4 [4057] 4057 root557Tail2 winner =
    scan edge (search edge 3) 4 [4057] 4057 root557Tail3 winner := by
  rw [root557Tail2, scan, if_neg (by decide), root557Candidates2_checked]
  change scan edge (search edge 3) 4 [4057] 4057 root557Tail3
    (if 3 ≤ root557Candidates2.length then search edge 3 [4057, 5077] root557Candidates2 9134 winner else winner) = _
  rw [if_neg (by decide)]
def root557Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root557Candidates3_checked : candidates edge 4 4057 5419 root557Tail4 winner.bound = root557Candidates3 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root557Step3 : scan edge (search edge 3) 4 [4057] 4057 root557Tail3 winner =
    scan edge (search edge 3) 4 [4057] 4057 root557Tail4 winner := by
  rw [root557Tail3, scan, if_neg (by decide), root557Candidates3_checked]
  change scan edge (search edge 3) 4 [4057] 4057 root557Tail4
    (if 3 ≤ root557Candidates3.length then search edge 3 [4057, 5419] root557Candidates3 9476 winner else winner) = _
  rw [if_neg (by decide)]
theorem root557Scan4 : scan edge (search edge 3) 4 [4057] 4057 root557Tail4 winner = winner := by rw [root557Tail4, scan, if_pos (by decide)]
theorem root557Scan3 : scan edge (search edge 3) 4 [4057] 4057 root557Tail3 winner = winner :=
  (root557Step3).trans root557Scan4
theorem root557Scan2 : scan edge (search edge 3) 4 [4057] 4057 root557Tail2 winner = winner :=
  (root557Step2).trans root557Scan3
theorem root557Scan1 : scan edge (search edge 3) 4 [4057] 4057 root557Tail1 winner = winner :=
  (root557Step1).trans root557Scan2
theorem root557Scan0 : scan edge (search edge 3) 4 [4057] 4057 root557Tail0 winner = winner :=
  (root557Step0).trans root557Scan1
set_option maxRecDepth 4096 in
theorem root557Branch : search edge 4 [4057] root557Candidates 4057 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4057] 4057 root557Tail0 winner = winner
  exact root557Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
