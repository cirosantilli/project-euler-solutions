import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root531Candidates : List Nat := Data.row3851
set_option maxRecDepth 8192 in
theorem root531Candidates_checked : candidates edge 5 0 3851 tail532 winner.bound = root531Candidates := by
  apply candidates_eq_row edge 5 0 3851 tail532 root531Candidates winner.bound tail532_ordered
  · decide
  · decide
  · funext q
    rfl
def root531Tail40 : List Nat := []
def root531Tail39 : List Nat := 21557 :: root531Tail40
def root531Tail38 : List Nat := 20051 :: root531Tail39
def root531Tail37 : List Nat := 19991 :: root531Tail38
def root531Tail36 : List Nat := 19463 :: root531Tail37
def root531Tail35 : List Nat := 18911 :: root531Tail36
def root531Tail34 : List Nat := 18593 :: root531Tail35
def root531Tail33 : List Nat := 17921 :: root531Tail34
def root531Tail32 : List Nat := 17891 :: root531Tail33
def root531Tail31 : List Nat := 17657 :: root531Tail32
def root531Tail30 : List Nat := 16811 :: root531Tail31
def root531Tail29 : List Nat := 16703 :: root531Tail30
def root531Tail28 : List Nat := 16661 :: root531Tail29
def root531Tail27 : List Nat := 15233 :: root531Tail28
def root531Tail26 : List Nat := 13877 :: root531Tail27
def root531Tail25 : List Nat := 13037 :: root531Tail26
def root531Tail24 : List Nat := 12821 :: root531Tail25
def root531Tail23 : List Nat := 12743 :: root531Tail24
def root531Tail22 : List Nat := 12041 :: root531Tail23
def root531Tail21 : List Nat := 11981 :: root531Tail22
def root531Tail20 : List Nat := 11717 :: root531Tail21
def root531Tail19 : List Nat := 10181 :: root531Tail20
def root531Tail18 : List Nat := 10133 :: root531Tail19
def root531Tail17 : List Nat := 9839 :: root531Tail18
def root531Tail16 : List Nat := 9521 :: root531Tail17
def root531Tail15 : List Nat := 9491 :: root531Tail16
def root531Tail14 : List Nat := 8663 :: root531Tail15
def root531Tail13 : List Nat := 7673 :: root531Tail14
def root531Tail12 : List Nat := 7547 :: root531Tail13
def root531Tail11 : List Nat := 7151 :: root531Tail12
def root531Tail10 : List Nat := 7043 :: root531Tail11
def root531Tail9 : List Nat := 6983 :: root531Tail10
def root531Tail8 : List Nat := 6899 :: root531Tail9
def root531Tail7 : List Nat := 6659 :: root531Tail8
def root531Tail6 : List Nat := 6521 :: root531Tail7
def root531Tail5 : List Nat := 5903 :: root531Tail6
def root531Tail4 : List Nat := 5651 :: root531Tail5
def root531Tail3 : List Nat := 4649 :: root531Tail4
def root531Tail2 : List Nat := 4271 :: root531Tail3
def root531Tail1 : List Nat := 4139 :: root531Tail2
def root531Tail0 : List Nat := 4013 :: root531Tail1
def root531Candidates0 : List Nat := [6659]
set_option maxRecDepth 8192 in
theorem root531Candidates0_checked : candidates edge 4 3851 4013 root531Tail1 winner.bound = root531Candidates0 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root531Step0 : scan edge (search edge 3) 4 [3851] 3851 root531Tail0 winner =
    scan edge (search edge 3) 4 [3851] 3851 root531Tail1 winner := by
  rw [root531Tail0, scan, if_neg (by decide), root531Candidates0_checked]
  change scan edge (search edge 3) 4 [3851] 3851 root531Tail1
    (if 3 ≤ root531Candidates0.length then search edge 3 [3851, 4013] root531Candidates0 7864 winner else winner) = _
  rw [if_neg (by decide)]
def root531Candidates1 : List Nat := [7673, 15233]
set_option maxRecDepth 8192 in
theorem root531Candidates1_checked : candidates edge 4 3851 4139 root531Tail2 winner.bound = root531Candidates1 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root531Step1 : scan edge (search edge 3) 4 [3851] 3851 root531Tail1 winner =
    scan edge (search edge 3) 4 [3851] 3851 root531Tail2 winner := by
  rw [root531Tail1, scan, if_neg (by decide), root531Candidates1_checked]
  change scan edge (search edge 3) 4 [3851] 3851 root531Tail2
    (if 3 ≤ root531Candidates1.length then search edge 3 [3851, 4139] root531Candidates1 7990 winner else winner) = _
  rw [if_neg (by decide)]
def root531Candidates2 : List Nat := [9491]
set_option maxRecDepth 8192 in
theorem root531Candidates2_checked : candidates edge 4 3851 4271 root531Tail3 winner.bound = root531Candidates2 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root531Step2 : scan edge (search edge 3) 4 [3851] 3851 root531Tail2 winner =
    scan edge (search edge 3) 4 [3851] 3851 root531Tail3 winner := by
  rw [root531Tail2, scan, if_neg (by decide), root531Candidates2_checked]
  change scan edge (search edge 3) 4 [3851] 3851 root531Tail3
    (if 3 ≤ root531Candidates2.length then search edge 3 [3851, 4271] root531Candidates2 8122 winner else winner) = _
  rw [if_neg (by decide)]
def root531Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root531Candidates3_checked : candidates edge 4 3851 4649 root531Tail4 winner.bound = root531Candidates3 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root531Step3 : scan edge (search edge 3) 4 [3851] 3851 root531Tail3 winner =
    scan edge (search edge 3) 4 [3851] 3851 root531Tail4 winner := by
  rw [root531Tail3, scan, if_neg (by decide), root531Candidates3_checked]
  change scan edge (search edge 3) 4 [3851] 3851 root531Tail4
    (if 3 ≤ root531Candidates3.length then search edge 3 [3851, 4649] root531Candidates3 8500 winner else winner) = _
  rw [if_neg (by decide)]
theorem root531Scan4 : scan edge (search edge 3) 4 [3851] 3851 root531Tail4 winner = winner := by rw [root531Tail4, scan, if_pos (by decide)]
theorem root531Scan3 : scan edge (search edge 3) 4 [3851] 3851 root531Tail3 winner = winner :=
  (root531Step3).trans root531Scan4
theorem root531Scan2 : scan edge (search edge 3) 4 [3851] 3851 root531Tail2 winner = winner :=
  (root531Step2).trans root531Scan3
theorem root531Scan1 : scan edge (search edge 3) 4 [3851] 3851 root531Tail1 winner = winner :=
  (root531Step1).trans root531Scan2
theorem root531Scan0 : scan edge (search edge 3) 4 [3851] 3851 root531Tail0 winner = winner :=
  (root531Step0).trans root531Scan1
set_option maxRecDepth 4096 in
theorem root531Branch : search edge 4 [3851] root531Candidates 3851 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3851] 3851 root531Tail0 winner = winner
  exact root531Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
