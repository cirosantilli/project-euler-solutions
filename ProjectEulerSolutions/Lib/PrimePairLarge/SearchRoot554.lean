import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root554Candidates : List Nat := Data.row4027
set_option maxRecDepth 8192 in
theorem root554Candidates_checked : candidates edge 5 0 4027 tail555 winner.bound = root554Candidates := by
  apply candidates_eq_row edge 5 0 4027 tail555 root554Candidates winner.bound tail555_ordered
  · decide
  · decide
  · funext q
    rfl
def root554Tail35 : List Nat := []
def root554Tail34 : List Nat := 20389 :: root554Tail35
def root554Tail33 : List Nat := 19717 :: root554Tail34
def root554Tail32 : List Nat := 18793 :: root554Tail33
def root554Tail31 : List Nat := 18181 :: root554Tail32
def root554Tail30 : List Nat := 16903 :: root554Tail31
def root554Tail29 : List Nat := 16879 :: root554Tail30
def root554Tail28 : List Nat := 16759 :: root554Tail29
def root554Tail27 : List Nat := 15727 :: root554Tail28
def root554Tail26 : List Nat := 15541 :: root554Tail27
def root554Tail25 : List Nat := 15427 :: root554Tail26
def root554Tail24 : List Nat := 14929 :: root554Tail25
def root554Tail23 : List Nat := 14461 :: root554Tail24
def root554Tail22 : List Nat := 14419 :: root554Tail23
def root554Tail21 : List Nat := 14083 :: root554Tail22
def root554Tail20 : List Nat := 13789 :: root554Tail21
def root554Tail19 : List Nat := 13669 :: root554Tail20
def root554Tail18 : List Nat := 13537 :: root554Tail19
def root554Tail17 : List Nat := 12577 :: root554Tail18
def root554Tail16 : List Nat := 12109 :: root554Tail17
def root554Tail15 : List Nat := 11443 :: root554Tail16
def root554Tail14 : List Nat := 9277 :: root554Tail15
def root554Tail13 : List Nat := 9181 :: root554Tail14
def root554Tail12 : List Nat := 9157 :: root554Tail13
def root554Tail11 : List Nat := 8863 :: root554Tail12
def root554Tail10 : List Nat := 8761 :: root554Tail11
def root554Tail9 : List Nat := 8677 :: root554Tail10
def root554Tail8 : List Nat := 7417 :: root554Tail9
def root554Tail7 : List Nat := 6949 :: root554Tail8
def root554Tail6 : List Nat := 6871 :: root554Tail7
def root554Tail5 : List Nat := 6199 :: root554Tail6
def root554Tail4 : List Nat := 5503 :: root554Tail5
def root554Tail3 : List Nat := 5413 :: root554Tail4
def root554Tail2 : List Nat := 5233 :: root554Tail3
def root554Tail1 : List Nat := 4933 :: root554Tail2
def root554Tail0 : List Nat := 4783 :: root554Tail1
def root554Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root554Candidates0_checked : candidates edge 4 4027 4783 root554Tail1 winner.bound = root554Candidates0 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root554Step0 : scan edge (search edge 3) 4 [4027] 4027 root554Tail0 winner =
    scan edge (search edge 3) 4 [4027] 4027 root554Tail1 winner := by
  rw [root554Tail0, scan, if_neg (by decide), root554Candidates0_checked]
  change scan edge (search edge 3) 4 [4027] 4027 root554Tail1
    (if 3 ≤ root554Candidates0.length then search edge 3 [4027, 4783] root554Candidates0 8810 winner else winner) = _
  rw [if_neg (by decide)]
def root554Candidates1 : List Nat := [12109, 13669, 15727]
set_option maxRecDepth 8192 in
theorem root554Candidates1_checked : candidates edge 4 4027 4933 root554Tail2 winner.bound = root554Candidates1 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root554Child1 : search edge 3 [4027, 4933] root554Candidates1 8960 winner = winner := by decide
theorem root554Step1 : scan edge (search edge 3) 4 [4027] 4027 root554Tail1 winner =
    scan edge (search edge 3) 4 [4027] 4027 root554Tail2 winner := by
  rw [root554Tail1, scan, if_neg (by decide), root554Candidates1_checked]
  change scan edge (search edge 3) 4 [4027] 4027 root554Tail2
    (if 3 ≤ root554Candidates1.length then search edge 3 [4027, 4933] root554Candidates1 8960 winner else winner) = _
  rw [if_pos (by decide), root554Child1]
def root554Candidates2 : List Nat := [8677]
set_option maxRecDepth 8192 in
theorem root554Candidates2_checked : candidates edge 4 4027 5233 root554Tail3 winner.bound = root554Candidates2 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root554Step2 : scan edge (search edge 3) 4 [4027] 4027 root554Tail2 winner =
    scan edge (search edge 3) 4 [4027] 4027 root554Tail3 winner := by
  rw [root554Tail2, scan, if_neg (by decide), root554Candidates2_checked]
  change scan edge (search edge 3) 4 [4027] 4027 root554Tail3
    (if 3 ≤ root554Candidates2.length then search edge 3 [4027, 5233] root554Candidates2 9260 winner else winner) = _
  rw [if_neg (by decide)]
def root554Candidates3 : List Nat := [13789]
set_option maxRecDepth 8192 in
theorem root554Candidates3_checked : candidates edge 4 4027 5413 root554Tail4 winner.bound = root554Candidates3 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root554Step3 : scan edge (search edge 3) 4 [4027] 4027 root554Tail3 winner =
    scan edge (search edge 3) 4 [4027] 4027 root554Tail4 winner := by
  rw [root554Tail3, scan, if_neg (by decide), root554Candidates3_checked]
  change scan edge (search edge 3) 4 [4027] 4027 root554Tail4
    (if 3 ≤ root554Candidates3.length then search edge 3 [4027, 5413] root554Candidates3 9440 winner else winner) = _
  rw [if_neg (by decide)]
theorem root554Scan4 : scan edge (search edge 3) 4 [4027] 4027 root554Tail4 winner = winner := by rw [root554Tail4, scan, if_pos (by decide)]
theorem root554Scan3 : scan edge (search edge 3) 4 [4027] 4027 root554Tail3 winner = winner :=
  (root554Step3).trans root554Scan4
theorem root554Scan2 : scan edge (search edge 3) 4 [4027] 4027 root554Tail2 winner = winner :=
  (root554Step2).trans root554Scan3
theorem root554Scan1 : scan edge (search edge 3) 4 [4027] 4027 root554Tail1 winner = winner :=
  (root554Step1).trans root554Scan2
theorem root554Scan0 : scan edge (search edge 3) 4 [4027] 4027 root554Tail0 winner = winner :=
  (root554Step0).trans root554Scan1
set_option maxRecDepth 4096 in
theorem root554Branch : search edge 4 [4027] root554Candidates 4027 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4027] 4027 root554Tail0 winner = winner
  exact root554Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
