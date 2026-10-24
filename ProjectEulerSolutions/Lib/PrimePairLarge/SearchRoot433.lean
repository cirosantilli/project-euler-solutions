import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root433Candidates : List Nat := Data.row3041
set_option maxRecDepth 8192 in
theorem root433Candidates_checked : candidates edge 5 0 3041 tail434 winner.bound = root433Candidates := by
  apply candidates_eq_row edge 5 0 3041 tail434 root433Candidates winner.bound tail434_ordered
  · decide
  · decide
  · funext q
    rfl
def root433Tail34 : List Nat := []
def root433Tail33 : List Nat := 21821 :: root433Tail34
def root433Tail32 : List Nat := 20759 :: root433Tail33
def root433Tail31 : List Nat := 20249 :: root433Tail32
def root433Tail30 : List Nat := 18671 :: root433Tail31
def root433Tail29 : List Nat := 17909 :: root433Tail30
def root433Tail28 : List Nat := 17729 :: root433Tail29
def root433Tail27 : List Nat := 16763 :: root433Tail28
def root433Tail26 : List Nat := 16487 :: root433Tail27
def root433Tail25 : List Nat := 16481 :: root433Tail26
def root433Tail24 : List Nat := 15761 :: root433Tail25
def root433Tail23 : List Nat := 15671 :: root433Tail24
def root433Tail22 : List Nat := 15473 :: root433Tail23
def root433Tail21 : List Nat := 14843 :: root433Tail22
def root433Tail20 : List Nat := 13001 :: root433Tail21
def root433Tail19 : List Nat := 12659 :: root433Tail20
def root433Tail18 : List Nat := 12323 :: root433Tail19
def root433Tail17 : List Nat := 11987 :: root433Tail18
def root433Tail16 : List Nat := 11903 :: root433Tail17
def root433Tail15 : List Nat := 10601 :: root433Tail16
def root433Tail14 : List Nat := 10133 :: root433Tail15
def root433Tail13 : List Nat := 9839 :: root433Tail14
def root433Tail12 : List Nat := 8963 :: root433Tail13
def root433Tail11 : List Nat := 8273 :: root433Tail12
def root433Tail10 : List Nat := 7529 :: root433Tail11
def root433Tail9 : List Nat := 7481 :: root433Tail10
def root433Tail8 : List Nat := 6719 :: root433Tail9
def root433Tail7 : List Nat := 6317 :: root433Tail8
def root433Tail6 : List Nat := 5897 :: root433Tail7
def root433Tail5 : List Nat := 4409 :: root433Tail6
def root433Tail4 : List Nat := 4337 :: root433Tail5
def root433Tail3 : List Nat := 4133 :: root433Tail4
def root433Tail2 : List Nat := 3533 :: root433Tail3
def root433Tail1 : List Nat := 3491 :: root433Tail2
def root433Tail0 : List Nat := 3209 :: root433Tail1
def root433Candidates0 : List Nat := [16487]
set_option maxRecDepth 8192 in
theorem root433Candidates0_checked : candidates edge 4 3041 3209 root433Tail1 winner.bound = root433Candidates0 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root433Step0 : scan edge (search edge 3) 4 [3041] 3041 root433Tail0 winner =
    scan edge (search edge 3) 4 [3041] 3041 root433Tail1 winner := by
  rw [root433Tail0, scan, if_neg (by decide), root433Candidates0_checked]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail1
    (if 3 ≤ root433Candidates0.length then search edge 3 [3041, 3209] root433Candidates0 6250 winner else winner) = _
  rw [if_neg (by decide)]
def root433Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root433Candidates1_checked : candidates edge 4 3041 3491 root433Tail2 winner.bound = root433Candidates1 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root433Step1 : scan edge (search edge 3) 4 [3041] 3041 root433Tail1 winner =
    scan edge (search edge 3) 4 [3041] 3041 root433Tail2 winner := by
  rw [root433Tail1, scan, if_neg (by decide), root433Candidates1_checked]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail2
    (if 3 ≤ root433Candidates1.length then search edge 3 [3041, 3491] root433Candidates1 6532 winner else winner) = _
  rw [if_neg (by decide)]
def root433Candidates2 : List Nat := [4133, 8273]
set_option maxRecDepth 8192 in
theorem root433Candidates2_checked : candidates edge 4 3041 3533 root433Tail3 winner.bound = root433Candidates2 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root433Step2 : scan edge (search edge 3) 4 [3041] 3041 root433Tail2 winner =
    scan edge (search edge 3) 4 [3041] 3041 root433Tail3 winner := by
  rw [root433Tail2, scan, if_neg (by decide), root433Candidates2_checked]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail3
    (if 3 ≤ root433Candidates2.length then search edge 3 [3041, 3533] root433Candidates2 6574 winner else winner) = _
  rw [if_neg (by decide)]
def root433Candidates3 : List Nat := [4409, 13001]
set_option maxRecDepth 8192 in
theorem root433Candidates3_checked : candidates edge 4 3041 4133 root433Tail4 winner.bound = root433Candidates3 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root433Step3 : scan edge (search edge 3) 4 [3041] 3041 root433Tail3 winner =
    scan edge (search edge 3) 4 [3041] 3041 root433Tail4 winner := by
  rw [root433Tail3, scan, if_neg (by decide), root433Candidates3_checked]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail4
    (if 3 ≤ root433Candidates3.length then search edge 3 [3041, 4133] root433Candidates3 7174 winner else winner) = _
  rw [if_neg (by decide)]
def root433Candidates4 : List Nat := [11903, 15473]
set_option maxRecDepth 8192 in
theorem root433Candidates4_checked : candidates edge 4 3041 4337 root433Tail5 winner.bound = root433Candidates4 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root433Step4 : scan edge (search edge 3) 4 [3041] 3041 root433Tail4 winner =
    scan edge (search edge 3) 4 [3041] 3041 root433Tail5 winner := by
  rw [root433Tail4, scan, if_neg (by decide), root433Candidates4_checked]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail5
    (if 3 ≤ root433Candidates4.length then search edge 3 [3041, 4337] root433Candidates4 7378 winner else winner) = _
  rw [if_neg (by decide)]
def root433Candidates5 : List Nat := [6719, 11987]
set_option maxRecDepth 8192 in
theorem root433Candidates5_checked : candidates edge 4 3041 4409 root433Tail6 winner.bound = root433Candidates5 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root433Step5 : scan edge (search edge 3) 4 [3041] 3041 root433Tail5 winner =
    scan edge (search edge 3) 4 [3041] 3041 root433Tail6 winner := by
  rw [root433Tail5, scan, if_neg (by decide), root433Candidates5_checked]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail6
    (if 3 ≤ root433Candidates5.length then search edge 3 [3041, 4409] root433Candidates5 7450 winner else winner) = _
  rw [if_neg (by decide)]
theorem root433Scan6 : scan edge (search edge 3) 4 [3041] 3041 root433Tail6 winner = winner := by rw [root433Tail6, scan, if_pos (by decide)]
theorem root433Scan5 : scan edge (search edge 3) 4 [3041] 3041 root433Tail5 winner = winner :=
  (root433Step5).trans root433Scan6
theorem root433Scan4 : scan edge (search edge 3) 4 [3041] 3041 root433Tail4 winner = winner :=
  (root433Step4).trans root433Scan5
theorem root433Scan3 : scan edge (search edge 3) 4 [3041] 3041 root433Tail3 winner = winner :=
  (root433Step3).trans root433Scan4
theorem root433Scan2 : scan edge (search edge 3) 4 [3041] 3041 root433Tail2 winner = winner :=
  (root433Step2).trans root433Scan3
theorem root433Scan1 : scan edge (search edge 3) 4 [3041] 3041 root433Tail1 winner = winner :=
  (root433Step1).trans root433Scan2
theorem root433Scan0 : scan edge (search edge 3) 4 [3041] 3041 root433Tail0 winner = winner :=
  (root433Step0).trans root433Scan1
set_option maxRecDepth 4096 in
theorem root433Branch : search edge 4 [3041] root433Candidates 3041 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3041] 3041 root433Tail0 winner = winner
  exact root433Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
