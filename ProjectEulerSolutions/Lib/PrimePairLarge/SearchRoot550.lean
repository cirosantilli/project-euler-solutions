import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root550Candidates : List Nat := Data.row4007
set_option maxRecDepth 8192 in
theorem root550Candidates_checked : candidates edge 5 0 4007 tail551 winner.bound = root550Candidates := by
  apply candidates_eq_row edge 5 0 4007 tail551 root550Candidates winner.bound tail551_ordered
  · decide
  · decide
  · funext q
    rfl
def root550Tail34 : List Nat := []
def root550Tail33 : List Nat := 21767 :: root550Tail34
def root550Tail32 : List Nat := 21599 :: root550Tail33
def root550Tail31 : List Nat := 20759 :: root550Tail32
def root550Tail30 : List Nat := 19997 :: root550Tail31
def root550Tail29 : List Nat := 19211 :: root550Tail30
def root550Tail28 : List Nat := 18233 :: root550Tail29
def root550Tail27 : List Nat := 17909 :: root550Tail28
def root550Tail26 : List Nat := 17579 :: root550Tail27
def root550Tail25 : List Nat := 17519 :: root550Tail26
def root550Tail24 : List Nat := 17117 :: root550Tail25
def root550Tail23 : List Nat := 16553 :: root550Tail24
def root550Tail22 : List Nat := 16091 :: root550Tail23
def root550Tail21 : List Nat := 15173 :: root550Tail22
def root550Tail20 : List Nat := 13487 :: root550Tail21
def root550Tail19 : List Nat := 12149 :: root550Tail20
def root550Tail18 : List Nat := 12113 :: root550Tail19
def root550Tail17 : List Nat := 11321 :: root550Tail18
def root550Tail16 : List Nat := 10883 :: root550Tail17
def root550Tail15 : List Nat := 8693 :: root550Tail16
def root550Tail14 : List Nat := 8369 :: root550Tail15
def root550Tail13 : List Nat := 8243 :: root550Tail14
def root550Tail12 : List Nat := 8081 :: root550Tail13
def root550Tail11 : List Nat := 7907 :: root550Tail12
def root550Tail10 : List Nat := 7853 :: root550Tail11
def root550Tail9 : List Nat := 7283 :: root550Tail10
def root550Tail8 : List Nat := 6353 :: root550Tail9
def root550Tail7 : List Nat := 6053 :: root550Tail8
def root550Tail6 : List Nat := 6011 :: root550Tail7
def root550Tail5 : List Nat := 5639 :: root550Tail6
def root550Tail4 : List Nat := 5417 :: root550Tail5
def root550Tail3 : List Nat := 4751 :: root550Tail4
def root550Tail2 : List Nat := 4337 :: root550Tail3
def root550Tail1 : List Nat := 4217 :: root550Tail2
def root550Tail0 : List Nat := 4091 :: root550Tail1
def root550Candidates0 : List Nat := [11321]
set_option maxRecDepth 8192 in
theorem root550Candidates0_checked : candidates edge 4 4007 4091 root550Tail1 winner.bound = root550Candidates0 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root550Step0 : scan edge (search edge 3) 4 [4007] 4007 root550Tail0 winner =
    scan edge (search edge 3) 4 [4007] 4007 root550Tail1 winner := by
  rw [root550Tail0, scan, if_neg (by decide), root550Candidates0_checked]
  change scan edge (search edge 3) 4 [4007] 4007 root550Tail1
    (if 3 ≤ root550Candidates0.length then search edge 3 [4007, 4091] root550Candidates0 8098 winner else winner) = _
  rw [if_neg (by decide)]
def root550Candidates1 : List Nat := [4337, 4751]
set_option maxRecDepth 8192 in
theorem root550Candidates1_checked : candidates edge 4 4007 4217 root550Tail2 winner.bound = root550Candidates1 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root550Step1 : scan edge (search edge 3) 4 [4007] 4007 root550Tail1 winner =
    scan edge (search edge 3) 4 [4007] 4007 root550Tail2 winner := by
  rw [root550Tail1, scan, if_neg (by decide), root550Candidates1_checked]
  change scan edge (search edge 3) 4 [4007] 4007 root550Tail2
    (if 3 ≤ root550Candidates1.length then search edge 3 [4007, 4217] root550Candidates1 8224 winner else winner) = _
  rw [if_neg (by decide)]
def root550Candidates2 : List Nat := [11321]
set_option maxRecDepth 8192 in
theorem root550Candidates2_checked : candidates edge 4 4007 4337 root550Tail3 winner.bound = root550Candidates2 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root550Step2 : scan edge (search edge 3) 4 [4007] 4007 root550Tail2 winner =
    scan edge (search edge 3) 4 [4007] 4007 root550Tail3 winner := by
  rw [root550Tail2, scan, if_neg (by decide), root550Candidates2_checked]
  change scan edge (search edge 3) 4 [4007] 4007 root550Tail3
    (if 3 ≤ root550Candidates2.length then search edge 3 [4007, 4337] root550Candidates2 8344 winner else winner) = _
  rw [if_neg (by decide)]
def root550Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root550Candidates3_checked : candidates edge 4 4007 4751 root550Tail4 winner.bound = root550Candidates3 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root550Step3 : scan edge (search edge 3) 4 [4007] 4007 root550Tail3 winner =
    scan edge (search edge 3) 4 [4007] 4007 root550Tail4 winner := by
  rw [root550Tail3, scan, if_neg (by decide), root550Candidates3_checked]
  change scan edge (search edge 3) 4 [4007] 4007 root550Tail4
    (if 3 ≤ root550Candidates3.length then search edge 3 [4007, 4751] root550Candidates3 8758 winner else winner) = _
  rw [if_neg (by decide)]
def root550Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root550Candidates4_checked : candidates edge 4 4007 5417 root550Tail5 winner.bound = root550Candidates4 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root550Step4 : scan edge (search edge 3) 4 [4007] 4007 root550Tail4 winner =
    scan edge (search edge 3) 4 [4007] 4007 root550Tail5 winner := by
  rw [root550Tail4, scan, if_neg (by decide), root550Candidates4_checked]
  change scan edge (search edge 3) 4 [4007] 4007 root550Tail5
    (if 3 ≤ root550Candidates4.length then search edge 3 [4007, 5417] root550Candidates4 9424 winner else winner) = _
  rw [if_neg (by decide)]
theorem root550Scan5 : scan edge (search edge 3) 4 [4007] 4007 root550Tail5 winner = winner := by rw [root550Tail5, scan, if_pos (by decide)]
theorem root550Scan4 : scan edge (search edge 3) 4 [4007] 4007 root550Tail4 winner = winner :=
  (root550Step4).trans root550Scan5
theorem root550Scan3 : scan edge (search edge 3) 4 [4007] 4007 root550Tail3 winner = winner :=
  (root550Step3).trans root550Scan4
theorem root550Scan2 : scan edge (search edge 3) 4 [4007] 4007 root550Tail2 winner = winner :=
  (root550Step2).trans root550Scan3
theorem root550Scan1 : scan edge (search edge 3) 4 [4007] 4007 root550Tail1 winner = winner :=
  (root550Step1).trans root550Scan2
theorem root550Scan0 : scan edge (search edge 3) 4 [4007] 4007 root550Tail0 winner = winner :=
  (root550Step0).trans root550Scan1
set_option maxRecDepth 4096 in
theorem root550Branch : search edge 4 [4007] root550Candidates 4007 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4007] 4007 root550Tail0 winner = winner
  exact root550Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
