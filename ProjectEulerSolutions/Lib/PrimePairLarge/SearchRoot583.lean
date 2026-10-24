import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root583Candidates : List Nat := Data.row4271
set_option maxRecDepth 8192 in
theorem root583Candidates_checked : candidates edge 5 0 4271 tail584 winner.bound = root583Candidates := by
  apply candidates_eq_row edge 5 0 4271 tail584 root583Candidates winner.bound tail584_ordered
  · decide
  · decide
  · funext q
    rfl
def root583Tail37 : List Nat := []
def root583Tail36 : List Nat := 21347 :: root583Tail37
def root583Tail35 : List Nat := 20663 :: root583Tail36
def root583Tail34 : List Nat := 20549 :: root583Tail35
def root583Tail33 : List Nat := 20063 :: root583Tail34
def root583Tail32 : List Nat := 19211 :: root583Tail33
def root583Tail31 : List Nat := 18701 :: root583Tail32
def root583Tail30 : List Nat := 18617 :: root583Tail31
def root583Tail29 : List Nat := 17189 :: root583Tail30
def root583Tail28 : List Nat := 17021 :: root583Tail29
def root583Tail27 : List Nat := 16883 :: root583Tail28
def root583Tail26 : List Nat := 16691 :: root583Tail27
def root583Tail25 : List Nat := 16451 :: root583Tail26
def root583Tail24 : List Nat := 16127 :: root583Tail25
def root583Tail23 : List Nat := 15443 :: root583Tail24
def root583Tail22 : List Nat := 15077 :: root583Tail23
def root583Tail21 : List Nat := 14591 :: root583Tail22
def root583Tail20 : List Nat := 14489 :: root583Tail21
def root583Tail19 : List Nat := 12809 :: root583Tail20
def root583Tail18 : List Nat := 12659 :: root583Tail19
def root583Tail17 : List Nat := 12113 :: root583Tail18
def root583Tail16 : List Nat := 11777 :: root583Tail17
def root583Tail15 : List Nat := 11549 :: root583Tail16
def root583Tail14 : List Nat := 10427 :: root583Tail15
def root583Tail13 : List Nat := 10247 :: root583Tail14
def root583Tail12 : List Nat := 10037 :: root583Tail13
def root583Tail11 : List Nat := 9491 :: root583Tail12
def root583Tail10 : List Nat := 9137 :: root583Tail11
def root583Tail9 : List Nat := 9029 :: root583Tail10
def root583Tail8 : List Nat := 8693 :: root583Tail9
def root583Tail7 : List Nat := 7823 :: root583Tail8
def root583Tail6 : List Nat := 6857 :: root583Tail7
def root583Tail5 : List Nat := 6131 :: root583Tail6
def root583Tail4 : List Nat := 5693 :: root583Tail5
def root583Tail3 : List Nat := 5573 :: root583Tail4
def root583Tail2 : List Nat := 5009 :: root583Tail3
def root583Tail1 : List Nat := 4673 :: root583Tail2
def root583Tail0 : List Nat := 4517 :: root583Tail1
def root583Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root583Candidates0_checked : candidates edge 4 4271 4517 root583Tail1 winner.bound = root583Candidates0 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root583Step0 : scan edge (search edge 3) 4 [4271] 4271 root583Tail0 winner =
    scan edge (search edge 3) 4 [4271] 4271 root583Tail1 winner := by
  rw [root583Tail0, scan, if_neg (by decide), root583Candidates0_checked]
  change scan edge (search edge 3) 4 [4271] 4271 root583Tail1
    (if 3 ≤ root583Candidates0.length then search edge 3 [4271, 4517] root583Candidates0 8788 winner else winner) = _
  rw [if_neg (by decide)]
def root583Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root583Candidates1_checked : candidates edge 4 4271 4673 root583Tail2 winner.bound = root583Candidates1 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root583Step1 : scan edge (search edge 3) 4 [4271] 4271 root583Tail1 winner =
    scan edge (search edge 3) 4 [4271] 4271 root583Tail2 winner := by
  rw [root583Tail1, scan, if_neg (by decide), root583Candidates1_checked]
  change scan edge (search edge 3) 4 [4271] 4271 root583Tail2
    (if 3 ≤ root583Candidates1.length then search edge 3 [4271, 4673] root583Candidates1 8944 winner else winner) = _
  rw [if_neg (by decide)]
def root583Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root583Candidates2_checked : candidates edge 4 4271 5009 root583Tail3 winner.bound = root583Candidates2 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root583Step2 : scan edge (search edge 3) 4 [4271] 4271 root583Tail2 winner =
    scan edge (search edge 3) 4 [4271] 4271 root583Tail3 winner := by
  rw [root583Tail2, scan, if_neg (by decide), root583Candidates2_checked]
  change scan edge (search edge 3) 4 [4271] 4271 root583Tail3
    (if 3 ≤ root583Candidates2.length then search edge 3 [4271, 5009] root583Candidates2 9280 winner else winner) = _
  rw [if_neg (by decide)]
theorem root583Scan3 : scan edge (search edge 3) 4 [4271] 4271 root583Tail3 winner = winner := by rw [root583Tail3, scan, if_pos (by decide)]
theorem root583Scan2 : scan edge (search edge 3) 4 [4271] 4271 root583Tail2 winner = winner :=
  (root583Step2).trans root583Scan3
theorem root583Scan1 : scan edge (search edge 3) 4 [4271] 4271 root583Tail1 winner = winner :=
  (root583Step1).trans root583Scan2
theorem root583Scan0 : scan edge (search edge 3) 4 [4271] 4271 root583Tail0 winner = winner :=
  (root583Step0).trans root583Scan1
set_option maxRecDepth 4096 in
theorem root583Branch : search edge 4 [4271] root583Candidates 4271 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4271] 4271 root583Tail0 winner = winner
  exact root583Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
