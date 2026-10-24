import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root563Candidates : List Nat := Data.row4111
set_option maxRecDepth 8192 in
theorem root563Candidates_checked : candidates edge 5 0 4111 tail564 winner.bound = root563Candidates := by
  apply candidates_eq_row edge 5 0 4111 tail564 root563Candidates winner.bound tail564_ordered
  · decide
  · decide
  · funext q
    rfl
def root563Tail44 : List Nat := []
def root563Tail43 : List Nat := 21817 :: root563Tail44
def root563Tail42 : List Nat := 21019 :: root563Tail43
def root563Tail41 : List Nat := 21001 :: root563Tail42
def root563Tail40 : List Nat := 20341 :: root563Tail41
def root563Tail39 : List Nat := 19843 :: root563Tail40
def root563Tail38 : List Nat := 19699 :: root563Tail39
def root563Tail37 : List Nat := 18919 :: root563Tail38
def root563Tail36 : List Nat := 18457 :: root563Tail37
def root563Tail35 : List Nat := 17467 :: root563Tail36
def root563Tail34 : List Nat := 17401 :: root563Tail35
def root563Tail33 : List Nat := 17359 :: root563Tail34
def root563Tail32 : List Nat := 16417 :: root563Tail33
def root563Tail31 : List Nat := 15877 :: root563Tail32
def root563Tail30 : List Nat := 14887 :: root563Tail31
def root563Tail29 : List Nat := 14551 :: root563Tail30
def root563Tail28 : List Nat := 13267 :: root563Tail29
def root563Tail27 : List Nat := 13249 :: root563Tail28
def root563Tail26 : List Nat := 12391 :: root563Tail27
def root563Tail25 : List Nat := 12157 :: root563Tail26
def root563Tail24 : List Nat := 12073 :: root563Tail25
def root563Tail23 : List Nat := 11593 :: root563Tail24
def root563Tail22 : List Nat := 11353 :: root563Tail23
def root563Tail21 : List Nat := 10957 :: root563Tail22
def root563Tail20 : List Nat := 10639 :: root563Tail21
def root563Tail19 : List Nat := 10627 :: root563Tail20
def root563Tail18 : List Nat := 10597 :: root563Tail19
def root563Tail17 : List Nat := 9811 :: root563Tail18
def root563Tail16 : List Nat := 9349 :: root563Tail17
def root563Tail15 : List Nat := 9181 :: root563Tail16
def root563Tail14 : List Nat := 8719 :: root563Tail15
def root563Tail13 : List Nat := 8167 :: root563Tail14
def root563Tail12 : List Nat := 7789 :: root563Tail13
def root563Tail11 : List Nat := 7369 :: root563Tail12
def root563Tail10 : List Nat := 6949 :: root563Tail11
def root563Tail9 : List Nat := 6679 :: root563Tail10
def root563Tail8 : List Nat := 6661 :: root563Tail9
def root563Tail7 : List Nat := 6637 :: root563Tail8
def root563Tail6 : List Nat := 6553 :: root563Tail7
def root563Tail5 : List Nat := 6073 :: root563Tail6
def root563Tail4 : List Nat := 5857 :: root563Tail5
def root563Tail3 : List Nat := 5527 :: root563Tail4
def root563Tail2 : List Nat := 5521 :: root563Tail3
def root563Tail1 : List Nat := 5119 :: root563Tail2
def root563Tail0 : List Nat := 4261 :: root563Tail1
def root563Candidates0 : List Nat := [12073, 13267]
set_option maxRecDepth 8192 in
theorem root563Candidates0_checked : candidates edge 4 4111 4261 root563Tail1 winner.bound = root563Candidates0 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root563Step0 : scan edge (search edge 3) 4 [4111] 4111 root563Tail0 winner =
    scan edge (search edge 3) 4 [4111] 4111 root563Tail1 winner := by
  rw [root563Tail0, scan, if_neg (by decide), root563Candidates0_checked]
  change scan edge (search edge 3) 4 [4111] 4111 root563Tail1
    (if 3 ≤ root563Candidates0.length then search edge 3 [4111, 4261] root563Candidates0 8372 winner else winner) = _
  rw [if_neg (by decide)]
def root563Candidates1 : List Nat := [6073, 6637, 10957]
set_option maxRecDepth 8192 in
theorem root563Candidates1_checked : candidates edge 4 4111 5119 root563Tail2 winner.bound = root563Candidates1 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root563Child1 : search edge 3 [4111, 5119] root563Candidates1 9230 winner = winner := by decide
theorem root563Step1 : scan edge (search edge 3) 4 [4111] 4111 root563Tail1 winner =
    scan edge (search edge 3) 4 [4111] 4111 root563Tail2 winner := by
  rw [root563Tail1, scan, if_neg (by decide), root563Candidates1_checked]
  change scan edge (search edge 3) 4 [4111] 4111 root563Tail2
    (if 3 ≤ root563Candidates1.length then search edge 3 [4111, 5119] root563Candidates1 9230 winner else winner) = _
  rw [if_pos (by decide), root563Child1]
theorem root563Scan2 : scan edge (search edge 3) 4 [4111] 4111 root563Tail2 winner = winner := by rw [root563Tail2, scan, if_pos (by decide)]
theorem root563Scan1 : scan edge (search edge 3) 4 [4111] 4111 root563Tail1 winner = winner :=
  (root563Step1).trans root563Scan2
theorem root563Scan0 : scan edge (search edge 3) 4 [4111] 4111 root563Tail0 winner = winner :=
  (root563Step0).trans root563Scan1
set_option maxRecDepth 4096 in
theorem root563Branch : search edge 4 [4111] root563Candidates 4111 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4111] 4111 root563Tail0 winner = winner
  exact root563Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
