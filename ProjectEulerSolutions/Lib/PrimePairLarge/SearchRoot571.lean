import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root571Candidates : List Nat := Data.row4177
set_option maxRecDepth 8192 in
theorem root571Candidates_checked : candidates edge 5 0 4177 tail572 winner.bound = root571Candidates := by
  apply candidates_eq_row edge 5 0 4177 tail572 root571Candidates winner.bound tail572_ordered
  · decide
  · decide
  · funext q
    rfl
def root571Tail31 : List Nat := []
def root571Tail30 : List Nat := 21169 :: root571Tail31
def root571Tail29 : List Nat := 20593 :: root571Tail30
def root571Tail28 : List Nat := 19441 :: root571Tail29
def root571Tail27 : List Nat := 17107 :: root571Tail28
def root571Tail26 : List Nat := 16417 :: root571Tail27
def root571Tail25 : List Nat := 16183 :: root571Tail26
def root571Tail24 : List Nat := 15937 :: root571Tail25
def root571Tail23 : List Nat := 15919 :: root571Tail24
def root571Tail22 : List Nat := 13963 :: root571Tail23
def root571Tail21 : List Nat := 13693 :: root571Tail22
def root571Tail20 : List Nat := 12487 :: root571Tail21
def root571Tail19 : List Nat := 11587 :: root571Tail20
def root571Tail18 : List Nat := 10303 :: root571Tail19
def root571Tail17 : List Nat := 9973 :: root571Tail18
def root571Tail16 : List Nat := 9781 :: root571Tail17
def root571Tail15 : List Nat := 8923 :: root571Tail16
def root571Tail14 : List Nat := 8521 :: root571Tail15
def root571Tail13 : List Nat := 8353 :: root571Tail14
def root571Tail12 : List Nat := 8311 :: root571Tail13
def root571Tail11 : List Nat := 8101 :: root571Tail12
def root571Tail10 : List Nat := 8053 :: root571Tail11
def root571Tail9 : List Nat := 7537 :: root571Tail10
def root571Tail8 : List Nat := 7297 :: root571Tail9
def root571Tail7 : List Nat := 6883 :: root571Tail8
def root571Tail6 : List Nat := 6781 :: root571Tail7
def root571Tail5 : List Nat := 6673 :: root571Tail6
def root571Tail4 : List Nat := 6379 :: root571Tail5
def root571Tail3 : List Nat := 5791 :: root571Tail4
def root571Tail2 : List Nat := 5323 :: root571Tail3
def root571Tail1 : List Nat := 4903 :: root571Tail2
def root571Tail0 : List Nat := 4363 :: root571Tail1
def root571Candidates0 : List Nat := [8311]
set_option maxRecDepth 8192 in
theorem root571Candidates0_checked : candidates edge 4 4177 4363 root571Tail1 winner.bound = root571Candidates0 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root571Step0 : scan edge (search edge 3) 4 [4177] 4177 root571Tail0 winner =
    scan edge (search edge 3) 4 [4177] 4177 root571Tail1 winner := by
  rw [root571Tail0, scan, if_neg (by decide), root571Candidates0_checked]
  change scan edge (search edge 3) 4 [4177] 4177 root571Tail1
    (if 3 ≤ root571Candidates0.length then search edge 3 [4177, 4363] root571Candidates0 8540 winner else winner) = _
  rw [if_neg (by decide)]
def root571Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root571Candidates1_checked : candidates edge 4 4177 4903 root571Tail2 winner.bound = root571Candidates1 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root571Step1 : scan edge (search edge 3) 4 [4177] 4177 root571Tail1 winner =
    scan edge (search edge 3) 4 [4177] 4177 root571Tail2 winner := by
  rw [root571Tail1, scan, if_neg (by decide), root571Candidates1_checked]
  change scan edge (search edge 3) 4 [4177] 4177 root571Tail2
    (if 3 ≤ root571Candidates1.length then search edge 3 [4177, 4903] root571Candidates1 9080 winner else winner) = _
  rw [if_neg (by decide)]
def root571Candidates2 : List Nat := [7537]
set_option maxRecDepth 8192 in
theorem root571Candidates2_checked : candidates edge 4 4177 5323 root571Tail3 winner.bound = root571Candidates2 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
theorem root571Step2 : scan edge (search edge 3) 4 [4177] 4177 root571Tail2 winner =
    scan edge (search edge 3) 4 [4177] 4177 root571Tail3 winner := by
  rw [root571Tail2, scan, if_neg (by decide), root571Candidates2_checked]
  change scan edge (search edge 3) 4 [4177] 4177 root571Tail3
    (if 3 ≤ root571Candidates2.length then search edge 3 [4177, 5323] root571Candidates2 9500 winner else winner) = _
  rw [if_neg (by decide)]
theorem root571Scan3 : scan edge (search edge 3) 4 [4177] 4177 root571Tail3 winner = winner := by rw [root571Tail3, scan, if_pos (by decide)]
theorem root571Scan2 : scan edge (search edge 3) 4 [4177] 4177 root571Tail2 winner = winner :=
  (root571Step2).trans root571Scan3
theorem root571Scan1 : scan edge (search edge 3) 4 [4177] 4177 root571Tail1 winner = winner :=
  (root571Step1).trans root571Scan2
theorem root571Scan0 : scan edge (search edge 3) 4 [4177] 4177 root571Tail0 winner = winner :=
  (root571Step0).trans root571Scan1
set_option maxRecDepth 4096 in
theorem root571Branch : search edge 4 [4177] root571Candidates 4177 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4177] 4177 root571Tail0 winner = winner
  exact root571Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
