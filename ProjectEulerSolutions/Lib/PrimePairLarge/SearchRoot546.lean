import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root546Candidates : List Nat := Data.row3967
set_option maxRecDepth 8192 in
theorem root546Candidates_checked : candidates edge 5 0 3967 tail547 winner.bound = root546Candidates := by
  apply candidates_eq_row edge 5 0 3967 tail547 root546Candidates winner.bound tail547_ordered
  · decide
  · decide
  · funext q
    rfl
def root546Tail34 : List Nat := []
def root546Tail33 : List Nat := 21649 :: root546Tail34
def root546Tail32 : List Nat := 21187 :: root546Tail33
def root546Tail31 : List Nat := 21001 :: root546Tail32
def root546Tail30 : List Nat := 20011 :: root546Tail31
def root546Tail29 : List Nat := 19777 :: root546Tail30
def root546Tail28 : List Nat := 18661 :: root546Tail29
def root546Tail27 : List Nat := 18313 :: root546Tail28
def root546Tail26 : List Nat := 17851 :: root546Tail27
def root546Tail25 : List Nat := 16381 :: root546Tail26
def root546Tail24 : List Nat := 16189 :: root546Tail25
def root546Tail23 : List Nat := 15973 :: root546Tail24
def root546Tail22 : List Nat := 13627 :: root546Tail23
def root546Tail21 : List Nat := 13249 :: root546Tail22
def root546Tail20 : List Nat := 12517 :: root546Tail21
def root546Tail19 : List Nat := 11971 :: root546Tail20
def root546Tail18 : List Nat := 11467 :: root546Tail19
def root546Tail17 : List Nat := 11173 :: root546Tail18
def root546Tail16 : List Nat := 9781 :: root546Tail17
def root546Tail15 : List Nat := 8893 :: root546Tail16
def root546Tail14 : List Nat := 8209 :: root546Tail15
def root546Tail13 : List Nat := 8053 :: root546Tail14
def root546Tail12 : List Nat := 7933 :: root546Tail13
def root546Tail11 : List Nat := 7927 :: root546Tail12
def root546Tail10 : List Nat := 7699 :: root546Tail11
def root546Tail9 : List Nat := 7177 :: root546Tail10
def root546Tail8 : List Nat := 7159 :: root546Tail9
def root546Tail7 : List Nat := 6271 :: root546Tail8
def root546Tail6 : List Nat := 6211 :: root546Tail7
def root546Tail5 : List Nat := 6067 :: root546Tail6
def root546Tail4 : List Nat := 5953 :: root546Tail5
def root546Tail3 : List Nat := 5791 :: root546Tail4
def root546Tail2 : List Nat := 5011 :: root546Tail3
def root546Tail1 : List Nat := 4783 :: root546Tail2
def root546Tail0 : List Nat := 4177 :: root546Tail1
def root546Candidates0 : List Nat := [5791, 8053, 9781]
set_option maxRecDepth 8192 in
theorem root546Candidates0_checked : candidates edge 4 3967 4177 root546Tail1 winner.bound = root546Candidates0 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root546Child0 : search edge 3 [3967, 4177] root546Candidates0 8144 winner = winner := by decide
theorem root546Step0 : scan edge (search edge 3) 4 [3967] 3967 root546Tail0 winner =
    scan edge (search edge 3) 4 [3967] 3967 root546Tail1 winner := by
  rw [root546Tail0, scan, if_neg (by decide), root546Candidates0_checked]
  change scan edge (search edge 3) 4 [3967] 3967 root546Tail1
    (if 3 ≤ root546Candidates0.length then search edge 3 [3967, 4177] root546Candidates0 8144 winner else winner) = _
  rw [if_pos (by decide), root546Child0]
def root546Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root546Candidates1_checked : candidates edge 4 3967 4783 root546Tail2 winner.bound = root546Candidates1 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root546Step1 : scan edge (search edge 3) 4 [3967] 3967 root546Tail1 winner =
    scan edge (search edge 3) 4 [3967] 3967 root546Tail2 winner := by
  rw [root546Tail1, scan, if_neg (by decide), root546Candidates1_checked]
  change scan edge (search edge 3) 4 [3967] 3967 root546Tail2
    (if 3 ≤ root546Candidates1.length then search edge 3 [3967, 4783] root546Candidates1 8750 winner else winner) = _
  rw [if_neg (by decide)]
def root546Candidates2 : List Nat := [6271, 8053]
set_option maxRecDepth 8192 in
theorem root546Candidates2_checked : candidates edge 4 3967 5011 root546Tail3 winner.bound = root546Candidates2 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root546Step2 : scan edge (search edge 3) 4 [3967] 3967 root546Tail2 winner =
    scan edge (search edge 3) 4 [3967] 3967 root546Tail3 winner := by
  rw [root546Tail2, scan, if_neg (by decide), root546Candidates2_checked]
  change scan edge (search edge 3) 4 [3967] 3967 root546Tail3
    (if 3 ≤ root546Candidates2.length then search edge 3 [3967, 5011] root546Candidates2 8978 winner else winner) = _
  rw [if_neg (by decide)]
theorem root546Scan3 : scan edge (search edge 3) 4 [3967] 3967 root546Tail3 winner = winner := by rw [root546Tail3, scan, if_pos (by decide)]
theorem root546Scan2 : scan edge (search edge 3) 4 [3967] 3967 root546Tail2 winner = winner :=
  (root546Step2).trans root546Scan3
theorem root546Scan1 : scan edge (search edge 3) 4 [3967] 3967 root546Tail1 winner = winner :=
  (root546Step1).trans root546Scan2
theorem root546Scan0 : scan edge (search edge 3) 4 [3967] 3967 root546Tail0 winner = winner :=
  (root546Step0).trans root546Scan1
set_option maxRecDepth 4096 in
theorem root546Branch : search edge 4 [3967] root546Candidates 3967 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3967] 3967 root546Tail0 winner = winner
  exact root546Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
