import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root516Candidates : List Nat := Data.row3719
set_option maxRecDepth 8192 in
theorem root516Candidates_checked : candidates edge 5 0 3719 tail517 winner.bound = root516Candidates := by
  apply candidates_eq_row edge 5 0 3719 tail517 root516Candidates winner.bound tail517_ordered
  · decide
  · decide
  · funext q
    rfl
def root516Tail24 : List Nat := []
def root516Tail23 : List Nat := 22157 :: root516Tail24
def root516Tail22 : List Nat := 19739 :: root516Tail23
def root516Tail21 : List Nat := 19157 :: root516Tail22
def root516Tail20 : List Nat := 17939 :: root516Tail21
def root516Tail19 : List Nat := 17417 :: root516Tail20
def root516Tail18 : List Nat := 17207 :: root516Tail19
def root516Tail17 : List Nat := 16487 :: root516Tail18
def root516Tail16 : List Nat := 16067 :: root516Tail17
def root516Tail15 : List Nat := 16001 :: root516Tail16
def root516Tail14 : List Nat := 15017 :: root516Tail15
def root516Tail13 : List Nat := 14489 :: root516Tail14
def root516Tail12 : List Nat := 13781 :: root516Tail13
def root516Tail11 : List Nat := 13697 :: root516Tail12
def root516Tail10 : List Nat := 13259 :: root516Tail11
def root516Tail9 : List Nat := 13187 :: root516Tail10
def root516Tail8 : List Nat := 12899 :: root516Tail9
def root516Tail7 : List Nat := 11621 :: root516Tail8
def root516Tail6 : List Nat := 10103 :: root516Tail7
def root516Tail5 : List Nat := 8663 :: root516Tail6
def root516Tail4 : List Nat := 6791 :: root516Tail5
def root516Tail3 : List Nat := 5591 :: root516Tail4
def root516Tail2 : List Nat := 5297 :: root516Tail3
def root516Tail1 : List Nat := 5273 :: root516Tail2
def root516Tail0 : List Nat := 4271 :: root516Tail1
def root516Candidates0 : List Nat := [14489]
set_option maxRecDepth 8192 in
theorem root516Candidates0_checked : candidates edge 4 3719 4271 root516Tail1 winner.bound = root516Candidates0 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root516Step0 : scan edge (search edge 3) 4 [3719] 3719 root516Tail0 winner =
    scan edge (search edge 3) 4 [3719] 3719 root516Tail1 winner := by
  rw [root516Tail0, scan, if_neg (by decide), root516Candidates0_checked]
  change scan edge (search edge 3) 4 [3719] 3719 root516Tail1
    (if 3 ≤ root516Candidates0.length then search edge 3 [3719, 4271] root516Candidates0 7990 winner else winner) = _
  rw [if_neg (by decide)]
def root516Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root516Candidates1_checked : candidates edge 4 3719 5273 root516Tail2 winner.bound = root516Candidates1 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root516Step1 : scan edge (search edge 3) 4 [3719] 3719 root516Tail1 winner =
    scan edge (search edge 3) 4 [3719] 3719 root516Tail2 winner := by
  rw [root516Tail1, scan, if_neg (by decide), root516Candidates1_checked]
  change scan edge (search edge 3) 4 [3719] 3719 root516Tail2
    (if 3 ≤ root516Candidates1.length then search edge 3 [3719, 5273] root516Candidates1 8992 winner else winner) = _
  rw [if_neg (by decide)]
def root516Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root516Candidates2_checked : candidates edge 4 3719 5297 root516Tail3 winner.bound = root516Candidates2 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root516Step2 : scan edge (search edge 3) 4 [3719] 3719 root516Tail2 winner =
    scan edge (search edge 3) 4 [3719] 3719 root516Tail3 winner := by
  rw [root516Tail2, scan, if_neg (by decide), root516Candidates2_checked]
  change scan edge (search edge 3) 4 [3719] 3719 root516Tail3
    (if 3 ≤ root516Candidates2.length then search edge 3 [3719, 5297] root516Candidates2 9016 winner else winner) = _
  rw [if_neg (by decide)]
theorem root516Scan3 : scan edge (search edge 3) 4 [3719] 3719 root516Tail3 winner = winner := by rw [root516Tail3, scan, if_pos (by decide)]
theorem root516Scan2 : scan edge (search edge 3) 4 [3719] 3719 root516Tail2 winner = winner :=
  (root516Step2).trans root516Scan3
theorem root516Scan1 : scan edge (search edge 3) 4 [3719] 3719 root516Tail1 winner = winner :=
  (root516Step1).trans root516Scan2
theorem root516Scan0 : scan edge (search edge 3) 4 [3719] 3719 root516Tail0 winner = winner :=
  (root516Step0).trans root516Scan1
set_option maxRecDepth 4096 in
theorem root516Branch : search edge 4 [3719] root516Candidates 3719 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3719] 3719 root516Tail0 winner = winner
  exact root516Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
