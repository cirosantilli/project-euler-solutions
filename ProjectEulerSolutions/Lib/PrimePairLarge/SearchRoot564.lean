import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root564Candidates : List Nat := Data.row4127
set_option maxRecDepth 8192 in
theorem root564Candidates_checked : candidates edge 5 0 4127 tail565 winner.bound = root564Candidates := by
  apply candidates_eq_row edge 5 0 4127 tail565 root564Candidates winner.bound tail565_ordered
  · decide
  · decide
  · funext q
    rfl
def root564Tail33 : List Nat := []
def root564Tail32 : List Nat := 20981 :: root564Tail33
def root564Tail31 : List Nat := 20879 :: root564Tail32
def root564Tail30 : List Nat := 20807 :: root564Tail31
def root564Tail29 : List Nat := 19913 :: root564Tail30
def root564Tail28 : List Nat := 19841 :: root564Tail29
def root564Tail27 : List Nat := 18539 :: root564Tail28
def root564Tail26 : List Nat := 18443 :: root564Tail27
def root564Tail25 : List Nat := 17489 :: root564Tail26
def root564Tail24 : List Nat := 17117 :: root564Tail25
def root564Tail23 : List Nat := 15551 :: root564Tail24
def root564Tail22 : List Nat := 14957 :: root564Tail23
def root564Tail21 : List Nat := 13751 :: root564Tail22
def root564Tail20 : List Nat := 13001 :: root564Tail21
def root564Tail19 : List Nat := 12983 :: root564Tail20
def root564Tail18 : List Nat := 12953 :: root564Tail19
def root564Tail17 : List Nat := 12917 :: root564Tail18
def root564Tail16 : List Nat := 12653 :: root564Tail17
def root564Tail15 : List Nat := 12203 :: root564Tail16
def root564Tail14 : List Nat := 11681 :: root564Tail15
def root564Tail13 : List Nat := 9767 :: root564Tail14
def root564Tail12 : List Nat := 9551 :: root564Tail13
def root564Tail11 : List Nat := 8933 :: root564Tail12
def root564Tail10 : List Nat := 8837 :: root564Tail11
def root564Tail9 : List Nat := 8291 :: root564Tail10
def root564Tail8 : List Nat := 8243 :: root564Tail9
def root564Tail7 : List Nat := 7949 :: root564Tail8
def root564Tail6 : List Nat := 7079 :: root564Tail7
def root564Tail5 : List Nat := 6959 :: root564Tail6
def root564Tail4 : List Nat := 6737 :: root564Tail5
def root564Tail3 : List Nat := 6581 :: root564Tail4
def root564Tail2 : List Nat := 5519 :: root564Tail3
def root564Tail1 : List Nat := 5309 :: root564Tail2
def root564Tail0 : List Nat := 4967 :: root564Tail1
def root564Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root564Candidates0_checked : candidates edge 4 4127 4967 root564Tail1 winner.bound = root564Candidates0 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root564Step0 : scan edge (search edge 3) 4 [4127] 4127 root564Tail0 winner =
    scan edge (search edge 3) 4 [4127] 4127 root564Tail1 winner := by
  rw [root564Tail0, scan, if_neg (by decide), root564Candidates0_checked]
  change scan edge (search edge 3) 4 [4127] 4127 root564Tail1
    (if 3 ≤ root564Candidates0.length then search edge 3 [4127, 4967] root564Candidates0 9094 winner else winner) = _
  rw [if_neg (by decide)]
def root564Candidates1 : List Nat := [15551]
set_option maxRecDepth 8192 in
theorem root564Candidates1_checked : candidates edge 4 4127 5309 root564Tail2 winner.bound = root564Candidates1 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root564Step1 : scan edge (search edge 3) 4 [4127] 4127 root564Tail1 winner =
    scan edge (search edge 3) 4 [4127] 4127 root564Tail2 winner := by
  rw [root564Tail1, scan, if_neg (by decide), root564Candidates1_checked]
  change scan edge (search edge 3) 4 [4127] 4127 root564Tail2
    (if 3 ≤ root564Candidates1.length then search edge 3 [4127, 5309] root564Candidates1 9436 winner else winner) = _
  rw [if_neg (by decide)]
theorem root564Scan2 : scan edge (search edge 3) 4 [4127] 4127 root564Tail2 winner = winner := by rw [root564Tail2, scan, if_pos (by decide)]
theorem root564Scan1 : scan edge (search edge 3) 4 [4127] 4127 root564Tail1 winner = winner :=
  (root564Step1).trans root564Scan2
theorem root564Scan0 : scan edge (search edge 3) 4 [4127] 4127 root564Tail0 winner = winner :=
  (root564Step0).trans root564Scan1
set_option maxRecDepth 4096 in
theorem root564Branch : search edge 4 [4127] root564Candidates 4127 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4127] 4127 root564Tail0 winner = winner
  exact root564Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
