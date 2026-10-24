import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root611Candidates : List Nat := Data.row4519
set_option maxRecDepth 8192 in
theorem root611Candidates_checked : candidates edge 5 0 4519 tail612 winner.bound = root611Candidates := by
  apply candidates_eq_row edge 5 0 4519 tail612 root611Candidates winner.bound tail612_ordered
  · decide
  · decide
  · funext q
    rfl
def root611Tail35 : List Nat := []
def root611Tail34 : List Nat := 21247 :: root611Tail35
def root611Tail33 : List Nat := 20857 :: root611Tail34
def root611Tail32 : List Nat := 20743 :: root611Tail33
def root611Tail31 : List Nat := 20107 :: root611Tail32
def root611Tail30 : List Nat := 19183 :: root611Tail31
def root611Tail29 : List Nat := 19141 :: root611Tail30
def root611Tail28 : List Nat := 18679 :: root611Tail29
def root611Tail27 : List Nat := 18289 :: root611Tail28
def root611Tail26 : List Nat := 17497 :: root611Tail27
def root611Tail25 : List Nat := 16651 :: root611Tail26
def root611Tail24 : List Nat := 16567 :: root611Tail25
def root611Tail23 : List Nat := 15733 :: root611Tail24
def root611Tail22 : List Nat := 15649 :: root611Tail23
def root611Tail21 : List Nat := 15319 :: root611Tail22
def root611Tail20 : List Nat := 14737 :: root611Tail21
def root611Tail19 : List Nat := 13921 :: root611Tail20
def root611Tail18 : List Nat := 13093 :: root611Tail19
def root611Tail17 : List Nat := 12589 :: root611Tail18
def root611Tail16 : List Nat := 12457 :: root611Tail17
def root611Tail15 : List Nat := 10453 :: root611Tail16
def root611Tail14 : List Nat := 10321 :: root611Tail15
def root611Tail13 : List Nat := 9907 :: root611Tail14
def root611Tail12 : List Nat := 9397 :: root611Tail13
def root611Tail11 : List Nat := 9199 :: root611Tail12
def root611Tail10 : List Nat := 9103 :: root611Tail11
def root611Tail9 : List Nat := 9013 :: root611Tail10
def root611Tail8 : List Nat := 8821 :: root611Tail9
def root611Tail7 : List Nat := 8737 :: root611Tail8
def root611Tail6 : List Nat := 6133 :: root611Tail7
def root611Tail5 : List Nat := 6043 :: root611Tail6
def root611Tail4 : List Nat := 5701 :: root611Tail5
def root611Tail3 : List Nat := 5419 :: root611Tail4
def root611Tail2 : List Nat := 5209 :: root611Tail3
def root611Tail1 : List Nat := 4729 :: root611Tail2
def root611Tail0 : List Nat := 4657 :: root611Tail1
def root611Candidates0 : List Nat := [8737]
set_option maxRecDepth 8192 in
theorem root611Candidates0_checked : candidates edge 4 4519 4657 root611Tail1 winner.bound = root611Candidates0 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root611Step0 : scan edge (search edge 3) 4 [4519] 4519 root611Tail0 winner =
    scan edge (search edge 3) 4 [4519] 4519 root611Tail1 winner := by
  rw [root611Tail0, scan, if_neg (by decide), root611Candidates0_checked]
  change scan edge (search edge 3) 4 [4519] 4519 root611Tail1
    (if 3 ≤ root611Candidates0.length then search edge 3 [4519, 4657] root611Candidates0 9176 winner else winner) = _
  rw [if_neg (by decide)]
def root611Candidates1 : List Nat := [12457, 16651]
set_option maxRecDepth 8192 in
theorem root611Candidates1_checked : candidates edge 4 4519 4729 root611Tail2 winner.bound = root611Candidates1 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
theorem root611Step1 : scan edge (search edge 3) 4 [4519] 4519 root611Tail1 winner =
    scan edge (search edge 3) 4 [4519] 4519 root611Tail2 winner := by
  rw [root611Tail1, scan, if_neg (by decide), root611Candidates1_checked]
  change scan edge (search edge 3) 4 [4519] 4519 root611Tail2
    (if 3 ≤ root611Candidates1.length then search edge 3 [4519, 4729] root611Candidates1 9248 winner else winner) = _
  rw [if_neg (by decide)]
def root611Candidates2 : List Nat := [5419]
set_option maxRecDepth 8192 in
theorem root611Candidates2_checked : candidates edge 4 4519 5209 root611Tail3 winner.bound = root611Candidates2 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root611Step2 : scan edge (search edge 3) 4 [4519] 4519 root611Tail2 winner =
    scan edge (search edge 3) 4 [4519] 4519 root611Tail3 winner := by
  rw [root611Tail2, scan, if_neg (by decide), root611Candidates2_checked]
  change scan edge (search edge 3) 4 [4519] 4519 root611Tail3
    (if 3 ≤ root611Candidates2.length then search edge 3 [4519, 5209] root611Candidates2 9728 winner else winner) = _
  rw [if_neg (by decide)]
theorem root611Scan3 : scan edge (search edge 3) 4 [4519] 4519 root611Tail3 winner = winner := by rw [root611Tail3, scan, if_pos (by decide)]
theorem root611Scan2 : scan edge (search edge 3) 4 [4519] 4519 root611Tail2 winner = winner :=
  (root611Step2).trans root611Scan3
theorem root611Scan1 : scan edge (search edge 3) 4 [4519] 4519 root611Tail1 winner = winner :=
  (root611Step1).trans root611Scan2
theorem root611Scan0 : scan edge (search edge 3) 4 [4519] 4519 root611Tail0 winner = winner :=
  (root611Step0).trans root611Scan1
set_option maxRecDepth 4096 in
theorem root611Branch : search edge 4 [4519] root611Candidates 4519 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4519] 4519 root611Tail0 winner = winner
  exact root611Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
