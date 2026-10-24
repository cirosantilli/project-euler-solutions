import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root542Candidates : List Nat := Data.row3929
set_option maxRecDepth 8192 in
theorem root542Candidates_checked : candidates edge 5 0 3929 tail543 winner.bound = root542Candidates := by
  apply candidates_eq_row edge 5 0 3929 tail543 root542Candidates winner.bound tail543_ordered
  · decide
  · decide
  · funext q
    rfl
def root542Tail35 : List Nat := []
def root542Tail34 : List Nat := 21929 :: root542Tail35
def root542Tail33 : List Nat := 20393 :: root542Tail34
def root542Tail32 : List Nat := 19913 :: root542Tail33
def root542Tail31 : List Nat := 19433 :: root542Tail32
def root542Tail30 : List Nat := 19139 :: root542Tail31
def root542Tail29 : List Nat := 19073 :: root542Tail30
def root542Tail28 : List Nat := 17747 :: root542Tail29
def root542Tail27 : List Nat := 17597 :: root542Tail28
def root542Tail26 : List Nat := 17417 :: root542Tail27
def root542Tail25 : List Nat := 17387 :: root542Tail26
def root542Tail24 : List Nat := 16319 :: root542Tail25
def root542Tail23 : List Nat := 15737 :: root542Tail24
def root542Tail22 : List Nat := 15401 :: root542Tail23
def root542Tail21 : List Nat := 14891 :: root542Tail22
def root542Tail20 : List Nat := 14723 :: root542Tail21
def root542Tail19 : List Nat := 13799 :: root542Tail20
def root542Tail18 : List Nat := 13049 :: root542Tail19
def root542Tail17 : List Nat := 12437 :: root542Tail18
def root542Tail16 : List Nat := 11549 :: root542Tail17
def root542Tail15 : List Nat := 11447 :: root542Tail16
def root542Tail14 : List Nat := 11171 :: root542Tail15
def root542Tail13 : List Nat := 10487 :: root542Tail14
def root542Tail12 : List Nat := 9941 :: root542Tail13
def root542Tail11 : List Nat := 9521 :: root542Tail12
def root542Tail10 : List Nat := 9209 :: root542Tail11
def root542Tail9 : List Nat := 9059 :: root542Tail10
def root542Tail8 : List Nat := 7691 :: root542Tail9
def root542Tail7 : List Nat := 7457 :: root542Tail8
def root542Tail6 : List Nat := 7109 :: root542Tail7
def root542Tail5 : List Nat := 7043 :: root542Tail6
def root542Tail4 : List Nat := 6221 :: root542Tail5
def root542Tail3 : List Nat := 6143 :: root542Tail4
def root542Tail2 : List Nat := 6053 :: root542Tail3
def root542Tail1 : List Nat := 4421 :: root542Tail2
def root542Tail0 : List Nat := 4349 :: root542Tail1
def root542Candidates0 : List Nat := [11171]
set_option maxRecDepth 8192 in
theorem root542Candidates0_checked : candidates edge 4 3929 4349 root542Tail1 winner.bound = root542Candidates0 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root542Step0 : scan edge (search edge 3) 4 [3929] 3929 root542Tail0 winner =
    scan edge (search edge 3) 4 [3929] 3929 root542Tail1 winner := by
  rw [root542Tail0, scan, if_neg (by decide), root542Candidates0_checked]
  change scan edge (search edge 3) 4 [3929] 3929 root542Tail1
    (if 3 ≤ root542Candidates0.length then search edge 3 [3929, 4349] root542Candidates0 8278 winner else winner) = _
  rw [if_neg (by decide)]
def root542Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root542Candidates1_checked : candidates edge 4 3929 4421 root542Tail2 winner.bound = root542Candidates1 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root542Step1 : scan edge (search edge 3) 4 [3929] 3929 root542Tail1 winner =
    scan edge (search edge 3) 4 [3929] 3929 root542Tail2 winner := by
  rw [root542Tail1, scan, if_neg (by decide), root542Candidates1_checked]
  change scan edge (search edge 3) 4 [3929] 3929 root542Tail2
    (if 3 ≤ root542Candidates1.length then search edge 3 [3929, 4421] root542Candidates1 8350 winner else winner) = _
  rw [if_neg (by decide)]
theorem root542Scan2 : scan edge (search edge 3) 4 [3929] 3929 root542Tail2 winner = winner := by rw [root542Tail2, scan, if_pos (by decide)]
theorem root542Scan1 : scan edge (search edge 3) 4 [3929] 3929 root542Tail1 winner = winner :=
  (root542Step1).trans root542Scan2
theorem root542Scan0 : scan edge (search edge 3) 4 [3929] 3929 root542Tail0 winner = winner :=
  (root542Step0).trans root542Scan1
set_option maxRecDepth 4096 in
theorem root542Branch : search edge 4 [3929] root542Candidates 3929 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3929] 3929 root542Tail0 winner = winner
  exact root542Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
