import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root548Candidates : List Nat := Data.row4001
set_option maxRecDepth 8192 in
theorem root548Candidates_checked : candidates edge 5 0 4001 tail549 winner.bound = root548Candidates := by
  apply candidates_eq_row edge 5 0 4001 tail549 root548Candidates winner.bound tail549_ordered
  · decide
  · decide
  · funext q
    rfl
def root548Tail34 : List Nat := []
def root548Tail33 : List Nat := 22031 :: root548Tail34
def root548Tail32 : List Nat := 21089 :: root548Tail33
def root548Tail31 : List Nat := 20807 :: root548Tail32
def root548Tail30 : List Nat := 20759 :: root548Tail31
def root548Tail29 : List Nat := 19997 :: root548Tail30
def root548Tail28 : List Nat := 19919 :: root548Tail29
def root548Tail27 : List Nat := 17609 :: root548Tail28
def root548Tail26 : List Nat := 16889 :: root548Tail27
def root548Tail25 : List Nat := 16319 :: root548Tail26
def root548Tail24 : List Nat := 16091 :: root548Tail25
def root548Tail23 : List Nat := 15773 :: root548Tail24
def root548Tail22 : List Nat := 15629 :: root548Tail23
def root548Tail21 : List Nat := 14723 :: root548Tail22
def root548Tail20 : List Nat := 13877 :: root548Tail21
def root548Tail19 : List Nat := 13619 :: root548Tail20
def root548Tail18 : List Nat := 12569 :: root548Tail19
def root548Tail17 : List Nat := 12329 :: root548Tail18
def root548Tail16 : List Nat := 10253 :: root548Tail17
def root548Tail15 : List Nat := 9689 :: root548Tail16
def root548Tail14 : List Nat := 9467 :: root548Tail15
def root548Tail13 : List Nat := 8543 :: root548Tail14
def root548Tail12 : List Nat := 8513 :: root548Tail13
def root548Tail11 : List Nat := 8219 :: root548Tail12
def root548Tail10 : List Nat := 7877 :: root548Tail11
def root548Tail9 : List Nat := 7583 :: root548Tail10
def root548Tail8 : List Nat := 7577 :: root548Tail9
def root548Tail7 : List Nat := 7121 :: root548Tail8
def root548Tail6 : List Nat := 6563 :: root548Tail7
def root548Tail5 : List Nat := 6329 :: root548Tail6
def root548Tail4 : List Nat := 6323 :: root548Tail5
def root548Tail3 : List Nat := 5783 :: root548Tail4
def root548Tail2 : List Nat := 4391 :: root548Tail3
def root548Tail1 : List Nat := 4349 :: root548Tail2
def root548Tail0 : List Nat := 4253 :: root548Tail1
def root548Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root548Candidates0_checked : candidates edge 4 4001 4253 root548Tail1 winner.bound = root548Candidates0 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root548Step0 : scan edge (search edge 3) 4 [4001] 4001 root548Tail0 winner =
    scan edge (search edge 3) 4 [4001] 4001 root548Tail1 winner := by
  rw [root548Tail0, scan, if_neg (by decide), root548Candidates0_checked]
  change scan edge (search edge 3) 4 [4001] 4001 root548Tail1
    (if 3 ≤ root548Candidates0.length then search edge 3 [4001, 4253] root548Candidates0 8254 winner else winner) = _
  rw [if_neg (by decide)]
def root548Candidates1 : List Nat := [15629]
set_option maxRecDepth 8192 in
theorem root548Candidates1_checked : candidates edge 4 4001 4349 root548Tail2 winner.bound = root548Candidates1 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root548Step1 : scan edge (search edge 3) 4 [4001] 4001 root548Tail1 winner =
    scan edge (search edge 3) 4 [4001] 4001 root548Tail2 winner := by
  rw [root548Tail1, scan, if_neg (by decide), root548Candidates1_checked]
  change scan edge (search edge 3) 4 [4001] 4001 root548Tail2
    (if 3 ≤ root548Candidates1.length then search edge 3 [4001, 4349] root548Candidates1 8350 winner else winner) = _
  rw [if_neg (by decide)]
def root548Candidates2 : List Nat := [8219]
set_option maxRecDepth 8192 in
theorem root548Candidates2_checked : candidates edge 4 4001 4391 root548Tail3 winner.bound = root548Candidates2 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root548Step2 : scan edge (search edge 3) 4 [4001] 4001 root548Tail2 winner =
    scan edge (search edge 3) 4 [4001] 4001 root548Tail3 winner := by
  rw [root548Tail2, scan, if_neg (by decide), root548Candidates2_checked]
  change scan edge (search edge 3) 4 [4001] 4001 root548Tail3
    (if 3 ≤ root548Candidates2.length then search edge 3 [4001, 4391] root548Candidates2 8392 winner else winner) = _
  rw [if_neg (by decide)]
theorem root548Scan3 : scan edge (search edge 3) 4 [4001] 4001 root548Tail3 winner = winner := by rw [root548Tail3, scan, if_pos (by decide)]
theorem root548Scan2 : scan edge (search edge 3) 4 [4001] 4001 root548Tail2 winner = winner :=
  (root548Step2).trans root548Scan3
theorem root548Scan1 : scan edge (search edge 3) 4 [4001] 4001 root548Tail1 winner = winner :=
  (root548Step1).trans root548Scan2
theorem root548Scan0 : scan edge (search edge 3) 4 [4001] 4001 root548Tail0 winner = winner :=
  (root548Step0).trans root548Scan1
set_option maxRecDepth 4096 in
theorem root548Branch : search edge 4 [4001] root548Candidates 4001 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4001] 4001 root548Tail0 winner = winner
  exact root548Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
