import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root581Candidates : List Nat := Data.row4259
set_option maxRecDepth 8192 in
theorem root581Candidates_checked : candidates edge 5 0 4259 tail582 winner.bound = root581Candidates := by
  apply candidates_eq_row edge 5 0 4259 tail582 root581Candidates winner.bound tail582_ordered
  · decide
  · decide
  · funext q
    rfl
def root581Tail28 : List Nat := []
def root581Tail27 : List Nat := 21227 :: root581Tail28
def root581Tail26 : List Nat := 19709 :: root581Tail27
def root581Tail25 : List Nat := 19001 :: root581Tail26
def root581Tail24 : List Nat := 18443 :: root581Tail25
def root581Tail23 : List Nat := 18329 :: root581Tail24
def root581Tail22 : List Nat := 17729 :: root581Tail23
def root581Tail21 : List Nat := 17351 :: root581Tail22
def root581Tail20 : List Nat := 15887 :: root581Tail21
def root581Tail19 : List Nat := 15881 :: root581Tail20
def root581Tail18 : List Nat := 15797 :: root581Tail19
def root581Tail17 : List Nat := 15383 :: root581Tail18
def root581Tail16 : List Nat := 12953 :: root581Tail17
def root581Tail15 : List Nat := 12743 :: root581Tail16
def root581Tail14 : List Nat := 11909 :: root581Tail15
def root581Tail13 : List Nat := 11597 :: root581Tail14
def root581Tail12 : List Nat := 11399 :: root581Tail13
def root581Tail11 : List Nat := 10847 :: root581Tail12
def root581Tail10 : List Nat := 10427 :: root581Tail11
def root581Tail9 : List Nat := 10181 :: root581Tail10
def root581Tail8 : List Nat := 10139 :: root581Tail9
def root581Tail7 : List Nat := 9677 :: root581Tail8
def root581Tail6 : List Nat := 9551 :: root581Tail7
def root581Tail5 : List Nat := 8609 :: root581Tail6
def root581Tail4 : List Nat := 6221 :: root581Tail5
def root581Tail3 : List Nat := 5807 :: root581Tail4
def root581Tail2 : List Nat := 5417 :: root581Tail3
def root581Tail1 : List Nat := 5087 :: root581Tail2
def root581Tail0 : List Nat := 4787 :: root581Tail1
def root581Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root581Candidates0_checked : candidates edge 4 4259 4787 root581Tail1 winner.bound = root581Candidates0 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root581Step0 : scan edge (search edge 3) 4 [4259] 4259 root581Tail0 winner =
    scan edge (search edge 3) 4 [4259] 4259 root581Tail1 winner := by
  rw [root581Tail0, scan, if_neg (by decide), root581Candidates0_checked]
  change scan edge (search edge 3) 4 [4259] 4259 root581Tail1
    (if 3 ≤ root581Candidates0.length then search edge 3 [4259, 4787] root581Candidates0 9046 winner else winner) = _
  rw [if_neg (by decide)]
def root581Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root581Candidates1_checked : candidates edge 4 4259 5087 root581Tail2 winner.bound = root581Candidates1 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root581Step1 : scan edge (search edge 3) 4 [4259] 4259 root581Tail1 winner =
    scan edge (search edge 3) 4 [4259] 4259 root581Tail2 winner := by
  rw [root581Tail1, scan, if_neg (by decide), root581Candidates1_checked]
  change scan edge (search edge 3) 4 [4259] 4259 root581Tail2
    (if 3 ≤ root581Candidates1.length then search edge 3 [4259, 5087] root581Candidates1 9346 winner else winner) = _
  rw [if_neg (by decide)]
def root581Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root581Candidates2_checked : candidates edge 4 4259 5417 root581Tail3 winner.bound = root581Candidates2 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root581Step2 : scan edge (search edge 3) 4 [4259] 4259 root581Tail2 winner =
    scan edge (search edge 3) 4 [4259] 4259 root581Tail3 winner := by
  rw [root581Tail2, scan, if_neg (by decide), root581Candidates2_checked]
  change scan edge (search edge 3) 4 [4259] 4259 root581Tail3
    (if 3 ≤ root581Candidates2.length then search edge 3 [4259, 5417] root581Candidates2 9676 winner else winner) = _
  rw [if_neg (by decide)]
theorem root581Scan3 : scan edge (search edge 3) 4 [4259] 4259 root581Tail3 winner = winner := by rw [root581Tail3, scan, if_pos (by decide)]
theorem root581Scan2 : scan edge (search edge 3) 4 [4259] 4259 root581Tail2 winner = winner :=
  (root581Step2).trans root581Scan3
theorem root581Scan1 : scan edge (search edge 3) 4 [4259] 4259 root581Tail1 winner = winner :=
  (root581Step1).trans root581Scan2
theorem root581Scan0 : scan edge (search edge 3) 4 [4259] 4259 root581Tail0 winner = winner :=
  (root581Step0).trans root581Scan1
set_option maxRecDepth 4096 in
theorem root581Branch : search edge 4 [4259] root581Candidates 4259 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4259] 4259 root581Tail0 winner = winner
  exact root581Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
