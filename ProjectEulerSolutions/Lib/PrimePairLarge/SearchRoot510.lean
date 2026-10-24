import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root510Candidates : List Nat := Data.row3673
set_option maxRecDepth 8192 in
theorem root510Candidates_checked : candidates edge 5 0 3673 tail511 winner.bound = root510Candidates := by
  apply candidates_eq_row edge 5 0 3673 tail511 root510Candidates winner.bound tail511_ordered
  · decide
  · decide
  · funext q
    rfl
def root510Tail25 : List Nat := []
def root510Tail24 : List Nat := 21061 :: root510Tail25
def root510Tail23 : List Nat := 20389 :: root510Tail24
def root510Tail22 : List Nat := 20029 :: root510Tail23
def root510Tail21 : List Nat := 19501 :: root510Tail22
def root510Tail20 : List Nat := 18859 :: root510Tail21
def root510Tail19 : List Nat := 18229 :: root510Tail20
def root510Tail18 : List Nat := 17761 :: root510Tail19
def root510Tail17 : List Nat := 17053 :: root510Tail18
def root510Tail16 : List Nat := 15307 :: root510Tail17
def root510Tail15 : List Nat := 15259 :: root510Tail16
def root510Tail14 : List Nat := 14713 :: root510Tail15
def root510Tail13 : List Nat := 14407 :: root510Tail14
def root510Tail12 : List Nat := 13183 :: root510Tail13
def root510Tail11 : List Nat := 11731 :: root510Tail12
def root510Tail10 : List Nat := 9421 :: root510Tail11
def root510Tail9 : List Nat := 9151 :: root510Tail10
def root510Tail8 : List Nat := 8629 :: root510Tail9
def root510Tail7 : List Nat := 8293 :: root510Tail8
def root510Tail6 : List Nat := 7717 :: root510Tail7
def root510Tail5 : List Nat := 7537 :: root510Tail6
def root510Tail4 : List Nat := 6691 :: root510Tail5
def root510Tail3 : List Nat := 6529 :: root510Tail4
def root510Tail2 : List Nat := 5581 :: root510Tail3
def root510Tail1 : List Nat := 4759 :: root510Tail2
def root510Tail0 : List Nat := 4129 :: root510Tail1
def root510Candidates0 : List Nat := [7717]
set_option maxRecDepth 8192 in
theorem root510Candidates0_checked : candidates edge 4 3673 4129 root510Tail1 winner.bound = root510Candidates0 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root510Step0 : scan edge (search edge 3) 4 [3673] 3673 root510Tail0 winner =
    scan edge (search edge 3) 4 [3673] 3673 root510Tail1 winner := by
  rw [root510Tail0, scan, if_neg (by decide), root510Candidates0_checked]
  change scan edge (search edge 3) 4 [3673] 3673 root510Tail1
    (if 3 ≤ root510Candidates0.length then search edge 3 [3673, 4129] root510Candidates0 7802 winner else winner) = _
  rw [if_neg (by decide)]
def root510Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root510Candidates1_checked : candidates edge 4 3673 4759 root510Tail2 winner.bound = root510Candidates1 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root510Step1 : scan edge (search edge 3) 4 [3673] 3673 root510Tail1 winner =
    scan edge (search edge 3) 4 [3673] 3673 root510Tail2 winner := by
  rw [root510Tail1, scan, if_neg (by decide), root510Candidates1_checked]
  change scan edge (search edge 3) 4 [3673] 3673 root510Tail2
    (if 3 ≤ root510Candidates1.length then search edge 3 [3673, 4759] root510Candidates1 8432 winner else winner) = _
  rw [if_neg (by decide)]
def root510Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root510Candidates2_checked : candidates edge 4 3673 5581 root510Tail3 winner.bound = root510Candidates2 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root510Step2 : scan edge (search edge 3) 4 [3673] 3673 root510Tail2 winner =
    scan edge (search edge 3) 4 [3673] 3673 root510Tail3 winner := by
  rw [root510Tail2, scan, if_neg (by decide), root510Candidates2_checked]
  change scan edge (search edge 3) 4 [3673] 3673 root510Tail3
    (if 3 ≤ root510Candidates2.length then search edge 3 [3673, 5581] root510Candidates2 9254 winner else winner) = _
  rw [if_neg (by decide)]
theorem root510Scan3 : scan edge (search edge 3) 4 [3673] 3673 root510Tail3 winner = winner := by rw [root510Tail3, scan, if_pos (by decide)]
theorem root510Scan2 : scan edge (search edge 3) 4 [3673] 3673 root510Tail2 winner = winner :=
  (root510Step2).trans root510Scan3
theorem root510Scan1 : scan edge (search edge 3) 4 [3673] 3673 root510Tail1 winner = winner :=
  (root510Step1).trans root510Scan2
theorem root510Scan0 : scan edge (search edge 3) 4 [3673] 3673 root510Tail0 winner = winner :=
  (root510Step0).trans root510Scan1
set_option maxRecDepth 4096 in
theorem root510Branch : search edge 4 [3673] root510Candidates 3673 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3673] 3673 root510Tail0 winner = winner
  exact root510Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
