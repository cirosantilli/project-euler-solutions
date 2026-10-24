import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root559Candidates : List Nat := Data.row4079
set_option maxRecDepth 8192 in
theorem root559Candidates_checked : candidates edge 5 0 4079 tail560 winner.bound = root559Candidates := by
  apply candidates_eq_row edge 5 0 4079 tail560 root559Candidates winner.bound tail560_ordered
  · decide
  · decide
  · funext q
    rfl
def root559Tail25 : List Nat := []
def root559Tail24 : List Nat := 21383 :: root559Tail25
def root559Tail23 : List Nat := 21149 :: root559Tail24
def root559Tail22 : List Nat := 20147 :: root559Tail23
def root559Tail21 : List Nat := 19997 :: root559Tail22
def root559Tail20 : List Nat := 19661 :: root559Tail21
def root559Tail19 : List Nat := 18047 :: root559Tail20
def root559Tail18 : List Nat := 14537 :: root559Tail19
def root559Tail17 : List Nat := 13553 :: root559Tail18
def root559Tail16 : List Nat := 13151 :: root559Tail17
def root559Tail15 : List Nat := 12203 :: root559Tail16
def root559Tail14 : List Nat := 11069 :: root559Tail15
def root559Tail13 : List Nat := 10103 :: root559Tail14
def root559Tail12 : List Nat := 9743 :: root559Tail13
def root559Tail11 : List Nat := 9323 :: root559Tail12
def root559Tail10 : List Nat := 6863 :: root559Tail11
def root559Tail9 : List Nat := 6131 :: root559Tail10
def root559Tail8 : List Nat := 6053 :: root559Tail9
def root559Tail7 : List Nat := 6029 :: root559Tail8
def root559Tail6 : List Nat := 5927 :: root559Tail7
def root559Tail5 : List Nat := 5897 :: root559Tail6
def root559Tail4 : List Nat := 5879 :: root559Tail5
def root559Tail3 : List Nat := 5441 :: root559Tail4
def root559Tail2 : List Nat := 5309 :: root559Tail3
def root559Tail1 : List Nat := 5231 :: root559Tail2
def root559Tail0 : List Nat := 4973 :: root559Tail1
def root559Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root559Candidates0_checked : candidates edge 4 4079 4973 root559Tail1 winner.bound = root559Candidates0 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root559Step0 : scan edge (search edge 3) 4 [4079] 4079 root559Tail0 winner =
    scan edge (search edge 3) 4 [4079] 4079 root559Tail1 winner := by
  rw [root559Tail0, scan, if_neg (by decide), root559Candidates0_checked]
  change scan edge (search edge 3) 4 [4079] 4079 root559Tail1
    (if 3 ≤ root559Candidates0.length then search edge 3 [4079, 4973] root559Candidates0 9052 winner else winner) = _
  rw [if_neg (by decide)]
def root559Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root559Candidates1_checked : candidates edge 4 4079 5231 root559Tail2 winner.bound = root559Candidates1 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root559Step1 : scan edge (search edge 3) 4 [4079] 4079 root559Tail1 winner =
    scan edge (search edge 3) 4 [4079] 4079 root559Tail2 winner := by
  rw [root559Tail1, scan, if_neg (by decide), root559Candidates1_checked]
  change scan edge (search edge 3) 4 [4079] 4079 root559Tail2
    (if 3 ≤ root559Candidates1.length then search edge 3 [4079, 5231] root559Candidates1 9310 winner else winner) = _
  rw [if_neg (by decide)]
def root559Candidates2 : List Nat := [9743, 13151]
set_option maxRecDepth 8192 in
theorem root559Candidates2_checked : candidates edge 4 4079 5309 root559Tail3 winner.bound = root559Candidates2 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root559Step2 : scan edge (search edge 3) 4 [4079] 4079 root559Tail2 winner =
    scan edge (search edge 3) 4 [4079] 4079 root559Tail3 winner := by
  rw [root559Tail2, scan, if_neg (by decide), root559Candidates2_checked]
  change scan edge (search edge 3) 4 [4079] 4079 root559Tail3
    (if 3 ≤ root559Candidates2.length then search edge 3 [4079, 5309] root559Candidates2 9388 winner else winner) = _
  rw [if_neg (by decide)]
def root559Candidates3 : List Nat := [13151]
set_option maxRecDepth 8192 in
theorem root559Candidates3_checked : candidates edge 4 4079 5441 root559Tail4 winner.bound = root559Candidates3 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root559Step3 : scan edge (search edge 3) 4 [4079] 4079 root559Tail3 winner =
    scan edge (search edge 3) 4 [4079] 4079 root559Tail4 winner := by
  rw [root559Tail3, scan, if_neg (by decide), root559Candidates3_checked]
  change scan edge (search edge 3) 4 [4079] 4079 root559Tail4
    (if 3 ≤ root559Candidates3.length then search edge 3 [4079, 5441] root559Candidates3 9520 winner else winner) = _
  rw [if_neg (by decide)]
theorem root559Scan4 : scan edge (search edge 3) 4 [4079] 4079 root559Tail4 winner = winner := by rw [root559Tail4, scan, if_pos (by decide)]
theorem root559Scan3 : scan edge (search edge 3) 4 [4079] 4079 root559Tail3 winner = winner :=
  (root559Step3).trans root559Scan4
theorem root559Scan2 : scan edge (search edge 3) 4 [4079] 4079 root559Tail2 winner = winner :=
  (root559Step2).trans root559Scan3
theorem root559Scan1 : scan edge (search edge 3) 4 [4079] 4079 root559Tail1 winner = winner :=
  (root559Step1).trans root559Scan2
theorem root559Scan0 : scan edge (search edge 3) 4 [4079] 4079 root559Tail0 winner = winner :=
  (root559Step0).trans root559Scan1
set_option maxRecDepth 4096 in
theorem root559Branch : search edge 4 [4079] root559Candidates 4079 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4079] 4079 root559Tail0 winner = winner
  exact root559Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
