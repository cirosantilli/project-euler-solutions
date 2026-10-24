import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root429Candidates : List Nat := Data.row3011
set_option maxRecDepth 8192 in
theorem root429Candidates_checked : candidates edge 5 0 3011 tail430 winner.bound = root429Candidates := by
  apply candidates_eq_row edge 5 0 3011 tail430 root429Candidates winner.bound tail430_ordered
  · decide
  · decide
  · funext q
    rfl
def root429Tail29 : List Nat := []
def root429Tail28 : List Nat := 22079 :: root429Tail29
def root429Tail27 : List Nat := 21767 :: root429Tail28
def root429Tail26 : List Nat := 21713 :: root429Tail27
def root429Tail25 : List Nat := 21347 :: root429Tail26
def root429Tail24 : List Nat := 20747 :: root429Tail25
def root429Tail23 : List Nat := 20231 :: root429Tail24
def root429Tail22 : List Nat := 19037 :: root429Tail23
def root429Tail21 : List Nat := 18899 :: root429Tail22
def root429Tail20 : List Nat := 17021 :: root429Tail21
def root429Tail19 : List Nat := 15791 :: root429Tail20
def root429Tail18 : List Nat := 14207 :: root429Tail19
def root429Tail17 : List Nat := 12923 :: root429Tail18
def root429Tail16 : List Nat := 11243 :: root429Tail17
def root429Tail15 : List Nat := 10151 :: root429Tail16
def root429Tail14 : List Nat := 9437 :: root429Tail15
def root429Tail13 : List Nat := 8369 :: root429Tail14
def root429Tail12 : List Nat := 8147 :: root429Tail13
def root429Tail11 : List Nat := 8009 :: root429Tail12
def root429Tail10 : List Nat := 7937 :: root429Tail11
def root429Tail9 : List Nat := 7883 :: root429Tail10
def root429Tail8 : List Nat := 7349 :: root429Tail9
def root429Tail7 : List Nat := 7127 :: root429Tail8
def root429Tail6 : List Nat := 6833 :: root429Tail7
def root429Tail5 : List Nat := 6353 :: root429Tail6
def root429Tail4 : List Nat := 6329 :: root429Tail5
def root429Tail3 : List Nat := 4817 :: root429Tail4
def root429Tail2 : List Nat := 4493 :: root429Tail3
def root429Tail1 : List Nat := 3329 :: root429Tail2
def root429Tail0 : List Nat := 3023 :: root429Tail1
def root429Candidates0 : List Nat := [9437]
set_option maxRecDepth 8192 in
theorem root429Candidates0_checked : candidates edge 4 3011 3023 root429Tail1 winner.bound = root429Candidates0 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root429Step0 : scan edge (search edge 3) 4 [3011] 3011 root429Tail0 winner =
    scan edge (search edge 3) 4 [3011] 3011 root429Tail1 winner := by
  rw [root429Tail0, scan, if_neg (by decide), root429Candidates0_checked]
  change scan edge (search edge 3) 4 [3011] 3011 root429Tail1
    (if 3 ≤ root429Candidates0.length then search edge 3 [3011, 3023] root429Candidates0 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root429Candidates1 : List Nat := [9437]
set_option maxRecDepth 8192 in
theorem root429Candidates1_checked : candidates edge 4 3011 3329 root429Tail2 winner.bound = root429Candidates1 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root429Step1 : scan edge (search edge 3) 4 [3011] 3011 root429Tail1 winner =
    scan edge (search edge 3) 4 [3011] 3011 root429Tail2 winner := by
  rw [root429Tail1, scan, if_neg (by decide), root429Candidates1_checked]
  change scan edge (search edge 3) 4 [3011] 3011 root429Tail2
    (if 3 ≤ root429Candidates1.length then search edge 3 [3011, 3329] root429Candidates1 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root429Candidates2 : List Nat := [7349, 10151, 11243]
set_option maxRecDepth 8192 in
theorem root429Candidates2_checked : candidates edge 4 3011 4493 root429Tail3 winner.bound = root429Candidates2 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root429Child2 : search edge 3 [3011, 4493] root429Candidates2 7504 winner = winner := by decide
theorem root429Step2 : scan edge (search edge 3) 4 [3011] 3011 root429Tail2 winner =
    scan edge (search edge 3) 4 [3011] 3011 root429Tail3 winner := by
  rw [root429Tail2, scan, if_neg (by decide), root429Candidates2_checked]
  change scan edge (search edge 3) 4 [3011] 3011 root429Tail3
    (if 3 ≤ root429Candidates2.length then search edge 3 [3011, 4493] root429Candidates2 7504 winner else winner) = _
  rw [if_pos (by decide), root429Child2]
def root429Candidates3 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root429Candidates3_checked : candidates edge 4 3011 4817 root429Tail4 winner.bound = root429Candidates3 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root429Step3 : scan edge (search edge 3) 4 [3011] 3011 root429Tail3 winner =
    scan edge (search edge 3) 4 [3011] 3011 root429Tail4 winner := by
  rw [root429Tail3, scan, if_neg (by decide), root429Candidates3_checked]
  change scan edge (search edge 3) 4 [3011] 3011 root429Tail4
    (if 3 ≤ root429Candidates3.length then search edge 3 [3011, 4817] root429Candidates3 7828 winner else winner) = _
  rw [if_neg (by decide)]
theorem root429Scan4 : scan edge (search edge 3) 4 [3011] 3011 root429Tail4 winner = winner := by rw [root429Tail4, scan, if_pos (by decide)]
theorem root429Scan3 : scan edge (search edge 3) 4 [3011] 3011 root429Tail3 winner = winner :=
  (root429Step3).trans root429Scan4
theorem root429Scan2 : scan edge (search edge 3) 4 [3011] 3011 root429Tail2 winner = winner :=
  (root429Step2).trans root429Scan3
theorem root429Scan1 : scan edge (search edge 3) 4 [3011] 3011 root429Tail1 winner = winner :=
  (root429Step1).trans root429Scan2
theorem root429Scan0 : scan edge (search edge 3) 4 [3011] 3011 root429Tail0 winner = winner :=
  (root429Step0).trans root429Scan1
set_option maxRecDepth 4096 in
theorem root429Branch : search edge 4 [3011] root429Candidates 3011 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3011] 3011 root429Tail0 winner = winner
  exact root429Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
