import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root601Candidates : List Nat := Data.row4447
set_option maxRecDepth 8192 in
theorem root601Candidates_checked : candidates edge 5 0 4447 tail602 winner.bound = root601Candidates := by
  apply candidates_eq_row edge 5 0 4447 tail602 root601Candidates winner.bound tail602_ordered
  · decide
  · decide
  · funext q
    rfl
def root601Tail30 : List Nat := []
def root601Tail29 : List Nat := 20407 :: root601Tail30
def root601Tail28 : List Nat := 20047 :: root601Tail29
def root601Tail27 : List Nat := 19441 :: root601Tail28
def root601Tail26 : List Nat := 19069 :: root601Tail27
def root601Tail25 : List Nat := 18787 :: root601Tail26
def root601Tail24 : List Nat := 18517 :: root601Tail25
def root601Tail23 : List Nat := 18289 :: root601Tail24
def root601Tail22 : List Nat := 17989 :: root601Tail23
def root601Tail21 : List Nat := 17761 :: root601Tail22
def root601Tail20 : List Nat := 17239 :: root601Tail21
def root601Tail19 : List Nat := 15259 :: root601Tail20
def root601Tail18 : List Nat := 15031 :: root601Tail19
def root601Tail17 : List Nat := 14419 :: root601Tail18
def root601Tail16 : List Nat := 12703 :: root601Tail17
def root601Tail15 : List Nat := 12553 :: root601Tail16
def root601Tail14 : List Nat := 11941 :: root601Tail15
def root601Tail13 : List Nat := 11677 :: root601Tail14
def root601Tail12 : List Nat := 10831 :: root601Tail13
def root601Tail11 : List Nat := 9811 :: root601Tail12
def root601Tail10 : List Nat := 8923 :: root601Tail11
def root601Tail9 : List Nat := 8839 :: root601Tail10
def root601Tail8 : List Nat := 8293 :: root601Tail9
def root601Tail7 : List Nat := 8101 :: root601Tail8
def root601Tail6 : List Nat := 7207 :: root601Tail7
def root601Tail5 : List Nat := 7057 :: root601Tail6
def root601Tail4 : List Nat := 6427 :: root601Tail5
def root601Tail3 : List Nat := 6067 :: root601Tail4
def root601Tail2 : List Nat := 5419 :: root601Tail3
def root601Tail1 : List Nat := 5023 :: root601Tail2
def root601Tail0 : List Nat := 4789 :: root601Tail1
def root601Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root601Candidates0_checked : candidates edge 4 4447 4789 root601Tail1 winner.bound = root601Candidates0 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root601Step0 : scan edge (search edge 3) 4 [4447] 4447 root601Tail0 winner =
    scan edge (search edge 3) 4 [4447] 4447 root601Tail1 winner := by
  rw [root601Tail0, scan, if_neg (by decide), root601Candidates0_checked]
  change scan edge (search edge 3) 4 [4447] 4447 root601Tail1
    (if 3 ≤ root601Candidates0.length then search edge 3 [4447, 4789] root601Candidates0 9236 winner else winner) = _
  rw [if_neg (by decide)]
def root601Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root601Candidates1_checked : candidates edge 4 4447 5023 root601Tail2 winner.bound = root601Candidates1 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root601Step1 : scan edge (search edge 3) 4 [4447] 4447 root601Tail1 winner =
    scan edge (search edge 3) 4 [4447] 4447 root601Tail2 winner := by
  rw [root601Tail1, scan, if_neg (by decide), root601Candidates1_checked]
  change scan edge (search edge 3) 4 [4447] 4447 root601Tail2
    (if 3 ≤ root601Candidates1.length then search edge 3 [4447, 5023] root601Candidates1 9470 winner else winner) = _
  rw [if_neg (by decide)]
theorem root601Scan2 : scan edge (search edge 3) 4 [4447] 4447 root601Tail2 winner = winner := by rw [root601Tail2, scan, if_pos (by decide)]
theorem root601Scan1 : scan edge (search edge 3) 4 [4447] 4447 root601Tail1 winner = winner :=
  (root601Step1).trans root601Scan2
theorem root601Scan0 : scan edge (search edge 3) 4 [4447] 4447 root601Tail0 winner = winner :=
  (root601Step0).trans root601Scan1
set_option maxRecDepth 4096 in
theorem root601Branch : search edge 4 [4447] root601Candidates 4447 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4447] 4447 root601Tail0 winner = winner
  exact root601Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
