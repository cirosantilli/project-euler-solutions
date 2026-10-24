import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root609Candidates : List Nat := Data.row4513
set_option maxRecDepth 8192 in
theorem root609Candidates_checked : candidates edge 5 0 4513 tail610 winner.bound = root609Candidates := by
  apply candidates_eq_row edge 5 0 4513 tail610 root609Candidates winner.bound tail610_ordered
  · decide
  · decide
  · funext q
    rfl
def root609Tail39 : List Nat := []
def root609Tail38 : List Nat := 21187 :: root609Tail39
def root609Tail37 : List Nat := 20929 :: root609Tail38
def root609Tail36 : List Nat := 20161 :: root609Tail37
def root609Tail35 : List Nat := 19993 :: root609Tail36
def root609Tail34 : List Nat := 19867 :: root609Tail35
def root609Tail33 : List Nat := 19717 :: root609Tail34
def root609Tail32 : List Nat := 17341 :: root609Tail33
def root609Tail31 : List Nat := 16963 :: root609Tail32
def root609Tail30 : List Nat := 15937 :: root609Tail31
def root609Tail29 : List Nat := 15583 :: root609Tail30
def root609Tail28 : List Nat := 15259 :: root609Tail29
def root609Tail27 : List Nat := 12637 :: root609Tail28
def root609Tail26 : List Nat := 12487 :: root609Tail27
def root609Tail25 : List Nat := 12409 :: root609Tail26
def root609Tail24 : List Nat := 11689 :: root609Tail25
def root609Tail23 : List Nat := 11551 :: root609Tail24
def root609Tail22 : List Nat := 11353 :: root609Tail23
def root609Tail21 : List Nat := 10771 :: root609Tail22
def root609Tail20 : List Nat := 10687 :: root609Tail21
def root609Tail19 : List Nat := 10597 :: root609Tail20
def root609Tail18 : List Nat := 10333 :: root609Tail19
def root609Tail17 : List Nat := 10099 :: root609Tail18
def root609Tail16 : List Nat := 10039 :: root609Tail17
def root609Tail15 : List Nat := 9781 :: root609Tail16
def root609Tail14 : List Nat := 9697 :: root609Tail15
def root609Tail13 : List Nat := 9613 :: root609Tail14
def root609Tail12 : List Nat := 9187 :: root609Tail13
def root609Tail11 : List Nat := 9151 :: root609Tail12
def root609Tail10 : List Nat := 9049 :: root609Tail11
def root609Tail9 : List Nat := 8707 :: root609Tail10
def root609Tail8 : List Nat := 8287 :: root609Tail9
def root609Tail7 : List Nat := 7723 :: root609Tail8
def root609Tail6 : List Nat := 6841 :: root609Tail7
def root609Tail5 : List Nat := 6397 :: root609Tail6
def root609Tail4 : List Nat := 6277 :: root609Tail5
def root609Tail3 : List Nat := 6163 :: root609Tail4
def root609Tail2 : List Nat := 5581 :: root609Tail3
def root609Tail1 : List Nat := 5437 :: root609Tail2
def root609Tail0 : List Nat := 4591 :: root609Tail1
def root609Candidates0 : List Nat := [8287, 11353, 12487]
set_option maxRecDepth 8192 in
theorem root609Candidates0_checked : candidates edge 4 4513 4591 root609Tail1 winner.bound = root609Candidates0 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root609Child0 : search edge 3 [4513, 4591] root609Candidates0 9104 winner = winner := by decide
theorem root609Step0 : scan edge (search edge 3) 4 [4513] 4513 root609Tail0 winner =
    scan edge (search edge 3) 4 [4513] 4513 root609Tail1 winner := by
  rw [root609Tail0, scan, if_neg (by decide), root609Candidates0_checked]
  change scan edge (search edge 3) 4 [4513] 4513 root609Tail1
    (if 3 ≤ root609Candidates0.length then search edge 3 [4513, 4591] root609Candidates0 9104 winner else winner) = _
  rw [if_pos (by decide), root609Child0]
theorem root609Scan1 : scan edge (search edge 3) 4 [4513] 4513 root609Tail1 winner = winner := by rw [root609Tail1, scan, if_pos (by decide)]
theorem root609Scan0 : scan edge (search edge 3) 4 [4513] 4513 root609Tail0 winner = winner :=
  (root609Step0).trans root609Scan1
set_option maxRecDepth 4096 in
theorem root609Branch : search edge 4 [4513] root609Candidates 4513 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4513] 4513 root609Tail0 winner = winner
  exact root609Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
