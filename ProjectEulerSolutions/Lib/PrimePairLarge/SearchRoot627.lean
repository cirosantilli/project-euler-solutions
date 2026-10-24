import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root627Candidates : List Nat := Data.row4657
set_option maxRecDepth 8192 in
theorem root627Candidates_checked : candidates edge 5 0 4657 tail628 winner.bound = root627Candidates := by
  apply candidates_eq_row edge 5 0 4657 tail628 root627Candidates winner.bound tail628_ordered
  · decide
  · decide
  · funext q
    rfl
def root627Tail31 : List Nat := []
def root627Tail30 : List Nat := 20743 :: root627Tail31
def root627Tail29 : List Nat := 20173 :: root627Tail30
def root627Tail28 : List Nat := 19867 :: root627Tail29
def root627Tail27 : List Nat := 19717 :: root627Tail28
def root627Tail26 : List Nat := 18013 :: root627Tail27
def root627Tail25 : List Nat := 17449 :: root627Tail26
def root627Tail24 : List Nat := 16963 :: root627Tail25
def root627Tail23 : List Nat := 16057 :: root627Tail24
def root627Tail22 : List Nat := 15307 :: root627Tail23
def root627Tail21 : List Nat := 14929 :: root627Tail22
def root627Tail20 : List Nat := 14797 :: root627Tail21
def root627Tail19 : List Nat := 14407 :: root627Tail20
def root627Tail18 : List Nat := 13903 :: root627Tail19
def root627Tail17 : List Nat := 13159 :: root627Tail18
def root627Tail16 : List Nat := 12853 :: root627Tail17
def root627Tail15 : List Nat := 11317 :: root627Tail16
def root627Tail14 : List Nat := 10837 :: root627Tail15
def root627Tail13 : List Nat := 10093 :: root627Tail14
def root627Tail12 : List Nat := 9433 :: root627Tail13
def root627Tail11 : List Nat := 9283 :: root627Tail12
def root627Tail10 : List Nat := 8971 :: root627Tail11
def root627Tail9 : List Nat := 8923 :: root627Tail10
def root627Tail8 : List Nat := 8737 :: root627Tail9
def root627Tail7 : List Nat := 8209 :: root627Tail8
def root627Tail6 : List Nat := 7933 :: root627Tail7
def root627Tail5 : List Nat := 7753 :: root627Tail6
def root627Tail4 : List Nat := 7243 :: root627Tail5
def root627Tail3 : List Nat := 7177 :: root627Tail4
def root627Tail2 : List Nat := 6217 :: root627Tail3
def root627Tail1 : List Nat := 5227 :: root627Tail2
def root627Tail0 : List Nat := 4789 :: root627Tail1
def root627Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root627Candidates0_checked : candidates edge 4 4657 4789 root627Tail1 winner.bound = root627Candidates0 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root627Step0 : scan edge (search edge 3) 4 [4657] 4657 root627Tail0 winner =
    scan edge (search edge 3) 4 [4657] 4657 root627Tail1 winner := by
  rw [root627Tail0, scan, if_neg (by decide), root627Candidates0_checked]
  change scan edge (search edge 3) 4 [4657] 4657 root627Tail1
    (if 3 ≤ root627Candidates0.length then search edge 3 [4657, 4789] root627Candidates0 9446 winner else winner) = _
  rw [if_neg (by decide)]
def root627Candidates1 : List Nat := [7243, 10093, 13159]
set_option maxRecDepth 8192 in
theorem root627Candidates1_checked : candidates edge 4 4657 5227 root627Tail2 winner.bound = root627Candidates1 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root627Child1 : search edge 3 [4657, 5227] root627Candidates1 9884 winner = winner := by decide
theorem root627Step1 : scan edge (search edge 3) 4 [4657] 4657 root627Tail1 winner =
    scan edge (search edge 3) 4 [4657] 4657 root627Tail2 winner := by
  rw [root627Tail1, scan, if_neg (by decide), root627Candidates1_checked]
  change scan edge (search edge 3) 4 [4657] 4657 root627Tail2
    (if 3 ≤ root627Candidates1.length then search edge 3 [4657, 5227] root627Candidates1 9884 winner else winner) = _
  rw [if_pos (by decide), root627Child1]
theorem root627Scan2 : scan edge (search edge 3) 4 [4657] 4657 root627Tail2 winner = winner := by rw [root627Tail2, scan, if_pos (by decide)]
theorem root627Scan1 : scan edge (search edge 3) 4 [4657] 4657 root627Tail1 winner = winner :=
  (root627Step1).trans root627Scan2
theorem root627Scan0 : scan edge (search edge 3) 4 [4657] 4657 root627Tail0 winner = winner :=
  (root627Step0).trans root627Scan1
set_option maxRecDepth 4096 in
theorem root627Branch : search edge 4 [4657] root627Candidates 4657 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4657] 4657 root627Tail0 winner = winner
  exact root627Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
