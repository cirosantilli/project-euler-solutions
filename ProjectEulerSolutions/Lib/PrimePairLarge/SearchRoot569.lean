import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root569Candidates : List Nat := Data.row4157
set_option maxRecDepth 8192 in
theorem root569Candidates_checked : candidates edge 5 0 4157 tail570 winner.bound = root569Candidates := by
  apply candidates_eq_row edge 5 0 4157 tail570 root569Candidates winner.bound tail570_ordered
  · decide
  · decide
  · funext q
    rfl
def root569Tail23 : List Nat := []
def root569Tail22 : List Nat := 21773 :: root569Tail23
def root569Tail21 : List Nat := 21701 :: root569Tail22
def root569Tail20 : List Nat := 21599 :: root569Tail21
def root569Tail19 : List Nat := 21221 :: root569Tail20
def root569Tail18 : List Nat := 20693 :: root569Tail19
def root569Tail17 : List Nat := 19211 :: root569Tail18
def root569Tail16 : List Nat := 17807 :: root569Tail17
def root569Tail15 : List Nat := 16547 :: root569Tail16
def root569Tail14 : List Nat := 15803 :: root569Tail15
def root569Tail13 : List Nat := 14771 :: root569Tail14
def root569Tail12 : List Nat := 14657 :: root569Tail13
def root569Tail11 : List Nat := 12953 :: root569Tail12
def root569Tail10 : List Nat := 11159 :: root569Tail11
def root569Tail9 : List Nat := 10259 :: root569Tail10
def root569Tail8 : List Nat := 9497 :: root569Tail9
def root569Tail7 : List Nat := 8741 :: root569Tail8
def root569Tail6 : List Nat := 7577 :: root569Tail7
def root569Tail5 : List Nat := 7433 :: root569Tail6
def root569Tail4 : List Nat := 7127 :: root569Tail5
def root569Tail3 : List Nat := 7043 :: root569Tail4
def root569Tail2 : List Nat := 6947 :: root569Tail3
def root569Tail1 : List Nat := 5717 :: root569Tail2
def root569Tail0 : List Nat := 5333 :: root569Tail1
def root569Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root569Candidates0_checked : candidates edge 4 4157 5333 root569Tail1 winner.bound = root569Candidates0 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root569Step0 : scan edge (search edge 3) 4 [4157] 4157 root569Tail0 winner =
    scan edge (search edge 3) 4 [4157] 4157 root569Tail1 winner := by
  rw [root569Tail0, scan, if_neg (by decide), root569Candidates0_checked]
  change scan edge (search edge 3) 4 [4157] 4157 root569Tail1
    (if 3 ≤ root569Candidates0.length then search edge 3 [4157, 5333] root569Candidates0 9490 winner else winner) = _
  rw [if_neg (by decide)]
theorem root569Scan1 : scan edge (search edge 3) 4 [4157] 4157 root569Tail1 winner = winner := by rw [root569Tail1, scan, if_pos (by decide)]
theorem root569Scan0 : scan edge (search edge 3) 4 [4157] 4157 root569Tail0 winner = winner :=
  (root569Step0).trans root569Scan1
set_option maxRecDepth 4096 in
theorem root569Branch : search edge 4 [4157] root569Candidates 4157 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4157] 4157 root569Tail0 winner = winner
  exact root569Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
