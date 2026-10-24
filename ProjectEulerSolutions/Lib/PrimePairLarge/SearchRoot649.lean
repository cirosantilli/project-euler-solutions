import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root649Candidates : List Nat := Data.row4871
set_option maxRecDepth 8192 in
theorem root649Candidates_checked : candidates edge 5 0 4871 tail650 winner.bound = root649Candidates := by
  apply candidates_eq_row edge 5 0 4871 tail650 root649Candidates winner.bound tail650_ordered
  · decide
  · decide
  · funext q
    rfl
def root649Tail36 : List Nat := []
def root649Tail35 : List Nat := 21101 :: root649Tail36
def root649Tail34 : List Nat := 20507 :: root649Tail35
def root649Tail33 : List Nat := 19727 :: root649Tail34
def root649Tail32 : List Nat := 19463 :: root649Tail33
def root649Tail31 : List Nat := 19403 :: root649Tail32
def root649Tail30 : List Nat := 17783 :: root649Tail31
def root649Tail29 : List Nat := 17333 :: root649Tail30
def root649Tail28 : List Nat := 16937 :: root649Tail29
def root649Tail27 : List Nat := 15287 :: root649Tail28
def root649Tail26 : List Nat := 14057 :: root649Tail27
def root649Tail25 : List Nat := 13997 :: root649Tail26
def root649Tail24 : List Nat := 13877 :: root649Tail25
def root649Tail23 : List Nat := 13553 :: root649Tail24
def root649Tail22 : List Nat := 12401 :: root649Tail23
def root649Tail21 : List Nat := 12071 :: root649Tail22
def root649Tail20 : List Nat := 11423 :: root649Tail21
def root649Tail19 : List Nat := 10937 :: root649Tail20
def root649Tail18 : List Nat := 10853 :: root649Tail19
def root649Tail17 : List Nat := 10259 :: root649Tail18
def root649Tail16 : List Nat := 10181 :: root649Tail17
def root649Tail15 : List Nat := 9887 :: root649Tail16
def root649Tail14 : List Nat := 9833 :: root649Tail15
def root649Tail13 : List Nat := 9791 :: root649Tail14
def root649Tail12 : List Nat := 8861 :: root649Tail13
def root649Tail11 : List Nat := 8627 :: root649Tail12
def root649Tail10 : List Nat := 8429 :: root649Tail11
def root649Tail9 : List Nat := 8237 :: root649Tail10
def root649Tail8 : List Nat := 8123 :: root649Tail9
def root649Tail7 : List Nat := 8069 :: root649Tail8
def root649Tail6 : List Nat := 7673 :: root649Tail7
def root649Tail5 : List Nat := 7649 :: root649Tail6
def root649Tail4 : List Nat := 6959 :: root649Tail5
def root649Tail3 : List Nat := 6947 :: root649Tail4
def root649Tail2 : List Nat := 6719 :: root649Tail3
def root649Tail1 : List Nat := 6389 :: root649Tail2
def root649Tail0 : List Nat := 5237 :: root649Tail1
def root649Candidates0 : List Nat := [10937, 12071]
set_option maxRecDepth 8192 in
theorem root649Candidates0_checked : candidates edge 4 4871 5237 root649Tail1 winner.bound = root649Candidates0 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root649Step0 : scan edge (search edge 3) 4 [4871] 4871 root649Tail0 winner =
    scan edge (search edge 3) 4 [4871] 4871 root649Tail1 winner := by
  rw [root649Tail0, scan, if_neg (by decide), root649Candidates0_checked]
  change scan edge (search edge 3) 4 [4871] 4871 root649Tail1
    (if 3 ≤ root649Candidates0.length then search edge 3 [4871, 5237] root649Candidates0 10108 winner else winner) = _
  rw [if_neg (by decide)]
theorem root649Scan1 : scan edge (search edge 3) 4 [4871] 4871 root649Tail1 winner = winner := by rw [root649Tail1, scan, if_pos (by decide)]
theorem root649Scan0 : scan edge (search edge 3) 4 [4871] 4871 root649Tail0 winner = winner :=
  (root649Step0).trans root649Scan1
set_option maxRecDepth 4096 in
theorem root649Branch : search edge 4 [4871] root649Candidates 4871 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4871] 4871 root649Tail0 winner = winner
  exact root649Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
