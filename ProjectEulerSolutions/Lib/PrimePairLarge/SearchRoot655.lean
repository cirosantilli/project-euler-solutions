import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root655Candidates : List Nat := Data.row4931
set_option maxRecDepth 8192 in
theorem root655Candidates_checked : candidates edge 5 0 4931 tail656 winner.bound = root655Candidates := by
  apply candidates_eq_row edge 5 0 4931 tail656 root655Candidates winner.bound tail656_ordered
  · decide
  · decide
  · funext q
    rfl
def root655Tail32 : List Nat := []
def root655Tail31 : List Nat := 20333 :: root655Tail32
def root655Tail30 : List Nat := 19421 :: root655Tail31
def root655Tail29 : List Nat := 19031 :: root655Tail30
def root655Tail28 : List Nat := 18701 :: root655Tail29
def root655Tail27 : List Nat := 17903 :: root655Tail28
def root655Tail26 : List Nat := 17747 :: root655Tail27
def root655Tail25 : List Nat := 16223 :: root655Tail26
def root655Tail24 : List Nat := 15383 :: root655Tail25
def root655Tail23 : List Nat := 15137 :: root655Tail24
def root655Tail22 : List Nat := 15017 :: root655Tail23
def root655Tail21 : List Nat := 14621 :: root655Tail22
def root655Tail20 : List Nat := 14327 :: root655Tail21
def root655Tail19 : List Nat := 13487 :: root655Tail20
def root655Tail18 : List Nat := 13457 :: root655Tail19
def root655Tail17 : List Nat := 13337 :: root655Tail18
def root655Tail16 : List Nat := 12953 :: root655Tail17
def root655Tail15 : List Nat := 11933 :: root655Tail16
def root655Tail14 : List Nat := 11681 :: root655Tail15
def root655Tail13 : List Nat := 11471 :: root655Tail14
def root655Tail12 : List Nat := 10973 :: root655Tail13
def root655Tail11 : List Nat := 10253 :: root655Tail12
def root655Tail10 : List Nat := 9833 :: root655Tail11
def root655Tail9 : List Nat := 9551 :: root655Tail10
def root655Tail8 : List Nat := 8501 :: root655Tail9
def root655Tail7 : List Nat := 7817 :: root655Tail8
def root655Tail6 : List Nat := 7703 :: root655Tail7
def root655Tail5 : List Nat := 7349 :: root655Tail6
def root655Tail4 : List Nat := 7109 :: root655Tail5
def root655Tail3 : List Nat := 6551 :: root655Tail4
def root655Tail2 : List Nat := 5843 :: root655Tail3
def root655Tail1 : List Nat := 5309 :: root655Tail2
def root655Tail0 : List Nat := 5009 :: root655Tail1
def root655Candidates0 : List Nat := [7109, 8501]
set_option maxRecDepth 8192 in
theorem root655Candidates0_checked : candidates edge 4 4931 5009 root655Tail1 winner.bound = root655Candidates0 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root655Step0 : scan edge (search edge 3) 4 [4931] 4931 root655Tail0 winner =
    scan edge (search edge 3) 4 [4931] 4931 root655Tail1 winner := by
  rw [root655Tail0, scan, if_neg (by decide), root655Candidates0_checked]
  change scan edge (search edge 3) 4 [4931] 4931 root655Tail1
    (if 3 ≤ root655Candidates0.length then search edge 3 [4931, 5009] root655Candidates0 9940 winner else winner) = _
  rw [if_neg (by decide)]
theorem root655Scan1 : scan edge (search edge 3) 4 [4931] 4931 root655Tail1 winner = winner := by rw [root655Tail1, scan, if_pos (by decide)]
theorem root655Scan0 : scan edge (search edge 3) 4 [4931] 4931 root655Tail0 winner = winner :=
  (root655Step0).trans root655Scan1
set_option maxRecDepth 4096 in
theorem root655Branch : search edge 4 [4931] root655Candidates 4931 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4931] 4931 root655Tail0 winner = winner
  exact root655Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
