import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root675Candidates : List Nat := Data.row5077
set_option maxRecDepth 8192 in
theorem root675Candidates_checked : candidates edge 5 0 5077 tail676 winner.bound = root675Candidates := by
  apply candidates_eq_row edge 5 0 5077 tail676 root675Candidates winner.bound tail676_ordered
  · decide
  · decide
  · funext q
    rfl
def root675Tail25 : List Nat := []
def root675Tail24 : List Nat := 20809 :: root675Tail25
def root675Tail23 : List Nat := 20749 :: root675Tail24
def root675Tail22 : List Nat := 20599 :: root675Tail23
def root675Tail21 : List Nat := 19507 :: root675Tail22
def root675Tail20 : List Nat := 18493 :: root675Tail21
def root675Tail19 : List Nat := 18223 :: root675Tail20
def root675Tail18 : List Nat := 16843 :: root675Tail19
def root675Tail17 : List Nat := 16633 :: root675Tail18
def root675Tail16 : List Nat := 16267 :: root675Tail17
def root675Tail15 : List Nat := 13879 :: root675Tail16
def root675Tail14 : List Nat := 13267 :: root675Tail15
def root675Tail13 : List Nat := 13159 :: root675Tail14
def root675Tail12 : List Nat := 11551 :: root675Tail13
def root675Tail11 : List Nat := 10663 :: root675Tail12
def root675Tail10 : List Nat := 10459 :: root675Tail11
def root675Tail9 : List Nat := 10141 :: root675Tail10
def root675Tail8 : List Nat := 9907 :: root675Tail9
def root675Tail7 : List Nat := 7873 :: root675Tail8
def root675Tail6 : List Nat := 7687 :: root675Tail7
def root675Tail5 : List Nat := 7459 :: root675Tail6
def root675Tail4 : List Nat := 6763 :: root675Tail5
def root675Tail3 : List Nat := 6367 :: root675Tail4
def root675Tail2 : List Nat := 6091 :: root675Tail3
def root675Tail1 : List Nat := 6043 :: root675Tail2
def root675Tail0 : List Nat := 5779 :: root675Tail1
theorem root675Scan0 : scan edge (search edge 3) 4 [5077] 5077 root675Tail0 winner = winner := by rw [root675Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root675Branch : search edge 4 [5077] root675Candidates 5077 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5077] 5077 root675Tail0 winner = winner
  exact root675Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
