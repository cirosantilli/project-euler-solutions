import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root678Candidates : List Nat := Data.row5099
set_option maxRecDepth 8192 in
theorem root678Candidates_checked : candidates edge 5 0 5099 tail679 winner.bound = root678Candidates := by
  apply candidates_eq_row edge 5 0 5099 tail679 root678Candidates winner.bound tail679_ordered
  · decide
  · decide
  · funext q
    rfl
def root678Tail26 : List Nat := []
def root678Tail25 : List Nat := 19841 :: root678Tail26
def root678Tail24 : List Nat := 17909 :: root678Tail25
def root678Tail23 : List Nat := 17477 :: root678Tail24
def root678Tail22 : List Nat := 17351 :: root678Tail23
def root678Tail21 : List Nat := 17189 :: root678Tail22
def root678Tail20 : List Nat := 14747 :: root678Tail21
def root678Tail19 : List Nat := 14621 :: root678Tail20
def root678Tail18 : List Nat := 13829 :: root678Tail19
def root678Tail17 : List Nat := 13619 :: root678Tail18
def root678Tail16 : List Nat := 13499 :: root678Tail17
def root678Tail15 : List Nat := 11939 :: root678Tail16
def root678Tail14 : List Nat := 9587 :: root678Tail15
def root678Tail13 : List Nat := 9467 :: root678Tail14
def root678Tail12 : List Nat := 9323 :: root678Tail13
def root678Tail11 : List Nat := 8663 :: root678Tail12
def root678Tail10 : List Nat := 8273 :: root678Tail11
def root678Tail9 : List Nat := 8231 :: root678Tail10
def root678Tail8 : List Nat := 7949 :: root678Tail9
def root678Tail7 : List Nat := 7853 :: root678Tail8
def root678Tail6 : List Nat := 7649 :: root678Tail7
def root678Tail5 : List Nat := 7523 :: root678Tail6
def root678Tail4 : List Nat := 7151 :: root678Tail5
def root678Tail3 : List Nat := 6857 :: root678Tail4
def root678Tail2 : List Nat := 5507 :: root678Tail3
def root678Tail1 : List Nat := 5501 :: root678Tail2
def root678Tail0 : List Nat := 5471 :: root678Tail1
theorem root678Scan0 : scan edge (search edge 3) 4 [5099] 5099 root678Tail0 winner = winner := by rw [root678Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root678Branch : search edge 4 [5099] root678Candidates 5099 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5099] 5099 root678Tail0 winner = winner
  exact root678Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
