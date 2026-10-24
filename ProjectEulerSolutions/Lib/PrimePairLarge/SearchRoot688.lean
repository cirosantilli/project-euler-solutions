import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root688Candidates : List Nat := Data.row5189
set_option maxRecDepth 8192 in
theorem root688Candidates_checked : candidates edge 5 0 5189 tail689 winner.bound = root688Candidates := by
  apply candidates_eq_row edge 5 0 5189 tail689 root688Candidates winner.bound tail689_ordered
  · decide
  · decide
  · funext q
    rfl
def root688Tail27 : List Nat := []
def root688Tail26 : List Nat := 20063 :: root688Tail27
def root688Tail25 : List Nat := 19727 :: root688Tail26
def root688Tail24 : List Nat := 19553 :: root688Tail25
def root688Tail23 : List Nat := 18731 :: root688Tail24
def root688Tail22 : List Nat := 18089 :: root688Tail23
def root688Tail21 : List Nat := 17327 :: root688Tail22
def root688Tail20 : List Nat := 17291 :: root688Tail21
def root688Tail19 : List Nat := 16889 :: root688Tail20
def root688Tail18 : List Nat := 16553 :: root688Tail19
def root688Tail17 : List Nat := 16529 :: root688Tail18
def root688Tail16 : List Nat := 15671 :: root688Tail17
def root688Tail15 : List Nat := 15461 :: root688Tail16
def root688Tail14 : List Nat := 15017 :: root688Tail15
def root688Tail13 : List Nat := 14813 :: root688Tail14
def root688Tail12 : List Nat := 14741 :: root688Tail13
def root688Tail11 : List Nat := 14723 :: root688Tail12
def root688Tail10 : List Nat := 12671 :: root688Tail11
def root688Tail9 : List Nat := 12473 :: root688Tail10
def root688Tail8 : List Nat := 11909 :: root688Tail9
def root688Tail7 : List Nat := 10331 :: root688Tail8
def root688Tail6 : List Nat := 10313 :: root688Tail7
def root688Tail5 : List Nat := 9539 :: root688Tail6
def root688Tail4 : List Nat := 9377 :: root688Tail5
def root688Tail3 : List Nat := 9209 :: root688Tail4
def root688Tail2 : List Nat := 8573 :: root688Tail3
def root688Tail1 : List Nat := 6491 :: root688Tail2
def root688Tail0 : List Nat := 6329 :: root688Tail1
theorem root688Scan0 : scan edge (search edge 3) 4 [5189] 5189 root688Tail0 winner = winner := by rw [root688Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root688Branch : search edge 4 [5189] root688Candidates 5189 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5189] 5189 root688Tail0 winner = winner
  exact root688Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
