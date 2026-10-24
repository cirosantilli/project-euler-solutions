import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root670Candidates : List Nat := Data.row5021
set_option maxRecDepth 8192 in
theorem root670Candidates_checked : candidates edge 5 0 5021 tail671 winner.bound = root670Candidates := by
  apply candidates_eq_row edge 5 0 5021 tail671 root670Candidates winner.bound tail671_ordered
  · decide
  · decide
  · funext q
    rfl
def root670Tail22 : List Nat := []
def root670Tail21 : List Nat := 20897 :: root670Tail22
def root670Tail20 : List Nat := 19997 :: root670Tail21
def root670Tail19 : List Nat := 19697 :: root670Tail20
def root670Tail18 : List Nat := 18713 :: root670Tail19
def root670Tail17 : List Nat := 18311 :: root670Tail18
def root670Tail16 : List Nat := 18059 :: root670Tail17
def root670Tail15 : List Nat := 17327 :: root670Tail16
def root670Tail14 : List Nat := 16493 :: root670Tail15
def root670Tail13 : List Nat := 14051 :: root670Tail14
def root670Tail12 : List Nat := 13109 :: root670Tail13
def root670Tail11 : List Nat := 11279 :: root670Tail12
def root670Tail10 : List Nat := 10799 :: root670Tail11
def root670Tail9 : List Nat := 10733 :: root670Tail10
def root670Tail8 : List Nat := 10163 :: root670Tail9
def root670Tail7 : List Nat := 9497 :: root670Tail8
def root670Tail6 : List Nat := 8849 :: root670Tail7
def root670Tail5 : List Nat := 8219 :: root670Tail6
def root670Tail4 : List Nat := 7127 :: root670Tail5
def root670Tail3 : List Nat := 6053 :: root670Tail4
def root670Tail2 : List Nat := 5987 :: root670Tail3
def root670Tail1 : List Nat := 5741 :: root670Tail2
def root670Tail0 : List Nat := 5507 :: root670Tail1
theorem root670Scan0 : scan edge (search edge 3) 4 [5021] 5021 root670Tail0 winner = winner := by rw [root670Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root670Branch : search edge 4 [5021] root670Candidates 5021 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5021] 5021 root670Tail0 winner = winner
  exact root670Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
