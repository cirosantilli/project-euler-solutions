import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root677Candidates : List Nat := Data.row5087
set_option maxRecDepth 8192 in
theorem root677Candidates_checked : candidates edge 5 0 5087 tail678 winner.bound = root677Candidates := by
  apply candidates_eq_row edge 5 0 5087 tail678 root677Candidates winner.bound tail678_ordered
  · decide
  · decide
  · funext q
    rfl
def root677Tail28 : List Nat := []
def root677Tail27 : List Nat := 20483 :: root677Tail28
def root677Tail26 : List Nat := 20063 :: root677Tail27
def root677Tail25 : List Nat := 19391 :: root677Tail26
def root677Tail24 : List Nat := 19301 :: root677Tail25
def root677Tail23 : List Nat := 17597 :: root677Tail24
def root677Tail22 : List Nat := 17207 :: root677Tail23
def root677Tail21 : List Nat := 15269 :: root677Tail22
def root677Tail20 : List Nat := 15149 :: root677Tail21
def root677Tail19 : List Nat := 14561 :: root677Tail20
def root677Tail18 : List Nat := 14411 :: root677Tail19
def root677Tail17 : List Nat := 13001 :: root677Tail18
def root677Tail16 : List Nat := 12917 :: root677Tail17
def root677Tail15 : List Nat := 11789 :: root677Tail16
def root677Tail14 : List Nat := 10733 :: root677Tail15
def root677Tail13 : List Nat := 10463 :: root677Tail14
def root677Tail12 : List Nat := 10457 :: root677Tail13
def root677Tail11 : List Nat := 9539 :: root677Tail12
def root677Tail10 : List Nat := 8147 :: root677Tail11
def root677Tail9 : List Nat := 7247 :: root677Tail10
def root677Tail8 : List Nat := 6803 :: root677Tail9
def root677Tail7 : List Nat := 6779 :: root677Tail8
def root677Tail6 : List Nat := 6719 :: root677Tail7
def root677Tail5 : List Nat := 6569 :: root677Tail6
def root677Tail4 : List Nat := 5927 :: root677Tail5
def root677Tail3 : List Nat := 5897 :: root677Tail4
def root677Tail2 : List Nat := 5861 :: root677Tail3
def root677Tail1 : List Nat := 5519 :: root677Tail2
def root677Tail0 : List Nat := 5273 :: root677Tail1
theorem root677Scan0 : scan edge (search edge 3) 4 [5087] 5087 root677Tail0 winner = winner := by rw [root677Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root677Branch : search edge 4 [5087] root677Candidates 5087 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5087] 5087 root677Tail0 winner = winner
  exact root677Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
