import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root662Candidates : List Nat := Data.row4969
set_option maxRecDepth 8192 in
theorem root662Candidates_checked : candidates edge 5 0 4969 tail663 winner.bound = root662Candidates := by
  apply candidates_eq_row edge 5 0 4969 tail663 root662Candidates winner.bound tail663_ordered
  · decide
  · decide
  · funext q
    rfl
def root662Tail31 : List Nat := []
def root662Tail30 : List Nat := 20959 :: root662Tail31
def root662Tail29 : List Nat := 19471 :: root662Tail30
def root662Tail28 : List Nat := 19183 :: root662Tail29
def root662Tail27 : List Nat := 18049 :: root662Tail28
def root662Tail26 : List Nat := 17791 :: root662Tail27
def root662Tail25 : List Nat := 17401 :: root662Tail26
def root662Tail24 : List Nat := 16603 :: root662Tail25
def root662Tail23 : List Nat := 15817 :: root662Tail24
def root662Tail22 : List Nat := 15373 :: root662Tail23
def root662Tail21 : List Nat := 15193 :: root662Tail22
def root662Tail20 : List Nat := 14293 :: root662Tail21
def root662Tail19 : List Nat := 13693 :: root662Tail20
def root662Tail18 : List Nat := 12487 :: root662Tail19
def root662Tail17 : List Nat := 12373 :: root662Tail18
def root662Tail16 : List Nat := 11497 :: root662Tail17
def root662Tail15 : List Nat := 11383 :: root662Tail16
def root662Tail14 : List Nat := 10891 :: root662Tail15
def root662Tail13 : List Nat := 10861 :: root662Tail14
def root662Tail12 : List Nat := 9817 :: root662Tail13
def root662Tail11 : List Nat := 9049 :: root662Tail12
def root662Tail10 : List Nat := 8923 :: root662Tail11
def root662Tail9 : List Nat := 8461 :: root662Tail10
def root662Tail8 : List Nat := 8329 :: root662Tail9
def root662Tail7 : List Nat := 7507 :: root662Tail8
def root662Tail6 : List Nat := 7321 :: root662Tail7
def root662Tail5 : List Nat := 6679 :: root662Tail6
def root662Tail4 : List Nat := 6397 :: root662Tail5
def root662Tail3 : List Nat := 6133 :: root662Tail4
def root662Tail2 : List Nat := 6067 :: root662Tail3
def root662Tail1 : List Nat := 5881 :: root662Tail2
def root662Tail0 : List Nat := 5827 :: root662Tail1
theorem root662Scan0 : scan edge (search edge 3) 4 [4969] 4969 root662Tail0 winner = winner := by rw [root662Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root662Branch : search edge 4 [4969] root662Candidates 4969 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4969] 4969 root662Tail0 winner = winner
  exact root662Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
