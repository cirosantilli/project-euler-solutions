import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root681Candidates : List Nat := Data.row5113
set_option maxRecDepth 8192 in
theorem root681Candidates_checked : candidates edge 5 0 5113 tail682 winner.bound = root681Candidates := by
  apply candidates_eq_row edge 5 0 5113 tail682 root681Candidates winner.bound tail682_ordered
  · decide
  · decide
  · funext q
    rfl
def root681Tail29 : List Nat := []
def root681Tail28 : List Nat := 20899 :: root681Tail29
def root681Tail27 : List Nat := 18679 :: root681Tail28
def root681Tail26 : List Nat := 18583 :: root681Tail27
def root681Tail25 : List Nat := 17707 :: root681Tail26
def root681Tail24 : List Nat := 15601 :: root681Tail25
def root681Tail23 : List Nat := 15319 :: root681Tail24
def root681Tail22 : List Nat := 14767 :: root681Tail23
def root681Tail21 : List Nat := 14737 :: root681Tail22
def root681Tail20 : List Nat := 13963 :: root681Tail21
def root681Tail19 : List Nat := 13513 :: root681Tail20
def root681Tail18 : List Nat := 13183 :: root681Tail19
def root681Tail17 : List Nat := 13177 :: root681Tail18
def root681Tail16 : List Nat := 12973 :: root681Tail17
def root681Tail15 : List Nat := 11113 :: root681Tail16
def root681Tail14 : List Nat := 9967 :: root681Tail15
def root681Tail13 : List Nat := 9343 :: root681Tail14
def root681Tail12 : List Nat := 9007 :: root681Tail13
def root681Tail11 : List Nat := 9001 :: root681Tail12
def root681Tail10 : List Nat := 8929 :: root681Tail11
def root681Tail9 : List Nat := 8641 :: root681Tail10
def root681Tail8 : List Nat := 8293 :: root681Tail9
def root681Tail7 : List Nat := 7789 :: root681Tail8
def root681Tail6 : List Nat := 7069 :: root681Tail7
def root681Tail5 : List Nat := 6991 :: root681Tail6
def root681Tail4 : List Nat := 6823 :: root681Tail5
def root681Tail3 : List Nat := 6793 :: root681Tail4
def root681Tail2 : List Nat := 5659 :: root681Tail3
def root681Tail1 : List Nat := 5449 :: root681Tail2
def root681Tail0 : List Nat := 5431 :: root681Tail1
theorem root681Scan0 : scan edge (search edge 3) 4 [5113] 5113 root681Tail0 winner = winner := by rw [root681Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root681Branch : search edge 4 [5113] root681Candidates 5113 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5113] 5113 root681Tail0 winner = winner
  exact root681Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
