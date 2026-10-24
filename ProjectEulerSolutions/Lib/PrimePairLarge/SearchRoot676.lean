import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root676Candidates : List Nat := Data.row5081
set_option maxRecDepth 8192 in
theorem root676Candidates_checked : candidates edge 5 0 5081 tail677 winner.bound = root676Candidates := by
  apply candidates_eq_row edge 5 0 5081 tail677 root676Candidates winner.bound tail677_ordered
  · decide
  · decide
  · funext q
    rfl
def root676Tail29 : List Nat := []
def root676Tail28 : List Nat := 20903 :: root676Tail29
def root676Tail27 : List Nat := 20399 :: root676Tail28
def root676Tail26 : List Nat := 19421 :: root676Tail27
def root676Tail25 : List Nat := 18749 :: root676Tail26
def root676Tail24 : List Nat := 18329 :: root676Tail25
def root676Tail23 : List Nat := 17921 :: root676Tail24
def root676Tail22 : List Nat := 17657 :: root676Tail23
def root676Tail21 : List Nat := 16493 :: root676Tail22
def root676Tail20 : List Nat := 16481 :: root676Tail21
def root676Tail19 : List Nat := 15809 :: root676Tail20
def root676Tail18 : List Nat := 13877 :: root676Tail19
def root676Tail17 : List Nat := 13679 :: root676Tail18
def root676Tail16 : List Nat := 13007 :: root676Tail17
def root676Tail15 : List Nat := 12821 :: root676Tail16
def root676Tail14 : List Nat := 12809 :: root676Tail15
def root676Tail13 : List Nat := 12401 :: root676Tail14
def root676Tail12 : List Nat := 11621 :: root676Tail13
def root676Tail11 : List Nat := 10193 :: root676Tail12
def root676Tail10 : List Nat := 10139 :: root676Tail11
def root676Tail9 : List Nat := 9203 :: root676Tail10
def root676Tail8 : List Nat := 8837 :: root676Tail9
def root676Tail7 : List Nat := 8513 :: root676Tail8
def root676Tail6 : List Nat := 8291 :: root676Tail7
def root676Tail5 : List Nat := 7817 :: root676Tail6
def root676Tail4 : List Nat := 7307 :: root676Tail5
def root676Tail3 : List Nat := 7253 :: root676Tail4
def root676Tail2 : List Nat := 7211 :: root676Tail3
def root676Tail1 : List Nat := 6971 :: root676Tail2
def root676Tail0 : List Nat := 5783 :: root676Tail1
theorem root676Scan0 : scan edge (search edge 3) 4 [5081] 5081 root676Tail0 winner = winner := by rw [root676Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root676Branch : search edge 4 [5081] root676Candidates 5081 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5081] 5081 root676Tail0 winner = winner
  exact root676Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
