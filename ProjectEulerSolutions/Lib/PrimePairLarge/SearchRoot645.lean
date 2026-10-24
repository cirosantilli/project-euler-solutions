import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root645Candidates : List Nat := Data.row4813
set_option maxRecDepth 8192 in
theorem root645Candidates_checked : candidates edge 5 0 4813 tail646 winner.bound = root645Candidates := by
  apply candidates_eq_row edge 5 0 4813 tail646 root645Candidates winner.bound tail646_ordered
  · decide
  · decide
  · funext q
    rfl
def root645Tail27 : List Nat := []
def root645Tail26 : List Nat := 20599 :: root645Tail27
def root645Tail25 : List Nat := 20107 :: root645Tail26
def root645Tail24 : List Nat := 19087 :: root645Tail25
def root645Tail23 : List Nat := 18919 :: root645Tail24
def root645Tail22 : List Nat := 18457 :: root645Tail23
def root645Tail21 : List Nat := 17257 :: root645Tail22
def root645Tail20 : List Nat := 17047 :: root645Tail21
def root645Tail19 : List Nat := 16963 :: root645Tail20
def root645Tail18 : List Nat := 16573 :: root645Tail19
def root645Tail17 : List Nat := 15643 :: root645Tail18
def root645Tail16 : List Nat := 14947 :: root645Tail17
def root645Tail15 : List Nat := 14737 :: root645Tail16
def root645Tail14 : List Nat := 13687 :: root645Tail15
def root645Tail13 : List Nat := 12553 :: root645Tail14
def root645Tail12 : List Nat := 12379 :: root645Tail13
def root645Tail11 : List Nat := 12241 :: root645Tail12
def root645Tail10 : List Nat := 9643 :: root645Tail11
def root645Tail9 : List Nat := 8863 :: root645Tail10
def root645Tail8 : List Nat := 8179 :: root645Tail9
def root645Tail7 : List Nat := 7351 :: root645Tail8
def root645Tail6 : List Nat := 7057 :: root645Tail7
def root645Tail5 : List Nat := 6661 :: root645Tail6
def root645Tail4 : List Nat := 6427 :: root645Tail5
def root645Tail3 : List Nat := 5869 :: root645Tail4
def root645Tail2 : List Nat := 5827 :: root645Tail3
def root645Tail1 : List Nat := 5701 :: root645Tail2
def root645Tail0 : List Nat := 5323 :: root645Tail1
theorem root645Scan0 : scan edge (search edge 3) 4 [4813] 4813 root645Tail0 winner = winner := by rw [root645Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root645Branch : search edge 4 [4813] root645Candidates 4813 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4813] 4813 root645Tail0 winner = winner
  exact root645Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
