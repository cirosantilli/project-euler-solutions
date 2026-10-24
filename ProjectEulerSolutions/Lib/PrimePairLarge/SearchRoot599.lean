import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root599Candidates : List Nat := Data.row4423
set_option maxRecDepth 8192 in
theorem root599Candidates_checked : candidates edge 5 0 4423 tail600 winner.bound = root599Candidates := by
  apply candidates_eq_row edge 5 0 4423 tail600 root599Candidates winner.bound tail600_ordered
  · decide
  · decide
  · funext q
    rfl
def root599Tail29 : List Nat := []
def root599Tail28 : List Nat := 21013 :: root599Tail29
def root599Tail27 : List Nat := 20887 :: root599Tail28
def root599Tail26 : List Nat := 20719 :: root599Tail27
def root599Tail25 : List Nat := 19333 :: root599Tail26
def root599Tail24 : List Nat := 17431 :: root599Tail25
def root599Tail23 : List Nat := 17011 :: root599Tail24
def root599Tail22 : List Nat := 16141 :: root599Tail23
def root599Tail21 : List Nat := 16087 :: root599Tail22
def root599Tail20 : List Nat := 15901 :: root599Tail21
def root599Tail19 : List Nat := 15823 :: root599Tail20
def root599Tail18 : List Nat := 15187 :: root599Tail19
def root599Tail17 : List Nat := 13693 :: root599Tail18
def root599Tail16 : List Nat := 13537 :: root599Tail17
def root599Tail15 : List Nat := 13411 :: root599Tail16
def root599Tail14 : List Nat := 12757 :: root599Tail15
def root599Tail13 : List Nat := 12109 :: root599Tail14
def root599Tail12 : List Nat := 10567 :: root599Tail13
def root599Tail11 : List Nat := 10357 :: root599Tail12
def root599Tail10 : List Nat := 10303 :: root599Tail11
def root599Tail9 : List Nat := 9817 :: root599Tail10
def root599Tail8 : List Nat := 9421 :: root599Tail9
def root599Tail7 : List Nat := 9343 :: root599Tail8
def root599Tail6 : List Nat := 9337 :: root599Tail7
def root599Tail5 : List Nat := 8863 :: root599Tail6
def root599Tail4 : List Nat := 8317 :: root599Tail5
def root599Tail3 : List Nat := 6133 :: root599Tail4
def root599Tail2 : List Nat := 5869 :: root599Tail3
def root599Tail1 : List Nat := 5689 :: root599Tail2
def root599Tail0 : List Nat := 5449 :: root599Tail1
theorem root599Scan0 : scan edge (search edge 3) 4 [4423] 4423 root599Tail0 winner = winner := by rw [root599Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root599Branch : search edge 4 [4423] root599Candidates 4423 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4423] 4423 root599Tail0 winner = winner
  exact root599Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
