import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root656Candidates : List Nat := Data.row4933
set_option maxRecDepth 8192 in
theorem root656Candidates_checked : candidates edge 5 0 4933 tail657 winner.bound = root656Candidates := by
  apply candidates_eq_row edge 5 0 4933 tail657 root656Candidates winner.bound tail657_ordered
  · decide
  · decide
  · funext q
    rfl
def root656Tail27 : List Nat := []
def root656Tail26 : List Nat := 20071 :: root656Tail27
def root656Tail25 : List Nat := 19963 :: root656Tail26
def root656Tail24 : List Nat := 19867 :: root656Tail25
def root656Tail23 : List Nat := 19441 :: root656Tail24
def root656Tail22 : List Nat := 18517 :: root656Tail23
def root656Tail21 : List Nat := 17971 :: root656Tail22
def root656Tail20 : List Nat := 17761 :: root656Tail21
def root656Tail19 : List Nat := 17257 :: root656Tail20
def root656Tail18 : List Nat := 17107 :: root656Tail19
def root656Tail17 : List Nat := 16267 :: root656Tail18
def root656Tail16 : List Nat := 15727 :: root656Tail17
def root656Tail15 : List Nat := 15643 :: root656Tail16
def root656Tail14 : List Nat := 15349 :: root656Tail15
def root656Tail13 : List Nat := 14293 :: root656Tail14
def root656Tail12 : List Nat := 14149 :: root656Tail13
def root656Tail11 : List Nat := 13669 :: root656Tail12
def root656Tail10 : List Nat := 13291 :: root656Tail11
def root656Tail9 : List Nat := 13159 :: root656Tail10
def root656Tail8 : List Nat := 12109 :: root656Tail9
def root656Tail7 : List Nat := 11131 :: root656Tail8
def root656Tail6 : List Nat := 10009 :: root656Tail7
def root656Tail5 : List Nat := 9421 :: root656Tail6
def root656Tail4 : List Nat := 8893 :: root656Tail5
def root656Tail3 : List Nat := 8629 :: root656Tail4
def root656Tail2 : List Nat := 6361 :: root656Tail3
def root656Tail1 : List Nat := 6271 :: root656Tail2
def root656Tail0 : List Nat := 5701 :: root656Tail1
theorem root656Scan0 : scan edge (search edge 3) 4 [4933] 4933 root656Tail0 winner = winner := by rw [root656Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root656Branch : search edge 4 [4933] root656Candidates 4933 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4933] 4933 root656Tail0 winner = winner
  exact root656Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
