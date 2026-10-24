import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root634Candidates : List Nat := Data.row4723
set_option maxRecDepth 8192 in
theorem root634Candidates_checked : candidates edge 5 0 4723 tail635 winner.bound = root634Candidates := by
  apply candidates_eq_row edge 5 0 4723 tail635 root634Candidates winner.bound tail635_ordered
  · decide
  · decide
  · funext q
    rfl
def root634Tail27 : List Nat := []
def root634Tail26 : List Nat := 21139 :: root634Tail27
def root634Tail25 : List Nat := 20389 :: root634Tail26
def root634Tail24 : List Nat := 19699 :: root634Tail25
def root634Tail23 : List Nat := 19207 :: root634Tail24
def root634Tail22 : List Nat := 18661 :: root634Tail23
def root634Tail21 : List Nat := 18439 :: root634Tail22
def root634Tail20 : List Nat := 17359 :: root634Tail21
def root634Tail19 : List Nat := 16561 :: root634Tail20
def root634Tail18 : List Nat := 16333 :: root634Tail19
def root634Tail17 : List Nat := 15919 :: root634Tail18
def root634Tail16 : List Nat := 15427 :: root634Tail17
def root634Tail15 : List Nat := 13249 :: root634Tail16
def root634Tail14 : List Nat := 13099 :: root634Tail15
def root634Tail13 : List Nat := 12979 :: root634Tail14
def root634Tail12 : List Nat := 10159 :: root634Tail13
def root634Tail11 : List Nat := 9901 :: root634Tail12
def root634Tail10 : List Nat := 9721 :: root634Tail11
def root634Tail9 : List Nat := 9613 :: root634Tail10
def root634Tail8 : List Nat := 8923 :: root634Tail9
def root634Tail7 : List Nat := 8839 :: root634Tail8
def root634Tail6 : List Nat := 8221 :: root634Tail7
def root634Tail5 : List Nat := 7369 :: root634Tail6
def root634Tail4 : List Nat := 7243 :: root634Tail5
def root634Tail3 : List Nat := 6361 :: root634Tail4
def root634Tail2 : List Nat := 5869 :: root634Tail3
def root634Tail1 : List Nat := 5623 :: root634Tail2
def root634Tail0 : List Nat := 5521 :: root634Tail1
theorem root634Scan0 : scan edge (search edge 3) 4 [4723] 4723 root634Tail0 winner = winner := by rw [root634Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root634Branch : search edge 4 [4723] root634Candidates 4723 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4723] 4723 root634Tail0 winner = winner
  exact root634Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
