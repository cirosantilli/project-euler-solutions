import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root686Candidates : List Nat := Data.row5171
set_option maxRecDepth 8192 in
theorem root686Candidates_checked : candidates edge 5 0 5171 tail687 winner.bound = root686Candidates := by
  apply candidates_eq_row edge 5 0 5171 tail687 root686Candidates winner.bound tail687_ordered
  · decide
  · decide
  · funext q
    rfl
def root686Tail24 : List Nat := []
def root686Tail23 : List Nat := 20693 :: root686Tail24
def root686Tail22 : List Nat := 20231 :: root686Tail23
def root686Tail21 : List Nat := 18713 :: root686Tail22
def root686Tail20 : List Nat := 18521 :: root686Tail21
def root686Tail19 : List Nat := 17387 :: root686Tail20
def root686Tail18 : List Nat := 15017 :: root686Tail19
def root686Tail17 : List Nat := 14537 :: root686Tail18
def root686Tail16 : List Nat := 14447 :: root686Tail17
def root686Tail15 : List Nat := 13967 :: root686Tail16
def root686Tail14 : List Nat := 13679 :: root686Tail15
def root686Tail13 : List Nat := 13109 :: root686Tail14
def root686Tail12 : List Nat := 11411 :: root686Tail13
def root686Tail11 : List Nat := 10457 :: root686Tail12
def root686Tail10 : List Nat := 10247 :: root686Tail11
def root686Tail9 : List Nat := 10211 :: root686Tail10
def root686Tail8 : List Nat := 10151 :: root686Tail9
def root686Tail7 : List Nat := 9431 :: root686Tail8
def root686Tail6 : List Nat := 8837 :: root686Tail7
def root686Tail5 : List Nat := 8111 :: root686Tail6
def root686Tail4 : List Nat := 8039 :: root686Tail5
def root686Tail3 : List Nat := 7187 :: root686Tail4
def root686Tail2 : List Nat := 5897 :: root686Tail3
def root686Tail1 : List Nat := 5393 :: root686Tail2
def root686Tail0 : List Nat := 5297 :: root686Tail1
theorem root686Scan0 : scan edge (search edge 3) 4 [5171] 5171 root686Tail0 winner = winner := by rw [root686Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root686Branch : search edge 4 [5171] root686Candidates 5171 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5171] 5171 root686Tail0 winner = winner
  exact root686Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
