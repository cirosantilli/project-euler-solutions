import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root682Candidates : List Nat := Data.row5119
set_option maxRecDepth 8192 in
theorem root682Candidates_checked : candidates edge 5 0 5119 tail683 winner.bound = root682Candidates := by
  apply candidates_eq_row edge 5 0 5119 tail683 root682Candidates winner.bound tail683_ordered
  · decide
  · decide
  · funext q
    rfl
def root682Tail30 : List Nat := []
def root682Tail29 : List Nat := 20707 :: root682Tail30
def root682Tail28 : List Nat := 20389 :: root682Tail29
def root682Tail27 : List Nat := 20089 :: root682Tail28
def root682Tail26 : List Nat := 19927 :: root682Tail27
def root682Tail25 : List Nat := 18541 :: root682Tail26
def root682Tail24 : List Nat := 18481 :: root682Tail25
def root682Tail23 : List Nat := 18013 :: root682Tail24
def root682Tail22 : List Nat := 17137 :: root682Tail23
def root682Tail21 : List Nat := 16963 :: root682Tail22
def root682Tail20 : List Nat := 14797 :: root682Tail21
def root682Tail19 : List Nat := 14173 :: root682Tail20
def root682Tail18 : List Nat := 12853 :: root682Tail19
def root682Tail17 : List Nat := 12601 :: root682Tail18
def root682Tail16 : List Nat := 12277 :: root682Tail17
def root682Tail15 : List Nat := 11731 :: root682Tail16
def root682Tail14 : List Nat := 10957 :: root682Tail15
def root682Tail13 : List Nat := 10177 :: root682Tail14
def root682Tail12 : List Nat := 9781 :: root682Tail13
def root682Tail11 : List Nat := 8677 :: root682Tail12
def root682Tail10 : List Nat := 8647 :: root682Tail11
def root682Tail9 : List Nat := 7621 :: root682Tail10
def root682Tail8 : List Nat := 7537 :: root682Tail9
def root682Tail7 : List Nat := 7459 :: root682Tail8
def root682Tail6 : List Nat := 7207 :: root682Tail7
def root682Tail5 : List Nat := 6637 :: root682Tail6
def root682Tail4 : List Nat := 6529 :: root682Tail5
def root682Tail3 : List Nat := 6337 :: root682Tail4
def root682Tail2 : List Nat := 6073 :: root682Tail3
def root682Tail1 : List Nat := 5503 :: root682Tail2
def root682Tail0 : List Nat := 5233 :: root682Tail1
theorem root682Scan0 : scan edge (search edge 3) 4 [5119] 5119 root682Tail0 winner = winner := by rw [root682Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root682Branch : search edge 4 [5119] root682Candidates 5119 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5119] 5119 root682Tail0 winner = winner
  exact root682Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
