import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root668Candidates : List Nat := Data.row5009
set_option maxRecDepth 8192 in
theorem root668Candidates_checked : candidates edge 5 0 5009 tail669 winner.bound = root668Candidates := by
  apply candidates_eq_row edge 5 0 5009 tail669 root668Candidates winner.bound tail669_ordered
  · decide
  · decide
  · funext q
    rfl
def root668Tail34 : List Nat := []
def root668Tail33 : List Nat := 20897 :: root668Tail34
def root668Tail32 : List Nat := 20261 :: root668Tail33
def root668Tail31 : List Nat := 20183 :: root668Tail32
def root668Tail30 : List Nat := 19973 :: root668Tail31
def root668Tail29 : List Nat := 19889 :: root668Tail30
def root668Tail28 : List Nat := 18797 :: root668Tail29
def root668Tail27 : List Nat := 18461 :: root668Tail28
def root668Tail26 : List Nat := 18233 :: root668Tail27
def root668Tail25 : List Nat := 16349 :: root668Tail26
def root668Tail24 : List Nat := 16139 :: root668Tail25
def root668Tail23 : List Nat := 16061 :: root668Tail24
def root668Tail22 : List Nat := 15923 :: root668Tail23
def root668Tail21 : List Nat := 15881 :: root668Tail22
def root668Tail20 : List Nat := 15131 :: root668Tail21
def root668Tail19 : List Nat := 14669 :: root668Tail20
def root668Tail18 : List Nat := 13043 :: root668Tail19
def root668Tail17 : List Nat := 12959 :: root668Tail18
def root668Tail16 : List Nat := 12413 :: root668Tail17
def root668Tail15 : List Nat := 11399 :: root668Tail16
def root668Tail14 : List Nat := 11057 :: root668Tail15
def root668Tail13 : List Nat := 10133 :: root668Tail14
def root668Tail12 : List Nat := 9839 :: root668Tail13
def root668Tail11 : List Nat := 9059 :: root668Tail12
def root668Tail10 : List Nat := 8501 :: root668Tail11
def root668Tail9 : List Nat := 8093 :: root668Tail10
def root668Tail8 : List Nat := 7757 :: root668Tail9
def root668Tail7 : List Nat := 7109 :: root668Tail8
def root668Tail6 : List Nat := 6947 :: root668Tail7
def root668Tail5 : List Nat := 6353 :: root668Tail6
def root668Tail4 : List Nat := 6197 :: root668Tail5
def root668Tail3 : List Nat := 6029 :: root668Tail4
def root668Tail2 : List Nat := 5861 :: root668Tail3
def root668Tail1 : List Nat := 5849 :: root668Tail2
def root668Tail0 : List Nat := 5351 :: root668Tail1
theorem root668Scan0 : scan edge (search edge 3) 4 [5009] 5009 root668Tail0 winner = winner := by rw [root668Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root668Branch : search edge 4 [5009] root668Candidates 5009 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5009] 5009 root668Tail0 winner = winner
  exact root668Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
