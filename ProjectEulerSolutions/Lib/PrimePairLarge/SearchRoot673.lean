import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root673Candidates : List Nat := Data.row5051
set_option maxRecDepth 8192 in
theorem root673Candidates_checked : candidates edge 5 0 5051 tail674 winner.bound = root673Candidates := by
  apply candidates_eq_row edge 5 0 5051 tail674 root673Candidates winner.bound tail674_ordered
  · decide
  · decide
  · funext q
    rfl
def root673Tail36 : List Nat := []
def root673Tail35 : List Nat := 20921 :: root673Tail36
def root673Tail34 : List Nat := 20333 :: root673Tail35
def root673Tail33 : List Nat := 19973 :: root673Tail34
def root673Tail32 : List Nat := 19889 :: root673Tail33
def root673Tail31 : List Nat := 19553 :: root673Tail32
def root673Tail30 : List Nat := 19469 :: root673Tail31
def root673Tail29 : List Nat := 18947 :: root673Tail30
def root673Tail28 : List Nat := 18233 :: root673Tail29
def root673Tail27 : List Nat := 17783 :: root673Tail28
def root673Tail26 : List Nat := 17729 :: root673Tail27
def root673Tail25 : List Nat := 17183 :: root673Tail26
def root673Tail24 : List Nat := 17117 :: root673Tail25
def root673Tail23 : List Nat := 15467 :: root673Tail24
def root673Tail22 : List Nat := 15137 :: root673Tail23
def root673Tail21 : List Nat := 13997 :: root673Tail22
def root673Tail20 : List Nat := 12539 :: root673Tail21
def root673Tail19 : List Nat := 12479 :: root673Tail20
def root673Tail18 : List Nat := 12323 :: root673Tail19
def root673Tail17 : List Nat := 11699 :: root673Tail18
def root673Tail16 : List Nat := 11681 :: root673Tail17
def root673Tail15 : List Nat := 11393 :: root673Tail16
def root673Tail14 : List Nat := 10391 :: root673Tail15
def root673Tail13 : List Nat := 10271 :: root673Tail14
def root673Tail12 : List Nat := 9929 :: root673Tail13
def root673Tail11 : List Nat := 9173 :: root673Tail12
def root673Tail10 : List Nat := 8969 :: root673Tail11
def root673Tail9 : List Nat := 7949 :: root673Tail10
def root673Tail8 : List Nat := 7283 :: root673Tail9
def root673Tail7 : List Nat := 7253 :: root673Tail8
def root673Tail6 : List Nat := 6863 :: root673Tail7
def root673Tail5 : List Nat := 6833 :: root673Tail6
def root673Tail4 : List Nat := 6737 :: root673Tail5
def root673Tail3 : List Nat := 6101 :: root673Tail4
def root673Tail2 : List Nat := 5939 :: root673Tail3
def root673Tail1 : List Nat := 5903 :: root673Tail2
def root673Tail0 : List Nat := 5867 :: root673Tail1
theorem root673Scan0 : scan edge (search edge 3) 4 [5051] 5051 root673Tail0 winner = winner := by rw [root673Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root673Branch : search edge 4 [5051] root673Candidates 5051 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5051] 5051 root673Tail0 winner = winner
  exact root673Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
