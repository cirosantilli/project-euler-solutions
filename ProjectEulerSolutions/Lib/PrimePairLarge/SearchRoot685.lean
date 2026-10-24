import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root685Candidates : List Nat := Data.row5167
set_option maxRecDepth 8192 in
theorem root685Candidates_checked : candidates edge 5 0 5167 tail686 winner.bound = root685Candidates := by
  apply candidates_eq_row edge 5 0 5167 tail686 root685Candidates winner.bound tail686_ordered
  · decide
  · decide
  · funext q
    rfl
def root685Tail28 : List Nat := []
def root685Tail27 : List Nat := 20857 :: root685Tail28
def root685Tail26 : List Nat := 19447 :: root685Tail27
def root685Tail25 : List Nat := 19309 :: root685Tail26
def root685Tail24 : List Nat := 19051 :: root685Tail25
def root685Tail23 : List Nat := 17599 :: root685Tail24
def root685Tail22 : List Nat := 16633 :: root685Tail23
def root685Tail21 : List Nat := 16561 :: root685Tail22
def root685Tail20 : List Nat := 16141 :: root685Tail21
def root685Tail19 : List Nat := 16087 :: root685Tail20
def root685Tail18 : List Nat := 15319 :: root685Tail19
def root685Tail17 : List Nat := 15061 :: root685Tail18
def root685Tail16 : List Nat := 13513 :: root685Tail17
def root685Tail15 : List Nat := 12799 :: root685Tail16
def root685Tail14 : List Nat := 11887 :: root685Tail15
def root685Tail13 : List Nat := 11329 :: root685Tail14
def root685Tail12 : List Nat := 10867 :: root685Tail13
def root685Tail11 : List Nat := 10837 :: root685Tail12
def root685Tail10 : List Nat := 10657 :: root685Tail11
def root685Tail9 : List Nat := 10501 :: root685Tail10
def root685Tail8 : List Nat := 10459 :: root685Tail9
def root685Tail7 : List Nat := 9619 :: root685Tail8
def root685Tail6 : List Nat := 8941 :: root685Tail7
def root685Tail5 : List Nat := 8761 :: root685Tail6
def root685Tail4 : List Nat := 7741 :: root685Tail5
def root685Tail3 : List Nat := 7177 :: root685Tail4
def root685Tail2 : List Nat := 6967 :: root685Tail3
def root685Tail1 : List Nat := 6661 :: root685Tail2
def root685Tail0 : List Nat := 5737 :: root685Tail1
theorem root685Scan0 : scan edge (search edge 3) 4 [5167] 5167 root685Tail0 winner = winner := by rw [root685Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root685Branch : search edge 4 [5167] root685Candidates 5167 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5167] 5167 root685Tail0 winner = winner
  exact root685Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
