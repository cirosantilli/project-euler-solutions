import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root683Candidates : List Nat := Data.row5147
set_option maxRecDepth 8192 in
theorem root683Candidates_checked : candidates edge 5 0 5147 tail684 winner.bound = root683Candidates := by
  apply candidates_eq_row edge 5 0 5147 tail684 root683Candidates winner.bound tail684_ordered
  · decide
  · decide
  · funext q
    rfl
def root683Tail23 : List Nat := []
def root683Tail22 : List Nat := 20201 :: root683Tail23
def root683Tail21 : List Nat := 18047 :: root683Tail22
def root683Tail20 : List Nat := 16067 :: root683Tail21
def root683Tail19 : List Nat := 15671 :: root683Tail20
def root683Tail18 : List Nat := 15329 :: root683Tail19
def root683Tail17 : List Nat := 15101 :: root683Tail18
def root683Tail16 : List Nat := 14957 :: root683Tail17
def root683Tail15 : List Nat := 14537 :: root683Tail16
def root683Tail14 : List Nat := 13613 :: root683Tail15
def root683Tail13 : List Nat := 13259 :: root683Tail14
def root683Tail12 : List Nat := 13007 :: root683Tail13
def root683Tail11 : List Nat := 12227 :: root683Tail12
def root683Tail10 : List Nat := 12161 :: root683Tail11
def root683Tail9 : List Nat := 9851 :: root683Tail10
def root683Tail8 : List Nat := 9203 :: root683Tail9
def root683Tail7 : List Nat := 7823 :: root683Tail8
def root683Tail6 : List Nat := 7487 :: root683Tail7
def root683Tail5 : List Nat := 7331 :: root683Tail6
def root683Tail4 : List Nat := 6959 :: root683Tail5
def root683Tail3 : List Nat := 6857 :: root683Tail4
def root683Tail2 : List Nat := 5297 :: root683Tail3
def root683Tail1 : List Nat := 5273 :: root683Tail2
def root683Tail0 : List Nat := 5231 :: root683Tail1
theorem root683Scan0 : scan edge (search edge 3) 4 [5147] 5147 root683Tail0 winner = winner := by rw [root683Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root683Branch : search edge 4 [5147] root683Candidates 5147 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5147] 5147 root683Tail0 winner = winner
  exact root683Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
