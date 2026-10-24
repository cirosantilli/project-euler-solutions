import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root661Candidates : List Nat := Data.row4967
set_option maxRecDepth 8192 in
theorem root661Candidates_checked : candidates edge 5 0 4967 tail662 winner.bound = root661Candidates := by
  apply candidates_eq_row edge 5 0 4967 tail662 root661Candidates winner.bound tail662_ordered
  · decide
  · decide
  · funext q
    rfl
def root661Tail36 : List Nat := []
def root661Tail35 : List Nat := 20897 :: root661Tail36
def root661Tail34 : List Nat := 20771 :: root661Tail35
def root661Tail33 : List Nat := 20129 :: root661Tail34
def root661Tail32 : List Nat := 20051 :: root661Tail33
def root661Tail31 : List Nat := 19463 :: root661Tail32
def root661Tail30 : List Nat := 19121 :: root661Tail31
def root661Tail29 : List Nat := 18713 :: root661Tail30
def root661Tail28 : List Nat := 18149 :: root661Tail29
def root661Tail27 : List Nat := 16061 :: root661Tail28
def root661Tail26 : List Nat := 15473 :: root661Tail27
def root661Tail25 : List Nat := 15467 :: root661Tail26
def root661Tail24 : List Nat := 15377 :: root661Tail25
def root661Tail23 : List Nat := 13577 :: root661Tail24
def root661Tail22 : List Nat := 13499 :: root661Tail23
def root661Tail21 : List Nat := 13331 :: root661Tail22
def root661Tail20 : List Nat := 12959 :: root661Tail21
def root661Tail19 : List Nat := 12911 :: root661Tail20
def root661Tail18 : List Nat := 12539 :: root661Tail19
def root661Tail17 : List Nat := 11393 :: root661Tail18
def root661Tail16 : List Nat := 11057 :: root661Tail17
def root661Tail15 : List Nat := 9857 :: root661Tail16
def root661Tail14 : List Nat := 8747 :: root661Tail15
def root661Tail13 : List Nat := 8501 :: root661Tail14
def root661Tail12 : List Nat := 8219 :: root661Tail13
def root661Tail11 : List Nat := 8117 :: root661Tail12
def root661Tail10 : List Nat := 7877 :: root661Tail11
def root661Tail9 : List Nat := 7547 :: root661Tail10
def root661Tail8 : List Nat := 7127 :: root661Tail9
def root661Tail7 : List Nat := 6827 :: root661Tail8
def root661Tail6 : List Nat := 5939 :: root661Tail7
def root661Tail5 : List Nat := 5903 :: root661Tail6
def root661Tail4 : List Nat := 5849 :: root661Tail5
def root661Tail3 : List Nat := 5573 :: root661Tail4
def root661Tail2 : List Nat := 5531 :: root661Tail3
def root661Tail1 : List Nat := 5441 :: root661Tail2
def root661Tail0 : List Nat := 5387 :: root661Tail1
theorem root661Scan0 : scan edge (search edge 3) 4 [4967] 4967 root661Tail0 winner = winner := by rw [root661Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root661Branch : search edge 4 [4967] root661Candidates 4967 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4967] 4967 root661Tail0 winner = winner
  exact root661Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
