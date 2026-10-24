import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root648Candidates : List Nat := Data.row4861
set_option maxRecDepth 8192 in
theorem root648Candidates_checked : candidates edge 5 0 4861 tail649 winner.bound = root648Candidates := by
  apply candidates_eq_row edge 5 0 4861 tail649 root648Candidates winner.bound tail649_ordered
  · decide
  · decide
  · funext q
    rfl
def root648Tail26 : List Nat := []
def root648Tail25 : List Nat := 20563 :: root648Tail26
def root648Tail24 : List Nat := 18637 :: root648Tail25
def root648Tail23 : List Nat := 18541 :: root648Tail24
def root648Tail22 : List Nat := 18517 :: root648Tail23
def root648Tail21 : List Nat := 18253 :: root648Tail22
def root648Tail20 : List Nat := 17449 :: root648Tail21
def root648Tail19 : List Nat := 16993 :: root648Tail20
def root648Tail18 : List Nat := 15451 :: root648Tail19
def root648Tail17 : List Nat := 15439 :: root648Tail18
def root648Tail16 : List Nat := 14341 :: root648Tail17
def root648Tail15 : List Nat := 12799 :: root648Tail16
def root648Tail14 : List Nat := 12577 :: root648Tail15
def root648Tail13 : List Nat := 11779 :: root648Tail14
def root648Tail12 : List Nat := 11317 :: root648Tail13
def root648Tail11 : List Nat := 10321 :: root648Tail12
def root648Tail10 : List Nat := 9007 :: root648Tail11
def root648Tail9 : List Nat := 7879 :: root648Tail10
def root648Tail8 : List Nat := 7573 :: root648Tail9
def root648Tail7 : List Nat := 7243 :: root648Tail8
def root648Tail6 : List Nat := 6829 :: root648Tail7
def root648Tail5 : List Nat := 6361 :: root648Tail6
def root648Tail4 : List Nat := 6229 :: root648Tail5
def root648Tail3 : List Nat := 6163 :: root648Tail4
def root648Tail2 : List Nat := 5953 :: root648Tail3
def root648Tail1 : List Nat := 5689 :: root648Tail2
def root648Tail0 : List Nat := 5659 :: root648Tail1
theorem root648Scan0 : scan edge (search edge 3) 4 [4861] 4861 root648Tail0 winner = winner := by rw [root648Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root648Branch : search edge 4 [4861] root648Candidates 4861 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4861] 4861 root648Tail0 winner = winner
  exact root648Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
