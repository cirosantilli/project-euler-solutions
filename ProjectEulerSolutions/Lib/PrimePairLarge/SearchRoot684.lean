import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root684Candidates : List Nat := Data.row5153
set_option maxRecDepth 8192 in
theorem root684Candidates_checked : candidates edge 5 0 5153 tail685 winner.bound = root684Candidates := by
  apply candidates_eq_row edge 5 0 5153 tail685 root684Candidates winner.bound tail685_ordered
  · decide
  · decide
  · funext q
    rfl
def root684Tail29 : List Nat := []
def root684Tail28 : List Nat := 19709 :: root684Tail29
def root684Tail27 : List Nat := 19001 :: root684Tail28
def root684Tail26 : List Nat := 17657 :: root684Tail27
def root684Tail25 : List Nat := 16493 :: root684Tail26
def root684Tail24 : List Nat := 15683 :: root684Tail25
def root684Tail23 : List Nat := 15107 :: root684Tail24
def root684Tail22 : List Nat := 14843 :: root684Tail23
def root684Tail21 : List Nat := 12923 :: root684Tail22
def root684Tail20 : List Nat := 12251 :: root684Tail21
def root684Tail19 : List Nat := 11969 :: root684Tail20
def root684Tail18 : List Nat := 11579 :: root684Tail19
def root684Tail17 : List Nat := 11087 :: root684Tail18
def root684Tail16 : List Nat := 10949 :: root684Tail17
def root684Tail15 : List Nat := 10559 :: root684Tail16
def root684Tail14 : List Nat := 10487 :: root684Tail15
def root684Tail13 : List Nat := 10079 :: root684Tail14
def root684Tail12 : List Nat := 9491 :: root684Tail13
def root684Tail11 : List Nat := 9311 :: root684Tail12
def root684Tail10 : List Nat := 9041 :: root684Tail11
def root684Tail9 : List Nat := 8819 :: root684Tail10
def root684Tail8 : List Nat := 8693 :: root684Tail9
def root684Tail7 : List Nat := 8297 :: root684Tail8
def root684Tail6 : List Nat := 7589 :: root684Tail7
def root684Tail5 : List Nat := 7529 :: root684Tail6
def root684Tail4 : List Nat := 6971 :: root684Tail5
def root684Tail3 : List Nat := 6101 :: root684Tail4
def root684Tail2 : List Nat := 6047 :: root684Tail3
def root684Tail1 : List Nat := 5867 :: root684Tail2
def root684Tail0 : List Nat := 5849 :: root684Tail1
theorem root684Scan0 : scan edge (search edge 3) 4 [5153] 5153 root684Tail0 winner = winner := by rw [root684Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root684Branch : search edge 4 [5153] root684Candidates 5153 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5153] 5153 root684Tail0 winner = winner
  exact root684Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
