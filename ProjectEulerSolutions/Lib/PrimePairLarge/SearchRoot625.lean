import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root625Candidates : List Nat := Data.row4649
set_option maxRecDepth 8192 in
theorem root625Candidates_checked : candidates edge 5 0 4649 tail626 winner.bound = root625Candidates := by
  apply candidates_eq_row edge 5 0 4649 tail626 root625Candidates winner.bound tail626_ordered
  · decide
  · decide
  · funext q
    rfl
def root625Tail36 : List Nat := []
def root625Tail35 : List Nat := 20411 :: root625Tail36
def root625Tail34 : List Nat := 20297 :: root625Tail35
def root625Tail33 : List Nat := 20219 :: root625Tail34
def root625Tail32 : List Nat := 20147 :: root625Tail33
def root625Tail31 : List Nat := 20129 :: root625Tail32
def root625Tail30 : List Nat := 19583 :: root625Tail31
def root625Tail29 : List Nat := 19421 :: root625Tail30
def root625Tail28 : List Nat := 18719 :: root625Tail29
def root625Tail27 : List Nat := 18119 :: root625Tail28
def root625Tail26 : List Nat := 18059 :: root625Tail27
def root625Tail25 : List Nat := 15683 :: root625Tail26
def root625Tail24 : List Nat := 15497 :: root625Tail25
def root625Tail23 : List Nat := 15329 :: root625Tail24
def root625Tail22 : List Nat := 15149 :: root625Tail23
def root625Tail21 : List Nat := 14951 :: root625Tail22
def root625Tail20 : List Nat := 14633 :: root625Tail21
def root625Tail19 : List Nat := 14627 :: root625Tail20
def root625Tail18 : List Nat := 14207 :: root625Tail19
def root625Tail17 : List Nat := 12893 :: root625Tail18
def root625Tail16 : List Nat := 12713 :: root625Tail17
def root625Tail15 : List Nat := 12671 :: root625Tail16
def root625Tail14 : List Nat := 12641 :: root625Tail15
def root625Tail13 : List Nat := 12251 :: root625Tail14
def root625Tail12 : List Nat := 12113 :: root625Tail13
def root625Tail11 : List Nat := 10163 :: root625Tail12
def root625Tail10 : List Nat := 9857 :: root625Tail11
def root625Tail9 : List Nat := 9461 :: root625Tail10
def root625Tail8 : List Nat := 9221 :: root625Tail9
def root625Tail7 : List Nat := 8537 :: root625Tail8
def root625Tail6 : List Nat := 8117 :: root625Tail7
def root625Tail5 : List Nat := 7853 :: root625Tail6
def root625Tail4 : List Nat := 7727 :: root625Tail5
def root625Tail3 : List Nat := 6719 :: root625Tail4
def root625Tail2 : List Nat := 6323 :: root625Tail3
def root625Tail1 : List Nat := 5987 :: root625Tail2
def root625Tail0 : List Nat := 5399 :: root625Tail1
theorem root625Scan0 : scan edge (search edge 3) 4 [4649] 4649 root625Tail0 winner = winner := by rw [root625Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root625Branch : search edge 4 [4649] root625Candidates 4649 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4649] 4649 root625Tail0 winner = winner
  exact root625Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
