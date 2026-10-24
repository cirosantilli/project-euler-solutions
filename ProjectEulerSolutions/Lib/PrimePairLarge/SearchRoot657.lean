import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root657Candidates : List Nat := Data.row4937
set_option maxRecDepth 8192 in
theorem root657Candidates_checked : candidates edge 5 0 4937 tail658 winner.bound = root657Candidates := by
  apply candidates_eq_row edge 5 0 4937 tail658 root657Candidates winner.bound tail658_ordered
  · decide
  · decide
  · funext q
    rfl
def root657Tail33 : List Nat := []
def root657Tail32 : List Nat := 20543 :: root657Tail33
def root657Tail31 : List Nat := 18521 :: root657Tail32
def root657Tail30 : List Nat := 18257 :: root657Tail31
def root657Tail29 : List Nat := 17891 :: root657Tail30
def root657Tail28 : List Nat := 17597 :: root657Tail29
def root657Tail27 : List Nat := 16319 :: root657Tail28
def root657Tail26 : List Nat := 15569 :: root657Tail27
def root657Tail25 : List Nat := 14831 :: root657Tail26
def root657Tail24 : List Nat := 14747 :: root657Tail25
def root657Tail23 : List Nat := 13691 :: root657Tail24
def root657Tail22 : List Nat := 13523 :: root657Tail23
def root657Tail21 : List Nat := 13049 :: root657Tail22
def root657Tail20 : List Nat := 12983 :: root657Tail21
def root657Tail19 : List Nat := 12893 :: root657Tail20
def root657Tail18 : List Nat := 12161 :: root657Tail19
def root657Tail17 : List Nat := 11927 :: root657Tail18
def root657Tail16 : List Nat := 10589 :: root657Tail17
def root657Tail15 : List Nat := 9851 :: root657Tail16
def root657Tail14 : List Nat := 9521 :: root657Tail15
def root657Tail13 : List Nat := 9419 :: root657Tail14
def root657Tail12 : List Nat := 9221 :: root657Tail13
def root657Tail11 : List Nat := 8573 :: root657Tail12
def root657Tail10 : List Nat := 7919 :: root657Tail11
def root657Tail9 : List Nat := 7841 :: root657Tail10
def root657Tail8 : List Nat := 7649 :: root657Tail9
def root657Tail7 : List Nat := 7583 :: root657Tail8
def root657Tail6 : List Nat := 7247 :: root657Tail7
def root657Tail5 : List Nat := 7103 :: root657Tail6
def root657Tail4 : List Nat := 7043 :: root657Tail5
def root657Tail3 : List Nat := 7001 :: root657Tail4
def root657Tail2 : List Nat := 6491 :: root657Tail3
def root657Tail1 : List Nat := 6053 :: root657Tail2
def root657Tail0 : List Nat := 5399 :: root657Tail1
theorem root657Scan0 : scan edge (search edge 3) 4 [4937] 4937 root657Tail0 winner = winner := by rw [root657Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root657Branch : search edge 4 [4937] root657Candidates 4937 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4937] 4937 root657Tail0 winner = winner
  exact root657Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
