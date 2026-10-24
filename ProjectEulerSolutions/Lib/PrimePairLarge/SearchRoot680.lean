import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root680Candidates : List Nat := Data.row5107
set_option maxRecDepth 8192 in
theorem root680Candidates_checked : candidates edge 5 0 5107 tail681 winner.bound = root680Candidates := by
  apply candidates_eq_row edge 5 0 5107 tail681 root680Candidates winner.bound tail681_ordered
  · decide
  · decide
  · funext q
    rfl
def root680Tail28 : List Nat := []
def root680Tail27 : List Nat := 20269 :: root680Tail28
def root680Tail26 : List Nat := 19801 :: root680Tail27
def root680Tail25 : List Nat := 18379 :: root680Tail26
def root680Tail24 : List Nat := 17959 :: root680Tail25
def root680Tail23 : List Nat := 17497 :: root680Tail24
def root680Tail22 : List Nat := 17341 :: root680Tail23
def root680Tail21 : List Nat := 17299 :: root680Tail22
def root680Tail20 : List Nat := 16417 :: root680Tail21
def root680Tail19 : List Nat := 16231 :: root680Tail20
def root680Tail18 : List Nat := 16033 :: root680Tail19
def root680Tail17 : List Nat := 14551 :: root680Tail18
def root680Tail16 : List Nat := 14107 :: root680Tail17
def root680Tail15 : List Nat := 13297 :: root680Tail16
def root680Tail14 : List Nat := 12721 :: root680Tail15
def root680Tail13 : List Nat := 12373 :: root680Tail14
def root680Tail12 : List Nat := 10987 :: root680Tail13
def root680Tail11 : List Nat := 10099 :: root680Tail12
def root680Tail10 : List Nat := 9901 :: root680Tail11
def root680Tail9 : List Nat := 9403 :: root680Tail10
def root680Tail8 : List Nat := 8641 :: root680Tail9
def root680Tail7 : List Nat := 8311 :: root680Tail8
def root680Tail6 : List Nat := 8017 :: root680Tail7
def root680Tail5 : List Nat := 7879 :: root680Tail6
def root680Tail4 : List Nat := 7717 :: root680Tail5
def root680Tail3 : List Nat := 6793 :: root680Tail4
def root680Tail2 : List Nat := 6763 :: root680Tail3
def root680Tail1 : List Nat := 5839 :: root680Tail2
def root680Tail0 : List Nat := 5779 :: root680Tail1
theorem root680Scan0 : scan edge (search edge 3) 4 [5107] 5107 root680Tail0 winner = winner := by rw [root680Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root680Branch : search edge 4 [5107] root680Candidates 5107 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5107] 5107 root680Tail0 winner = winner
  exact root680Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
