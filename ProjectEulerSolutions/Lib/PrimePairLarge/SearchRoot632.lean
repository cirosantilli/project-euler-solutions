import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root632Candidates : List Nat := Data.row4703
set_option maxRecDepth 8192 in
theorem root632Candidates_checked : candidates edge 5 0 4703 tail633 winner.bound = root632Candidates := by
  apply candidates_eq_row edge 5 0 4703 tail633 root632Candidates winner.bound tail633_ordered
  · decide
  · decide
  · funext q
    rfl
def root632Tail31 : List Nat := []
def root632Tail30 : List Nat := 21017 :: root632Tail31
def root632Tail29 : List Nat := 21011 :: root632Tail30
def root632Tail28 : List Nat := 20051 :: root632Tail29
def root632Tail27 : List Nat := 19991 :: root632Tail28
def root632Tail26 : List Nat := 19697 :: root632Tail27
def root632Tail25 : List Nat := 18869 :: root632Tail26
def root632Tail24 : List Nat := 17609 :: root632Tail25
def root632Tail23 : List Nat := 17483 :: root632Tail24
def root632Tail22 : List Nat := 16607 :: root632Tail23
def root632Tail21 : List Nat := 16127 :: root632Tail22
def root632Tail20 : List Nat := 15641 :: root632Tail21
def root632Tail19 : List Nat := 13751 :: root632Tail20
def root632Tail18 : List Nat := 13397 :: root632Tail19
def root632Tail17 : List Nat := 13331 :: root632Tail18
def root632Tail16 : List Nat := 11483 :: root632Tail17
def root632Tail15 : List Nat := 11057 :: root632Tail16
def root632Tail14 : List Nat := 9749 :: root632Tail15
def root632Tail13 : List Nat := 9743 :: root632Tail14
def root632Tail12 : List Nat := 9371 :: root632Tail13
def root632Tail11 : List Nat := 8837 :: root632Tail12
def root632Tail10 : List Nat := 8627 :: root632Tail11
def root632Tail9 : List Nat := 8543 :: root632Tail10
def root632Tail8 : List Nat := 8081 :: root632Tail9
def root632Tail7 : List Nat := 6653 :: root632Tail8
def root632Tail6 : List Nat := 6359 :: root632Tail7
def root632Tail5 : List Nat := 6329 :: root632Tail6
def root632Tail4 : List Nat := 6257 :: root632Tail5
def root632Tail3 : List Nat := 6221 :: root632Tail4
def root632Tail2 : List Nat := 6131 :: root632Tail3
def root632Tail1 : List Nat := 5657 :: root632Tail2
def root632Tail0 : List Nat := 5531 :: root632Tail1
theorem root632Scan0 : scan edge (search edge 3) 4 [4703] 4703 root632Tail0 winner = winner := by rw [root632Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root632Branch : search edge 4 [4703] root632Candidates 4703 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4703] 4703 root632Tail0 winner = winner
  exact root632Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
