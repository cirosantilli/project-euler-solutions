import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root595Candidates : List Nat := Data.row4391
set_option maxRecDepth 8192 in
theorem root595Candidates_checked : candidates edge 5 0 4391 tail596 winner.bound = root595Candidates := by
  apply candidates_eq_row edge 5 0 4391 tail596 root595Candidates winner.bound tail596_ordered
  · decide
  · decide
  · funext q
    rfl
def root595Tail26 : List Nat := []
def root595Tail25 : List Nat := 21383 :: root595Tail26
def root595Tail24 : List Nat := 19997 :: root595Tail25
def root595Tail23 : List Nat := 18587 :: root595Tail24
def root595Tail22 : List Nat := 18521 :: root595Tail23
def root595Tail21 : List Nat := 17519 :: root595Tail22
def root595Tail20 : List Nat := 16361 :: root595Tail21
def root595Tail19 : List Nat := 16193 :: root595Tail20
def root595Tail18 : List Nat := 14867 :: root595Tail19
def root595Tail17 : List Nat := 14831 :: root595Tail18
def root595Tail16 : List Nat := 14813 :: root595Tail17
def root595Tail15 : List Nat := 13397 :: root595Tail16
def root595Tail14 : List Nat := 12647 :: root595Tail15
def root595Tail13 : List Nat := 12497 :: root595Tail14
def root595Tail12 : List Nat := 12203 :: root595Tail13
def root595Tail11 : List Nat := 11579 :: root595Tail12
def root595Tail10 : List Nat := 10499 :: root595Tail11
def root595Tail9 : List Nat := 10211 :: root595Tail10
def root595Tail8 : List Nat := 9539 :: root595Tail9
def root595Tail7 : List Nat := 8999 :: root595Tail8
def root595Tail6 : List Nat := 8783 :: root595Tail7
def root595Tail5 : List Nat := 8219 :: root595Tail6
def root595Tail4 : List Nat := 8111 :: root595Tail5
def root595Tail3 : List Nat := 7901 :: root595Tail4
def root595Tail2 : List Nat := 6869 :: root595Tail3
def root595Tail1 : List Nat := 6263 :: root595Tail2
def root595Tail0 : List Nat := 5639 :: root595Tail1
theorem root595Scan0 : scan edge (search edge 3) 4 [4391] 4391 root595Tail0 winner = winner := by rw [root595Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root595Branch : search edge 4 [4391] root595Candidates 4391 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4391] 4391 root595Tail0 winner = winner
  exact root595Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
