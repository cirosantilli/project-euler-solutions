import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root598Candidates : List Nat := Data.row4421
set_option maxRecDepth 8192 in
theorem root598Candidates_checked : candidates edge 5 0 4421 tail599 winner.bound = root598Candidates := by
  apply candidates_eq_row edge 5 0 4421 tail599 root598Candidates winner.bound tail599_ordered
  · decide
  · decide
  · funext q
    rfl
def root598Tail37 : List Nat := []
def root598Tail36 : List Nat := 20771 :: root598Tail37
def root598Tail35 : List Nat := 20759 :: root598Tail36
def root598Tail34 : List Nat := 19763 :: root598Tail35
def root598Tail33 : List Nat := 19661 :: root598Tail34
def root598Tail32 : List Nat := 18947 :: root598Tail33
def root598Tail31 : List Nat := 18671 :: root598Tail32
def root598Tail30 : List Nat := 18617 :: root598Tail31
def root598Tail29 : List Nat := 18077 :: root598Tail30
def root598Tail28 : List Nat := 16931 :: root598Tail29
def root598Tail27 : List Nat := 16427 :: root598Tail28
def root598Tail26 : List Nat := 15803 :: root598Tail27
def root598Tail25 : List Nat := 15629 :: root598Tail26
def root598Tail24 : List Nat := 15101 :: root598Tail25
def root598Tail23 : List Nat := 14243 :: root598Tail24
def root598Tail22 : List Nat := 13241 :: root598Tail23
def root598Tail21 : List Nat := 13163 :: root598Tail22
def root598Tail20 : List Nat := 12401 :: root598Tail21
def root598Tail19 : List Nat := 11597 :: root598Tail20
def root598Tail18 : List Nat := 11471 :: root598Tail19
def root598Tail17 : List Nat := 11393 :: root598Tail18
def root598Tail16 : List Nat := 11279 :: root598Tail17
def root598Tail15 : List Nat := 10883 :: root598Tail16
def root598Tail14 : List Nat := 10631 :: root598Tail15
def root598Tail13 : List Nat := 10463 :: root598Tail14
def root598Tail12 : List Nat := 10139 :: root598Tail13
def root598Tail11 : List Nat := 9887 :: root598Tail12
def root598Tail10 : List Nat := 9803 :: root598Tail11
def root598Tail9 : List Nat := 9587 :: root598Tail10
def root598Tail8 : List Nat := 8039 :: root598Tail9
def root598Tail7 : List Nat := 7673 :: root598Tail8
def root598Tail6 : List Nat := 7577 :: root598Tail7
def root598Tail5 : List Nat := 6899 :: root598Tail6
def root598Tail4 : List Nat := 6869 :: root598Tail5
def root598Tail3 : List Nat := 6791 :: root598Tail4
def root598Tail2 : List Nat := 6563 :: root598Tail3
def root598Tail1 : List Nat := 6113 :: root598Tail2
def root598Tail0 : List Nat := 5783 :: root598Tail1
theorem root598Scan0 : scan edge (search edge 3) 4 [4421] 4421 root598Tail0 winner = winner := by rw [root598Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root598Branch : search edge 4 [4421] root598Candidates 4421 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4421] 4421 root598Tail0 winner = winner
  exact root598Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
