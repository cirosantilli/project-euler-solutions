import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root689Candidates : List Nat := Data.row5197
set_option maxRecDepth 8192 in
theorem root689Candidates_checked : candidates edge 5 0 5197 tail690 winner.bound = root689Candidates := by
  apply candidates_eq_row edge 5 0 5197 tail690 root689Candidates winner.bound tail690_ordered
  · decide
  · decide
  · funext q
    rfl
def root689Tail31 : List Nat := []
def root689Tail30 : List Nat := 20599 :: root689Tail31
def root689Tail29 : List Nat := 20347 :: root689Tail30
def root689Tail28 : List Nat := 20101 :: root689Tail29
def root689Tail27 : List Nat := 19441 :: root689Tail28
def root689Tail26 : List Nat := 19381 :: root689Tail27
def root689Tail25 : List Nat := 18553 :: root689Tail26
def root689Tail24 : List Nat := 17791 :: root689Tail25
def root689Tail23 : List Nat := 15451 :: root689Tail24
def root689Tail22 : List Nat := 15277 :: root689Tail23
def root689Tail21 : List Nat := 15013 :: root689Tail22
def root689Tail20 : List Nat := 14653 :: root689Tail21
def root689Tail19 : List Nat := 14341 :: root689Tail20
def root689Tail18 : List Nat := 13933 :: root689Tail19
def root689Tail17 : List Nat := 12421 :: root689Tail18
def root689Tail16 : List Nat := 12409 :: root689Tail17
def root689Tail15 : List Nat := 12289 :: root689Tail16
def root689Tail14 : List Nat := 11443 :: root689Tail15
def root689Tail13 : List Nat := 10831 :: root689Tail14
def root689Tail12 : List Nat := 10729 :: root689Tail13
def root689Tail11 : List Nat := 10657 :: root689Tail12
def root689Tail10 : List Nat := 9679 :: root689Tail11
def root689Tail9 : List Nat := 8929 :: root689Tail10
def root689Tail8 : List Nat := 8629 :: root689Tail9
def root689Tail7 : List Nat := 8389 :: root689Tail8
def root689Tail6 : List Nat := 8221 :: root689Tail7
def root689Tail5 : List Nat := 8209 :: root689Tail6
def root689Tail4 : List Nat := 7963 :: root689Tail5
def root689Tail3 : List Nat := 7591 :: root689Tail4
def root689Tail2 : List Nat := 6997 :: root689Tail3
def root689Tail1 : List Nat := 6733 :: root689Tail2
def root689Tail0 : List Nat := 5701 :: root689Tail1
theorem root689Scan0 : scan edge (search edge 3) 4 [5197] 5197 root689Tail0 winner = winner := by rw [root689Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root689Branch : search edge 4 [5197] root689Candidates 5197 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5197] 5197 root689Tail0 winner = winner
  exact root689Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
