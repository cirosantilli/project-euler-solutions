import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root620Candidates : List Nat := Data.row4603
set_option maxRecDepth 8192 in
theorem root620Candidates_checked : candidates edge 5 0 4603 tail621 winner.bound = root620Candidates := by
  apply candidates_eq_row edge 5 0 4603 tail621 root620Candidates winner.bound tail621_ordered
  · decide
  · decide
  · funext q
    rfl
def root620Tail27 : List Nat := []
def root620Tail26 : List Nat := 20533 :: root620Tail27
def root620Tail25 : List Nat := 20347 :: root620Tail26
def root620Tail24 : List Nat := 18583 :: root620Tail25
def root620Tail23 : List Nat := 18127 :: root620Tail24
def root620Tail22 : List Nat := 17707 :: root620Tail23
def root620Tail21 : List Nat := 17491 :: root620Tail22
def root620Tail20 : List Nat := 17257 :: root620Tail21
def root620Tail19 : List Nat := 16573 :: root620Tail20
def root620Tail18 : List Nat := 14983 :: root620Tail19
def root620Tail17 : List Nat := 13291 :: root620Tail18
def root620Tail16 : List Nat := 12841 :: root620Tail17
def root620Tail15 : List Nat := 12637 :: root620Tail16
def root620Tail14 : List Nat := 12421 :: root620Tail15
def root620Tail13 : List Nat := 11503 :: root620Tail14
def root620Tail12 : List Nat := 9901 :: root620Tail13
def root620Tail11 : List Nat := 9769 :: root620Tail12
def root620Tail10 : List Nat := 9601 :: root620Tail11
def root620Tail9 : List Nat := 9241 :: root620Tail10
def root620Tail8 : List Nat := 8761 :: root620Tail9
def root620Tail7 : List Nat := 8731 :: root620Tail8
def root620Tail6 : List Nat := 8599 :: root620Tail7
def root620Tail5 : List Nat := 7603 :: root620Tail6
def root620Tail4 : List Nat := 6577 :: root620Tail5
def root620Tail3 : List Nat := 6421 :: root620Tail4
def root620Tail2 : List Nat := 6379 :: root620Tail3
def root620Tail1 : List Nat := 6073 :: root620Tail2
def root620Tail0 : List Nat := 5569 :: root620Tail1
theorem root620Scan0 : scan edge (search edge 3) 4 [4603] 4603 root620Tail0 winner = winner := by rw [root620Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root620Branch : search edge 4 [4603] root620Candidates 4603 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4603] 4603 root620Tail0 winner = winner
  exact root620Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
