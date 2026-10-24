import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root687Candidates : List Nat := Data.row5179
set_option maxRecDepth 8192 in
theorem root687Candidates_checked : candidates edge 5 0 5179 tail688 winner.bound = root687Candidates := by
  apply candidates_eq_row edge 5 0 5179 tail688 root687Candidates winner.bound tail688_ordered
  · decide
  · decide
  · funext q
    rfl
def root687Tail27 : List Nat := []
def root687Tail26 : List Nat := 20707 :: root687Tail27
def root687Tail25 : List Nat := 20593 :: root687Tail26
def root687Tail24 : List Nat := 20269 :: root687Tail25
def root687Tail23 : List Nat := 20023 :: root687Tail24
def root687Tail22 : List Nat := 19069 :: root687Tail23
def root687Tail21 : List Nat := 19009 :: root687Tail22
def root687Tail20 : List Nat := 17419 :: root687Tail21
def root687Tail19 : List Nat := 17317 :: root687Tail20
def root687Tail18 : List Nat := 15859 :: root687Tail19
def root687Tail17 : List Nat := 14083 :: root687Tail18
def root687Tail16 : List Nat := 14029 :: root687Tail17
def root687Tail15 : List Nat := 12841 :: root687Tail16
def root687Tail14 : List Nat := 11491 :: root687Tail15
def root687Tail13 : List Nat := 11437 :: root687Tail14
def root687Tail12 : List Nat := 11287 :: root687Tail13
def root687Tail11 : List Nat := 9661 :: root687Tail12
def root687Tail10 : List Nat := 9619 :: root687Tail11
def root687Tail9 : List Nat := 9439 :: root687Tail10
def root687Tail8 : List Nat := 9337 :: root687Tail9
def root687Tail7 : List Nat := 9043 :: root687Tail8
def root687Tail6 : List Nat := 8863 :: root687Tail7
def root687Tail5 : List Nat := 8209 :: root687Tail6
def root687Tail4 : List Nat := 7873 :: root687Tail5
def root687Tail3 : List Nat := 6763 :: root687Tail4
def root687Tail2 : List Nat := 6301 :: root687Tail3
def root687Tail1 : List Nat := 5851 :: root687Tail2
def root687Tail0 : List Nat := 5683 :: root687Tail1
theorem root687Scan0 : scan edge (search edge 3) 4 [5179] 5179 root687Tail0 winner = winner := by rw [root687Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root687Branch : search edge 4 [5179] root687Candidates 5179 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5179] 5179 root687Tail0 winner = winner
  exact root687Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
