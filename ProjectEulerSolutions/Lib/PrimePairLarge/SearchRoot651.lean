import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root651Candidates : List Nat := Data.row4889
set_option maxRecDepth 8192 in
theorem root651Candidates_checked : candidates edge 5 0 4889 tail652 winner.bound = root651Candidates := by
  apply candidates_eq_row edge 5 0 4889 tail652 root651Candidates winner.bound tail652_ordered
  · decide
  · decide
  · funext q
    rfl
def root651Tail24 : List Nat := []
def root651Tail23 : List Nat := 20249 :: root651Tail24
def root651Tail22 : List Nat := 18149 :: root651Tail23
def root651Tail21 : List Nat := 17987 :: root651Tail22
def root651Tail20 : List Nat := 17609 :: root651Tail21
def root651Tail19 : List Nat := 15137 :: root651Tail20
def root651Tail18 : List Nat := 15101 :: root651Tail19
def root651Tail17 : List Nat := 14843 :: root651Tail18
def root651Tail16 : List Nat := 13883 :: root651Tail17
def root651Tail15 : List Nat := 13781 :: root651Tail16
def root651Tail14 : List Nat := 12101 :: root651Tail15
def root651Tail13 : List Nat := 11987 :: root651Tail14
def root651Tail12 : List Nat := 11903 :: root651Tail13
def root651Tail11 : List Nat := 11597 :: root651Tail12
def root651Tail10 : List Nat := 9833 :: root651Tail11
def root651Tail9 : List Nat := 9743 :: root651Tail10
def root651Tail8 : List Nat := 8237 :: root651Tail9
def root651Tail7 : List Nat := 8123 :: root651Tail8
def root651Tail6 : List Nat := 7643 :: root651Tail7
def root651Tail5 : List Nat := 7481 :: root651Tail6
def root651Tail4 : List Nat := 7187 :: root651Tail5
def root651Tail3 : List Nat := 6311 :: root651Tail4
def root651Tail2 : List Nat := 6053 :: root651Tail3
def root651Tail1 : List Nat := 5669 :: root651Tail2
def root651Tail0 : List Nat := 5333 :: root651Tail1
theorem root651Scan0 : scan edge (search edge 3) 4 [4889] 4889 root651Tail0 winner = winner := by rw [root651Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root651Branch : search edge 4 [4889] root651Candidates 4889 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4889] 4889 root651Tail0 winner = winner
  exact root651Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
