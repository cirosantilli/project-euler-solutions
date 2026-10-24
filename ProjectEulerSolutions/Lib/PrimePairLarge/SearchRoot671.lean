import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root671Candidates : List Nat := Data.row5023
set_option maxRecDepth 8192 in
theorem root671Candidates_checked : candidates edge 5 0 5023 tail672 winner.bound = root671Candidates := by
  apply candidates_eq_row edge 5 0 5023 tail672 root671Candidates winner.bound tail672_ordered
  · decide
  · decide
  · funext q
    rfl
def root671Tail33 : List Nat := []
def root671Tail32 : List Nat := 20983 :: root671Tail33
def root671Tail31 : List Nat := 20011 :: root671Tail32
def root671Tail30 : List Nat := 18211 :: root671Tail31
def root671Tail29 : List Nat := 18133 :: root671Tail30
def root671Tail28 : List Nat := 17509 :: root671Tail29
def root671Tail27 : List Nat := 17239 :: root671Tail28
def root671Tail26 : List Nat := 16993 :: root671Tail27
def root671Tail25 : List Nat := 16651 :: root671Tail26
def root671Tail24 : List Nat := 16447 :: root671Tail25
def root671Tail23 : List Nat := 16417 :: root671Tail24
def root671Tail22 : List Nat := 15277 :: root671Tail23
def root671Tail21 : List Nat := 14767 :: root671Tail22
def root671Tail20 : List Nat := 13687 :: root671Tail21
def root671Tail19 : List Nat := 13477 :: root671Tail20
def root671Tail18 : List Nat := 13177 :: root671Tail19
def root671Tail17 : List Nat := 12457 :: root671Tail18
def root671Tail16 : List Nat := 11959 :: root671Tail17
def root671Tail15 : List Nat := 11827 :: root671Tail16
def root671Tail14 : List Nat := 11119 :: root671Tail15
def root671Tail13 : List Nat := 11071 :: root671Tail14
def root671Tail12 : List Nat := 10903 :: root671Tail13
def root671Tail11 : List Nat := 10687 :: root671Tail12
def root671Tail10 : List Nat := 9949 :: root671Tail11
def root671Tail9 : List Nat := 9907 :: root671Tail10
def root671Tail8 : List Nat := 9769 :: root671Tail9
def root671Tail7 : List Nat := 8779 :: root671Tail8
def root671Tail6 : List Nat := 8269 :: root671Tail7
def root671Tail5 : List Nat := 8059 :: root671Tail6
def root671Tail4 : List Nat := 7753 :: root671Tail5
def root671Tail3 : List Nat := 7477 :: root671Tail4
def root671Tail2 : List Nat := 7039 :: root671Tail3
def root671Tail1 : List Nat := 6841 :: root671Tail2
def root671Tail0 : List Nat := 5443 :: root671Tail1
theorem root671Scan0 : scan edge (search edge 3) 4 [5023] 5023 root671Tail0 winner = winner := by rw [root671Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root671Branch : search edge 4 [5023] root671Candidates 5023 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5023] 5023 root671Tail0 winner = winner
  exact root671Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
