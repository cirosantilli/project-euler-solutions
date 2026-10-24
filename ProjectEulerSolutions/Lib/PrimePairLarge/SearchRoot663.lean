import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root663Candidates : List Nat := Data.row4973
set_option maxRecDepth 8192 in
theorem root663Candidates_checked : candidates edge 5 0 4973 tail664 winner.bound = root663Candidates := by
  apply candidates_eq_row edge 5 0 4973 tail664 root663Candidates winner.bound tail664_ordered
  · decide
  · decide
  · funext q
    rfl
def root663Tail36 : List Nat := []
def root663Tail35 : List Nat := 20051 :: root663Tail36
def root663Tail34 : List Nat := 19913 :: root663Tail35
def root663Tail33 : List Nat := 19583 :: root663Tail34
def root663Tail32 : List Nat := 19421 :: root663Tail33
def root663Tail31 : List Nat := 19157 :: root663Tail32
def root663Tail30 : List Nat := 18587 :: root663Tail31
def root663Tail29 : List Nat := 18329 :: root663Tail30
def root663Tail28 : List Nat := 17939 :: root663Tail29
def root663Tail27 : List Nat := 17783 :: root663Tail28
def root663Tail26 : List Nat := 17519 :: root663Tail27
def root663Tail25 : List Nat := 17033 :: root663Tail26
def root663Tail24 : List Nat := 16529 :: root663Tail25
def root663Tail23 : List Nat := 16427 :: root663Tail24
def root663Tail22 : List Nat := 16349 :: root663Tail23
def root663Tail21 : List Nat := 16319 :: root663Tail22
def root663Tail20 : List Nat := 15629 :: root663Tail21
def root663Tail19 : List Nat := 14087 :: root663Tail20
def root663Tail18 : List Nat := 13841 :: root663Tail19
def root663Tail17 : List Nat := 13757 :: root663Tail18
def root663Tail16 : List Nat := 12941 :: root663Tail17
def root663Tail15 : List Nat := 11699 :: root663Tail16
def root663Tail14 : List Nat := 11261 :: root663Tail15
def root663Tail13 : List Nat := 10631 :: root663Tail14
def root663Tail12 : List Nat := 10601 :: root663Tail13
def root663Tail11 : List Nat := 10433 :: root663Tail12
def root663Tail10 : List Nat := 10223 :: root663Tail11
def root663Tail9 : List Nat := 10133 :: root663Tail10
def root663Tail8 : List Nat := 9629 :: root663Tail9
def root663Tail7 : List Nat := 9413 :: root663Tail8
def root663Tail6 : List Nat := 7901 :: root663Tail7
def root663Tail5 : List Nat := 7829 :: root663Tail6
def root663Tail4 : List Nat := 6569 :: root663Tail5
def root663Tail3 : List Nat := 6389 :: root663Tail4
def root663Tail2 : List Nat := 6221 :: root663Tail3
def root663Tail1 : List Nat := 5717 :: root663Tail2
def root663Tail0 : List Nat := 5303 :: root663Tail1
theorem root663Scan0 : scan edge (search edge 3) 4 [4973] 4973 root663Tail0 winner = winner := by rw [root663Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root663Branch : search edge 4 [4973] root663Candidates 4973 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4973] 4973 root663Tail0 winner = winner
  exact root663Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
