import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root614Candidates : List Nat := Data.row4549
set_option maxRecDepth 8192 in
theorem root614Candidates_checked : candidates edge 5 0 4549 tail615 winner.bound = root614Candidates := by
  apply candidates_eq_row edge 5 0 4549 tail615 root614Candidates winner.bound tail615_ordered
  · decide
  · decide
  · funext q
    rfl
def root614Tail33 : List Nat := []
def root614Tail32 : List Nat := 19753 :: root614Tail33
def root614Tail31 : List Nat := 19681 :: root614Tail32
def root614Tail30 : List Nat := 19417 :: root614Tail31
def root614Tail29 : List Nat := 18301 :: root614Tail30
def root614Tail28 : List Nat := 17707 :: root614Tail29
def root614Tail27 : List Nat := 17401 :: root614Tail28
def root614Tail26 : List Nat := 17041 :: root614Tail27
def root614Tail25 : List Nat := 16381 :: root614Tail26
def root614Tail24 : List Nat := 16339 :: root614Tail25
def root614Tail23 : List Nat := 15859 :: root614Tail24
def root614Tail22 : List Nat := 15667 :: root614Tail23
def root614Tail21 : List Nat := 15277 :: root614Tail22
def root614Tail20 : List Nat := 14461 :: root614Tail21
def root614Tail19 : List Nat := 14011 :: root614Tail20
def root614Tail18 : List Nat := 12967 :: root614Tail19
def root614Tail17 : List Nat := 12547 :: root614Tail18
def root614Tail16 : List Nat := 12109 :: root614Tail17
def root614Tail15 : List Nat := 11239 :: root614Tail16
def root614Tail14 : List Nat := 11161 :: root614Tail15
def root614Tail13 : List Nat := 10273 :: root614Tail14
def root614Tail12 : List Nat := 9631 :: root614Tail13
def root614Tail11 : List Nat := 9133 :: root614Tail12
def root614Tail10 : List Nat := 9127 :: root614Tail11
def root614Tail9 : List Nat := 8287 :: root614Tail10
def root614Tail8 : List Nat := 7717 :: root614Tail9
def root614Tail7 : List Nat := 6571 :: root614Tail8
def root614Tail6 : List Nat := 6361 :: root614Tail7
def root614Tail5 : List Nat := 6343 :: root614Tail6
def root614Tail4 : List Nat := 6151 :: root614Tail5
def root614Tail3 : List Nat := 6133 :: root614Tail4
def root614Tail2 : List Nat := 5851 :: root614Tail3
def root614Tail1 : List Nat := 5839 :: root614Tail2
def root614Tail0 : List Nat := 5563 :: root614Tail1
theorem root614Scan0 : scan edge (search edge 3) 4 [4549] 4549 root614Tail0 winner = winner := by rw [root614Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root614Branch : search edge 4 [4549] root614Candidates 4549 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4549] 4549 root614Tail0 winner = winner
  exact root614Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
