import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root679Candidates : List Nat := Data.row5101
set_option maxRecDepth 8192 in
theorem root679Candidates_checked : candidates edge 5 0 5101 tail680 winner.bound = root679Candidates := by
  apply candidates_eq_row edge 5 0 5101 tail680 root679Candidates winner.bound tail680_ordered
  · decide
  · decide
  · funext q
    rfl
def root679Tail39 : List Nat := []
def root679Tail38 : List Nat := 20029 :: root679Tail39
def root679Tail37 : List Nat := 19531 :: root679Tail38
def root679Tail36 : List Nat := 19501 :: root679Tail37
def root679Tail35 : List Nat := 19447 :: root679Tail36
def root679Tail34 : List Nat := 18691 :: root679Tail35
def root679Tail33 : List Nat := 18013 :: root679Tail34
def root679Tail32 : List Nat := 17737 :: root679Tail33
def root679Tail31 : List Nat := 17359 :: root679Tail32
def root679Tail30 : List Nat := 17137 :: root679Tail31
def root679Tail29 : List Nat := 17107 :: root679Tail30
def root679Tail28 : List Nat := 16903 :: root679Tail29
def root679Tail27 : List Nat := 16843 :: root679Tail28
def root679Tail26 : List Nat := 16561 :: root679Tail27
def root679Tail25 : List Nat := 16339 :: root679Tail26
def root679Tail24 : List Nat := 16141 :: root679Tail25
def root679Tail23 : List Nat := 15913 :: root679Tail24
def root679Tail22 : List Nat := 15391 :: root679Tail23
def root679Tail21 : List Nat := 15241 :: root679Tail22
def root679Tail20 : List Nat := 15031 :: root679Tail21
def root679Tail19 : List Nat := 14887 :: root679Tail20
def root679Tail18 : List Nat := 14821 :: root679Tail19
def root679Tail17 : List Nat := 14551 :: root679Tail18
def root679Tail16 : List Nat := 13789 :: root679Tail17
def root679Tail15 : List Nat := 12853 :: root679Tail16
def root679Tail14 : List Nat := 12613 :: root679Tail15
def root679Tail13 : List Nat := 12553 :: root679Tail14
def root679Tail12 : List Nat := 10831 :: root679Tail13
def root679Tail11 : List Nat := 10429 :: root679Tail12
def root679Tail10 : List Nat := 10369 :: root679Tail11
def root679Tail9 : List Nat := 9931 :: root679Tail10
def root679Tail8 : List Nat := 8293 :: root679Tail9
def root679Tail7 : List Nat := 8263 :: root679Tail8
def root679Tail6 : List Nat := 7237 :: root679Tail7
def root679Tail5 : List Nat := 7069 :: root679Tail6
def root679Tail4 : List Nat := 6883 :: root679Tail5
def root679Tail3 : List Nat := 6571 :: root679Tail4
def root679Tail2 : List Nat := 6397 :: root679Tail3
def root679Tail1 : List Nat := 6151 :: root679Tail2
def root679Tail0 : List Nat := 5953 :: root679Tail1
theorem root679Scan0 : scan edge (search edge 3) 4 [5101] 5101 root679Tail0 winner = winner := by rw [root679Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root679Branch : search edge 4 [5101] root679Candidates 5101 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5101] 5101 root679Tail0 winner = winner
  exact root679Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
