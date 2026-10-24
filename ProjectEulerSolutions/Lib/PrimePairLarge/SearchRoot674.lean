import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root674Candidates : List Nat := Data.row5059
set_option maxRecDepth 8192 in
theorem root674Candidates_checked : candidates edge 5 0 5059 tail675 winner.bound = root674Candidates := by
  apply candidates_eq_row edge 5 0 5059 tail675 root674Candidates winner.bound tail675_ordered
  · decide
  · decide
  · funext q
    rfl
def root674Tail33 : List Nat := []
def root674Tail32 : List Nat := 20707 :: root674Tail33
def root674Tail31 : List Nat := 20047 :: root674Tail32
def root674Tail30 : List Nat := 19501 :: root674Tail31
def root674Tail29 : List Nat := 19087 :: root674Tail30
def root674Tail28 : List Nat := 18313 :: root674Tail29
def root674Tail27 : List Nat := 18223 :: root674Tail28
def root674Tail26 : List Nat := 16693 :: root674Tail27
def root674Tail25 : List Nat := 16381 :: root674Tail26
def root674Tail24 : List Nat := 15601 :: root674Tail25
def root674Tail23 : List Nat := 14551 :: root674Tail24
def root674Tail22 : List Nat := 13921 :: root674Tail23
def root674Tail21 : List Nat := 13711 :: root674Tail22
def root674Tail20 : List Nat := 13159 :: root674Tail21
def root674Tail19 : List Nat := 12907 :: root674Tail20
def root674Tail18 : List Nat := 12073 :: root674Tail19
def root674Tail17 : List Nat := 11437 :: root674Tail18
def root674Tail16 : List Nat := 11299 :: root674Tail17
def root674Tail15 : List Nat := 10789 :: root674Tail16
def root674Tail14 : List Nat := 10729 :: root674Tail15
def root674Tail13 : List Nat := 10711 :: root674Tail14
def root674Tail12 : List Nat := 10687 :: root674Tail13
def root674Tail11 : List Nat := 10039 :: root674Tail12
def root674Tail10 : List Nat := 9649 :: root674Tail11
def root674Tail9 : List Nat := 9613 :: root674Tail10
def root674Tail8 : List Nat := 9319 :: root674Tail9
def root674Tail7 : List Nat := 9187 :: root674Tail8
def root674Tail6 : List Nat := 9127 :: root674Tail7
def root674Tail5 : List Nat := 9007 :: root674Tail6
def root674Tail4 : List Nat := 8269 :: root674Tail5
def root674Tail3 : List Nat := 7213 :: root674Tail4
def root674Tail2 : List Nat := 6691 :: root674Tail3
def root674Tail1 : List Nat := 5659 :: root674Tail2
def root674Tail0 : List Nat := 5323 :: root674Tail1
theorem root674Scan0 : scan edge (search edge 3) 4 [5059] 5059 root674Tail0 winner = winner := by rw [root674Tail0, scan, if_pos (by decide)]
set_option maxRecDepth 4096 in
theorem root674Branch : search edge 4 [5059] root674Candidates 5059 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5059] 5059 root674Tail0 winner = winner
  exact root674Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
