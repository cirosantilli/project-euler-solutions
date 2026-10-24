import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root650Candidates : List Nat := Data.row4877
set_option maxRecDepth 8192 in
theorem root650Candidates_checked : candidates edge 5 0 4877 tail651 winner.bound = root650Candidates := by
  apply candidates_eq_row edge 5 0 4877 tail651 root650Candidates winner.bound tail651_ordered
  · decide
  · decide
  · funext q
    rfl
def root650Tail40 : List Nat := []
def root650Tail39 : List Nat := 20879 :: root650Tail40
def root650Tail38 : List Nat := 20771 :: root650Tail39
def root650Tail37 : List Nat := 20627 :: root650Tail38
def root650Tail36 : List Nat := 20393 :: root650Tail37
def root650Tail35 : List Nat := 19319 :: root650Tail36
def root650Tail34 : List Nat := 18461 :: root650Tail35
def root650Tail33 : List Nat := 17729 :: root650Tail34
def root650Tail32 : List Nat := 17627 :: root650Tail33
def root650Tail31 : List Nat := 16883 :: root650Tail32
def root650Tail30 : List Nat := 16631 :: root650Tail31
def root650Tail29 : List Nat := 14939 :: root650Tail30
def root650Tail28 : List Nat := 14891 :: root650Tail29
def root650Tail27 : List Nat := 14771 :: root650Tail28
def root650Tail26 : List Nat := 13103 :: root650Tail27
def root650Tail25 : List Nat := 12809 :: root650Tail26
def root650Tail24 : List Nat := 12143 :: root650Tail25
def root650Tail23 : List Nat := 11681 :: root650Tail24
def root650Tail22 : List Nat := 11177 :: root650Tail23
def root650Tail21 : List Nat := 11087 :: root650Tail22
def root650Tail20 : List Nat := 9161 :: root650Tail21
def root650Tail19 : List Nat := 8573 :: root650Tail20
def root650Tail18 : List Nat := 8291 :: root650Tail19
def root650Tail17 : List Nat := 8231 :: root650Tail18
def root650Tail16 : List Nat := 8123 :: root650Tail17
def root650Tail15 : List Nat := 7643 :: root650Tail16
def root650Tail14 : List Nat := 7559 :: root650Tail15
def root650Tail13 : List Nat := 7349 :: root650Tail14
def root650Tail12 : List Nat := 7307 :: root650Tail13
def root650Tail11 : List Nat := 6977 :: root650Tail12
def root650Tail10 : List Nat := 6947 :: root650Tail11
def root650Tail9 : List Nat := 6719 :: root650Tail10
def root650Tail8 : List Nat := 6551 :: root650Tail9
def root650Tail7 : List Nat := 6359 :: root650Tail8
def root650Tail6 : List Nat := 6257 :: root650Tail7
def root650Tail5 : List Nat := 5981 :: root650Tail6
def root650Tail4 : List Nat := 5813 :: root650Tail5
def root650Tail3 : List Nat := 5711 :: root650Tail4
def root650Tail2 : List Nat := 5693 :: root650Tail3
def root650Tail1 : List Nat := 5501 :: root650Tail2
def root650Tail0 : List Nat := 5171 :: root650Tail1
def root650Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root650Candidates0_checked : candidates edge 4 4877 5171 root650Tail1 winner.bound = root650Candidates0 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root650Step0 : scan edge (search edge 3) 4 [4877] 4877 root650Tail0 winner =
    scan edge (search edge 3) 4 [4877] 4877 root650Tail1 winner := by
  rw [root650Tail0, scan, if_neg (by decide), root650Candidates0_checked]
  change scan edge (search edge 3) 4 [4877] 4877 root650Tail1
    (if 3 ≤ root650Candidates0.length then search edge 3 [4877, 5171] root650Candidates0 10048 winner else winner) = _
  rw [if_neg (by decide)]
theorem root650Scan1 : scan edge (search edge 3) 4 [4877] 4877 root650Tail1 winner = winner := by rw [root650Tail1, scan, if_pos (by decide)]
theorem root650Scan0 : scan edge (search edge 3) 4 [4877] 4877 root650Tail0 winner = winner :=
  (root650Step0).trans root650Scan1
set_option maxRecDepth 4096 in
theorem root650Branch : search edge 4 [4877] root650Candidates 4877 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4877] 4877 root650Tail0 winner = winner
  exact root650Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
