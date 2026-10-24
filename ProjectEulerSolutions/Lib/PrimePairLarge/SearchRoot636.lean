import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root636Candidates : List Nat := Data.row4733
set_option maxRecDepth 8192 in
theorem root636Candidates_checked : candidates edge 5 0 4733 tail637 winner.bound = root636Candidates := by
  apply candidates_eq_row edge 5 0 4733 tail637 root636Candidates winner.bound tail637_ordered
  · decide
  · decide
  · funext q
    rfl
def root636Tail35 : List Nat := []
def root636Tail34 : List Nat := 20873 :: root636Tail35
def root636Tail33 : List Nat := 19961 :: root636Tail34
def root636Tail32 : List Nat := 19853 :: root636Tail33
def root636Tail31 : List Nat := 19433 :: root636Tail32
def root636Tail30 : List Nat := 18617 :: root636Tail31
def root636Tail29 : List Nat := 18413 :: root636Tail30
def root636Tail28 : List Nat := 17909 :: root636Tail29
def root636Tail27 : List Nat := 16223 :: root636Tail28
def root636Tail26 : List Nat := 16001 :: root636Tail27
def root636Tail25 : List Nat := 15959 :: root636Tail26
def root636Tail24 : List Nat := 15527 :: root636Tail25
def root636Tail23 : List Nat := 15329 :: root636Tail24
def root636Tail22 : List Nat := 14207 :: root636Tail23
def root636Tail21 : List Nat := 13469 :: root636Tail22
def root636Tail20 : List Nat := 13187 :: root636Tail21
def root636Tail19 : List Nat := 12923 :: root636Tail20
def root636Tail18 : List Nat := 12821 :: root636Tail19
def root636Tail17 : List Nat := 11903 :: root636Tail18
def root636Tail16 : List Nat := 10289 :: root636Tail17
def root636Tail15 : List Nat := 10259 :: root636Tail16
def root636Tail14 : List Nat := 10037 :: root636Tail15
def root636Tail13 : List Nat := 9437 :: root636Tail14
def root636Tail12 : List Nat := 9029 :: root636Tail13
def root636Tail11 : List Nat := 8849 :: root636Tail12
def root636Tail10 : List Nat := 7919 :: root636Tail11
def root636Tail9 : List Nat := 7907 :: root636Tail10
def root636Tail8 : List Nat := 7793 :: root636Tail9
def root636Tail7 : List Nat := 7517 :: root636Tail8
def root636Tail6 : List Nat := 7331 :: root636Tail7
def root636Tail5 : List Nat := 7211 :: root636Tail6
def root636Tail4 : List Nat := 7079 :: root636Tail5
def root636Tail3 : List Nat := 6911 :: root636Tail4
def root636Tail2 : List Nat := 6689 :: root636Tail3
def root636Tail1 : List Nat := 6269 :: root636Tail2
def root636Tail0 : List Nat := 5237 :: root636Tail1
def root636Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root636Candidates0_checked : candidates edge 4 4733 5237 root636Tail1 winner.bound = root636Candidates0 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root636Step0 : scan edge (search edge 3) 4 [4733] 4733 root636Tail0 winner =
    scan edge (search edge 3) 4 [4733] 4733 root636Tail1 winner := by
  rw [root636Tail0, scan, if_neg (by decide), root636Candidates0_checked]
  change scan edge (search edge 3) 4 [4733] 4733 root636Tail1
    (if 3 ≤ root636Candidates0.length then search edge 3 [4733, 5237] root636Candidates0 9970 winner else winner) = _
  rw [if_neg (by decide)]
theorem root636Scan1 : scan edge (search edge 3) 4 [4733] 4733 root636Tail1 winner = winner := by rw [root636Tail1, scan, if_pos (by decide)]
theorem root636Scan0 : scan edge (search edge 3) 4 [4733] 4733 root636Tail0 winner = winner :=
  (root636Step0).trans root636Scan1
set_option maxRecDepth 4096 in
theorem root636Branch : search edge 4 [4733] root636Candidates 4733 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4733] 4733 root636Tail0 winner = winner
  exact root636Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
