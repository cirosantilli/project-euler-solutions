import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root640Candidates : List Nat := Data.row4787
set_option maxRecDepth 8192 in
theorem root640Candidates_checked : candidates edge 5 0 4787 tail641 winner.bound = root640Candidates := by
  apply candidates_eq_row edge 5 0 4787 tail641 root640Candidates winner.bound tail641_ordered
  · decide
  · decide
  · funext q
    rfl
def root640Tail35 : List Nat := []
def root640Tail34 : List Nat := 20639 :: root640Tail35
def root640Tail33 : List Nat := 20147 :: root640Tail34
def root640Tail32 : List Nat := 19751 :: root640Tail33
def root640Tail31 : List Nat := 19289 :: root640Tail32
def root640Tail30 : List Nat := 19139 :: root640Tail31
def root640Tail29 : List Nat := 18593 :: root640Tail30
def root640Tail28 : List Nat := 17417 :: root640Tail29
def root640Tail27 : List Nat := 17387 :: root640Tail28
def root640Tail26 : List Nat := 17333 :: root640Tail27
def root640Tail25 : List Nat := 15641 :: root640Tail26
def root640Tail24 : List Nat := 14699 :: root640Tail25
def root640Tail23 : List Nat := 14087 :: root640Tail24
def root640Tail22 : List Nat := 13901 :: root640Tail23
def root640Tail21 : List Nat := 13553 :: root640Tail22
def root640Tail20 : List Nat := 12893 :: root640Tail21
def root640Tail19 : List Nat := 12281 :: root640Tail20
def root640Tail18 : List Nat := 11411 :: root640Tail19
def root640Tail17 : List Nat := 10949 :: root640Tail18
def root640Tail16 : List Nat := 10889 :: root640Tail17
def root640Tail15 : List Nat := 10781 :: root640Tail16
def root640Tail14 : List Nat := 10457 :: root640Tail15
def root640Tail13 : List Nat := 9929 :: root640Tail14
def root640Tail12 : List Nat := 9371 :: root640Tail13
def root640Tail11 : List Nat := 9257 :: root640Tail12
def root640Tail10 : List Nat := 8573 :: root640Tail11
def root640Tail9 : List Nat := 8009 :: root640Tail10
def root640Tail8 : List Nat := 7649 :: root640Tail9
def root640Tail7 : List Nat := 7433 :: root640Tail8
def root640Tail6 : List Nat := 7229 :: root640Tail7
def root640Tail5 : List Nat := 7187 :: root640Tail6
def root640Tail4 : List Nat := 7013 :: root640Tail5
def root640Tail3 : List Nat := 6047 :: root640Tail4
def root640Tail2 : List Nat := 5981 :: root640Tail3
def root640Tail1 : List Nat := 5237 :: root640Tail2
def root640Tail0 : List Nat := 5099 :: root640Tail1
def root640Candidates0 : List Nat := [7649]
set_option maxRecDepth 8192 in
theorem root640Candidates0_checked : candidates edge 4 4787 5099 root640Tail1 winner.bound = root640Candidates0 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root640Step0 : scan edge (search edge 3) 4 [4787] 4787 root640Tail0 winner =
    scan edge (search edge 3) 4 [4787] 4787 root640Tail1 winner := by
  rw [root640Tail0, scan, if_neg (by decide), root640Candidates0_checked]
  change scan edge (search edge 3) 4 [4787] 4787 root640Tail1
    (if 3 ≤ root640Candidates0.length then search edge 3 [4787, 5099] root640Candidates0 9886 winner else winner) = _
  rw [if_neg (by decide)]
def root640Candidates1 : List Nat := [8009, 12281]
set_option maxRecDepth 8192 in
theorem root640Candidates1_checked : candidates edge 4 4787 5237 root640Tail2 winner.bound = root640Candidates1 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root640Step1 : scan edge (search edge 3) 4 [4787] 4787 root640Tail1 winner =
    scan edge (search edge 3) 4 [4787] 4787 root640Tail2 winner := by
  rw [root640Tail1, scan, if_neg (by decide), root640Candidates1_checked]
  change scan edge (search edge 3) 4 [4787] 4787 root640Tail2
    (if 3 ≤ root640Candidates1.length then search edge 3 [4787, 5237] root640Candidates1 10024 winner else winner) = _
  rw [if_neg (by decide)]
theorem root640Scan2 : scan edge (search edge 3) 4 [4787] 4787 root640Tail2 winner = winner := by rw [root640Tail2, scan, if_pos (by decide)]
theorem root640Scan1 : scan edge (search edge 3) 4 [4787] 4787 root640Tail1 winner = winner :=
  (root640Step1).trans root640Scan2
theorem root640Scan0 : scan edge (search edge 3) 4 [4787] 4787 root640Tail0 winner = winner :=
  (root640Step0).trans root640Scan1
set_option maxRecDepth 4096 in
theorem root640Branch : search edge 4 [4787] root640Candidates 4787 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4787] 4787 root640Tail0 winner = winner
  exact root640Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
