import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root654Candidates : List Nat := Data.row4919
set_option maxRecDepth 8192 in
theorem root654Candidates_checked : candidates edge 5 0 4919 tail655 winner.bound = root654Candidates := by
  apply candidates_eq_row edge 5 0 4919 tail655 root654Candidates winner.bound tail655_ordered
  · decide
  · decide
  · funext q
    rfl
def root654Tail36 : List Nat := []
def root654Tail35 : List Nat := 20201 :: root654Tail36
def root654Tail34 : List Nat := 19979 :: root654Tail35
def root654Tail33 : List Nat := 19889 :: root654Tail34
def root654Tail32 : List Nat := 19661 :: root654Tail33
def root654Tail31 : List Nat := 18131 :: root654Tail32
def root654Tail30 : List Nat := 17351 :: root654Tail31
def root654Tail29 : List Nat := 17159 :: root654Tail30
def root654Tail28 : List Nat := 15527 :: root654Tail29
def root654Tail27 : List Nat := 13907 :: root654Tail28
def root654Tail26 : List Nat := 13841 :: root654Tail27
def root654Tail25 : List Nat := 13691 :: root654Tail26
def root654Tail24 : List Nat := 13337 :: root654Tail25
def root654Tail23 : List Nat := 11597 :: root654Tail24
def root654Tail22 : List Nat := 11093 :: root654Tail23
def root654Tail21 : List Nat := 10781 :: root654Tail22
def root654Tail20 : List Nat := 10739 :: root654Tail21
def root654Tail19 : List Nat := 10709 :: root654Tail20
def root654Tail18 : List Nat := 10631 :: root654Tail19
def root654Tail17 : List Nat := 10163 :: root654Tail18
def root654Tail16 : List Nat := 10061 :: root654Tail17
def root654Tail15 : List Nat := 9203 :: root654Tail16
def root654Tail14 : List Nat := 8783 :: root654Tail15
def root654Tail13 : List Nat := 8741 :: root654Tail14
def root654Tail12 : List Nat := 8627 :: root654Tail13
def root654Tail11 : List Nat := 7877 :: root654Tail12
def root654Tail10 : List Nat := 7829 :: root654Tail11
def root654Tail9 : List Nat := 7607 :: root654Tail10
def root654Tail8 : List Nat := 7559 :: root654Tail9
def root654Tail7 : List Nat := 7481 :: root654Tail8
def root654Tail6 : List Nat := 6869 :: root654Tail7
def root654Tail5 : List Nat := 6803 :: root654Tail6
def root654Tail4 : List Nat := 6599 :: root654Tail5
def root654Tail3 : List Nat := 6197 :: root654Tail4
def root654Tail2 : List Nat := 5897 :: root654Tail3
def root654Tail1 : List Nat := 5189 :: root654Tail2
def root654Tail0 : List Nat := 5147 :: root654Tail1
def root654Candidates0 : List Nat := [9203]
set_option maxRecDepth 8192 in
theorem root654Candidates0_checked : candidates edge 4 4919 5147 root654Tail1 winner.bound = root654Candidates0 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root654Step0 : scan edge (search edge 3) 4 [4919] 4919 root654Tail0 winner =
    scan edge (search edge 3) 4 [4919] 4919 root654Tail1 winner := by
  rw [root654Tail0, scan, if_neg (by decide), root654Candidates0_checked]
  change scan edge (search edge 3) 4 [4919] 4919 root654Tail1
    (if 3 ≤ root654Candidates0.length then search edge 3 [4919, 5147] root654Candidates0 10066 winner else winner) = _
  rw [if_neg (by decide)]
def root654Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root654Candidates1_checked : candidates edge 4 4919 5189 root654Tail2 winner.bound = root654Candidates1 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root654Step1 : scan edge (search edge 3) 4 [4919] 4919 root654Tail1 winner =
    scan edge (search edge 3) 4 [4919] 4919 root654Tail2 winner := by
  rw [root654Tail1, scan, if_neg (by decide), root654Candidates1_checked]
  change scan edge (search edge 3) 4 [4919] 4919 root654Tail2
    (if 3 ≤ root654Candidates1.length then search edge 3 [4919, 5189] root654Candidates1 10108 winner else winner) = _
  rw [if_neg (by decide)]
theorem root654Scan2 : scan edge (search edge 3) 4 [4919] 4919 root654Tail2 winner = winner := by rw [root654Tail2, scan, if_pos (by decide)]
theorem root654Scan1 : scan edge (search edge 3) 4 [4919] 4919 root654Tail1 winner = winner :=
  (root654Step1).trans root654Scan2
theorem root654Scan0 : scan edge (search edge 3) 4 [4919] 4919 root654Tail0 winner = winner :=
  (root654Step0).trans root654Scan1
set_option maxRecDepth 4096 in
theorem root654Branch : search edge 4 [4919] root654Candidates 4919 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4919] 4919 root654Tail0 winner = winner
  exact root654Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
