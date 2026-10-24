import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root555Candidates : List Nat := Data.row4049
set_option maxRecDepth 8192 in
theorem root555Candidates_checked : candidates edge 5 0 4049 tail556 winner.bound = root555Candidates := by
  apply candidates_eq_row edge 5 0 4049 tail556 root555Candidates winner.bound tail556_ordered
  · decide
  · decide
  · funext q
    rfl
def root555Tail36 : List Nat := []
def root555Tail35 : List Nat := 21467 :: root555Tail36
def root555Tail34 : List Nat := 20627 :: root555Tail35
def root555Tail33 : List Nat := 20249 :: root555Tail34
def root555Tail32 : List Nat := 19763 :: root555Tail33
def root555Tail31 : List Nat := 19751 :: root555Tail32
def root555Tail30 : List Nat := 19289 :: root555Tail31
def root555Tail29 : List Nat := 19259 :: root555Tail30
def root555Tail28 : List Nat := 17393 :: root555Tail29
def root555Tail27 : List Nat := 17351 :: root555Tail28
def root555Tail26 : List Nat := 17099 :: root555Tail27
def root555Tail25 : List Nat := 16649 :: root555Tail26
def root555Tail24 : List Nat := 14621 :: root555Tail25
def root555Tail23 : List Nat := 14303 :: root555Tail24
def root555Tail22 : List Nat := 14249 :: root555Tail23
def root555Tail21 : List Nat := 13967 :: root555Tail22
def root555Tail20 : List Nat := 13877 :: root555Tail21
def root555Tail19 : List Nat := 13151 :: root555Tail20
def root555Tail18 : List Nat := 12899 :: root555Tail19
def root555Tail17 : List Nat := 12791 :: root555Tail18
def root555Tail16 : List Nat := 12689 :: root555Tail17
def root555Tail15 : List Nat := 12653 :: root555Tail16
def root555Tail14 : List Nat := 12329 :: root555Tail15
def root555Tail13 : List Nat := 12323 :: root555Tail14
def root555Tail12 : List Nat := 11939 :: root555Tail13
def root555Tail11 : List Nat := 11519 :: root555Tail12
def root555Tail10 : List Nat := 11279 :: root555Tail11
def root555Tail9 : List Nat := 10139 :: root555Tail10
def root555Tail8 : List Nat := 9623 :: root555Tail9
def root555Tail7 : List Nat := 8111 :: root555Tail8
def root555Tail6 : List Nat := 7703 :: root555Tail7
def root555Tail5 : List Nat := 7649 :: root555Tail6
def root555Tail4 : List Nat := 5927 :: root555Tail5
def root555Tail3 : List Nat := 5261 :: root555Tail4
def root555Tail2 : List Nat := 4919 :: root555Tail3
def root555Tail1 : List Nat := 4877 :: root555Tail2
def root555Tail0 : List Nat := 4133 :: root555Tail1
def root555Candidates0 : List Nat := [8111, 12329]
set_option maxRecDepth 8192 in
theorem root555Candidates0_checked : candidates edge 4 4049 4133 root555Tail1 winner.bound = root555Candidates0 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root555Step0 : scan edge (search edge 3) 4 [4049] 4049 root555Tail0 winner =
    scan edge (search edge 3) 4 [4049] 4049 root555Tail1 winner := by
  rw [root555Tail0, scan, if_neg (by decide), root555Candidates0_checked]
  change scan edge (search edge 3) 4 [4049] 4049 root555Tail1
    (if 3 ≤ root555Candidates0.length then search edge 3 [4049, 4133] root555Candidates0 8182 winner else winner) = _
  rw [if_neg (by decide)]
def root555Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root555Candidates1_checked : candidates edge 4 4049 4877 root555Tail2 winner.bound = root555Candidates1 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root555Step1 : scan edge (search edge 3) 4 [4049] 4049 root555Tail1 winner =
    scan edge (search edge 3) 4 [4049] 4049 root555Tail2 winner := by
  rw [root555Tail1, scan, if_neg (by decide), root555Candidates1_checked]
  change scan edge (search edge 3) 4 [4049] 4049 root555Tail2
    (if 3 ≤ root555Candidates1.length then search edge 3 [4049, 4877] root555Candidates1 8926 winner else winner) = _
  rw [if_neg (by decide)]
def root555Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root555Candidates2_checked : candidates edge 4 4049 4919 root555Tail3 winner.bound = root555Candidates2 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root555Step2 : scan edge (search edge 3) 4 [4049] 4049 root555Tail2 winner =
    scan edge (search edge 3) 4 [4049] 4049 root555Tail3 winner := by
  rw [root555Tail2, scan, if_neg (by decide), root555Candidates2_checked]
  change scan edge (search edge 3) 4 [4049] 4049 root555Tail3
    (if 3 ≤ root555Candidates2.length then search edge 3 [4049, 4919] root555Candidates2 8968 winner else winner) = _
  rw [if_neg (by decide)]
def root555Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root555Candidates3_checked : candidates edge 4 4049 5261 root555Tail4 winner.bound = root555Candidates3 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root555Step3 : scan edge (search edge 3) 4 [4049] 4049 root555Tail3 winner =
    scan edge (search edge 3) 4 [4049] 4049 root555Tail4 winner := by
  rw [root555Tail3, scan, if_neg (by decide), root555Candidates3_checked]
  change scan edge (search edge 3) 4 [4049] 4049 root555Tail4
    (if 3 ≤ root555Candidates3.length then search edge 3 [4049, 5261] root555Candidates3 9310 winner else winner) = _
  rw [if_neg (by decide)]
theorem root555Scan4 : scan edge (search edge 3) 4 [4049] 4049 root555Tail4 winner = winner := by rw [root555Tail4, scan, if_pos (by decide)]
theorem root555Scan3 : scan edge (search edge 3) 4 [4049] 4049 root555Tail3 winner = winner :=
  (root555Step3).trans root555Scan4
theorem root555Scan2 : scan edge (search edge 3) 4 [4049] 4049 root555Tail2 winner = winner :=
  (root555Step2).trans root555Scan3
theorem root555Scan1 : scan edge (search edge 3) 4 [4049] 4049 root555Tail1 winner = winner :=
  (root555Step1).trans root555Scan2
theorem root555Scan0 : scan edge (search edge 3) 4 [4049] 4049 root555Tail0 winner = winner :=
  (root555Step0).trans root555Scan1
set_option maxRecDepth 4096 in
theorem root555Branch : search edge 4 [4049] root555Candidates 4049 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4049] 4049 root555Tail0 winner = winner
  exact root555Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
