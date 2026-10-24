import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root577Candidates : List Nat := Data.row4231
set_option maxRecDepth 8192 in
theorem root577Candidates_checked : candidates edge 5 0 4231 tail578 winner.bound = root577Candidates := by
  apply candidates_eq_row edge 5 0 4231 tail578 root577Candidates winner.bound tail578_ordered
  · decide
  · decide
  · funext q
    rfl
def root577Tail35 : List Nat := []
def root577Tail34 : List Nat := 21499 :: root577Tail35
def root577Tail33 : List Nat := 19687 :: root577Tail34
def root577Tail32 : List Nat := 18637 :: root577Tail33
def root577Tail31 : List Nat := 18379 :: root577Tail32
def root577Tail30 : List Nat := 18169 :: root577Tail31
def root577Tail29 : List Nat := 17203 :: root577Tail30
def root577Tail28 : List Nat := 16921 :: root577Tail29
def root577Tail27 : List Nat := 15991 :: root577Tail28
def root577Tail26 : List Nat := 14947 :: root577Tail27
def root577Tail25 : List Nat := 14563 :: root577Tail26
def root577Tail24 : List Nat := 14551 :: root577Tail25
def root577Tail23 : List Nat := 14143 :: root577Tail24
def root577Tail22 : List Nat := 14011 :: root577Tail23
def root577Tail21 : List Nat := 13729 :: root577Tail22
def root577Tail20 : List Nat := 13591 :: root577Tail21
def root577Tail19 : List Nat := 12967 :: root577Tail20
def root577Tail18 : List Nat := 12763 :: root577Tail19
def root577Tail17 : List Nat := 12253 :: root577Tail18
def root577Tail16 : List Nat := 11527 :: root577Tail17
def root577Tail15 : List Nat := 10789 :: root577Tail16
def root577Tail14 : List Nat := 10477 :: root577Tail15
def root577Tail13 : List Nat := 9769 :: root577Tail14
def root577Tail12 : List Nat := 9721 :: root577Tail13
def root577Tail11 : List Nat := 9547 :: root577Tail12
def root577Tail10 : List Nat := 8707 :: root577Tail11
def root577Tail9 : List Nat := 8623 :: root577Tail10
def root577Tail8 : List Nat := 7789 :: root577Tail9
def root577Tail7 : List Nat := 7027 :: root577Tail8
def root577Tail6 : List Nat := 6577 :: root577Tail7
def root577Tail5 : List Nat := 6529 :: root577Tail6
def root577Tail4 : List Nat := 6151 :: root577Tail5
def root577Tail3 : List Nat := 6037 :: root577Tail4
def root577Tail2 : List Nat := 5197 :: root577Tail3
def root577Tail1 : List Nat := 4933 :: root577Tail2
def root577Tail0 : List Nat := 4339 :: root577Tail1
def root577Candidates0 : List Nat := [14563]
set_option maxRecDepth 8192 in
theorem root577Candidates0_checked : candidates edge 4 4231 4339 root577Tail1 winner.bound = root577Candidates0 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root577Step0 : scan edge (search edge 3) 4 [4231] 4231 root577Tail0 winner =
    scan edge (search edge 3) 4 [4231] 4231 root577Tail1 winner := by
  rw [root577Tail0, scan, if_neg (by decide), root577Candidates0_checked]
  change scan edge (search edge 3) 4 [4231] 4231 root577Tail1
    (if 3 ≤ root577Candidates0.length then search edge 3 [4231, 4339] root577Candidates0 8570 winner else winner) = _
  rw [if_neg (by decide)]
def root577Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root577Candidates1_checked : candidates edge 4 4231 4933 root577Tail2 winner.bound = root577Candidates1 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root577Step1 : scan edge (search edge 3) 4 [4231] 4231 root577Tail1 winner =
    scan edge (search edge 3) 4 [4231] 4231 root577Tail2 winner := by
  rw [root577Tail1, scan, if_neg (by decide), root577Candidates1_checked]
  change scan edge (search edge 3) 4 [4231] 4231 root577Tail2
    (if 3 ≤ root577Candidates1.length then search edge 3 [4231, 4933] root577Candidates1 9164 winner else winner) = _
  rw [if_neg (by decide)]
def root577Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root577Candidates2_checked : candidates edge 4 4231 5197 root577Tail3 winner.bound = root577Candidates2 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root577Step2 : scan edge (search edge 3) 4 [4231] 4231 root577Tail2 winner =
    scan edge (search edge 3) 4 [4231] 4231 root577Tail3 winner := by
  rw [root577Tail2, scan, if_neg (by decide), root577Candidates2_checked]
  change scan edge (search edge 3) 4 [4231] 4231 root577Tail3
    (if 3 ≤ root577Candidates2.length then search edge 3 [4231, 5197] root577Candidates2 9428 winner else winner) = _
  rw [if_neg (by decide)]
theorem root577Scan3 : scan edge (search edge 3) 4 [4231] 4231 root577Tail3 winner = winner := by rw [root577Tail3, scan, if_pos (by decide)]
theorem root577Scan2 : scan edge (search edge 3) 4 [4231] 4231 root577Tail2 winner = winner :=
  (root577Step2).trans root577Scan3
theorem root577Scan1 : scan edge (search edge 3) 4 [4231] 4231 root577Tail1 winner = winner :=
  (root577Step1).trans root577Scan2
theorem root577Scan0 : scan edge (search edge 3) 4 [4231] 4231 root577Tail0 winner = winner :=
  (root577Step0).trans root577Scan1
set_option maxRecDepth 4096 in
theorem root577Branch : search edge 4 [4231] root577Candidates 4231 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4231] 4231 root577Tail0 winner = winner
  exact root577Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
