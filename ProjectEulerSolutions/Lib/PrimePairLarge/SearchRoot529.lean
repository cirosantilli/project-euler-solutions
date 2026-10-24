import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root529Candidates : List Nat := Data.row3833
set_option maxRecDepth 8192 in
theorem root529Candidates_checked : candidates edge 5 0 3833 tail530 winner.bound = root529Candidates := by
  apply candidates_eq_row edge 5 0 3833 tail530 root529Candidates winner.bound tail530_ordered
  · decide
  · decide
  · funext q
    rfl
def root529Tail34 : List Nat := []
def root529Tail33 : List Nat := 22073 :: root529Tail34
def root529Tail32 : List Nat := 22067 :: root529Tail33
def root529Tail31 : List Nat := 21383 :: root529Tail32
def root529Tail30 : List Nat := 21149 :: root529Tail31
def root529Tail29 : List Nat := 19793 :: root529Tail30
def root529Tail28 : List Nat := 19457 :: root529Tail29
def root529Tail27 : List Nat := 18743 :: root529Tail28
def root529Tail26 : List Nat := 18539 :: root529Tail27
def root529Tail25 : List Nat := 18371 :: root529Tail26
def root529Tail24 : List Nat := 15767 :: root529Tail25
def root529Tail23 : List Nat := 15629 :: root529Tail24
def root529Tail22 : List Nat := 14891 :: root529Tail23
def root529Tail21 : List Nat := 14753 :: root529Tail22
def root529Tail20 : List Nat := 14621 :: root529Tail21
def root529Tail19 : List Nat := 13913 :: root529Tail20
def root529Tail18 : List Nat := 13829 :: root529Tail19
def root529Tail17 : List Nat := 13001 :: root529Tail18
def root529Tail16 : List Nat := 11657 :: root529Tail17
def root529Tail15 : List Nat := 11447 :: root529Tail16
def root529Tail14 : List Nat := 11177 :: root529Tail15
def root529Tail13 : List Nat := 10799 :: root529Tail14
def root529Tail12 : List Nat := 9857 :: root529Tail13
def root529Tail11 : List Nat := 7841 :: root529Tail12
def root529Tail10 : List Nat := 6983 :: root529Tail11
def root529Tail9 : List Nat := 6911 :: root529Tail10
def root529Tail8 : List Nat := 6833 :: root529Tail9
def root529Tail7 : List Nat := 6827 :: root529Tail8
def root529Tail6 : List Nat := 6449 :: root529Tail7
def root529Tail5 : List Nat := 6269 :: root529Tail6
def root529Tail4 : List Nat := 5813 :: root529Tail5
def root529Tail3 : List Nat := 5657 :: root529Tail4
def root529Tail2 : List Nat := 4889 :: root529Tail3
def root529Tail1 : List Nat := 4733 :: root529Tail2
def root529Tail0 : List Nat := 4253 :: root529Tail1
def root529Candidates0 : List Nat := [5657]
set_option maxRecDepth 8192 in
theorem root529Candidates0_checked : candidates edge 4 3833 4253 root529Tail1 winner.bound = root529Candidates0 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root529Step0 : scan edge (search edge 3) 4 [3833] 3833 root529Tail0 winner =
    scan edge (search edge 3) 4 [3833] 3833 root529Tail1 winner := by
  rw [root529Tail0, scan, if_neg (by decide), root529Candidates0_checked]
  change scan edge (search edge 3) 4 [3833] 3833 root529Tail1
    (if 3 ≤ root529Candidates0.length then search edge 3 [3833, 4253] root529Candidates0 8086 winner else winner) = _
  rw [if_neg (by decide)]
def root529Candidates1 : List Nat := [6269, 6911]
set_option maxRecDepth 8192 in
theorem root529Candidates1_checked : candidates edge 4 3833 4733 root529Tail2 winner.bound = root529Candidates1 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root529Step1 : scan edge (search edge 3) 4 [3833] 3833 root529Tail1 winner =
    scan edge (search edge 3) 4 [3833] 3833 root529Tail2 winner := by
  rw [root529Tail1, scan, if_neg (by decide), root529Candidates1_checked]
  change scan edge (search edge 3) 4 [3833] 3833 root529Tail2
    (if 3 ≤ root529Candidates1.length then search edge 3 [3833, 4733] root529Candidates1 8566 winner else winner) = _
  rw [if_neg (by decide)]
def root529Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root529Candidates2_checked : candidates edge 4 3833 4889 root529Tail3 winner.bound = root529Candidates2 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root529Step2 : scan edge (search edge 3) 4 [3833] 3833 root529Tail2 winner =
    scan edge (search edge 3) 4 [3833] 3833 root529Tail3 winner := by
  rw [root529Tail2, scan, if_neg (by decide), root529Candidates2_checked]
  change scan edge (search edge 3) 4 [3833] 3833 root529Tail3
    (if 3 ≤ root529Candidates2.length then search edge 3 [3833, 4889] root529Candidates2 8722 winner else winner) = _
  rw [if_neg (by decide)]
theorem root529Scan3 : scan edge (search edge 3) 4 [3833] 3833 root529Tail3 winner = winner := by rw [root529Tail3, scan, if_pos (by decide)]
theorem root529Scan2 : scan edge (search edge 3) 4 [3833] 3833 root529Tail2 winner = winner :=
  (root529Step2).trans root529Scan3
theorem root529Scan1 : scan edge (search edge 3) 4 [3833] 3833 root529Tail1 winner = winner :=
  (root529Step1).trans root529Scan2
theorem root529Scan0 : scan edge (search edge 3) 4 [3833] 3833 root529Tail0 winner = winner :=
  (root529Step0).trans root529Scan1
set_option maxRecDepth 4096 in
theorem root529Branch : search edge 4 [3833] root529Candidates 3833 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3833] 3833 root529Tail0 winner = winner
  exact root529Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
