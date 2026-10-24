import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root580Candidates : List Nat := Data.row4253
set_option maxRecDepth 8192 in
theorem root580Candidates_checked : candidates edge 5 0 4253 tail581 winner.bound = root580Candidates := by
  apply candidates_eq_row edge 5 0 4253 tail581 root580Candidates winner.bound tail581_ordered
  · decide
  · decide
  · funext q
    rfl
def root580Tail25 : List Nat := []
def root580Tail24 : List Nat := 21599 :: root580Tail25
def root580Tail23 : List Nat := 20543 :: root580Tail24
def root580Tail22 : List Nat := 19709 :: root580Tail23
def root580Tail21 : List Nat := 17099 :: root580Tail22
def root580Tail20 : List Nat := 16937 :: root580Tail21
def root580Tail19 : List Nat := 15461 :: root580Tail20
def root580Tail18 : List Nat := 13997 :: root580Tail19
def root580Tail17 : List Nat := 12101 :: root580Tail18
def root580Tail16 : List Nat := 11867 :: root580Tail17
def root580Tail15 : List Nat := 11813 :: root580Tail16
def root580Tail14 : List Nat := 11681 :: root580Tail15
def root580Tail13 : List Nat := 11519 :: root580Tail14
def root580Tail12 : List Nat := 11489 :: root580Tail13
def root580Tail11 : List Nat := 9941 :: root580Tail12
def root580Tail10 : List Nat := 9521 :: root580Tail11
def root580Tail9 : List Nat := 9293 :: root580Tail10
def root580Tail8 : List Nat := 8849 :: root580Tail9
def root580Tail7 : List Nat := 8831 :: root580Tail8
def root580Tail6 : List Nat := 7829 :: root580Tail7
def root580Tail5 : List Nat := 6917 :: root580Tail6
def root580Tail4 : List Nat := 5939 :: root580Tail5
def root580Tail3 : List Nat := 5657 :: root580Tail4
def root580Tail2 : List Nat := 4937 :: root580Tail3
def root580Tail1 : List Nat := 4463 :: root580Tail2
def root580Tail0 : List Nat := 4397 :: root580Tail1
def root580Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root580Candidates0_checked : candidates edge 4 4253 4397 root580Tail1 winner.bound = root580Candidates0 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root580Step0 : scan edge (search edge 3) 4 [4253] 4253 root580Tail0 winner =
    scan edge (search edge 3) 4 [4253] 4253 root580Tail1 winner := by
  rw [root580Tail0, scan, if_neg (by decide), root580Candidates0_checked]
  change scan edge (search edge 3) 4 [4253] 4253 root580Tail1
    (if 3 ≤ root580Candidates0.length then search edge 3 [4253, 4397] root580Candidates0 8650 winner else winner) = _
  rw [if_neg (by decide)]
def root580Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root580Candidates1_checked : candidates edge 4 4253 4463 root580Tail2 winner.bound = root580Candidates1 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root580Step1 : scan edge (search edge 3) 4 [4253] 4253 root580Tail1 winner =
    scan edge (search edge 3) 4 [4253] 4253 root580Tail2 winner := by
  rw [root580Tail1, scan, if_neg (by decide), root580Candidates1_checked]
  change scan edge (search edge 3) 4 [4253] 4253 root580Tail2
    (if 3 ≤ root580Candidates1.length then search edge 3 [4253, 4463] root580Candidates1 8716 winner else winner) = _
  rw [if_neg (by decide)]
def root580Candidates2 : List Nat := [9521]
set_option maxRecDepth 8192 in
theorem root580Candidates2_checked : candidates edge 4 4253 4937 root580Tail3 winner.bound = root580Candidates2 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root580Step2 : scan edge (search edge 3) 4 [4253] 4253 root580Tail2 winner =
    scan edge (search edge 3) 4 [4253] 4253 root580Tail3 winner := by
  rw [root580Tail2, scan, if_neg (by decide), root580Candidates2_checked]
  change scan edge (search edge 3) 4 [4253] 4253 root580Tail3
    (if 3 ≤ root580Candidates2.length then search edge 3 [4253, 4937] root580Candidates2 9190 winner else winner) = _
  rw [if_neg (by decide)]
theorem root580Scan3 : scan edge (search edge 3) 4 [4253] 4253 root580Tail3 winner = winner := by rw [root580Tail3, scan, if_pos (by decide)]
theorem root580Scan2 : scan edge (search edge 3) 4 [4253] 4253 root580Tail2 winner = winner :=
  (root580Step2).trans root580Scan3
theorem root580Scan1 : scan edge (search edge 3) 4 [4253] 4253 root580Tail1 winner = winner :=
  (root580Step1).trans root580Scan2
theorem root580Scan0 : scan edge (search edge 3) 4 [4253] 4253 root580Tail0 winner = winner :=
  (root580Step0).trans root580Scan1
set_option maxRecDepth 4096 in
theorem root580Branch : search edge 4 [4253] root580Candidates 4253 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4253] 4253 root580Tail0 winner = winner
  exact root580Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
