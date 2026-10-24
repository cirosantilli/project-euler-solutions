import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root455Candidates : List Nat := Data.row3251
set_option maxRecDepth 8192 in
theorem root455Candidates_checked : candidates edge 5 0 3251 tail456 winner.bound = root455Candidates := by
  apply candidates_eq_row edge 5 0 3251 tail456 root455Candidates winner.bound tail456_ordered
  · decide
  · decide
  · funext q
    rfl
def root455Tail27 : List Nat := []
def root455Tail26 : List Nat := 22409 :: root455Tail27
def root455Tail25 : List Nat := 20183 :: root455Tail26
def root455Tail24 : List Nat := 19919 :: root455Tail25
def root455Tail23 : List Nat := 19427 :: root455Tail24
def root455Tail22 : List Nat := 19031 :: root455Tail23
def root455Tail21 : List Nat := 17321 :: root455Tail22
def root455Tail20 : List Nat := 17183 :: root455Tail21
def root455Tail19 : List Nat := 16067 :: root455Tail20
def root455Tail18 : List Nat := 15671 :: root455Tail19
def root455Tail17 : List Nat := 15629 :: root455Tail18
def root455Tail16 : List Nat := 14831 :: root455Tail17
def root455Tail15 : List Nat := 14033 :: root455Tail16
def root455Tail14 : List Nat := 13907 :: root455Tail15
def root455Tail13 : List Nat := 13163 :: root455Tail14
def root455Tail12 : List Nat := 11867 :: root455Tail13
def root455Tail11 : List Nat := 9497 :: root455Tail12
def root455Tail10 : List Nat := 9413 :: root455Tail11
def root455Tail9 : List Nat := 8963 :: root455Tail10
def root455Tail8 : List Nat := 8273 :: root455Tail9
def root455Tail7 : List Nat := 7229 :: root455Tail8
def root455Tail6 : List Nat := 5039 :: root455Tail7
def root455Tail5 : List Nat := 4973 :: root455Tail6
def root455Tail4 : List Nat := 4721 :: root455Tail5
def root455Tail3 : List Nat := 4409 :: root455Tail4
def root455Tail2 : List Nat := 4253 :: root455Tail3
def root455Tail1 : List Nat := 4241 :: root455Tail2
def root455Tail0 : List Nat := 3413 :: root455Tail1
def root455Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root455Candidates0_checked : candidates edge 4 3251 3413 root455Tail1 winner.bound = root455Candidates0 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root455Step0 : scan edge (search edge 3) 4 [3251] 3251 root455Tail0 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail1 winner := by
  rw [root455Tail0, scan, if_neg (by decide), root455Candidates0_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail1
    (if 3 ≤ root455Candidates0.length then search edge 3 [3251, 3413] root455Candidates0 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root455Candidates1 : List Nat := [9413]
set_option maxRecDepth 8192 in
theorem root455Candidates1_checked : candidates edge 4 3251 4241 root455Tail2 winner.bound = root455Candidates1 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root455Step1 : scan edge (search edge 3) 4 [3251] 3251 root455Tail1 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail2 winner := by
  rw [root455Tail1, scan, if_neg (by decide), root455Candidates1_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail2
    (if 3 ≤ root455Candidates1.length then search edge 3 [3251, 4241] root455Candidates1 7492 winner else winner) = _
  rw [if_neg (by decide)]
def root455Candidates2 : List Nat := [11867]
set_option maxRecDepth 8192 in
theorem root455Candidates2_checked : candidates edge 4 3251 4253 root455Tail3 winner.bound = root455Candidates2 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root455Step2 : scan edge (search edge 3) 4 [3251] 3251 root455Tail2 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail3 winner := by
  rw [root455Tail2, scan, if_neg (by decide), root455Candidates2_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail3
    (if 3 ≤ root455Candidates2.length then search edge 3 [3251, 4253] root455Candidates2 7504 winner else winner) = _
  rw [if_neg (by decide)]
def root455Candidates3 : List Nat := [9413]
set_option maxRecDepth 8192 in
theorem root455Candidates3_checked : candidates edge 4 3251 4409 root455Tail4 winner.bound = root455Candidates3 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root455Step3 : scan edge (search edge 3) 4 [3251] 3251 root455Tail3 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail4 winner := by
  rw [root455Tail3, scan, if_neg (by decide), root455Candidates3_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail4
    (if 3 ≤ root455Candidates3.length then search edge 3 [3251, 4409] root455Candidates3 7660 winner else winner) = _
  rw [if_neg (by decide)]
def root455Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root455Candidates4_checked : candidates edge 4 3251 4721 root455Tail5 winner.bound = root455Candidates4 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root455Step4 : scan edge (search edge 3) 4 [3251] 3251 root455Tail4 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail5 winner := by
  rw [root455Tail4, scan, if_neg (by decide), root455Candidates4_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail5
    (if 3 ≤ root455Candidates4.length then search edge 3 [3251, 4721] root455Candidates4 7972 winner else winner) = _
  rw [if_neg (by decide)]
def root455Candidates5 : List Nat := [9413, 15629]
set_option maxRecDepth 8192 in
theorem root455Candidates5_checked : candidates edge 4 3251 4973 root455Tail6 winner.bound = root455Candidates5 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root455Step5 : scan edge (search edge 3) 4 [3251] 3251 root455Tail5 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail6 winner := by
  rw [root455Tail5, scan, if_neg (by decide), root455Candidates5_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail6
    (if 3 ≤ root455Candidates5.length then search edge 3 [3251, 4973] root455Candidates5 8224 winner else winner) = _
  rw [if_neg (by decide)]
def root455Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root455Candidates6_checked : candidates edge 4 3251 5039 root455Tail7 winner.bound = root455Candidates6 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root455Step6 : scan edge (search edge 3) 4 [3251] 3251 root455Tail6 winner =
    scan edge (search edge 3) 4 [3251] 3251 root455Tail7 winner := by
  rw [root455Tail6, scan, if_neg (by decide), root455Candidates6_checked]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail7
    (if 3 ≤ root455Candidates6.length then search edge 3 [3251, 5039] root455Candidates6 8290 winner else winner) = _
  rw [if_neg (by decide)]
theorem root455Scan7 : scan edge (search edge 3) 4 [3251] 3251 root455Tail7 winner = winner := by rw [root455Tail7, scan, if_pos (by decide)]
theorem root455Scan6 : scan edge (search edge 3) 4 [3251] 3251 root455Tail6 winner = winner :=
  (root455Step6).trans root455Scan7
theorem root455Scan5 : scan edge (search edge 3) 4 [3251] 3251 root455Tail5 winner = winner :=
  (root455Step5).trans root455Scan6
theorem root455Scan4 : scan edge (search edge 3) 4 [3251] 3251 root455Tail4 winner = winner :=
  (root455Step4).trans root455Scan5
theorem root455Scan3 : scan edge (search edge 3) 4 [3251] 3251 root455Tail3 winner = winner :=
  (root455Step3).trans root455Scan4
theorem root455Scan2 : scan edge (search edge 3) 4 [3251] 3251 root455Tail2 winner = winner :=
  (root455Step2).trans root455Scan3
theorem root455Scan1 : scan edge (search edge 3) 4 [3251] 3251 root455Tail1 winner = winner :=
  (root455Step1).trans root455Scan2
theorem root455Scan0 : scan edge (search edge 3) 4 [3251] 3251 root455Tail0 winner = winner :=
  (root455Step0).trans root455Scan1
set_option maxRecDepth 4096 in
theorem root455Branch : search edge 4 [3251] root455Candidates 3251 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3251] 3251 root455Tail0 winner = winner
  exact root455Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
