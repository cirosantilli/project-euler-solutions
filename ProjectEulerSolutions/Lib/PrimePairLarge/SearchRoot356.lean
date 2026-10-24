import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root356Candidates : List Nat := Data.row2417
set_option maxRecDepth 8192 in
theorem root356Candidates_checked : candidates edge 5 0 2417 tail357 winner.bound = root356Candidates := by
  apply candidates_eq_row edge 5 0 2417 tail357 root356Candidates winner.bound tail357_ordered
  · decide
  · decide
  · funext q
    rfl
def root356Tail48 : List Nat := []
def root356Tail47 : List Nat := 23099 :: root356Tail48
def root356Tail46 : List Nat := 22973 :: root356Tail47
def root356Tail45 : List Nat := 22727 :: root356Tail46
def root356Tail44 : List Nat := 21617 :: root356Tail45
def root356Tail43 : List Nat := 20753 :: root356Tail44
def root356Tail42 : List Nat := 20693 :: root356Tail43
def root356Tail41 : List Nat := 19661 :: root356Tail42
def root356Tail40 : List Nat := 19427 :: root356Tail41
def root356Tail39 : List Nat := 18947 :: root356Tail40
def root356Tail38 : List Nat := 18341 :: root356Tail39
def root356Tail37 : List Nat := 18251 :: root356Tail38
def root356Tail36 : List Nat := 18143 :: root356Tail37
def root356Tail35 : List Nat := 17579 :: root356Tail36
def root356Tail34 : List Nat := 17123 :: root356Tail35
def root356Tail33 : List Nat := 15647 :: root356Tail34
def root356Tail32 : List Nat := 14939 :: root356Tail33
def root356Tail31 : List Nat := 14489 :: root356Tail32
def root356Tail30 : List Nat := 13553 :: root356Tail31
def root356Tail29 : List Nat := 13523 :: root356Tail30
def root356Tail28 : List Nat := 13229 :: root356Tail29
def root356Tail27 : List Nat := 12377 :: root356Tail28
def root356Tail26 : List Nat := 12101 :: root356Tail27
def root356Tail25 : List Nat := 11801 :: root356Tail26
def root356Tail24 : List Nat := 11159 :: root356Tail25
def root356Tail23 : List Nat := 10463 :: root356Tail24
def root356Tail22 : List Nat := 9461 :: root356Tail23
def root356Tail21 : List Nat := 9437 :: root356Tail22
def root356Tail20 : List Nat := 8951 :: root356Tail21
def root356Tail19 : List Nat := 8867 :: root356Tail20
def root356Tail18 : List Nat := 8747 :: root356Tail19
def root356Tail17 : List Nat := 8237 :: root356Tail18
def root356Tail16 : List Nat := 7877 :: root356Tail17
def root356Tail15 : List Nat := 7691 :: root356Tail16
def root356Tail14 : List Nat := 7103 :: root356Tail15
def root356Tail13 : List Nat := 7019 :: root356Tail14
def root356Tail12 : List Nat := 6857 :: root356Tail13
def root356Tail11 : List Nat := 6599 :: root356Tail12
def root356Tail10 : List Nat := 6563 :: root356Tail11
def root356Tail9 : List Nat := 6389 :: root356Tail10
def root356Tail8 : List Nat := 5927 :: root356Tail9
def root356Tail7 : List Nat := 5903 :: root356Tail8
def root356Tail6 : List Nat := 5303 :: root356Tail7
def root356Tail5 : List Nat := 5153 :: root356Tail6
def root356Tail4 : List Nat := 4253 :: root356Tail5
def root356Tail3 : List Nat := 4229 :: root356Tail4
def root356Tail2 : List Nat := 4001 :: root356Tail3
def root356Tail1 : List Nat := 3449 :: root356Tail2
def root356Tail0 : List Nat := 2741 :: root356Tail1
def root356Candidates0 : List Nat := [4253, 8747]
set_option maxRecDepth 8192 in
theorem root356Candidates0_checked : candidates edge 4 2417 2741 root356Tail1 winner.bound = root356Candidates0 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root356Step0 : scan edge (search edge 3) 4 [2417] 2417 root356Tail0 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail1 winner := by
  rw [root356Tail0, scan, if_neg (by decide), root356Candidates0_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail1
    (if 3 ≤ root356Candidates0.length then search edge 3 [2417, 2741] root356Candidates0 5158 winner else winner) = _
  rw [if_neg (by decide)]
def root356Candidates1 : List Nat := [13229]
set_option maxRecDepth 8192 in
theorem root356Candidates1_checked : candidates edge 4 2417 3449 root356Tail2 winner.bound = root356Candidates1 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root356Step1 : scan edge (search edge 3) 4 [2417] 2417 root356Tail1 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail2 winner := by
  rw [root356Tail1, scan, if_neg (by decide), root356Candidates1_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail2
    (if 3 ≤ root356Candidates1.length then search edge 3 [2417, 3449] root356Candidates1 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root356Candidates2 : List Nat := [4253, 6563, 7877]
set_option maxRecDepth 8192 in
theorem root356Candidates2_checked : candidates edge 4 2417 4001 root356Tail3 winner.bound = root356Candidates2 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root356Child2 : search edge 3 [2417, 4001] root356Candidates2 6418 winner = winner := by decide
theorem root356Step2 : scan edge (search edge 3) 4 [2417] 2417 root356Tail2 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail3 winner := by
  rw [root356Tail2, scan, if_neg (by decide), root356Candidates2_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail3
    (if 3 ≤ root356Candidates2.length then search edge 3 [2417, 4001] root356Candidates2 6418 winner else winner) = _
  rw [if_pos (by decide), root356Child2]
def root356Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root356Candidates3_checked : candidates edge 4 2417 4229 root356Tail4 winner.bound = root356Candidates3 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root356Step3 : scan edge (search edge 3) 4 [2417] 2417 root356Tail3 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail4 winner := by
  rw [root356Tail3, scan, if_neg (by decide), root356Candidates3_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail4
    (if 3 ≤ root356Candidates3.length then search edge 3 [2417, 4229] root356Candidates3 6646 winner else winner) = _
  rw [if_neg (by decide)]
def root356Candidates4 : List Nat := [12101]
set_option maxRecDepth 8192 in
theorem root356Candidates4_checked : candidates edge 4 2417 4253 root356Tail5 winner.bound = root356Candidates4 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root356Step4 : scan edge (search edge 3) 4 [2417] 2417 root356Tail4 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail5 winner := by
  rw [root356Tail4, scan, if_neg (by decide), root356Candidates4_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail5
    (if 3 ≤ root356Candidates4.length then search edge 3 [2417, 4253] root356Candidates4 6670 winner else winner) = _
  rw [if_neg (by decide)]
def root356Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root356Candidates5_checked : candidates edge 4 2417 5153 root356Tail6 winner.bound = root356Candidates5 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root356Step5 : scan edge (search edge 3) 4 [2417] 2417 root356Tail5 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail6 winner := by
  rw [root356Tail5, scan, if_neg (by decide), root356Candidates5_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail6
    (if 3 ≤ root356Candidates5.length then search edge 3 [2417, 5153] root356Candidates5 7570 winner else winner) = _
  rw [if_neg (by decide)]
def root356Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root356Candidates6_checked : candidates edge 4 2417 5303 root356Tail7 winner.bound = root356Candidates6 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root356Step6 : scan edge (search edge 3) 4 [2417] 2417 root356Tail6 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail7 winner := by
  rw [root356Tail6, scan, if_neg (by decide), root356Candidates6_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail7
    (if 3 ≤ root356Candidates6.length then search edge 3 [2417, 5303] root356Candidates6 7720 winner else winner) = _
  rw [if_neg (by decide)]
def root356Candidates7 : List Nat := [7877, 13553, 17579]
set_option maxRecDepth 8192 in
theorem root356Candidates7_checked : candidates edge 4 2417 5903 root356Tail8 winner.bound = root356Candidates7 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root356Child7 : search edge 3 [2417, 5903] root356Candidates7 8320 winner = winner := by decide
theorem root356Step7 : scan edge (search edge 3) 4 [2417] 2417 root356Tail7 winner =
    scan edge (search edge 3) 4 [2417] 2417 root356Tail8 winner := by
  rw [root356Tail7, scan, if_neg (by decide), root356Candidates7_checked]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail8
    (if 3 ≤ root356Candidates7.length then search edge 3 [2417, 5903] root356Candidates7 8320 winner else winner) = _
  rw [if_pos (by decide), root356Child7]
theorem root356Scan8 : scan edge (search edge 3) 4 [2417] 2417 root356Tail8 winner = winner := by rw [root356Tail8, scan, if_pos (by decide)]
theorem root356Scan7 : scan edge (search edge 3) 4 [2417] 2417 root356Tail7 winner = winner :=
  (root356Step7).trans root356Scan8
theorem root356Scan6 : scan edge (search edge 3) 4 [2417] 2417 root356Tail6 winner = winner :=
  (root356Step6).trans root356Scan7
theorem root356Scan5 : scan edge (search edge 3) 4 [2417] 2417 root356Tail5 winner = winner :=
  (root356Step5).trans root356Scan6
theorem root356Scan4 : scan edge (search edge 3) 4 [2417] 2417 root356Tail4 winner = winner :=
  (root356Step4).trans root356Scan5
theorem root356Scan3 : scan edge (search edge 3) 4 [2417] 2417 root356Tail3 winner = winner :=
  (root356Step3).trans root356Scan4
theorem root356Scan2 : scan edge (search edge 3) 4 [2417] 2417 root356Tail2 winner = winner :=
  (root356Step2).trans root356Scan3
theorem root356Scan1 : scan edge (search edge 3) 4 [2417] 2417 root356Tail1 winner = winner :=
  (root356Step1).trans root356Scan2
theorem root356Scan0 : scan edge (search edge 3) 4 [2417] 2417 root356Tail0 winner = winner :=
  (root356Step0).trans root356Scan1
set_option maxRecDepth 4096 in
theorem root356Branch : search edge 4 [2417] root356Candidates 2417 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2417] 2417 root356Tail0 winner = winner
  exact root356Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
