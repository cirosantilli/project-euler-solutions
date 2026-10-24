import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root198Candidates : List Nat := Data.row1229
set_option maxRecDepth 8192 in
theorem root198Candidates_checked : candidates edge 5 0 1229 tail199 winner.bound = root198Candidates := by
  apply candidates_eq_row edge 5 0 1229 tail199 root198Candidates winner.bound tail199_ordered
  · decide
  · decide
  · funext q
    rfl
def root198Tail45 : List Nat := []
def root198Tail44 : List Nat := 24413 :: root198Tail45
def root198Tail43 : List Nat := 22859 :: root198Tail44
def root198Tail42 : List Nat := 22433 :: root198Tail43
def root198Tail41 : List Nat := 22067 :: root198Tail42
def root198Tail40 : List Nat := 21767 :: root198Tail41
def root198Tail39 : List Nat := 21521 :: root198Tail40
def root198Tail38 : List Nat := 21059 :: root198Tail39
def root198Tail37 : List Nat := 20249 :: root198Tail38
def root198Tail36 : List Nat := 19919 :: root198Tail37
def root198Tail35 : List Nat := 19289 :: root198Tail36
def root198Tail34 : List Nat := 19259 :: root198Tail35
def root198Tail33 : List Nat := 18947 :: root198Tail34
def root198Tail32 : List Nat := 17579 :: root198Tail33
def root198Tail31 : List Nat := 17351 :: root198Tail32
def root198Tail30 : List Nat := 16649 :: root198Tail31
def root198Tail29 : List Nat := 16427 :: root198Tail30
def root198Tail28 : List Nat := 16067 :: root198Tail29
def root198Tail27 : List Nat := 15263 :: root198Tail28
def root198Tail26 : List Nat := 14543 :: root198Tail27
def root198Tail25 : List Nat := 14087 :: root198Tail26
def root198Tail24 : List Nat := 12611 :: root198Tail25
def root198Tail23 : List Nat := 12497 :: root198Tail24
def root198Tail22 : List Nat := 12239 :: root198Tail23
def root198Tail21 : List Nat := 11783 :: root198Tail22
def root198Tail20 : List Nat := 11681 :: root198Tail21
def root198Tail19 : List Nat := 10979 :: root198Tail20
def root198Tail18 : List Nat := 10589 :: root198Tail19
def root198Tail17 : List Nat := 9473 :: root198Tail18
def root198Tail16 : List Nat := 9461 :: root198Tail17
def root198Tail15 : List Nat := 7907 :: root198Tail16
def root198Tail14 : List Nat := 7877 :: root198Tail15
def root198Tail13 : List Nat := 6353 :: root198Tail14
def root198Tail12 : List Nat := 5861 :: root198Tail13
def root198Tail11 : List Nat := 5741 :: root198Tail12
def root198Tail10 : List Nat := 5693 :: root198Tail11
def root198Tail9 : List Nat := 5399 :: root198Tail10
def root198Tail8 : List Nat := 4229 :: root198Tail9
def root198Tail7 : List Nat := 4127 :: root198Tail8
def root198Tail6 : List Nat := 3881 :: root198Tail7
def root198Tail5 : List Nat := 3389 :: root198Tail6
def root198Tail4 : List Nat := 3089 :: root198Tail5
def root198Tail3 : List Nat := 2963 :: root198Tail4
def root198Tail2 : List Nat := 2663 :: root198Tail3
def root198Tail1 : List Nat := 2081 :: root198Tail2
def root198Tail0 : List Nat := 1607 :: root198Tail1
def root198Candidates0 : List Nat := [7877, 19919]
set_option maxRecDepth 8192 in
theorem root198Candidates0_checked : candidates edge 4 1229 1607 root198Tail1 winner.bound = root198Candidates0 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
theorem root198Step0 : scan edge (search edge 3) 4 [1229] 1229 root198Tail0 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail1 winner := by
  rw [root198Tail0, scan, if_neg (by decide), root198Candidates0_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail1
    (if 3 ≤ root198Candidates0.length then search edge 3 [1229, 1607] root198Candidates0 2836 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates1 : List Nat := [4127]
set_option maxRecDepth 8192 in
theorem root198Candidates1_checked : candidates edge 4 1229 2081 root198Tail2 winner.bound = root198Candidates1 := by
  unfold candidates
  have row : edge 2081 = (SortedNatTree.contains FastRows.row2081) := by
    change Data.row2081.contains = (SortedNatTree.contains FastRows.row2081)
    exact FastRows.row2081_checked.symm
  rw [row]
  decide
theorem root198Step1 : scan edge (search edge 3) 4 [1229] 1229 root198Tail1 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail2 winner := by
  rw [root198Tail1, scan, if_neg (by decide), root198Candidates1_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail2
    (if 3 ≤ root198Candidates1.length then search edge 3 [1229, 2081] root198Candidates1 3310 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates2 : List Nat := [21521]
set_option maxRecDepth 8192 in
theorem root198Candidates2_checked : candidates edge 4 1229 2663 root198Tail3 winner.bound = root198Candidates2 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root198Step2 : scan edge (search edge 3) 4 [1229] 1229 root198Tail2 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail3 winner := by
  rw [root198Tail2, scan, if_neg (by decide), root198Candidates2_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail3
    (if 3 ≤ root198Candidates2.length then search edge 3 [1229, 2663] root198Candidates2 3892 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates3 : List Nat := [5693, 5861]
set_option maxRecDepth 8192 in
theorem root198Candidates3_checked : candidates edge 4 1229 2963 root198Tail4 winner.bound = root198Candidates3 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root198Step3 : scan edge (search edge 3) 4 [1229] 1229 root198Tail3 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail4 winner := by
  rw [root198Tail3, scan, if_neg (by decide), root198Candidates3_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail4
    (if 3 ≤ root198Candidates3.length then search edge 3 [1229, 2963] root198Candidates3 4192 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root198Candidates4_checked : candidates edge 4 1229 3089 root198Tail5 winner.bound = root198Candidates4 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root198Step4 : scan edge (search edge 3) 4 [1229] 1229 root198Tail4 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail5 winner := by
  rw [root198Tail4, scan, if_neg (by decide), root198Candidates4_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail5
    (if 3 ≤ root198Candidates4.length then search edge 3 [1229, 3089] root198Candidates4 4318 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates5 : List Nat := [15263]
set_option maxRecDepth 8192 in
theorem root198Candidates5_checked : candidates edge 4 1229 3389 root198Tail6 winner.bound = root198Candidates5 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
theorem root198Step5 : scan edge (search edge 3) 4 [1229] 1229 root198Tail5 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail6 winner := by
  rw [root198Tail5, scan, if_neg (by decide), root198Candidates5_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail6
    (if 3 ≤ root198Candidates5.length then search edge 3 [1229, 3389] root198Candidates5 4618 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates6 : List Nat := [10589, 15263]
set_option maxRecDepth 8192 in
theorem root198Candidates6_checked : candidates edge 4 1229 3881 root198Tail7 winner.bound = root198Candidates6 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root198Step6 : scan edge (search edge 3) 4 [1229] 1229 root198Tail6 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail7 winner := by
  rw [root198Tail6, scan, if_neg (by decide), root198Candidates6_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail7
    (if 3 ≤ root198Candidates6.length then search edge 3 [1229, 3881] root198Candidates6 5110 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates7 : List Nat := [11681]
set_option maxRecDepth 8192 in
theorem root198Candidates7_checked : candidates edge 4 1229 4127 root198Tail8 winner.bound = root198Candidates7 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root198Step7 : scan edge (search edge 3) 4 [1229] 1229 root198Tail7 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail8 winner := by
  rw [root198Tail7, scan, if_neg (by decide), root198Candidates7_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail8
    (if 3 ≤ root198Candidates7.length then search edge 3 [1229, 4127] root198Candidates7 5356 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates8 : List Nat := [5399]
set_option maxRecDepth 8192 in
theorem root198Candidates8_checked : candidates edge 4 1229 4229 root198Tail9 winner.bound = root198Candidates8 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root198Step8 : scan edge (search edge 3) 4 [1229] 1229 root198Tail8 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail9 winner := by
  rw [root198Tail8, scan, if_neg (by decide), root198Candidates8_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail9
    (if 3 ≤ root198Candidates8.length then search edge 3 [1229, 4229] root198Candidates8 5458 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root198Candidates9_checked : candidates edge 4 1229 5399 root198Tail10 winner.bound = root198Candidates9 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root198Step9 : scan edge (search edge 3) 4 [1229] 1229 root198Tail9 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail10 winner := by
  rw [root198Tail9, scan, if_neg (by decide), root198Candidates9_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail10
    (if 3 ≤ root198Candidates9.length then search edge 3 [1229, 5399] root198Candidates9 6628 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root198Candidates10_checked : candidates edge 4 1229 5693 root198Tail11 winner.bound = root198Candidates10 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root198Step10 : scan edge (search edge 3) 4 [1229] 1229 root198Tail10 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail11 winner := by
  rw [root198Tail10, scan, if_neg (by decide), root198Candidates10_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail11
    (if 3 ≤ root198Candidates10.length then search edge 3 [1229, 5693] root198Candidates10 6922 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root198Candidates11_checked : candidates edge 4 1229 5741 root198Tail12 winner.bound = root198Candidates11 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
theorem root198Step11 : scan edge (search edge 3) 4 [1229] 1229 root198Tail11 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail12 winner := by
  rw [root198Tail11, scan, if_neg (by decide), root198Candidates11_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail12
    (if 3 ≤ root198Candidates11.length then search edge 3 [1229, 5741] root198Candidates11 6970 winner else winner) = _
  rw [if_neg (by decide)]
def root198Candidates12 : List Nat := [11681]
set_option maxRecDepth 8192 in
theorem root198Candidates12_checked : candidates edge 4 1229 5861 root198Tail13 winner.bound = root198Candidates12 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root198Step12 : scan edge (search edge 3) 4 [1229] 1229 root198Tail12 winner =
    scan edge (search edge 3) 4 [1229] 1229 root198Tail13 winner := by
  rw [root198Tail12, scan, if_neg (by decide), root198Candidates12_checked]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail13
    (if 3 ≤ root198Candidates12.length then search edge 3 [1229, 5861] root198Candidates12 7090 winner else winner) = _
  rw [if_neg (by decide)]
theorem root198Scan13 : scan edge (search edge 3) 4 [1229] 1229 root198Tail13 winner = winner := by rw [root198Tail13, scan, if_pos (by decide)]
theorem root198Scan12 : scan edge (search edge 3) 4 [1229] 1229 root198Tail12 winner = winner :=
  (root198Step12).trans root198Scan13
theorem root198Scan11 : scan edge (search edge 3) 4 [1229] 1229 root198Tail11 winner = winner :=
  (root198Step11).trans root198Scan12
theorem root198Scan10 : scan edge (search edge 3) 4 [1229] 1229 root198Tail10 winner = winner :=
  (root198Step10).trans root198Scan11
theorem root198Scan9 : scan edge (search edge 3) 4 [1229] 1229 root198Tail9 winner = winner :=
  (root198Step9).trans root198Scan10
theorem root198Scan8 : scan edge (search edge 3) 4 [1229] 1229 root198Tail8 winner = winner :=
  (root198Step8).trans root198Scan9
theorem root198Scan7 : scan edge (search edge 3) 4 [1229] 1229 root198Tail7 winner = winner :=
  (root198Step7).trans root198Scan8
theorem root198Scan6 : scan edge (search edge 3) 4 [1229] 1229 root198Tail6 winner = winner :=
  (root198Step6).trans root198Scan7
theorem root198Scan5 : scan edge (search edge 3) 4 [1229] 1229 root198Tail5 winner = winner :=
  (root198Step5).trans root198Scan6
theorem root198Scan4 : scan edge (search edge 3) 4 [1229] 1229 root198Tail4 winner = winner :=
  (root198Step4).trans root198Scan5
theorem root198Scan3 : scan edge (search edge 3) 4 [1229] 1229 root198Tail3 winner = winner :=
  (root198Step3).trans root198Scan4
theorem root198Scan2 : scan edge (search edge 3) 4 [1229] 1229 root198Tail2 winner = winner :=
  (root198Step2).trans root198Scan3
theorem root198Scan1 : scan edge (search edge 3) 4 [1229] 1229 root198Tail1 winner = winner :=
  (root198Step1).trans root198Scan2
theorem root198Scan0 : scan edge (search edge 3) 4 [1229] 1229 root198Tail0 winner = winner :=
  (root198Step0).trans root198Scan1
set_option maxRecDepth 4096 in
theorem root198Branch : search edge 4 [1229] root198Candidates 1229 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1229] 1229 root198Tail0 winner = winner
  exact root198Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
