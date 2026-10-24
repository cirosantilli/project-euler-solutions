import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root170Candidates : List Nat := Data.row1031
set_option maxRecDepth 8192 in
theorem root170Candidates_checked : candidates edge 5 0 1031 tail171 winner.bound = root170Candidates := by
  apply candidates_eq_row edge 5 0 1031 tail171 root170Candidates winner.bound tail171_ordered
  · decide
  · decide
  · funext q
    rfl
def root170Tail53 : List Nat := []
def root170Tail52 : List Nat := 24767 :: root170Tail53
def root170Tail51 : List Nat := 24407 :: root170Tail52
def root170Tail50 : List Nat := 23561 :: root170Tail51
def root170Tail49 : List Nat := 23399 :: root170Tail50
def root170Tail48 : List Nat := 22871 :: root170Tail49
def root170Tail47 : List Nat := 21803 :: root170Tail48
def root170Tail46 : List Nat := 20939 :: root170Tail47
def root170Tail45 : List Nat := 20333 :: root170Tail46
def root170Tail44 : List Nat := 19949 :: root170Tail45
def root170Tail43 : List Nat := 18797 :: root170Tail44
def root170Tail42 : List Nat := 17681 :: root170Tail43
def root170Tail41 : List Nat := 17117 :: root170Tail42
def root170Tail40 : List Nat := 17093 :: root170Tail41
def root170Tail39 : List Nat := 16703 :: root170Tail40
def root170Tail38 : List Nat := 16421 :: root170Tail39
def root170Tail37 : List Nat := 15737 :: root170Tail38
def root170Tail36 : List Nat := 15017 :: root170Tail37
def root170Tail35 : List Nat := 14951 :: root170Tail36
def root170Tail34 : List Nat := 14519 :: root170Tail35
def root170Tail33 : List Nat := 14489 :: root170Tail34
def root170Tail32 : List Nat := 14447 :: root170Tail33
def root170Tail31 : List Nat := 14051 :: root170Tail32
def root170Tail30 : List Nat := 13463 :: root170Tail31
def root170Tail29 : List Nat := 13103 :: root170Tail30
def root170Tail28 : List Nat := 12923 :: root170Tail29
def root170Tail27 : List Nat := 11867 :: root170Tail28
def root170Tail26 : List Nat := 11831 :: root170Tail27
def root170Tail25 : List Nat := 11321 :: root170Tail26
def root170Tail24 : List Nat := 10781 :: root170Tail25
def root170Tail23 : List Nat := 10691 :: root170Tail24
def root170Tail22 : List Nat := 10463 :: root170Tail23
def root170Tail21 : List Nat := 9851 :: root170Tail22
def root170Tail20 : List Nat := 9839 :: root170Tail21
def root170Tail19 : List Nat := 9371 :: root170Tail20
def root170Tail18 : List Nat := 8573 :: root170Tail19
def root170Tail17 : List Nat := 7529 :: root170Tail18
def root170Tail16 : List Nat := 7253 :: root170Tail17
def root170Tail15 : List Nat := 7193 :: root170Tail16
def root170Tail14 : List Nat := 6899 :: root170Tail15
def root170Tail13 : List Nat := 5237 :: root170Tail14
def root170Tail12 : List Nat := 5003 :: root170Tail13
def root170Tail11 : List Nat := 4643 :: root170Tail12
def root170Tail10 : List Nat := 4253 :: root170Tail11
def root170Tail9 : List Nat := 3533 :: root170Tail10
def root170Tail8 : List Nat := 2927 :: root170Tail9
def root170Tail7 : List Nat := 2531 :: root170Tail8
def root170Tail6 : List Nat := 2207 :: root170Tail7
def root170Tail5 : List Nat := 1997 :: root170Tail6
def root170Tail4 : List Nat := 1913 :: root170Tail5
def root170Tail3 : List Nat := 1811 :: root170Tail4
def root170Tail2 : List Nat := 1709 :: root170Tail3
def root170Tail1 : List Nat := 1601 :: root170Tail2
def root170Tail0 : List Nat := 1481 :: root170Tail1
def root170Candidates0 : List Nat := [7529]
set_option maxRecDepth 8192 in
theorem root170Candidates0_checked : candidates edge 4 1031 1481 root170Tail1 winner.bound = root170Candidates0 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
theorem root170Step0 : scan edge (search edge 3) 4 [1031] 1031 root170Tail0 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail1 winner := by
  rw [root170Tail0, scan, if_neg (by decide), root170Candidates0_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail1
    (if 3 ≤ root170Candidates0.length then search edge 3 [1031, 1481] root170Candidates0 2512 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates1 : List Nat := [1811, 2207, 3533, 10463, 13103, 14489, 14951, 15017]
set_option maxRecDepth 8192 in
theorem root170Candidates1_checked : candidates edge 4 1031 1601 root170Tail2 winner.bound = root170Candidates1 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root170Child1 : search edge 3 [1031, 1601] root170Candidates1 2632 winner = winner := by decide
theorem root170Step1 : scan edge (search edge 3) 4 [1031] 1031 root170Tail1 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail2 winner := by
  rw [root170Tail1, scan, if_neg (by decide), root170Candidates1_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail2
    (if 3 ≤ root170Candidates1.length then search edge 3 [1031, 1601] root170Candidates1 2632 winner else winner) = _
  rw [if_pos (by decide), root170Child1]
def root170Candidates2 : List Nat := [1913, 19949]
set_option maxRecDepth 8192 in
theorem root170Candidates2_checked : candidates edge 4 1031 1709 root170Tail3 winner.bound = root170Candidates2 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
theorem root170Step2 : scan edge (search edge 3) 4 [1031] 1031 root170Tail2 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail3 winner := by
  rw [root170Tail2, scan, if_neg (by decide), root170Candidates2_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail3
    (if 3 ≤ root170Candidates2.length then search edge 3 [1031, 1709] root170Candidates2 2740 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates3 : List Nat := [10691]
set_option maxRecDepth 8192 in
theorem root170Candidates3_checked : candidates edge 4 1031 1811 root170Tail4 winner.bound = root170Candidates3 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
theorem root170Step3 : scan edge (search edge 3) 4 [1031] 1031 root170Tail3 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail4 winner := by
  rw [root170Tail3, scan, if_neg (by decide), root170Candidates3_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail4
    (if 3 ≤ root170Candidates3.length then search edge 3 [1031, 1811] root170Candidates3 2842 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates4 : List Nat := [7253, 9851, 22871]
set_option maxRecDepth 8192 in
theorem root170Candidates4_checked : candidates edge 4 1031 1913 root170Tail5 winner.bound = root170Candidates4 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root170Child4 : search edge 3 [1031, 1913] root170Candidates4 2944 winner = winner := by decide
theorem root170Step4 : scan edge (search edge 3) 4 [1031] 1031 root170Tail4 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail5 winner := by
  rw [root170Tail4, scan, if_neg (by decide), root170Candidates4_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail5
    (if 3 ≤ root170Candidates4.length then search edge 3 [1031, 1913] root170Candidates4 2944 winner else winner) = _
  rw [if_pos (by decide), root170Child4]
def root170Candidates5 : List Nat := [14951, 17681]
set_option maxRecDepth 8192 in
theorem root170Candidates5_checked : candidates edge 4 1031 1997 root170Tail6 winner.bound = root170Candidates5 := by
  unfold candidates
  have row : edge 1997 = (SortedNatTree.contains FastRows.row1997) := by
    change Data.row1997.contains = (SortedNatTree.contains FastRows.row1997)
    exact FastRows.row1997_checked.symm
  rw [row]
  decide
theorem root170Step5 : scan edge (search edge 3) 4 [1031] 1031 root170Tail5 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail6 winner := by
  rw [root170Tail5, scan, if_neg (by decide), root170Candidates5_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail6
    (if 3 ≤ root170Candidates5.length then search edge 3 [1031, 1997] root170Candidates5 3028 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates6 : List Nat := [7193, 15017, 16421]
set_option maxRecDepth 8192 in
theorem root170Candidates6_checked : candidates edge 4 1031 2207 root170Tail7 winner.bound = root170Candidates6 := by
  unfold candidates
  have row : edge 2207 = (SortedNatTree.contains FastRows.row2207) := by
    change Data.row2207.contains = (SortedNatTree.contains FastRows.row2207)
    exact FastRows.row2207_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root170Child6 : search edge 3 [1031, 2207] root170Candidates6 3238 winner = winner := by decide
theorem root170Step6 : scan edge (search edge 3) 4 [1031] 1031 root170Tail6 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail7 winner := by
  rw [root170Tail6, scan, if_neg (by decide), root170Candidates6_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail7
    (if 3 ≤ root170Candidates6.length then search edge 3 [1031, 2207] root170Candidates6 3238 winner else winner) = _
  rw [if_pos (by decide), root170Child6]
def root170Candidates7 : List Nat := [4643, 11867]
set_option maxRecDepth 8192 in
theorem root170Candidates7_checked : candidates edge 4 1031 2531 root170Tail8 winner.bound = root170Candidates7 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
theorem root170Step7 : scan edge (search edge 3) 4 [1031] 1031 root170Tail7 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail8 winner := by
  rw [root170Tail7, scan, if_neg (by decide), root170Candidates7_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail8
    (if 3 ≤ root170Candidates7.length then search edge 3 [1031, 2531] root170Candidates7 3562 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates8 : List Nat := [13103]
set_option maxRecDepth 8192 in
theorem root170Candidates8_checked : candidates edge 4 1031 2927 root170Tail9 winner.bound = root170Candidates8 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root170Step8 : scan edge (search edge 3) 4 [1031] 1031 root170Tail8 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail9 winner := by
  rw [root170Tail8, scan, if_neg (by decide), root170Candidates8_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail9
    (if 3 ≤ root170Candidates8.length then search edge 3 [1031, 2927] root170Candidates8 3958 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates9 : List Nat := [10691]
set_option maxRecDepth 8192 in
theorem root170Candidates9_checked : candidates edge 4 1031 3533 root170Tail10 winner.bound = root170Candidates9 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root170Step9 : scan edge (search edge 3) 4 [1031] 1031 root170Tail9 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail10 winner := by
  rw [root170Tail9, scan, if_neg (by decide), root170Candidates9_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail10
    (if 3 ≤ root170Candidates9.length then search edge 3 [1031, 3533] root170Candidates9 4564 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates10 : List Nat := [11867]
set_option maxRecDepth 8192 in
theorem root170Candidates10_checked : candidates edge 4 1031 4253 root170Tail11 winner.bound = root170Candidates10 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root170Step10 : scan edge (search edge 3) 4 [1031] 1031 root170Tail10 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail11 winner := by
  rw [root170Tail10, scan, if_neg (by decide), root170Candidates10_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail11
    (if 3 ≤ root170Candidates10.length then search edge 3 [1031, 4253] root170Candidates10 5284 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates11 : List Nat := [5237]
set_option maxRecDepth 8192 in
theorem root170Candidates11_checked : candidates edge 4 1031 4643 root170Tail12 winner.bound = root170Candidates11 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root170Step11 : scan edge (search edge 3) 4 [1031] 1031 root170Tail11 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail12 winner := by
  rw [root170Tail11, scan, if_neg (by decide), root170Candidates11_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail12
    (if 3 ≤ root170Candidates11.length then search edge 3 [1031, 4643] root170Candidates11 5674 winner else winner) = _
  rw [if_neg (by decide)]
def root170Candidates12 : List Nat := [14051, 14447, 14519]
set_option maxRecDepth 8192 in
theorem root170Candidates12_checked : candidates edge 4 1031 5003 root170Tail13 winner.bound = root170Candidates12 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root170Child12 : search edge 3 [1031, 5003] root170Candidates12 6034 winner = winner := by decide
theorem root170Step12 : scan edge (search edge 3) 4 [1031] 1031 root170Tail12 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail13 winner := by
  rw [root170Tail12, scan, if_neg (by decide), root170Candidates12_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail13
    (if 3 ≤ root170Candidates12.length then search edge 3 [1031, 5003] root170Candidates12 6034 winner else winner) = _
  rw [if_pos (by decide), root170Child12]
def root170Candidates13 : List Nat := [13103, 14447]
set_option maxRecDepth 8192 in
theorem root170Candidates13_checked : candidates edge 4 1031 5237 root170Tail14 winner.bound = root170Candidates13 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root170Step13 : scan edge (search edge 3) 4 [1031] 1031 root170Tail13 winner =
    scan edge (search edge 3) 4 [1031] 1031 root170Tail14 winner := by
  rw [root170Tail13, scan, if_neg (by decide), root170Candidates13_checked]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail14
    (if 3 ≤ root170Candidates13.length then search edge 3 [1031, 5237] root170Candidates13 6268 winner else winner) = _
  rw [if_neg (by decide)]
theorem root170Scan14 : scan edge (search edge 3) 4 [1031] 1031 root170Tail14 winner = winner := by rw [root170Tail14, scan, if_pos (by decide)]
theorem root170Scan13 : scan edge (search edge 3) 4 [1031] 1031 root170Tail13 winner = winner :=
  (root170Step13).trans root170Scan14
theorem root170Scan12 : scan edge (search edge 3) 4 [1031] 1031 root170Tail12 winner = winner :=
  (root170Step12).trans root170Scan13
theorem root170Scan11 : scan edge (search edge 3) 4 [1031] 1031 root170Tail11 winner = winner :=
  (root170Step11).trans root170Scan12
theorem root170Scan10 : scan edge (search edge 3) 4 [1031] 1031 root170Tail10 winner = winner :=
  (root170Step10).trans root170Scan11
theorem root170Scan9 : scan edge (search edge 3) 4 [1031] 1031 root170Tail9 winner = winner :=
  (root170Step9).trans root170Scan10
theorem root170Scan8 : scan edge (search edge 3) 4 [1031] 1031 root170Tail8 winner = winner :=
  (root170Step8).trans root170Scan9
theorem root170Scan7 : scan edge (search edge 3) 4 [1031] 1031 root170Tail7 winner = winner :=
  (root170Step7).trans root170Scan8
theorem root170Scan6 : scan edge (search edge 3) 4 [1031] 1031 root170Tail6 winner = winner :=
  (root170Step6).trans root170Scan7
theorem root170Scan5 : scan edge (search edge 3) 4 [1031] 1031 root170Tail5 winner = winner :=
  (root170Step5).trans root170Scan6
theorem root170Scan4 : scan edge (search edge 3) 4 [1031] 1031 root170Tail4 winner = winner :=
  (root170Step4).trans root170Scan5
theorem root170Scan3 : scan edge (search edge 3) 4 [1031] 1031 root170Tail3 winner = winner :=
  (root170Step3).trans root170Scan4
theorem root170Scan2 : scan edge (search edge 3) 4 [1031] 1031 root170Tail2 winner = winner :=
  (root170Step2).trans root170Scan3
theorem root170Scan1 : scan edge (search edge 3) 4 [1031] 1031 root170Tail1 winner = winner :=
  (root170Step1).trans root170Scan2
theorem root170Scan0 : scan edge (search edge 3) 4 [1031] 1031 root170Tail0 winner = winner :=
  (root170Step0).trans root170Scan1
set_option maxRecDepth 4096 in
theorem root170Branch : search edge 4 [1031] root170Candidates 1031 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1031] 1031 root170Tail0 winner = winner
  exact root170Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
