import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root160Candidates : List Nat := Data.row967
set_option maxRecDepth 8192 in
theorem root160Candidates_checked : candidates edge 5 0 967 tail161 winner.bound = root160Candidates := by
  apply candidates_eq_row edge 5 0 967 tail161 root160Candidates winner.bound tail161_ordered
  · decide
  · decide
  · funext q
    rfl
def root160Tail53 : List Nat := []
def root160Tail52 : List Nat := 24967 :: root160Tail53
def root160Tail51 : List Nat := 24391 :: root160Tail52
def root160Tail50 : List Nat := 24379 :: root160Tail51
def root160Tail49 : List Nat := 24049 :: root160Tail50
def root160Tail48 : List Nat := 23677 :: root160Tail49
def root160Tail47 : List Nat := 22777 :: root160Tail48
def root160Tail46 : List Nat := 22111 :: root160Tail47
def root160Tail45 : List Nat := 21649 :: root160Tail46
def root160Tail44 : List Nat := 20719 :: root160Tail45
def root160Tail43 : List Nat := 19891 :: root160Tail44
def root160Tail42 : List Nat := 19501 :: root160Tail43
def root160Tail41 : List Nat := 19213 :: root160Tail42
def root160Tail40 : List Nat := 18679 :: root160Tail41
def root160Tail39 : List Nat := 18553 :: root160Tail40
def root160Tail38 : List Nat := 17443 :: root160Tail39
def root160Tail37 : List Nat := 16453 :: root160Tail38
def root160Tail36 : List Nat := 16183 :: root160Tail37
def root160Tail35 : List Nat := 16069 :: root160Tail36
def root160Tail34 : List Nat := 15973 :: root160Tail35
def root160Tail33 : List Nat := 15061 :: root160Tail34
def root160Tail32 : List Nat := 14713 :: root160Tail33
def root160Tail31 : List Nat := 13009 :: root160Tail32
def root160Tail30 : List Nat := 12919 :: root160Tail31
def root160Tail29 : List Nat := 12697 :: root160Tail30
def root160Tail28 : List Nat := 12379 :: root160Tail29
def root160Tail27 : List Nat := 12373 :: root160Tail28
def root160Tail26 : List Nat := 11161 :: root160Tail27
def root160Tail25 : List Nat := 10993 :: root160Tail26
def root160Tail24 : List Nat := 10303 :: root160Tail25
def root160Tail23 : List Nat := 9277 :: root160Tail24
def root160Tail22 : List Nat := 8761 :: root160Tail23
def root160Tail21 : List Nat := 8269 :: root160Tail22
def root160Tail20 : List Nat := 8017 :: root160Tail21
def root160Tail19 : List Nat := 6967 :: root160Tail20
def root160Tail18 : List Nat := 6949 :: root160Tail19
def root160Tail17 : List Nat := 6829 :: root160Tail18
def root160Tail16 : List Nat := 6607 :: root160Tail17
def root160Tail15 : List Nat := 6421 :: root160Tail16
def root160Tail14 : List Nat := 5857 :: root160Tail15
def root160Tail13 : List Nat := 5779 :: root160Tail14
def root160Tail12 : List Nat := 5623 :: root160Tail13
def root160Tail11 : List Nat := 5011 :: root160Tail12
def root160Tail10 : List Nat := 4507 :: root160Tail11
def root160Tail9 : List Nat := 3583 :: root160Tail10
def root160Tail8 : List Nat := 3511 :: root160Tail9
def root160Tail7 : List Nat := 3049 :: root160Tail8
def root160Tail6 : List Nat := 2791 :: root160Tail7
def root160Tail5 : List Nat := 2557 :: root160Tail6
def root160Tail4 : List Nat := 2389 :: root160Tail5
def root160Tail3 : List Nat := 1951 :: root160Tail4
def root160Tail2 : List Nat := 1663 :: root160Tail3
def root160Tail1 : List Nat := 1381 :: root160Tail2
def root160Tail0 : List Nat := 1117 :: root160Tail1
def root160Candidates0 : List Nat := [3511, 10993, 15061]
set_option maxRecDepth 8192 in
theorem root160Candidates0_checked : candidates edge 4 967 1117 root160Tail1 winner.bound = root160Candidates0 := by
  unfold candidates
  have row : edge 1117 = (SortedNatTree.contains FastRows.row1117) := by
    change Data.row1117.contains = (SortedNatTree.contains FastRows.row1117)
    exact FastRows.row1117_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root160Child0 : search edge 3 [967, 1117] root160Candidates0 2084 winner = winner := by decide
theorem root160Step0 : scan edge (search edge 3) 4 [967] 967 root160Tail0 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail1 winner := by
  rw [root160Tail0, scan, if_neg (by decide), root160Candidates0_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail1
    (if 3 ≤ root160Candidates0.length then search edge 3 [967, 1117] root160Candidates0 2084 winner else winner) = _
  rw [if_pos (by decide), root160Child0]
def root160Candidates1 : List Nat := [8017]
set_option maxRecDepth 8192 in
theorem root160Candidates1_checked : candidates edge 4 967 1381 root160Tail2 winner.bound = root160Candidates1 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root160Step1 : scan edge (search edge 3) 4 [967] 967 root160Tail1 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail2 winner := by
  rw [root160Tail1, scan, if_neg (by decide), root160Candidates1_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail2
    (if 3 ≤ root160Candidates1.length then search edge 3 [967, 1381] root160Candidates1 2348 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root160Candidates2_checked : candidates edge 4 967 1663 root160Tail3 winner.bound = root160Candidates2 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
theorem root160Step2 : scan edge (search edge 3) 4 [967] 967 root160Tail2 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail3 winner := by
  rw [root160Tail2, scan, if_neg (by decide), root160Candidates2_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail3
    (if 3 ≤ root160Candidates2.length then search edge 3 [967, 1663] root160Candidates2 2630 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root160Candidates3_checked : candidates edge 4 967 1951 root160Tail4 winner.bound = root160Candidates3 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
theorem root160Step3 : scan edge (search edge 3) 4 [967] 967 root160Tail3 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail4 winner := by
  rw [root160Tail3, scan, if_neg (by decide), root160Candidates3_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail4
    (if 3 ≤ root160Candidates3.length then search edge 3 [967, 1951] root160Candidates3 2918 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates4 : List Nat := [8761]
set_option maxRecDepth 8192 in
theorem root160Candidates4_checked : candidates edge 4 967 2389 root160Tail5 winner.bound = root160Candidates4 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
theorem root160Step4 : scan edge (search edge 3) 4 [967] 967 root160Tail4 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail5 winner := by
  rw [root160Tail4, scan, if_neg (by decide), root160Candidates4_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail5
    (if 3 ≤ root160Candidates4.length then search edge 3 [967, 2389] root160Candidates4 3356 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root160Candidates5_checked : candidates edge 4 967 2557 root160Tail6 winner.bound = root160Candidates5 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
theorem root160Step5 : scan edge (search edge 3) 4 [967] 967 root160Tail5 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail6 winner := by
  rw [root160Tail5, scan, if_neg (by decide), root160Candidates5_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail6
    (if 3 ≤ root160Candidates5.length then search edge 3 [967, 2557] root160Candidates5 3524 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates6 : List Nat := [6607]
set_option maxRecDepth 8192 in
theorem root160Candidates6_checked : candidates edge 4 967 2791 root160Tail7 winner.bound = root160Candidates6 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root160Step6 : scan edge (search edge 3) 4 [967] 967 root160Tail6 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail7 winner := by
  rw [root160Tail6, scan, if_neg (by decide), root160Candidates6_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail7
    (if 3 ≤ root160Candidates6.length then search edge 3 [967, 2791] root160Candidates6 3758 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root160Candidates7_checked : candidates edge 4 967 3049 root160Tail8 winner.bound = root160Candidates7 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root160Step7 : scan edge (search edge 3) 4 [967] 967 root160Tail7 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail8 winner := by
  rw [root160Tail7, scan, if_neg (by decide), root160Candidates7_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail8
    (if 3 ≤ root160Candidates7.length then search edge 3 [967, 3049] root160Candidates7 4016 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates8 : List Nat := [6967, 11161, 12919]
set_option maxRecDepth 8192 in
theorem root160Candidates8_checked : candidates edge 4 967 3511 root160Tail9 winner.bound = root160Candidates8 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root160Child8 : search edge 3 [967, 3511] root160Candidates8 4478 winner = winner := by decide
theorem root160Step8 : scan edge (search edge 3) 4 [967] 967 root160Tail8 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail9 winner := by
  rw [root160Tail8, scan, if_neg (by decide), root160Candidates8_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail9
    (if 3 ≤ root160Candidates8.length then search edge 3 [967, 3511] root160Candidates8 4478 winner else winner) = _
  rw [if_pos (by decide), root160Child8]
def root160Candidates9 : List Nat := [5857, 14713]
set_option maxRecDepth 8192 in
theorem root160Candidates9_checked : candidates edge 4 967 3583 root160Tail10 winner.bound = root160Candidates9 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root160Step9 : scan edge (search edge 3) 4 [967] 967 root160Tail9 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail10 winner := by
  rw [root160Tail9, scan, if_neg (by decide), root160Candidates9_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail10
    (if 3 ≤ root160Candidates9.length then search edge 3 [967, 3583] root160Candidates9 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root160Candidates10_checked : candidates edge 4 967 4507 root160Tail11 winner.bound = root160Candidates10 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root160Step10 : scan edge (search edge 3) 4 [967] 967 root160Tail10 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail11 winner := by
  rw [root160Tail10, scan, if_neg (by decide), root160Candidates10_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail11
    (if 3 ≤ root160Candidates10.length then search edge 3 [967, 4507] root160Candidates10 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates11 : List Nat := [6949, 13009]
set_option maxRecDepth 8192 in
theorem root160Candidates11_checked : candidates edge 4 967 5011 root160Tail12 winner.bound = root160Candidates11 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root160Step11 : scan edge (search edge 3) 4 [967] 967 root160Tail11 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail12 winner := by
  rw [root160Tail11, scan, if_neg (by decide), root160Candidates11_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail12
    (if 3 ≤ root160Candidates11.length then search edge 3 [967, 5011] root160Candidates11 5978 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates12 : List Nat := [6421, 6949]
set_option maxRecDepth 8192 in
theorem root160Candidates12_checked : candidates edge 4 967 5623 root160Tail13 winner.bound = root160Candidates12 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root160Step12 : scan edge (search edge 3) 4 [967] 967 root160Tail12 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail13 winner := by
  rw [root160Tail12, scan, if_neg (by decide), root160Candidates12_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail13
    (if 3 ≤ root160Candidates12.length then search edge 3 [967, 5623] root160Candidates12 6590 winner else winner) = _
  rw [if_neg (by decide)]
def root160Candidates13 : List Nat := [6421, 8269, 8761]
set_option maxRecDepth 8192 in
theorem root160Candidates13_checked : candidates edge 4 967 5779 root160Tail14 winner.bound = root160Candidates13 := by
  unfold candidates
  have row : edge 5779 = (SortedNatTree.contains FastRows.row5779) := by
    change Data.row5779.contains = (SortedNatTree.contains FastRows.row5779)
    exact FastRows.row5779_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root160Child13 : search edge 3 [967, 5779] root160Candidates13 6746 winner = winner := by decide
theorem root160Step13 : scan edge (search edge 3) 4 [967] 967 root160Tail13 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail14 winner := by
  rw [root160Tail13, scan, if_neg (by decide), root160Candidates13_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail14
    (if 3 ≤ root160Candidates13.length then search edge 3 [967, 5779] root160Candidates13 6746 winner else winner) = _
  rw [if_pos (by decide), root160Child13]
def root160Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root160Candidates14_checked : candidates edge 4 967 5857 root160Tail15 winner.bound = root160Candidates14 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root160Step14 : scan edge (search edge 3) 4 [967] 967 root160Tail14 winner =
    scan edge (search edge 3) 4 [967] 967 root160Tail15 winner := by
  rw [root160Tail14, scan, if_neg (by decide), root160Candidates14_checked]
  change scan edge (search edge 3) 4 [967] 967 root160Tail15
    (if 3 ≤ root160Candidates14.length then search edge 3 [967, 5857] root160Candidates14 6824 winner else winner) = _
  rw [if_neg (by decide)]
theorem root160Scan15 : scan edge (search edge 3) 4 [967] 967 root160Tail15 winner = winner := by rw [root160Tail15, scan, if_pos (by decide)]
theorem root160Scan14 : scan edge (search edge 3) 4 [967] 967 root160Tail14 winner = winner :=
  (root160Step14).trans root160Scan15
theorem root160Scan13 : scan edge (search edge 3) 4 [967] 967 root160Tail13 winner = winner :=
  (root160Step13).trans root160Scan14
theorem root160Scan12 : scan edge (search edge 3) 4 [967] 967 root160Tail12 winner = winner :=
  (root160Step12).trans root160Scan13
theorem root160Scan11 : scan edge (search edge 3) 4 [967] 967 root160Tail11 winner = winner :=
  (root160Step11).trans root160Scan12
theorem root160Scan10 : scan edge (search edge 3) 4 [967] 967 root160Tail10 winner = winner :=
  (root160Step10).trans root160Scan11
theorem root160Scan9 : scan edge (search edge 3) 4 [967] 967 root160Tail9 winner = winner :=
  (root160Step9).trans root160Scan10
theorem root160Scan8 : scan edge (search edge 3) 4 [967] 967 root160Tail8 winner = winner :=
  (root160Step8).trans root160Scan9
theorem root160Scan7 : scan edge (search edge 3) 4 [967] 967 root160Tail7 winner = winner :=
  (root160Step7).trans root160Scan8
theorem root160Scan6 : scan edge (search edge 3) 4 [967] 967 root160Tail6 winner = winner :=
  (root160Step6).trans root160Scan7
theorem root160Scan5 : scan edge (search edge 3) 4 [967] 967 root160Tail5 winner = winner :=
  (root160Step5).trans root160Scan6
theorem root160Scan4 : scan edge (search edge 3) 4 [967] 967 root160Tail4 winner = winner :=
  (root160Step4).trans root160Scan5
theorem root160Scan3 : scan edge (search edge 3) 4 [967] 967 root160Tail3 winner = winner :=
  (root160Step3).trans root160Scan4
theorem root160Scan2 : scan edge (search edge 3) 4 [967] 967 root160Tail2 winner = winner :=
  (root160Step2).trans root160Scan3
theorem root160Scan1 : scan edge (search edge 3) 4 [967] 967 root160Tail1 winner = winner :=
  (root160Step1).trans root160Scan2
theorem root160Scan0 : scan edge (search edge 3) 4 [967] 967 root160Tail0 winner = winner :=
  (root160Step0).trans root160Scan1
set_option maxRecDepth 4096 in
theorem root160Branch : search edge 4 [967] root160Candidates 967 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [967] 967 root160Tail0 winner = winner
  exact root160Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
