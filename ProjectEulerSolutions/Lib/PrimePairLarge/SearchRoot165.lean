import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root165Candidates : List Nat := Data.row997
set_option maxRecDepth 8192 in
theorem root165Candidates_checked : candidates edge 5 0 997 tail166 winner.bound = root165Candidates := by
  apply candidates_eq_row edge 5 0 997 tail166 root165Candidates winner.bound tail166_ordered
  · decide
  · decide
  · funext q
    rfl
def root165Tail57 : List Nat := []
def root165Tail56 : List Nat := 24841 :: root165Tail57
def root165Tail55 : List Nat := 23977 :: root165Tail56
def root165Tail54 : List Nat := 23623 :: root165Tail55
def root165Tail53 : List Nat := 23557 :: root165Tail54
def root165Tail52 : List Nat := 23017 :: root165Tail53
def root165Tail51 : List Nat := 23011 :: root165Tail52
def root165Tail50 : List Nat := 22741 :: root165Tail51
def root165Tail49 : List Nat := 22171 :: root165Tail50
def root165Tail48 : List Nat := 21319 :: root165Tail49
def root165Tail47 : List Nat := 21313 :: root165Tail48
def root165Tail46 : List Nat := 20287 :: root165Tail47
def root165Tail45 : List Nat := 19471 :: root165Tail46
def root165Tail44 : List Nat := 18121 :: root165Tail45
def root165Tail43 : List Nat := 18097 :: root165Tail44
def root165Tail42 : List Nat := 18013 :: root165Tail43
def root165Tail41 : List Nat := 17707 :: root165Tail42
def root165Tail40 : List Nat := 17659 :: root165Tail41
def root165Tail39 : List Nat := 17623 :: root165Tail40
def root165Tail38 : List Nat := 17467 :: root165Tail39
def root165Tail37 : List Nat := 17011 :: root165Tail38
def root165Tail36 : List Nat := 16087 :: root165Tail37
def root165Tail35 : List Nat := 15937 :: root165Tail36
def root165Tail34 : List Nat := 15901 :: root165Tail35
def root165Tail33 : List Nat := 15877 :: root165Tail34
def root165Tail32 : List Nat := 15493 :: root165Tail33
def root165Tail31 : List Nat := 15331 :: root165Tail32
def root165Tail30 : List Nat := 14341 :: root165Tail31
def root165Tail29 : List Nat := 14281 :: root165Tail30
def root165Tail28 : List Nat := 14221 :: root165Tail29
def root165Tail27 : List Nat := 14107 :: root165Tail28
def root165Tail26 : List Nat := 13249 :: root165Tail27
def root165Tail25 : List Nat := 13147 :: root165Tail26
def root165Tail24 : List Nat := 12577 :: root165Tail25
def root165Tail23 : List Nat := 11197 :: root165Tail24
def root165Tail22 : List Nat := 11083 :: root165Tail23
def root165Tail21 : List Nat := 10939 :: root165Tail22
def root165Tail20 : List Nat := 10753 :: root165Tail21
def root165Tail19 : List Nat := 10333 :: root165Tail20
def root165Tail18 : List Nat := 10099 :: root165Tail19
def root165Tail17 : List Nat := 9091 :: root165Tail18
def root165Tail16 : List Nat := 7741 :: root165Tail17
def root165Tail15 : List Nat := 7057 :: root165Tail16
def root165Tail14 : List Nat := 6763 :: root165Tail15
def root165Tail13 : List Nat := 6481 :: root165Tail14
def root165Tail12 : List Nat := 6091 :: root165Tail13
def root165Tail11 : List Nat := 5569 :: root165Tail12
def root165Tail10 : List Nat := 4969 :: root165Tail11
def root165Tail9 : List Nat := 4513 :: root165Tail10
def root165Tail8 : List Nat := 4243 :: root165Tail9
def root165Tail7 : List Nat := 4177 :: root165Tail8
def root165Tail6 : List Nat := 3331 :: root165Tail7
def root165Tail5 : List Nat := 3217 :: root165Tail6
def root165Tail4 : List Nat := 2617 :: root165Tail5
def root165Tail3 : List Nat := 2377 :: root165Tail4
def root165Tail2 : List Nat := 2029 :: root165Tail3
def root165Tail1 : List Nat := 1993 :: root165Tail2
def root165Tail0 : List Nat := 1483 :: root165Tail1
def root165Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root165Candidates0_checked : candidates edge 4 997 1483 root165Tail1 winner.bound = root165Candidates0 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
theorem root165Step0 : scan edge (search edge 3) 4 [997] 997 root165Tail0 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail1 winner := by
  rw [root165Tail0, scan, if_neg (by decide), root165Candidates0_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail1
    (if 3 ≤ root165Candidates0.length then search edge 3 [997, 1483] root165Candidates0 2480 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates1 : List Nat := [6481, 12577]
set_option maxRecDepth 8192 in
theorem root165Candidates1_checked : candidates edge 4 997 1993 root165Tail2 winner.bound = root165Candidates1 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
theorem root165Step1 : scan edge (search edge 3) 4 [997] 997 root165Tail1 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail2 winner := by
  rw [root165Tail1, scan, if_neg (by decide), root165Candidates1_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail2
    (if 3 ≤ root165Candidates1.length then search edge 3 [997, 1993] root165Candidates1 2990 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates2 : List Nat := [19471]
set_option maxRecDepth 8192 in
theorem root165Candidates2_checked : candidates edge 4 997 2029 root165Tail3 winner.bound = root165Candidates2 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
theorem root165Step2 : scan edge (search edge 3) 4 [997] 997 root165Tail2 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail3 winner := by
  rw [root165Tail2, scan, if_neg (by decide), root165Candidates2_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail3
    (if 3 ≤ root165Candidates2.length then search edge 3 [997, 2029] root165Candidates2 3026 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates3 : List Nat := [22171]
set_option maxRecDepth 8192 in
theorem root165Candidates3_checked : candidates edge 4 997 2377 root165Tail4 winner.bound = root165Candidates3 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
theorem root165Step3 : scan edge (search edge 3) 4 [997] 997 root165Tail3 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail4 winner := by
  rw [root165Tail3, scan, if_neg (by decide), root165Candidates3_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail4
    (if 3 ≤ root165Candidates3.length then search edge 3 [997, 2377] root165Candidates3 3374 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates4 : List Nat := [7741, 17707]
set_option maxRecDepth 8192 in
theorem root165Candidates4_checked : candidates edge 4 997 2617 root165Tail5 winner.bound = root165Candidates4 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
theorem root165Step4 : scan edge (search edge 3) 4 [997] 997 root165Tail4 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail5 winner := by
  rw [root165Tail4, scan, if_neg (by decide), root165Candidates4_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail5
    (if 3 ≤ root165Candidates4.length then search edge 3 [997, 2617] root165Candidates4 3614 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates5 : List Nat := [11197]
set_option maxRecDepth 8192 in
theorem root165Candidates5_checked : candidates edge 4 997 3217 root165Tail6 winner.bound = root165Candidates5 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root165Step5 : scan edge (search edge 3) 4 [997] 997 root165Tail5 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail6 winner := by
  rw [root165Tail5, scan, if_neg (by decide), root165Candidates5_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail6
    (if 3 ≤ root165Candidates5.length then search edge 3 [997, 3217] root165Candidates5 4214 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates6 : List Nat := [10333, 17011]
set_option maxRecDepth 8192 in
theorem root165Candidates6_checked : candidates edge 4 997 3331 root165Tail7 winner.bound = root165Candidates6 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root165Step6 : scan edge (search edge 3) 4 [997] 997 root165Tail6 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail7 winner := by
  rw [root165Tail6, scan, if_neg (by decide), root165Candidates6_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail7
    (if 3 ≤ root165Candidates6.length then search edge 3 [997, 3331] root165Candidates6 4328 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates7 : List Nat := [15937]
set_option maxRecDepth 8192 in
theorem root165Candidates7_checked : candidates edge 4 997 4177 root165Tail8 winner.bound = root165Candidates7 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root165Step7 : scan edge (search edge 3) 4 [997] 997 root165Tail7 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail8 winner := by
  rw [root165Tail7, scan, if_neg (by decide), root165Candidates7_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail8
    (if 3 ≤ root165Candidates7.length then search edge 3 [997, 4177] root165Candidates7 5174 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates8 : List Nat := [11197]
set_option maxRecDepth 8192 in
theorem root165Candidates8_checked : candidates edge 4 997 4243 root165Tail9 winner.bound = root165Candidates8 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root165Step8 : scan edge (search edge 3) 4 [997] 997 root165Tail8 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail9 winner := by
  rw [root165Tail8, scan, if_neg (by decide), root165Candidates8_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail9
    (if 3 ≤ root165Candidates8.length then search edge 3 [997, 4243] root165Candidates8 5240 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates9 : List Nat := [10099, 10333, 15937]
set_option maxRecDepth 8192 in
theorem root165Candidates9_checked : candidates edge 4 997 4513 root165Tail10 winner.bound = root165Candidates9 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root165Child9 : search edge 3 [997, 4513] root165Candidates9 5510 winner = winner := by decide
theorem root165Step9 : scan edge (search edge 3) 4 [997] 997 root165Tail9 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail10 winner := by
  rw [root165Tail9, scan, if_neg (by decide), root165Candidates9_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail10
    (if 3 ≤ root165Candidates9.length then search edge 3 [997, 4513] root165Candidates9 5510 winner else winner) = _
  rw [if_pos (by decide), root165Child9]
def root165Candidates10 : List Nat := [19471]
set_option maxRecDepth 8192 in
theorem root165Candidates10_checked : candidates edge 4 997 4969 root165Tail11 winner.bound = root165Candidates10 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root165Step10 : scan edge (search edge 3) 4 [997] 997 root165Tail10 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail11 winner := by
  rw [root165Tail10, scan, if_neg (by decide), root165Candidates10_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail11
    (if 3 ≤ root165Candidates10.length then search edge 3 [997, 4969] root165Candidates10 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root165Candidates11_checked : candidates edge 4 997 5569 root165Tail12 winner.bound = root165Candidates11 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root165Step11 : scan edge (search edge 3) 4 [997] 997 root165Tail11 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail12 winner := by
  rw [root165Tail11, scan, if_neg (by decide), root165Candidates11_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail12
    (if 3 ≤ root165Candidates11.length then search edge 3 [997, 5569] root165Candidates11 6566 winner else winner) = _
  rw [if_neg (by decide)]
def root165Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root165Candidates12_checked : candidates edge 4 997 6091 root165Tail13 winner.bound = root165Candidates12 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root165Step12 : scan edge (search edge 3) 4 [997] 997 root165Tail12 winner =
    scan edge (search edge 3) 4 [997] 997 root165Tail13 winner := by
  rw [root165Tail12, scan, if_neg (by decide), root165Candidates12_checked]
  change scan edge (search edge 3) 4 [997] 997 root165Tail13
    (if 3 ≤ root165Candidates12.length then search edge 3 [997, 6091] root165Candidates12 7088 winner else winner) = _
  rw [if_neg (by decide)]
theorem root165Scan13 : scan edge (search edge 3) 4 [997] 997 root165Tail13 winner = winner := by rw [root165Tail13, scan, if_pos (by decide)]
theorem root165Scan12 : scan edge (search edge 3) 4 [997] 997 root165Tail12 winner = winner :=
  (root165Step12).trans root165Scan13
theorem root165Scan11 : scan edge (search edge 3) 4 [997] 997 root165Tail11 winner = winner :=
  (root165Step11).trans root165Scan12
theorem root165Scan10 : scan edge (search edge 3) 4 [997] 997 root165Tail10 winner = winner :=
  (root165Step10).trans root165Scan11
theorem root165Scan9 : scan edge (search edge 3) 4 [997] 997 root165Tail9 winner = winner :=
  (root165Step9).trans root165Scan10
theorem root165Scan8 : scan edge (search edge 3) 4 [997] 997 root165Tail8 winner = winner :=
  (root165Step8).trans root165Scan9
theorem root165Scan7 : scan edge (search edge 3) 4 [997] 997 root165Tail7 winner = winner :=
  (root165Step7).trans root165Scan8
theorem root165Scan6 : scan edge (search edge 3) 4 [997] 997 root165Tail6 winner = winner :=
  (root165Step6).trans root165Scan7
theorem root165Scan5 : scan edge (search edge 3) 4 [997] 997 root165Tail5 winner = winner :=
  (root165Step5).trans root165Scan6
theorem root165Scan4 : scan edge (search edge 3) 4 [997] 997 root165Tail4 winner = winner :=
  (root165Step4).trans root165Scan5
theorem root165Scan3 : scan edge (search edge 3) 4 [997] 997 root165Tail3 winner = winner :=
  (root165Step3).trans root165Scan4
theorem root165Scan2 : scan edge (search edge 3) 4 [997] 997 root165Tail2 winner = winner :=
  (root165Step2).trans root165Scan3
theorem root165Scan1 : scan edge (search edge 3) 4 [997] 997 root165Tail1 winner = winner :=
  (root165Step1).trans root165Scan2
theorem root165Scan0 : scan edge (search edge 3) 4 [997] 997 root165Tail0 winner = winner :=
  (root165Step0).trans root165Scan1
set_option maxRecDepth 4096 in
theorem root165Branch : search edge 4 [997] root165Candidates 997 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [997] 997 root165Tail0 winner = winner
  exact root165Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
