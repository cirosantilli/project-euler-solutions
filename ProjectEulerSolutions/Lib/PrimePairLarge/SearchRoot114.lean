import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root114Candidates : List Nat := Data.row643
set_option maxRecDepth 8192 in
theorem root114Candidates_checked : candidates edge 5 0 643 tail115 winner.bound = root114Candidates := by
  apply candidates_eq_row edge 5 0 643 tail115 root114Candidates winner.bound tail115_ordered
  · decide
  · decide
  · funext q
    rfl
def root114Tail55 : List Nat := []
def root114Tail54 : List Nat := 25243 :: root114Tail55
def root114Tail53 : List Nat := 25117 :: root114Tail54
def root114Tail52 : List Nat := 24133 :: root114Tail53
def root114Tail51 : List Nat := 23773 :: root114Tail52
def root114Tail50 : List Nat := 23431 :: root114Tail51
def root114Tail49 : List Nat := 21757 :: root114Tail50
def root114Tail48 : List Nat := 21529 :: root114Tail49
def root114Tail47 : List Nat := 21211 :: root114Tail48
def root114Tail46 : List Nat := 20983 :: root114Tail47
def root114Tail45 : List Nat := 20479 :: root114Tail46
def root114Tail44 : List Nat := 20407 :: root114Tail45
def root114Tail43 : List Nat := 20233 :: root114Tail44
def root114Tail42 : List Nat := 19867 :: root114Tail43
def root114Tail41 : List Nat := 19819 :: root114Tail42
def root114Tail40 : List Nat := 19231 :: root114Tail41
def root114Tail39 : List Nat := 19069 :: root114Tail40
def root114Tail38 : List Nat := 18757 :: root114Tail39
def root114Tail37 : List Nat := 18181 :: root114Tail38
def root114Tail36 : List Nat := 17419 :: root114Tail37
def root114Tail35 : List Nat := 16657 :: root114Tail36
def root114Tail34 : List Nat := 15739 :: root114Tail35
def root114Tail33 : List Nat := 15373 :: root114Tail34
def root114Tail32 : List Nat := 15313 :: root114Tail33
def root114Tail31 : List Nat := 14821 :: root114Tail32
def root114Tail30 : List Nat := 14737 :: root114Tail31
def root114Tail29 : List Nat := 14479 :: root114Tail30
def root114Tail28 : List Nat := 14401 :: root114Tail29
def root114Tail27 : List Nat := 14149 :: root114Tail28
def root114Tail26 : List Nat := 13441 :: root114Tail27
def root114Tail25 : List Nat := 12841 :: root114Tail26
def root114Tail24 : List Nat := 12097 :: root114Tail25
def root114Tail23 : List Nat := 12043 :: root114Tail24
def root114Tail22 : List Nat := 10111 :: root114Tail23
def root114Tail21 : List Nat := 9439 :: root114Tail22
def root114Tail20 : List Nat := 9397 :: root114Tail21
def root114Tail19 : List Nat := 9007 :: root114Tail20
def root114Tail18 : List Nat := 8923 :: root114Tail19
def root114Tail17 : List Nat := 8191 :: root114Tail18
def root114Tail16 : List Nat := 8059 :: root114Tail17
def root114Tail15 : List Nat := 7477 :: root114Tail16
def root114Tail14 : List Nat := 7393 :: root114Tail15
def root114Tail13 : List Nat := 6037 :: root114Tail14
def root114Tail12 : List Nat := 6007 :: root114Tail13
def root114Tail11 : List Nat := 5563 :: root114Tail12
def root114Tail10 : List Nat := 5413 :: root114Tail11
def root114Tail9 : List Nat := 5119 :: root114Tail10
def root114Tail8 : List Nat := 4111 :: root114Tail9
def root114Tail7 : List Nat := 2917 :: root114Tail8
def root114Tail6 : List Nat := 2383 :: root114Tail7
def root114Tail5 : List Nat := 2053 :: root114Tail6
def root114Tail4 : List Nat := 2017 :: root114Tail5
def root114Tail3 : List Nat := 1993 :: root114Tail4
def root114Tail2 : List Nat := 1213 :: root114Tail3
def root114Tail1 : List Nat := 991 :: root114Tail2
def root114Tail0 : List Nat := 751 :: root114Tail1
def root114Candidates0 : List Nat := [2383]
set_option maxRecDepth 8192 in
theorem root114Candidates0_checked : candidates edge 4 643 751 root114Tail1 winner.bound = root114Candidates0 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
theorem root114Step0 : scan edge (search edge 3) 4 [643] 643 root114Tail0 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail1 winner := by
  rw [root114Tail0, scan, if_neg (by decide), root114Candidates0_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail1
    (if 3 ≤ root114Candidates0.length then search edge 3 [643, 751] root114Candidates0 1394 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates1 : List Nat := [2017, 15739]
set_option maxRecDepth 8192 in
theorem root114Candidates1_checked : candidates edge 4 643 991 root114Tail2 winner.bound = root114Candidates1 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
theorem root114Step1 : scan edge (search edge 3) 4 [643] 643 root114Tail1 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail2 winner := by
  rw [root114Tail1, scan, if_neg (by decide), root114Candidates1_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail2
    (if 3 ≤ root114Candidates1.length then search edge 3 [643, 991] root114Candidates1 1634 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates2 : List Nat := [2053]
set_option maxRecDepth 8192 in
theorem root114Candidates2_checked : candidates edge 4 643 1213 root114Tail3 winner.bound = root114Candidates2 := by
  unfold candidates
  have row : edge 1213 = (SortedNatTree.contains FastRows.row1213) := by
    change Data.row1213.contains = (SortedNatTree.contains FastRows.row1213)
    exact FastRows.row1213_checked.symm
  rw [row]
  decide
theorem root114Step2 : scan edge (search edge 3) 4 [643] 643 root114Tail2 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail3 winner := by
  rw [root114Tail2, scan, if_neg (by decide), root114Candidates2_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail3
    (if 3 ≤ root114Candidates2.length then search edge 3 [643, 1213] root114Candidates2 1856 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates3_checked : candidates edge 4 643 1993 root114Tail4 winner.bound = root114Candidates3 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
theorem root114Step3 : scan edge (search edge 3) 4 [643] 643 root114Tail3 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail4 winner := by
  rw [root114Tail3, scan, if_neg (by decide), root114Candidates3_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail4
    (if 3 ≤ root114Candidates3.length then search edge 3 [643, 1993] root114Candidates3 2636 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates4_checked : candidates edge 4 643 2017 root114Tail5 winner.bound = root114Candidates4 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root114Step4 : scan edge (search edge 3) 4 [643] 643 root114Tail4 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail5 winner := by
  rw [root114Tail4, scan, if_neg (by decide), root114Candidates4_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail5
    (if 3 ≤ root114Candidates4.length then search edge 3 [643, 2017] root114Candidates4 2660 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates5 : List Nat := [8923, 12097, 18181, 19069]
set_option maxRecDepth 8192 in
theorem root114Candidates5_checked : candidates edge 4 643 2053 root114Tail6 winner.bound = root114Candidates5 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root114Child5 : search edge 3 [643, 2053] root114Candidates5 2696 winner = winner := by decide
theorem root114Step5 : scan edge (search edge 3) 4 [643] 643 root114Tail5 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail6 winner := by
  rw [root114Tail5, scan, if_neg (by decide), root114Candidates5_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail6
    (if 3 ≤ root114Candidates5.length then search edge 3 [643, 2053] root114Candidates5 2696 winner else winner) = _
  rw [if_pos (by decide), root114Child5]
def root114Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates6_checked : candidates edge 4 643 2383 root114Tail7 winner.bound = root114Candidates6 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root114Step6 : scan edge (search edge 3) 4 [643] 643 root114Tail6 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail7 winner := by
  rw [root114Tail6, scan, if_neg (by decide), root114Candidates6_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail7
    (if 3 ≤ root114Candidates6.length then search edge 3 [643, 2383] root114Candidates6 3026 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates7_checked : candidates edge 4 643 2917 root114Tail8 winner.bound = root114Candidates7 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root114Step7 : scan edge (search edge 3) 4 [643] 643 root114Tail7 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail8 winner := by
  rw [root114Tail7, scan, if_neg (by decide), root114Candidates7_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail8
    (if 3 ≤ root114Candidates7.length then search edge 3 [643, 2917] root114Candidates7 3560 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates8 : List Nat := [5119]
set_option maxRecDepth 8192 in
theorem root114Candidates8_checked : candidates edge 4 643 4111 root114Tail9 winner.bound = root114Candidates8 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root114Step8 : scan edge (search edge 3) 4 [643] 643 root114Tail8 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail9 winner := by
  rw [root114Tail8, scan, if_neg (by decide), root114Candidates8_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail9
    (if 3 ≤ root114Candidates8.length then search edge 3 [643, 4111] root114Candidates8 4754 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates9_checked : candidates edge 4 643 5119 root114Tail10 winner.bound = root114Candidates9 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root114Step9 : scan edge (search edge 3) 4 [643] 643 root114Tail9 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail10 winner := by
  rw [root114Tail9, scan, if_neg (by decide), root114Candidates9_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail10
    (if 3 ≤ root114Candidates9.length then search edge 3 [643, 5119] root114Candidates9 5762 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates10_checked : candidates edge 4 643 5413 root114Tail11 winner.bound = root114Candidates10 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root114Step10 : scan edge (search edge 3) 4 [643] 643 root114Tail10 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail11 winner := by
  rw [root114Tail10, scan, if_neg (by decide), root114Candidates10_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail11
    (if 3 ≤ root114Candidates10.length then search edge 3 [643, 5413] root114Candidates10 6056 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates11 : List Nat := [6037]
set_option maxRecDepth 8192 in
theorem root114Candidates11_checked : candidates edge 4 643 5563 root114Tail12 winner.bound = root114Candidates11 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root114Step11 : scan edge (search edge 3) 4 [643] 643 root114Tail11 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail12 winner := by
  rw [root114Tail11, scan, if_neg (by decide), root114Candidates11_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail12
    (if 3 ≤ root114Candidates11.length then search edge 3 [643, 5563] root114Candidates11 6206 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates12_checked : candidates edge 4 643 6007 root114Tail13 winner.bound = root114Candidates12 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root114Step12 : scan edge (search edge 3) 4 [643] 643 root114Tail12 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail13 winner := by
  rw [root114Tail12, scan, if_neg (by decide), root114Candidates12_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail13
    (if 3 ≤ root114Candidates12.length then search edge 3 [643, 6007] root114Candidates12 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root114Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root114Candidates13_checked : candidates edge 4 643 6037 root114Tail14 winner.bound = root114Candidates13 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root114Step13 : scan edge (search edge 3) 4 [643] 643 root114Tail13 winner =
    scan edge (search edge 3) 4 [643] 643 root114Tail14 winner := by
  rw [root114Tail13, scan, if_neg (by decide), root114Candidates13_checked]
  change scan edge (search edge 3) 4 [643] 643 root114Tail14
    (if 3 ≤ root114Candidates13.length then search edge 3 [643, 6037] root114Candidates13 6680 winner else winner) = _
  rw [if_neg (by decide)]
theorem root114Scan14 : scan edge (search edge 3) 4 [643] 643 root114Tail14 winner = winner := by rw [root114Tail14, scan, if_pos (by decide)]
theorem root114Scan13 : scan edge (search edge 3) 4 [643] 643 root114Tail13 winner = winner :=
  (root114Step13).trans root114Scan14
theorem root114Scan12 : scan edge (search edge 3) 4 [643] 643 root114Tail12 winner = winner :=
  (root114Step12).trans root114Scan13
theorem root114Scan11 : scan edge (search edge 3) 4 [643] 643 root114Tail11 winner = winner :=
  (root114Step11).trans root114Scan12
theorem root114Scan10 : scan edge (search edge 3) 4 [643] 643 root114Tail10 winner = winner :=
  (root114Step10).trans root114Scan11
theorem root114Scan9 : scan edge (search edge 3) 4 [643] 643 root114Tail9 winner = winner :=
  (root114Step9).trans root114Scan10
theorem root114Scan8 : scan edge (search edge 3) 4 [643] 643 root114Tail8 winner = winner :=
  (root114Step8).trans root114Scan9
theorem root114Scan7 : scan edge (search edge 3) 4 [643] 643 root114Tail7 winner = winner :=
  (root114Step7).trans root114Scan8
theorem root114Scan6 : scan edge (search edge 3) 4 [643] 643 root114Tail6 winner = winner :=
  (root114Step6).trans root114Scan7
theorem root114Scan5 : scan edge (search edge 3) 4 [643] 643 root114Tail5 winner = winner :=
  (root114Step5).trans root114Scan6
theorem root114Scan4 : scan edge (search edge 3) 4 [643] 643 root114Tail4 winner = winner :=
  (root114Step4).trans root114Scan5
theorem root114Scan3 : scan edge (search edge 3) 4 [643] 643 root114Tail3 winner = winner :=
  (root114Step3).trans root114Scan4
theorem root114Scan2 : scan edge (search edge 3) 4 [643] 643 root114Tail2 winner = winner :=
  (root114Step2).trans root114Scan3
theorem root114Scan1 : scan edge (search edge 3) 4 [643] 643 root114Tail1 winner = winner :=
  (root114Step1).trans root114Scan2
theorem root114Scan0 : scan edge (search edge 3) 4 [643] 643 root114Tail0 winner = winner :=
  (root114Step0).trans root114Scan1
set_option maxRecDepth 4096 in
theorem root114Branch : search edge 4 [643] root114Candidates 643 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [643] 643 root114Tail0 winner = winner
  exact root114Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
