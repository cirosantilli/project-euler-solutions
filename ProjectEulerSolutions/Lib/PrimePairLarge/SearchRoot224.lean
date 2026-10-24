import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root224Candidates : List Nat := Data.row1433
set_option maxRecDepth 8192 in
theorem root224Candidates_checked : candidates edge 5 0 1433 tail225 winner.bound = root224Candidates := by
  apply candidates_eq_row edge 5 0 1433 tail225 root224Candidates winner.bound tail225_ordered
  · decide
  · decide
  · funext q
    rfl
def root224Tail49 : List Nat := []
def root224Tail48 : List Nat := 24473 :: root224Tail49
def root224Tail47 : List Nat := 24359 :: root224Tail48
def root224Tail46 : List Nat := 24029 :: root224Tail47
def root224Tail45 : List Nat := 23561 :: root224Tail46
def root224Tail44 : List Nat := 22709 :: root224Tail45
def root224Tail43 : List Nat := 21863 :: root224Tail44
def root224Tail42 : List Nat := 21341 :: root224Tail43
def root224Tail41 : List Nat := 20147 :: root224Tail42
def root224Tail40 : List Nat := 19949 :: root224Tail41
def root224Tail39 : List Nat := 19853 :: root224Tail40
def root224Tail38 : List Nat := 19427 :: root224Tail39
def root224Tail37 : List Nat := 18803 :: root224Tail38
def root224Tail36 : List Nat := 17981 :: root224Tail37
def root224Tail35 : List Nat := 17957 :: root224Tail36
def root224Tail34 : List Nat := 17579 :: root224Tail35
def root224Tail33 : List Nat := 16091 :: root224Tail34
def root224Tail32 : List Nat := 15773 :: root224Tail33
def root224Tail31 : List Nat := 15329 :: root224Tail32
def root224Tail30 : List Nat := 15017 :: root224Tail31
def root224Tail29 : List Nat := 14621 :: root224Tail30
def root224Tail28 : List Nat := 14447 :: root224Tail29
def root224Tail27 : List Nat := 14177 :: root224Tail28
def root224Tail26 : List Nat := 13187 :: root224Tail27
def root224Tail25 : List Nat := 12893 :: root224Tail26
def root224Tail24 : List Nat := 12641 :: root224Tail25
def root224Tail23 : List Nat := 12503 :: root224Tail24
def root224Tail22 : List Nat := 11969 :: root224Tail23
def root224Tail21 : List Nat := 9929 :: root224Tail22
def root224Tail20 : List Nat := 8693 :: root224Tail21
def root224Tail19 : List Nat := 8297 :: root224Tail20
def root224Tail18 : List Nat := 7937 :: root224Tail19
def root224Tail17 : List Nat := 7793 :: root224Tail18
def root224Tail16 : List Nat := 7607 :: root224Tail17
def root224Tail15 : List Nat := 6761 :: root224Tail16
def root224Tail14 : List Nat := 6737 :: root224Tail15
def root224Tail13 : List Nat := 6299 :: root224Tail14
def root224Tail12 : List Nat := 6029 :: root224Tail13
def root224Tail11 : List Nat := 5903 :: root224Tail12
def root224Tail10 : List Nat := 5501 :: root224Tail11
def root224Tail9 : List Nat := 5393 :: root224Tail10
def root224Tail8 : List Nat := 4643 :: root224Tail9
def root224Tail7 : List Nat := 4637 :: root224Tail8
def root224Tail6 : List Nat := 4283 :: root224Tail7
def root224Tail5 : List Nat := 4157 :: root224Tail6
def root224Tail4 : List Nat := 3323 :: root224Tail5
def root224Tail3 : List Nat := 3089 :: root224Tail4
def root224Tail2 : List Nat := 2999 :: root224Tail3
def root224Tail1 : List Nat := 2957 :: root224Tail2
def root224Tail0 : List Nat := 2663 :: root224Tail1
def root224Candidates0 : List Nat := [7607, 21863]
set_option maxRecDepth 8192 in
theorem root224Candidates0_checked : candidates edge 4 1433 2663 root224Tail1 winner.bound = root224Candidates0 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root224Step0 : scan edge (search edge 3) 4 [1433] 1433 root224Tail0 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail1 winner := by
  rw [root224Tail0, scan, if_neg (by decide), root224Candidates0_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail1
    (if 3 ≤ root224Candidates0.length then search edge 3 [1433, 2663] root224Candidates0 4096 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates1_checked : candidates edge 4 1433 2957 root224Tail2 winner.bound = root224Candidates1 := by
  unfold candidates
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row]
  decide
theorem root224Step1 : scan edge (search edge 3) 4 [1433] 1433 root224Tail1 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail2 winner := by
  rw [root224Tail1, scan, if_neg (by decide), root224Candidates1_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail2
    (if 3 ≤ root224Candidates1.length then search edge 3 [1433, 2957] root224Candidates1 4390 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates2_checked : candidates edge 4 1433 2999 root224Tail3 winner.bound = root224Candidates2 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root224Step2 : scan edge (search edge 3) 4 [1433] 1433 root224Tail2 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail3 winner := by
  rw [root224Tail2, scan, if_neg (by decide), root224Candidates2_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail3
    (if 3 ≤ root224Candidates2.length then search edge 3 [1433, 2999] root224Candidates2 4432 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates3 : List Nat := [18803]
set_option maxRecDepth 8192 in
theorem root224Candidates3_checked : candidates edge 4 1433 3089 root224Tail4 winner.bound = root224Candidates3 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root224Step3 : scan edge (search edge 3) 4 [1433] 1433 root224Tail3 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail4 winner := by
  rw [root224Tail3, scan, if_neg (by decide), root224Candidates3_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail4
    (if 3 ≤ root224Candidates3.length then search edge 3 [1433, 3089] root224Candidates3 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates4 : List Nat := [5393]
set_option maxRecDepth 8192 in
theorem root224Candidates4_checked : candidates edge 4 1433 3323 root224Tail5 winner.bound = root224Candidates4 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
theorem root224Step4 : scan edge (search edge 3) 4 [1433] 1433 root224Tail4 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail5 winner := by
  rw [root224Tail4, scan, if_neg (by decide), root224Candidates4_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail5
    (if 3 ≤ root224Candidates4.length then search edge 3 [1433, 3323] root224Candidates4 4756 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates5_checked : candidates edge 4 1433 4157 root224Tail6 winner.bound = root224Candidates5 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root224Step5 : scan edge (search edge 3) 4 [1433] 1433 root224Tail5 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail6 winner := by
  rw [root224Tail5, scan, if_neg (by decide), root224Candidates5_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail6
    (if 3 ≤ root224Candidates5.length then search edge 3 [1433, 4157] root224Candidates5 5590 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates6_checked : candidates edge 4 1433 4283 root224Tail7 winner.bound = root224Candidates6 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root224Step6 : scan edge (search edge 3) 4 [1433] 1433 root224Tail6 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail7 winner := by
  rw [root224Tail6, scan, if_neg (by decide), root224Candidates6_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail7
    (if 3 ≤ root224Candidates6.length then search edge 3 [1433, 4283] root224Candidates6 5716 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates7_checked : candidates edge 4 1433 4637 root224Tail8 winner.bound = root224Candidates7 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root224Step7 : scan edge (search edge 3) 4 [1433] 1433 root224Tail7 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail8 winner := by
  rw [root224Tail7, scan, if_neg (by decide), root224Candidates7_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail8
    (if 3 ≤ root224Candidates7.length then search edge 3 [1433, 4637] root224Candidates7 6070 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates8_checked : candidates edge 4 1433 4643 root224Tail9 winner.bound = root224Candidates8 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root224Step8 : scan edge (search edge 3) 4 [1433] 1433 root224Tail8 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail9 winner := by
  rw [root224Tail8, scan, if_neg (by decide), root224Candidates8_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail9
    (if 3 ≤ root224Candidates8.length then search edge 3 [1433, 4643] root224Candidates8 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates9 : List Nat := [7937]
set_option maxRecDepth 8192 in
theorem root224Candidates9_checked : candidates edge 4 1433 5393 root224Tail10 winner.bound = root224Candidates9 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root224Step9 : scan edge (search edge 3) 4 [1433] 1433 root224Tail9 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail10 winner := by
  rw [root224Tail9, scan, if_neg (by decide), root224Candidates9_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail10
    (if 3 ≤ root224Candidates9.length then search edge 3 [1433, 5393] root224Candidates9 6826 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root224Candidates10_checked : candidates edge 4 1433 5501 root224Tail11 winner.bound = root224Candidates10 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root224Step10 : scan edge (search edge 3) 4 [1433] 1433 root224Tail10 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail11 winner := by
  rw [root224Tail10, scan, if_neg (by decide), root224Candidates10_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail11
    (if 3 ≤ root224Candidates10.length then search edge 3 [1433, 5501] root224Candidates10 6934 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates11 : List Nat := [17579]
set_option maxRecDepth 8192 in
theorem root224Candidates11_checked : candidates edge 4 1433 5903 root224Tail12 winner.bound = root224Candidates11 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root224Step11 : scan edge (search edge 3) 4 [1433] 1433 root224Tail11 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail12 winner := by
  rw [root224Tail11, scan, if_neg (by decide), root224Candidates11_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail12
    (if 3 ≤ root224Candidates11.length then search edge 3 [1433, 5903] root224Candidates11 7336 winner else winner) = _
  rw [if_neg (by decide)]
def root224Candidates12 : List Nat := [7607]
set_option maxRecDepth 8192 in
theorem root224Candidates12_checked : candidates edge 4 1433 6029 root224Tail13 winner.bound = root224Candidates12 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root224Step12 : scan edge (search edge 3) 4 [1433] 1433 root224Tail12 winner =
    scan edge (search edge 3) 4 [1433] 1433 root224Tail13 winner := by
  rw [root224Tail12, scan, if_neg (by decide), root224Candidates12_checked]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail13
    (if 3 ≤ root224Candidates12.length then search edge 3 [1433, 6029] root224Candidates12 7462 winner else winner) = _
  rw [if_neg (by decide)]
theorem root224Scan13 : scan edge (search edge 3) 4 [1433] 1433 root224Tail13 winner = winner := by rw [root224Tail13, scan, if_pos (by decide)]
theorem root224Scan12 : scan edge (search edge 3) 4 [1433] 1433 root224Tail12 winner = winner :=
  (root224Step12).trans root224Scan13
theorem root224Scan11 : scan edge (search edge 3) 4 [1433] 1433 root224Tail11 winner = winner :=
  (root224Step11).trans root224Scan12
theorem root224Scan10 : scan edge (search edge 3) 4 [1433] 1433 root224Tail10 winner = winner :=
  (root224Step10).trans root224Scan11
theorem root224Scan9 : scan edge (search edge 3) 4 [1433] 1433 root224Tail9 winner = winner :=
  (root224Step9).trans root224Scan10
theorem root224Scan8 : scan edge (search edge 3) 4 [1433] 1433 root224Tail8 winner = winner :=
  (root224Step8).trans root224Scan9
theorem root224Scan7 : scan edge (search edge 3) 4 [1433] 1433 root224Tail7 winner = winner :=
  (root224Step7).trans root224Scan8
theorem root224Scan6 : scan edge (search edge 3) 4 [1433] 1433 root224Tail6 winner = winner :=
  (root224Step6).trans root224Scan7
theorem root224Scan5 : scan edge (search edge 3) 4 [1433] 1433 root224Tail5 winner = winner :=
  (root224Step5).trans root224Scan6
theorem root224Scan4 : scan edge (search edge 3) 4 [1433] 1433 root224Tail4 winner = winner :=
  (root224Step4).trans root224Scan5
theorem root224Scan3 : scan edge (search edge 3) 4 [1433] 1433 root224Tail3 winner = winner :=
  (root224Step3).trans root224Scan4
theorem root224Scan2 : scan edge (search edge 3) 4 [1433] 1433 root224Tail2 winner = winner :=
  (root224Step2).trans root224Scan3
theorem root224Scan1 : scan edge (search edge 3) 4 [1433] 1433 root224Tail1 winner = winner :=
  (root224Step1).trans root224Scan2
theorem root224Scan0 : scan edge (search edge 3) 4 [1433] 1433 root224Tail0 winner = winner :=
  (root224Step0).trans root224Scan1
set_option maxRecDepth 4096 in
theorem root224Branch : search edge 4 [1433] root224Candidates 1433 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1433] 1433 root224Tail0 winner = winner
  exact root224Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
