import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root159Candidates : List Nat := Data.row953
set_option maxRecDepth 8192 in
theorem root159Candidates_checked : candidates edge 5 0 953 tail160 winner.bound = root159Candidates := by
  apply candidates_eq_row edge 5 0 953 tail160 root159Candidates winner.bound tail160_ordered
  · decide
  · decide
  · funext q
    rfl
def root159Tail49 : List Nat := []
def root159Tail48 : List Nat := 24071 :: root159Tail49
def root159Tail47 : List Nat := 23537 :: root159Tail48
def root159Tail46 : List Nat := 23447 :: root159Tail47
def root159Tail45 : List Nat := 22697 :: root159Tail46
def root159Tail44 : List Nat := 22109 :: root159Tail45
def root159Tail43 : List Nat := 21089 :: root159Tail44
def root159Tail42 : List Nat := 20333 :: root159Tail43
def root159Tail41 : List Nat := 20051 :: root159Tail42
def root159Tail40 : List Nat := 19997 :: root159Tail41
def root159Tail39 : List Nat := 19571 :: root159Tail40
def root159Tail38 : List Nat := 19463 :: root159Tail39
def root159Tail37 : List Nat := 19421 :: root159Tail38
def root159Tail36 : List Nat := 19121 :: root159Tail37
def root159Tail35 : List Nat := 18869 :: root159Tail36
def root159Tail34 : List Nat := 17909 :: root159Tail35
def root159Tail33 : List Nat := 17597 :: root159Tail34
def root159Tail32 : List Nat := 16937 :: root159Tail33
def root159Tail31 : List Nat := 16889 :: root159Tail32
def root159Tail30 : List Nat := 15263 :: root159Tail31
def root159Tail29 : List Nat := 14591 :: root159Tail30
def root159Tail28 : List Nat := 13649 :: root159Tail29
def root159Tail27 : List Nat := 13619 :: root159Tail28
def root159Tail26 : List Nat := 11003 :: root159Tail27
def root159Tail25 : List Nat := 10889 :: root159Tail26
def root159Tail24 : List Nat := 10463 :: root159Tail25
def root159Tail23 : List Nat := 10181 :: root159Tail24
def root159Tail22 : List Nat := 10139 :: root159Tail23
def root159Tail21 : List Nat := 9743 :: root159Tail22
def root159Tail20 : List Nat := 9623 :: root159Tail21
def root159Tail19 : List Nat := 9281 :: root159Tail20
def root159Tail18 : List Nat := 9029 :: root159Tail19
def root159Tail17 : List Nat := 8423 :: root159Tail18
def root159Tail16 : List Nat := 7727 :: root159Tail17
def root159Tail15 : List Nat := 7457 :: root159Tail16
def root159Tail14 : List Nat := 7109 :: root159Tail15
def root159Tail13 : List Nat := 6029 :: root159Tail14
def root159Tail12 : List Nat := 5651 :: root159Tail13
def root159Tail11 : List Nat := 5417 :: root159Tail12
def root159Tail10 : List Nat := 5039 :: root159Tail11
def root159Tail9 : List Nat := 4871 :: root159Tail10
def root159Tail8 : List Nat := 4799 :: root159Tail9
def root159Tail7 : List Nat := 4337 :: root159Tail8
def root159Tail6 : List Nat := 3533 :: root159Tail7
def root159Tail5 : List Nat := 2687 :: root159Tail6
def root159Tail4 : List Nat := 2609 :: root159Tail5
def root159Tail3 : List Nat := 2333 :: root159Tail4
def root159Tail2 : List Nat := 2027 :: root159Tail3
def root159Tail1 : List Nat := 1523 :: root159Tail2
def root159Tail0 : List Nat := 1367 :: root159Tail1
def root159Candidates0 : List Nat := [2027]
set_option maxRecDepth 8192 in
theorem root159Candidates0_checked : candidates edge 4 953 1367 root159Tail1 winner.bound = root159Candidates0 := by
  unfold candidates
  have row : edge 1367 = (SortedNatTree.contains FastRows.row1367) := by
    change Data.row1367.contains = (SortedNatTree.contains FastRows.row1367)
    exact FastRows.row1367_checked.symm
  rw [row]
  decide
theorem root159Step0 : scan edge (search edge 3) 4 [953] 953 root159Tail0 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail1 winner := by
  rw [root159Tail0, scan, if_neg (by decide), root159Candidates0_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail1
    (if 3 ≤ root159Candidates0.length then search edge 3 [953, 1367] root159Candidates0 2320 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root159Candidates1_checked : candidates edge 4 953 1523 root159Tail2 winner.bound = root159Candidates1 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
theorem root159Step1 : scan edge (search edge 3) 4 [953] 953 root159Tail1 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail2 winner := by
  rw [root159Tail1, scan, if_neg (by decide), root159Candidates1_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail2
    (if 3 ≤ root159Candidates1.length then search edge 3 [953, 1523] root159Candidates1 2476 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates2 : List Nat := [7457, 10181, 10463, 10889, 15263]
set_option maxRecDepth 8192 in
theorem root159Candidates2_checked : candidates edge 4 953 2027 root159Tail3 winner.bound = root159Candidates2 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root159Child2 : search edge 3 [953, 2027] root159Candidates2 2980 winner = winner := by decide
theorem root159Step2 : scan edge (search edge 3) 4 [953] 953 root159Tail2 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail3 winner := by
  rw [root159Tail2, scan, if_neg (by decide), root159Candidates2_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail3
    (if 3 ≤ root159Candidates2.length then search edge 3 [953, 2027] root159Candidates2 2980 winner else winner) = _
  rw [if_pos (by decide), root159Child2]
def root159Candidates3 : List Nat := [3533]
set_option maxRecDepth 8192 in
theorem root159Candidates3_checked : candidates edge 4 953 2333 root159Tail4 winner.bound = root159Candidates3 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
theorem root159Step3 : scan edge (search edge 3) 4 [953] 953 root159Tail3 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail4 winner := by
  rw [root159Tail3, scan, if_neg (by decide), root159Candidates3_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail4
    (if 3 ≤ root159Candidates3.length then search edge 3 [953, 2333] root159Candidates3 3286 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates4 : List Nat := [5651]
set_option maxRecDepth 8192 in
theorem root159Candidates4_checked : candidates edge 4 953 2609 root159Tail5 winner.bound = root159Candidates4 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
theorem root159Step4 : scan edge (search edge 3) 4 [953] 953 root159Tail4 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail5 winner := by
  rw [root159Tail4, scan, if_neg (by decide), root159Candidates4_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail5
    (if 3 ≤ root159Candidates4.length then search edge 3 [953, 2609] root159Candidates4 3562 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates5 : List Nat := [19421]
set_option maxRecDepth 8192 in
theorem root159Candidates5_checked : candidates edge 4 953 2687 root159Tail6 winner.bound = root159Candidates5 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root159Step5 : scan edge (search edge 3) 4 [953] 953 root159Tail5 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail6 winner := by
  rw [root159Tail5, scan, if_neg (by decide), root159Candidates5_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail6
    (if 3 ≤ root159Candidates5.length then search edge 3 [953, 2687] root159Candidates5 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root159Candidates6_checked : candidates edge 4 953 3533 root159Tail7 winner.bound = root159Candidates6 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root159Step6 : scan edge (search edge 3) 4 [953] 953 root159Tail6 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail7 winner := by
  rw [root159Tail6, scan, if_neg (by decide), root159Candidates6_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail7
    (if 3 ≤ root159Candidates6.length then search edge 3 [953, 3533] root159Candidates6 4486 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root159Candidates7_checked : candidates edge 4 953 4337 root159Tail8 winner.bound = root159Candidates7 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root159Step7 : scan edge (search edge 3) 4 [953] 953 root159Tail7 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail8 winner := by
  rw [root159Tail7, scan, if_neg (by decide), root159Candidates7_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail8
    (if 3 ≤ root159Candidates7.length then search edge 3 [953, 4337] root159Candidates7 5290 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates8 : List Nat := [10181, 19421]
set_option maxRecDepth 8192 in
theorem root159Candidates8_checked : candidates edge 4 953 4799 root159Tail9 winner.bound = root159Candidates8 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root159Step8 : scan edge (search edge 3) 4 [953] 953 root159Tail8 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail9 winner := by
  rw [root159Tail8, scan, if_neg (by decide), root159Candidates8_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail9
    (if 3 ≤ root159Candidates8.length then search edge 3 [953, 4799] root159Candidates8 5752 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates9 : List Nat := [10181, 16937, 19463]
set_option maxRecDepth 8192 in
theorem root159Candidates9_checked : candidates edge 4 953 4871 root159Tail10 winner.bound = root159Candidates9 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root159Child9 : search edge 3 [953, 4871] root159Candidates9 5824 winner = winner := by decide
theorem root159Step9 : scan edge (search edge 3) 4 [953] 953 root159Tail9 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail10 winner := by
  rw [root159Tail9, scan, if_neg (by decide), root159Candidates9_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail10
    (if 3 ≤ root159Candidates9.length then search edge 3 [953, 4871] root159Candidates9 5824 winner else winner) = _
  rw [if_pos (by decide), root159Child9]
def root159Candidates10 : List Nat := [7727, 19421]
set_option maxRecDepth 8192 in
theorem root159Candidates10_checked : candidates edge 4 953 5039 root159Tail11 winner.bound = root159Candidates10 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root159Step10 : scan edge (search edge 3) 4 [953] 953 root159Tail10 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail11 winner := by
  rw [root159Tail10, scan, if_neg (by decide), root159Candidates10_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail11
    (if 3 ≤ root159Candidates10.length then search edge 3 [953, 5039] root159Candidates10 5992 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates11 : List Nat := [9029, 19121]
set_option maxRecDepth 8192 in
theorem root159Candidates11_checked : candidates edge 4 953 5417 root159Tail12 winner.bound = root159Candidates11 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root159Step11 : scan edge (search edge 3) 4 [953] 953 root159Tail11 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail12 winner := by
  rw [root159Tail11, scan, if_neg (by decide), root159Candidates11_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail12
    (if 3 ≤ root159Candidates11.length then search edge 3 [953, 5417] root159Candidates11 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root159Candidates12_checked : candidates edge 4 953 5651 root159Tail13 winner.bound = root159Candidates12 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root159Step12 : scan edge (search edge 3) 4 [953] 953 root159Tail12 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail13 winner := by
  rw [root159Tail12, scan, if_neg (by decide), root159Candidates12_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail13
    (if 3 ≤ root159Candidates12.length then search edge 3 [953, 5651] root159Candidates12 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root159Candidates13 : List Nat := [8423, 9623, 11003]
set_option maxRecDepth 8192 in
theorem root159Candidates13_checked : candidates edge 4 953 6029 root159Tail14 winner.bound = root159Candidates13 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root159Child13 : search edge 3 [953, 6029] root159Candidates13 6982 winner = winner := by decide
theorem root159Step13 : scan edge (search edge 3) 4 [953] 953 root159Tail13 winner =
    scan edge (search edge 3) 4 [953] 953 root159Tail14 winner := by
  rw [root159Tail13, scan, if_neg (by decide), root159Candidates13_checked]
  change scan edge (search edge 3) 4 [953] 953 root159Tail14
    (if 3 ≤ root159Candidates13.length then search edge 3 [953, 6029] root159Candidates13 6982 winner else winner) = _
  rw [if_pos (by decide), root159Child13]
theorem root159Scan14 : scan edge (search edge 3) 4 [953] 953 root159Tail14 winner = winner := by rw [root159Tail14, scan, if_pos (by decide)]
theorem root159Scan13 : scan edge (search edge 3) 4 [953] 953 root159Tail13 winner = winner :=
  (root159Step13).trans root159Scan14
theorem root159Scan12 : scan edge (search edge 3) 4 [953] 953 root159Tail12 winner = winner :=
  (root159Step12).trans root159Scan13
theorem root159Scan11 : scan edge (search edge 3) 4 [953] 953 root159Tail11 winner = winner :=
  (root159Step11).trans root159Scan12
theorem root159Scan10 : scan edge (search edge 3) 4 [953] 953 root159Tail10 winner = winner :=
  (root159Step10).trans root159Scan11
theorem root159Scan9 : scan edge (search edge 3) 4 [953] 953 root159Tail9 winner = winner :=
  (root159Step9).trans root159Scan10
theorem root159Scan8 : scan edge (search edge 3) 4 [953] 953 root159Tail8 winner = winner :=
  (root159Step8).trans root159Scan9
theorem root159Scan7 : scan edge (search edge 3) 4 [953] 953 root159Tail7 winner = winner :=
  (root159Step7).trans root159Scan8
theorem root159Scan6 : scan edge (search edge 3) 4 [953] 953 root159Tail6 winner = winner :=
  (root159Step6).trans root159Scan7
theorem root159Scan5 : scan edge (search edge 3) 4 [953] 953 root159Tail5 winner = winner :=
  (root159Step5).trans root159Scan6
theorem root159Scan4 : scan edge (search edge 3) 4 [953] 953 root159Tail4 winner = winner :=
  (root159Step4).trans root159Scan5
theorem root159Scan3 : scan edge (search edge 3) 4 [953] 953 root159Tail3 winner = winner :=
  (root159Step3).trans root159Scan4
theorem root159Scan2 : scan edge (search edge 3) 4 [953] 953 root159Tail2 winner = winner :=
  (root159Step2).trans root159Scan3
theorem root159Scan1 : scan edge (search edge 3) 4 [953] 953 root159Tail1 winner = winner :=
  (root159Step1).trans root159Scan2
theorem root159Scan0 : scan edge (search edge 3) 4 [953] 953 root159Tail0 winner = winner :=
  (root159Step0).trans root159Scan1
set_option maxRecDepth 4096 in
theorem root159Branch : search edge 4 [953] root159Candidates 953 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [953] 953 root159Tail0 winner = winner
  exact root159Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
