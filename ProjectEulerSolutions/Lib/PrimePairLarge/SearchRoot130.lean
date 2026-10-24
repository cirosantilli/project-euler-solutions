import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root130Candidates : List Nat := Data.row751
set_option maxRecDepth 8192 in
theorem root130Candidates_checked : candidates edge 5 0 751 tail131 winner.bound = root130Candidates := by
  apply candidates_eq_row edge 5 0 751 tail131 root130Candidates winner.bound tail131_ordered
  · decide
  · decide
  · funext q
    rfl
def root130Tail56 : List Nat := []
def root130Tail55 : List Nat := 24979 :: root130Tail56
def root130Tail54 : List Nat := 24391 :: root130Tail55
def root130Tail53 : List Nat := 24223 :: root130Tail54
def root130Tail52 : List Nat := 23857 :: root130Tail53
def root130Tail51 : List Nat := 23173 :: root130Tail52
def root130Tail50 : List Nat := 22051 :: root130Tail51
def root130Tail49 : List Nat := 22039 :: root130Tail50
def root130Tail48 : List Nat := 21481 :: root130Tail49
def root130Tail47 : List Nat := 19543 :: root130Tail48
def root130Tail46 : List Nat := 19501 :: root130Tail47
def root130Tail45 : List Nat := 19477 :: root130Tail46
def root130Tail44 : List Nat := 19423 :: root130Tail45
def root130Tail43 : List Nat := 19249 :: root130Tail44
def root130Tail42 : List Nat := 18553 :: root130Tail43
def root130Tail41 : List Nat := 18481 :: root130Tail42
def root130Tail40 : List Nat := 18451 :: root130Tail41
def root130Tail39 : List Nat := 18397 :: root130Tail40
def root130Tail38 : List Nat := 17737 :: root130Tail39
def root130Tail37 : List Nat := 17491 :: root130Tail38
def root130Tail36 : List Nat := 17401 :: root130Tail37
def root130Tail35 : List Nat := 17359 :: root130Tail36
def root130Tail34 : List Nat := 16759 :: root130Tail35
def root130Tail33 : List Nat := 16231 :: root130Tail34
def root130Tail32 : List Nat := 16057 :: root130Tail33
def root130Tail31 : List Nat := 15919 :: root130Tail32
def root130Tail30 : List Nat := 15331 :: root130Tail31
def root130Tail29 : List Nat := 15091 :: root130Tail30
def root130Tail28 : List Nat := 14887 :: root130Tail29
def root130Tail27 : List Nat := 13693 :: root130Tail28
def root130Tail26 : List Nat := 13627 :: root130Tail27
def root130Tail25 : List Nat := 13513 :: root130Tail26
def root130Tail24 : List Nat := 13417 :: root130Tail25
def root130Tail23 : List Nat := 12577 :: root130Tail24
def root130Tail22 : List Nat := 11689 :: root130Tail23
def root130Tail21 : List Nat := 10531 :: root130Tail22
def root130Tail20 : List Nat := 10141 :: root130Tail21
def root130Tail19 : List Nat := 9901 :: root130Tail20
def root130Tail18 : List Nat := 9817 :: root130Tail19
def root130Tail17 : List Nat := 8893 :: root130Tail18
def root130Tail16 : List Nat := 8431 :: root130Tail17
def root130Tail15 : List Nat := 7669 :: root130Tail16
def root130Tail14 : List Nat := 7591 :: root130Tail15
def root130Tail13 : List Nat := 5281 :: root130Tail14
def root130Tail12 : List Nat := 4597 :: root130Tail13
def root130Tail11 : List Nat := 4231 :: root130Tail12
def root130Tail10 : List Nat := 3547 :: root130Tail11
def root130Tail9 : List Nat := 3361 :: root130Tail10
def root130Tail8 : List Nat := 3343 :: root130Tail9
def root130Tail7 : List Nat := 2671 :: root130Tail8
def root130Tail6 : List Nat := 2437 :: root130Tail7
def root130Tail5 : List Nat := 2383 :: root130Tail6
def root130Tail4 : List Nat := 2377 :: root130Tail5
def root130Tail3 : List Nat := 1789 :: root130Tail4
def root130Tail2 : List Nat := 1033 :: root130Tail3
def root130Tail1 : List Nat := 997 :: root130Tail2
def root130Tail0 : List Nat := 787 :: root130Tail1
def root130Candidates0 : List Nat := [2671, 12577, 19423, 21481]
set_option maxRecDepth 8192 in
theorem root130Candidates0_checked : candidates edge 4 751 787 root130Tail1 winner.bound = root130Candidates0 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root130Child0 : search edge 3 [751, 787] root130Candidates0 1538 winner = winner := by decide
theorem root130Step0 : scan edge (search edge 3) 4 [751] 751 root130Tail0 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail1 winner := by
  rw [root130Tail0, scan, if_neg (by decide), root130Candidates0_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail1
    (if 3 ≤ root130Candidates0.length then search edge 3 [751, 787] root130Candidates0 1538 winner else winner) = _
  rw [if_pos (by decide), root130Child0]
def root130Candidates1 : List Nat := [2377, 12577, 15331]
set_option maxRecDepth 8192 in
theorem root130Candidates1_checked : candidates edge 4 751 997 root130Tail2 winner.bound = root130Candidates1 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root130Child1 : search edge 3 [751, 997] root130Candidates1 1748 winner = winner := by decide
theorem root130Step1 : scan edge (search edge 3) 4 [751] 751 root130Tail1 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail2 winner := by
  rw [root130Tail1, scan, if_neg (by decide), root130Candidates1_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail2
    (if 3 ≤ root130Candidates1.length then search edge 3 [751, 997] root130Candidates1 1748 winner else winner) = _
  rw [if_pos (by decide), root130Child1]
def root130Candidates2 : List Nat := [4231, 16231, 23173]
set_option maxRecDepth 8192 in
theorem root130Candidates2_checked : candidates edge 4 751 1033 root130Tail3 winner.bound = root130Candidates2 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root130Child2 : search edge 3 [751, 1033] root130Candidates2 1784 winner = winner := by decide
theorem root130Step2 : scan edge (search edge 3) 4 [751] 751 root130Tail2 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail3 winner := by
  rw [root130Tail2, scan, if_neg (by decide), root130Candidates2_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail3
    (if 3 ≤ root130Candidates2.length then search edge 3 [751, 1033] root130Candidates2 1784 winner else winner) = _
  rw [if_pos (by decide), root130Child2]
def root130Candidates3 : List Nat := [2377]
set_option maxRecDepth 8192 in
theorem root130Candidates3_checked : candidates edge 4 751 1789 root130Tail4 winner.bound = root130Candidates3 := by
  unfold candidates
  have row : edge 1789 = (SortedNatTree.contains FastRows.row1789) := by
    change Data.row1789.contains = (SortedNatTree.contains FastRows.row1789)
    exact FastRows.row1789_checked.symm
  rw [row]
  decide
theorem root130Step3 : scan edge (search edge 3) 4 [751] 751 root130Tail3 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail4 winner := by
  rw [root130Tail3, scan, if_neg (by decide), root130Candidates3_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail4
    (if 3 ≤ root130Candidates3.length then search edge 3 [751, 1789] root130Candidates3 2540 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates4 : List Nat := [13627, 19477]
set_option maxRecDepth 8192 in
theorem root130Candidates4_checked : candidates edge 4 751 2377 root130Tail5 winner.bound = root130Candidates4 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
theorem root130Step4 : scan edge (search edge 3) 4 [751] 751 root130Tail4 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail5 winner := by
  rw [root130Tail4, scan, if_neg (by decide), root130Candidates4_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail5
    (if 3 ≤ root130Candidates4.length then search edge 3 [751, 2377] root130Candidates4 3128 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates5 : List Nat := [8431]
set_option maxRecDepth 8192 in
theorem root130Candidates5_checked : candidates edge 4 751 2383 root130Tail6 winner.bound = root130Candidates5 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root130Step5 : scan edge (search edge 3) 4 [751] 751 root130Tail5 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail6 winner := by
  rw [root130Tail5, scan, if_neg (by decide), root130Candidates5_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail6
    (if 3 ≤ root130Candidates5.length then search edge 3 [751, 2383] root130Candidates5 3134 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates6 : List Nat := [3361, 18451]
set_option maxRecDepth 8192 in
theorem root130Candidates6_checked : candidates edge 4 751 2437 root130Tail7 winner.bound = root130Candidates6 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root130Step6 : scan edge (search edge 3) 4 [751] 751 root130Tail6 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail7 winner := by
  rw [root130Tail6, scan, if_neg (by decide), root130Candidates6_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail7
    (if 3 ≤ root130Candidates6.length then search edge 3 [751, 2437] root130Candidates6 3188 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates7 : List Nat := [3361, 4231]
set_option maxRecDepth 8192 in
theorem root130Candidates7_checked : candidates edge 4 751 2671 root130Tail8 winner.bound = root130Candidates7 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root130Step7 : scan edge (search edge 3) 4 [751] 751 root130Tail7 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail8 winner := by
  rw [root130Tail7, scan, if_neg (by decide), root130Candidates7_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail8
    (if 3 ≤ root130Candidates7.length then search edge 3 [751, 2671] root130Candidates7 3422 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates8 : List Nat := [19477]
set_option maxRecDepth 8192 in
theorem root130Candidates8_checked : candidates edge 4 751 3343 root130Tail9 winner.bound = root130Candidates8 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root130Step8 : scan edge (search edge 3) 4 [751] 751 root130Tail8 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail9 winner := by
  rw [root130Tail8, scan, if_neg (by decide), root130Candidates8_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail9
    (if 3 ≤ root130Candidates8.length then search edge 3 [751, 3343] root130Candidates8 4094 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root130Candidates9_checked : candidates edge 4 751 3361 root130Tail10 winner.bound = root130Candidates9 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root130Step9 : scan edge (search edge 3) 4 [751] 751 root130Tail9 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail10 winner := by
  rw [root130Tail9, scan, if_neg (by decide), root130Candidates9_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail10
    (if 3 ≤ root130Candidates9.length then search edge 3 [751, 3361] root130Candidates9 4112 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates10 : List Nat := [8431, 9817, 18451, 19423]
set_option maxRecDepth 8192 in
theorem root130Candidates10_checked : candidates edge 4 751 3547 root130Tail11 winner.bound = root130Candidates10 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root130Child10 : search edge 3 [751, 3547] root130Candidates10 4298 winner = winner := by decide
theorem root130Step10 : scan edge (search edge 3) 4 [751] 751 root130Tail10 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail11 winner := by
  rw [root130Tail10, scan, if_neg (by decide), root130Candidates10_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail11
    (if 3 ≤ root130Candidates10.length then search edge 3 [751, 3547] root130Candidates10 4298 winner else winner) = _
  rw [if_pos (by decide), root130Child10]
def root130Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root130Candidates11_checked : candidates edge 4 751 4231 root130Tail12 winner.bound = root130Candidates11 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root130Step11 : scan edge (search edge 3) 4 [751] 751 root130Tail11 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail12 winner := by
  rw [root130Tail11, scan, if_neg (by decide), root130Candidates11_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail12
    (if 3 ≤ root130Candidates11.length then search edge 3 [751, 4231] root130Candidates11 4982 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates12 : List Nat := [15331]
set_option maxRecDepth 8192 in
theorem root130Candidates12_checked : candidates edge 4 751 4597 root130Tail13 winner.bound = root130Candidates12 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root130Step12 : scan edge (search edge 3) 4 [751] 751 root130Tail12 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail13 winner := by
  rw [root130Tail12, scan, if_neg (by decide), root130Candidates12_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail13
    (if 3 ≤ root130Candidates12.length then search edge 3 [751, 4597] root130Candidates12 5348 winner else winner) = _
  rw [if_neg (by decide)]
def root130Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root130Candidates13_checked : candidates edge 4 751 5281 root130Tail14 winner.bound = root130Candidates13 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root130Step13 : scan edge (search edge 3) 4 [751] 751 root130Tail13 winner =
    scan edge (search edge 3) 4 [751] 751 root130Tail14 winner := by
  rw [root130Tail13, scan, if_neg (by decide), root130Candidates13_checked]
  change scan edge (search edge 3) 4 [751] 751 root130Tail14
    (if 3 ≤ root130Candidates13.length then search edge 3 [751, 5281] root130Candidates13 6032 winner else winner) = _
  rw [if_neg (by decide)]
theorem root130Scan14 : scan edge (search edge 3) 4 [751] 751 root130Tail14 winner = winner := by rw [root130Tail14, scan, if_pos (by decide)]
theorem root130Scan13 : scan edge (search edge 3) 4 [751] 751 root130Tail13 winner = winner :=
  (root130Step13).trans root130Scan14
theorem root130Scan12 : scan edge (search edge 3) 4 [751] 751 root130Tail12 winner = winner :=
  (root130Step12).trans root130Scan13
theorem root130Scan11 : scan edge (search edge 3) 4 [751] 751 root130Tail11 winner = winner :=
  (root130Step11).trans root130Scan12
theorem root130Scan10 : scan edge (search edge 3) 4 [751] 751 root130Tail10 winner = winner :=
  (root130Step10).trans root130Scan11
theorem root130Scan9 : scan edge (search edge 3) 4 [751] 751 root130Tail9 winner = winner :=
  (root130Step9).trans root130Scan10
theorem root130Scan8 : scan edge (search edge 3) 4 [751] 751 root130Tail8 winner = winner :=
  (root130Step8).trans root130Scan9
theorem root130Scan7 : scan edge (search edge 3) 4 [751] 751 root130Tail7 winner = winner :=
  (root130Step7).trans root130Scan8
theorem root130Scan6 : scan edge (search edge 3) 4 [751] 751 root130Tail6 winner = winner :=
  (root130Step6).trans root130Scan7
theorem root130Scan5 : scan edge (search edge 3) 4 [751] 751 root130Tail5 winner = winner :=
  (root130Step5).trans root130Scan6
theorem root130Scan4 : scan edge (search edge 3) 4 [751] 751 root130Tail4 winner = winner :=
  (root130Step4).trans root130Scan5
theorem root130Scan3 : scan edge (search edge 3) 4 [751] 751 root130Tail3 winner = winner :=
  (root130Step3).trans root130Scan4
theorem root130Scan2 : scan edge (search edge 3) 4 [751] 751 root130Tail2 winner = winner :=
  (root130Step2).trans root130Scan3
theorem root130Scan1 : scan edge (search edge 3) 4 [751] 751 root130Tail1 winner = winner :=
  (root130Step1).trans root130Scan2
theorem root130Scan0 : scan edge (search edge 3) 4 [751] 751 root130Tail0 winner = winner :=
  (root130Step0).trans root130Scan1
set_option maxRecDepth 4096 in
theorem root130Branch : search edge 4 [751] root130Candidates 751 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [751] 751 root130Tail0 winner = winner
  exact root130Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
