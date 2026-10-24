import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root205Candidates : List Nat := Data.row1283
set_option maxRecDepth 8192 in
theorem root205Candidates_checked : candidates edge 5 0 1283 tail206 winner.bound = root205Candidates := by
  apply candidates_eq_row edge 5 0 1283 tail206 root205Candidates winner.bound tail206_ordered
  · decide
  · decide
  · funext q
    rfl
def root205Tail63 : List Nat := []
def root205Tail62 : List Nat := 22871 :: root205Tail63
def root205Tail61 : List Nat := 22541 :: root205Tail62
def root205Tail60 : List Nat := 22157 :: root205Tail61
def root205Tail59 : List Nat := 22133 :: root205Tail60
def root205Tail58 : List Nat := 21107 :: root205Tail59
def root205Tail57 : List Nat := 20903 :: root205Tail58
def root205Tail56 : List Nat := 20441 :: root205Tail57
def root205Tail55 : List Nat := 20021 :: root205Tail56
def root205Tail54 : List Nat := 19739 :: root205Tail55
def root205Tail53 : List Nat := 19427 :: root205Tail54
def root205Tail52 : List Nat := 18731 :: root205Tail53
def root205Tail51 : List Nat := 18461 :: root205Tail52
def root205Tail50 : List Nat := 17789 :: root205Tail51
def root205Tail49 : List Nat := 16673 :: root205Tail50
def root205Tail48 : List Nat := 16451 :: root205Tail49
def root205Tail47 : List Nat := 16193 :: root205Tail48
def root205Tail46 : List Nat := 15101 :: root205Tail47
def root205Tail45 : List Nat := 15017 :: root205Tail46
def root205Tail44 : List Nat := 14867 :: root205Tail45
def root205Tail43 : List Nat := 14561 :: root205Tail44
def root205Tail42 : List Nat := 14411 :: root205Tail43
def root205Tail41 : List Nat := 13217 :: root205Tail42
def root205Tail40 : List Nat := 12983 :: root205Tail41
def root205Tail39 : List Nat := 12941 :: root205Tail40
def root205Tail38 : List Nat := 12641 :: root205Tail39
def root205Tail37 : List Nat := 12227 :: root205Tail38
def root205Tail36 : List Nat := 11933 :: root205Tail37
def root205Tail35 : List Nat := 11789 :: root205Tail36
def root205Tail34 : List Nat := 11657 :: root205Tail35
def root205Tail33 : List Nat := 10733 :: root205Tail34
def root205Tail32 : List Nat := 10313 :: root205Tail33
def root205Tail31 : List Nat := 9689 :: root205Tail32
def root205Tail30 : List Nat := 9419 :: root205Tail31
def root205Tail29 : List Nat := 9371 :: root205Tail30
def root205Tail28 : List Nat := 9137 :: root205Tail29
def root205Tail27 : List Nat := 8951 :: root205Tail28
def root205Tail26 : List Nat := 8747 :: root205Tail27
def root205Tail25 : List Nat := 8537 :: root205Tail26
def root205Tail24 : List Nat := 8513 :: root205Tail25
def root205Tail23 : List Nat := 8237 :: root205Tail24
def root205Tail22 : List Nat := 7949 :: root205Tail23
def root205Tail21 : List Nat := 7757 :: root205Tail22
def root205Tail20 : List Nat := 7247 :: root205Tail21
def root205Tail19 : List Nat := 7001 :: root205Tail20
def root205Tail18 : List Nat := 6869 :: root205Tail19
def root205Tail17 : List Nat := 6827 :: root205Tail18
def root205Tail16 : List Nat := 6287 :: root205Tail17
def root205Tail15 : List Nat := 6221 :: root205Tail16
def root205Tail14 : List Nat := 5807 :: root205Tail15
def root205Tail13 : List Nat := 5441 :: root205Tail14
def root205Tail12 : List Nat := 5021 :: root205Tail13
def root205Tail11 : List Nat := 4397 :: root205Tail12
def root205Tail10 : List Nat := 4127 :: root205Tail11
def root205Tail9 : List Nat := 3719 :: root205Tail10
def root205Tail8 : List Nat := 3617 :: root205Tail9
def root205Tail7 : List Nat := 3593 :: root205Tail8
def root205Tail6 : List Nat := 2801 :: root205Tail7
def root205Tail5 : List Nat := 2741 :: root205Tail6
def root205Tail4 : List Nat := 2423 :: root205Tail5
def root205Tail3 : List Nat := 2111 :: root205Tail4
def root205Tail2 : List Nat := 1913 :: root205Tail3
def root205Tail1 : List Nat := 1667 :: root205Tail2
def root205Tail0 : List Nat := 1619 :: root205Tail1
def root205Candidates0 : List Nat := [4127, 7949, 8513, 8747, 8951, 10733, 16451, 16673, 19739]
set_option maxRecDepth 8192 in
theorem root205Candidates0_checked : candidates edge 4 1283 1619 root205Tail1 winner.bound = root205Candidates0 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root205Child0 : search edge 3 [1283, 1619] root205Candidates0 2902 winner = winner := by decide
theorem root205Step0 : scan edge (search edge 3) 4 [1283] 1283 root205Tail0 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail1 winner := by
  rw [root205Tail0, scan, if_neg (by decide), root205Candidates0_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail1
    (if 3 ≤ root205Candidates0.length then search edge 3 [1283, 1619] root205Candidates0 2902 winner else winner) = _
  rw [if_pos (by decide), root205Child0]
def root205Candidates1 : List Nat := [6287, 9137]
set_option maxRecDepth 8192 in
theorem root205Candidates1_checked : candidates edge 4 1283 1667 root205Tail2 winner.bound = root205Candidates1 := by
  unfold candidates
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row]
  decide
theorem root205Step1 : scan edge (search edge 3) 4 [1283] 1283 root205Tail1 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail2 winner := by
  rw [root205Tail1, scan, if_neg (by decide), root205Candidates1_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail2
    (if 3 ≤ root205Candidates1.length then search edge 3 [1283, 1667] root205Candidates1 2950 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates2 : List Nat := [2741, 11789]
set_option maxRecDepth 8192 in
theorem root205Candidates2_checked : candidates edge 4 1283 1913 root205Tail3 winner.bound = root205Candidates2 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
theorem root205Step2 : scan edge (search edge 3) 4 [1283] 1283 root205Tail2 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail3 winner := by
  rw [root205Tail2, scan, if_neg (by decide), root205Candidates2_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail3
    (if 3 ≤ root205Candidates2.length then search edge 3 [1283, 1913] root205Candidates2 3196 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates3 : List Nat := [4127, 10313, 10733, 21107]
set_option maxRecDepth 8192 in
theorem root205Candidates3_checked : candidates edge 4 1283 2111 root205Tail4 winner.bound = root205Candidates3 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root205Child3 : search edge 3 [1283, 2111] root205Candidates3 3394 winner = winner := by decide
theorem root205Step3 : scan edge (search edge 3) 4 [1283] 1283 root205Tail3 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail4 winner := by
  rw [root205Tail3, scan, if_neg (by decide), root205Candidates3_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail4
    (if 3 ≤ root205Candidates3.length then search edge 3 [1283, 2111] root205Candidates3 3394 winner else winner) = _
  rw [if_pos (by decide), root205Child3]
def root205Candidates4 : List Nat := [5021]
set_option maxRecDepth 8192 in
theorem root205Candidates4_checked : candidates edge 4 1283 2423 root205Tail5 winner.bound = root205Candidates4 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root205Step4 : scan edge (search edge 3) 4 [1283] 1283 root205Tail4 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail5 winner := by
  rw [root205Tail4, scan, if_neg (by decide), root205Candidates4_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail5
    (if 3 ≤ root205Candidates4.length then search edge 3 [1283, 2423] root205Candidates4 3706 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates5 : List Nat := [7757, 8537, 8747, 14411]
set_option maxRecDepth 8192 in
theorem root205Candidates5_checked : candidates edge 4 1283 2741 root205Tail6 winner.bound = root205Candidates5 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root205Child5 : search edge 3 [1283, 2741] root205Candidates5 4024 winner = winner := by decide
theorem root205Step5 : scan edge (search edge 3) 4 [1283] 1283 root205Tail5 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail6 winner := by
  rw [root205Tail5, scan, if_neg (by decide), root205Candidates5_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail6
    (if 3 ≤ root205Candidates5.length then search edge 3 [1283, 2741] root205Candidates5 4024 winner else winner) = _
  rw [if_pos (by decide), root205Child5]
def root205Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root205Candidates6_checked : candidates edge 4 1283 2801 root205Tail7 winner.bound = root205Candidates6 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
theorem root205Step6 : scan edge (search edge 3) 4 [1283] 1283 root205Tail6 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail7 winner := by
  rw [root205Tail6, scan, if_neg (by decide), root205Candidates6_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail7
    (if 3 ≤ root205Candidates6.length then search edge 3 [1283, 2801] root205Candidates6 4084 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates7 : List Nat := [3719, 5021, 18461]
set_option maxRecDepth 8192 in
theorem root205Candidates7_checked : candidates edge 4 1283 3593 root205Tail8 winner.bound = root205Candidates7 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root205Child7 : search edge 3 [1283, 3593] root205Candidates7 4876 winner = winner := by decide
theorem root205Step7 : scan edge (search edge 3) 4 [1283] 1283 root205Tail7 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail8 winner := by
  rw [root205Tail7, scan, if_neg (by decide), root205Candidates7_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail8
    (if 3 ≤ root205Candidates7.length then search edge 3 [1283, 3593] root205Candidates7 4876 winner else winner) = _
  rw [if_pos (by decide), root205Child7]
def root205Candidates8 : List Nat := [10313, 16673]
set_option maxRecDepth 8192 in
theorem root205Candidates8_checked : candidates edge 4 1283 3617 root205Tail9 winner.bound = root205Candidates8 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root205Step8 : scan edge (search edge 3) 4 [1283] 1283 root205Tail8 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail9 winner := by
  rw [root205Tail8, scan, if_neg (by decide), root205Candidates8_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail9
    (if 3 ≤ root205Candidates8.length then search edge 3 [1283, 3617] root205Candidates8 4900 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates9 : List Nat := [15017, 19739]
set_option maxRecDepth 8192 in
theorem root205Candidates9_checked : candidates edge 4 1283 3719 root205Tail10 winner.bound = root205Candidates9 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root205Step9 : scan edge (search edge 3) 4 [1283] 1283 root205Tail9 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail10 winner := by
  rw [root205Tail9, scan, if_neg (by decide), root205Candidates9_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail10
    (if 3 ≤ root205Candidates9.length then search edge 3 [1283, 3719] root205Candidates9 5002 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates10 : List Nat := [7949, 12983]
set_option maxRecDepth 8192 in
theorem root205Candidates10_checked : candidates edge 4 1283 4127 root205Tail11 winner.bound = root205Candidates10 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root205Step10 : scan edge (search edge 3) 4 [1283] 1283 root205Tail10 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail11 winner := by
  rw [root205Tail10, scan, if_neg (by decide), root205Candidates10_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail11
    (if 3 ≤ root205Candidates10.length then search edge 3 [1283, 4127] root205Candidates10 5410 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates11 : List Nat := [8513, 16451]
set_option maxRecDepth 8192 in
theorem root205Candidates11_checked : candidates edge 4 1283 4397 root205Tail12 winner.bound = root205Candidates11 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root205Step11 : scan edge (search edge 3) 4 [1283] 1283 root205Tail11 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail12 winner := by
  rw [root205Tail11, scan, if_neg (by decide), root205Candidates11_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail12
    (if 3 ≤ root205Candidates11.length then search edge 3 [1283, 4397] root205Candidates11 5680 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates12 : List Nat := [10733]
set_option maxRecDepth 8192 in
theorem root205Candidates12_checked : candidates edge 4 1283 5021 root205Tail13 winner.bound = root205Candidates12 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root205Step12 : scan edge (search edge 3) 4 [1283] 1283 root205Tail12 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail13 winner := by
  rw [root205Tail12, scan, if_neg (by decide), root205Candidates12_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail13
    (if 3 ≤ root205Candidates12.length then search edge 3 [1283, 5021] root205Candidates12 6304 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates13 : List Nat := [6287, 11657]
set_option maxRecDepth 8192 in
theorem root205Candidates13_checked : candidates edge 4 1283 5441 root205Tail14 winner.bound = root205Candidates13 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root205Step13 : scan edge (search edge 3) 4 [1283] 1283 root205Tail13 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail14 winner := by
  rw [root205Tail13, scan, if_neg (by decide), root205Candidates13_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail14
    (if 3 ≤ root205Candidates13.length then search edge 3 [1283, 5441] root205Candidates13 6724 winner else winner) = _
  rw [if_neg (by decide)]
def root205Candidates14 : List Nat := [7949, 9689, 15101, 17789]
set_option maxRecDepth 8192 in
theorem root205Candidates14_checked : candidates edge 4 1283 5807 root205Tail15 winner.bound = root205Candidates14 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root205Child14 : search edge 3 [1283, 5807] root205Candidates14 7090 winner = winner := by decide
theorem root205Step14 : scan edge (search edge 3) 4 [1283] 1283 root205Tail14 winner =
    scan edge (search edge 3) 4 [1283] 1283 root205Tail15 winner := by
  rw [root205Tail14, scan, if_neg (by decide), root205Candidates14_checked]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail15
    (if 3 ≤ root205Candidates14.length then search edge 3 [1283, 5807] root205Candidates14 7090 winner else winner) = _
  rw [if_pos (by decide), root205Child14]
theorem root205Scan15 : scan edge (search edge 3) 4 [1283] 1283 root205Tail15 winner = winner := by rw [root205Tail15, scan, if_pos (by decide)]
theorem root205Scan14 : scan edge (search edge 3) 4 [1283] 1283 root205Tail14 winner = winner :=
  (root205Step14).trans root205Scan15
theorem root205Scan13 : scan edge (search edge 3) 4 [1283] 1283 root205Tail13 winner = winner :=
  (root205Step13).trans root205Scan14
theorem root205Scan12 : scan edge (search edge 3) 4 [1283] 1283 root205Tail12 winner = winner :=
  (root205Step12).trans root205Scan13
theorem root205Scan11 : scan edge (search edge 3) 4 [1283] 1283 root205Tail11 winner = winner :=
  (root205Step11).trans root205Scan12
theorem root205Scan10 : scan edge (search edge 3) 4 [1283] 1283 root205Tail10 winner = winner :=
  (root205Step10).trans root205Scan11
theorem root205Scan9 : scan edge (search edge 3) 4 [1283] 1283 root205Tail9 winner = winner :=
  (root205Step9).trans root205Scan10
theorem root205Scan8 : scan edge (search edge 3) 4 [1283] 1283 root205Tail8 winner = winner :=
  (root205Step8).trans root205Scan9
theorem root205Scan7 : scan edge (search edge 3) 4 [1283] 1283 root205Tail7 winner = winner :=
  (root205Step7).trans root205Scan8
theorem root205Scan6 : scan edge (search edge 3) 4 [1283] 1283 root205Tail6 winner = winner :=
  (root205Step6).trans root205Scan7
theorem root205Scan5 : scan edge (search edge 3) 4 [1283] 1283 root205Tail5 winner = winner :=
  (root205Step5).trans root205Scan6
theorem root205Scan4 : scan edge (search edge 3) 4 [1283] 1283 root205Tail4 winner = winner :=
  (root205Step4).trans root205Scan5
theorem root205Scan3 : scan edge (search edge 3) 4 [1283] 1283 root205Tail3 winner = winner :=
  (root205Step3).trans root205Scan4
theorem root205Scan2 : scan edge (search edge 3) 4 [1283] 1283 root205Tail2 winner = winner :=
  (root205Step2).trans root205Scan3
theorem root205Scan1 : scan edge (search edge 3) 4 [1283] 1283 root205Tail1 winner = winner :=
  (root205Step1).trans root205Scan2
theorem root205Scan0 : scan edge (search edge 3) 4 [1283] 1283 root205Tail0 winner = winner :=
  (root205Step0).trans root205Scan1
set_option maxRecDepth 4096 in
theorem root205Branch : search edge 4 [1283] root205Candidates 1283 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1283] 1283 root205Tail0 winner = winner
  exact root205Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
