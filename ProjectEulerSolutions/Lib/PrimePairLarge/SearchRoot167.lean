import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root167Candidates : List Nat := Data.row1013
set_option maxRecDepth 8192 in
theorem root167Candidates_checked : candidates edge 5 0 1013 tail168 winner.bound = root167Candidates := by
  apply candidates_eq_row edge 5 0 1013 tail168 root167Candidates winner.bound tail168_ordered
  · decide
  · decide
  · funext q
    rfl
def root167Tail63 : List Nat := []
def root167Tail62 : List Nat := 24677 :: root167Tail63
def root167Tail61 : List Nat := 24593 :: root167Tail62
def root167Tail60 : List Nat := 24533 :: root167Tail61
def root167Tail59 : List Nat := 23549 :: root167Tail60
def root167Tail58 : List Nat := 23291 :: root167Tail59
def root167Tail57 : List Nat := 22721 :: root167Tail58
def root167Tail56 : List Nat := 22709 :: root167Tail57
def root167Tail55 : List Nat := 22643 :: root167Tail56
def root167Tail54 : List Nat := 22469 :: root167Tail55
def root167Tail53 : List Nat := 21863 :: root167Tail54
def root167Tail52 : List Nat := 21569 :: root167Tail53
def root167Tail51 : List Nat := 21023 :: root167Tail52
def root167Tail50 : List Nat := 20903 :: root167Tail51
def root167Tail49 : List Nat := 19973 :: root167Tail50
def root167Tail48 : List Nat := 19319 :: root167Tail49
def root167Tail47 : List Nat := 19181 :: root167Tail48
def root167Tail46 : List Nat := 18803 :: root167Tail47
def root167Tail45 : List Nat := 18143 :: root167Tail46
def root167Tail44 : List Nat := 18047 :: root167Tail45
def root167Tail43 : List Nat := 17681 :: root167Tail44
def root167Tail42 : List Nat := 16703 :: root167Tail43
def root167Tail41 : List Nat := 16673 :: root167Tail42
def root167Tail40 : List Nat := 16427 :: root167Tail41
def root167Tail39 : List Nat := 15749 :: root167Tail40
def root167Tail38 : List Nat := 15077 :: root167Tail39
def root167Tail37 : List Nat := 14957 :: root167Tail38
def root167Tail36 : List Nat := 14891 :: root167Tail37
def root167Tail35 : List Nat := 14561 :: root167Tail36
def root167Tail34 : List Nat := 13103 :: root167Tail35
def root167Tail33 : List Nat := 11831 :: root167Tail34
def root167Tail32 : List Nat := 11783 :: root167Tail33
def root167Tail31 : List Nat := 11411 :: root167Tail32
def root167Tail30 : List Nat := 11003 :: root167Tail31
def root167Tail29 : List Nat := 10151 :: root167Tail30
def root167Tail28 : List Nat := 9719 :: root167Tail29
def root167Tail27 : List Nat := 8867 :: root167Tail28
def root167Tail26 : List Nat := 8231 :: root167Tail27
def root167Tail25 : List Nat := 7937 :: root167Tail26
def root167Tail24 : List Nat := 7577 :: root167Tail25
def root167Tail23 : List Nat := 7433 :: root167Tail24
def root167Tail22 : List Nat := 7331 :: root167Tail23
def root167Tail21 : List Nat := 7247 :: root167Tail22
def root167Tail20 : List Nat := 7187 :: root167Tail21
def root167Tail19 : List Nat := 6761 :: root167Tail20
def root167Tail18 : List Nat := 6719 :: root167Tail19
def root167Tail17 : List Nat := 6323 :: root167Tail18
def root167Tail16 : List Nat := 6221 :: root167Tail17
def root167Tail15 : List Nat := 6047 :: root167Tail16
def root167Tail14 : List Nat := 5087 :: root167Tail15
def root167Tail13 : List Nat := 5039 :: root167Tail14
def root167Tail12 : List Nat := 4349 :: root167Tail13
def root167Tail11 : List Nat := 4259 :: root167Tail12
def root167Tail10 : List Nat := 4139 :: root167Tail11
def root167Tail9 : List Nat := 4133 :: root167Tail10
def root167Tail8 : List Nat := 3803 :: root167Tail9
def root167Tail7 : List Nat := 3659 :: root167Tail8
def root167Tail6 : List Nat := 2543 :: root167Tail7
def root167Tail5 : List Nat := 2273 :: root167Tail6
def root167Tail4 : List Nat := 1811 :: root167Tail5
def root167Tail3 : List Nat := 1721 :: root167Tail4
def root167Tail2 : List Nat := 1697 :: root167Tail3
def root167Tail1 : List Nat := 1097 :: root167Tail2
def root167Tail0 : List Nat := 1091 :: root167Tail1
def root167Candidates0 : List Nat := [1697, 19319, 20903, 22721]
set_option maxRecDepth 8192 in
theorem root167Candidates0_checked : candidates edge 4 1013 1091 root167Tail1 winner.bound = root167Candidates0 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root167Child0 : search edge 3 [1013, 1091] root167Candidates0 2104 winner = winner := by decide
theorem root167Step0 : scan edge (search edge 3) 4 [1013] 1013 root167Tail0 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail1 winner := by
  rw [root167Tail0, scan, if_neg (by decide), root167Candidates0_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail1
    (if 3 ≤ root167Candidates0.length then search edge 3 [1013, 1091] root167Candidates0 2104 winner else winner) = _
  rw [if_pos (by decide), root167Child0]
def root167Candidates1 : List Nat := [5087, 6323]
set_option maxRecDepth 8192 in
theorem root167Candidates1_checked : candidates edge 4 1013 1097 root167Tail2 winner.bound = root167Candidates1 := by
  unfold candidates
  have row : edge 1097 = (SortedNatTree.contains FastRows.row1097) := by
    change Data.row1097.contains = (SortedNatTree.contains FastRows.row1097)
    exact FastRows.row1097_checked.symm
  rw [row]
  decide
theorem root167Step1 : scan edge (search edge 3) 4 [1013] 1013 root167Tail1 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail2 winner := by
  rw [root167Tail1, scan, if_neg (by decide), root167Candidates1_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail2
    (if 3 ≤ root167Candidates1.length then search edge 3 [1013, 1097] root167Candidates1 2110 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates2 : List Nat := [5087]
set_option maxRecDepth 8192 in
theorem root167Candidates2_checked : candidates edge 4 1013 1697 root167Tail3 winner.bound = root167Candidates2 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root167Step2 : scan edge (search edge 3) 4 [1013] 1013 root167Tail2 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail3 winner := by
  rw [root167Tail2, scan, if_neg (by decide), root167Candidates2_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail3
    (if 3 ≤ root167Candidates2.length then search edge 3 [1013, 1697] root167Candidates2 2710 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates3 : List Nat := [8231, 14561]
set_option maxRecDepth 8192 in
theorem root167Candidates3_checked : candidates edge 4 1013 1721 root167Tail4 winner.bound = root167Candidates3 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root167Step3 : scan edge (search edge 3) 4 [1013] 1013 root167Tail3 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail4 winner := by
  rw [root167Tail3, scan, if_neg (by decide), root167Candidates3_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail4
    (if 3 ≤ root167Candidates3.length then search edge 3 [1013, 1721] root167Candidates3 2734 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates4 : List Nat := [5039]
set_option maxRecDepth 8192 in
theorem root167Candidates4_checked : candidates edge 4 1013 1811 root167Tail5 winner.bound = root167Candidates4 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
theorem root167Step4 : scan edge (search edge 3) 4 [1013] 1013 root167Tail4 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail5 winner := by
  rw [root167Tail4, scan, if_neg (by decide), root167Candidates4_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail5
    (if 3 ≤ root167Candidates4.length then search edge 3 [1013, 1811] root167Candidates4 2824 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates5 : List Nat := [19181, 21023]
set_option maxRecDepth 8192 in
theorem root167Candidates5_checked : candidates edge 4 1013 2273 root167Tail6 winner.bound = root167Candidates5 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
theorem root167Step5 : scan edge (search edge 3) 4 [1013] 1013 root167Tail5 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail6 winner := by
  rw [root167Tail5, scan, if_neg (by decide), root167Candidates5_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail6
    (if 3 ≤ root167Candidates5.length then search edge 3 [1013, 2273] root167Candidates5 3286 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates6 : List Nat := [4349, 7331, 19973]
set_option maxRecDepth 8192 in
theorem root167Candidates6_checked : candidates edge 4 1013 2543 root167Tail7 winner.bound = root167Candidates6 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root167Child6 : search edge 3 [1013, 2543] root167Candidates6 3556 winner = winner := by decide
theorem root167Step6 : scan edge (search edge 3) 4 [1013] 1013 root167Tail6 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail7 winner := by
  rw [root167Tail6, scan, if_neg (by decide), root167Candidates6_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail7
    (if 3 ≤ root167Candidates6.length then search edge 3 [1013, 2543] root167Candidates6 3556 winner else winner) = _
  rw [if_pos (by decide), root167Child6]
def root167Candidates7 : List Nat := [3803, 6047, 14561, 14957]
set_option maxRecDepth 8192 in
theorem root167Candidates7_checked : candidates edge 4 1013 3659 root167Tail8 winner.bound = root167Candidates7 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root167Child7 : search edge 3 [1013, 3659] root167Candidates7 4672 winner = winner := by decide
theorem root167Step7 : scan edge (search edge 3) 4 [1013] 1013 root167Tail7 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail8 winner := by
  rw [root167Tail7, scan, if_neg (by decide), root167Candidates7_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail8
    (if 3 ≤ root167Candidates7.length then search edge 3 [1013, 3659] root167Candidates7 4672 winner else winner) = _
  rw [if_pos (by decide), root167Child7]
def root167Candidates8 : List Nat := [5087]
set_option maxRecDepth 8192 in
theorem root167Candidates8_checked : candidates edge 4 1013 3803 root167Tail9 winner.bound = root167Candidates8 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root167Step8 : scan edge (search edge 3) 4 [1013] 1013 root167Tail8 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail9 winner := by
  rw [root167Tail8, scan, if_neg (by decide), root167Candidates8_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail9
    (if 3 ≤ root167Candidates8.length then search edge 3 [1013, 3803] root167Candidates8 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates9 : List Nat := [7433]
set_option maxRecDepth 8192 in
theorem root167Candidates9_checked : candidates edge 4 1013 4133 root167Tail10 winner.bound = root167Candidates9 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root167Step9 : scan edge (search edge 3) 4 [1013] 1013 root167Tail9 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail10 winner := by
  rw [root167Tail9, scan, if_neg (by decide), root167Candidates9_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail10
    (if 3 ≤ root167Candidates9.length then search edge 3 [1013, 4133] root167Candidates9 5146 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates10 : List Nat := [8867, 13103]
set_option maxRecDepth 8192 in
theorem root167Candidates10_checked : candidates edge 4 1013 4139 root167Tail11 winner.bound = root167Candidates10 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root167Step10 : scan edge (search edge 3) 4 [1013] 1013 root167Tail10 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail11 winner := by
  rw [root167Tail10, scan, if_neg (by decide), root167Candidates10_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail11
    (if 3 ≤ root167Candidates10.length then search edge 3 [1013, 4139] root167Candidates10 5152 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates11 : List Nat := [5087, 6221]
set_option maxRecDepth 8192 in
theorem root167Candidates11_checked : candidates edge 4 1013 4259 root167Tail12 winner.bound = root167Candidates11 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root167Step11 : scan edge (search edge 3) 4 [1013] 1013 root167Tail11 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail12 winner := by
  rw [root167Tail11, scan, if_neg (by decide), root167Candidates11_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail12
    (if 3 ≤ root167Candidates11.length then search edge 3 [1013, 4259] root167Candidates11 5272 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root167Candidates12_checked : candidates edge 4 1013 4349 root167Tail13 winner.bound = root167Candidates12 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root167Step12 : scan edge (search edge 3) 4 [1013] 1013 root167Tail12 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail13 winner := by
  rw [root167Tail12, scan, if_neg (by decide), root167Candidates12_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail13
    (if 3 ≤ root167Candidates12.length then search edge 3 [1013, 4349] root167Candidates12 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates13 : List Nat := [8867]
set_option maxRecDepth 8192 in
theorem root167Candidates13_checked : candidates edge 4 1013 5039 root167Tail14 winner.bound = root167Candidates13 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root167Step13 : scan edge (search edge 3) 4 [1013] 1013 root167Tail13 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail14 winner := by
  rw [root167Tail13, scan, if_neg (by decide), root167Candidates13_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail14
    (if 3 ≤ root167Candidates13.length then search edge 3 [1013, 5039] root167Candidates13 6052 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates14 : List Nat := [6719, 7247, 14561]
set_option maxRecDepth 8192 in
theorem root167Candidates14_checked : candidates edge 4 1013 5087 root167Tail15 winner.bound = root167Candidates14 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root167Child14 : search edge 3 [1013, 5087] root167Candidates14 6100 winner = winner := by decide
theorem root167Step14 : scan edge (search edge 3) 4 [1013] 1013 root167Tail14 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail15 winner := by
  rw [root167Tail14, scan, if_neg (by decide), root167Candidates14_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail15
    (if 3 ≤ root167Candidates14.length then search edge 3 [1013, 5087] root167Candidates14 6100 winner else winner) = _
  rw [if_pos (by decide), root167Child14]
def root167Candidates15 : List Nat := [6719, 17681]
set_option maxRecDepth 8192 in
theorem root167Candidates15_checked : candidates edge 4 1013 6047 root167Tail16 winner.bound = root167Candidates15 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root167Step15 : scan edge (search edge 3) 4 [1013] 1013 root167Tail15 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail16 winner := by
  rw [root167Tail15, scan, if_neg (by decide), root167Candidates15_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail16
    (if 3 ≤ root167Candidates15.length then search edge 3 [1013, 6047] root167Candidates15 7060 winner else winner) = _
  rw [if_neg (by decide)]
def root167Candidates16 : List Nat := [7577, 9719]
set_option maxRecDepth 8192 in
theorem root167Candidates16_checked : candidates edge 4 1013 6221 root167Tail17 winner.bound = root167Candidates16 := by
  unfold candidates
  have row : edge 6221 = (SortedNatTree.contains FastRows.row6221) := by
    change Data.row6221.contains = (SortedNatTree.contains FastRows.row6221)
    exact FastRows.row6221_checked.symm
  rw [row]
  decide
theorem root167Step16 : scan edge (search edge 3) 4 [1013] 1013 root167Tail16 winner =
    scan edge (search edge 3) 4 [1013] 1013 root167Tail17 winner := by
  rw [root167Tail16, scan, if_neg (by decide), root167Candidates16_checked]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail17
    (if 3 ≤ root167Candidates16.length then search edge 3 [1013, 6221] root167Candidates16 7234 winner else winner) = _
  rw [if_neg (by decide)]
theorem root167Scan17 : scan edge (search edge 3) 4 [1013] 1013 root167Tail17 winner = winner := by rw [root167Tail17, scan, if_pos (by decide)]
theorem root167Scan16 : scan edge (search edge 3) 4 [1013] 1013 root167Tail16 winner = winner :=
  (root167Step16).trans root167Scan17
theorem root167Scan15 : scan edge (search edge 3) 4 [1013] 1013 root167Tail15 winner = winner :=
  (root167Step15).trans root167Scan16
theorem root167Scan14 : scan edge (search edge 3) 4 [1013] 1013 root167Tail14 winner = winner :=
  (root167Step14).trans root167Scan15
theorem root167Scan13 : scan edge (search edge 3) 4 [1013] 1013 root167Tail13 winner = winner :=
  (root167Step13).trans root167Scan14
theorem root167Scan12 : scan edge (search edge 3) 4 [1013] 1013 root167Tail12 winner = winner :=
  (root167Step12).trans root167Scan13
theorem root167Scan11 : scan edge (search edge 3) 4 [1013] 1013 root167Tail11 winner = winner :=
  (root167Step11).trans root167Scan12
theorem root167Scan10 : scan edge (search edge 3) 4 [1013] 1013 root167Tail10 winner = winner :=
  (root167Step10).trans root167Scan11
theorem root167Scan9 : scan edge (search edge 3) 4 [1013] 1013 root167Tail9 winner = winner :=
  (root167Step9).trans root167Scan10
theorem root167Scan8 : scan edge (search edge 3) 4 [1013] 1013 root167Tail8 winner = winner :=
  (root167Step8).trans root167Scan9
theorem root167Scan7 : scan edge (search edge 3) 4 [1013] 1013 root167Tail7 winner = winner :=
  (root167Step7).trans root167Scan8
theorem root167Scan6 : scan edge (search edge 3) 4 [1013] 1013 root167Tail6 winner = winner :=
  (root167Step6).trans root167Scan7
theorem root167Scan5 : scan edge (search edge 3) 4 [1013] 1013 root167Tail5 winner = winner :=
  (root167Step5).trans root167Scan6
theorem root167Scan4 : scan edge (search edge 3) 4 [1013] 1013 root167Tail4 winner = winner :=
  (root167Step4).trans root167Scan5
theorem root167Scan3 : scan edge (search edge 3) 4 [1013] 1013 root167Tail3 winner = winner :=
  (root167Step3).trans root167Scan4
theorem root167Scan2 : scan edge (search edge 3) 4 [1013] 1013 root167Tail2 winner = winner :=
  (root167Step2).trans root167Scan3
theorem root167Scan1 : scan edge (search edge 3) 4 [1013] 1013 root167Tail1 winner = winner :=
  (root167Step1).trans root167Scan2
theorem root167Scan0 : scan edge (search edge 3) 4 [1013] 1013 root167Tail0 winner = winner :=
  (root167Step0).trans root167Scan1
set_option maxRecDepth 4096 in
theorem root167Branch : search edge 4 [1013] root167Candidates 1013 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1013] 1013 root167Tail0 winner = winner
  exact root167Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
