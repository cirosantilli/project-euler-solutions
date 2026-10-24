import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root115Candidates : List Nat := Data.row647
set_option maxRecDepth 8192 in
theorem root115Candidates_checked : candidates edge 5 0 647 tail116 winner.bound = root115Candidates := by
  apply candidates_eq_row edge 5 0 647 tail116 root115Candidates winner.bound tail116_ordered
  · decide
  · decide
  · funext q
    rfl
def root115Tail53 : List Nat := []
def root115Tail52 : List Nat := 25127 :: root115Tail53
def root115Tail51 : List Nat := 24677 :: root115Tail52
def root115Tail50 : List Nat := 24623 :: root115Tail51
def root115Tail49 : List Nat := 24509 :: root115Tail50
def root115Tail48 : List Nat := 22817 :: root115Tail49
def root115Tail47 : List Nat := 22727 :: root115Tail48
def root115Tail46 : List Nat := 22271 :: root115Tail47
def root115Tail45 : List Nat := 21803 :: root115Tail46
def root115Tail44 : List Nat := 21599 :: root115Tail45
def root115Tail43 : List Nat := 21323 :: root115Tail44
def root115Tail42 : List Nat := 20297 :: root115Tail43
def root115Tail41 : List Nat := 19583 :: root115Tail42
def root115Tail40 : List Nat := 19559 :: root115Tail41
def root115Tail39 : List Nat := 19541 :: root115Tail40
def root115Tail38 : List Nat := 18713 :: root115Tail39
def root115Tail37 : List Nat := 17291 :: root115Tail38
def root115Tail36 : List Nat := 15149 :: root115Tail37
def root115Tail35 : List Nat := 14891 :: root115Tail36
def root115Tail34 : List Nat := 14669 :: root115Tail35
def root115Tail33 : List Nat := 14561 :: root115Tail34
def root115Tail32 : List Nat := 13613 :: root115Tail33
def root115Tail31 : List Nat := 13367 :: root115Tail32
def root115Tail30 : List Nat := 13217 :: root115Tail31
def root115Tail29 : List Nat := 12941 :: root115Tail30
def root115Tail28 : List Nat := 11831 :: root115Tail29
def root115Tail27 : List Nat := 11159 :: root115Tail28
def root115Tail26 : List Nat := 10739 :: root115Tail27
def root115Tail25 : List Nat := 10301 :: root115Tail26
def root115Tail24 : List Nat := 9923 :: root115Tail25
def root115Tail23 : List Nat := 9521 :: root115Tail24
def root115Tail22 : List Nat := 8273 :: root115Tail23
def root115Tail21 : List Nat := 7949 :: root115Tail22
def root115Tail20 : List Nat := 7253 :: root115Tail21
def root115Tail19 : List Nat := 6329 :: root115Tail20
def root115Tail18 : List Nat := 5879 :: root115Tail19
def root115Tail17 : List Nat := 5471 :: root115Tail18
def root115Tail16 : List Nat := 5387 :: root115Tail17
def root115Tail15 : List Nat := 5153 :: root115Tail16
def root115Tail14 : List Nat := 4493 :: root115Tail15
def root115Tail13 : List Nat := 4241 :: root115Tail14
def root115Tail12 : List Nat := 4019 :: root115Tail13
def root115Tail11 : List Nat := 3911 :: root115Tail12
def root115Tail10 : List Nat := 3407 :: root115Tail11
def root115Tail9 : List Nat := 3347 :: root115Tail10
def root115Tail8 : List Nat := 3011 :: root115Tail9
def root115Tail7 : List Nat := 1889 :: root115Tail8
def root115Tail6 : List Nat := 1811 :: root115Tail7
def root115Tail5 : List Nat := 1733 :: root115Tail6
def root115Tail4 : List Nat := 1667 :: root115Tail5
def root115Tail3 : List Nat := 1511 :: root115Tail4
def root115Tail2 : List Nat := 1019 :: root115Tail3
def root115Tail1 : List Nat := 953 :: root115Tail2
def root115Tail0 : List Nat := 821 :: root115Tail1
def root115Candidates0 : List Nat := [22727]
set_option maxRecDepth 8192 in
theorem root115Candidates0_checked : candidates edge 4 647 821 root115Tail1 winner.bound = root115Candidates0 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
theorem root115Step0 : scan edge (search edge 3) 4 [647] 647 root115Tail0 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail1 winner := by
  rw [root115Tail0, scan, if_neg (by decide), root115Candidates0_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail1
    (if 3 ≤ root115Candidates0.length then search edge 3 [647, 821] root115Candidates0 1468 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root115Candidates1_checked : candidates edge 4 647 953 root115Tail2 winner.bound = root115Candidates1 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
theorem root115Step1 : scan edge (search edge 3) 4 [647] 647 root115Tail1 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail2 winner := by
  rw [root115Tail1, scan, if_neg (by decide), root115Candidates1_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail2
    (if 3 ≤ root115Candidates1.length then search edge 3 [647, 953] root115Candidates1 1600 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates2 : List Nat := [7949, 13367]
set_option maxRecDepth 8192 in
theorem root115Candidates2_checked : candidates edge 4 647 1019 root115Tail3 winner.bound = root115Candidates2 := by
  unfold candidates
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row]
  decide
theorem root115Step2 : scan edge (search edge 3) 4 [647] 647 root115Tail2 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail3 winner := by
  rw [root115Tail2, scan, if_neg (by decide), root115Candidates2_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail3
    (if 3 ≤ root115Candidates2.length then search edge 3 [647, 1019] root115Candidates2 1666 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates3 : List Nat := [3011, 15149]
set_option maxRecDepth 8192 in
theorem root115Candidates3_checked : candidates edge 4 647 1511 root115Tail4 winner.bound = root115Candidates3 := by
  unfold candidates
  have row : edge 1511 = (SortedNatTree.contains FastRows.row1511) := by
    change Data.row1511.contains = (SortedNatTree.contains FastRows.row1511)
    exact FastRows.row1511_checked.symm
  rw [row]
  decide
theorem root115Step3 : scan edge (search edge 3) 4 [647] 647 root115Tail3 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail4 winner := by
  rw [root115Tail3, scan, if_neg (by decide), root115Candidates3_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail4
    (if 3 ≤ root115Candidates3.length then search edge 3 [647, 1511] root115Candidates3 2158 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates4 : List Nat := [6329]
set_option maxRecDepth 8192 in
theorem root115Candidates4_checked : candidates edge 4 647 1667 root115Tail5 winner.bound = root115Candidates4 := by
  unfold candidates
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row]
  decide
theorem root115Step4 : scan edge (search edge 3) 4 [647] 647 root115Tail4 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail5 winner := by
  rw [root115Tail4, scan, if_neg (by decide), root115Candidates4_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail5
    (if 3 ≤ root115Candidates4.length then search edge 3 [647, 1667] root115Candidates4 2314 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates5 : List Nat := [3347, 7949, 14891]
set_option maxRecDepth 8192 in
theorem root115Candidates5_checked : candidates edge 4 647 1733 root115Tail6 winner.bound = root115Candidates5 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root115Child5 : search edge 3 [647, 1733] root115Candidates5 2380 winner = winner := by decide
theorem root115Step5 : scan edge (search edge 3) 4 [647] 647 root115Tail5 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail6 winner := by
  rw [root115Tail5, scan, if_neg (by decide), root115Candidates5_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail6
    (if 3 ≤ root115Candidates5.length then search edge 3 [647, 1733] root115Candidates5 2380 winner else winner) = _
  rw [if_pos (by decide), root115Child5]
def root115Candidates6 : List Nat := [18713]
set_option maxRecDepth 8192 in
theorem root115Candidates6_checked : candidates edge 4 647 1811 root115Tail7 winner.bound = root115Candidates6 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
theorem root115Step6 : scan edge (search edge 3) 4 [647] 647 root115Tail6 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail7 winner := by
  rw [root115Tail6, scan, if_neg (by decide), root115Candidates6_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail7
    (if 3 ≤ root115Candidates6.length then search edge 3 [647, 1811] root115Candidates6 2458 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates7 : List Nat := [4019, 17291]
set_option maxRecDepth 8192 in
theorem root115Candidates7_checked : candidates edge 4 647 1889 root115Tail8 winner.bound = root115Candidates7 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root115Step7 : scan edge (search edge 3) 4 [647] 647 root115Tail7 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail8 winner := by
  rw [root115Tail7, scan, if_neg (by decide), root115Candidates7_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail8
    (if 3 ≤ root115Candidates7.length then search edge 3 [647, 1889] root115Candidates7 2536 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates8 : List Nat := [4493, 6329]
set_option maxRecDepth 8192 in
theorem root115Candidates8_checked : candidates edge 4 647 3011 root115Tail9 winner.bound = root115Candidates8 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root115Step8 : scan edge (search edge 3) 4 [647] 647 root115Tail8 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail9 winner := by
  rw [root115Tail8, scan, if_neg (by decide), root115Candidates8_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail9
    (if 3 ≤ root115Candidates8.length then search edge 3 [647, 3011] root115Candidates8 3658 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates9 : List Nat := [8273]
set_option maxRecDepth 8192 in
theorem root115Candidates9_checked : candidates edge 4 647 3347 root115Tail10 winner.bound = root115Candidates9 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root115Step9 : scan edge (search edge 3) 4 [647] 647 root115Tail9 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail10 winner := by
  rw [root115Tail9, scan, if_neg (by decide), root115Candidates9_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail10
    (if 3 ≤ root115Candidates9.length then search edge 3 [647, 3347] root115Candidates9 3994 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root115Candidates10_checked : candidates edge 4 647 3407 root115Tail11 winner.bound = root115Candidates10 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root115Step10 : scan edge (search edge 3) 4 [647] 647 root115Tail10 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail11 winner := by
  rw [root115Tail10, scan, if_neg (by decide), root115Candidates10_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail11
    (if 3 ≤ root115Candidates10.length then search edge 3 [647, 3407] root115Candidates10 4054 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates11 : List Nat := [10739]
set_option maxRecDepth 8192 in
theorem root115Candidates11_checked : candidates edge 4 647 3911 root115Tail12 winner.bound = root115Candidates11 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root115Step11 : scan edge (search edge 3) 4 [647] 647 root115Tail11 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail12 winner := by
  rw [root115Tail11, scan, if_neg (by decide), root115Candidates11_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail12
    (if 3 ≤ root115Candidates11.length then search edge 3 [647, 3911] root115Candidates11 4558 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates12 : List Nat := [5471]
set_option maxRecDepth 8192 in
theorem root115Candidates12_checked : candidates edge 4 647 4019 root115Tail13 winner.bound = root115Candidates12 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root115Step12 : scan edge (search edge 3) 4 [647] 647 root115Tail12 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail13 winner := by
  rw [root115Tail12, scan, if_neg (by decide), root115Candidates12_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail13
    (if 3 ≤ root115Candidates12.length then search edge 3 [647, 4019] root115Candidates12 4666 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root115Candidates13_checked : candidates edge 4 647 4241 root115Tail14 winner.bound = root115Candidates13 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root115Step13 : scan edge (search edge 3) 4 [647] 647 root115Tail13 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail14 winner := by
  rw [root115Tail13, scan, if_neg (by decide), root115Candidates13_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail14
    (if 3 ≤ root115Candidates13.length then search edge 3 [647, 4241] root115Candidates13 4888 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates14 : List Nat := [20297]
set_option maxRecDepth 8192 in
theorem root115Candidates14_checked : candidates edge 4 647 4493 root115Tail15 winner.bound = root115Candidates14 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root115Step14 : scan edge (search edge 3) 4 [647] 647 root115Tail14 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail15 winner := by
  rw [root115Tail14, scan, if_neg (by decide), root115Candidates14_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail15
    (if 3 ≤ root115Candidates14.length then search edge 3 [647, 4493] root115Candidates14 5140 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root115Candidates15_checked : candidates edge 4 647 5153 root115Tail16 winner.bound = root115Candidates15 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root115Step15 : scan edge (search edge 3) 4 [647] 647 root115Tail15 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail16 winner := by
  rw [root115Tail15, scan, if_neg (by decide), root115Candidates15_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail16
    (if 3 ≤ root115Candidates15.length then search edge 3 [647, 5153] root115Candidates15 5800 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root115Candidates16_checked : candidates edge 4 647 5387 root115Tail17 winner.bound = root115Candidates16 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root115Step16 : scan edge (search edge 3) 4 [647] 647 root115Tail16 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail17 winner := by
  rw [root115Tail16, scan, if_neg (by decide), root115Candidates16_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail17
    (if 3 ≤ root115Candidates16.length then search edge 3 [647, 5387] root115Candidates16 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root115Candidates17_checked : candidates edge 4 647 5471 root115Tail18 winner.bound = root115Candidates17 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root115Step17 : scan edge (search edge 3) 4 [647] 647 root115Tail17 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail18 winner := by
  rw [root115Tail17, scan, if_neg (by decide), root115Candidates17_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail18
    (if 3 ≤ root115Candidates17.length then search edge 3 [647, 5471] root115Candidates17 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates18 : List Nat := [8273]
set_option maxRecDepth 8192 in
theorem root115Candidates18_checked : candidates edge 4 647 5879 root115Tail19 winner.bound = root115Candidates18 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root115Step18 : scan edge (search edge 3) 4 [647] 647 root115Tail18 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail19 winner := by
  rw [root115Tail18, scan, if_neg (by decide), root115Candidates18_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail19
    (if 3 ≤ root115Candidates18.length then search edge 3 [647, 5879] root115Candidates18 6526 winner else winner) = _
  rw [if_neg (by decide)]
def root115Candidates19 : List Nat := [10739, 14561]
set_option maxRecDepth 8192 in
theorem root115Candidates19_checked : candidates edge 4 647 6329 root115Tail20 winner.bound = root115Candidates19 := by
  unfold candidates
  have row : edge 6329 = (SortedNatTree.contains FastRows.row6329) := by
    change Data.row6329.contains = (SortedNatTree.contains FastRows.row6329)
    exact FastRows.row6329_checked.symm
  rw [row]
  decide
theorem root115Step19 : scan edge (search edge 3) 4 [647] 647 root115Tail19 winner =
    scan edge (search edge 3) 4 [647] 647 root115Tail20 winner := by
  rw [root115Tail19, scan, if_neg (by decide), root115Candidates19_checked]
  change scan edge (search edge 3) 4 [647] 647 root115Tail20
    (if 3 ≤ root115Candidates19.length then search edge 3 [647, 6329] root115Candidates19 6976 winner else winner) = _
  rw [if_neg (by decide)]
theorem root115Scan20 : scan edge (search edge 3) 4 [647] 647 root115Tail20 winner = winner := by rw [root115Tail20, scan, if_pos (by decide)]
theorem root115Scan19 : scan edge (search edge 3) 4 [647] 647 root115Tail19 winner = winner :=
  (root115Step19).trans root115Scan20
theorem root115Scan18 : scan edge (search edge 3) 4 [647] 647 root115Tail18 winner = winner :=
  (root115Step18).trans root115Scan19
theorem root115Scan17 : scan edge (search edge 3) 4 [647] 647 root115Tail17 winner = winner :=
  (root115Step17).trans root115Scan18
theorem root115Scan16 : scan edge (search edge 3) 4 [647] 647 root115Tail16 winner = winner :=
  (root115Step16).trans root115Scan17
theorem root115Scan15 : scan edge (search edge 3) 4 [647] 647 root115Tail15 winner = winner :=
  (root115Step15).trans root115Scan16
theorem root115Scan14 : scan edge (search edge 3) 4 [647] 647 root115Tail14 winner = winner :=
  (root115Step14).trans root115Scan15
theorem root115Scan13 : scan edge (search edge 3) 4 [647] 647 root115Tail13 winner = winner :=
  (root115Step13).trans root115Scan14
theorem root115Scan12 : scan edge (search edge 3) 4 [647] 647 root115Tail12 winner = winner :=
  (root115Step12).trans root115Scan13
theorem root115Scan11 : scan edge (search edge 3) 4 [647] 647 root115Tail11 winner = winner :=
  (root115Step11).trans root115Scan12
theorem root115Scan10 : scan edge (search edge 3) 4 [647] 647 root115Tail10 winner = winner :=
  (root115Step10).trans root115Scan11
theorem root115Scan9 : scan edge (search edge 3) 4 [647] 647 root115Tail9 winner = winner :=
  (root115Step9).trans root115Scan10
theorem root115Scan8 : scan edge (search edge 3) 4 [647] 647 root115Tail8 winner = winner :=
  (root115Step8).trans root115Scan9
theorem root115Scan7 : scan edge (search edge 3) 4 [647] 647 root115Tail7 winner = winner :=
  (root115Step7).trans root115Scan8
theorem root115Scan6 : scan edge (search edge 3) 4 [647] 647 root115Tail6 winner = winner :=
  (root115Step6).trans root115Scan7
theorem root115Scan5 : scan edge (search edge 3) 4 [647] 647 root115Tail5 winner = winner :=
  (root115Step5).trans root115Scan6
theorem root115Scan4 : scan edge (search edge 3) 4 [647] 647 root115Tail4 winner = winner :=
  (root115Step4).trans root115Scan5
theorem root115Scan3 : scan edge (search edge 3) 4 [647] 647 root115Tail3 winner = winner :=
  (root115Step3).trans root115Scan4
theorem root115Scan2 : scan edge (search edge 3) 4 [647] 647 root115Tail2 winner = winner :=
  (root115Step2).trans root115Scan3
theorem root115Scan1 : scan edge (search edge 3) 4 [647] 647 root115Tail1 winner = winner :=
  (root115Step1).trans root115Scan2
theorem root115Scan0 : scan edge (search edge 3) 4 [647] 647 root115Tail0 winner = winner :=
  (root115Step0).trans root115Scan1
set_option maxRecDepth 4096 in
theorem root115Branch : search edge 4 [647] root115Candidates 647 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [647] 647 root115Tail0 winner = winner
  exact root115Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
