import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root248Candidates : List Nat := Data.row1597
set_option maxRecDepth 8192 in
theorem root248Candidates_checked : candidates edge 5 0 1597 tail249 winner.bound = root248Candidates := by
  apply candidates_eq_row edge 5 0 1597 tail249 root248Candidates winner.bound tail249_ordered
  · decide
  · decide
  · funext q
    rfl
def root248Tail46 : List Nat := []
def root248Tail45 : List Nat := 22567 :: root248Tail46
def root248Tail44 : List Nat := 22501 :: root248Tail45
def root248Tail43 : List Nat := 22063 :: root248Tail44
def root248Tail42 : List Nat := 21031 :: root248Tail43
def root248Tail41 : List Nat := 20149 :: root248Tail42
def root248Tail40 : List Nat := 19447 :: root248Tail41
def root248Tail39 : List Nat := 18523 :: root248Tail40
def root248Tail38 : List Nat := 17011 :: root248Tail39
def root248Tail37 : List Nat := 16477 :: root248Tail38
def root248Tail36 : List Nat := 16111 :: root248Tail37
def root248Tail35 : List Nat := 15859 :: root248Tail36
def root248Tail34 : List Nat := 13537 :: root248Tail35
def root248Tail33 : List Nat := 13441 :: root248Tail34
def root248Tail32 : List Nat := 12919 :: root248Tail33
def root248Tail31 : List Nat := 12211 :: root248Tail32
def root248Tail30 : List Nat := 11833 :: root248Tail31
def root248Tail29 : List Nat := 11677 :: root248Tail30
def root248Tail28 : List Nat := 10531 :: root248Tail29
def root248Tail27 : List Nat := 10273 :: root248Tail28
def root248Tail26 : List Nat := 9829 :: root248Tail27
def root248Tail25 : List Nat := 9811 :: root248Tail26
def root248Tail24 : List Nat := 9661 :: root248Tail25
def root248Tail23 : List Nat := 9631 :: root248Tail24
def root248Tail22 : List Nat := 8803 :: root248Tail23
def root248Tail21 : List Nat := 8527 :: root248Tail22
def root248Tail20 : List Nat := 8269 :: root248Tail21
def root248Tail19 : List Nat := 7699 :: root248Tail20
def root248Tail18 : List Nat := 7573 :: root248Tail19
def root248Tail17 : List Nat := 7333 :: root248Tail18
def root248Tail16 : List Nat := 5503 :: root248Tail17
def root248Tail15 : List Nat := 4657 :: root248Tail16
def root248Tail14 : List Nat := 4363 :: root248Tail15
def root248Tail13 : List Nat := 4201 :: root248Tail14
def root248Tail12 : List Nat := 3823 :: root248Tail13
def root248Tail11 : List Nat := 3691 :: root248Tail12
def root248Tail10 : List Nat := 3469 :: root248Tail11
def root248Tail9 : List Nat := 3301 :: root248Tail10
def root248Tail8 : List Nat := 3163 :: root248Tail9
def root248Tail7 : List Nat := 3121 :: root248Tail8
def root248Tail6 : List Nat := 2731 :: root248Tail7
def root248Tail5 : List Nat := 2713 :: root248Tail6
def root248Tail4 : List Nat := 2617 :: root248Tail5
def root248Tail3 : List Nat := 2251 :: root248Tail4
def root248Tail2 : List Nat := 2113 :: root248Tail3
def root248Tail1 : List Nat := 1873 :: root248Tail2
def root248Tail0 : List Nat := 1777 :: root248Tail1
def root248Candidates0 : List Nat := [3121, 12211, 18523]
set_option maxRecDepth 8192 in
theorem root248Candidates0_checked : candidates edge 4 1597 1777 root248Tail1 winner.bound = root248Candidates0 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root248Child0 : search edge 3 [1597, 1777] root248Candidates0 3374 winner = winner := by decide
theorem root248Step0 : scan edge (search edge 3) 4 [1597] 1597 root248Tail0 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail1 winner := by
  rw [root248Tail0, scan, if_neg (by decide), root248Candidates0_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail1
    (if 3 ≤ root248Candidates0.length then search edge 3 [1597, 1777] root248Candidates0 3374 winner else winner) = _
  rw [if_pos (by decide), root248Child0]
def root248Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root248Candidates1_checked : candidates edge 4 1597 1873 root248Tail2 winner.bound = root248Candidates1 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root248Step1 : scan edge (search edge 3) 4 [1597] 1597 root248Tail1 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail2 winner := by
  rw [root248Tail1, scan, if_neg (by decide), root248Candidates1_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail2
    (if 3 ≤ root248Candidates1.length then search edge 3 [1597, 1873] root248Candidates1 3470 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates2 : List Nat := [3121, 12919]
set_option maxRecDepth 8192 in
theorem root248Candidates2_checked : candidates edge 4 1597 2113 root248Tail3 winner.bound = root248Candidates2 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root248Step2 : scan edge (search edge 3) 4 [1597] 1597 root248Tail2 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail3 winner := by
  rw [root248Tail2, scan, if_neg (by decide), root248Candidates2_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail3
    (if 3 ≤ root248Candidates2.length then search edge 3 [1597, 2113] root248Candidates2 3710 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates3 : List Nat := [4201, 5503]
set_option maxRecDepth 8192 in
theorem root248Candidates3_checked : candidates edge 4 1597 2251 root248Tail4 winner.bound = root248Candidates3 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
theorem root248Step3 : scan edge (search edge 3) 4 [1597] 1597 root248Tail3 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail4 winner := by
  rw [root248Tail3, scan, if_neg (by decide), root248Candidates3_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail4
    (if 3 ≤ root248Candidates3.length then search edge 3 [1597, 2251] root248Candidates3 3848 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates4 : List Nat := [9661]
set_option maxRecDepth 8192 in
theorem root248Candidates4_checked : candidates edge 4 1597 2617 root248Tail5 winner.bound = root248Candidates4 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
theorem root248Step4 : scan edge (search edge 3) 4 [1597] 1597 root248Tail4 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail5 winner := by
  rw [root248Tail4, scan, if_neg (by decide), root248Candidates4_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail5
    (if 3 ≤ root248Candidates4.length then search edge 3 [1597, 2617] root248Candidates4 4214 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root248Candidates5_checked : candidates edge 4 1597 2713 root248Tail6 winner.bound = root248Candidates5 := by
  unfold candidates
  have row : edge 2713 = (SortedNatTree.contains FastRows.row2713) := by
    change Data.row2713.contains = (SortedNatTree.contains FastRows.row2713)
    exact FastRows.row2713_checked.symm
  rw [row]
  decide
theorem root248Step5 : scan edge (search edge 3) 4 [1597] 1597 root248Tail5 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail6 winner := by
  rw [root248Tail5, scan, if_neg (by decide), root248Candidates5_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail6
    (if 3 ≤ root248Candidates5.length then search edge 3 [1597, 2713] root248Candidates5 4310 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates6 : List Nat := [3301, 4657, 9661, 10531]
set_option maxRecDepth 8192 in
theorem root248Candidates6_checked : candidates edge 4 1597 2731 root248Tail7 winner.bound = root248Candidates6 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root248Child6 : search edge 3 [1597, 2731] root248Candidates6 4328 winner = winner := by decide
theorem root248Step6 : scan edge (search edge 3) 4 [1597] 1597 root248Tail6 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail7 winner := by
  rw [root248Tail6, scan, if_neg (by decide), root248Candidates6_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail7
    (if 3 ≤ root248Candidates6.length then search edge 3 [1597, 2731] root248Candidates6 4328 winner else winner) = _
  rw [if_pos (by decide), root248Child6]
def root248Candidates7 : List Nat := [4201, 11677]
set_option maxRecDepth 8192 in
theorem root248Candidates7_checked : candidates edge 4 1597 3121 root248Tail8 winner.bound = root248Candidates7 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
theorem root248Step7 : scan edge (search edge 3) 4 [1597] 1597 root248Tail7 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail8 winner := by
  rw [root248Tail7, scan, if_neg (by decide), root248Candidates7_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail8
    (if 3 ≤ root248Candidates7.length then search edge 3 [1597, 3121] root248Candidates7 4718 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates8 : List Nat := [8527, 13537, 18523]
set_option maxRecDepth 8192 in
theorem root248Candidates8_checked : candidates edge 4 1597 3163 root248Tail9 winner.bound = root248Candidates8 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root248Child8 : search edge 3 [1597, 3163] root248Candidates8 4760 winner = winner := by decide
theorem root248Step8 : scan edge (search edge 3) 4 [1597] 1597 root248Tail8 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail9 winner := by
  rw [root248Tail8, scan, if_neg (by decide), root248Candidates8_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail9
    (if 3 ≤ root248Candidates8.length then search edge 3 [1597, 3163] root248Candidates8 4760 winner else winner) = _
  rw [if_pos (by decide), root248Child8]
def root248Candidates9 : List Nat := [4363, 10531]
set_option maxRecDepth 8192 in
theorem root248Candidates9_checked : candidates edge 4 1597 3301 root248Tail10 winner.bound = root248Candidates9 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root248Step9 : scan edge (search edge 3) 4 [1597] 1597 root248Tail9 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail10 winner := by
  rw [root248Tail9, scan, if_neg (by decide), root248Candidates9_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail10
    (if 3 ≤ root248Candidates9.length then search edge 3 [1597, 3301] root248Candidates9 4898 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates10 : List Nat := [5503, 10273]
set_option maxRecDepth 8192 in
theorem root248Candidates10_checked : candidates edge 4 1597 3469 root248Tail11 winner.bound = root248Candidates10 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root248Step10 : scan edge (search edge 3) 4 [1597] 1597 root248Tail10 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail11 winner := by
  rw [root248Tail10, scan, if_neg (by decide), root248Candidates10_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail11
    (if 3 ≤ root248Candidates10.length then search edge 3 [1597, 3469] root248Candidates10 5066 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates11 : List Nat := [7333]
set_option maxRecDepth 8192 in
theorem root248Candidates11_checked : candidates edge 4 1597 3691 root248Tail12 winner.bound = root248Candidates11 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
theorem root248Step11 : scan edge (search edge 3) 4 [1597] 1597 root248Tail11 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail12 winner := by
  rw [root248Tail11, scan, if_neg (by decide), root248Candidates11_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail12
    (if 3 ≤ root248Candidates11.length then search edge 3 [1597, 3691] root248Candidates11 5288 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates12 : List Nat := [4201, 11833, 15859]
set_option maxRecDepth 8192 in
theorem root248Candidates12_checked : candidates edge 4 1597 3823 root248Tail13 winner.bound = root248Candidates12 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root248Child12 : search edge 3 [1597, 3823] root248Candidates12 5420 winner = winner := by decide
theorem root248Step12 : scan edge (search edge 3) 4 [1597] 1597 root248Tail12 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail13 winner := by
  rw [root248Tail12, scan, if_neg (by decide), root248Candidates12_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail13
    (if 3 ≤ root248Candidates12.length then search edge 3 [1597, 3823] root248Candidates12 5420 winner else winner) = _
  rw [if_pos (by decide), root248Child12]
def root248Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root248Candidates13_checked : candidates edge 4 1597 4201 root248Tail14 winner.bound = root248Candidates13 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root248Step13 : scan edge (search edge 3) 4 [1597] 1597 root248Tail13 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail14 winner := by
  rw [root248Tail13, scan, if_neg (by decide), root248Candidates13_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail14
    (if 3 ≤ root248Candidates13.length then search edge 3 [1597, 4201] root248Candidates13 5798 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root248Candidates14_checked : candidates edge 4 1597 4363 root248Tail15 winner.bound = root248Candidates14 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root248Step14 : scan edge (search edge 3) 4 [1597] 1597 root248Tail14 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail15 winner := by
  rw [root248Tail14, scan, if_neg (by decide), root248Candidates14_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail15
    (if 3 ≤ root248Candidates14.length then search edge 3 [1597, 4363] root248Candidates14 5960 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root248Candidates15_checked : candidates edge 4 1597 4657 root248Tail16 winner.bound = root248Candidates15 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root248Step15 : scan edge (search edge 3) 4 [1597] 1597 root248Tail15 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail16 winner := by
  rw [root248Tail15, scan, if_neg (by decide), root248Candidates15_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail16
    (if 3 ≤ root248Candidates15.length then search edge 3 [1597, 4657] root248Candidates15 6254 winner else winner) = _
  rw [if_neg (by decide)]
def root248Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root248Candidates16_checked : candidates edge 4 1597 5503 root248Tail17 winner.bound = root248Candidates16 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root248Step16 : scan edge (search edge 3) 4 [1597] 1597 root248Tail16 winner =
    scan edge (search edge 3) 4 [1597] 1597 root248Tail17 winner := by
  rw [root248Tail16, scan, if_neg (by decide), root248Candidates16_checked]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail17
    (if 3 ≤ root248Candidates16.length then search edge 3 [1597, 5503] root248Candidates16 7100 winner else winner) = _
  rw [if_neg (by decide)]
theorem root248Scan17 : scan edge (search edge 3) 4 [1597] 1597 root248Tail17 winner = winner := by rw [root248Tail17, scan, if_pos (by decide)]
theorem root248Scan16 : scan edge (search edge 3) 4 [1597] 1597 root248Tail16 winner = winner :=
  (root248Step16).trans root248Scan17
theorem root248Scan15 : scan edge (search edge 3) 4 [1597] 1597 root248Tail15 winner = winner :=
  (root248Step15).trans root248Scan16
theorem root248Scan14 : scan edge (search edge 3) 4 [1597] 1597 root248Tail14 winner = winner :=
  (root248Step14).trans root248Scan15
theorem root248Scan13 : scan edge (search edge 3) 4 [1597] 1597 root248Tail13 winner = winner :=
  (root248Step13).trans root248Scan14
theorem root248Scan12 : scan edge (search edge 3) 4 [1597] 1597 root248Tail12 winner = winner :=
  (root248Step12).trans root248Scan13
theorem root248Scan11 : scan edge (search edge 3) 4 [1597] 1597 root248Tail11 winner = winner :=
  (root248Step11).trans root248Scan12
theorem root248Scan10 : scan edge (search edge 3) 4 [1597] 1597 root248Tail10 winner = winner :=
  (root248Step10).trans root248Scan11
theorem root248Scan9 : scan edge (search edge 3) 4 [1597] 1597 root248Tail9 winner = winner :=
  (root248Step9).trans root248Scan10
theorem root248Scan8 : scan edge (search edge 3) 4 [1597] 1597 root248Tail8 winner = winner :=
  (root248Step8).trans root248Scan9
theorem root248Scan7 : scan edge (search edge 3) 4 [1597] 1597 root248Tail7 winner = winner :=
  (root248Step7).trans root248Scan8
theorem root248Scan6 : scan edge (search edge 3) 4 [1597] 1597 root248Tail6 winner = winner :=
  (root248Step6).trans root248Scan7
theorem root248Scan5 : scan edge (search edge 3) 4 [1597] 1597 root248Tail5 winner = winner :=
  (root248Step5).trans root248Scan6
theorem root248Scan4 : scan edge (search edge 3) 4 [1597] 1597 root248Tail4 winner = winner :=
  (root248Step4).trans root248Scan5
theorem root248Scan3 : scan edge (search edge 3) 4 [1597] 1597 root248Tail3 winner = winner :=
  (root248Step3).trans root248Scan4
theorem root248Scan2 : scan edge (search edge 3) 4 [1597] 1597 root248Tail2 winner = winner :=
  (root248Step2).trans root248Scan3
theorem root248Scan1 : scan edge (search edge 3) 4 [1597] 1597 root248Tail1 winner = winner :=
  (root248Step1).trans root248Scan2
theorem root248Scan0 : scan edge (search edge 3) 4 [1597] 1597 root248Tail0 winner = winner :=
  (root248Step0).trans root248Scan1
set_option maxRecDepth 4096 in
theorem root248Branch : search edge 4 [1597] root248Candidates 1597 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1597] 1597 root248Tail0 winner = winner
  exact root248Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
