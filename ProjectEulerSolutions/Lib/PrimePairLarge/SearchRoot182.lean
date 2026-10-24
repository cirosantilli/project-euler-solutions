import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root182Candidates : List Nat := Data.row1103
set_option maxRecDepth 8192 in
theorem root182Candidates_checked : candidates edge 5 0 1103 tail183 winner.bound = root182Candidates := by
  apply candidates_eq_row edge 5 0 1103 tail183 root182Candidates winner.bound tail183_ordered
  · decide
  · decide
  · funext q
    rfl
def root182Tail48 : List Nat := []
def root182Tail47 : List Nat := 23909 :: root182Tail48
def root182Tail46 : List Nat := 23819 :: root182Tail47
def root182Tail45 : List Nat := 23531 :: root182Tail46
def root182Tail44 : List Nat := 23459 :: root182Tail45
def root182Tail43 : List Nat := 23399 :: root182Tail44
def root182Tail42 : List Nat := 23321 :: root182Tail43
def root182Tail41 : List Nat := 22073 :: root182Tail42
def root182Tail40 : List Nat := 21569 :: root182Tail41
def root182Tail39 : List Nat := 21269 :: root182Tail40
def root182Tail38 : List Nat := 20549 :: root182Tail39
def root182Tail37 : List Nat := 19211 :: root182Tail38
def root182Tail36 : List Nat := 18701 :: root182Tail37
def root182Tail35 : List Nat := 18443 :: root182Tail36
def root182Tail34 : List Nat := 17159 :: root182Tail35
def root182Tail33 : List Nat := 17117 :: root182Tail34
def root182Tail32 : List Nat := 16649 :: root182Tail33
def root182Tail31 : List Nat := 16187 :: root182Tail32
def root182Tail30 : List Nat := 15641 :: root182Tail31
def root182Tail29 : List Nat := 14639 :: root182Tail30
def root182Tail28 : List Nat := 14327 :: root182Tail29
def root182Tail27 : List Nat := 13457 :: root182Tail28
def root182Tail26 : List Nat := 13331 :: root182Tail27
def root182Tail25 : List Nat := 13127 :: root182Tail26
def root182Tail24 : List Nat := 12401 :: root182Tail25
def root182Tail23 : List Nat := 12197 :: root182Tail24
def root182Tail22 : List Nat := 11699 :: root182Tail23
def root182Tail21 : List Nat := 11489 :: root182Tail22
def root182Tail20 : List Nat := 8123 :: root182Tail21
def root182Tail19 : List Nat := 7529 :: root182Tail20
def root182Tail18 : List Nat := 6983 :: root182Tail19
def root182Tail17 : List Nat := 6329 :: root182Tail18
def root182Tail16 : List Nat := 6317 :: root182Tail17
def root182Tail15 : List Nat := 6143 :: root182Tail16
def root182Tail14 : List Nat := 6029 :: root182Tail15
def root182Tail13 : List Nat := 5987 :: root182Tail14
def root182Tail12 : List Nat := 5813 :: root182Tail13
def root182Tail11 : List Nat := 5807 :: root182Tail12
def root182Tail10 : List Nat := 5237 :: root182Tail11
def root182Tail9 : List Nat := 3557 :: root182Tail10
def root182Tail8 : List Nat := 3299 :: root182Tail9
def root182Tail7 : List Nat := 3041 :: root182Tail8
def root182Tail6 : List Nat := 2927 :: root182Tail7
def root182Tail5 : List Nat := 2909 :: root182Tail6
def root182Tail4 : List Nat := 2081 :: root182Tail5
def root182Tail3 : List Nat := 2039 :: root182Tail4
def root182Tail2 : List Nat := 1733 :: root182Tail3
def root182Tail1 : List Nat := 1373 :: root182Tail2
def root182Tail0 : List Nat := 1187 :: root182Tail1
def root182Candidates0 : List Nat := [6983]
set_option maxRecDepth 8192 in
theorem root182Candidates0_checked : candidates edge 4 1103 1187 root182Tail1 winner.bound = root182Candidates0 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
theorem root182Step0 : scan edge (search edge 3) 4 [1103] 1103 root182Tail0 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail1 winner := by
  rw [root182Tail0, scan, if_neg (by decide), root182Candidates0_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail1
    (if 3 ≤ root182Candidates0.length then search edge 3 [1103, 1187] root182Candidates0 2290 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates1 : List Nat := [6143]
set_option maxRecDepth 8192 in
theorem root182Candidates1_checked : candidates edge 4 1103 1373 root182Tail2 winner.bound = root182Candidates1 := by
  unfold candidates
  have row : edge 1373 = (SortedNatTree.contains FastRows.row1373) := by
    change Data.row1373.contains = (SortedNatTree.contains FastRows.row1373)
    exact FastRows.row1373_checked.symm
  rw [row]
  decide
theorem root182Step1 : scan edge (search edge 3) 4 [1103] 1103 root182Tail1 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail2 winner := by
  rw [root182Tail1, scan, if_neg (by decide), root182Candidates1_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail2
    (if 3 ≤ root182Candidates1.length then search edge 3 [1103, 1373] root182Candidates1 2476 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates2 : List Nat := [2927, 12401, 13457]
set_option maxRecDepth 8192 in
theorem root182Candidates2_checked : candidates edge 4 1103 1733 root182Tail3 winner.bound = root182Candidates2 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root182Child2 : search edge 3 [1103, 1733] root182Candidates2 2836 winner = winner := by decide
theorem root182Step2 : scan edge (search edge 3) 4 [1103] 1103 root182Tail2 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail3 winner := by
  rw [root182Tail2, scan, if_neg (by decide), root182Candidates2_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail3
    (if 3 ≤ root182Candidates2.length then search edge 3 [1103, 1733] root182Candidates2 2836 winner else winner) = _
  rw [if_pos (by decide), root182Child2]
def root182Candidates3 : List Nat := [3299]
set_option maxRecDepth 8192 in
theorem root182Candidates3_checked : candidates edge 4 1103 2039 root182Tail4 winner.bound = root182Candidates3 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
theorem root182Step3 : scan edge (search edge 3) 4 [1103] 1103 root182Tail3 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail4 winner := by
  rw [root182Tail3, scan, if_neg (by decide), root182Candidates3_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail4
    (if 3 ≤ root182Candidates3.length then search edge 3 [1103, 2039] root182Candidates3 3142 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates4 : List Nat := [5807, 11489]
set_option maxRecDepth 8192 in
theorem root182Candidates4_checked : candidates edge 4 1103 2081 root182Tail5 winner.bound = root182Candidates4 := by
  unfold candidates
  have row : edge 2081 = (SortedNatTree.contains FastRows.row2081) := by
    change Data.row2081.contains = (SortedNatTree.contains FastRows.row2081)
    exact FastRows.row2081_checked.symm
  rw [row]
  decide
theorem root182Step4 : scan edge (search edge 3) 4 [1103] 1103 root182Tail4 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail5 winner := by
  rw [root182Tail4, scan, if_neg (by decide), root182Candidates4_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail5
    (if 3 ≤ root182Candidates4.length then search edge 3 [1103, 2081] root182Candidates4 3184 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates5 : List Nat := [3557, 5813, 6983]
set_option maxRecDepth 8192 in
theorem root182Candidates5_checked : candidates edge 4 1103 2909 root182Tail6 winner.bound = root182Candidates5 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root182Child5 : search edge 3 [1103, 2909] root182Candidates5 4012 winner = winner := by decide
theorem root182Step5 : scan edge (search edge 3) 4 [1103] 1103 root182Tail5 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail6 winner := by
  rw [root182Tail5, scan, if_neg (by decide), root182Candidates5_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail6
    (if 3 ≤ root182Candidates5.length then search edge 3 [1103, 2909] root182Candidates5 4012 winner else winner) = _
  rw [if_pos (by decide), root182Child5]
def root182Candidates6 : List Nat := [6029, 6983, 13457]
set_option maxRecDepth 8192 in
theorem root182Candidates6_checked : candidates edge 4 1103 2927 root182Tail7 winner.bound = root182Candidates6 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root182Child6 : search edge 3 [1103, 2927] root182Candidates6 4030 winner = winner := by decide
theorem root182Step6 : scan edge (search edge 3) 4 [1103] 1103 root182Tail6 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail7 winner := by
  rw [root182Tail6, scan, if_neg (by decide), root182Candidates6_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail7
    (if 3 ≤ root182Candidates6.length then search edge 3 [1103, 2927] root182Candidates6 4030 winner else winner) = _
  rw [if_pos (by decide), root182Child6]
def root182Candidates7 : List Nat := [6317, 7529]
set_option maxRecDepth 8192 in
theorem root182Candidates7_checked : candidates edge 4 1103 3041 root182Tail8 winner.bound = root182Candidates7 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root182Step7 : scan edge (search edge 3) 4 [1103] 1103 root182Tail7 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail8 winner := by
  rw [root182Tail7, scan, if_neg (by decide), root182Candidates7_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail8
    (if 3 ≤ root182Candidates7.length then search edge 3 [1103, 3041] root182Candidates7 4144 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root182Candidates8_checked : candidates edge 4 1103 3299 root182Tail9 winner.bound = root182Candidates8 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root182Step8 : scan edge (search edge 3) 4 [1103] 1103 root182Tail8 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail9 winner := by
  rw [root182Tail8, scan, if_neg (by decide), root182Candidates8_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail9
    (if 3 ≤ root182Candidates8.length then search edge 3 [1103, 3299] root182Candidates8 4402 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates9 : List Nat := [6329, 21269]
set_option maxRecDepth 8192 in
theorem root182Candidates9_checked : candidates edge 4 1103 3557 root182Tail10 winner.bound = root182Candidates9 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root182Step9 : scan edge (search edge 3) 4 [1103] 1103 root182Tail9 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail10 winner := by
  rw [root182Tail9, scan, if_neg (by decide), root182Candidates9_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail10
    (if 3 ≤ root182Candidates9.length then search edge 3 [1103, 3557] root182Candidates9 4660 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates10 : List Nat := [12197]
set_option maxRecDepth 8192 in
theorem root182Candidates10_checked : candidates edge 4 1103 5237 root182Tail11 winner.bound = root182Candidates10 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root182Step10 : scan edge (search edge 3) 4 [1103] 1103 root182Tail10 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail11 winner := by
  rw [root182Tail10, scan, if_neg (by decide), root182Candidates10_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail11
    (if 3 ≤ root182Candidates10.length then search edge 3 [1103, 5237] root182Candidates10 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates11 : List Nat := [6329]
set_option maxRecDepth 8192 in
theorem root182Candidates11_checked : candidates edge 4 1103 5807 root182Tail12 winner.bound = root182Candidates11 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root182Step11 : scan edge (search edge 3) 4 [1103] 1103 root182Tail11 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail12 winner := by
  rw [root182Tail11, scan, if_neg (by decide), root182Candidates11_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail12
    (if 3 ≤ root182Candidates11.length then search edge 3 [1103, 5807] root182Candidates11 6910 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates12 : List Nat := [14639]
set_option maxRecDepth 8192 in
theorem root182Candidates12_checked : candidates edge 4 1103 5813 root182Tail13 winner.bound = root182Candidates12 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root182Step12 : scan edge (search edge 3) 4 [1103] 1103 root182Tail12 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail13 winner := by
  rw [root182Tail12, scan, if_neg (by decide), root182Candidates12_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail13
    (if 3 ≤ root182Candidates12.length then search edge 3 [1103, 5813] root182Candidates12 6916 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root182Candidates13_checked : candidates edge 4 1103 5987 root182Tail14 winner.bound = root182Candidates13 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root182Step13 : scan edge (search edge 3) 4 [1103] 1103 root182Tail13 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail14 winner := by
  rw [root182Tail13, scan, if_neg (by decide), root182Candidates13_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail14
    (if 3 ≤ root182Candidates13.length then search edge 3 [1103, 5987] root182Candidates13 7090 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates14 : List Nat := [7529]
set_option maxRecDepth 8192 in
theorem root182Candidates14_checked : candidates edge 4 1103 6029 root182Tail15 winner.bound = root182Candidates14 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root182Step14 : scan edge (search edge 3) 4 [1103] 1103 root182Tail14 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail15 winner := by
  rw [root182Tail14, scan, if_neg (by decide), root182Candidates14_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail15
    (if 3 ≤ root182Candidates14.length then search edge 3 [1103, 6029] root182Candidates14 7132 winner else winner) = _
  rw [if_neg (by decide)]
def root182Candidates15 : List Nat := [14639]
set_option maxRecDepth 8192 in
theorem root182Candidates15_checked : candidates edge 4 1103 6143 root182Tail16 winner.bound = root182Candidates15 := by
  unfold candidates
  have row : edge 6143 = (SortedNatTree.contains FastRows.row6143) := by
    change Data.row6143.contains = (SortedNatTree.contains FastRows.row6143)
    exact FastRows.row6143_checked.symm
  rw [row]
  decide
theorem root182Step15 : scan edge (search edge 3) 4 [1103] 1103 root182Tail15 winner =
    scan edge (search edge 3) 4 [1103] 1103 root182Tail16 winner := by
  rw [root182Tail15, scan, if_neg (by decide), root182Candidates15_checked]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail16
    (if 3 ≤ root182Candidates15.length then search edge 3 [1103, 6143] root182Candidates15 7246 winner else winner) = _
  rw [if_neg (by decide)]
theorem root182Scan16 : scan edge (search edge 3) 4 [1103] 1103 root182Tail16 winner = winner := by rw [root182Tail16, scan, if_pos (by decide)]
theorem root182Scan15 : scan edge (search edge 3) 4 [1103] 1103 root182Tail15 winner = winner :=
  (root182Step15).trans root182Scan16
theorem root182Scan14 : scan edge (search edge 3) 4 [1103] 1103 root182Tail14 winner = winner :=
  (root182Step14).trans root182Scan15
theorem root182Scan13 : scan edge (search edge 3) 4 [1103] 1103 root182Tail13 winner = winner :=
  (root182Step13).trans root182Scan14
theorem root182Scan12 : scan edge (search edge 3) 4 [1103] 1103 root182Tail12 winner = winner :=
  (root182Step12).trans root182Scan13
theorem root182Scan11 : scan edge (search edge 3) 4 [1103] 1103 root182Tail11 winner = winner :=
  (root182Step11).trans root182Scan12
theorem root182Scan10 : scan edge (search edge 3) 4 [1103] 1103 root182Tail10 winner = winner :=
  (root182Step10).trans root182Scan11
theorem root182Scan9 : scan edge (search edge 3) 4 [1103] 1103 root182Tail9 winner = winner :=
  (root182Step9).trans root182Scan10
theorem root182Scan8 : scan edge (search edge 3) 4 [1103] 1103 root182Tail8 winner = winner :=
  (root182Step8).trans root182Scan9
theorem root182Scan7 : scan edge (search edge 3) 4 [1103] 1103 root182Tail7 winner = winner :=
  (root182Step7).trans root182Scan8
theorem root182Scan6 : scan edge (search edge 3) 4 [1103] 1103 root182Tail6 winner = winner :=
  (root182Step6).trans root182Scan7
theorem root182Scan5 : scan edge (search edge 3) 4 [1103] 1103 root182Tail5 winner = winner :=
  (root182Step5).trans root182Scan6
theorem root182Scan4 : scan edge (search edge 3) 4 [1103] 1103 root182Tail4 winner = winner :=
  (root182Step4).trans root182Scan5
theorem root182Scan3 : scan edge (search edge 3) 4 [1103] 1103 root182Tail3 winner = winner :=
  (root182Step3).trans root182Scan4
theorem root182Scan2 : scan edge (search edge 3) 4 [1103] 1103 root182Tail2 winner = winner :=
  (root182Step2).trans root182Scan3
theorem root182Scan1 : scan edge (search edge 3) 4 [1103] 1103 root182Tail1 winner = winner :=
  (root182Step1).trans root182Scan2
theorem root182Scan0 : scan edge (search edge 3) 4 [1103] 1103 root182Tail0 winner = winner :=
  (root182Step0).trans root182Scan1
set_option maxRecDepth 4096 in
theorem root182Branch : search edge 4 [1103] root182Candidates 1103 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1103] 1103 root182Tail0 winner = winner
  exact root182Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
