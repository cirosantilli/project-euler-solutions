import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root154Candidates : List Nat := Data.row919
set_option maxRecDepth 8192 in
theorem root154Candidates_checked : candidates edge 5 0 919 tail155 winner.bound = root154Candidates := by
  apply candidates_eq_row edge 5 0 919 tail155 root154Candidates winner.bound tail155_ordered
  · decide
  · decide
  · funext q
    rfl
def root154Tail52 : List Nat := []
def root154Tail51 : List Nat := 24181 :: root154Tail52
def root154Tail50 : List Nat := 24121 :: root154Tail51
def root154Tail49 : List Nat := 23899 :: root154Tail50
def root154Tail48 : List Nat := 23719 :: root154Tail49
def root154Tail47 : List Nat := 23509 :: root154Tail48
def root154Tail46 : List Nat := 22807 :: root154Tail47
def root154Tail45 : List Nat := 22189 :: root154Tail46
def root154Tail44 : List Nat := 21559 :: root154Tail45
def root154Tail43 : List Nat := 21481 :: root154Tail44
def root154Tail42 : List Nat := 21187 :: root154Tail43
def root154Tail41 : List Nat := 21157 :: root154Tail42
def root154Tail40 : List Nat := 20509 :: root154Tail41
def root154Tail39 : List Nat := 19603 :: root154Tail40
def root154Tail38 : List Nat := 18481 :: root154Tail39
def root154Tail37 : List Nat := 18199 :: root154Tail38
def root154Tail36 : List Nat := 17851 :: root154Tail37
def root154Tail35 : List Nat := 17209 :: root154Tail36
def root154Tail34 : List Nat := 17191 :: root154Tail35
def root154Tail33 : List Nat := 16729 :: root154Tail34
def root154Tail32 : List Nat := 14437 :: root154Tail33
def root154Tail31 : List Nat := 14293 :: root154Tail32
def root154Tail30 : List Nat := 13933 :: root154Tail31
def root154Tail29 : List Nat := 13597 :: root154Tail30
def root154Tail28 : List Nat := 12109 :: root154Tail29
def root154Tail27 : List Nat := 12097 :: root154Tail28
def root154Tail26 : List Nat := 11467 :: root154Tail27
def root154Tail25 : List Nat := 11047 :: root154Tail26
def root154Tail24 : List Nat := 10459 :: root154Tail25
def root154Tail23 : List Nat := 10099 :: root154Tail24
def root154Tail22 : List Nat := 9661 :: root154Tail23
def root154Tail21 : List Nat := 9649 :: root154Tail22
def root154Tail20 : List Nat := 7207 :: root154Tail21
def root154Tail19 : List Nat := 7039 :: root154Tail20
def root154Tail18 : List Nat := 6961 :: root154Tail19
def root154Tail17 : List Nat := 6679 :: root154Tail18
def root154Tail16 : List Nat := 5881 :: root154Tail17
def root154Tail15 : List Nat := 5839 :: root154Tail16
def root154Tail14 : List Nat := 5821 :: root154Tail15
def root154Tail13 : List Nat := 5167 :: root154Tail14
def root154Tail12 : List Nat := 4597 :: root154Tail13
def root154Tail11 : List Nat := 4567 :: root154Tail12
def root154Tail10 : List Nat := 4441 :: root154Tail11
def root154Tail9 : List Nat := 4231 :: root154Tail10
def root154Tail8 : List Nat := 3643 :: root154Tail9
def root154Tail7 : List Nat := 3463 :: root154Tail8
def root154Tail6 : List Nat := 2749 :: root154Tail7
def root154Tail5 : List Nat := 2017 :: root154Tail6
def root154Tail4 : List Nat := 1627 :: root154Tail5
def root154Tail3 : List Nat := 1279 :: root154Tail4
def root154Tail2 : List Nat := 1201 :: root154Tail3
def root154Tail1 : List Nat := 1069 :: root154Tail2
def root154Tail0 : List Nat := 937 :: root154Tail1
def root154Candidates0 : List Nat := [22807, 23719, 23899]
set_option maxRecDepth 8192 in
theorem root154Candidates0_checked : candidates edge 4 919 937 root154Tail1 winner.bound = root154Candidates0 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root154Child0 : search edge 3 [919, 937] root154Candidates0 1856 winner = winner := by decide
theorem root154Step0 : scan edge (search edge 3) 4 [919] 919 root154Tail0 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail1 winner := by
  rw [root154Tail0, scan, if_neg (by decide), root154Candidates0_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail1
    (if 3 ≤ root154Candidates0.length then search edge 3 [919, 937] root154Candidates0 1856 winner else winner) = _
  rw [if_pos (by decide), root154Child0]
def root154Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates1_checked : candidates edge 4 919 1069 root154Tail2 winner.bound = root154Candidates1 := by
  unfold candidates
  have row : edge 1069 = (SortedNatTree.contains FastRows.row1069) := by
    change Data.row1069.contains = (SortedNatTree.contains FastRows.row1069)
    exact FastRows.row1069_checked.symm
  rw [row]
  decide
theorem root154Step1 : scan edge (search edge 3) 4 [919] 919 root154Tail1 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail2 winner := by
  rw [root154Tail1, scan, if_neg (by decide), root154Candidates1_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail2
    (if 3 ≤ root154Candidates1.length then search edge 3 [919, 1069] root154Candidates1 1988 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates2 : List Nat := [2017, 5167, 7207]
set_option maxRecDepth 8192 in
theorem root154Candidates2_checked : candidates edge 4 919 1201 root154Tail3 winner.bound = root154Candidates2 := by
  unfold candidates
  have row : edge 1201 = (SortedNatTree.contains FastRows.row1201) := by
    change Data.row1201.contains = (SortedNatTree.contains FastRows.row1201)
    exact FastRows.row1201_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root154Child2 : search edge 3 [919, 1201] root154Candidates2 2120 winner = winner := by decide
theorem root154Step2 : scan edge (search edge 3) 4 [919] 919 root154Tail2 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail3 winner := by
  rw [root154Tail2, scan, if_neg (by decide), root154Candidates2_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail3
    (if 3 ≤ root154Candidates2.length then search edge 3 [919, 1201] root154Candidates2 2120 winner else winner) = _
  rw [if_pos (by decide), root154Child2]
def root154Candidates3 : List Nat := [3463, 18199]
set_option maxRecDepth 8192 in
theorem root154Candidates3_checked : candidates edge 4 919 1279 root154Tail4 winner.bound = root154Candidates3 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
theorem root154Step3 : scan edge (search edge 3) 4 [919] 919 root154Tail3 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail4 winner := by
  rw [root154Tail3, scan, if_neg (by decide), root154Candidates3_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail4
    (if 3 ≤ root154Candidates3.length then search edge 3 [919, 1279] root154Candidates3 2198 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates4 : List Nat := [3463, 5821, 6961]
set_option maxRecDepth 8192 in
theorem root154Candidates4_checked : candidates edge 4 919 1627 root154Tail5 winner.bound = root154Candidates4 := by
  unfold candidates
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root154Child4 : search edge 3 [919, 1627] root154Candidates4 2546 winner = winner := by decide
theorem root154Step4 : scan edge (search edge 3) 4 [919] 919 root154Tail4 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail5 winner := by
  rw [root154Tail4, scan, if_neg (by decide), root154Candidates4_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail5
    (if 3 ≤ root154Candidates4.length then search edge 3 [919, 1627] root154Candidates4 2546 winner else winner) = _
  rw [if_pos (by decide), root154Child4]
def root154Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates5_checked : candidates edge 4 919 2017 root154Tail6 winner.bound = root154Candidates5 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root154Step5 : scan edge (search edge 3) 4 [919] 919 root154Tail5 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail6 winner := by
  rw [root154Tail5, scan, if_neg (by decide), root154Candidates5_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail6
    (if 3 ≤ root154Candidates5.length then search edge 3 [919, 2017] root154Candidates5 2936 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates6 : List Nat := [3643, 11467, 17191]
set_option maxRecDepth 8192 in
theorem root154Candidates6_checked : candidates edge 4 919 2749 root154Tail7 winner.bound = root154Candidates6 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root154Child6 : search edge 3 [919, 2749] root154Candidates6 3668 winner = winner := by decide
theorem root154Step6 : scan edge (search edge 3) 4 [919] 919 root154Tail6 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail7 winner := by
  rw [root154Tail6, scan, if_neg (by decide), root154Candidates6_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail7
    (if 3 ≤ root154Candidates6.length then search edge 3 [919, 2749] root154Candidates6 3668 winner else winner) = _
  rw [if_pos (by decide), root154Child6]
def root154Candidates7 : List Nat := [10099]
set_option maxRecDepth 8192 in
theorem root154Candidates7_checked : candidates edge 4 919 3463 root154Tail8 winner.bound = root154Candidates7 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root154Step7 : scan edge (search edge 3) 4 [919] 919 root154Tail7 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail8 winner := by
  rw [root154Tail7, scan, if_neg (by decide), root154Candidates7_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail8
    (if 3 ≤ root154Candidates7.length then search edge 3 [919, 3463] root154Candidates7 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates8 : List Nat := [21157, 21187]
set_option maxRecDepth 8192 in
theorem root154Candidates8_checked : candidates edge 4 919 3643 root154Tail9 winner.bound = root154Candidates8 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root154Step8 : scan edge (search edge 3) 4 [919] 919 root154Tail8 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail9 winner := by
  rw [root154Tail8, scan, if_neg (by decide), root154Candidates8_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail9
    (if 3 ≤ root154Candidates8.length then search edge 3 [919, 3643] root154Candidates8 4562 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates9_checked : candidates edge 4 919 4231 root154Tail10 winner.bound = root154Candidates9 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root154Step9 : scan edge (search edge 3) 4 [919] 919 root154Tail9 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail10 winner := by
  rw [root154Tail9, scan, if_neg (by decide), root154Candidates9_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail10
    (if 3 ≤ root154Candidates9.length then search edge 3 [919, 4231] root154Candidates9 5150 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates10_checked : candidates edge 4 919 4441 root154Tail11 winner.bound = root154Candidates10 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root154Step10 : scan edge (search edge 3) 4 [919] 919 root154Tail10 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail11 winner := by
  rw [root154Tail10, scan, if_neg (by decide), root154Candidates10_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail11
    (if 3 ≤ root154Candidates10.length then search edge 3 [919, 4441] root154Candidates10 5360 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates11 : List Nat := [7039, 14437]
set_option maxRecDepth 8192 in
theorem root154Candidates11_checked : candidates edge 4 919 4567 root154Tail12 winner.bound = root154Candidates11 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root154Step11 : scan edge (search edge 3) 4 [919] 919 root154Tail11 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail12 winner := by
  rw [root154Tail11, scan, if_neg (by decide), root154Candidates11_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail12
    (if 3 ≤ root154Candidates11.length then search edge 3 [919, 4567] root154Candidates11 5486 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates12_checked : candidates edge 4 919 4597 root154Tail13 winner.bound = root154Candidates12 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root154Step12 : scan edge (search edge 3) 4 [919] 919 root154Tail12 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail13 winner := by
  rw [root154Tail12, scan, if_neg (by decide), root154Candidates12_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail13
    (if 3 ≤ root154Candidates12.length then search edge 3 [919, 4597] root154Candidates12 5516 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates13 : List Nat := [10459]
set_option maxRecDepth 8192 in
theorem root154Candidates13_checked : candidates edge 4 919 5167 root154Tail14 winner.bound = root154Candidates13 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root154Step13 : scan edge (search edge 3) 4 [919] 919 root154Tail13 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail14 winner := by
  rw [root154Tail13, scan, if_neg (by decide), root154Candidates13_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail14
    (if 3 ≤ root154Candidates13.length then search edge 3 [919, 5167] root154Candidates13 6086 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates14_checked : candidates edge 4 919 5821 root154Tail15 winner.bound = root154Candidates14 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root154Step14 : scan edge (search edge 3) 4 [919] 919 root154Tail14 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail15 winner := by
  rw [root154Tail14, scan, if_neg (by decide), root154Candidates14_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail15
    (if 3 ≤ root154Candidates14.length then search edge 3 [919, 5821] root154Candidates14 6740 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates15 : List Nat := [11047]
set_option maxRecDepth 8192 in
theorem root154Candidates15_checked : candidates edge 4 919 5839 root154Tail16 winner.bound = root154Candidates15 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root154Step15 : scan edge (search edge 3) 4 [919] 919 root154Tail15 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail16 winner := by
  rw [root154Tail15, scan, if_neg (by decide), root154Candidates15_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail16
    (if 3 ≤ root154Candidates15.length then search edge 3 [919, 5839] root154Candidates15 6758 winner else winner) = _
  rw [if_neg (by decide)]
def root154Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root154Candidates16_checked : candidates edge 4 919 5881 root154Tail17 winner.bound = root154Candidates16 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root154Step16 : scan edge (search edge 3) 4 [919] 919 root154Tail16 winner =
    scan edge (search edge 3) 4 [919] 919 root154Tail17 winner := by
  rw [root154Tail16, scan, if_neg (by decide), root154Candidates16_checked]
  change scan edge (search edge 3) 4 [919] 919 root154Tail17
    (if 3 ≤ root154Candidates16.length then search edge 3 [919, 5881] root154Candidates16 6800 winner else winner) = _
  rw [if_neg (by decide)]
theorem root154Scan17 : scan edge (search edge 3) 4 [919] 919 root154Tail17 winner = winner := by rw [root154Tail17, scan, if_pos (by decide)]
theorem root154Scan16 : scan edge (search edge 3) 4 [919] 919 root154Tail16 winner = winner :=
  (root154Step16).trans root154Scan17
theorem root154Scan15 : scan edge (search edge 3) 4 [919] 919 root154Tail15 winner = winner :=
  (root154Step15).trans root154Scan16
theorem root154Scan14 : scan edge (search edge 3) 4 [919] 919 root154Tail14 winner = winner :=
  (root154Step14).trans root154Scan15
theorem root154Scan13 : scan edge (search edge 3) 4 [919] 919 root154Tail13 winner = winner :=
  (root154Step13).trans root154Scan14
theorem root154Scan12 : scan edge (search edge 3) 4 [919] 919 root154Tail12 winner = winner :=
  (root154Step12).trans root154Scan13
theorem root154Scan11 : scan edge (search edge 3) 4 [919] 919 root154Tail11 winner = winner :=
  (root154Step11).trans root154Scan12
theorem root154Scan10 : scan edge (search edge 3) 4 [919] 919 root154Tail10 winner = winner :=
  (root154Step10).trans root154Scan11
theorem root154Scan9 : scan edge (search edge 3) 4 [919] 919 root154Tail9 winner = winner :=
  (root154Step9).trans root154Scan10
theorem root154Scan8 : scan edge (search edge 3) 4 [919] 919 root154Tail8 winner = winner :=
  (root154Step8).trans root154Scan9
theorem root154Scan7 : scan edge (search edge 3) 4 [919] 919 root154Tail7 winner = winner :=
  (root154Step7).trans root154Scan8
theorem root154Scan6 : scan edge (search edge 3) 4 [919] 919 root154Tail6 winner = winner :=
  (root154Step6).trans root154Scan7
theorem root154Scan5 : scan edge (search edge 3) 4 [919] 919 root154Tail5 winner = winner :=
  (root154Step5).trans root154Scan6
theorem root154Scan4 : scan edge (search edge 3) 4 [919] 919 root154Tail4 winner = winner :=
  (root154Step4).trans root154Scan5
theorem root154Scan3 : scan edge (search edge 3) 4 [919] 919 root154Tail3 winner = winner :=
  (root154Step3).trans root154Scan4
theorem root154Scan2 : scan edge (search edge 3) 4 [919] 919 root154Tail2 winner = winner :=
  (root154Step2).trans root154Scan3
theorem root154Scan1 : scan edge (search edge 3) 4 [919] 919 root154Tail1 winner = winner :=
  (root154Step1).trans root154Scan2
theorem root154Scan0 : scan edge (search edge 3) 4 [919] 919 root154Tail0 winner = winner :=
  (root154Step0).trans root154Scan1
set_option maxRecDepth 4096 in
theorem root154Branch : search edge 4 [919] root154Candidates 919 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [919] 919 root154Tail0 winner = winner
  exact root154Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
