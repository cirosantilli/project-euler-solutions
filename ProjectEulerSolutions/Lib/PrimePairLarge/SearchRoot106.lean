import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root106Candidates : List Nat := Data.row599
set_option maxRecDepth 8192 in
theorem root106Candidates_checked : candidates edge 5 0 599 tail107 winner.bound = root106Candidates := by
  apply candidates_eq_row edge 5 0 599 tail107 root106Candidates winner.bound tail107_ordered
  · decide
  · decide
  · funext q
    rfl
def root106Tail61 : List Nat := []
def root106Tail60 : List Nat := 25367 :: root106Tail61
def root106Tail59 : List Nat := 24989 :: root106Tail60
def root106Tail58 : List Nat := 24971 :: root106Tail59
def root106Tail57 : List Nat := 23627 :: root106Tail58
def root106Tail56 : List Nat := 23333 :: root106Tail57
def root106Tail55 : List Nat := 23057 :: root106Tail56
def root106Tail54 : List Nat := 22871 :: root106Tail55
def root106Tail53 : List Nat := 21803 :: root106Tail54
def root106Tail52 : List Nat := 21023 :: root106Tail53
def root106Tail51 : List Nat := 20261 :: root106Tail52
def root106Tail50 : List Nat := 20249 :: root106Tail51
def root106Tail49 : List Nat := 20183 :: root106Tail50
def root106Tail48 : List Nat := 19697 :: root106Tail49
def root106Tail47 : List Nat := 19469 :: root106Tail48
def root106Tail46 : List Nat := 19463 :: root106Tail47
def root106Tail45 : List Nat := 19403 :: root106Tail46
def root106Tail44 : List Nat := 19259 :: root106Tail45
def root106Tail43 : List Nat := 17807 :: root106Tail44
def root106Tail42 : List Nat := 17489 :: root106Tail43
def root106Tail41 : List Nat := 17477 :: root106Tail42
def root106Tail40 : List Nat := 17387 :: root106Tail41
def root106Tail39 : List Nat := 16943 :: root106Tail40
def root106Tail38 : List Nat := 16889 :: root106Tail39
def root106Tail37 : List Nat := 15923 :: root106Tail38
def root106Tail36 : List Nat := 15881 :: root106Tail37
def root106Tail35 : List Nat := 15773 :: root106Tail36
def root106Tail34 : List Nat := 15629 :: root106Tail35
def root106Tail33 : List Nat := 15287 :: root106Tail34
def root106Tail32 : List Nat := 15269 :: root106Tail33
def root106Tail31 : List Nat := 14051 :: root106Tail32
def root106Tail30 : List Nat := 12959 :: root106Tail31
def root106Tail29 : List Nat := 12917 :: root106Tail30
def root106Tail28 : List Nat := 12473 :: root106Tail29
def root106Tail27 : List Nat := 12227 :: root106Tail28
def root106Tail26 : List Nat := 12071 :: root106Tail27
def root106Tail25 : List Nat := 11279 :: root106Tail26
def root106Tail24 : List Nat := 10709 :: root106Tail25
def root106Tail23 : List Nat := 10247 :: root106Tail24
def root106Tail22 : List Nat := 9551 :: root106Tail23
def root106Tail21 : List Nat := 9479 :: root106Tail22
def root106Tail20 : List Nat := 9341 :: root106Tail21
def root106Tail19 : List Nat := 8039 :: root106Tail20
def root106Tail18 : List Nat := 7883 :: root106Tail19
def root106Tail17 : List Nat := 7877 :: root106Tail18
def root106Tail16 : List Nat := 7841 :: root106Tail17
def root106Tail15 : List Nat := 7517 :: root106Tail16
def root106Tail14 : List Nat := 6047 :: root106Tail15
def root106Tail13 : List Nat := 5573 :: root106Tail14
def root106Tail12 : List Nat := 5399 :: root106Tail13
def root106Tail11 : List Nat := 5309 :: root106Tail12
def root106Tail10 : List Nat := 5081 :: root106Tail11
def root106Tail9 : List Nat := 4091 :: root106Tail10
def root106Tail8 : List Nat := 3761 :: root106Tail9
def root106Tail7 : List Nat := 3671 :: root106Tail8
def root106Tail6 : List Nat := 3557 :: root106Tail7
def root106Tail5 : List Nat := 3023 :: root106Tail6
def root106Tail4 : List Nat := 1907 :: root106Tail5
def root106Tail3 : List Nat := 1823 :: root106Tail4
def root106Tail2 : List Nat := 1709 :: root106Tail3
def root106Tail1 : List Nat := 941 :: root106Tail2
def root106Tail0 : List Nat := 719 :: root106Tail1
def root106Candidates0 : List Nat := [19697, 23057]
set_option maxRecDepth 8192 in
theorem root106Candidates0_checked : candidates edge 4 599 719 root106Tail1 winner.bound = root106Candidates0 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
theorem root106Step0 : scan edge (search edge 3) 4 [599] 599 root106Tail0 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail1 winner := by
  rw [root106Tail0, scan, if_neg (by decide), root106Candidates0_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail1
    (if 3 ≤ root106Candidates0.length then search edge 3 [599, 719] root106Candidates0 1318 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates1 : List Nat := [1709, 6047, 12959, 21803, 23333]
set_option maxRecDepth 8192 in
theorem root106Candidates1_checked : candidates edge 4 599 941 root106Tail2 winner.bound = root106Candidates1 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root106Child1 : search edge 3 [599, 941] root106Candidates1 1540 winner = winner := by decide
theorem root106Step1 : scan edge (search edge 3) 4 [599] 599 root106Tail1 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail2 winner := by
  rw [root106Tail1, scan, if_neg (by decide), root106Candidates1_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail2
    (if 3 ≤ root106Candidates1.length then search edge 3 [599, 941] root106Candidates1 1540 winner else winner) = _
  rw [if_pos (by decide), root106Child1]
def root106Candidates2 : List Nat := [7877]
set_option maxRecDepth 8192 in
theorem root106Candidates2_checked : candidates edge 4 599 1709 root106Tail3 winner.bound = root106Candidates2 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
theorem root106Step2 : scan edge (search edge 3) 4 [599] 599 root106Tail2 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail3 winner := by
  rw [root106Tail2, scan, if_neg (by decide), root106Candidates2_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail3
    (if 3 ≤ root106Candidates2.length then search edge 3 [599, 1709] root106Candidates2 2308 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates3 : List Nat := [3023, 9341, 15629, 19469]
set_option maxRecDepth 8192 in
theorem root106Candidates3_checked : candidates edge 4 599 1823 root106Tail4 winner.bound = root106Candidates3 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root106Child3 : search edge 3 [599, 1823] root106Candidates3 2422 winner = winner := by decide
theorem root106Step3 : scan edge (search edge 3) 4 [599] 599 root106Tail3 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail4 winner := by
  rw [root106Tail3, scan, if_neg (by decide), root106Candidates3_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail4
    (if 3 ≤ root106Candidates3.length then search edge 3 [599, 1823] root106Candidates3 2422 winner else winner) = _
  rw [if_pos (by decide), root106Child3]
def root106Candidates4 : List Nat := [9341, 12227, 17489, 20183]
set_option maxRecDepth 8192 in
theorem root106Candidates4_checked : candidates edge 4 599 1907 root106Tail5 winner.bound = root106Candidates4 := by
  unfold candidates
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root106Child4 : search edge 3 [599, 1907] root106Candidates4 2506 winner = winner := by decide
theorem root106Step4 : scan edge (search edge 3) 4 [599] 599 root106Tail4 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail5 winner := by
  rw [root106Tail4, scan, if_neg (by decide), root106Candidates4_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail5
    (if 3 ≤ root106Candidates4.length then search edge 3 [599, 1907] root106Candidates4 2506 winner else winner) = _
  rw [if_pos (by decide), root106Child4]
def root106Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root106Candidates5_checked : candidates edge 4 599 3023 root106Tail6 winner.bound = root106Candidates5 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root106Step5 : scan edge (search edge 3) 4 [599] 599 root106Tail5 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail6 winner := by
  rw [root106Tail5, scan, if_neg (by decide), root106Candidates5_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail6
    (if 3 ≤ root106Candidates5.length then search edge 3 [599, 3023] root106Candidates5 3622 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root106Candidates6_checked : candidates edge 4 599 3557 root106Tail7 winner.bound = root106Candidates6 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root106Step6 : scan edge (search edge 3) 4 [599] 599 root106Tail6 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail7 winner := by
  rw [root106Tail6, scan, if_neg (by decide), root106Candidates6_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail7
    (if 3 ≤ root106Candidates6.length then search edge 3 [599, 3557] root106Candidates6 4156 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates7 : List Nat := [4091]
set_option maxRecDepth 8192 in
theorem root106Candidates7_checked : candidates edge 4 599 3671 root106Tail8 winner.bound = root106Candidates7 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root106Step7 : scan edge (search edge 3) 4 [599] 599 root106Tail7 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail8 winner := by
  rw [root106Tail7, scan, if_neg (by decide), root106Candidates7_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail8
    (if 3 ≤ root106Candidates7.length then search edge 3 [599, 3671] root106Candidates7 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates8 : List Nat := [19259]
set_option maxRecDepth 8192 in
theorem root106Candidates8_checked : candidates edge 4 599 3761 root106Tail9 winner.bound = root106Candidates8 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root106Step8 : scan edge (search edge 3) 4 [599] 599 root106Tail8 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail9 winner := by
  rw [root106Tail8, scan, if_neg (by decide), root106Candidates8_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail9
    (if 3 ≤ root106Candidates8.length then search edge 3 [599, 3761] root106Candidates8 4360 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root106Candidates9_checked : candidates edge 4 599 4091 root106Tail10 winner.bound = root106Candidates9 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root106Step9 : scan edge (search edge 3) 4 [599] 599 root106Tail9 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail10 winner := by
  rw [root106Tail9, scan, if_neg (by decide), root106Candidates9_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail10
    (if 3 ≤ root106Candidates9.length then search edge 3 [599, 4091] root106Candidates9 4690 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root106Candidates10_checked : candidates edge 4 599 5081 root106Tail11 winner.bound = root106Candidates10 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root106Step10 : scan edge (search edge 3) 4 [599] 599 root106Tail10 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail11 winner := by
  rw [root106Tail10, scan, if_neg (by decide), root106Candidates10_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail11
    (if 3 ≤ root106Candidates10.length then search edge 3 [599, 5081] root106Candidates10 5680 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root106Candidates11_checked : candidates edge 4 599 5309 root106Tail12 winner.bound = root106Candidates11 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root106Step11 : scan edge (search edge 3) 4 [599] 599 root106Tail11 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail12 winner := by
  rw [root106Tail11, scan, if_neg (by decide), root106Candidates11_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail12
    (if 3 ≤ root106Candidates11.length then search edge 3 [599, 5309] root106Candidates11 5908 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates12 : List Nat := [15269, 15287, 15773]
set_option maxRecDepth 8192 in
theorem root106Candidates12_checked : candidates edge 4 599 5399 root106Tail13 winner.bound = root106Candidates12 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root106Child12 : search edge 3 [599, 5399] root106Candidates12 5998 winner = winner := by decide
theorem root106Step12 : scan edge (search edge 3) 4 [599] 599 root106Tail12 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail13 winner := by
  rw [root106Tail12, scan, if_neg (by decide), root106Candidates12_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail13
    (if 3 ≤ root106Candidates12.length then search edge 3 [599, 5399] root106Candidates12 5998 winner else winner) = _
  rw [if_pos (by decide), root106Child12]
def root106Candidates13 : List Nat := [17489]
set_option maxRecDepth 8192 in
theorem root106Candidates13_checked : candidates edge 4 599 5573 root106Tail14 winner.bound = root106Candidates13 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root106Step13 : scan edge (search edge 3) 4 [599] 599 root106Tail13 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail14 winner := by
  rw [root106Tail13, scan, if_neg (by decide), root106Candidates13_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail14
    (if 3 ≤ root106Candidates13.length then search edge 3 [599, 5573] root106Candidates13 6172 winner else winner) = _
  rw [if_neg (by decide)]
def root106Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root106Candidates14_checked : candidates edge 4 599 6047 root106Tail15 winner.bound = root106Candidates14 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root106Step14 : scan edge (search edge 3) 4 [599] 599 root106Tail14 winner =
    scan edge (search edge 3) 4 [599] 599 root106Tail15 winner := by
  rw [root106Tail14, scan, if_neg (by decide), root106Candidates14_checked]
  change scan edge (search edge 3) 4 [599] 599 root106Tail15
    (if 3 ≤ root106Candidates14.length then search edge 3 [599, 6047] root106Candidates14 6646 winner else winner) = _
  rw [if_neg (by decide)]
theorem root106Scan15 : scan edge (search edge 3) 4 [599] 599 root106Tail15 winner = winner := by rw [root106Tail15, scan, if_pos (by decide)]
theorem root106Scan14 : scan edge (search edge 3) 4 [599] 599 root106Tail14 winner = winner :=
  (root106Step14).trans root106Scan15
theorem root106Scan13 : scan edge (search edge 3) 4 [599] 599 root106Tail13 winner = winner :=
  (root106Step13).trans root106Scan14
theorem root106Scan12 : scan edge (search edge 3) 4 [599] 599 root106Tail12 winner = winner :=
  (root106Step12).trans root106Scan13
theorem root106Scan11 : scan edge (search edge 3) 4 [599] 599 root106Tail11 winner = winner :=
  (root106Step11).trans root106Scan12
theorem root106Scan10 : scan edge (search edge 3) 4 [599] 599 root106Tail10 winner = winner :=
  (root106Step10).trans root106Scan11
theorem root106Scan9 : scan edge (search edge 3) 4 [599] 599 root106Tail9 winner = winner :=
  (root106Step9).trans root106Scan10
theorem root106Scan8 : scan edge (search edge 3) 4 [599] 599 root106Tail8 winner = winner :=
  (root106Step8).trans root106Scan9
theorem root106Scan7 : scan edge (search edge 3) 4 [599] 599 root106Tail7 winner = winner :=
  (root106Step7).trans root106Scan8
theorem root106Scan6 : scan edge (search edge 3) 4 [599] 599 root106Tail6 winner = winner :=
  (root106Step6).trans root106Scan7
theorem root106Scan5 : scan edge (search edge 3) 4 [599] 599 root106Tail5 winner = winner :=
  (root106Step5).trans root106Scan6
theorem root106Scan4 : scan edge (search edge 3) 4 [599] 599 root106Tail4 winner = winner :=
  (root106Step4).trans root106Scan5
theorem root106Scan3 : scan edge (search edge 3) 4 [599] 599 root106Tail3 winner = winner :=
  (root106Step3).trans root106Scan4
theorem root106Scan2 : scan edge (search edge 3) 4 [599] 599 root106Tail2 winner = winner :=
  (root106Step2).trans root106Scan3
theorem root106Scan1 : scan edge (search edge 3) 4 [599] 599 root106Tail1 winner = winner :=
  (root106Step1).trans root106Scan2
theorem root106Scan0 : scan edge (search edge 3) 4 [599] 599 root106Tail0 winner = winner :=
  (root106Step0).trans root106Scan1
set_option maxRecDepth 4096 in
theorem root106Branch : search edge 4 [599] root106Candidates 599 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [599] 599 root106Tail0 winner = winner
  exact root106Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
