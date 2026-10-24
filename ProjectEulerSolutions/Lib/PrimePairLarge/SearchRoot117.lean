import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root117Candidates : List Nat := Data.row659
set_option maxRecDepth 8192 in
theorem root117Candidates_checked : candidates edge 5 0 659 tail118 winner.bound = root117Candidates := by
  apply candidates_eq_row edge 5 0 659 tail118 root117Candidates winner.bound tail118_ordered
  · decide
  · decide
  · funext q
    rfl
def root117Tail57 : List Nat := []
def root117Tail56 : List Nat := 25367 :: root117Tail57
def root117Tail55 : List Nat := 25121 :: root117Tail56
def root117Tail54 : List Nat := 24077 :: root117Tail55
def root117Tail53 : List Nat := 23567 :: root117Tail54
def root117Tail52 : List Nat := 23537 :: root117Tail53
def root117Tail51 : List Nat := 22697 :: root117Tail52
def root117Tail50 : List Nat := 22643 :: root117Tail51
def root117Tail49 : List Nat := 22511 :: root117Tail50
def root117Tail48 : List Nat := 22013 :: root117Tail49
def root117Tail47 : List Nat := 21929 :: root117Tail48
def root117Tail46 : List Nat := 21893 :: root117Tail47
def root117Tail45 : List Nat := 21269 :: root117Tail46
def root117Tail44 : List Nat := 20123 :: root117Tail45
def root117Tail43 : List Nat := 18233 :: root117Tail44
def root117Tail42 : List Nat := 18143 :: root117Tail43
def root117Tail41 : List Nat := 17987 :: root117Tail42
def root117Tail40 : List Nat := 17807 :: root117Tail41
def root117Tail39 : List Nat := 17519 :: root117Tail40
def root117Tail38 : List Nat := 16547 :: root117Tail39
def root117Tail37 : List Nat := 15887 :: root117Tail38
def root117Tail36 : List Nat := 15797 :: root117Tail37
def root117Tail35 : List Nat := 15053 :: root117Tail36
def root117Tail34 : List Nat := 14327 :: root117Tail35
def root117Tail33 : List Nat := 14243 :: root117Tail34
def root117Tail32 : List Nat := 14087 :: root117Tail33
def root117Tail31 : List Nat := 14081 :: root117Tail32
def root117Tail30 : List Nat := 13751 :: root117Tail31
def root117Tail29 : List Nat := 12527 :: root117Tail30
def root117Tail28 : List Nat := 12101 :: root117Tail29
def root117Tail27 : List Nat := 11981 :: root117Tail28
def root117Tail26 : List Nat := 11939 :: root117Tail27
def root117Tail25 : List Nat := 11777 :: root117Tail26
def root117Tail24 : List Nat := 10007 :: root117Tail25
def root117Tail23 : List Nat := 9839 :: root117Tail24
def root117Tail22 : List Nat := 9743 :: root117Tail23
def root117Tail21 : List Nat := 9221 :: root117Tail22
def root117Tail20 : List Nat := 8951 :: root117Tail21
def root117Tail19 : List Nat := 8819 :: root117Tail20
def root117Tail18 : List Nat := 8087 :: root117Tail19
def root117Tail17 : List Nat := 7583 :: root117Tail18
def root117Tail16 : List Nat := 7517 :: root117Tail17
def root117Tail15 : List Nat := 7307 :: root117Tail16
def root117Tail14 : List Nat := 6089 :: root117Tail15
def root117Tail13 : List Nat := 5009 :: root117Tail14
def root117Tail12 : List Nat := 4799 :: root117Tail13
def root117Tail11 : List Nat := 4139 :: root117Tail12
def root117Tail10 : List Nat := 4127 :: root117Tail11
def root117Tail9 : List Nat := 3593 :: root117Tail10
def root117Tail8 : List Nat := 3407 :: root117Tail9
def root117Tail7 : List Nat := 3371 :: root117Tail8
def root117Tail6 : List Nat := 3329 :: root117Tail7
def root117Tail5 : List Nat := 3023 :: root117Tail6
def root117Tail4 : List Nat := 2741 :: root117Tail5
def root117Tail3 : List Nat := 2393 :: root117Tail4
def root117Tail2 : List Nat := 1493 :: root117Tail3
def root117Tail1 : List Nat := 983 :: root117Tail2
def root117Tail0 : List Nat := 947 :: root117Tail1
def root117Candidates0 : List Nat := [5009, 18233, 21269]
set_option maxRecDepth 8192 in
theorem root117Candidates0_checked : candidates edge 4 659 947 root117Tail1 winner.bound = root117Candidates0 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root117Child0 : search edge 3 [659, 947] root117Candidates0 1606 winner = winner := by decide
theorem root117Step0 : scan edge (search edge 3) 4 [659] 659 root117Tail0 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail1 winner := by
  rw [root117Tail0, scan, if_neg (by decide), root117Candidates0_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail1
    (if 3 ≤ root117Candidates0.length then search edge 3 [659, 947] root117Candidates0 1606 winner else winner) = _
  rw [if_pos (by decide), root117Child0]
def root117Candidates1 : List Nat := [10007]
set_option maxRecDepth 8192 in
theorem root117Candidates1_checked : candidates edge 4 659 983 root117Tail2 winner.bound = root117Candidates1 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root117Step1 : scan edge (search edge 3) 4 [659] 659 root117Tail1 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail2 winner := by
  rw [root117Tail1, scan, if_neg (by decide), root117Candidates1_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail2
    (if 3 ≤ root117Candidates1.length then search edge 3 [659, 983] root117Candidates1 1642 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates2 : List Nat := [3329, 12101]
set_option maxRecDepth 8192 in
theorem root117Candidates2_checked : candidates edge 4 659 1493 root117Tail3 winner.bound = root117Candidates2 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root117Step2 : scan edge (search edge 3) 4 [659] 659 root117Tail2 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail3 winner := by
  rw [root117Tail2, scan, if_neg (by decide), root117Candidates2_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail3
    (if 3 ≤ root117Candidates2.length then search edge 3 [659, 1493] root117Candidates2 2152 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates3 : List Nat := [11981, 14087]
set_option maxRecDepth 8192 in
theorem root117Candidates3_checked : candidates edge 4 659 2393 root117Tail4 winner.bound = root117Candidates3 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root117Step3 : scan edge (search edge 3) 4 [659] 659 root117Tail3 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail4 winner := by
  rw [root117Tail3, scan, if_neg (by decide), root117Candidates3_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail4
    (if 3 ≤ root117Candidates3.length then search edge 3 [659, 2393] root117Candidates3 3052 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates4 : List Nat := [3329, 4799, 20123]
set_option maxRecDepth 8192 in
theorem root117Candidates4_checked : candidates edge 4 659 2741 root117Tail5 winner.bound = root117Candidates4 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root117Child4 : search edge 3 [659, 2741] root117Candidates4 3400 winner = winner := by decide
theorem root117Step4 : scan edge (search edge 3) 4 [659] 659 root117Tail4 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail5 winner := by
  rw [root117Tail4, scan, if_neg (by decide), root117Candidates4_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail5
    (if 3 ≤ root117Candidates4.length then search edge 3 [659, 2741] root117Candidates4 3400 winner else winner) = _
  rw [if_pos (by decide), root117Child4]
def root117Candidates5 : List Nat := [21893]
set_option maxRecDepth 8192 in
theorem root117Candidates5_checked : candidates edge 4 659 3023 root117Tail6 winner.bound = root117Candidates5 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root117Step5 : scan edge (search edge 3) 4 [659] 659 root117Tail5 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail6 winner := by
  rw [root117Tail5, scan, if_neg (by decide), root117Candidates5_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail6
    (if 3 ≤ root117Candidates5.length then search edge 3 [659, 3023] root117Candidates5 3682 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates6 : List Nat := [4127]
set_option maxRecDepth 8192 in
theorem root117Candidates6_checked : candidates edge 4 659 3329 root117Tail7 winner.bound = root117Candidates6 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root117Step6 : scan edge (search edge 3) 4 [659] 659 root117Tail6 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail7 winner := by
  rw [root117Tail6, scan, if_neg (by decide), root117Candidates6_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail7
    (if 3 ≤ root117Candidates6.length then search edge 3 [659, 3329] root117Candidates6 3988 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates7 : List Nat := [4799, 5009, 12101]
set_option maxRecDepth 8192 in
theorem root117Candidates7_checked : candidates edge 4 659 3371 root117Tail8 winner.bound = root117Candidates7 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root117Child7 : search edge 3 [659, 3371] root117Candidates7 4030 winner = winner := by decide
theorem root117Step7 : scan edge (search edge 3) 4 [659] 659 root117Tail7 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail8 winner := by
  rw [root117Tail7, scan, if_neg (by decide), root117Candidates7_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail8
    (if 3 ≤ root117Candidates7.length then search edge 3 [659, 3371] root117Candidates7 4030 winner else winner) = _
  rw [if_pos (by decide), root117Child7]
def root117Candidates8 : List Nat := [8819, 9743, 12101]
set_option maxRecDepth 8192 in
theorem root117Candidates8_checked : candidates edge 4 659 3407 root117Tail9 winner.bound = root117Candidates8 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root117Child8 : search edge 3 [659, 3407] root117Candidates8 4066 winner = winner := by decide
theorem root117Step8 : scan edge (search edge 3) 4 [659] 659 root117Tail8 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail9 winner := by
  rw [root117Tail8, scan, if_neg (by decide), root117Candidates8_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail9
    (if 3 ≤ root117Candidates8.length then search edge 3 [659, 3407] root117Candidates8 4066 winner else winner) = _
  rw [if_pos (by decide), root117Child8]
def root117Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root117Candidates9_checked : candidates edge 4 659 3593 root117Tail10 winner.bound = root117Candidates9 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root117Step9 : scan edge (search edge 3) 4 [659] 659 root117Tail9 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail10 winner := by
  rw [root117Tail9, scan, if_neg (by decide), root117Candidates9_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail10
    (if 3 ≤ root117Candidates9.length then search edge 3 [659, 3593] root117Candidates9 4252 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates10 : List Nat := [13751]
set_option maxRecDepth 8192 in
theorem root117Candidates10_checked : candidates edge 4 659 4127 root117Tail11 winner.bound = root117Candidates10 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root117Step10 : scan edge (search edge 3) 4 [659] 659 root117Tail10 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail11 winner := by
  rw [root117Tail10, scan, if_neg (by decide), root117Candidates10_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail11
    (if 3 ≤ root117Candidates10.length then search edge 3 [659, 4127] root117Candidates10 4786 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root117Candidates11_checked : candidates edge 4 659 4139 root117Tail12 winner.bound = root117Candidates11 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root117Step11 : scan edge (search edge 3) 4 [659] 659 root117Tail11 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail12 winner := by
  rw [root117Tail11, scan, if_neg (by decide), root117Candidates11_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail12
    (if 3 ≤ root117Candidates11.length then search edge 3 [659, 4139] root117Candidates11 4798 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates12 : List Nat := [7583]
set_option maxRecDepth 8192 in
theorem root117Candidates12_checked : candidates edge 4 659 4799 root117Tail13 winner.bound = root117Candidates12 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root117Step12 : scan edge (search edge 3) 4 [659] 659 root117Tail12 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail13 winner := by
  rw [root117Tail12, scan, if_neg (by decide), root117Candidates12_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail13
    (if 3 ≤ root117Candidates12.length then search edge 3 [659, 4799] root117Candidates12 5458 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates13 : List Nat := [9839, 18233]
set_option maxRecDepth 8192 in
theorem root117Candidates13_checked : candidates edge 4 659 5009 root117Tail14 winner.bound = root117Candidates13 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root117Step13 : scan edge (search edge 3) 4 [659] 659 root117Tail13 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail14 winner := by
  rw [root117Tail13, scan, if_neg (by decide), root117Candidates13_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail14
    (if 3 ≤ root117Candidates13.length then search edge 3 [659, 5009] root117Candidates13 5668 winner else winner) = _
  rw [if_neg (by decide)]
def root117Candidates14 : List Nat := [17807]
set_option maxRecDepth 8192 in
theorem root117Candidates14_checked : candidates edge 4 659 6089 root117Tail15 winner.bound = root117Candidates14 := by
  unfold candidates
  have row : edge 6089 = (SortedNatTree.contains FastRows.row6089) := by
    change Data.row6089.contains = (SortedNatTree.contains FastRows.row6089)
    exact FastRows.row6089_checked.symm
  rw [row]
  decide
theorem root117Step14 : scan edge (search edge 3) 4 [659] 659 root117Tail14 winner =
    scan edge (search edge 3) 4 [659] 659 root117Tail15 winner := by
  rw [root117Tail14, scan, if_neg (by decide), root117Candidates14_checked]
  change scan edge (search edge 3) 4 [659] 659 root117Tail15
    (if 3 ≤ root117Candidates14.length then search edge 3 [659, 6089] root117Candidates14 6748 winner else winner) = _
  rw [if_neg (by decide)]
theorem root117Scan15 : scan edge (search edge 3) 4 [659] 659 root117Tail15 winner = winner := by rw [root117Tail15, scan, if_pos (by decide)]
theorem root117Scan14 : scan edge (search edge 3) 4 [659] 659 root117Tail14 winner = winner :=
  (root117Step14).trans root117Scan15
theorem root117Scan13 : scan edge (search edge 3) 4 [659] 659 root117Tail13 winner = winner :=
  (root117Step13).trans root117Scan14
theorem root117Scan12 : scan edge (search edge 3) 4 [659] 659 root117Tail12 winner = winner :=
  (root117Step12).trans root117Scan13
theorem root117Scan11 : scan edge (search edge 3) 4 [659] 659 root117Tail11 winner = winner :=
  (root117Step11).trans root117Scan12
theorem root117Scan10 : scan edge (search edge 3) 4 [659] 659 root117Tail10 winner = winner :=
  (root117Step10).trans root117Scan11
theorem root117Scan9 : scan edge (search edge 3) 4 [659] 659 root117Tail9 winner = winner :=
  (root117Step9).trans root117Scan10
theorem root117Scan8 : scan edge (search edge 3) 4 [659] 659 root117Tail8 winner = winner :=
  (root117Step8).trans root117Scan9
theorem root117Scan7 : scan edge (search edge 3) 4 [659] 659 root117Tail7 winner = winner :=
  (root117Step7).trans root117Scan8
theorem root117Scan6 : scan edge (search edge 3) 4 [659] 659 root117Tail6 winner = winner :=
  (root117Step6).trans root117Scan7
theorem root117Scan5 : scan edge (search edge 3) 4 [659] 659 root117Tail5 winner = winner :=
  (root117Step5).trans root117Scan6
theorem root117Scan4 : scan edge (search edge 3) 4 [659] 659 root117Tail4 winner = winner :=
  (root117Step4).trans root117Scan5
theorem root117Scan3 : scan edge (search edge 3) 4 [659] 659 root117Tail3 winner = winner :=
  (root117Step3).trans root117Scan4
theorem root117Scan2 : scan edge (search edge 3) 4 [659] 659 root117Tail2 winner = winner :=
  (root117Step2).trans root117Scan3
theorem root117Scan1 : scan edge (search edge 3) 4 [659] 659 root117Tail1 winner = winner :=
  (root117Step1).trans root117Scan2
theorem root117Scan0 : scan edge (search edge 3) 4 [659] 659 root117Tail0 winner = winner :=
  (root117Step0).trans root117Scan1
set_option maxRecDepth 4096 in
theorem root117Branch : search edge 4 [659] root117Candidates 659 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [659] 659 root117Tail0 winner = winner
  exact root117Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
