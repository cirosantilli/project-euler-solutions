import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root240Candidates : List Nat := Data.row1543
set_option maxRecDepth 8192 in
theorem root240Candidates_checked : candidates edge 5 0 1543 tail241 winner.bound = root240Candidates := by
  apply candidates_eq_row edge 5 0 1543 tail241 root240Candidates winner.bound tail241_ordered
  · decide
  · decide
  · funext q
    rfl
def root240Tail55 : List Nat := []
def root240Tail54 : List Nat := 24133 :: root240Tail55
def root240Tail53 : List Nat := 24097 :: root240Tail54
def root240Tail52 : List Nat := 23131 :: root240Tail53
def root240Tail51 : List Nat := 22381 :: root240Tail52
def root240Tail50 : List Nat := 22171 :: root240Tail51
def root240Tail49 : List Nat := 22027 :: root240Tail50
def root240Tail48 : List Nat := 21613 :: root240Tail49
def root240Tail47 : List Nat := 21397 :: root240Tail48
def root240Tail46 : List Nat := 21313 :: root240Tail47
def root240Tail45 : List Nat := 20359 :: root240Tail46
def root240Tail44 : List Nat := 20143 :: root240Tail45
def root240Tail43 : List Nat := 19609 :: root240Tail44
def root240Tail42 : List Nat := 19507 :: root240Tail43
def root240Tail41 : List Nat := 18517 :: root240Tail42
def root240Tail40 : List Nat := 17539 :: root240Tail41
def root240Tail39 : List Nat := 17299 :: root240Tail40
def root240Tail38 : List Nat := 17167 :: root240Tail39
def root240Tail37 : List Nat := 16573 :: root240Tail38
def root240Tail36 : List Nat := 16417 :: root240Tail37
def root240Tail35 : List Nat := 15823 :: root240Tail36
def root240Tail34 : List Nat := 14983 :: root240Tail35
def root240Tail33 : List Nat := 14731 :: root240Tail34
def root240Tail32 : List Nat := 14653 :: root240Tail33
def root240Tail31 : List Nat := 12721 :: root240Tail32
def root240Tail30 : List Nat := 12577 :: root240Tail31
def root240Tail29 : List Nat := 12073 :: root240Tail30
def root240Tail28 : List Nat := 11953 :: root240Tail29
def root240Tail27 : List Nat := 11743 :: root240Tail28
def root240Tail26 : List Nat := 11161 :: root240Tail27
def root240Tail25 : List Nat := 10369 :: root240Tail26
def root240Tail24 : List Nat := 9883 :: root240Tail25
def root240Tail23 : List Nat := 9679 :: root240Tail24
def root240Tail22 : List Nat := 9511 :: root240Tail23
def root240Tail21 : List Nat := 8539 :: root240Tail22
def root240Tail20 : List Nat := 8053 :: root240Tail21
def root240Tail19 : List Nat := 7879 :: root240Tail20
def root240Tail18 : List Nat := 7159 :: root240Tail19
def root240Tail17 : List Nat := 6823 :: root240Tail18
def root240Tail16 : List Nat := 6571 :: root240Tail17
def root240Tail15 : List Nat := 6451 :: root240Tail16
def root240Tail14 : List Nat := 5857 :: root240Tail15
def root240Tail13 : List Nat := 5689 :: root240Tail14
def root240Tail12 : List Nat := 5443 :: root240Tail13
def root240Tail11 : List Nat := 4423 :: root240Tail12
def root240Tail10 : List Nat := 4273 :: root240Tail11
def root240Tail9 : List Nat := 4129 :: root240Tail10
def root240Tail8 : List Nat := 4093 :: root240Tail9
def root240Tail7 : List Nat := 3967 :: root240Tail8
def root240Tail6 : List Nat := 3931 :: root240Tail7
def root240Tail5 : List Nat := 3511 :: root240Tail6
def root240Tail4 : List Nat := 3463 :: root240Tail5
def root240Tail3 : List Nat := 3217 :: root240Tail4
def root240Tail2 : List Nat := 1777 :: root240Tail3
def root240Tail1 : List Nat := 1693 :: root240Tail2
def root240Tail0 : List Nat := 1579 :: root240Tail1
def root240Candidates0 : List Nat := [17167]
set_option maxRecDepth 8192 in
theorem root240Candidates0_checked : candidates edge 4 1543 1579 root240Tail1 winner.bound = root240Candidates0 := by
  unfold candidates
  have row : edge 1579 = (SortedNatTree.contains FastRows.row1579) := by
    change Data.row1579.contains = (SortedNatTree.contains FastRows.row1579)
    exact FastRows.row1579_checked.symm
  rw [row]
  decide
theorem root240Step0 : scan edge (search edge 3) 4 [1543] 1543 root240Tail0 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail1 winner := by
  rw [root240Tail0, scan, if_neg (by decide), root240Candidates0_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail1
    (if 3 ≤ root240Candidates0.length then search edge 3 [1543, 1579] root240Candidates0 3122 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates1 : List Nat := [11953]
set_option maxRecDepth 8192 in
theorem root240Candidates1_checked : candidates edge 4 1543 1693 root240Tail2 winner.bound = root240Candidates1 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
theorem root240Step1 : scan edge (search edge 3) 4 [1543] 1543 root240Tail1 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail2 winner := by
  rw [root240Tail1, scan, if_neg (by decide), root240Candidates1_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail2
    (if 3 ≤ root240Candidates1.length then search edge 3 [1543, 1693] root240Candidates1 3236 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates2 : List Nat := [3463, 20143]
set_option maxRecDepth 8192 in
theorem root240Candidates2_checked : candidates edge 4 1543 1777 root240Tail3 winner.bound = root240Candidates2 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
theorem root240Step2 : scan edge (search edge 3) 4 [1543] 1543 root240Tail2 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail3 winner := by
  rw [root240Tail2, scan, if_neg (by decide), root240Candidates2_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail3
    (if 3 ≤ root240Candidates2.length then search edge 3 [1543, 1777] root240Candidates2 3320 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates3 : List Nat := [3511, 3931]
set_option maxRecDepth 8192 in
theorem root240Candidates3_checked : candidates edge 4 1543 3217 root240Tail4 winner.bound = root240Candidates3 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root240Step3 : scan edge (search edge 3) 4 [1543] 1543 root240Tail3 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail4 winner := by
  rw [root240Tail3, scan, if_neg (by decide), root240Candidates3_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail4
    (if 3 ≤ root240Candidates3.length then search edge 3 [1543, 3217] root240Candidates3 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates4 : List Nat := [8053, 9883]
set_option maxRecDepth 8192 in
theorem root240Candidates4_checked : candidates edge 4 1543 3463 root240Tail5 winner.bound = root240Candidates4 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root240Step4 : scan edge (search edge 3) 4 [1543] 1543 root240Tail4 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail5 winner := by
  rw [root240Tail4, scan, if_neg (by decide), root240Candidates4_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail5
    (if 3 ≤ root240Candidates4.length then search edge 3 [1543, 3463] root240Candidates4 5006 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates5 : List Nat := [7879, 11161]
set_option maxRecDepth 8192 in
theorem root240Candidates5_checked : candidates edge 4 1543 3511 root240Tail6 winner.bound = root240Candidates5 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root240Step5 : scan edge (search edge 3) 4 [1543] 1543 root240Tail5 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail6 winner := by
  rw [root240Tail5, scan, if_neg (by decide), root240Candidates5_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail6
    (if 3 ≤ root240Candidates5.length then search edge 3 [1543, 3511] root240Candidates5 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root240Candidates6_checked : candidates edge 4 1543 3931 root240Tail7 winner.bound = root240Candidates6 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root240Step6 : scan edge (search edge 3) 4 [1543] 1543 root240Tail6 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail7 winner := by
  rw [root240Tail6, scan, if_neg (by decide), root240Candidates6_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail7
    (if 3 ≤ root240Candidates6.length then search edge 3 [1543, 3931] root240Candidates6 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates7 : List Nat := [7159, 8053]
set_option maxRecDepth 8192 in
theorem root240Candidates7_checked : candidates edge 4 1543 3967 root240Tail8 winner.bound = root240Candidates7 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root240Step7 : scan edge (search edge 3) 4 [1543] 1543 root240Tail7 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail8 winner := by
  rw [root240Tail7, scan, if_neg (by decide), root240Candidates7_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail8
    (if 3 ≤ root240Candidates7.length then search edge 3 [1543, 3967] root240Candidates7 5510 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates8 : List Nat := [6823, 16417]
set_option maxRecDepth 8192 in
theorem root240Candidates8_checked : candidates edge 4 1543 4093 root240Tail9 winner.bound = root240Candidates8 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root240Step8 : scan edge (search edge 3) 4 [1543] 1543 root240Tail8 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail9 winner := by
  rw [root240Tail8, scan, if_neg (by decide), root240Candidates8_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail9
    (if 3 ≤ root240Candidates8.length then search edge 3 [1543, 4093] root240Candidates8 5636 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates9 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root240Candidates9_checked : candidates edge 4 1543 4129 root240Tail10 winner.bound = root240Candidates9 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root240Step9 : scan edge (search edge 3) 4 [1543] 1543 root240Tail9 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail10 winner := by
  rw [root240Tail9, scan, if_neg (by decide), root240Candidates9_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail10
    (if 3 ≤ root240Candidates9.length then search edge 3 [1543, 4129] root240Candidates9 5672 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates10 : List Nat := [9883, 19609]
set_option maxRecDepth 8192 in
theorem root240Candidates10_checked : candidates edge 4 1543 4273 root240Tail11 winner.bound = root240Candidates10 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root240Step10 : scan edge (search edge 3) 4 [1543] 1543 root240Tail10 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail11 winner := by
  rw [root240Tail10, scan, if_neg (by decide), root240Candidates10_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail11
    (if 3 ≤ root240Candidates10.length then search edge 3 [1543, 4273] root240Candidates10 5816 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates11 : List Nat := [5689, 15823]
set_option maxRecDepth 8192 in
theorem root240Candidates11_checked : candidates edge 4 1543 4423 root240Tail12 winner.bound = root240Candidates11 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root240Step11 : scan edge (search edge 3) 4 [1543] 1543 root240Tail11 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail12 winner := by
  rw [root240Tail11, scan, if_neg (by decide), root240Candidates11_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail12
    (if 3 ≤ root240Candidates11.length then search edge 3 [1543, 4423] root240Candidates11 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root240Candidates12_checked : candidates edge 4 1543 5443 root240Tail13 winner.bound = root240Candidates12 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root240Step12 : scan edge (search edge 3) 4 [1543] 1543 root240Tail12 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail13 winner := by
  rw [root240Tail12, scan, if_neg (by decide), root240Candidates12_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail13
    (if 3 ≤ root240Candidates12.length then search edge 3 [1543, 5443] root240Candidates12 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates13 : List Nat := [6823, 9679]
set_option maxRecDepth 8192 in
theorem root240Candidates13_checked : candidates edge 4 1543 5689 root240Tail14 winner.bound = root240Candidates13 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root240Step13 : scan edge (search edge 3) 4 [1543] 1543 root240Tail13 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail14 winner := by
  rw [root240Tail13, scan, if_neg (by decide), root240Candidates13_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail14
    (if 3 ≤ root240Candidates13.length then search edge 3 [1543, 5689] root240Candidates13 7232 winner else winner) = _
  rw [if_neg (by decide)]
def root240Candidates14 : List Nat := [9511]
set_option maxRecDepth 8192 in
theorem root240Candidates14_checked : candidates edge 4 1543 5857 root240Tail15 winner.bound = root240Candidates14 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root240Step14 : scan edge (search edge 3) 4 [1543] 1543 root240Tail14 winner =
    scan edge (search edge 3) 4 [1543] 1543 root240Tail15 winner := by
  rw [root240Tail14, scan, if_neg (by decide), root240Candidates14_checked]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail15
    (if 3 ≤ root240Candidates14.length then search edge 3 [1543, 5857] root240Candidates14 7400 winner else winner) = _
  rw [if_neg (by decide)]
theorem root240Scan15 : scan edge (search edge 3) 4 [1543] 1543 root240Tail15 winner = winner := by rw [root240Tail15, scan, if_pos (by decide)]
theorem root240Scan14 : scan edge (search edge 3) 4 [1543] 1543 root240Tail14 winner = winner :=
  (root240Step14).trans root240Scan15
theorem root240Scan13 : scan edge (search edge 3) 4 [1543] 1543 root240Tail13 winner = winner :=
  (root240Step13).trans root240Scan14
theorem root240Scan12 : scan edge (search edge 3) 4 [1543] 1543 root240Tail12 winner = winner :=
  (root240Step12).trans root240Scan13
theorem root240Scan11 : scan edge (search edge 3) 4 [1543] 1543 root240Tail11 winner = winner :=
  (root240Step11).trans root240Scan12
theorem root240Scan10 : scan edge (search edge 3) 4 [1543] 1543 root240Tail10 winner = winner :=
  (root240Step10).trans root240Scan11
theorem root240Scan9 : scan edge (search edge 3) 4 [1543] 1543 root240Tail9 winner = winner :=
  (root240Step9).trans root240Scan10
theorem root240Scan8 : scan edge (search edge 3) 4 [1543] 1543 root240Tail8 winner = winner :=
  (root240Step8).trans root240Scan9
theorem root240Scan7 : scan edge (search edge 3) 4 [1543] 1543 root240Tail7 winner = winner :=
  (root240Step7).trans root240Scan8
theorem root240Scan6 : scan edge (search edge 3) 4 [1543] 1543 root240Tail6 winner = winner :=
  (root240Step6).trans root240Scan7
theorem root240Scan5 : scan edge (search edge 3) 4 [1543] 1543 root240Tail5 winner = winner :=
  (root240Step5).trans root240Scan6
theorem root240Scan4 : scan edge (search edge 3) 4 [1543] 1543 root240Tail4 winner = winner :=
  (root240Step4).trans root240Scan5
theorem root240Scan3 : scan edge (search edge 3) 4 [1543] 1543 root240Tail3 winner = winner :=
  (root240Step3).trans root240Scan4
theorem root240Scan2 : scan edge (search edge 3) 4 [1543] 1543 root240Tail2 winner = winner :=
  (root240Step2).trans root240Scan3
theorem root240Scan1 : scan edge (search edge 3) 4 [1543] 1543 root240Tail1 winner = winner :=
  (root240Step1).trans root240Scan2
theorem root240Scan0 : scan edge (search edge 3) 4 [1543] 1543 root240Tail0 winner = winner :=
  (root240Step0).trans root240Scan1
set_option maxRecDepth 4096 in
theorem root240Branch : search edge 4 [1543] root240Candidates 1543 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1543] 1543 root240Tail0 winner = winner
  exact root240Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
