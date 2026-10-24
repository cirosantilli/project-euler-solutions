import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root84Candidates : List Nat := Data.row449
set_option maxRecDepth 8192 in
theorem root84Candidates_checked : candidates edge 5 0 449 tail85 winner.bound = root84Candidates := by
  apply candidates_eq_row edge 5 0 449 tail85 root84Candidates winner.bound tail85_ordered
  · decide
  · decide
  · funext q
    rfl
def root84Tail54 : List Nat := []
def root84Tail53 : List Nat := 25247 :: root84Tail54
def root84Tail52 : List Nat := 24953 :: root84Tail53
def root84Tail51 : List Nat := 24071 :: root84Tail52
def root84Tail50 : List Nat := 23909 :: root84Tail51
def root84Tail49 : List Nat := 23609 :: root84Tail50
def root84Tail48 : List Nat := 23321 :: root84Tail49
def root84Tail47 : List Nat := 23057 :: root84Tail48
def root84Tail46 : List Nat := 22739 :: root84Tail47
def root84Tail45 : List Nat := 22433 :: root84Tail46
def root84Tail44 : List Nat := 19079 :: root84Tail45
def root84Tail43 : List Nat := 18899 :: root84Tail44
def root84Tail42 : List Nat := 17981 :: root84Tail43
def root84Tail41 : List Nat := 16901 :: root84Tail42
def root84Tail40 : List Nat := 15383 :: root84Tail41
def root84Tail39 : List Nat := 14843 :: root84Tail40
def root84Tail38 : List Nat := 14627 :: root84Tail39
def root84Tail37 : List Nat := 13151 :: root84Tail38
def root84Tail36 : List Nat := 12899 :: root84Tail37
def root84Tail35 : List Nat := 12821 :: root84Tail36
def root84Tail34 : List Nat := 12611 :: root84Tail35
def root84Tail33 : List Nat := 11927 :: root84Tail34
def root84Tail32 : List Nat := 11903 :: root84Tail33
def root84Tail31 : List Nat := 11423 :: root84Tail32
def root84Tail30 : List Nat := 11351 :: root84Tail31
def root84Tail29 : List Nat := 10169 :: root84Tail30
def root84Tail28 : List Nat := 10079 :: root84Tail29
def root84Tail27 : List Nat := 9923 :: root84Tail28
def root84Tail26 : List Nat := 9839 :: root84Tail27
def root84Tail25 : List Nat := 9587 :: root84Tail26
def root84Tail24 : List Nat := 9323 :: root84Tail25
def root84Tail23 : List Nat := 8609 :: root84Tail24
def root84Tail22 : List Nat := 7919 :: root84Tail23
def root84Tail21 : List Nat := 7817 :: root84Tail22
def root84Tail20 : List Nat := 7583 :: root84Tail21
def root84Tail19 : List Nat := 7109 :: root84Tail20
def root84Tail18 : List Nat := 6599 :: root84Tail19
def root84Tail17 : List Nat := 6491 :: root84Tail18
def root84Tail16 : List Nat := 6353 :: root84Tail17
def root84Tail15 : List Nat := 5801 :: root84Tail16
def root84Tail14 : List Nat := 5297 :: root84Tail15
def root84Tail13 : List Nat := 5171 :: root84Tail14
def root84Tail12 : List Nat := 5009 :: root84Tail13
def root84Tail11 : List Nat := 4157 :: root84Tail12
def root84Tail10 : List Nat := 3659 :: root84Tail11
def root84Tail9 : List Nat := 2459 :: root84Tail10
def root84Tail8 : List Nat := 2441 :: root84Tail9
def root84Tail7 : List Nat := 2309 :: root84Tail8
def root84Tail6 : List Nat := 2273 :: root84Tail7
def root84Tail5 : List Nat := 2069 :: root84Tail6
def root84Tail4 : List Nat := 1871 :: root84Tail5
def root84Tail3 : List Nat := 941 :: root84Tail4
def root84Tail2 : List Nat := 821 :: root84Tail3
def root84Tail1 : List Nat := 563 :: root84Tail2
def root84Tail0 : List Nat := 557 :: root84Tail1
def root84Candidates0 : List Nat := [6491, 16901]
set_option maxRecDepth 8192 in
theorem root84Candidates0_checked : candidates edge 4 449 557 root84Tail1 winner.bound = root84Candidates0 := by
  unfold candidates
  have row : edge 557 = (SortedNatTree.contains FastRows.row557) := by
    change Data.row557.contains = (SortedNatTree.contains FastRows.row557)
    exact FastRows.row557_checked.symm
  rw [row]
  decide
theorem root84Step0 : scan edge (search edge 3) 4 [449] 449 root84Tail0 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail1 winner := by
  rw [root84Tail0, scan, if_neg (by decide), root84Candidates0_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail1
    (if 3 ≤ root84Candidates0.length then search edge 3 [449, 557] root84Candidates0 1006 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates1 : List Nat := [5297, 13151, 23909]
set_option maxRecDepth 8192 in
theorem root84Candidates1_checked : candidates edge 4 449 563 root84Tail2 winner.bound = root84Candidates1 := by
  unfold candidates
  have row : edge 563 = (SortedNatTree.contains FastRows.row563) := by
    change Data.row563.contains = (SortedNatTree.contains FastRows.row563)
    exact FastRows.row563_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root84Child1 : search edge 3 [449, 563] root84Candidates1 1012 winner = winner := by decide
theorem root84Step1 : scan edge (search edge 3) 4 [449] 449 root84Tail1 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail2 winner := by
  rw [root84Tail1, scan, if_neg (by decide), root84Candidates1_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail2
    (if 3 ≤ root84Candidates1.length then search edge 3 [449, 563] root84Candidates1 1012 winner else winner) = _
  rw [if_pos (by decide), root84Child1]
def root84Candidates2 : List Nat := [10169, 11423, 16901, 18899, 23057]
set_option maxRecDepth 8192 in
theorem root84Candidates2_checked : candidates edge 4 449 821 root84Tail3 winner.bound = root84Candidates2 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root84Child2 : search edge 3 [449, 821] root84Candidates2 1270 winner = winner := by decide
theorem root84Step2 : scan edge (search edge 3) 4 [449] 449 root84Tail2 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail3 winner := by
  rw [root84Tail2, scan, if_neg (by decide), root84Candidates2_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail3
    (if 3 ≤ root84Candidates2.length then search edge 3 [449, 821] root84Candidates2 1270 winner else winner) = _
  rw [if_pos (by decide), root84Child2]
def root84Candidates3 : List Nat := [9923]
set_option maxRecDepth 8192 in
theorem root84Candidates3_checked : candidates edge 4 449 941 root84Tail4 winner.bound = root84Candidates3 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
theorem root84Step3 : scan edge (search edge 3) 4 [449] 449 root84Tail3 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail4 winner := by
  rw [root84Tail3, scan, if_neg (by decide), root84Candidates3_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail4
    (if 3 ≤ root84Candidates3.length then search edge 3 [449, 941] root84Candidates3 1390 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates4 : List Nat := [11927]
set_option maxRecDepth 8192 in
theorem root84Candidates4_checked : candidates edge 4 449 1871 root84Tail5 winner.bound = root84Candidates4 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
theorem root84Step4 : scan edge (search edge 3) 4 [449] 449 root84Tail4 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail5 winner := by
  rw [root84Tail4, scan, if_neg (by decide), root84Candidates4_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail5
    (if 3 ≤ root84Candidates4.length then search edge 3 [449, 1871] root84Candidates4 2320 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates5 : List Nat := [14627, 22739]
set_option maxRecDepth 8192 in
theorem root84Candidates5_checked : candidates edge 4 449 2069 root84Tail6 winner.bound = root84Candidates5 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
theorem root84Step5 : scan edge (search edge 3) 4 [449] 449 root84Tail5 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail6 winner := by
  rw [root84Tail5, scan, if_neg (by decide), root84Candidates5_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail6
    (if 3 ≤ root84Candidates5.length then search edge 3 [449, 2069] root84Candidates5 2518 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates6 : List Nat := [9323]
set_option maxRecDepth 8192 in
theorem root84Candidates6_checked : candidates edge 4 449 2273 root84Tail7 winner.bound = root84Candidates6 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
theorem root84Step6 : scan edge (search edge 3) 4 [449] 449 root84Tail6 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail7 winner := by
  rw [root84Tail6, scan, if_neg (by decide), root84Candidates6_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail7
    (if 3 ≤ root84Candidates6.length then search edge 3 [449, 2273] root84Candidates6 2722 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates7 : List Nat := [7817]
set_option maxRecDepth 8192 in
theorem root84Candidates7_checked : candidates edge 4 449 2309 root84Tail8 winner.bound = root84Candidates7 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
theorem root84Step7 : scan edge (search edge 3) 4 [449] 449 root84Tail7 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail8 winner := by
  rw [root84Tail7, scan, if_neg (by decide), root84Candidates7_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail8
    (if 3 ≤ root84Candidates7.length then search edge 3 [449, 2309] root84Candidates7 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates8 : List Nat := [4157, 5297, 10169, 11423]
set_option maxRecDepth 8192 in
theorem root84Candidates8_checked : candidates edge 4 449 2441 root84Tail9 winner.bound = root84Candidates8 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root84Child8 : search edge 3 [449, 2441] root84Candidates8 2890 winner = winner := by decide
theorem root84Step8 : scan edge (search edge 3) 4 [449] 449 root84Tail8 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail9 winner := by
  rw [root84Tail8, scan, if_neg (by decide), root84Candidates8_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail9
    (if 3 ≤ root84Candidates8.length then search edge 3 [449, 2441] root84Candidates8 2890 winner else winner) = _
  rw [if_pos (by decide), root84Child8]
def root84Candidates9 : List Nat := [5171, 6353]
set_option maxRecDepth 8192 in
theorem root84Candidates9_checked : candidates edge 4 449 2459 root84Tail10 winner.bound = root84Candidates9 := by
  unfold candidates
  have row : edge 2459 = (SortedNatTree.contains FastRows.row2459) := by
    change Data.row2459.contains = (SortedNatTree.contains FastRows.row2459)
    exact FastRows.row2459_checked.symm
  rw [row]
  decide
theorem root84Step9 : scan edge (search edge 3) 4 [449] 449 root84Tail9 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail10 winner := by
  rw [root84Tail9, scan, if_neg (by decide), root84Candidates9_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail10
    (if 3 ≤ root84Candidates9.length then search edge 3 [449, 2459] root84Candidates9 2908 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates10 : List Nat := [4157]
set_option maxRecDepth 8192 in
theorem root84Candidates10_checked : candidates edge 4 449 3659 root84Tail11 winner.bound = root84Candidates10 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
theorem root84Step10 : scan edge (search edge 3) 4 [449] 449 root84Tail10 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail11 winner := by
  rw [root84Tail10, scan, if_neg (by decide), root84Candidates10_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail11
    (if 3 ≤ root84Candidates10.length then search edge 3 [449, 3659] root84Candidates10 4108 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root84Candidates11_checked : candidates edge 4 449 4157 root84Tail12 winner.bound = root84Candidates11 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root84Step11 : scan edge (search edge 3) 4 [449] 449 root84Tail11 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail12 winner := by
  rw [root84Tail11, scan, if_neg (by decide), root84Candidates11_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail12
    (if 3 ≤ root84Candidates11.length then search edge 3 [449, 4157] root84Candidates11 4606 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates12 : List Nat := [6353, 7109, 9839]
set_option maxRecDepth 8192 in
theorem root84Candidates12_checked : candidates edge 4 449 5009 root84Tail13 winner.bound = root84Candidates12 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root84Child12 : search edge 3 [449, 5009] root84Candidates12 5458 winner = winner := by decide
theorem root84Step12 : scan edge (search edge 3) 4 [449] 449 root84Tail12 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail13 winner := by
  rw [root84Tail12, scan, if_neg (by decide), root84Candidates12_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail13
    (if 3 ≤ root84Candidates12.length then search edge 3 [449, 5009] root84Candidates12 5458 winner else winner) = _
  rw [if_pos (by decide), root84Child12]
def root84Candidates13 : List Nat := [5297]
set_option maxRecDepth 8192 in
theorem root84Candidates13_checked : candidates edge 4 449 5171 root84Tail14 winner.bound = root84Candidates13 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root84Step13 : scan edge (search edge 3) 4 [449] 449 root84Tail13 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail14 winner := by
  rw [root84Tail13, scan, if_neg (by decide), root84Candidates13_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail14
    (if 3 ≤ root84Candidates13.length then search edge 3 [449, 5171] root84Candidates13 5620 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root84Candidates14_checked : candidates edge 4 449 5297 root84Tail15 winner.bound = root84Candidates14 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root84Step14 : scan edge (search edge 3) 4 [449] 449 root84Tail14 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail15 winner := by
  rw [root84Tail14, scan, if_neg (by decide), root84Candidates14_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail15
    (if 3 ≤ root84Candidates14.length then search edge 3 [449, 5297] root84Candidates14 5746 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates15 : List Nat := [7919, 18899]
set_option maxRecDepth 8192 in
theorem root84Candidates15_checked : candidates edge 4 449 5801 root84Tail16 winner.bound = root84Candidates15 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root84Step15 : scan edge (search edge 3) 4 [449] 449 root84Tail15 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail16 winner := by
  rw [root84Tail15, scan, if_neg (by decide), root84Candidates15_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail16
    (if 3 ≤ root84Candidates15.length then search edge 3 [449, 5801] root84Candidates15 6250 winner else winner) = _
  rw [if_neg (by decide)]
def root84Candidates16 : List Nat := [9923]
set_option maxRecDepth 8192 in
theorem root84Candidates16_checked : candidates edge 4 449 6353 root84Tail17 winner.bound = root84Candidates16 := by
  unfold candidates
  have row : edge 6353 = (SortedNatTree.contains FastRows.row6353) := by
    change Data.row6353.contains = (SortedNatTree.contains FastRows.row6353)
    exact FastRows.row6353_checked.symm
  rw [row]
  decide
theorem root84Step16 : scan edge (search edge 3) 4 [449] 449 root84Tail16 winner =
    scan edge (search edge 3) 4 [449] 449 root84Tail17 winner := by
  rw [root84Tail16, scan, if_neg (by decide), root84Candidates16_checked]
  change scan edge (search edge 3) 4 [449] 449 root84Tail17
    (if 3 ≤ root84Candidates16.length then search edge 3 [449, 6353] root84Candidates16 6802 winner else winner) = _
  rw [if_neg (by decide)]
theorem root84Scan17 : scan edge (search edge 3) 4 [449] 449 root84Tail17 winner = winner := by rw [root84Tail17, scan, if_pos (by decide)]
theorem root84Scan16 : scan edge (search edge 3) 4 [449] 449 root84Tail16 winner = winner :=
  (root84Step16).trans root84Scan17
theorem root84Scan15 : scan edge (search edge 3) 4 [449] 449 root84Tail15 winner = winner :=
  (root84Step15).trans root84Scan16
theorem root84Scan14 : scan edge (search edge 3) 4 [449] 449 root84Tail14 winner = winner :=
  (root84Step14).trans root84Scan15
theorem root84Scan13 : scan edge (search edge 3) 4 [449] 449 root84Tail13 winner = winner :=
  (root84Step13).trans root84Scan14
theorem root84Scan12 : scan edge (search edge 3) 4 [449] 449 root84Tail12 winner = winner :=
  (root84Step12).trans root84Scan13
theorem root84Scan11 : scan edge (search edge 3) 4 [449] 449 root84Tail11 winner = winner :=
  (root84Step11).trans root84Scan12
theorem root84Scan10 : scan edge (search edge 3) 4 [449] 449 root84Tail10 winner = winner :=
  (root84Step10).trans root84Scan11
theorem root84Scan9 : scan edge (search edge 3) 4 [449] 449 root84Tail9 winner = winner :=
  (root84Step9).trans root84Scan10
theorem root84Scan8 : scan edge (search edge 3) 4 [449] 449 root84Tail8 winner = winner :=
  (root84Step8).trans root84Scan9
theorem root84Scan7 : scan edge (search edge 3) 4 [449] 449 root84Tail7 winner = winner :=
  (root84Step7).trans root84Scan8
theorem root84Scan6 : scan edge (search edge 3) 4 [449] 449 root84Tail6 winner = winner :=
  (root84Step6).trans root84Scan7
theorem root84Scan5 : scan edge (search edge 3) 4 [449] 449 root84Tail5 winner = winner :=
  (root84Step5).trans root84Scan6
theorem root84Scan4 : scan edge (search edge 3) 4 [449] 449 root84Tail4 winner = winner :=
  (root84Step4).trans root84Scan5
theorem root84Scan3 : scan edge (search edge 3) 4 [449] 449 root84Tail3 winner = winner :=
  (root84Step3).trans root84Scan4
theorem root84Scan2 : scan edge (search edge 3) 4 [449] 449 root84Tail2 winner = winner :=
  (root84Step2).trans root84Scan3
theorem root84Scan1 : scan edge (search edge 3) 4 [449] 449 root84Tail1 winner = winner :=
  (root84Step1).trans root84Scan2
theorem root84Scan0 : scan edge (search edge 3) 4 [449] 449 root84Tail0 winner = winner :=
  (root84Step0).trans root84Scan1
set_option maxRecDepth 4096 in
theorem root84Branch : search edge 4 [449] root84Candidates 449 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [449] 449 root84Tail0 winner = winner
  exact root84Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
