import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root70Candidates : List Nat := Data.row367
set_option maxRecDepth 8192 in
theorem root70Candidates_checked : candidates edge 5 0 367 tail71 winner.bound = root70Candidates := by
  apply candidates_eq_row edge 5 0 367 tail71 root70Candidates winner.bound tail71_ordered
  · decide
  · decide
  · funext q
    rfl
def root70Tail72 : List Nat := []
def root70Tail71 : List Nat := 25303 :: root70Tail72
def root70Tail70 : List Nat := 24847 :: root70Tail71
def root70Tail69 : List Nat := 24691 :: root70Tail70
def root70Tail68 : List Nat := 24007 :: root70Tail69
def root70Tail67 : List Nat := 23671 :: root70Tail68
def root70Tail66 : List Nat := 23581 :: root70Tail67
def root70Tail65 : List Nat := 23473 :: root70Tail66
def root70Tail64 : List Nat := 23059 :: root70Tail65
def root70Tail63 : List Nat := 22783 :: root70Tail64
def root70Tail62 : List Nat := 22717 :: root70Tail63
def root70Tail61 : List Nat := 21799 :: root70Tail62
def root70Tail60 : List Nat := 21673 :: root70Tail61
def root70Tail59 : List Nat := 21589 :: root70Tail60
def root70Tail58 : List Nat := 21001 :: root70Tail59
def root70Tail57 : List Nat := 19777 :: root70Tail58
def root70Tail56 : List Nat := 19759 :: root70Tail57
def root70Tail55 : List Nat := 19489 :: root70Tail56
def root70Tail54 : List Nat := 18757 :: root70Tail55
def root70Tail53 : List Nat := 18397 :: root70Tail54
def root70Tail52 : List Nat := 17971 :: root70Tail53
def root70Tail51 : List Nat := 17851 :: root70Tail52
def root70Tail50 : List Nat := 17341 :: root70Tail51
def root70Tail49 : List Nat := 17047 :: root70Tail50
def root70Tail48 : List Nat := 16573 :: root70Tail49
def root70Tail47 : List Nat := 15859 :: root70Tail48
def root70Tail46 : List Nat := 15643 :: root70Tail47
def root70Tail45 : List Nat := 15313 :: root70Tail46
def root70Tail44 : List Nat := 14947 :: root70Tail45
def root70Tail43 : List Nat := 14341 :: root70Tail44
def root70Tail42 : List Nat := 14071 :: root70Tail43
def root70Tail41 : List Nat := 13759 :: root70Tail42
def root70Tail40 : List Nat := 13441 :: root70Tail41
def root70Tail39 : List Nat := 13297 :: root70Tail40
def root70Tail38 : List Nat := 13147 :: root70Tail39
def root70Tail37 : List Nat := 12703 :: root70Tail38
def root70Tail36 : List Nat := 12007 :: root70Tail37
def root70Tail35 : List Nat := 11923 :: root70Tail36
def root70Tail34 : List Nat := 11839 :: root70Tail35
def root70Tail33 : List Nat := 10627 :: root70Tail34
def root70Tail32 : List Nat := 9967 :: root70Tail33
def root70Tail31 : List Nat := 9679 :: root70Tail32
def root70Tail30 : List Nat := 9601 :: root70Tail31
def root70Tail29 : List Nat := 8929 :: root70Tail30
def root70Tail28 : List Nat := 8923 :: root70Tail29
def root70Tail27 : List Nat := 8887 :: root70Tail28
def root70Tail26 : List Nat := 8839 :: root70Tail27
def root70Tail25 : List Nat := 8821 :: root70Tail26
def root70Tail24 : List Nat := 8377 :: root70Tail25
def root70Tail23 : List Nat := 8317 :: root70Tail24
def root70Tail22 : List Nat := 7789 :: root70Tail23
def root70Tail21 : List Nat := 7573 :: root70Tail22
def root70Tail20 : List Nat := 7459 :: root70Tail21
def root70Tail19 : List Nat := 7207 :: root70Tail20
def root70Tail18 : List Nat := 6091 :: root70Tail19
def root70Tail17 : List Nat := 5563 :: root70Tail18
def root70Tail16 : List Nat := 5059 :: root70Tail17
def root70Tail15 : List Nat := 4663 :: root70Tail16
def root70Tail14 : List Nat := 4597 :: root70Tail15
def root70Tail13 : List Nat := 4339 :: root70Tail14
def root70Tail12 : List Nat := 3529 :: root70Tail13
def root70Tail11 : List Nat := 3463 :: root70Tail12
def root70Tail10 : List Nat := 3217 :: root70Tail11
def root70Tail9 : List Nat := 2791 :: root70Tail10
def root70Tail8 : List Nat := 2659 :: root70Tail9
def root70Tail7 : List Nat := 2521 :: root70Tail8
def root70Tail6 : List Nat := 2293 :: root70Tail7
def root70Tail5 : List Nat := 1531 :: root70Tail6
def root70Tail4 : List Nat := 1117 :: root70Tail5
def root70Tail3 : List Nat := 1021 :: root70Tail4
def root70Tail2 : List Nat := 751 :: root70Tail3
def root70Tail1 : List Nat := 613 :: root70Tail2
def root70Tail0 : List Nat := 457 :: root70Tail1
def root70Candidates0 : List Nat := [8923]
set_option maxRecDepth 8192 in
theorem root70Candidates0_checked : candidates edge 4 367 457 root70Tail1 winner.bound = root70Candidates0 := by
  unfold candidates
  have row : edge 457 = (SortedNatTree.contains FastRows.row457) := by
    change Data.row457.contains = (SortedNatTree.contains FastRows.row457)
    exact FastRows.row457_checked.symm
  rw [row]
  decide
theorem root70Step0 : scan edge (search edge 3) 4 [367] 367 root70Tail0 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail1 winner := by
  rw [root70Tail0, scan, if_neg (by decide), root70Candidates0_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail1
    (if 3 ≤ root70Candidates0.length then search edge 3 [367, 457] root70Candidates0 824 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates1 : List Nat := [4339, 15643, 19759, 22783]
set_option maxRecDepth 8192 in
theorem root70Candidates1_checked : candidates edge 4 367 613 root70Tail2 winner.bound = root70Candidates1 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root70Child1 : search edge 3 [367, 613] root70Candidates1 980 winner = winner := by decide
theorem root70Step1 : scan edge (search edge 3) 4 [367] 367 root70Tail1 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail2 winner := by
  rw [root70Tail1, scan, if_neg (by decide), root70Candidates1_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail2
    (if 3 ≤ root70Candidates1.length then search edge 3 [367, 613] root70Candidates1 980 winner else winner) = _
  rw [if_pos (by decide), root70Child1]
def root70Candidates2 : List Nat := [4597, 18397]
set_option maxRecDepth 8192 in
theorem root70Candidates2_checked : candidates edge 4 367 751 root70Tail3 winner.bound = root70Candidates2 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
theorem root70Step2 : scan edge (search edge 3) 4 [367] 367 root70Tail2 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail3 winner := by
  rw [root70Tail2, scan, if_neg (by decide), root70Candidates2_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail3
    (if 3 ≤ root70Candidates2.length then search edge 3 [367, 751] root70Candidates2 1118 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates3 : List Nat := [3529, 4597]
set_option maxRecDepth 8192 in
theorem root70Candidates3_checked : candidates edge 4 367 1021 root70Tail4 winner.bound = root70Candidates3 := by
  unfold candidates
  have row : edge 1021 = (SortedNatTree.contains FastRows.row1021) := by
    change Data.row1021.contains = (SortedNatTree.contains FastRows.row1021)
    exact FastRows.row1021_checked.symm
  rw [row]
  decide
theorem root70Step3 : scan edge (search edge 3) 4 [367] 367 root70Tail3 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail4 winner := by
  rw [root70Tail3, scan, if_neg (by decide), root70Candidates3_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail4
    (if 3 ≤ root70Candidates3.length then search edge 3 [367, 1021] root70Candidates3 1388 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates4 : List Nat := [8887, 9601, 13759, 16573, 17971]
set_option maxRecDepth 8192 in
theorem root70Candidates4_checked : candidates edge 4 367 1117 root70Tail5 winner.bound = root70Candidates4 := by
  unfold candidates
  have row : edge 1117 = (SortedNatTree.contains FastRows.row1117) := by
    change Data.row1117.contains = (SortedNatTree.contains FastRows.row1117)
    exact FastRows.row1117_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root70Child4 : search edge 3 [367, 1117] root70Candidates4 1484 winner = winner := by decide
theorem root70Step4 : scan edge (search edge 3) 4 [367] 367 root70Tail4 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail5 winner := by
  rw [root70Tail4, scan, if_neg (by decide), root70Candidates4_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail5
    (if 3 ≤ root70Candidates4.length then search edge 3 [367, 1117] root70Candidates4 1484 winner else winner) = _
  rw [if_pos (by decide), root70Child4]
def root70Candidates5 : List Nat := [5059, 7789, 19777]
set_option maxRecDepth 8192 in
theorem root70Candidates5_checked : candidates edge 4 367 1531 root70Tail6 winner.bound = root70Candidates5 := by
  unfold candidates
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root70Child5 : search edge 3 [367, 1531] root70Candidates5 1898 winner = winner := by decide
theorem root70Step5 : scan edge (search edge 3) 4 [367] 367 root70Tail5 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail6 winner := by
  rw [root70Tail5, scan, if_neg (by decide), root70Candidates5_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail6
    (if 3 ≤ root70Candidates5.length then search edge 3 [367, 1531] root70Candidates5 1898 winner else winner) = _
  rw [if_pos (by decide), root70Child5]
def root70Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root70Candidates6_checked : candidates edge 4 367 2293 root70Tail7 winner.bound = root70Candidates6 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root70Step6 : scan edge (search edge 3) 4 [367] 367 root70Tail6 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail7 winner := by
  rw [root70Tail6, scan, if_neg (by decide), root70Candidates6_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail7
    (if 3 ≤ root70Candidates6.length then search edge 3 [367, 2293] root70Candidates6 2660 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates7 : List Nat := [8887, 8929, 16573]
set_option maxRecDepth 8192 in
theorem root70Candidates7_checked : candidates edge 4 367 2521 root70Tail8 winner.bound = root70Candidates7 := by
  unfold candidates
  have row : edge 2521 = (SortedNatTree.contains FastRows.row2521) := by
    change Data.row2521.contains = (SortedNatTree.contains FastRows.row2521)
    exact FastRows.row2521_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root70Child7 : search edge 3 [367, 2521] root70Candidates7 2888 winner = winner := by decide
theorem root70Step7 : scan edge (search edge 3) 4 [367] 367 root70Tail7 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail8 winner := by
  rw [root70Tail7, scan, if_neg (by decide), root70Candidates7_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail8
    (if 3 ≤ root70Candidates7.length then search edge 3 [367, 2521] root70Candidates7 2888 winner else winner) = _
  rw [if_pos (by decide), root70Child7]
def root70Candidates8 : List Nat := [10627]
set_option maxRecDepth 8192 in
theorem root70Candidates8_checked : candidates edge 4 367 2659 root70Tail9 winner.bound = root70Candidates8 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
theorem root70Step8 : scan edge (search edge 3) 4 [367] 367 root70Tail8 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail9 winner := by
  rw [root70Tail8, scan, if_neg (by decide), root70Candidates8_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail9
    (if 3 ≤ root70Candidates8.length then search edge 3 [367, 2659] root70Candidates8 3026 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root70Candidates9_checked : candidates edge 4 367 2791 root70Tail10 winner.bound = root70Candidates9 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root70Step9 : scan edge (search edge 3) 4 [367] 367 root70Tail9 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail10 winner := by
  rw [root70Tail9, scan, if_neg (by decide), root70Candidates9_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail10
    (if 3 ≤ root70Candidates9.length then search edge 3 [367, 2791] root70Candidates9 3158 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates10 : List Nat := [13759]
set_option maxRecDepth 8192 in
theorem root70Candidates10_checked : candidates edge 4 367 3217 root70Tail11 winner.bound = root70Candidates10 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root70Step10 : scan edge (search edge 3) 4 [367] 367 root70Tail10 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail11 winner := by
  rw [root70Tail10, scan, if_neg (by decide), root70Candidates10_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail11
    (if 3 ≤ root70Candidates10.length then search edge 3 [367, 3217] root70Candidates10 3584 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root70Candidates11_checked : candidates edge 4 367 3463 root70Tail12 winner.bound = root70Candidates11 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root70Step11 : scan edge (search edge 3) 4 [367] 367 root70Tail11 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail12 winner := by
  rw [root70Tail11, scan, if_neg (by decide), root70Candidates11_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail12
    (if 3 ≤ root70Candidates11.length then search edge 3 [367, 3463] root70Candidates11 3830 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates12 : List Nat := [21589]
set_option maxRecDepth 8192 in
theorem root70Candidates12_checked : candidates edge 4 367 3529 root70Tail13 winner.bound = root70Candidates12 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root70Step12 : scan edge (search edge 3) 4 [367] 367 root70Tail12 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail13 winner := by
  rw [root70Tail12, scan, if_neg (by decide), root70Candidates12_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail13
    (if 3 ≤ root70Candidates12.length then search edge 3 [367, 3529] root70Candidates12 3896 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates13 : List Nat := [8317, 9679]
set_option maxRecDepth 8192 in
theorem root70Candidates13_checked : candidates edge 4 367 4339 root70Tail14 winner.bound = root70Candidates13 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root70Step13 : scan edge (search edge 3) 4 [367] 367 root70Tail13 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail14 winner := by
  rw [root70Tail13, scan, if_neg (by decide), root70Candidates13_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail14
    (if 3 ≤ root70Candidates13.length then search edge 3 [367, 4339] root70Candidates13 4706 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates14 : List Nat := [5563, 8923, 14071]
set_option maxRecDepth 8192 in
theorem root70Candidates14_checked : candidates edge 4 367 4597 root70Tail15 winner.bound = root70Candidates14 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root70Child14 : search edge 3 [367, 4597] root70Candidates14 4964 winner = winner := by decide
theorem root70Step14 : scan edge (search edge 3) 4 [367] 367 root70Tail14 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail15 winner := by
  rw [root70Tail14, scan, if_neg (by decide), root70Candidates14_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail15
    (if 3 ≤ root70Candidates14.length then search edge 3 [367, 4597] root70Candidates14 4964 winner else winner) = _
  rw [if_pos (by decide), root70Child14]
def root70Candidates15 : List Nat := [9601, 17851]
set_option maxRecDepth 8192 in
theorem root70Candidates15_checked : candidates edge 4 367 4663 root70Tail16 winner.bound = root70Candidates15 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root70Step15 : scan edge (search edge 3) 4 [367] 367 root70Tail15 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail16 winner := by
  rw [root70Tail15, scan, if_neg (by decide), root70Candidates15_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail16
    (if 3 ≤ root70Candidates15.length then search edge 3 [367, 4663] root70Candidates15 5030 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root70Candidates16_checked : candidates edge 4 367 5059 root70Tail17 winner.bound = root70Candidates16 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root70Step16 : scan edge (search edge 3) 4 [367] 367 root70Tail16 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail17 winner := by
  rw [root70Tail16, scan, if_neg (by decide), root70Candidates16_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail17
    (if 3 ≤ root70Candidates16.length then search edge 3 [367, 5059] root70Candidates16 5426 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates17 : List Nat := [12703, 14947]
set_option maxRecDepth 8192 in
theorem root70Candidates17_checked : candidates edge 4 367 5563 root70Tail18 winner.bound = root70Candidates17 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root70Step17 : scan edge (search edge 3) 4 [367] 367 root70Tail17 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail18 winner := by
  rw [root70Tail17, scan, if_neg (by decide), root70Candidates17_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail18
    (if 3 ≤ root70Candidates17.length then search edge 3 [367, 5563] root70Candidates17 5930 winner else winner) = _
  rw [if_neg (by decide)]
def root70Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root70Candidates18_checked : candidates edge 4 367 6091 root70Tail19 winner.bound = root70Candidates18 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root70Step18 : scan edge (search edge 3) 4 [367] 367 root70Tail18 winner =
    scan edge (search edge 3) 4 [367] 367 root70Tail19 winner := by
  rw [root70Tail18, scan, if_neg (by decide), root70Candidates18_checked]
  change scan edge (search edge 3) 4 [367] 367 root70Tail19
    (if 3 ≤ root70Candidates18.length then search edge 3 [367, 6091] root70Candidates18 6458 winner else winner) = _
  rw [if_neg (by decide)]
theorem root70Scan19 : scan edge (search edge 3) 4 [367] 367 root70Tail19 winner = winner := by rw [root70Tail19, scan, if_pos (by decide)]
theorem root70Scan18 : scan edge (search edge 3) 4 [367] 367 root70Tail18 winner = winner :=
  (root70Step18).trans root70Scan19
theorem root70Scan17 : scan edge (search edge 3) 4 [367] 367 root70Tail17 winner = winner :=
  (root70Step17).trans root70Scan18
theorem root70Scan16 : scan edge (search edge 3) 4 [367] 367 root70Tail16 winner = winner :=
  (root70Step16).trans root70Scan17
theorem root70Scan15 : scan edge (search edge 3) 4 [367] 367 root70Tail15 winner = winner :=
  (root70Step15).trans root70Scan16
theorem root70Scan14 : scan edge (search edge 3) 4 [367] 367 root70Tail14 winner = winner :=
  (root70Step14).trans root70Scan15
theorem root70Scan13 : scan edge (search edge 3) 4 [367] 367 root70Tail13 winner = winner :=
  (root70Step13).trans root70Scan14
theorem root70Scan12 : scan edge (search edge 3) 4 [367] 367 root70Tail12 winner = winner :=
  (root70Step12).trans root70Scan13
theorem root70Scan11 : scan edge (search edge 3) 4 [367] 367 root70Tail11 winner = winner :=
  (root70Step11).trans root70Scan12
theorem root70Scan10 : scan edge (search edge 3) 4 [367] 367 root70Tail10 winner = winner :=
  (root70Step10).trans root70Scan11
theorem root70Scan9 : scan edge (search edge 3) 4 [367] 367 root70Tail9 winner = winner :=
  (root70Step9).trans root70Scan10
theorem root70Scan8 : scan edge (search edge 3) 4 [367] 367 root70Tail8 winner = winner :=
  (root70Step8).trans root70Scan9
theorem root70Scan7 : scan edge (search edge 3) 4 [367] 367 root70Tail7 winner = winner :=
  (root70Step7).trans root70Scan8
theorem root70Scan6 : scan edge (search edge 3) 4 [367] 367 root70Tail6 winner = winner :=
  (root70Step6).trans root70Scan7
theorem root70Scan5 : scan edge (search edge 3) 4 [367] 367 root70Tail5 winner = winner :=
  (root70Step5).trans root70Scan6
theorem root70Scan4 : scan edge (search edge 3) 4 [367] 367 root70Tail4 winner = winner :=
  (root70Step4).trans root70Scan5
theorem root70Scan3 : scan edge (search edge 3) 4 [367] 367 root70Tail3 winner = winner :=
  (root70Step3).trans root70Scan4
theorem root70Scan2 : scan edge (search edge 3) 4 [367] 367 root70Tail2 winner = winner :=
  (root70Step2).trans root70Scan3
theorem root70Scan1 : scan edge (search edge 3) 4 [367] 367 root70Tail1 winner = winner :=
  (root70Step1).trans root70Scan2
theorem root70Scan0 : scan edge (search edge 3) 4 [367] 367 root70Tail0 winner = winner :=
  (root70Step0).trans root70Scan1
set_option maxRecDepth 4096 in
theorem root70Branch : search edge 4 [367] root70Candidates 367 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [367] 367 root70Tail0 winner = winner
  exact root70Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
