import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root90Candidates : List Nat := Data.row487
set_option maxRecDepth 8192 in
theorem root90Candidates_checked : candidates edge 5 0 487 tail91 winner.bound = root90Candidates := by
  apply candidates_eq_row edge 5 0 487 tail91 root90Candidates winner.bound tail91_ordered
  · decide
  · decide
  · funext q
    rfl
def root90Tail65 : List Nat := []
def root90Tail64 : List Nat := 25471 :: root90Tail65
def root90Tail63 : List Nat := 23929 :: root90Tail64
def root90Tail62 : List Nat := 23833 :: root90Tail63
def root90Tail61 : List Nat := 23743 :: root90Tail62
def root90Tail60 : List Nat := 21727 :: root90Tail61
def root90Tail59 : List Nat := 21661 :: root90Tail60
def root90Tail58 : List Nat := 21601 :: root90Tail59
def root90Tail57 : List Nat := 21577 :: root90Tail58
def root90Tail56 : List Nat := 21397 :: root90Tail57
def root90Tail55 : List Nat := 21019 :: root90Tail56
def root90Tail54 : List Nat := 19963 :: root90Tail55
def root90Tail53 : List Nat := 19081 :: root90Tail54
def root90Tail52 : List Nat := 18913 :: root90Tail53
def root90Tail51 : List Nat := 18517 :: root90Tail52
def root90Tail50 : List Nat := 17293 :: root90Tail51
def root90Tail49 : List Nat := 17029 :: root90Tail50
def root90Tail48 : List Nat := 16519 :: root90Tail49
def root90Tail47 : List Nat := 16417 :: root90Tail48
def root90Tail46 : List Nat := 15739 :: root90Tail47
def root90Tail45 : List Nat := 15361 :: root90Tail46
def root90Tail44 : List Nat := 15193 :: root90Tail45
def root90Tail43 : List Nat := 14821 :: root90Tail44
def root90Tail42 : List Nat := 12823 :: root90Tail43
def root90Tail41 : List Nat := 12739 :: root90Tail42
def root90Tail40 : List Nat := 12619 :: root90Tail41
def root90Tail39 : List Nat := 11959 :: root90Tail40
def root90Tail38 : List Nat := 11863 :: root90Tail39
def root90Tail37 : List Nat := 11287 :: root90Tail38
def root90Tail36 : List Nat := 11059 :: root90Tail37
def root90Tail35 : List Nat := 10891 :: root90Tail36
def root90Tail34 : List Nat := 10477 :: root90Tail35
def root90Tail33 : List Nat := 10303 :: root90Tail34
def root90Tail32 : List Nat := 10267 :: root90Tail33
def root90Tail31 : List Nat := 9883 :: root90Tail32
def root90Tail30 : List Nat := 9871 :: root90Tail31
def root90Tail29 : List Nat := 9733 :: root90Tail30
def root90Tail28 : List Nat := 9631 :: root90Tail29
def root90Tail27 : List Nat := 9547 :: root90Tail28
def root90Tail26 : List Nat := 9319 :: root90Tail27
def root90Tail25 : List Nat := 8941 :: root90Tail26
def root90Tail24 : List Nat := 8893 :: root90Tail25
def root90Tail23 : List Nat := 8527 :: root90Tail24
def root90Tail22 : List Nat := 8431 :: root90Tail23
def root90Tail21 : List Nat := 8233 :: root90Tail22
def root90Tail20 : List Nat := 6709 :: root90Tail21
def root90Tail19 : List Nat := 6607 :: root90Tail20
def root90Tail18 : List Nat := 6301 :: root90Tail19
def root90Tail17 : List Nat := 6133 :: root90Tail18
def root90Tail16 : List Nat := 5881 :: root90Tail17
def root90Tail15 : List Nat := 5407 :: root90Tail16
def root90Tail14 : List Nat := 5113 :: root90Tail15
def root90Tail13 : List Nat := 4339 :: root90Tail14
def root90Tail12 : List Nat := 3697 :: root90Tail13
def root90Tail11 : List Nat := 3559 :: root90Tail12
def root90Tail10 : List Nat := 3433 :: root90Tail11
def root90Tail9 : List Nat := 3163 :: root90Tail10
def root90Tail8 : List Nat := 2311 :: root90Tail9
def root90Tail7 : List Nat := 1723 :: root90Tail8
def root90Tail6 : List Nat := 1567 :: root90Tail7
def root90Tail5 : List Nat := 1297 :: root90Tail6
def root90Tail4 : List Nat := 1093 :: root90Tail5
def root90Tail3 : List Nat := 769 :: root90Tail4
def root90Tail2 : List Nat := 757 :: root90Tail3
def root90Tail1 : List Nat := 727 :: root90Tail2
def root90Tail0 : List Nat := 601 :: root90Tail1
def root90Candidates0 : List Nat := [23929]
set_option maxRecDepth 8192 in
theorem root90Candidates0_checked : candidates edge 4 487 601 root90Tail1 winner.bound = root90Candidates0 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
theorem root90Step0 : scan edge (search edge 3) 4 [487] 487 root90Tail0 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail1 winner := by
  rw [root90Tail0, scan, if_neg (by decide), root90Candidates0_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail1
    (if 3 ≤ root90Candidates0.length then search edge 3 [487, 601] root90Candidates0 1088 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates1 : List Nat := [1723]
set_option maxRecDepth 8192 in
theorem root90Candidates1_checked : candidates edge 4 487 727 root90Tail2 winner.bound = root90Candidates1 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
theorem root90Step1 : scan edge (search edge 3) 4 [487] 487 root90Tail1 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail2 winner := by
  rw [root90Tail1, scan, if_neg (by decide), root90Candidates1_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail2
    (if 3 ≤ root90Candidates1.length then search edge 3 [487, 727] root90Candidates1 1214 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates2 : List Nat := [11959, 16519, 23929]
set_option maxRecDepth 8192 in
theorem root90Candidates2_checked : candidates edge 4 487 757 root90Tail3 winner.bound = root90Candidates2 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root90Child2 : search edge 3 [487, 757] root90Candidates2 1244 winner = winner := by decide
theorem root90Step2 : scan edge (search edge 3) 4 [487] 487 root90Tail2 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail3 winner := by
  rw [root90Tail2, scan, if_neg (by decide), root90Candidates2_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail3
    (if 3 ≤ root90Candidates2.length then search edge 3 [487, 757] root90Candidates2 1244 winner else winner) = _
  rw [if_pos (by decide), root90Child2]
def root90Candidates3 : List Nat := [5113, 6133, 16417]
set_option maxRecDepth 8192 in
theorem root90Candidates3_checked : candidates edge 4 487 769 root90Tail4 winner.bound = root90Candidates3 := by
  unfold candidates
  have row : edge 769 = (SortedNatTree.contains FastRows.row769) := by
    change Data.row769.contains = (SortedNatTree.contains FastRows.row769)
    exact FastRows.row769_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root90Child3 : search edge 3 [487, 769] root90Candidates3 1256 winner = winner := by decide
theorem root90Step3 : scan edge (search edge 3) 4 [487] 487 root90Tail3 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail4 winner := by
  rw [root90Tail3, scan, if_neg (by decide), root90Candidates3_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail4
    (if 3 ≤ root90Candidates3.length then search edge 3 [487, 769] root90Candidates3 1256 winner else winner) = _
  rw [if_pos (by decide), root90Child3]
def root90Candidates4 : List Nat := [9883, 12823]
set_option maxRecDepth 8192 in
theorem root90Candidates4_checked : candidates edge 4 487 1093 root90Tail5 winner.bound = root90Candidates4 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
theorem root90Step4 : scan edge (search edge 3) 4 [487] 487 root90Tail4 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail5 winner := by
  rw [root90Tail4, scan, if_neg (by decide), root90Candidates4_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail5
    (if 3 ≤ root90Candidates4.length then search edge 3 [487, 1093] root90Candidates4 1580 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates5 : List Nat := [16417]
set_option maxRecDepth 8192 in
theorem root90Candidates5_checked : candidates edge 4 487 1297 root90Tail6 winner.bound = root90Candidates5 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
theorem root90Step5 : scan edge (search edge 3) 4 [487] 487 root90Tail5 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail6 winner := by
  rw [root90Tail5, scan, if_neg (by decide), root90Candidates5_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail6
    (if 3 ≤ root90Candidates5.length then search edge 3 [487, 1297] root90Candidates5 1784 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates6 : List Nat := [2311, 9733]
set_option maxRecDepth 8192 in
theorem root90Candidates6_checked : candidates edge 4 487 1567 root90Tail7 winner.bound = root90Candidates6 := by
  unfold candidates
  have row : edge 1567 = (SortedNatTree.contains FastRows.row1567) := by
    change Data.row1567.contains = (SortedNatTree.contains FastRows.row1567)
    exact FastRows.row1567_checked.symm
  rw [row]
  decide
theorem root90Step6 : scan edge (search edge 3) 4 [487] 487 root90Tail6 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail7 winner := by
  rw [root90Tail6, scan, if_neg (by decide), root90Candidates6_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail7
    (if 3 ≤ root90Candidates6.length then search edge 3 [487, 1567] root90Candidates6 2054 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates7 : List Nat := [8233]
set_option maxRecDepth 8192 in
theorem root90Candidates7_checked : candidates edge 4 487 1723 root90Tail8 winner.bound = root90Candidates7 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
theorem root90Step7 : scan edge (search edge 3) 4 [487] 487 root90Tail7 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail8 winner := by
  rw [root90Tail7, scan, if_neg (by decide), root90Candidates7_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail8
    (if 3 ≤ root90Candidates7.length then search edge 3 [487, 1723] root90Candidates7 2210 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates8 : List Nat := [8941, 21661]
set_option maxRecDepth 8192 in
theorem root90Candidates8_checked : candidates edge 4 487 2311 root90Tail9 winner.bound = root90Candidates8 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root90Step8 : scan edge (search edge 3) 4 [487] 487 root90Tail8 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail9 winner := by
  rw [root90Tail8, scan, if_neg (by decide), root90Candidates8_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail9
    (if 3 ≤ root90Candidates8.length then search edge 3 [487, 2311] root90Candidates8 2798 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates9 : List Nat := [6301, 8527, 21727]
set_option maxRecDepth 8192 in
theorem root90Candidates9_checked : candidates edge 4 487 3163 root90Tail10 winner.bound = root90Candidates9 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root90Child9 : search edge 3 [487, 3163] root90Candidates9 3650 winner = winner := by decide
theorem root90Step9 : scan edge (search edge 3) 4 [487] 487 root90Tail9 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail10 winner := by
  rw [root90Tail9, scan, if_neg (by decide), root90Candidates9_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail10
    (if 3 ≤ root90Candidates9.length then search edge 3 [487, 3163] root90Candidates9 3650 winner else winner) = _
  rw [if_pos (by decide), root90Child9]
def root90Candidates10 : List Nat := [5407, 6607]
set_option maxRecDepth 8192 in
theorem root90Candidates10_checked : candidates edge 4 487 3433 root90Tail11 winner.bound = root90Candidates10 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root90Step10 : scan edge (search edge 3) 4 [487] 487 root90Tail10 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail11 winner := by
  rw [root90Tail10, scan, if_neg (by decide), root90Candidates10_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail11
    (if 3 ≤ root90Candidates10.length then search edge 3 [487, 3433] root90Candidates10 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates11 : List Nat := [6607]
set_option maxRecDepth 8192 in
theorem root90Candidates11_checked : candidates edge 4 487 3559 root90Tail12 winner.bound = root90Candidates11 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
theorem root90Step11 : scan edge (search edge 3) 4 [487] 487 root90Tail11 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail12 winner := by
  rw [root90Tail11, scan, if_neg (by decide), root90Candidates11_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail12
    (if 3 ≤ root90Candidates11.length then search edge 3 [487, 3559] root90Candidates11 4046 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates12 : List Nat := [8941, 11059, 15739]
set_option maxRecDepth 8192 in
theorem root90Candidates12_checked : candidates edge 4 487 3697 root90Tail13 winner.bound = root90Candidates12 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root90Child12 : search edge 3 [487, 3697] root90Candidates12 4184 winner = winner := by decide
theorem root90Step12 : scan edge (search edge 3) 4 [487] 487 root90Tail12 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail13 winner := by
  rw [root90Tail12, scan, if_neg (by decide), root90Candidates12_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail13
    (if 3 ≤ root90Candidates12.length then search edge 3 [487, 3697] root90Candidates12 4184 winner else winner) = _
  rw [if_pos (by decide), root90Child12]
def root90Candidates13 : List Nat := [10303]
set_option maxRecDepth 8192 in
theorem root90Candidates13_checked : candidates edge 4 487 4339 root90Tail14 winner.bound = root90Candidates13 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root90Step13 : scan edge (search edge 3) 4 [487] 487 root90Tail13 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail14 winner := by
  rw [root90Tail13, scan, if_neg (by decide), root90Candidates13_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail14
    (if 3 ≤ root90Candidates13.length then search edge 3 [487, 4339] root90Candidates13 4826 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root90Candidates14_checked : candidates edge 4 487 5113 root90Tail15 winner.bound = root90Candidates14 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root90Step14 : scan edge (search edge 3) 4 [487] 487 root90Tail14 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail15 winner := by
  rw [root90Tail14, scan, if_neg (by decide), root90Candidates14_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail15
    (if 3 ≤ root90Candidates14.length then search edge 3 [487, 5113] root90Candidates14 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root90Candidates15_checked : candidates edge 4 487 5407 root90Tail16 winner.bound = root90Candidates15 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root90Step15 : scan edge (search edge 3) 4 [487] 487 root90Tail15 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail16 winner := by
  rw [root90Tail15, scan, if_neg (by decide), root90Candidates15_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail16
    (if 3 ≤ root90Candidates15.length then search edge 3 [487, 5407] root90Candidates15 5894 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates16 : List Nat := [9547, 17293]
set_option maxRecDepth 8192 in
theorem root90Candidates16_checked : candidates edge 4 487 5881 root90Tail17 winner.bound = root90Candidates16 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root90Step16 : scan edge (search edge 3) 4 [487] 487 root90Tail16 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail17 winner := by
  rw [root90Tail16, scan, if_neg (by decide), root90Candidates16_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail17
    (if 3 ≤ root90Candidates16.length then search edge 3 [487, 5881] root90Candidates16 6368 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates17 : List Nat := [9319]
set_option maxRecDepth 8192 in
theorem root90Candidates17_checked : candidates edge 4 487 6133 root90Tail18 winner.bound = root90Candidates17 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root90Step17 : scan edge (search edge 3) 4 [487] 487 root90Tail17 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail18 winner := by
  rw [root90Tail17, scan, if_neg (by decide), root90Candidates17_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail18
    (if 3 ≤ root90Candidates17.length then search edge 3 [487, 6133] root90Candidates17 6620 winner else winner) = _
  rw [if_neg (by decide)]
def root90Candidates18 : List Nat := [6607, 8233, 9883]
set_option maxRecDepth 8192 in
theorem root90Candidates18_checked : candidates edge 4 487 6301 root90Tail19 winner.bound = root90Candidates18 := by
  unfold candidates
  have row : edge 6301 = (SortedNatTree.contains FastRows.row6301) := by
    change Data.row6301.contains = (SortedNatTree.contains FastRows.row6301)
    exact FastRows.row6301_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root90Child18 : search edge 3 [487, 6301] root90Candidates18 6788 winner = winner := by decide
theorem root90Step18 : scan edge (search edge 3) 4 [487] 487 root90Tail18 winner =
    scan edge (search edge 3) 4 [487] 487 root90Tail19 winner := by
  rw [root90Tail18, scan, if_neg (by decide), root90Candidates18_checked]
  change scan edge (search edge 3) 4 [487] 487 root90Tail19
    (if 3 ≤ root90Candidates18.length then search edge 3 [487, 6301] root90Candidates18 6788 winner else winner) = _
  rw [if_pos (by decide), root90Child18]
theorem root90Scan19 : scan edge (search edge 3) 4 [487] 487 root90Tail19 winner = winner := by rw [root90Tail19, scan, if_pos (by decide)]
theorem root90Scan18 : scan edge (search edge 3) 4 [487] 487 root90Tail18 winner = winner :=
  (root90Step18).trans root90Scan19
theorem root90Scan17 : scan edge (search edge 3) 4 [487] 487 root90Tail17 winner = winner :=
  (root90Step17).trans root90Scan18
theorem root90Scan16 : scan edge (search edge 3) 4 [487] 487 root90Tail16 winner = winner :=
  (root90Step16).trans root90Scan17
theorem root90Scan15 : scan edge (search edge 3) 4 [487] 487 root90Tail15 winner = winner :=
  (root90Step15).trans root90Scan16
theorem root90Scan14 : scan edge (search edge 3) 4 [487] 487 root90Tail14 winner = winner :=
  (root90Step14).trans root90Scan15
theorem root90Scan13 : scan edge (search edge 3) 4 [487] 487 root90Tail13 winner = winner :=
  (root90Step13).trans root90Scan14
theorem root90Scan12 : scan edge (search edge 3) 4 [487] 487 root90Tail12 winner = winner :=
  (root90Step12).trans root90Scan13
theorem root90Scan11 : scan edge (search edge 3) 4 [487] 487 root90Tail11 winner = winner :=
  (root90Step11).trans root90Scan12
theorem root90Scan10 : scan edge (search edge 3) 4 [487] 487 root90Tail10 winner = winner :=
  (root90Step10).trans root90Scan11
theorem root90Scan9 : scan edge (search edge 3) 4 [487] 487 root90Tail9 winner = winner :=
  (root90Step9).trans root90Scan10
theorem root90Scan8 : scan edge (search edge 3) 4 [487] 487 root90Tail8 winner = winner :=
  (root90Step8).trans root90Scan9
theorem root90Scan7 : scan edge (search edge 3) 4 [487] 487 root90Tail7 winner = winner :=
  (root90Step7).trans root90Scan8
theorem root90Scan6 : scan edge (search edge 3) 4 [487] 487 root90Tail6 winner = winner :=
  (root90Step6).trans root90Scan7
theorem root90Scan5 : scan edge (search edge 3) 4 [487] 487 root90Tail5 winner = winner :=
  (root90Step5).trans root90Scan6
theorem root90Scan4 : scan edge (search edge 3) 4 [487] 487 root90Tail4 winner = winner :=
  (root90Step4).trans root90Scan5
theorem root90Scan3 : scan edge (search edge 3) 4 [487] 487 root90Tail3 winner = winner :=
  (root90Step3).trans root90Scan4
theorem root90Scan2 : scan edge (search edge 3) 4 [487] 487 root90Tail2 winner = winner :=
  (root90Step2).trans root90Scan3
theorem root90Scan1 : scan edge (search edge 3) 4 [487] 487 root90Tail1 winner = winner :=
  (root90Step1).trans root90Scan2
theorem root90Scan0 : scan edge (search edge 3) 4 [487] 487 root90Tail0 winner = winner :=
  (root90Step0).trans root90Scan1
set_option maxRecDepth 4096 in
theorem root90Branch : search edge 4 [487] root90Candidates 487 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [487] 487 root90Tail0 winner = winner
  exact root90Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
