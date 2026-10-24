import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root97Candidates : List Nat := Data.row541
set_option maxRecDepth 8192 in
theorem root97Candidates_checked : candidates edge 5 0 541 tail98 winner.bound = root97Candidates := by
  apply candidates_eq_row edge 5 0 541 tail98 root97Candidates winner.bound tail98_ordered
  · decide
  · decide
  · funext q
    rfl
def root97Tail69 : List Nat := []
def root97Tail68 : List Nat := 24781 :: root97Tail69
def root97Tail67 : List Nat := 24373 :: root97Tail68
def root97Tail66 : List Nat := 23059 :: root97Tail67
def root97Tail65 : List Nat := 23029 :: root97Tail66
def root97Tail64 : List Nat := 22573 :: root97Tail65
def root97Tail63 : List Nat := 21871 :: root97Tail64
def root97Tail62 : List Nat := 21577 :: root97Tail63
def root97Tail61 : List Nat := 21313 :: root97Tail62
def root97Tail60 : List Nat := 21169 :: root97Tail61
def root97Tail59 : List Nat := 21061 :: root97Tail60
def root97Tail58 : List Nat := 21019 :: root97Tail59
def root97Tail57 : List Nat := 20887 :: root97Tail58
def root97Tail56 : List Nat := 20719 :: root97Tail57
def root97Tail55 : List Nat := 19801 :: root97Tail56
def root97Tail54 : List Nat := 19531 :: root97Tail55
def root97Tail53 : List Nat := 19477 :: root97Tail54
def root97Tail52 : List Nat := 19267 :: root97Tail53
def root97Tail51 : List Nat := 18913 :: root97Tail52
def root97Tail50 : List Nat := 18541 :: root97Tail51
def root97Tail49 : List Nat := 18133 :: root97Tail50
def root97Tail48 : List Nat := 17911 :: root97Tail49
def root97Tail47 : List Nat := 17881 :: root97Tail48
def root97Tail46 : List Nat := 17419 :: root97Tail47
def root97Tail45 : List Nat := 17359 :: root97Tail46
def root97Tail44 : List Nat := 17209 :: root97Tail45
def root97Tail43 : List Nat := 17053 :: root97Tail44
def root97Tail42 : List Nat := 16927 :: root97Tail43
def root97Tail41 : List Nat := 16519 :: root97Tail42
def root97Tail40 : List Nat := 15889 :: root97Tail41
def root97Tail39 : List Nat := 15643 :: root97Tail40
def root97Tail38 : List Nat := 15583 :: root97Tail39
def root97Tail37 : List Nat := 15541 :: root97Tail38
def root97Tail36 : List Nat := 15307 :: root97Tail37
def root97Tail35 : List Nat := 14713 :: root97Tail36
def root97Tail34 : List Nat := 13903 :: root97Tail35
def root97Tail33 : List Nat := 13537 :: root97Tail34
def root97Tail32 : List Nat := 12433 :: root97Tail33
def root97Tail31 : List Nat := 12043 :: root97Tail32
def root97Tail30 : List Nat := 10141 :: root97Tail31
def root97Tail29 : List Nat := 9439 :: root97Tail30
def root97Tail28 : List Nat := 9397 :: root97Tail29
def root97Tail27 : List Nat := 9391 :: root97Tail28
def root97Tail26 : List Nat := 9157 :: root97Tail27
def root97Tail25 : List Nat := 9049 :: root97Tail26
def root97Tail24 : List Nat := 9013 :: root97Tail25
def root97Tail23 : List Nat := 8719 :: root97Tail24
def root97Tail22 : List Nat := 8599 :: root97Tail23
def root97Tail21 : List Nat := 8233 :: root97Tail22
def root97Tail20 : List Nat := 8167 :: root97Tail21
def root97Tail19 : List Nat := 7369 :: root97Tail20
def root97Tail18 : List Nat := 6991 :: root97Tail19
def root97Tail17 : List Nat := 6709 :: root97Tail18
def root97Tail16 : List Nat := 5479 :: root97Tail17
def root97Tail15 : List Nat := 5101 :: root97Tail16
def root97Tail14 : List Nat := 4261 :: root97Tail15
def root97Tail13 : List Nat := 4231 :: root97Tail14
def root97Tail12 : List Nat := 4159 :: root97Tail13
def root97Tail11 : List Nat := 3631 :: root97Tail12
def root97Tail10 : List Nat := 3301 :: root97Tail11
def root97Tail9 : List Nat := 2887 :: root97Tail10
def root97Tail8 : List Nat := 2791 :: root97Tail9
def root97Tail7 : List Nat := 2467 :: root97Tail8
def root97Tail6 : List Nat := 1621 :: root97Tail7
def root97Tail5 : List Nat := 1093 :: root97Tail6
def root97Tail4 : List Nat := 1033 :: root97Tail5
def root97Tail3 : List Nat := 991 :: root97Tail4
def root97Tail2 : List Nat := 727 :: root97Tail3
def root97Tail1 : List Nat := 661 :: root97Tail2
def root97Tail0 : List Nat := 571 :: root97Tail1
def root97Candidates0 : List Nat := [8233, 16519, 19477, 19801]
set_option maxRecDepth 8192 in
theorem root97Candidates0_checked : candidates edge 4 541 571 root97Tail1 winner.bound = root97Candidates0 := by
  unfold candidates
  have row : edge 571 = (SortedNatTree.contains FastRows.row571) := by
    change Data.row571.contains = (SortedNatTree.contains FastRows.row571)
    exact FastRows.row571_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root97Child0 : search edge 3 [541, 571] root97Candidates0 1112 winner = winner := by decide
theorem root97Step0 : scan edge (search edge 3) 4 [541] 541 root97Tail0 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail1 winner := by
  rw [root97Tail0, scan, if_neg (by decide), root97Candidates0_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail1
    (if 3 ≤ root97Candidates0.length then search edge 3 [541, 571] root97Candidates0 1112 winner else winner) = _
  rw [if_pos (by decide), root97Child0]
def root97Candidates1 : List Nat := [2791, 9157]
set_option maxRecDepth 8192 in
theorem root97Candidates1_checked : candidates edge 4 541 661 root97Tail2 winner.bound = root97Candidates1 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
theorem root97Step1 : scan edge (search edge 3) 4 [541] 541 root97Tail1 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail2 winner := by
  rw [root97Tail1, scan, if_neg (by decide), root97Candidates1_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail2
    (if 3 ≤ root97Candidates1.length then search edge 3 [541, 661] root97Candidates1 1202 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates2 : List Nat := [9157, 19267]
set_option maxRecDepth 8192 in
theorem root97Candidates2_checked : candidates edge 4 541 727 root97Tail3 winner.bound = root97Candidates2 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
theorem root97Step2 : scan edge (search edge 3) 4 [541] 541 root97Tail2 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail3 winner := by
  rw [root97Tail2, scan, if_neg (by decide), root97Candidates2_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail3
    (if 3 ≤ root97Candidates2.length then search edge 3 [541, 727] root97Candidates2 1268 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates3 : List Nat := [15307, 21871]
set_option maxRecDepth 8192 in
theorem root97Candidates3_checked : candidates edge 4 541 991 root97Tail4 winner.bound = root97Candidates3 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
theorem root97Step3 : scan edge (search edge 3) 4 [541] 541 root97Tail3 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail4 winner := by
  rw [root97Tail3, scan, if_neg (by decide), root97Candidates3_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail4
    (if 3 ≤ root97Candidates3.length then search edge 3 [541, 991] root97Candidates3 1532 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates4 : List Nat := [4231, 8719, 19267]
set_option maxRecDepth 8192 in
theorem root97Candidates4_checked : candidates edge 4 541 1033 root97Tail5 winner.bound = root97Candidates4 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root97Child4 : search edge 3 [541, 1033] root97Candidates4 1574 winner = winner := by decide
theorem root97Step4 : scan edge (search edge 3) 4 [541] 541 root97Tail4 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail5 winner := by
  rw [root97Tail4, scan, if_neg (by decide), root97Candidates4_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail5
    (if 3 ≤ root97Candidates4.length then search edge 3 [541, 1033] root97Candidates4 1574 winner else winner) = _
  rw [if_pos (by decide), root97Child4]
def root97Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root97Candidates5_checked : candidates edge 4 541 1093 root97Tail6 winner.bound = root97Candidates5 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
theorem root97Step5 : scan edge (search edge 3) 4 [541] 541 root97Tail5 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail6 winner := by
  rw [root97Tail5, scan, if_neg (by decide), root97Candidates5_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail6
    (if 3 ≤ root97Candidates5.length then search edge 3 [541, 1093] root97Candidates5 1634 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates6 : List Nat := [17881]
set_option maxRecDepth 8192 in
theorem root97Candidates6_checked : candidates edge 4 541 1621 root97Tail7 winner.bound = root97Candidates6 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
theorem root97Step6 : scan edge (search edge 3) 4 [541] 541 root97Tail6 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail7 winner := by
  rw [root97Tail6, scan, if_neg (by decide), root97Candidates6_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail7
    (if 3 ≤ root97Candidates6.length then search edge 3 [541, 1621] root97Candidates6 2162 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates7 : List Nat := [8167, 19801]
set_option maxRecDepth 8192 in
theorem root97Candidates7_checked : candidates edge 4 541 2467 root97Tail8 winner.bound = root97Candidates7 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root97Step7 : scan edge (search edge 3) 4 [541] 541 root97Tail7 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail8 winner := by
  rw [root97Tail7, scan, if_neg (by decide), root97Candidates7_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail8
    (if 3 ≤ root97Candidates7.length then search edge 3 [541, 2467] root97Candidates7 3008 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates8 : List Nat := [7369, 15541]
set_option maxRecDepth 8192 in
theorem root97Candidates8_checked : candidates edge 4 541 2791 root97Tail9 winner.bound = root97Candidates8 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root97Step8 : scan edge (search edge 3) 4 [541] 541 root97Tail8 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail9 winner := by
  rw [root97Tail8, scan, if_neg (by decide), root97Candidates8_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail9
    (if 3 ≤ root97Candidates8.length then search edge 3 [541, 2791] root97Candidates8 3332 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates9 : List Nat := [4231, 6709, 9439, 19477, 22573]
set_option maxRecDepth 8192 in
theorem root97Candidates9_checked : candidates edge 4 541 2887 root97Tail10 winner.bound = root97Candidates9 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root97Child9 : search edge 3 [541, 2887] root97Candidates9 3428 winner = winner := by decide
theorem root97Step9 : scan edge (search edge 3) 4 [541] 541 root97Tail9 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail10 winner := by
  rw [root97Tail9, scan, if_neg (by decide), root97Candidates9_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail10
    (if 3 ≤ root97Candidates9.length then search edge 3 [541, 2887] root97Candidates9 3428 winner else winner) = _
  rw [if_pos (by decide), root97Child9]
def root97Candidates10 : List Nat := [18541, 21061]
set_option maxRecDepth 8192 in
theorem root97Candidates10_checked : candidates edge 4 541 3301 root97Tail11 winner.bound = root97Candidates10 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root97Step10 : scan edge (search edge 3) 4 [541] 541 root97Tail10 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail11 winner := by
  rw [root97Tail10, scan, if_neg (by decide), root97Candidates10_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail11
    (if 3 ≤ root97Candidates10.length then search edge 3 [541, 3301] root97Candidates10 3842 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates11 : List Nat := [13537, 16519, 21019]
set_option maxRecDepth 8192 in
theorem root97Candidates11_checked : candidates edge 4 541 3631 root97Tail12 winner.bound = root97Candidates11 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root97Child11 : search edge 3 [541, 3631] root97Candidates11 4172 winner = winner := by decide
theorem root97Step11 : scan edge (search edge 3) 4 [541] 541 root97Tail11 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail12 winner := by
  rw [root97Tail11, scan, if_neg (by decide), root97Candidates11_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail12
    (if 3 ≤ root97Candidates11.length then search edge 3 [541, 3631] root97Candidates11 4172 winner else winner) = _
  rw [if_pos (by decide), root97Child11]
def root97Candidates12 : List Nat := [8233, 15541, 17209]
set_option maxRecDepth 8192 in
theorem root97Candidates12_checked : candidates edge 4 541 4159 root97Tail13 winner.bound = root97Candidates12 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root97Child12 : search edge 3 [541, 4159] root97Candidates12 4700 winner = winner := by decide
theorem root97Step12 : scan edge (search edge 3) 4 [541] 541 root97Tail12 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail13 winner := by
  rw [root97Tail12, scan, if_neg (by decide), root97Candidates12_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail13
    (if 3 ≤ root97Candidates12.length then search edge 3 [541, 4159] root97Candidates12 4700 winner else winner) = _
  rw [if_pos (by decide), root97Child12]
def root97Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root97Candidates13_checked : candidates edge 4 541 4231 root97Tail14 winner.bound = root97Candidates13 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root97Step13 : scan edge (search edge 3) 4 [541] 541 root97Tail13 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail14 winner := by
  rw [root97Tail13, scan, if_neg (by decide), root97Candidates13_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail14
    (if 3 ≤ root97Candidates13.length then search edge 3 [541, 4231] root97Candidates13 4772 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates14 : List Nat := [9397, 12433, 18541, 20719, 21019]
set_option maxRecDepth 8192 in
theorem root97Candidates14_checked : candidates edge 4 541 4261 root97Tail15 winner.bound = root97Candidates14 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root97Child14 : search edge 3 [541, 4261] root97Candidates14 4802 winner = winner := by decide
theorem root97Step14 : scan edge (search edge 3) 4 [541] 541 root97Tail14 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail15 winner := by
  rw [root97Tail14, scan, if_neg (by decide), root97Candidates14_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail15
    (if 3 ≤ root97Candidates14.length then search edge 3 [541, 4261] root97Candidates14 4802 winner else winner) = _
  rw [if_pos (by decide), root97Child14]
def root97Candidates15 : List Nat := [17359, 19531]
set_option maxRecDepth 8192 in
theorem root97Candidates15_checked : candidates edge 4 541 5101 root97Tail16 winner.bound = root97Candidates15 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root97Step15 : scan edge (search edge 3) 4 [541] 541 root97Tail15 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail16 winner := by
  rw [root97Tail15, scan, if_neg (by decide), root97Candidates15_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail16
    (if 3 ≤ root97Candidates15.length then search edge 3 [541, 5101] root97Candidates15 5642 winner else winner) = _
  rw [if_neg (by decide)]
def root97Candidates16 : List Nat := [13537]
set_option maxRecDepth 8192 in
theorem root97Candidates16_checked : candidates edge 4 541 5479 root97Tail17 winner.bound = root97Candidates16 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root97Step16 : scan edge (search edge 3) 4 [541] 541 root97Tail16 winner =
    scan edge (search edge 3) 4 [541] 541 root97Tail17 winner := by
  rw [root97Tail16, scan, if_neg (by decide), root97Candidates16_checked]
  change scan edge (search edge 3) 4 [541] 541 root97Tail17
    (if 3 ≤ root97Candidates16.length then search edge 3 [541, 5479] root97Candidates16 6020 winner else winner) = _
  rw [if_neg (by decide)]
theorem root97Scan17 : scan edge (search edge 3) 4 [541] 541 root97Tail17 winner = winner := by rw [root97Tail17, scan, if_pos (by decide)]
theorem root97Scan16 : scan edge (search edge 3) 4 [541] 541 root97Tail16 winner = winner :=
  (root97Step16).trans root97Scan17
theorem root97Scan15 : scan edge (search edge 3) 4 [541] 541 root97Tail15 winner = winner :=
  (root97Step15).trans root97Scan16
theorem root97Scan14 : scan edge (search edge 3) 4 [541] 541 root97Tail14 winner = winner :=
  (root97Step14).trans root97Scan15
theorem root97Scan13 : scan edge (search edge 3) 4 [541] 541 root97Tail13 winner = winner :=
  (root97Step13).trans root97Scan14
theorem root97Scan12 : scan edge (search edge 3) 4 [541] 541 root97Tail12 winner = winner :=
  (root97Step12).trans root97Scan13
theorem root97Scan11 : scan edge (search edge 3) 4 [541] 541 root97Tail11 winner = winner :=
  (root97Step11).trans root97Scan12
theorem root97Scan10 : scan edge (search edge 3) 4 [541] 541 root97Tail10 winner = winner :=
  (root97Step10).trans root97Scan11
theorem root97Scan9 : scan edge (search edge 3) 4 [541] 541 root97Tail9 winner = winner :=
  (root97Step9).trans root97Scan10
theorem root97Scan8 : scan edge (search edge 3) 4 [541] 541 root97Tail8 winner = winner :=
  (root97Step8).trans root97Scan9
theorem root97Scan7 : scan edge (search edge 3) 4 [541] 541 root97Tail7 winner = winner :=
  (root97Step7).trans root97Scan8
theorem root97Scan6 : scan edge (search edge 3) 4 [541] 541 root97Tail6 winner = winner :=
  (root97Step6).trans root97Scan7
theorem root97Scan5 : scan edge (search edge 3) 4 [541] 541 root97Tail5 winner = winner :=
  (root97Step5).trans root97Scan6
theorem root97Scan4 : scan edge (search edge 3) 4 [541] 541 root97Tail4 winner = winner :=
  (root97Step4).trans root97Scan5
theorem root97Scan3 : scan edge (search edge 3) 4 [541] 541 root97Tail3 winner = winner :=
  (root97Step3).trans root97Scan4
theorem root97Scan2 : scan edge (search edge 3) 4 [541] 541 root97Tail2 winner = winner :=
  (root97Step2).trans root97Scan3
theorem root97Scan1 : scan edge (search edge 3) 4 [541] 541 root97Tail1 winner = winner :=
  (root97Step1).trans root97Scan2
theorem root97Scan0 : scan edge (search edge 3) 4 [541] 541 root97Tail0 winner = winner :=
  (root97Step0).trans root97Scan1
set_option maxRecDepth 4096 in
theorem root97Branch : search edge 4 [541] root97Candidates 541 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [541] 541 root97Tail0 winner = winner
  exact root97Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
