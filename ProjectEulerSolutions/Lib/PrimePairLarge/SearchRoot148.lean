import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root148Candidates : List Nat := Data.row877
set_option maxRecDepth 8192 in
theorem root148Candidates_checked : candidates edge 5 0 877 tail149 winner.bound = root148Candidates := by
  apply candidates_eq_row edge 5 0 877 tail149 root148Candidates winner.bound tail149_ordered
  · decide
  · decide
  · funext q
    rfl
def root148Tail65 : List Nat := []
def root148Tail64 : List Nat := 24907 :: root148Tail65
def root148Tail63 : List Nat := 24781 :: root148Tail64
def root148Tail62 : List Nat := 23833 :: root148Tail63
def root148Tail61 : List Nat := 23131 :: root148Tail62
def root148Tail60 : List Nat := 22573 :: root148Tail61
def root148Tail59 : List Nat := 22543 :: root148Tail60
def root148Tail58 : List Nat := 22441 :: root148Tail59
def root148Tail57 : List Nat := 22273 :: root148Tail58
def root148Tail56 : List Nat := 22039 :: root148Tail57
def root148Tail55 : List Nat := 22027 :: root148Tail56
def root148Tail54 : List Nat := 21577 :: root148Tail55
def root148Tail53 : List Nat := 21211 :: root148Tail54
def root148Tail52 : List Nat := 20161 :: root148Tail53
def root148Tail51 : List Nat := 19543 :: root148Tail52
def root148Tail50 : List Nat := 18049 :: root148Tail51
def root148Tail49 : List Nat := 17911 :: root148Tail50
def root148Tail48 : List Nat := 17683 :: root148Tail49
def root148Tail47 : List Nat := 16903 :: root148Tail48
def root148Tail46 : List Nat := 16339 :: root148Tail47
def root148Tail45 : List Nat := 15919 :: root148Tail46
def root148Tail44 : List Nat := 15193 :: root148Tail45
def root148Tail43 : List Nat := 15013 :: root148Tail44
def root148Tail42 : List Nat := 14983 :: root148Tail43
def root148Tail41 : List Nat := 14551 :: root148Tail42
def root148Tail40 : List Nat := 13933 :: root148Tail41
def root148Tail39 : List Nat := 13597 :: root148Tail40
def root148Tail38 : List Nat := 13399 :: root148Tail39
def root148Tail37 : List Nat := 13033 :: root148Tail38
def root148Tail36 : List Nat := 12451 :: root148Tail37
def root148Tail35 : List Nat := 11779 :: root148Tail36
def root148Tail34 : List Nat := 11311 :: root148Tail35
def root148Tail33 : List Nat := 11149 :: root148Tail34
def root148Tail32 : List Nat := 10771 :: root148Tail33
def root148Tail31 : List Nat := 10267 :: root148Tail32
def root148Tail30 : List Nat := 9241 :: root148Tail31
def root148Tail29 : List Nat := 8461 :: root148Tail30
def root148Tail28 : List Nat := 8389 :: root148Tail29
def root148Tail27 : List Nat := 8293 :: root148Tail28
def root148Tail26 : List Nat := 8191 :: root148Tail27
def root148Tail25 : List Nat := 7759 :: root148Tail26
def root148Tail24 : List Nat := 7669 :: root148Tail25
def root148Tail23 : List Nat := 7243 :: root148Tail24
def root148Tail22 : List Nat := 6991 :: root148Tail23
def root148Tail21 : List Nat := 6637 :: root148Tail22
def root148Tail20 : List Nat := 6529 :: root148Tail21
def root148Tail19 : List Nat := 6373 :: root148Tail20
def root148Tail18 : List Nat := 6121 :: root148Tail19
def root148Tail17 : List Nat := 5743 :: root148Tail18
def root148Tail16 : List Nat := 5659 :: root148Tail17
def root148Tail15 : List Nat := 5527 :: root148Tail16
def root148Tail14 : List Nat := 5419 :: root148Tail15
def root148Tail13 : List Nat := 4723 :: root148Tail14
def root148Tail12 : List Nat := 4483 :: root148Tail13
def root148Tail11 : List Nat := 3823 :: root148Tail12
def root148Tail10 : List Nat := 3727 :: root148Tail11
def root148Tail9 : List Nat := 3469 :: root148Tail10
def root148Tail8 : List Nat := 2917 :: root148Tail9
def root148Tail7 : List Nat := 2551 :: root148Tail8
def root148Tail6 : List Nat := 2371 :: root148Tail7
def root148Tail5 : List Nat := 2251 :: root148Tail6
def root148Tail4 : List Nat := 1621 :: root148Tail5
def root148Tail3 : List Nat := 1327 :: root148Tail4
def root148Tail2 : List Nat := 997 :: root148Tail3
def root148Tail1 : List Nat := 937 :: root148Tail2
def root148Tail0 : List Nat := 883 :: root148Tail1
def root148Candidates0 : List Nat := [11779, 18049, 22543, 22573]
set_option maxRecDepth 8192 in
theorem root148Candidates0_checked : candidates edge 4 877 883 root148Tail1 winner.bound = root148Candidates0 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root148Child0 : search edge 3 [877, 883] root148Candidates0 1760 winner = winner := by decide
theorem root148Step0 : scan edge (search edge 3) 4 [877] 877 root148Tail0 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail1 winner := by
  rw [root148Tail0, scan, if_neg (by decide), root148Candidates0_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail1
    (if 3 ≤ root148Candidates0.length then search edge 3 [877, 883] root148Candidates0 1760 winner else winner) = _
  rw [if_pos (by decide), root148Child0]
def root148Candidates1 : List Nat := [16339, 22039]
set_option maxRecDepth 8192 in
theorem root148Candidates1_checked : candidates edge 4 877 937 root148Tail2 winner.bound = root148Candidates1 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
theorem root148Step1 : scan edge (search edge 3) 4 [877] 877 root148Tail1 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail2 winner := by
  rw [root148Tail1, scan, if_neg (by decide), root148Candidates1_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail2
    (if 3 ≤ root148Candidates1.length then search edge 3 [877, 937] root148Candidates1 1814 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root148Candidates2_checked : candidates edge 4 877 997 root148Tail3 winner.bound = root148Candidates2 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
theorem root148Step2 : scan edge (search edge 3) 4 [877] 877 root148Tail2 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail3 winner := by
  rw [root148Tail2, scan, if_neg (by decide), root148Candidates2_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail3
    (if 3 ≤ root148Candidates2.length then search edge 3 [877, 997] root148Candidates2 1874 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates3 : List Nat := [2551, 9241, 22573]
set_option maxRecDepth 8192 in
theorem root148Candidates3_checked : candidates edge 4 877 1327 root148Tail4 winner.bound = root148Candidates3 := by
  unfold candidates
  have row : edge 1327 = (SortedNatTree.contains FastRows.row1327) := by
    change Data.row1327.contains = (SortedNatTree.contains FastRows.row1327)
    exact FastRows.row1327_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root148Child3 : search edge 3 [877, 1327] root148Candidates3 2204 winner = winner := by decide
theorem root148Step3 : scan edge (search edge 3) 4 [877] 877 root148Tail3 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail4 winner := by
  rw [root148Tail3, scan, if_neg (by decide), root148Candidates3_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail4
    (if 3 ≤ root148Candidates3.length then search edge 3 [877, 1327] root148Candidates3 2204 winner else winner) = _
  rw [if_pos (by decide), root148Child3]
def root148Candidates4 : List Nat := [3823, 5527, 11149, 15193]
set_option maxRecDepth 8192 in
theorem root148Candidates4_checked : candidates edge 4 877 1621 root148Tail5 winner.bound = root148Candidates4 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root148Child4 : search edge 3 [877, 1621] root148Candidates4 2498 winner = winner := by decide
theorem root148Step4 : scan edge (search edge 3) 4 [877] 877 root148Tail4 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail5 winner := by
  rw [root148Tail4, scan, if_neg (by decide), root148Candidates4_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail5
    (if 3 ≤ root148Candidates4.length then search edge 3 [877, 1621] root148Candidates4 2498 winner else winner) = _
  rw [if_pos (by decide), root148Child4]
def root148Candidates5 : List Nat := [7669, 8191, 22039]
set_option maxRecDepth 8192 in
theorem root148Candidates5_checked : candidates edge 4 877 2251 root148Tail6 winner.bound = root148Candidates5 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root148Child5 : search edge 3 [877, 2251] root148Candidates5 3128 winner = winner := by decide
theorem root148Step5 : scan edge (search edge 3) 4 [877] 877 root148Tail5 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail6 winner := by
  rw [root148Tail5, scan, if_neg (by decide), root148Candidates5_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail6
    (if 3 ≤ root148Candidates5.length then search edge 3 [877, 2251] root148Candidates5 3128 winner else winner) = _
  rw [if_pos (by decide), root148Child5]
def root148Candidates6 : List Nat := [2917, 17911]
set_option maxRecDepth 8192 in
theorem root148Candidates6_checked : candidates edge 4 877 2371 root148Tail7 winner.bound = root148Candidates6 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root148Step6 : scan edge (search edge 3) 4 [877] 877 root148Tail6 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail7 winner := by
  rw [root148Tail6, scan, if_neg (by decide), root148Candidates6_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail7
    (if 3 ≤ root148Candidates6.length then search edge 3 [877, 2371] root148Candidates6 3248 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates7 : List Nat := [18049]
set_option maxRecDepth 8192 in
theorem root148Candidates7_checked : candidates edge 4 877 2551 root148Tail8 winner.bound = root148Candidates7 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root148Step7 : scan edge (search edge 3) 4 [877] 877 root148Tail7 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail8 winner := by
  rw [root148Tail7, scan, if_neg (by decide), root148Candidates7_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail8
    (if 3 ≤ root148Candidates7.length then search edge 3 [877, 2551] root148Candidates7 3428 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates8 : List Nat := [10267, 11149]
set_option maxRecDepth 8192 in
theorem root148Candidates8_checked : candidates edge 4 877 2917 root148Tail9 winner.bound = root148Candidates8 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root148Step8 : scan edge (search edge 3) 4 [877] 877 root148Tail8 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail9 winner := by
  rw [root148Tail8, scan, if_neg (by decide), root148Candidates8_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail9
    (if 3 ≤ root148Candidates8.length then search edge 3 [877, 2917] root148Candidates8 3794 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root148Candidates9_checked : candidates edge 4 877 3469 root148Tail10 winner.bound = root148Candidates9 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root148Step9 : scan edge (search edge 3) 4 [877] 877 root148Tail9 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail10 winner := by
  rw [root148Tail9, scan, if_neg (by decide), root148Candidates9_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail10
    (if 3 ≤ root148Candidates9.length then search edge 3 [877, 3469] root148Candidates9 4346 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates10 : List Nat := [5659]
set_option maxRecDepth 8192 in
theorem root148Candidates10_checked : candidates edge 4 877 3727 root148Tail11 winner.bound = root148Candidates10 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root148Step10 : scan edge (search edge 3) 4 [877] 877 root148Tail10 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail11 winner := by
  rw [root148Tail10, scan, if_neg (by decide), root148Candidates10_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail11
    (if 3 ≤ root148Candidates10.length then search edge 3 [877, 3727] root148Candidates10 4604 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root148Candidates11_checked : candidates edge 4 877 3823 root148Tail12 winner.bound = root148Candidates11 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root148Step11 : scan edge (search edge 3) 4 [877] 877 root148Tail11 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail12 winner := by
  rw [root148Tail11, scan, if_neg (by decide), root148Candidates11_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail12
    (if 3 ≤ root148Candidates11.length then search edge 3 [877, 3823] root148Candidates11 4700 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates12 : List Nat := [7759]
set_option maxRecDepth 8192 in
theorem root148Candidates12_checked : candidates edge 4 877 4483 root148Tail13 winner.bound = root148Candidates12 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root148Step12 : scan edge (search edge 3) 4 [877] 877 root148Tail12 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail13 winner := by
  rw [root148Tail12, scan, if_neg (by decide), root148Candidates12_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail13
    (if 3 ≤ root148Candidates12.length then search edge 3 [877, 4483] root148Candidates12 5360 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates13 : List Nat := [7243, 15919]
set_option maxRecDepth 8192 in
theorem root148Candidates13_checked : candidates edge 4 877 4723 root148Tail14 winner.bound = root148Candidates13 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root148Step13 : scan edge (search edge 3) 4 [877] 877 root148Tail13 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail14 winner := by
  rw [root148Tail13, scan, if_neg (by decide), root148Candidates13_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail14
    (if 3 ≤ root148Candidates13.length then search edge 3 [877, 4723] root148Candidates13 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates14 : List Nat := [9241, 13399]
set_option maxRecDepth 8192 in
theorem root148Candidates14_checked : candidates edge 4 877 5419 root148Tail15 winner.bound = root148Candidates14 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root148Step14 : scan edge (search edge 3) 4 [877] 877 root148Tail14 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail15 winner := by
  rw [root148Tail14, scan, if_neg (by decide), root148Candidates14_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail15
    (if 3 ≤ root148Candidates14.length then search edge 3 [877, 5419] root148Candidates14 6296 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates15 : List Nat := [14983]
set_option maxRecDepth 8192 in
theorem root148Candidates15_checked : candidates edge 4 877 5527 root148Tail16 winner.bound = root148Candidates15 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root148Step15 : scan edge (search edge 3) 4 [877] 877 root148Tail15 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail16 winner := by
  rw [root148Tail15, scan, if_neg (by decide), root148Candidates15_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail16
    (if 3 ≤ root148Candidates15.length then search edge 3 [877, 5527] root148Candidates15 6404 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates16 : List Nat := [6121, 7759, 8461, 11779, 14551]
set_option maxRecDepth 8192 in
theorem root148Candidates16_checked : candidates edge 4 877 5659 root148Tail17 winner.bound = root148Candidates16 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root148Child16 : search edge 3 [877, 5659] root148Candidates16 6536 winner = winner := by decide
theorem root148Step16 : scan edge (search edge 3) 4 [877] 877 root148Tail16 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail17 winner := by
  rw [root148Tail16, scan, if_neg (by decide), root148Candidates16_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail17
    (if 3 ≤ root148Candidates16.length then search edge 3 [877, 5659] root148Candidates16 6536 winner else winner) = _
  rw [if_pos (by decide), root148Child16]
def root148Candidates17 : List Nat := [7759, 14551]
set_option maxRecDepth 8192 in
theorem root148Candidates17_checked : candidates edge 4 877 5743 root148Tail18 winner.bound = root148Candidates17 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root148Step17 : scan edge (search edge 3) 4 [877] 877 root148Tail17 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail18 winner := by
  rw [root148Tail17, scan, if_neg (by decide), root148Candidates17_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail18
    (if 3 ≤ root148Candidates17.length then search edge 3 [877, 5743] root148Candidates17 6620 winner else winner) = _
  rw [if_neg (by decide)]
def root148Candidates18 : List Nat := [8389]
set_option maxRecDepth 8192 in
theorem root148Candidates18_checked : candidates edge 4 877 6121 root148Tail19 winner.bound = root148Candidates18 := by
  unfold candidates
  have row : edge 6121 = (SortedNatTree.contains FastRows.row6121) := by
    change Data.row6121.contains = (SortedNatTree.contains FastRows.row6121)
    exact FastRows.row6121_checked.symm
  rw [row]
  decide
theorem root148Step18 : scan edge (search edge 3) 4 [877] 877 root148Tail18 winner =
    scan edge (search edge 3) 4 [877] 877 root148Tail19 winner := by
  rw [root148Tail18, scan, if_neg (by decide), root148Candidates18_checked]
  change scan edge (search edge 3) 4 [877] 877 root148Tail19
    (if 3 ≤ root148Candidates18.length then search edge 3 [877, 6121] root148Candidates18 6998 winner else winner) = _
  rw [if_neg (by decide)]
theorem root148Scan19 : scan edge (search edge 3) 4 [877] 877 root148Tail19 winner = winner := by rw [root148Tail19, scan, if_pos (by decide)]
theorem root148Scan18 : scan edge (search edge 3) 4 [877] 877 root148Tail18 winner = winner :=
  (root148Step18).trans root148Scan19
theorem root148Scan17 : scan edge (search edge 3) 4 [877] 877 root148Tail17 winner = winner :=
  (root148Step17).trans root148Scan18
theorem root148Scan16 : scan edge (search edge 3) 4 [877] 877 root148Tail16 winner = winner :=
  (root148Step16).trans root148Scan17
theorem root148Scan15 : scan edge (search edge 3) 4 [877] 877 root148Tail15 winner = winner :=
  (root148Step15).trans root148Scan16
theorem root148Scan14 : scan edge (search edge 3) 4 [877] 877 root148Tail14 winner = winner :=
  (root148Step14).trans root148Scan15
theorem root148Scan13 : scan edge (search edge 3) 4 [877] 877 root148Tail13 winner = winner :=
  (root148Step13).trans root148Scan14
theorem root148Scan12 : scan edge (search edge 3) 4 [877] 877 root148Tail12 winner = winner :=
  (root148Step12).trans root148Scan13
theorem root148Scan11 : scan edge (search edge 3) 4 [877] 877 root148Tail11 winner = winner :=
  (root148Step11).trans root148Scan12
theorem root148Scan10 : scan edge (search edge 3) 4 [877] 877 root148Tail10 winner = winner :=
  (root148Step10).trans root148Scan11
theorem root148Scan9 : scan edge (search edge 3) 4 [877] 877 root148Tail9 winner = winner :=
  (root148Step9).trans root148Scan10
theorem root148Scan8 : scan edge (search edge 3) 4 [877] 877 root148Tail8 winner = winner :=
  (root148Step8).trans root148Scan9
theorem root148Scan7 : scan edge (search edge 3) 4 [877] 877 root148Tail7 winner = winner :=
  (root148Step7).trans root148Scan8
theorem root148Scan6 : scan edge (search edge 3) 4 [877] 877 root148Tail6 winner = winner :=
  (root148Step6).trans root148Scan7
theorem root148Scan5 : scan edge (search edge 3) 4 [877] 877 root148Tail5 winner = winner :=
  (root148Step5).trans root148Scan6
theorem root148Scan4 : scan edge (search edge 3) 4 [877] 877 root148Tail4 winner = winner :=
  (root148Step4).trans root148Scan5
theorem root148Scan3 : scan edge (search edge 3) 4 [877] 877 root148Tail3 winner = winner :=
  (root148Step3).trans root148Scan4
theorem root148Scan2 : scan edge (search edge 3) 4 [877] 877 root148Tail2 winner = winner :=
  (root148Step2).trans root148Scan3
theorem root148Scan1 : scan edge (search edge 3) 4 [877] 877 root148Tail1 winner = winner :=
  (root148Step1).trans root148Scan2
theorem root148Scan0 : scan edge (search edge 3) 4 [877] 877 root148Tail0 winner = winner :=
  (root148Step0).trans root148Scan1
set_option maxRecDepth 4096 in
theorem root148Branch : search edge 4 [877] root148Candidates 877 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [877] 877 root148Tail0 winner = winner
  exact root148Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
