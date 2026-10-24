import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root37Candidates : List Nat := Data.row173
set_option maxRecDepth 8192 in
theorem root37Candidates_checked : candidates edge 5 0 173 tail38 winner.bound = root37Candidates := by
  apply candidates_eq_row edge 5 0 173 tail38 root37Candidates winner.bound tail38_ordered
  · decide
  · decide
  · funext q
    rfl
def root37Tail69 : List Nat := []
def root37Tail68 : List Nat := 25799 :: root37Tail69
def root37Tail67 : List Nat := 25643 :: root37Tail68
def root37Tail66 : List Nat := 25097 :: root37Tail67
def root37Tail65 : List Nat := 25013 :: root37Tail66
def root37Tail64 : List Nat := 24137 :: root37Tail65
def root37Tail63 : List Nat := 23447 :: root37Tail64
def root37Tail62 : List Nat := 23081 :: root37Tail63
def root37Tail61 : List Nat := 22259 :: root37Tail62
def root37Tail60 : List Nat := 22079 :: root37Tail61
def root37Tail59 : List Nat := 22073 :: root37Tail60
def root37Tail58 : List Nat := 21191 :: root37Tail59
def root37Tail57 : List Nat := 20549 :: root37Tail58
def root37Tail56 : List Nat := 20117 :: root37Tail57
def root37Tail55 : List Nat := 19949 :: root37Tail56
def root37Tail54 : List Nat := 19853 :: root37Tail55
def root37Tail53 : List Nat := 19571 :: root37Tail54
def root37Tail52 : List Nat := 19301 :: root37Tail53
def root37Tail51 : List Nat := 18131 :: root37Tail52
def root37Tail50 : List Nat := 17903 :: root37Tail51
def root37Tail49 : List Nat := 17123 :: root37Tail50
def root37Tail48 : List Nat := 16547 :: root37Tail49
def root37Tail47 : List Nat := 16433 :: root37Tail48
def root37Tail46 : List Nat := 16223 :: root37Tail47
def root37Tail45 : List Nat := 15443 :: root37Tail46
def root37Tail44 : List Nat := 14741 :: root37Tail45
def root37Tail43 : List Nat := 14543 :: root37Tail44
def root37Tail42 : List Nat := 13619 :: root37Tail43
def root37Tail41 : List Nat := 13523 :: root37Tail42
def root37Tail40 : List Nat := 13229 :: root37Tail41
def root37Tail39 : List Nat := 13187 :: root37Tail40
def root37Tail38 : List Nat := 12791 :: root37Tail39
def root37Tail37 : List Nat := 12611 :: root37Tail38
def root37Tail36 : List Nat := 12527 :: root37Tail37
def root37Tail35 : List Nat := 12473 :: root37Tail36
def root37Tail34 : List Nat := 12203 :: root37Tail35
def root37Tail33 : List Nat := 11633 :: root37Tail34
def root37Tail32 : List Nat := 10739 :: root37Tail33
def root37Tail31 : List Nat := 10427 :: root37Tail32
def root37Tail30 : List Nat := 10343 :: root37Tail31
def root37Tail29 : List Nat := 9587 :: root37Tail30
def root37Tail28 : List Nat := 9461 :: root37Tail29
def root37Tail27 : List Nat := 9209 :: root37Tail28
def root37Tail26 : List Nat := 9173 :: root37Tail27
def root37Tail25 : List Nat := 9041 :: root37Tail26
def root37Tail24 : List Nat := 8609 :: root37Tail25
def root37Tail23 : List Nat := 8171 :: root37Tail24
def root37Tail22 : List Nat := 8117 :: root37Tail23
def root37Tail21 : List Nat := 8039 :: root37Tail22
def root37Tail20 : List Nat := 7559 :: root37Tail21
def root37Tail19 : List Nat := 7529 :: root37Tail20
def root37Tail18 : List Nat := 7331 :: root37Tail19
def root37Tail17 : List Nat := 5813 :: root37Tail18
def root37Tail16 : List Nat := 5807 :: root37Tail17
def root37Tail15 : List Nat := 5333 :: root37Tail16
def root37Tail14 : List Nat := 4937 :: root37Tail15
def root37Tail13 : List Nat := 4787 :: root37Tail14
def root37Tail12 : List Nat := 4721 :: root37Tail13
def root37Tail11 : List Nat := 4673 :: root37Tail12
def root37Tail10 : List Nat := 3929 :: root37Tail11
def root37Tail9 : List Nat := 3041 :: root37Tail10
def root37Tail8 : List Nat := 2333 :: root37Tail9
def root37Tail7 : List Nat := 2039 :: root37Tail8
def root37Tail6 : List Nat := 1091 :: root37Tail7
def root37Tail5 : List Nat := 743 :: root37Tail6
def root37Tail4 : List Nat := 659 :: root37Tail5
def root37Tail3 : List Nat := 431 :: root37Tail4
def root37Tail2 : List Nat := 347 :: root37Tail3
def root37Tail1 : List Nat := 293 :: root37Tail2
def root37Tail0 : List Nat := 191 :: root37Tail1
def root37Candidates0 : List Nat := [4673, 5813, 9461, 13187, 19853, 19949, 22079]
set_option maxRecDepth 8192 in
theorem root37Candidates0_checked : candidates edge 4 173 191 root37Tail1 winner.bound = root37Candidates0 := by
  unfold candidates
  have row : edge 191 = (SortedNatTree.contains FastRows.row191) := by
    change Data.row191.contains = (SortedNatTree.contains FastRows.row191)
    exact FastRows.row191_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root37Child0 : search edge 3 [173, 191] root37Candidates0 364 winner = winner := by decide
theorem root37Step0 : scan edge (search edge 3) 4 [173] 173 root37Tail0 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail1 winner := by
  rw [root37Tail0, scan, if_neg (by decide), root37Candidates0_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail1
    (if 3 ≤ root37Candidates0.length then search edge 3 [173, 191] root37Candidates0 364 winner else winner) = _
  rw [if_pos (by decide), root37Child0]
def root37Candidates1 : List Nat := [16223, 17903, 23447]
set_option maxRecDepth 8192 in
theorem root37Candidates1_checked : candidates edge 4 173 293 root37Tail2 winner.bound = root37Candidates1 := by
  unfold candidates
  have row : edge 293 = (SortedNatTree.contains FastRows.row293) := by
    change Data.row293.contains = (SortedNatTree.contains FastRows.row293)
    exact FastRows.row293_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root37Child1 : search edge 3 [173, 293] root37Candidates1 466 winner = winner := by decide
theorem root37Step1 : scan edge (search edge 3) 4 [173] 173 root37Tail1 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail2 winner := by
  rw [root37Tail1, scan, if_neg (by decide), root37Candidates1_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail2
    (if 3 ≤ root37Candidates1.length then search edge 3 [173, 293] root37Candidates1 466 winner else winner) = _
  rw [if_pos (by decide), root37Child1]
def root37Candidates2 : List Nat := [7559, 19571, 22073]
set_option maxRecDepth 8192 in
theorem root37Candidates2_checked : candidates edge 4 173 347 root37Tail3 winner.bound = root37Candidates2 := by
  unfold candidates
  have row : edge 347 = (SortedNatTree.contains FastRows.row347) := by
    change Data.row347.contains = (SortedNatTree.contains FastRows.row347)
    exact FastRows.row347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root37Child2 : search edge 3 [173, 347] root37Candidates2 520 winner = winner := by decide
theorem root37Step2 : scan edge (search edge 3) 4 [173] 173 root37Tail2 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail3 winner := by
  rw [root37Tail2, scan, if_neg (by decide), root37Candidates2_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail3
    (if 3 ≤ root37Candidates2.length then search edge 3 [173, 347] root37Candidates2 520 winner else winner) = _
  rw [if_pos (by decide), root37Child2]
def root37Candidates3 : List Nat := [12203, 19301, 20117, 25013]
set_option maxRecDepth 8192 in
theorem root37Candidates3_checked : candidates edge 4 173 431 root37Tail4 winner.bound = root37Candidates3 := by
  unfold candidates
  have row : edge 431 = (SortedNatTree.contains FastRows.row431) := by
    change Data.row431.contains = (SortedNatTree.contains FastRows.row431)
    exact FastRows.row431_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root37Child3 : search edge 3 [173, 431] root37Candidates3 604 winner = winner := by decide
theorem root37Step3 : scan edge (search edge 3) 4 [173] 173 root37Tail3 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail4 winner := by
  rw [root37Tail3, scan, if_neg (by decide), root37Candidates3_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail4
    (if 3 ≤ root37Candidates3.length then search edge 3 [173, 431] root37Candidates3 604 winner else winner) = _
  rw [if_pos (by decide), root37Child3]
def root37Candidates4 : List Nat := [12527, 16547]
set_option maxRecDepth 8192 in
theorem root37Candidates4_checked : candidates edge 4 173 659 root37Tail5 winner.bound = root37Candidates4 := by
  unfold candidates
  have row : edge 659 = (SortedNatTree.contains FastRows.row659) := by
    change Data.row659.contains = (SortedNatTree.contains FastRows.row659)
    exact FastRows.row659_checked.symm
  rw [row]
  decide
theorem root37Step4 : scan edge (search edge 3) 4 [173] 173 root37Tail4 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail5 winner := by
  rw [root37Tail4, scan, if_neg (by decide), root37Candidates4_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail5
    (if 3 ≤ root37Candidates4.length then search edge 3 [173, 659] root37Candidates4 832 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates5 : List Nat := [8117]
set_option maxRecDepth 8192 in
theorem root37Candidates5_checked : candidates edge 4 173 743 root37Tail6 winner.bound = root37Candidates5 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
theorem root37Step5 : scan edge (search edge 3) 4 [173] 173 root37Tail5 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail6 winner := by
  rw [root37Tail5, scan, if_neg (by decide), root37Candidates5_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail6
    (if 3 ≤ root37Candidates5.length then search edge 3 [173, 743] root37Candidates5 916 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates6 : List Nat := [3041, 9173]
set_option maxRecDepth 8192 in
theorem root37Candidates6_checked : candidates edge 4 173 1091 root37Tail7 winner.bound = root37Candidates6 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
theorem root37Step6 : scan edge (search edge 3) 4 [173] 173 root37Tail6 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail7 winner := by
  rw [root37Tail6, scan, if_neg (by decide), root37Candidates6_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail7
    (if 3 ≤ root37Candidates6.length then search edge 3 [173, 1091] root37Candidates6 1264 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates7 : List Nat := [4937]
set_option maxRecDepth 8192 in
theorem root37Candidates7_checked : candidates edge 4 173 2039 root37Tail8 winner.bound = root37Candidates7 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
theorem root37Step7 : scan edge (search edge 3) 4 [173] 173 root37Tail7 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail8 winner := by
  rw [root37Tail7, scan, if_neg (by decide), root37Candidates7_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail8
    (if 3 ≤ root37Candidates7.length then search edge 3 [173, 2039] root37Candidates7 2212 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates8 : List Nat := [12203, 18131, 19301, 19949]
set_option maxRecDepth 8192 in
theorem root37Candidates8_checked : candidates edge 4 173 2333 root37Tail9 winner.bound = root37Candidates8 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root37Child8 : search edge 3 [173, 2333] root37Candidates8 2506 winner = winner := by decide
theorem root37Step8 : scan edge (search edge 3) 4 [173] 173 root37Tail8 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail9 winner := by
  rw [root37Tail8, scan, if_neg (by decide), root37Candidates8_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail9
    (if 3 ≤ root37Candidates8.length then search edge 3 [173, 2333] root37Candidates8 2506 winner else winner) = _
  rw [if_pos (by decide), root37Child8]
def root37Candidates9 : List Nat := [7529]
set_option maxRecDepth 8192 in
theorem root37Candidates9_checked : candidates edge 4 173 3041 root37Tail10 winner.bound = root37Candidates9 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root37Step9 : scan edge (search edge 3) 4 [173] 173 root37Tail9 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail10 winner := by
  rw [root37Tail9, scan, if_neg (by decide), root37Candidates9_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail10
    (if 3 ≤ root37Candidates9.length then search edge 3 [173, 3041] root37Candidates9 3214 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates10 : List Nat := [9209]
set_option maxRecDepth 8192 in
theorem root37Candidates10_checked : candidates edge 4 173 3929 root37Tail11 winner.bound = root37Candidates10 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root37Step10 : scan edge (search edge 3) 4 [173] 173 root37Tail10 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail11 winner := by
  rw [root37Tail10, scan, if_neg (by decide), root37Candidates10_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail11
    (if 3 ≤ root37Candidates10.length then search edge 3 [173, 3929] root37Candidates10 4102 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root37Candidates11_checked : candidates edge 4 173 4673 root37Tail12 winner.bound = root37Candidates11 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root37Step11 : scan edge (search edge 3) 4 [173] 173 root37Tail11 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail12 winner := by
  rw [root37Tail11, scan, if_neg (by decide), root37Candidates11_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail12
    (if 3 ≤ root37Candidates11.length then search edge 3 [173, 4673] root37Candidates11 4846 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates12 : List Nat := [5813, 9209]
set_option maxRecDepth 8192 in
theorem root37Candidates12_checked : candidates edge 4 173 4721 root37Tail13 winner.bound = root37Candidates12 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root37Step12 : scan edge (search edge 3) 4 [173] 173 root37Tail12 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail13 winner := by
  rw [root37Tail12, scan, if_neg (by decide), root37Candidates12_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail13
    (if 3 ≤ root37Candidates12.length then search edge 3 [173, 4721] root37Candidates12 4894 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root37Candidates13_checked : candidates edge 4 173 4787 root37Tail14 winner.bound = root37Candidates13 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root37Step13 : scan edge (search edge 3) 4 [173] 173 root37Tail13 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail14 winner := by
  rw [root37Tail13, scan, if_neg (by decide), root37Candidates13_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail14
    (if 3 ≤ root37Candidates13.length then search edge 3 [173, 4787] root37Candidates13 4960 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates14 : List Nat := [13523]
set_option maxRecDepth 8192 in
theorem root37Candidates14_checked : candidates edge 4 173 4937 root37Tail15 winner.bound = root37Candidates14 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root37Step14 : scan edge (search edge 3) 4 [173] 173 root37Tail14 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail15 winner := by
  rw [root37Tail14, scan, if_neg (by decide), root37Candidates14_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail15
    (if 3 ≤ root37Candidates14.length then search edge 3 [173, 4937] root37Candidates14 5110 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root37Candidates15_checked : candidates edge 4 173 5333 root37Tail16 winner.bound = root37Candidates15 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root37Step15 : scan edge (search edge 3) 4 [173] 173 root37Tail15 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail16 winner := by
  rw [root37Tail15, scan, if_neg (by decide), root37Candidates15_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail16
    (if 3 ≤ root37Candidates15.length then search edge 3 [173, 5333] root37Candidates15 5506 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates16 : List Nat := [9587]
set_option maxRecDepth 8192 in
theorem root37Candidates16_checked : candidates edge 4 173 5807 root37Tail17 winner.bound = root37Candidates16 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root37Step16 : scan edge (search edge 3) 4 [173] 173 root37Tail16 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail17 winner := by
  rw [root37Tail16, scan, if_neg (by decide), root37Candidates16_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail17
    (if 3 ≤ root37Candidates16.length then search edge 3 [173, 5807] root37Candidates16 5980 winner else winner) = _
  rw [if_neg (by decide)]
def root37Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root37Candidates17_checked : candidates edge 4 173 5813 root37Tail18 winner.bound = root37Candidates17 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root37Step17 : scan edge (search edge 3) 4 [173] 173 root37Tail17 winner =
    scan edge (search edge 3) 4 [173] 173 root37Tail18 winner := by
  rw [root37Tail17, scan, if_neg (by decide), root37Candidates17_checked]
  change scan edge (search edge 3) 4 [173] 173 root37Tail18
    (if 3 ≤ root37Candidates17.length then search edge 3 [173, 5813] root37Candidates17 5986 winner else winner) = _
  rw [if_neg (by decide)]
theorem root37Scan18 : scan edge (search edge 3) 4 [173] 173 root37Tail18 winner = winner := by rw [root37Tail18, scan, if_pos (by decide)]
theorem root37Scan17 : scan edge (search edge 3) 4 [173] 173 root37Tail17 winner = winner :=
  (root37Step17).trans root37Scan18
theorem root37Scan16 : scan edge (search edge 3) 4 [173] 173 root37Tail16 winner = winner :=
  (root37Step16).trans root37Scan17
theorem root37Scan15 : scan edge (search edge 3) 4 [173] 173 root37Tail15 winner = winner :=
  (root37Step15).trans root37Scan16
theorem root37Scan14 : scan edge (search edge 3) 4 [173] 173 root37Tail14 winner = winner :=
  (root37Step14).trans root37Scan15
theorem root37Scan13 : scan edge (search edge 3) 4 [173] 173 root37Tail13 winner = winner :=
  (root37Step13).trans root37Scan14
theorem root37Scan12 : scan edge (search edge 3) 4 [173] 173 root37Tail12 winner = winner :=
  (root37Step12).trans root37Scan13
theorem root37Scan11 : scan edge (search edge 3) 4 [173] 173 root37Tail11 winner = winner :=
  (root37Step11).trans root37Scan12
theorem root37Scan10 : scan edge (search edge 3) 4 [173] 173 root37Tail10 winner = winner :=
  (root37Step10).trans root37Scan11
theorem root37Scan9 : scan edge (search edge 3) 4 [173] 173 root37Tail9 winner = winner :=
  (root37Step9).trans root37Scan10
theorem root37Scan8 : scan edge (search edge 3) 4 [173] 173 root37Tail8 winner = winner :=
  (root37Step8).trans root37Scan9
theorem root37Scan7 : scan edge (search edge 3) 4 [173] 173 root37Tail7 winner = winner :=
  (root37Step7).trans root37Scan8
theorem root37Scan6 : scan edge (search edge 3) 4 [173] 173 root37Tail6 winner = winner :=
  (root37Step6).trans root37Scan7
theorem root37Scan5 : scan edge (search edge 3) 4 [173] 173 root37Tail5 winner = winner :=
  (root37Step5).trans root37Scan6
theorem root37Scan4 : scan edge (search edge 3) 4 [173] 173 root37Tail4 winner = winner :=
  (root37Step4).trans root37Scan5
theorem root37Scan3 : scan edge (search edge 3) 4 [173] 173 root37Tail3 winner = winner :=
  (root37Step3).trans root37Scan4
theorem root37Scan2 : scan edge (search edge 3) 4 [173] 173 root37Tail2 winner = winner :=
  (root37Step2).trans root37Scan3
theorem root37Scan1 : scan edge (search edge 3) 4 [173] 173 root37Tail1 winner = winner :=
  (root37Step1).trans root37Scan2
theorem root37Scan0 : scan edge (search edge 3) 4 [173] 173 root37Tail0 winner = winner :=
  (root37Step0).trans root37Scan1
set_option maxRecDepth 4096 in
theorem root37Branch : search edge 4 [173] root37Candidates 173 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [173] 173 root37Tail0 winner = winner
  exact root37Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
