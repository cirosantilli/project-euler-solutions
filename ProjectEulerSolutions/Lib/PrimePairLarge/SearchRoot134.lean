import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root134Candidates : List Nat := Data.row773
set_option maxRecDepth 8192 in
theorem root134Candidates_checked : candidates edge 5 0 773 tail135 winner.bound = root134Candidates := by
  apply candidates_eq_row edge 5 0 773 tail135 root134Candidates winner.bound tail135_ordered
  · decide
  · decide
  · funext q
    rfl
def root134Tail64 : List Nat := []
def root134Tail63 : List Nat := 25229 :: root134Tail64
def root134Tail62 : List Nat := 25037 :: root134Tail63
def root134Tail61 : List Nat := 24971 :: root134Tail62
def root134Tail60 : List Nat := 24473 :: root134Tail61
def root134Tail59 : List Nat := 24029 :: root134Tail60
def root134Tail58 : List Nat := 23993 :: root134Tail59
def root134Tail57 : List Nat := 23669 :: root134Tail58
def root134Tail56 : List Nat := 23087 :: root134Tail57
def root134Tail55 : List Nat := 23063 :: root134Tail56
def root134Tail54 : List Nat := 22541 :: root134Tail55
def root134Tail53 : List Nat := 22073 :: root134Tail54
def root134Tail52 : List Nat := 20921 :: root134Tail53
def root134Tail51 : List Nat := 20129 :: root134Tail52
def root134Tail50 : List Nat := 18719 :: root134Tail51
def root134Tail49 : List Nat := 17987 :: root134Tail50
def root134Tail48 : List Nat := 17729 :: root134Tail49
def root134Tail47 : List Nat := 17477 :: root134Tail48
def root134Tail46 : List Nat := 17417 :: root134Tail47
def root134Tail45 : List Nat := 16937 :: root134Tail46
def root134Tail44 : List Nat := 16619 :: root134Tail45
def root134Tail43 : List Nat := 16433 :: root134Tail44
def root134Tail42 : List Nat := 16319 :: root134Tail43
def root134Tail41 : List Nat := 16091 :: root134Tail42
def root134Tail40 : List Nat := 16067 :: root134Tail41
def root134Tail39 : List Nat := 15083 :: root134Tail40
def root134Tail38 : List Nat := 14639 :: root134Tail39
def root134Tail37 : List Nat := 14543 :: root134Tail38
def root134Tail36 : List Nat := 13829 :: root134Tail37
def root134Tail35 : List Nat := 13259 :: root134Tail36
def root134Tail34 : List Nat := 12899 :: root134Tail35
def root134Tail33 : List Nat := 12119 :: root134Tail34
def root134Tail32 : List Nat := 11933 :: root134Tail33
def root134Tail31 : List Nat := 11909 :: root134Tail32
def root134Tail30 : List Nat := 11867 :: root134Tail31
def root134Tail29 : List Nat := 11681 :: root134Tail30
def root134Tail28 : List Nat := 10949 :: root134Tail29
def root134Tail27 : List Nat := 10847 :: root134Tail28
def root134Tail26 : List Nat := 10529 :: root134Tail27
def root134Tail25 : List Nat := 10007 :: root134Tail26
def root134Tail24 : List Nat := 9461 :: root134Tail25
def root134Tail23 : List Nat := 9323 :: root134Tail24
def root134Tail22 : List Nat := 9311 :: root134Tail23
def root134Tail21 : List Nat := 8867 :: root134Tail22
def root134Tail20 : List Nat := 7547 :: root134Tail21
def root134Tail19 : List Nat := 6269 :: root134Tail20
def root134Tail18 : List Nat := 6263 :: root134Tail19
def root134Tail17 : List Nat := 6101 :: root134Tail18
def root134Tail16 : List Nat := 5927 :: root134Tail17
def root134Tail15 : List Nat := 5711 :: root134Tail16
def root134Tail14 : List Nat := 5669 :: root134Tail15
def root134Tail13 : List Nat := 5381 :: root134Tail14
def root134Tail12 : List Nat := 5081 :: root134Tail13
def root134Tail11 : List Nat := 4409 :: root134Tail12
def root134Tail10 : List Nat := 4241 :: root134Tail11
def root134Tail9 : List Nat := 3701 :: root134Tail10
def root134Tail8 : List Nat := 2591 :: root134Tail9
def root134Tail7 : List Nat := 2339 :: root134Tail8
def root134Tail6 : List Nat := 2129 :: root134Tail7
def root134Tail5 : List Nat := 2063 :: root134Tail6
def root134Tail4 : List Nat := 1877 :: root134Tail5
def root134Tail3 : List Nat := 1721 :: root134Tail4
def root134Tail2 : List Nat := 1667 :: root134Tail3
def root134Tail1 : List Nat := 1439 :: root134Tail2
def root134Tail0 : List Nat := 953 :: root134Tail1
def root134Candidates0 : List Nat := [16937]
set_option maxRecDepth 8192 in
theorem root134Candidates0_checked : candidates edge 4 773 953 root134Tail1 winner.bound = root134Candidates0 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
theorem root134Step0 : scan edge (search edge 3) 4 [773] 773 root134Tail0 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail1 winner := by
  rw [root134Tail0, scan, if_neg (by decide), root134Candidates0_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail1
    (if 3 ≤ root134Candidates0.length then search edge 3 [773, 953] root134Candidates0 1726 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates1 : List Nat := [14543]
set_option maxRecDepth 8192 in
theorem root134Candidates1_checked : candidates edge 4 773 1439 root134Tail2 winner.bound = root134Candidates1 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
theorem root134Step1 : scan edge (search edge 3) 4 [773] 773 root134Tail1 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail2 winner := by
  rw [root134Tail1, scan, if_neg (by decide), root134Candidates1_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail2
    (if 3 ≤ root134Candidates1.length then search edge 3 [773, 1439] root134Candidates1 2212 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates2 : List Nat := [5669, 17987, 20129]
set_option maxRecDepth 8192 in
theorem root134Candidates2_checked : candidates edge 4 773 1667 root134Tail3 winner.bound = root134Candidates2 := by
  unfold candidates
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root134Child2 : search edge 3 [773, 1667] root134Candidates2 2440 winner = winner := by decide
theorem root134Step2 : scan edge (search edge 3) 4 [773] 773 root134Tail2 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail3 winner := by
  rw [root134Tail2, scan, if_neg (by decide), root134Candidates2_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail3
    (if 3 ≤ root134Candidates2.length then search edge 3 [773, 1667] root134Candidates2 2440 winner else winner) = _
  rw [if_pos (by decide), root134Child2]
def root134Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root134Candidates3_checked : candidates edge 4 773 1721 root134Tail4 winner.bound = root134Candidates3 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root134Step3 : scan edge (search edge 3) 4 [773] 773 root134Tail3 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail4 winner := by
  rw [root134Tail3, scan, if_neg (by decide), root134Candidates3_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail4
    (if 3 ≤ root134Candidates3.length then search edge 3 [773, 1721] root134Candidates3 2494 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates4 : List Nat := [5711, 23087]
set_option maxRecDepth 8192 in
theorem root134Candidates4_checked : candidates edge 4 773 1877 root134Tail5 winner.bound = root134Candidates4 := by
  unfold candidates
  have row : edge 1877 = (SortedNatTree.contains FastRows.row1877) := by
    change Data.row1877.contains = (SortedNatTree.contains FastRows.row1877)
    exact FastRows.row1877_checked.symm
  rw [row]
  decide
theorem root134Step4 : scan edge (search edge 3) 4 [773] 773 root134Tail4 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail5 winner := by
  rw [root134Tail4, scan, if_neg (by decide), root134Candidates4_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail5
    (if 3 ≤ root134Candidates4.length then search edge 3 [773, 1877] root134Candidates4 2650 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates5 : List Nat := [5711, 18719]
set_option maxRecDepth 8192 in
theorem root134Candidates5_checked : candidates edge 4 773 2063 root134Tail6 winner.bound = root134Candidates5 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root134Step5 : scan edge (search edge 3) 4 [773] 773 root134Tail5 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail6 winner := by
  rw [root134Tail5, scan, if_neg (by decide), root134Candidates5_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail6
    (if 3 ≤ root134Candidates5.length then search edge 3 [773, 2063] root134Candidates5 2836 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates6 : List Nat := [6269]
set_option maxRecDepth 8192 in
theorem root134Candidates6_checked : candidates edge 4 773 2129 root134Tail7 winner.bound = root134Candidates6 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root134Step6 : scan edge (search edge 3) 4 [773] 773 root134Tail6 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail7 winner := by
  rw [root134Tail6, scan, if_neg (by decide), root134Candidates6_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail7
    (if 3 ≤ root134Candidates6.length then search edge 3 [773, 2129] root134Candidates6 2902 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates7 : List Nat := [5081, 6101]
set_option maxRecDepth 8192 in
theorem root134Candidates7_checked : candidates edge 4 773 2339 root134Tail8 winner.bound = root134Candidates7 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root134Step7 : scan edge (search edge 3) 4 [773] 773 root134Tail7 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail8 winner := by
  rw [root134Tail7, scan, if_neg (by decide), root134Candidates7_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail8
    (if 3 ≤ root134Candidates7.length then search edge 3 [773, 2339] root134Candidates7 3112 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates8 : List Nat := [22541]
set_option maxRecDepth 8192 in
theorem root134Candidates8_checked : candidates edge 4 773 2591 root134Tail9 winner.bound = root134Candidates8 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root134Step8 : scan edge (search edge 3) 4 [773] 773 root134Tail8 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail9 winner := by
  rw [root134Tail8, scan, if_neg (by decide), root134Candidates8_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail9
    (if 3 ≤ root134Candidates8.length then search edge 3 [773, 2591] root134Candidates8 3364 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates9 : List Nat := [5081, 11867, 14639]
set_option maxRecDepth 8192 in
theorem root134Candidates9_checked : candidates edge 4 773 3701 root134Tail10 winner.bound = root134Candidates9 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root134Child9 : search edge 3 [773, 3701] root134Candidates9 4474 winner = winner := by decide
theorem root134Step9 : scan edge (search edge 3) 4 [773] 773 root134Tail9 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail10 winner := by
  rw [root134Tail9, scan, if_neg (by decide), root134Candidates9_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail10
    (if 3 ≤ root134Candidates9.length then search edge 3 [773, 3701] root134Candidates9 4474 winner else winner) = _
  rw [if_pos (by decide), root134Child9]
def root134Candidates10 : List Nat := [9323, 10949, 16937]
set_option maxRecDepth 8192 in
theorem root134Candidates10_checked : candidates edge 4 773 4241 root134Tail11 winner.bound = root134Candidates10 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root134Child10 : search edge 3 [773, 4241] root134Candidates10 5014 winner = winner := by decide
theorem root134Step10 : scan edge (search edge 3) 4 [773] 773 root134Tail10 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail11 winner := by
  rw [root134Tail10, scan, if_neg (by decide), root134Candidates10_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail11
    (if 3 ≤ root134Candidates10.length then search edge 3 [773, 4241] root134Candidates10 5014 winner else winner) = _
  rw [if_pos (by decide), root134Child10]
def root134Candidates11 : List Nat := [18719]
set_option maxRecDepth 8192 in
theorem root134Candidates11_checked : candidates edge 4 773 4409 root134Tail12 winner.bound = root134Candidates11 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root134Step11 : scan edge (search edge 3) 4 [773] 773 root134Tail11 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail12 winner := by
  rw [root134Tail11, scan, if_neg (by decide), root134Candidates11_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail12
    (if 3 ≤ root134Candidates11.length then search edge 3 [773, 4409] root134Candidates11 5182 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root134Candidates12_checked : candidates edge 4 773 5081 root134Tail13 winner.bound = root134Candidates12 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root134Step12 : scan edge (search edge 3) 4 [773] 773 root134Tail12 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail13 winner := by
  rw [root134Tail12, scan, if_neg (by decide), root134Candidates12_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail13
    (if 3 ≤ root134Candidates12.length then search edge 3 [773, 5081] root134Candidates12 5854 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root134Candidates13_checked : candidates edge 4 773 5381 root134Tail14 winner.bound = root134Candidates13 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root134Step13 : scan edge (search edge 3) 4 [773] 773 root134Tail13 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail14 winner := by
  rw [root134Tail13, scan, if_neg (by decide), root134Candidates13_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail14
    (if 3 ≤ root134Candidates13.length then search edge 3 [773, 5381] root134Candidates13 6154 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates14 : List Nat := [13829, 16937]
set_option maxRecDepth 8192 in
theorem root134Candidates14_checked : candidates edge 4 773 5669 root134Tail15 winner.bound = root134Candidates14 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root134Step14 : scan edge (search edge 3) 4 [773] 773 root134Tail14 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail15 winner := by
  rw [root134Tail14, scan, if_neg (by decide), root134Candidates14_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail15
    (if 3 ≤ root134Candidates14.length then search edge 3 [773, 5669] root134Candidates14 6442 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root134Candidates15_checked : candidates edge 4 773 5711 root134Tail16 winner.bound = root134Candidates15 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root134Step15 : scan edge (search edge 3) 4 [773] 773 root134Tail15 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail16 winner := by
  rw [root134Tail15, scan, if_neg (by decide), root134Candidates15_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail16
    (if 3 ≤ root134Candidates15.length then search edge 3 [773, 5711] root134Candidates15 6484 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root134Candidates16_checked : candidates edge 4 773 5927 root134Tail17 winner.bound = root134Candidates16 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root134Step16 : scan edge (search edge 3) 4 [773] 773 root134Tail16 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail17 winner := by
  rw [root134Tail16, scan, if_neg (by decide), root134Candidates16_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail17
    (if 3 ≤ root134Candidates16.length then search edge 3 [773, 5927] root134Candidates16 6700 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates17 : List Nat := [11867, 16433]
set_option maxRecDepth 8192 in
theorem root134Candidates17_checked : candidates edge 4 773 6101 root134Tail18 winner.bound = root134Candidates17 := by
  unfold candidates
  have row : edge 6101 = (SortedNatTree.contains FastRows.row6101) := by
    change Data.row6101.contains = (SortedNatTree.contains FastRows.row6101)
    exact FastRows.row6101_checked.symm
  rw [row]
  decide
theorem root134Step17 : scan edge (search edge 3) 4 [773] 773 root134Tail17 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail18 winner := by
  rw [root134Tail17, scan, if_neg (by decide), root134Candidates17_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail18
    (if 3 ≤ root134Candidates17.length then search edge 3 [773, 6101] root134Candidates17 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates18 : List Nat := [16619]
set_option maxRecDepth 8192 in
theorem root134Candidates18_checked : candidates edge 4 773 6263 root134Tail19 winner.bound = root134Candidates18 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root134Step18 : scan edge (search edge 3) 4 [773] 773 root134Tail18 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail19 winner := by
  rw [root134Tail18, scan, if_neg (by decide), root134Candidates18_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail19
    (if 3 ≤ root134Candidates18.length then search edge 3 [773, 6263] root134Candidates18 7036 winner else winner) = _
  rw [if_neg (by decide)]
def root134Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root134Candidates19_checked : candidates edge 4 773 6269 root134Tail20 winner.bound = root134Candidates19 := by
  unfold candidates
  have row : edge 6269 = (SortedNatTree.contains FastRows.row6269) := by
    change Data.row6269.contains = (SortedNatTree.contains FastRows.row6269)
    exact FastRows.row6269_checked.symm
  rw [row]
  decide
theorem root134Step19 : scan edge (search edge 3) 4 [773] 773 root134Tail19 winner =
    scan edge (search edge 3) 4 [773] 773 root134Tail20 winner := by
  rw [root134Tail19, scan, if_neg (by decide), root134Candidates19_checked]
  change scan edge (search edge 3) 4 [773] 773 root134Tail20
    (if 3 ≤ root134Candidates19.length then search edge 3 [773, 6269] root134Candidates19 7042 winner else winner) = _
  rw [if_neg (by decide)]
theorem root134Scan20 : scan edge (search edge 3) 4 [773] 773 root134Tail20 winner = winner := by rw [root134Tail20, scan, if_pos (by decide)]
theorem root134Scan19 : scan edge (search edge 3) 4 [773] 773 root134Tail19 winner = winner :=
  (root134Step19).trans root134Scan20
theorem root134Scan18 : scan edge (search edge 3) 4 [773] 773 root134Tail18 winner = winner :=
  (root134Step18).trans root134Scan19
theorem root134Scan17 : scan edge (search edge 3) 4 [773] 773 root134Tail17 winner = winner :=
  (root134Step17).trans root134Scan18
theorem root134Scan16 : scan edge (search edge 3) 4 [773] 773 root134Tail16 winner = winner :=
  (root134Step16).trans root134Scan17
theorem root134Scan15 : scan edge (search edge 3) 4 [773] 773 root134Tail15 winner = winner :=
  (root134Step15).trans root134Scan16
theorem root134Scan14 : scan edge (search edge 3) 4 [773] 773 root134Tail14 winner = winner :=
  (root134Step14).trans root134Scan15
theorem root134Scan13 : scan edge (search edge 3) 4 [773] 773 root134Tail13 winner = winner :=
  (root134Step13).trans root134Scan14
theorem root134Scan12 : scan edge (search edge 3) 4 [773] 773 root134Tail12 winner = winner :=
  (root134Step12).trans root134Scan13
theorem root134Scan11 : scan edge (search edge 3) 4 [773] 773 root134Tail11 winner = winner :=
  (root134Step11).trans root134Scan12
theorem root134Scan10 : scan edge (search edge 3) 4 [773] 773 root134Tail10 winner = winner :=
  (root134Step10).trans root134Scan11
theorem root134Scan9 : scan edge (search edge 3) 4 [773] 773 root134Tail9 winner = winner :=
  (root134Step9).trans root134Scan10
theorem root134Scan8 : scan edge (search edge 3) 4 [773] 773 root134Tail8 winner = winner :=
  (root134Step8).trans root134Scan9
theorem root134Scan7 : scan edge (search edge 3) 4 [773] 773 root134Tail7 winner = winner :=
  (root134Step7).trans root134Scan8
theorem root134Scan6 : scan edge (search edge 3) 4 [773] 773 root134Tail6 winner = winner :=
  (root134Step6).trans root134Scan7
theorem root134Scan5 : scan edge (search edge 3) 4 [773] 773 root134Tail5 winner = winner :=
  (root134Step5).trans root134Scan6
theorem root134Scan4 : scan edge (search edge 3) 4 [773] 773 root134Tail4 winner = winner :=
  (root134Step4).trans root134Scan5
theorem root134Scan3 : scan edge (search edge 3) 4 [773] 773 root134Tail3 winner = winner :=
  (root134Step3).trans root134Scan4
theorem root134Scan2 : scan edge (search edge 3) 4 [773] 773 root134Tail2 winner = winner :=
  (root134Step2).trans root134Scan3
theorem root134Scan1 : scan edge (search edge 3) 4 [773] 773 root134Tail1 winner = winner :=
  (root134Step1).trans root134Scan2
theorem root134Scan0 : scan edge (search edge 3) 4 [773] 773 root134Tail0 winner = winner :=
  (root134Step0).trans root134Scan1
set_option maxRecDepth 4096 in
theorem root134Branch : search edge 4 [773] root134Candidates 773 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [773] 773 root134Tail0 winner = winner
  exact root134Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
