import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root61Candidates : List Nat := Data.row311
set_option maxRecDepth 8192 in
theorem root61Candidates_checked : candidates edge 5 0 311 tail62 winner.bound = root61Candidates := by
  apply candidates_eq_row edge 5 0 311 tail62 root61Candidates winner.bound tail62_ordered
  · decide
  · decide
  · funext q
    rfl
def root61Tail73 : List Nat := []
def root61Tail72 : List Nat := 25253 :: root61Tail73
def root61Tail71 : List Nat := 25247 :: root61Tail72
def root61Tail70 : List Nat := 24971 :: root61Tail71
def root61Tail69 : List Nat := 24851 :: root61Tail70
def root61Tail68 : List Nat := 24809 :: root61Tail69
def root61Tail67 : List Nat := 24593 :: root61Tail68
def root61Tail66 : List Nat := 22943 :: root61Tail67
def root61Tail65 : List Nat := 22811 :: root61Tail66
def root61Tail64 : List Nat := 22727 :: root61Tail65
def root61Tail63 : List Nat := 22643 :: root61Tail64
def root61Tail62 : List Nat := 22271 :: root61Tail63
def root61Tail61 : List Nat := 22031 :: root61Tail62
def root61Tail60 : List Nat := 21803 :: root61Tail61
def root61Tail59 : List Nat := 21557 :: root61Tail60
def root61Tail58 : List Nat := 20543 :: root61Tail59
def root61Tail57 : List Nat := 19013 :: root61Tail58
def root61Tail56 : List Nat := 18911 :: root61Tail57
def root61Tail55 : List Nat := 18869 :: root61Tail56
def root61Tail54 : List Nat := 18587 :: root61Tail55
def root61Tail53 : List Nat := 17579 :: root61Tail54
def root61Tail52 : List Nat := 17477 :: root61Tail53
def root61Tail51 : List Nat := 17351 :: root61Tail52
def root61Tail50 : List Nat := 16871 :: root61Tail51
def root61Tail49 : List Nat := 16139 :: root61Tail50
def root61Tail48 : List Nat := 15629 :: root61Tail49
def root61Tail47 : List Nat := 15401 :: root61Tail48
def root61Tail46 : List Nat := 14747 :: root61Tail47
def root61Tail45 : List Nat := 13829 :: root61Tail46
def root61Tail44 : List Nat := 13217 :: root61Tail45
def root61Tail43 : List Nat := 12539 :: root61Tail44
def root61Tail42 : List Nat := 11681 :: root61Tail43
def root61Tail41 : List Nat := 11657 :: root61Tail42
def root61Tail40 : List Nat := 11351 :: root61Tail41
def root61Tail39 : List Nat := 11243 :: root61Tail40
def root61Tail38 : List Nat := 11177 :: root61Tail39
def root61Tail37 : List Nat := 10631 :: root61Tail38
def root61Tail36 : List Nat := 10091 :: root61Tail37
def root61Tail35 : List Nat := 9923 :: root61Tail36
def root61Tail34 : List Nat := 8867 :: root61Tail35
def root61Tail33 : List Nat := 8807 :: root61Tail34
def root61Tail32 : List Nat := 8699 :: root61Tail33
def root61Tail31 : List Nat := 8663 :: root61Tail32
def root61Tail30 : List Nat := 8387 :: root61Tail31
def root61Tail29 : List Nat := 8237 :: root61Tail30
def root61Tail28 : List Nat := 8069 :: root61Tail29
def root61Tail27 : List Nat := 7559 :: root61Tail28
def root61Tail26 : List Nat := 7523 :: root61Tail27
def root61Tail25 : List Nat := 7211 :: root61Tail26
def root61Tail24 : List Nat := 7013 :: root61Tail25
def root61Tail23 : List Nat := 6761 :: root61Tail24
def root61Tail22 : List Nat := 6263 :: root61Tail23
def root61Tail21 : List Nat := 6131 :: root61Tail22
def root61Tail20 : List Nat := 6011 :: root61Tail21
def root61Tail19 : List Nat := 5279 :: root61Tail20
def root61Tail18 : List Nat := 5237 :: root61Tail19
def root61Tail17 : List Nat := 5171 :: root61Tail18
def root61Tail16 : List Nat := 4967 :: root61Tail17
def root61Tail15 : List Nat := 4871 :: root61Tail16
def root61Tail14 : List Nat := 4019 :: root61Tail15
def root61Tail13 : List Nat := 3863 :: root61Tail14
def root61Tail12 : List Nat := 2909 :: root61Tail13
def root61Tail11 : List Nat := 2609 :: root61Tail12
def root61Tail10 : List Nat := 2441 :: root61Tail11
def root61Tail9 : List Nat := 2381 :: root61Tail10
def root61Tail8 : List Nat := 2141 :: root61Tail9
def root61Tail7 : List Nat := 2111 :: root61Tail8
def root61Tail6 : List Nat := 1223 :: root61Tail7
def root61Tail5 : List Nat := 881 :: root61Tail6
def root61Tail4 : List Nat := 827 :: root61Tail5
def root61Tail3 : List Nat := 821 :: root61Tail4
def root61Tail2 : List Nat := 677 :: root61Tail3
def root61Tail1 : List Nat := 653 :: root61Tail2
def root61Tail0 : List Nat := 359 :: root61Tail1
def root61Candidates0 : List Nat := [11177]
set_option maxRecDepth 8192 in
theorem root61Candidates0_checked : candidates edge 4 311 359 root61Tail1 winner.bound = root61Candidates0 := by
  unfold candidates
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row]
  decide
theorem root61Step0 : scan edge (search edge 3) 4 [311] 311 root61Tail0 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail1 winner := by
  rw [root61Tail0, scan, if_neg (by decide), root61Candidates0_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail1
    (if 3 ≤ root61Candidates0.length then search edge 3 [311, 359] root61Candidates0 670 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates1 : List Nat := [7211, 11681]
set_option maxRecDepth 8192 in
theorem root61Candidates1_checked : candidates edge 4 311 653 root61Tail2 winner.bound = root61Candidates1 := by
  unfold candidates
  have row : edge 653 = (SortedNatTree.contains FastRows.row653) := by
    change Data.row653.contains = (SortedNatTree.contains FastRows.row653)
    exact FastRows.row653_checked.symm
  rw [row]
  decide
theorem root61Step1 : scan edge (search edge 3) 4 [311] 311 root61Tail1 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail2 winner := by
  rw [root61Tail1, scan, if_neg (by decide), root61Candidates1_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail2
    (if 3 ≤ root61Candidates1.length then search edge 3 [311, 653] root61Candidates1 964 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates2 : List Nat := [827, 3863, 13829, 22943]
set_option maxRecDepth 8192 in
theorem root61Candidates2_checked : candidates edge 4 311 677 root61Tail3 winner.bound = root61Candidates2 := by
  unfold candidates
  have row : edge 677 = (SortedNatTree.contains FastRows.row677) := by
    change Data.row677.contains = (SortedNatTree.contains FastRows.row677)
    exact FastRows.row677_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child2 : search edge 3 [311, 677] root61Candidates2 988 winner = winner := by decide
theorem root61Step2 : scan edge (search edge 3) 4 [311] 311 root61Tail2 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail3 winner := by
  rw [root61Tail2, scan, if_neg (by decide), root61Candidates2_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail3
    (if 3 ≤ root61Candidates2.length then search edge 3 [311, 677] root61Candidates2 988 winner else winner) = _
  rw [if_pos (by decide), root61Child2]
def root61Candidates3 : List Nat := [827, 7523, 8663, 22727]
set_option maxRecDepth 8192 in
theorem root61Candidates3_checked : candidates edge 4 311 821 root61Tail4 winner.bound = root61Candidates3 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child3 : search edge 3 [311, 821] root61Candidates3 1132 winner = winner := by decide
theorem root61Step3 : scan edge (search edge 3) 4 [311] 311 root61Tail3 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail4 winner := by
  rw [root61Tail3, scan, if_neg (by decide), root61Candidates3_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail4
    (if 3 ≤ root61Candidates3.length then search edge 3 [311, 821] root61Candidates3 1132 winner else winner) = _
  rw [if_pos (by decide), root61Child3]
def root61Candidates4 : List Nat := [2909, 8663, 10631, 14747, 17477, 20543, 22643]
set_option maxRecDepth 8192 in
theorem root61Candidates4_checked : candidates edge 4 311 827 root61Tail5 winner.bound = root61Candidates4 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child4 : search edge 3 [311, 827] root61Candidates4 1138 winner = winner := by decide
theorem root61Step4 : scan edge (search edge 3) 4 [311] 311 root61Tail4 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail5 winner := by
  rw [root61Tail4, scan, if_neg (by decide), root61Candidates4_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail5
    (if 3 ≤ root61Candidates4.length then search edge 3 [311, 827] root61Candidates4 1138 winner else winner) = _
  rw [if_pos (by decide), root61Child4]
def root61Candidates5 : List Nat := [11351, 12539, 22943]
set_option maxRecDepth 8192 in
theorem root61Candidates5_checked : candidates edge 4 311 881 root61Tail6 winner.bound = root61Candidates5 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child5 : search edge 3 [311, 881] root61Candidates5 1192 winner = winner := by decide
theorem root61Step5 : scan edge (search edge 3) 4 [311] 311 root61Tail5 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail6 winner := by
  rw [root61Tail5, scan, if_neg (by decide), root61Candidates5_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail6
    (if 3 ≤ root61Candidates5.length then search edge 3 [311, 881] root61Candidates5 1192 winner else winner) = _
  rw [if_pos (by decide), root61Child5]
def root61Candidates6 : List Nat := [2141, 6131, 6761, 7523, 7559, 8867, 22727]
set_option maxRecDepth 8192 in
theorem root61Candidates6_checked : candidates edge 4 311 1223 root61Tail7 winner.bound = root61Candidates6 := by
  unfold candidates
  have row : edge 1223 = (SortedNatTree.contains FastRows.row1223) := by
    change Data.row1223.contains = (SortedNatTree.contains FastRows.row1223)
    exact FastRows.row1223_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child6 : search edge 3 [311, 1223] root61Candidates6 1534 winner = winner := by decide
theorem root61Step6 : scan edge (search edge 3) 4 [311] 311 root61Tail6 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail7 winner := by
  rw [root61Tail6, scan, if_neg (by decide), root61Candidates6_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail7
    (if 3 ≤ root61Candidates6.length then search edge 3 [311, 1223] root61Candidates6 1534 winner else winner) = _
  rw [if_pos (by decide), root61Child6]
def root61Candidates7 : List Nat := [4019, 14747, 21557]
set_option maxRecDepth 8192 in
theorem root61Candidates7_checked : candidates edge 4 311 2111 root61Tail8 winner.bound = root61Candidates7 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child7 : search edge 3 [311, 2111] root61Candidates7 2422 winner = winner := by decide
theorem root61Step7 : scan edge (search edge 3) 4 [311] 311 root61Tail7 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail8 winner := by
  rw [root61Tail7, scan, if_neg (by decide), root61Candidates7_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail8
    (if 3 ≤ root61Candidates7.length then search edge 3 [311, 2111] root61Candidates7 2422 winner else winner) = _
  rw [if_pos (by decide), root61Child7]
def root61Candidates8 : List Nat := [11351, 19013]
set_option maxRecDepth 8192 in
theorem root61Candidates8_checked : candidates edge 4 311 2141 root61Tail9 winner.bound = root61Candidates8 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root61Step8 : scan edge (search edge 3) 4 [311] 311 root61Tail8 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail9 winner := by
  rw [root61Tail8, scan, if_neg (by decide), root61Candidates8_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail9
    (if 3 ≤ root61Candidates8.length then search edge 3 [311, 2141] root61Candidates8 2452 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates9 : List Nat := [8807, 9923]
set_option maxRecDepth 8192 in
theorem root61Candidates9_checked : candidates edge 4 311 2381 root61Tail10 winner.bound = root61Candidates9 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
theorem root61Step9 : scan edge (search edge 3) 4 [311] 311 root61Tail9 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail10 winner := by
  rw [root61Tail9, scan, if_neg (by decide), root61Candidates9_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail10
    (if 3 ≤ root61Candidates9.length then search edge 3 [311, 2381] root61Candidates9 2692 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates10 : List Nat := [17351]
set_option maxRecDepth 8192 in
theorem root61Candidates10_checked : candidates edge 4 311 2441 root61Tail11 winner.bound = root61Candidates10 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
theorem root61Step10 : scan edge (search edge 3) 4 [311] 311 root61Tail10 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail11 winner := by
  rw [root61Tail10, scan, if_neg (by decide), root61Candidates10_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail11
    (if 3 ≤ root61Candidates10.length then search edge 3 [311, 2441] root61Candidates10 2752 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates11 : List Nat := [8069, 21803]
set_option maxRecDepth 8192 in
theorem root61Candidates11_checked : candidates edge 4 311 2609 root61Tail12 winner.bound = root61Candidates11 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
theorem root61Step11 : scan edge (search edge 3) 4 [311] 311 root61Tail11 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail12 winner := by
  rw [root61Tail11, scan, if_neg (by decide), root61Candidates11_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail12
    (if 3 ≤ root61Candidates11.length then search edge 3 [311, 2609] root61Candidates11 2920 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates12 : List Nat := [10091]
set_option maxRecDepth 8192 in
theorem root61Candidates12_checked : candidates edge 4 311 2909 root61Tail13 winner.bound = root61Candidates12 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
theorem root61Step12 : scan edge (search edge 3) 4 [311] 311 root61Tail12 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail13 winner := by
  rw [root61Tail12, scan, if_neg (by decide), root61Candidates12_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail13
    (if 3 ≤ root61Candidates12.length then search edge 3 [311, 2909] root61Candidates12 3220 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates13 : List Nat := [8867, 19013]
set_option maxRecDepth 8192 in
theorem root61Candidates13_checked : candidates edge 4 311 3863 root61Tail14 winner.bound = root61Candidates13 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root61Step13 : scan edge (search edge 3) 4 [311] 311 root61Tail13 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail14 winner := by
  rw [root61Tail13, scan, if_neg (by decide), root61Candidates13_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail14
    (if 3 ≤ root61Candidates13.length then search edge 3 [311, 3863] root61Candidates13 4174 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates14 : List Nat := [7211]
set_option maxRecDepth 8192 in
theorem root61Candidates14_checked : candidates edge 4 311 4019 root61Tail15 winner.bound = root61Candidates14 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root61Step14 : scan edge (search edge 3) 4 [311] 311 root61Tail14 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail15 winner := by
  rw [root61Tail14, scan, if_neg (by decide), root61Candidates14_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail15
    (if 3 ≤ root61Candidates14.length then search edge 3 [311, 4019] root61Candidates14 4330 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates15 : List Nat := [5237, 8069, 8237]
set_option maxRecDepth 8192 in
theorem root61Candidates15_checked : candidates edge 4 311 4871 root61Tail16 winner.bound = root61Candidates15 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root61Child15 : search edge 3 [311, 4871] root61Candidates15 5182 winner = winner := by decide
theorem root61Step15 : scan edge (search edge 3) 4 [311] 311 root61Tail15 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail16 winner := by
  rw [root61Tail15, scan, if_neg (by decide), root61Candidates15_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail16
    (if 3 ≤ root61Candidates15.length then search edge 3 [311, 4871] root61Candidates15 5182 winner else winner) = _
  rw [if_pos (by decide), root61Child15]
def root61Candidates16 : List Nat := [12539]
set_option maxRecDepth 8192 in
theorem root61Candidates16_checked : candidates edge 4 311 4967 root61Tail17 winner.bound = root61Candidates16 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root61Step16 : scan edge (search edge 3) 4 [311] 311 root61Tail16 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail17 winner := by
  rw [root61Tail16, scan, if_neg (by decide), root61Candidates16_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail17
    (if 3 ≤ root61Candidates16.length then search edge 3 [311, 4967] root61Candidates16 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root61Candidates17_checked : candidates edge 4 311 5171 root61Tail18 winner.bound = root61Candidates17 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root61Step17 : scan edge (search edge 3) 4 [311] 311 root61Tail17 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail18 winner := by
  rw [root61Tail17, scan, if_neg (by decide), root61Candidates17_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail18
    (if 3 ≤ root61Candidates17.length then search edge 3 [311, 5171] root61Candidates17 5482 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates18 : List Nat := [6761, 13217]
set_option maxRecDepth 8192 in
theorem root61Candidates18_checked : candidates edge 4 311 5237 root61Tail19 winner.bound = root61Candidates18 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root61Step18 : scan edge (search edge 3) 4 [311] 311 root61Tail18 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail19 winner := by
  rw [root61Tail18, scan, if_neg (by decide), root61Candidates18_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail19
    (if 3 ≤ root61Candidates18.length then search edge 3 [311, 5237] root61Candidates18 5548 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates19 : List Nat := [13829]
set_option maxRecDepth 8192 in
theorem root61Candidates19_checked : candidates edge 4 311 5279 root61Tail20 winner.bound = root61Candidates19 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root61Step19 : scan edge (search edge 3) 4 [311] 311 root61Tail19 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail20 winner := by
  rw [root61Tail19, scan, if_neg (by decide), root61Candidates19_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail20
    (if 3 ≤ root61Candidates19.length then search edge 3 [311, 5279] root61Candidates19 5590 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates20 : List Nat := [16871]
set_option maxRecDepth 8192 in
theorem root61Candidates20_checked : candidates edge 4 311 6011 root61Tail21 winner.bound = root61Candidates20 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root61Step20 : scan edge (search edge 3) 4 [311] 311 root61Tail20 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail21 winner := by
  rw [root61Tail20, scan, if_neg (by decide), root61Candidates20_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail21
    (if 3 ≤ root61Candidates20.length then search edge 3 [311, 6011] root61Candidates20 6322 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates21 : List Nat := [13217, 18911]
set_option maxRecDepth 8192 in
theorem root61Candidates21_checked : candidates edge 4 311 6131 root61Tail22 winner.bound = root61Candidates21 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root61Step21 : scan edge (search edge 3) 4 [311] 311 root61Tail21 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail22 winner := by
  rw [root61Tail21, scan, if_neg (by decide), root61Candidates21_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail22
    (if 3 ≤ root61Candidates21.length then search edge 3 [311, 6131] root61Candidates21 6442 winner else winner) = _
  rw [if_neg (by decide)]
def root61Candidates22 : List Nat := [11177, 15401]
set_option maxRecDepth 8192 in
theorem root61Candidates22_checked : candidates edge 4 311 6263 root61Tail23 winner.bound = root61Candidates22 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root61Step22 : scan edge (search edge 3) 4 [311] 311 root61Tail22 winner =
    scan edge (search edge 3) 4 [311] 311 root61Tail23 winner := by
  rw [root61Tail22, scan, if_neg (by decide), root61Candidates22_checked]
  change scan edge (search edge 3) 4 [311] 311 root61Tail23
    (if 3 ≤ root61Candidates22.length then search edge 3 [311, 6263] root61Candidates22 6574 winner else winner) = _
  rw [if_neg (by decide)]
theorem root61Scan23 : scan edge (search edge 3) 4 [311] 311 root61Tail23 winner = winner := by rw [root61Tail23, scan, if_pos (by decide)]
theorem root61Scan22 : scan edge (search edge 3) 4 [311] 311 root61Tail22 winner = winner :=
  (root61Step22).trans root61Scan23
theorem root61Scan21 : scan edge (search edge 3) 4 [311] 311 root61Tail21 winner = winner :=
  (root61Step21).trans root61Scan22
theorem root61Scan20 : scan edge (search edge 3) 4 [311] 311 root61Tail20 winner = winner :=
  (root61Step20).trans root61Scan21
theorem root61Scan19 : scan edge (search edge 3) 4 [311] 311 root61Tail19 winner = winner :=
  (root61Step19).trans root61Scan20
theorem root61Scan18 : scan edge (search edge 3) 4 [311] 311 root61Tail18 winner = winner :=
  (root61Step18).trans root61Scan19
theorem root61Scan17 : scan edge (search edge 3) 4 [311] 311 root61Tail17 winner = winner :=
  (root61Step17).trans root61Scan18
theorem root61Scan16 : scan edge (search edge 3) 4 [311] 311 root61Tail16 winner = winner :=
  (root61Step16).trans root61Scan17
theorem root61Scan15 : scan edge (search edge 3) 4 [311] 311 root61Tail15 winner = winner :=
  (root61Step15).trans root61Scan16
theorem root61Scan14 : scan edge (search edge 3) 4 [311] 311 root61Tail14 winner = winner :=
  (root61Step14).trans root61Scan15
theorem root61Scan13 : scan edge (search edge 3) 4 [311] 311 root61Tail13 winner = winner :=
  (root61Step13).trans root61Scan14
theorem root61Scan12 : scan edge (search edge 3) 4 [311] 311 root61Tail12 winner = winner :=
  (root61Step12).trans root61Scan13
theorem root61Scan11 : scan edge (search edge 3) 4 [311] 311 root61Tail11 winner = winner :=
  (root61Step11).trans root61Scan12
theorem root61Scan10 : scan edge (search edge 3) 4 [311] 311 root61Tail10 winner = winner :=
  (root61Step10).trans root61Scan11
theorem root61Scan9 : scan edge (search edge 3) 4 [311] 311 root61Tail9 winner = winner :=
  (root61Step9).trans root61Scan10
theorem root61Scan8 : scan edge (search edge 3) 4 [311] 311 root61Tail8 winner = winner :=
  (root61Step8).trans root61Scan9
theorem root61Scan7 : scan edge (search edge 3) 4 [311] 311 root61Tail7 winner = winner :=
  (root61Step7).trans root61Scan8
theorem root61Scan6 : scan edge (search edge 3) 4 [311] 311 root61Tail6 winner = winner :=
  (root61Step6).trans root61Scan7
theorem root61Scan5 : scan edge (search edge 3) 4 [311] 311 root61Tail5 winner = winner :=
  (root61Step5).trans root61Scan6
theorem root61Scan4 : scan edge (search edge 3) 4 [311] 311 root61Tail4 winner = winner :=
  (root61Step4).trans root61Scan5
theorem root61Scan3 : scan edge (search edge 3) 4 [311] 311 root61Tail3 winner = winner :=
  (root61Step3).trans root61Scan4
theorem root61Scan2 : scan edge (search edge 3) 4 [311] 311 root61Tail2 winner = winner :=
  (root61Step2).trans root61Scan3
theorem root61Scan1 : scan edge (search edge 3) 4 [311] 311 root61Tail1 winner = winner :=
  (root61Step1).trans root61Scan2
theorem root61Scan0 : scan edge (search edge 3) 4 [311] 311 root61Tail0 winner = winner :=
  (root61Step0).trans root61Scan1
set_option maxRecDepth 4096 in
theorem root61Branch : search edge 4 [311] root61Candidates 311 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [311] 311 root61Tail0 winner = winner
  exact root61Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
