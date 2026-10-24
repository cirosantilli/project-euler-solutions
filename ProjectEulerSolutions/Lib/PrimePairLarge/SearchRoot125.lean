import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root125Candidates : List Nat := Data.row719
set_option maxRecDepth 8192 in
theorem root125Candidates_checked : candidates edge 5 0 719 tail126 winner.bound = root125Candidates := by
  apply candidates_eq_row edge 5 0 719 tail126 root125Candidates winner.bound tail126_ordered
  · decide
  · decide
  · funext q
    rfl
def root125Tail61 : List Nat := []
def root125Tail60 : List Nat := 24851 :: root125Tail61
def root125Tail59 : List Nat := 24137 :: root125Tail60
def root125Tail58 : List Nat := 23057 :: root125Tail59
def root125Tail57 : List Nat := 22973 :: root125Tail58
def root125Tail56 : List Nat := 22133 :: root125Tail57
def root125Tail55 : List Nat := 20939 :: root125Tail56
def root125Tail54 : List Nat := 20759 :: root125Tail55
def root125Tail53 : List Nat := 20117 :: root125Tail54
def root125Tail52 : List Nat := 19763 :: root125Tail53
def root125Tail51 : List Nat := 19709 :: root125Tail52
def root125Tail50 : List Nat := 19697 :: root125Tail51
def root125Tail49 : List Nat := 19553 :: root125Tail50
def root125Tail48 : List Nat := 19037 :: root125Tail49
def root125Tail47 : List Nat := 18311 :: root125Tail48
def root125Tail46 : List Nat := 18257 :: root125Tail47
def root125Tail45 : List Nat := 17669 :: root125Tail46
def root125Tail44 : List Nat := 17609 :: root125Tail45
def root125Tail43 : List Nat := 16451 :: root125Tail44
def root125Tail42 : List Nat := 16223 :: root125Tail43
def root125Tail41 : List Nat := 15761 :: root125Tail42
def root125Tail40 : List Nat := 15497 :: root125Tail41
def root125Tail39 : List Nat := 14657 :: root125Tail40
def root125Tail38 : List Nat := 14207 :: root125Tail39
def root125Tail37 : List Nat := 14009 :: root125Tail38
def root125Tail36 : List Nat := 12569 :: root125Tail37
def root125Tail35 : List Nat := 12413 :: root125Tail36
def root125Tail34 : List Nat := 12119 :: root125Tail35
def root125Tail33 : List Nat := 11321 :: root125Tail34
def root125Tail32 : List Nat := 11243 :: root125Tail33
def root125Tail31 : List Nat := 11159 :: root125Tail32
def root125Tail30 : List Nat := 10331 :: root125Tail31
def root125Tail29 : List Nat := 9173 :: root125Tail30
def root125Tail28 : List Nat := 9029 :: root125Tail29
def root125Tail27 : List Nat := 8963 :: root125Tail28
def root125Tail26 : List Nat := 8297 :: root125Tail27
def root125Tail25 : List Nat := 7937 :: root125Tail26
def root125Tail24 : List Nat := 7793 :: root125Tail25
def root125Tail23 : List Nat := 7433 :: root125Tail24
def root125Tail22 : List Nat := 7283 :: root125Tail23
def root125Tail21 : List Nat := 7079 :: root125Tail22
def root125Tail20 : List Nat := 6869 :: root125Tail21
def root125Tail19 : List Nat := 6323 :: root125Tail20
def root125Tail18 : List Nat := 6317 :: root125Tail19
def root125Tail17 : List Nat := 5807 :: root125Tail18
def root125Tail16 : List Nat := 5387 :: root125Tail17
def root125Tail15 : List Nat := 5261 :: root125Tail16
def root125Tail14 : List Nat := 5039 :: root125Tail15
def root125Tail13 : List Nat := 4973 :: root125Tail14
def root125Tail12 : List Nat := 4799 :: root125Tail13
def root125Tail11 : List Nat := 3833 :: root125Tail12
def root125Tail10 : List Nat := 3257 :: root125Tail11
def root125Tail9 : List Nat := 2579 :: root125Tail10
def root125Tail8 : List Nat := 2543 :: root125Tail9
def root125Tail7 : List Nat := 2333 :: root125Tail8
def root125Tail6 : List Nat := 2027 :: root125Tail7
def root125Tail5 : List Nat := 1997 :: root125Tail6
def root125Tail4 : List Nat := 1439 :: root125Tail5
def root125Tail3 : List Nat := 1319 :: root125Tail4
def root125Tail2 : List Nat := 947 :: root125Tail3
def root125Tail1 : List Nat := 911 :: root125Tail2
def root125Tail0 : List Nat := 773 :: root125Tail1
def root125Candidates0 : List Nat := [1439, 12119]
set_option maxRecDepth 8192 in
theorem root125Candidates0_checked : candidates edge 4 719 773 root125Tail1 winner.bound = root125Candidates0 := by
  unfold candidates
  have row : edge 773 = (SortedNatTree.contains FastRows.row773) := by
    change Data.row773.contains = (SortedNatTree.contains FastRows.row773)
    exact FastRows.row773_checked.symm
  rw [row]
  decide
theorem root125Step0 : scan edge (search edge 3) 4 [719] 719 root125Tail0 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail1 winner := by
  rw [root125Tail0, scan, if_neg (by decide), root125Candidates0_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail1
    (if 3 ≤ root125Candidates0.length then search edge 3 [719, 773] root125Candidates0 1492 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates1 : List Nat := [947]
set_option maxRecDepth 8192 in
theorem root125Candidates1_checked : candidates edge 4 719 911 root125Tail2 winner.bound = root125Candidates1 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root125Step1 : scan edge (search edge 3) 4 [719] 719 root125Tail1 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail2 winner := by
  rw [root125Tail1, scan, if_neg (by decide), root125Candidates1_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail2
    (if 3 ≤ root125Candidates1.length then search edge 3 [719, 911] root125Candidates1 1630 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates2 : List Nat := [1439, 20117, 22973]
set_option maxRecDepth 8192 in
theorem root125Candidates2_checked : candidates edge 4 719 947 root125Tail3 winner.bound = root125Candidates2 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root125Child2 : search edge 3 [719, 947] root125Candidates2 1666 winner = winner := by decide
theorem root125Step2 : scan edge (search edge 3) 4 [719] 719 root125Tail2 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail3 winner := by
  rw [root125Tail2, scan, if_neg (by decide), root125Candidates2_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail3
    (if 3 ≤ root125Candidates2.length then search edge 3 [719, 947] root125Candidates2 1666 winner else winner) = _
  rw [if_pos (by decide), root125Child2]
def root125Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root125Candidates3_checked : candidates edge 4 719 1319 root125Tail4 winner.bound = root125Candidates3 := by
  unfold candidates
  have row : edge 1319 = (SortedNatTree.contains FastRows.row1319) := by
    change Data.row1319.contains = (SortedNatTree.contains FastRows.row1319)
    exact FastRows.row1319_checked.symm
  rw [row]
  decide
theorem root125Step3 : scan edge (search edge 3) 4 [719] 719 root125Tail3 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail4 winner := by
  rw [root125Tail3, scan, if_neg (by decide), root125Candidates3_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail4
    (if 3 ≤ root125Candidates3.length then search edge 3 [719, 1319] root125Candidates3 2038 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates4 : List Nat := [2579, 3833, 5261, 6869]
set_option maxRecDepth 8192 in
theorem root125Candidates4_checked : candidates edge 4 719 1439 root125Tail5 winner.bound = root125Candidates4 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root125Child4 : search edge 3 [719, 1439] root125Candidates4 2158 winner = winner := by decide
theorem root125Step4 : scan edge (search edge 3) 4 [719] 719 root125Tail4 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail5 winner := by
  rw [root125Tail4, scan, if_neg (by decide), root125Candidates4_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail5
    (if 3 ≤ root125Candidates4.length then search edge 3 [719, 1439] root125Candidates4 2158 winner else winner) = _
  rw [if_pos (by decide), root125Child4]
def root125Candidates5 : List Nat := [23057]
set_option maxRecDepth 8192 in
theorem root125Candidates5_checked : candidates edge 4 719 1997 root125Tail6 winner.bound = root125Candidates5 := by
  unfold candidates
  have row : edge 1997 = (SortedNatTree.contains FastRows.row1997) := by
    change Data.row1997.contains = (SortedNatTree.contains FastRows.row1997)
    exact FastRows.row1997_checked.symm
  rw [row]
  decide
theorem root125Step5 : scan edge (search edge 3) 4 [719] 719 root125Tail5 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail6 winner := by
  rw [root125Tail5, scan, if_neg (by decide), root125Candidates5_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail6
    (if 3 ≤ root125Candidates5.length then search edge 3 [719, 1997] root125Candidates5 2716 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates6 : List Nat := [19763]
set_option maxRecDepth 8192 in
theorem root125Candidates6_checked : candidates edge 4 719 2027 root125Tail7 winner.bound = root125Candidates6 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root125Step6 : scan edge (search edge 3) 4 [719] 719 root125Tail6 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail7 winner := by
  rw [root125Tail6, scan, if_neg (by decide), root125Candidates6_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail7
    (if 3 ≤ root125Candidates6.length then search edge 3 [719, 2027] root125Candidates6 2746 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates7 : List Nat := [5261, 12119, 16451]
set_option maxRecDepth 8192 in
theorem root125Candidates7_checked : candidates edge 4 719 2333 root125Tail8 winner.bound = root125Candidates7 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root125Child7 : search edge 3 [719, 2333] root125Candidates7 3052 winner = winner := by decide
theorem root125Step7 : scan edge (search edge 3) 4 [719] 719 root125Tail7 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail8 winner := by
  rw [root125Tail7, scan, if_neg (by decide), root125Candidates7_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail8
    (if 3 ≤ root125Candidates7.length then search edge 3 [719, 2333] root125Candidates7 3052 winner else winner) = _
  rw [if_pos (by decide), root125Child7]
def root125Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root125Candidates8_checked : candidates edge 4 719 2543 root125Tail9 winner.bound = root125Candidates8 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
theorem root125Step8 : scan edge (search edge 3) 4 [719] 719 root125Tail8 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail9 winner := by
  rw [root125Tail8, scan, if_neg (by decide), root125Candidates8_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail9
    (if 3 ≤ root125Candidates8.length then search edge 3 [719, 2543] root125Candidates8 3262 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates9 : List Nat := [20759]
set_option maxRecDepth 8192 in
theorem root125Candidates9_checked : candidates edge 4 719 2579 root125Tail10 winner.bound = root125Candidates9 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root125Step9 : scan edge (search edge 3) 4 [719] 719 root125Tail9 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail10 winner := by
  rw [root125Tail9, scan, if_neg (by decide), root125Candidates9_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail10
    (if 3 ≤ root125Candidates9.length then search edge 3 [719, 2579] root125Candidates9 3298 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates10 : List Nat := [18257, 20939]
set_option maxRecDepth 8192 in
theorem root125Candidates10_checked : candidates edge 4 719 3257 root125Tail11 winner.bound = root125Candidates10 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root125Step10 : scan edge (search edge 3) 4 [719] 719 root125Tail10 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail11 winner := by
  rw [root125Tail10, scan, if_neg (by decide), root125Candidates10_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail11
    (if 3 ≤ root125Candidates10.length then search edge 3 [719, 3257] root125Candidates10 3976 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root125Candidates11_checked : candidates edge 4 719 3833 root125Tail12 winner.bound = root125Candidates11 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root125Step11 : scan edge (search edge 3) 4 [719] 719 root125Tail11 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail12 winner := by
  rw [root125Tail11, scan, if_neg (by decide), root125Candidates11_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail12
    (if 3 ≤ root125Candidates11.length then search edge 3 [719, 3833] root125Candidates11 4552 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates12 : List Nat := [11321, 12569]
set_option maxRecDepth 8192 in
theorem root125Candidates12_checked : candidates edge 4 719 4799 root125Tail13 winner.bound = root125Candidates12 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root125Step12 : scan edge (search edge 3) 4 [719] 719 root125Tail12 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail13 winner := by
  rw [root125Tail12, scan, if_neg (by decide), root125Candidates12_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail13
    (if 3 ≤ root125Candidates12.length then search edge 3 [719, 4799] root125Candidates12 5518 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root125Candidates13_checked : candidates edge 4 719 4973 root125Tail14 winner.bound = root125Candidates13 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root125Step13 : scan edge (search edge 3) 4 [719] 719 root125Tail13 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail14 winner := by
  rw [root125Tail13, scan, if_neg (by decide), root125Candidates13_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail14
    (if 3 ≤ root125Candidates13.length then search edge 3 [719, 4973] root125Candidates13 5692 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates14 : List Nat := [6317, 15761, 19037]
set_option maxRecDepth 8192 in
theorem root125Candidates14_checked : candidates edge 4 719 5039 root125Tail15 winner.bound = root125Candidates14 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root125Child14 : search edge 3 [719, 5039] root125Candidates14 5758 winner = winner := by decide
theorem root125Step14 : scan edge (search edge 3) 4 [719] 719 root125Tail14 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail15 winner := by
  rw [root125Tail14, scan, if_neg (by decide), root125Candidates14_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail15
    (if 3 ≤ root125Candidates14.length then search edge 3 [719, 5039] root125Candidates14 5758 winner else winner) = _
  rw [if_pos (by decide), root125Child14]
def root125Candidates15 : List Nat := [14207, 19037]
set_option maxRecDepth 8192 in
theorem root125Candidates15_checked : candidates edge 4 719 5261 root125Tail16 winner.bound = root125Candidates15 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root125Step15 : scan edge (search edge 3) 4 [719] 719 root125Tail15 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail16 winner := by
  rw [root125Tail15, scan, if_neg (by decide), root125Candidates15_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail16
    (if 3 ≤ root125Candidates15.length then search edge 3 [719, 5261] root125Candidates15 5980 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates16 : List Nat := [5807, 9173, 12569, 16223]
set_option maxRecDepth 8192 in
theorem root125Candidates16_checked : candidates edge 4 719 5387 root125Tail17 winner.bound = root125Candidates16 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root125Child16 : search edge 3 [719, 5387] root125Candidates16 6106 winner = winner := by decide
theorem root125Step16 : scan edge (search edge 3) 4 [719] 719 root125Tail16 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail17 winner := by
  rw [root125Tail16, scan, if_neg (by decide), root125Candidates16_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail17
    (if 3 ≤ root125Candidates16.length then search edge 3 [719, 5387] root125Candidates16 6106 winner else winner) = _
  rw [if_pos (by decide), root125Child16]
def root125Candidates17 : List Nat := [8297, 11321]
set_option maxRecDepth 8192 in
theorem root125Candidates17_checked : candidates edge 4 719 5807 root125Tail18 winner.bound = root125Candidates17 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root125Step17 : scan edge (search edge 3) 4 [719] 719 root125Tail17 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail18 winner := by
  rw [root125Tail17, scan, if_neg (by decide), root125Candidates17_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail18
    (if 3 ≤ root125Candidates17.length then search edge 3 [719, 5807] root125Candidates17 6526 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates18 : List Nat := [14009, 16223]
set_option maxRecDepth 8192 in
theorem root125Candidates18_checked : candidates edge 4 719 6317 root125Tail19 winner.bound = root125Candidates18 := by
  unfold candidates
  have row : edge 6317 = (SortedNatTree.contains FastRows.row6317) := by
    change Data.row6317.contains = (SortedNatTree.contains FastRows.row6317)
    exact FastRows.row6317_checked.symm
  rw [row]
  decide
theorem root125Step18 : scan edge (search edge 3) 4 [719] 719 root125Tail18 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail19 winner := by
  rw [root125Tail18, scan, if_neg (by decide), root125Candidates18_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail19
    (if 3 ≤ root125Candidates18.length then search edge 3 [719, 6317] root125Candidates18 7036 winner else winner) = _
  rw [if_neg (by decide)]
def root125Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root125Candidates19_checked : candidates edge 4 719 6323 root125Tail20 winner.bound = root125Candidates19 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root125Step19 : scan edge (search edge 3) 4 [719] 719 root125Tail19 winner =
    scan edge (search edge 3) 4 [719] 719 root125Tail20 winner := by
  rw [root125Tail19, scan, if_neg (by decide), root125Candidates19_checked]
  change scan edge (search edge 3) 4 [719] 719 root125Tail20
    (if 3 ≤ root125Candidates19.length then search edge 3 [719, 6323] root125Candidates19 7042 winner else winner) = _
  rw [if_neg (by decide)]
theorem root125Scan20 : scan edge (search edge 3) 4 [719] 719 root125Tail20 winner = winner := by rw [root125Tail20, scan, if_pos (by decide)]
theorem root125Scan19 : scan edge (search edge 3) 4 [719] 719 root125Tail19 winner = winner :=
  (root125Step19).trans root125Scan20
theorem root125Scan18 : scan edge (search edge 3) 4 [719] 719 root125Tail18 winner = winner :=
  (root125Step18).trans root125Scan19
theorem root125Scan17 : scan edge (search edge 3) 4 [719] 719 root125Tail17 winner = winner :=
  (root125Step17).trans root125Scan18
theorem root125Scan16 : scan edge (search edge 3) 4 [719] 719 root125Tail16 winner = winner :=
  (root125Step16).trans root125Scan17
theorem root125Scan15 : scan edge (search edge 3) 4 [719] 719 root125Tail15 winner = winner :=
  (root125Step15).trans root125Scan16
theorem root125Scan14 : scan edge (search edge 3) 4 [719] 719 root125Tail14 winner = winner :=
  (root125Step14).trans root125Scan15
theorem root125Scan13 : scan edge (search edge 3) 4 [719] 719 root125Tail13 winner = winner :=
  (root125Step13).trans root125Scan14
theorem root125Scan12 : scan edge (search edge 3) 4 [719] 719 root125Tail12 winner = winner :=
  (root125Step12).trans root125Scan13
theorem root125Scan11 : scan edge (search edge 3) 4 [719] 719 root125Tail11 winner = winner :=
  (root125Step11).trans root125Scan12
theorem root125Scan10 : scan edge (search edge 3) 4 [719] 719 root125Tail10 winner = winner :=
  (root125Step10).trans root125Scan11
theorem root125Scan9 : scan edge (search edge 3) 4 [719] 719 root125Tail9 winner = winner :=
  (root125Step9).trans root125Scan10
theorem root125Scan8 : scan edge (search edge 3) 4 [719] 719 root125Tail8 winner = winner :=
  (root125Step8).trans root125Scan9
theorem root125Scan7 : scan edge (search edge 3) 4 [719] 719 root125Tail7 winner = winner :=
  (root125Step7).trans root125Scan8
theorem root125Scan6 : scan edge (search edge 3) 4 [719] 719 root125Tail6 winner = winner :=
  (root125Step6).trans root125Scan7
theorem root125Scan5 : scan edge (search edge 3) 4 [719] 719 root125Tail5 winner = winner :=
  (root125Step5).trans root125Scan6
theorem root125Scan4 : scan edge (search edge 3) 4 [719] 719 root125Tail4 winner = winner :=
  (root125Step4).trans root125Scan5
theorem root125Scan3 : scan edge (search edge 3) 4 [719] 719 root125Tail3 winner = winner :=
  (root125Step3).trans root125Scan4
theorem root125Scan2 : scan edge (search edge 3) 4 [719] 719 root125Tail2 winner = winner :=
  (root125Step2).trans root125Scan3
theorem root125Scan1 : scan edge (search edge 3) 4 [719] 719 root125Tail1 winner = winner :=
  (root125Step1).trans root125Scan2
theorem root125Scan0 : scan edge (search edge 3) 4 [719] 719 root125Tail0 winner = winner :=
  (root125Step0).trans root125Scan1
set_option maxRecDepth 4096 in
theorem root125Branch : search edge 4 [719] root125Candidates 719 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [719] 719 root125Tail0 winner = winner
  exact root125Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
