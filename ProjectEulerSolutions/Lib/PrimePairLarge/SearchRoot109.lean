import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root109Candidates : List Nat := Data.row613
set_option maxRecDepth 8192 in
theorem root109Candidates_checked : candidates edge 5 0 613 tail110 winner.bound = root109Candidates := by
  apply candidates_eq_row edge 5 0 613 tail110 root109Candidates winner.bound tail110_ordered
  · decide
  · decide
  · funext q
    rfl
def root109Tail73 : List Nat := []
def root109Tail72 : List Nat := 25321 :: root109Tail73
def root109Tail71 : List Nat := 24379 :: root109Tail72
def root109Tail70 : List Nat := 24121 :: root109Tail71
def root109Tail69 : List Nat := 23593 :: root109Tail70
def root109Tail68 : List Nat := 23071 :: root109Tail69
def root109Tail67 : List Nat := 23011 :: root109Tail68
def root109Tail66 : List Nat := 22783 :: root109Tail67
def root109Tail65 : List Nat := 22189 :: root109Tail66
def root109Tail64 : List Nat := 22003 :: root109Tail65
def root109Tail63 : List Nat := 20443 :: root109Tail64
def root109Tail62 : List Nat := 20407 :: root109Tail63
def root109Tail61 : List Nat := 20089 :: root109Tail62
def root109Tail60 : List Nat := 19861 :: root109Tail61
def root109Tail59 : List Nat := 19759 :: root109Tail60
def root109Tail58 : List Nat := 19501 :: root109Tail59
def root109Tail57 : List Nat := 18793 :: root109Tail58
def root109Tail56 : List Nat := 18679 :: root109Tail57
def root109Tail55 : List Nat := 18121 :: root109Tail56
def root109Tail54 : List Nat := 17923 :: root109Tail55
def root109Tail53 : List Nat := 17359 :: root109Tail54
def root109Tail52 : List Nat := 17167 :: root109Tail53
def root109Tail51 : List Nat := 16747 :: root109Tail52
def root109Tail50 : List Nat := 16273 :: root109Tail51
def root109Tail49 : List Nat := 16267 :: root109Tail50
def root109Tail48 : List Nat := 15973 :: root109Tail49
def root109Tail47 : List Nat := 15643 :: root109Tail48
def root109Tail46 : List Nat := 14923 :: root109Tail47
def root109Tail45 : List Nat := 14713 :: root109Tail46
def root109Tail44 : List Nat := 14593 :: root109Tail45
def root109Tail43 : List Nat := 13789 :: root109Tail44
def root109Tail42 : List Nat := 13411 :: root109Tail43
def root109Tail41 : List Nat := 13309 :: root109Tail42
def root109Tail40 : List Nat := 13267 :: root109Tail41
def root109Tail39 : List Nat := 13099 :: root109Tail40
def root109Tail38 : List Nat := 12763 :: root109Tail39
def root109Tail37 : List Nat := 12577 :: root109Tail38
def root109Tail36 : List Nat := 12073 :: root109Tail37
def root109Tail35 : List Nat := 11941 :: root109Tail36
def root109Tail34 : List Nat := 11863 :: root109Tail35
def root109Tail33 : List Nat := 11251 :: root109Tail34
def root109Tail32 : List Nat := 9949 :: root109Tail33
def root109Tail31 : List Nat := 9901 :: root109Tail32
def root109Tail30 : List Nat := 8893 :: root109Tail31
def root109Tail29 : List Nat := 8443 :: root109Tail30
def root109Tail28 : List Nat := 8191 :: root109Tail29
def root109Tail27 : List Nat := 7177 :: root109Tail28
def root109Tail26 : List Nat := 7069 :: root109Tail27
def root109Tail25 : List Nat := 6967 :: root109Tail26
def root109Tail24 : List Nat := 6733 :: root109Tail25
def root109Tail23 : List Nat := 6679 :: root109Tail24
def root109Tail22 : List Nat := 6673 :: root109Tail23
def root109Tail21 : List Nat := 6547 :: root109Tail22
def root109Tail20 : List Nat := 6379 :: root109Tail21
def root109Tail19 : List Nat := 5743 :: root109Tail20
def root109Tail18 : List Nat := 5581 :: root109Tail19
def root109Tail17 : List Nat := 4801 :: root109Tail18
def root109Tail16 : List Nat := 4363 :: root109Tail17
def root109Tail15 : List Nat := 4339 :: root109Tail16
def root109Tail14 : List Nat := 4153 :: root109Tail15
def root109Tail13 : List Nat := 4129 :: root109Tail14
def root109Tail12 : List Nat := 4021 :: root109Tail13
def root109Tail11 : List Nat := 3943 :: root109Tail12
def root109Tail10 : List Nat := 3769 :: root109Tail11
def root109Tail9 : List Nat := 3643 :: root109Tail10
def root109Tail8 : List Nat := 3037 :: root109Tail9
def root109Tail7 : List Nat := 2851 :: root109Tail8
def root109Tail6 : List Nat := 2089 :: root109Tail7
def root109Tail5 : List Nat := 1753 :: root109Tail6
def root109Tail4 : List Nat := 1303 :: root109Tail5
def root109Tail3 : List Nat := 883 :: root109Tail4
def root109Tail2 : List Nat := 829 :: root109Tail3
def root109Tail1 : List Nat := 673 :: root109Tail2
def root109Tail0 : List Nat := 661 :: root109Tail1
def root109Candidates0 : List Nat := [883]
set_option maxRecDepth 8192 in
theorem root109Candidates0_checked : candidates edge 4 613 661 root109Tail1 winner.bound = root109Candidates0 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
theorem root109Step0 : scan edge (search edge 3) 4 [613] 613 root109Tail0 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail1 winner := by
  rw [root109Tail0, scan, if_neg (by decide), root109Candidates0_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail1
    (if 3 ≤ root109Candidates0.length then search edge 3 [613, 661] root109Candidates0 1274 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates1 : List Nat := [7177, 12577, 14593]
set_option maxRecDepth 8192 in
theorem root109Candidates1_checked : candidates edge 4 613 673 root109Tail2 winner.bound = root109Candidates1 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root109Child1 : search edge 3 [613, 673] root109Candidates1 1286 winner = winner := by decide
theorem root109Step1 : scan edge (search edge 3) 4 [613] 613 root109Tail1 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail2 winner := by
  rw [root109Tail1, scan, if_neg (by decide), root109Candidates1_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail2
    (if 3 ≤ root109Candidates1.length then search edge 3 [613, 673] root109Candidates1 1286 winner else winner) = _
  rw [if_pos (by decide), root109Child1]
def root109Candidates2 : List Nat := [15643]
set_option maxRecDepth 8192 in
theorem root109Candidates2_checked : candidates edge 4 613 829 root109Tail3 winner.bound = root109Candidates2 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
theorem root109Step2 : scan edge (search edge 3) 4 [613] 613 root109Tail2 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail3 winner := by
  rw [root109Tail2, scan, if_neg (by decide), root109Candidates2_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail3
    (if 3 ≤ root109Candidates2.length then search edge 3 [613, 829] root109Candidates2 1442 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates3 : List Nat := [14593, 15973, 17167, 18679]
set_option maxRecDepth 8192 in
theorem root109Candidates3_checked : candidates edge 4 613 883 root109Tail4 winner.bound = root109Candidates3 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root109Child3 : search edge 3 [613, 883] root109Candidates3 1496 winner = winner := by decide
theorem root109Step3 : scan edge (search edge 3) 4 [613] 613 root109Tail3 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail4 winner := by
  rw [root109Tail3, scan, if_neg (by decide), root109Candidates3_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail4
    (if 3 ≤ root109Candidates3.length then search edge 3 [613, 883] root109Candidates3 1496 winner else winner) = _
  rw [if_pos (by decide), root109Child3]
def root109Candidates4 : List Nat := [4363]
set_option maxRecDepth 8192 in
theorem root109Candidates4_checked : candidates edge 4 613 1303 root109Tail5 winner.bound = root109Candidates4 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
theorem root109Step4 : scan edge (search edge 3) 4 [613] 613 root109Tail4 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail5 winner := by
  rw [root109Tail4, scan, if_neg (by decide), root109Candidates4_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail5
    (if 3 ≤ root109Candidates4.length then search edge 3 [613, 1303] root109Candidates4 1916 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates5 : List Nat := [3643, 8893, 13267, 13309]
set_option maxRecDepth 8192 in
theorem root109Candidates5_checked : candidates edge 4 613 1753 root109Tail6 winner.bound = root109Candidates5 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root109Child5 : search edge 3 [613, 1753] root109Candidates5 2366 winner = winner := by decide
theorem root109Step5 : scan edge (search edge 3) 4 [613] 613 root109Tail5 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail6 winner := by
  rw [root109Tail5, scan, if_neg (by decide), root109Candidates5_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail6
    (if 3 ≤ root109Candidates5.length then search edge 3 [613, 1753] root109Candidates5 2366 winner else winner) = _
  rw [if_pos (by decide), root109Child5]
def root109Candidates6 : List Nat := [3643, 18121]
set_option maxRecDepth 8192 in
theorem root109Candidates6_checked : candidates edge 4 613 2089 root109Tail7 winner.bound = root109Candidates6 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
theorem root109Step6 : scan edge (search edge 3) 4 [613] 613 root109Tail6 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail7 winner := by
  rw [root109Tail6, scan, if_neg (by decide), root109Candidates6_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail7
    (if 3 ≤ root109Candidates6.length then search edge 3 [613, 2089] root109Candidates6 2702 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root109Candidates7_checked : candidates edge 4 613 2851 root109Tail8 winner.bound = root109Candidates7 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root109Step7 : scan edge (search edge 3) 4 [613] 613 root109Tail7 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail8 winner := by
  rw [root109Tail7, scan, if_neg (by decide), root109Candidates7_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail8
    (if 3 ≤ root109Candidates7.length then search edge 3 [613, 2851] root109Candidates7 3464 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates8 : List Nat := [20089]
set_option maxRecDepth 8192 in
theorem root109Candidates8_checked : candidates edge 4 613 3037 root109Tail9 winner.bound = root109Candidates8 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
theorem root109Step8 : scan edge (search edge 3) 4 [613] 613 root109Tail8 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail9 winner := by
  rw [root109Tail8, scan, if_neg (by decide), root109Candidates8_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail9
    (if 3 ≤ root109Candidates8.length then search edge 3 [613, 3037] root109Candidates8 3650 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root109Candidates9_checked : candidates edge 4 613 3643 root109Tail10 winner.bound = root109Candidates9 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root109Step9 : scan edge (search edge 3) 4 [613] 613 root109Tail9 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail10 winner := by
  rw [root109Tail9, scan, if_neg (by decide), root109Candidates9_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail10
    (if 3 ≤ root109Candidates9.length then search edge 3 [613, 3643] root109Candidates9 4256 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates10 : List Nat := [16747]
set_option maxRecDepth 8192 in
theorem root109Candidates10_checked : candidates edge 4 613 3769 root109Tail11 winner.bound = root109Candidates10 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root109Step10 : scan edge (search edge 3) 4 [613] 613 root109Tail10 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail11 winner := by
  rw [root109Tail10, scan, if_neg (by decide), root109Candidates10_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail11
    (if 3 ≤ root109Candidates10.length then search edge 3 [613, 3769] root109Candidates10 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates11 : List Nat := [5581, 14593]
set_option maxRecDepth 8192 in
theorem root109Candidates11_checked : candidates edge 4 613 3943 root109Tail12 winner.bound = root109Candidates11 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root109Step11 : scan edge (search edge 3) 4 [613] 613 root109Tail11 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail12 winner := by
  rw [root109Tail11, scan, if_neg (by decide), root109Candidates11_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail12
    (if 3 ≤ root109Candidates11.length then search edge 3 [613, 3943] root109Candidates11 4556 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates12 : List Nat := [6733, 17923]
set_option maxRecDepth 8192 in
theorem root109Candidates12_checked : candidates edge 4 613 4021 root109Tail13 winner.bound = root109Candidates12 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root109Step12 : scan edge (search edge 3) 4 [613] 613 root109Tail12 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail13 winner := by
  rw [root109Tail12, scan, if_neg (by decide), root109Candidates12_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail13
    (if 3 ≤ root109Candidates12.length then search edge 3 [613, 4021] root109Candidates12 4634 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root109Candidates13_checked : candidates edge 4 613 4129 root109Tail14 winner.bound = root109Candidates13 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root109Step13 : scan edge (search edge 3) 4 [613] 613 root109Tail13 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail14 winner := by
  rw [root109Tail13, scan, if_neg (by decide), root109Candidates13_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail14
    (if 3 ≤ root109Candidates13.length then search edge 3 [613, 4129] root109Candidates13 4742 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root109Candidates14_checked : candidates edge 4 613 4153 root109Tail15 winner.bound = root109Candidates14 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
theorem root109Step14 : scan edge (search edge 3) 4 [613] 613 root109Tail14 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail15 winner := by
  rw [root109Tail14, scan, if_neg (by decide), root109Candidates14_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail15
    (if 3 ≤ root109Candidates14.length then search edge 3 [613, 4153] root109Candidates14 4766 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root109Candidates15_checked : candidates edge 4 613 4339 root109Tail16 winner.bound = root109Candidates15 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root109Step15 : scan edge (search edge 3) 4 [613] 613 root109Tail15 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail16 winner := by
  rw [root109Tail15, scan, if_neg (by decide), root109Candidates15_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail16
    (if 3 ≤ root109Candidates15.length then search edge 3 [613, 4339] root109Candidates15 4952 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates16 : List Nat := [16267]
set_option maxRecDepth 8192 in
theorem root109Candidates16_checked : candidates edge 4 613 4363 root109Tail17 winner.bound = root109Candidates16 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root109Step16 : scan edge (search edge 3) 4 [613] 613 root109Tail16 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail17 winner := by
  rw [root109Tail16, scan, if_neg (by decide), root109Candidates16_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail17
    (if 3 ≤ root109Candidates16.length then search edge 3 [613, 4363] root109Candidates16 4976 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates17 : List Nat := [6733, 19501]
set_option maxRecDepth 8192 in
theorem root109Candidates17_checked : candidates edge 4 613 4801 root109Tail18 winner.bound = root109Candidates17 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root109Step17 : scan edge (search edge 3) 4 [613] 613 root109Tail17 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail18 winner := by
  rw [root109Tail17, scan, if_neg (by decide), root109Candidates17_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail18
    (if 3 ≤ root109Candidates17.length then search edge 3 [613, 4801] root109Candidates17 5414 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates18 : List Nat := [9949, 18793]
set_option maxRecDepth 8192 in
theorem root109Candidates18_checked : candidates edge 4 613 5581 root109Tail19 winner.bound = root109Candidates18 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root109Step18 : scan edge (search edge 3) 4 [613] 613 root109Tail18 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail19 winner := by
  rw [root109Tail18, scan, if_neg (by decide), root109Candidates18_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail19
    (if 3 ≤ root109Candidates18.length then search edge 3 [613, 5581] root109Candidates18 6194 winner else winner) = _
  rw [if_neg (by decide)]
def root109Candidates19 : List Nat := [16747]
set_option maxRecDepth 8192 in
theorem root109Candidates19_checked : candidates edge 4 613 5743 root109Tail20 winner.bound = root109Candidates19 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root109Step19 : scan edge (search edge 3) 4 [613] 613 root109Tail19 winner =
    scan edge (search edge 3) 4 [613] 613 root109Tail20 winner := by
  rw [root109Tail19, scan, if_neg (by decide), root109Candidates19_checked]
  change scan edge (search edge 3) 4 [613] 613 root109Tail20
    (if 3 ≤ root109Candidates19.length then search edge 3 [613, 5743] root109Candidates19 6356 winner else winner) = _
  rw [if_neg (by decide)]
theorem root109Scan20 : scan edge (search edge 3) 4 [613] 613 root109Tail20 winner = winner := by rw [root109Tail20, scan, if_pos (by decide)]
theorem root109Scan19 : scan edge (search edge 3) 4 [613] 613 root109Tail19 winner = winner :=
  (root109Step19).trans root109Scan20
theorem root109Scan18 : scan edge (search edge 3) 4 [613] 613 root109Tail18 winner = winner :=
  (root109Step18).trans root109Scan19
theorem root109Scan17 : scan edge (search edge 3) 4 [613] 613 root109Tail17 winner = winner :=
  (root109Step17).trans root109Scan18
theorem root109Scan16 : scan edge (search edge 3) 4 [613] 613 root109Tail16 winner = winner :=
  (root109Step16).trans root109Scan17
theorem root109Scan15 : scan edge (search edge 3) 4 [613] 613 root109Tail15 winner = winner :=
  (root109Step15).trans root109Scan16
theorem root109Scan14 : scan edge (search edge 3) 4 [613] 613 root109Tail14 winner = winner :=
  (root109Step14).trans root109Scan15
theorem root109Scan13 : scan edge (search edge 3) 4 [613] 613 root109Tail13 winner = winner :=
  (root109Step13).trans root109Scan14
theorem root109Scan12 : scan edge (search edge 3) 4 [613] 613 root109Tail12 winner = winner :=
  (root109Step12).trans root109Scan13
theorem root109Scan11 : scan edge (search edge 3) 4 [613] 613 root109Tail11 winner = winner :=
  (root109Step11).trans root109Scan12
theorem root109Scan10 : scan edge (search edge 3) 4 [613] 613 root109Tail10 winner = winner :=
  (root109Step10).trans root109Scan11
theorem root109Scan9 : scan edge (search edge 3) 4 [613] 613 root109Tail9 winner = winner :=
  (root109Step9).trans root109Scan10
theorem root109Scan8 : scan edge (search edge 3) 4 [613] 613 root109Tail8 winner = winner :=
  (root109Step8).trans root109Scan9
theorem root109Scan7 : scan edge (search edge 3) 4 [613] 613 root109Tail7 winner = winner :=
  (root109Step7).trans root109Scan8
theorem root109Scan6 : scan edge (search edge 3) 4 [613] 613 root109Tail6 winner = winner :=
  (root109Step6).trans root109Scan7
theorem root109Scan5 : scan edge (search edge 3) 4 [613] 613 root109Tail5 winner = winner :=
  (root109Step5).trans root109Scan6
theorem root109Scan4 : scan edge (search edge 3) 4 [613] 613 root109Tail4 winner = winner :=
  (root109Step4).trans root109Scan5
theorem root109Scan3 : scan edge (search edge 3) 4 [613] 613 root109Tail3 winner = winner :=
  (root109Step3).trans root109Scan4
theorem root109Scan2 : scan edge (search edge 3) 4 [613] 613 root109Tail2 winner = winner :=
  (root109Step2).trans root109Scan3
theorem root109Scan1 : scan edge (search edge 3) 4 [613] 613 root109Tail1 winner = winner :=
  (root109Step1).trans root109Scan2
theorem root109Scan0 : scan edge (search edge 3) 4 [613] 613 root109Tail0 winner = winner :=
  (root109Step0).trans root109Scan1
set_option maxRecDepth 4096 in
theorem root109Branch : search edge 4 [613] root109Candidates 613 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [613] 613 root109Tail0 winner = winner
  exact root109Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
