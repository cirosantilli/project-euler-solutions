import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root164Candidates : List Nat := Data.row991
set_option maxRecDepth 8192 in
theorem root164Candidates_checked : candidates edge 5 0 991 tail165 winner.bound = root164Candidates := by
  apply candidates_eq_row edge 5 0 991 tail165 root164Candidates winner.bound tail165_ordered
  · decide
  · decide
  · funext q
    rfl
def root164Tail66 : List Nat := []
def root164Tail65 : List Nat := 24967 :: root164Tail66
def root164Tail64 : List Nat := 24379 :: root164Tail65
def root164Tail63 : List Nat := 23929 :: root164Tail64
def root164Tail62 : List Nat := 23917 :: root164Tail63
def root164Tail61 : List Nat := 23539 :: root164Tail62
def root164Tail60 : List Nat := 23509 :: root164Tail61
def root164Tail59 : List Nat := 23173 :: root164Tail60
def root164Tail58 : List Nat := 23053 :: root164Tail59
def root164Tail57 : List Nat := 22063 :: root164Tail58
def root164Tail56 : List Nat := 22039 :: root164Tail57
def root164Tail55 : List Nat := 22027 :: root164Tail56
def root164Tail54 : List Nat := 21961 :: root164Tail55
def root164Tail53 : List Nat := 21937 :: root164Tail54
def root164Tail52 : List Nat := 21871 :: root164Tail53
def root164Tail51 : List Nat := 21247 :: root164Tail52
def root164Tail50 : List Nat := 21163 :: root164Tail51
def root164Tail49 : List Nat := 20809 :: root164Tail50
def root164Tail48 : List Nat := 20599 :: root164Tail49
def root164Tail47 : List Nat := 20389 :: root164Tail48
def root164Tail46 : List Nat := 20341 :: root164Tail47
def root164Tail45 : List Nat := 19081 :: root164Tail46
def root164Tail44 : List Nat := 18787 :: root164Tail45
def root164Tail43 : List Nat := 18553 :: root164Tail44
def root164Tail42 : List Nat := 18493 :: root164Tail43
def root164Tail41 : List Nat := 18457 :: root164Tail42
def root164Tail40 : List Nat := 17257 :: root164Tail41
def root164Tail39 : List Nat := 16267 :: root164Tail40
def root164Tail38 : List Nat := 16189 :: root164Tail39
def root164Tail37 : List Nat := 15937 :: root164Tail38
def root164Tail36 : List Nat := 15739 :: root164Tail37
def root164Tail35 : List Nat := 15307 :: root164Tail36
def root164Tail34 : List Nat := 14341 :: root164Tail35
def root164Tail33 : List Nat := 13711 :: root164Tail34
def root164Tail32 : List Nat := 13099 :: root164Tail33
def root164Tail31 : List Nat := 12829 :: root164Tail32
def root164Tail30 : List Nat := 12109 :: root164Tail31
def root164Tail29 : List Nat := 11839 :: root164Tail30
def root164Tail28 : List Nat := 11161 :: root164Tail29
def root164Tail27 : List Nat := 10267 :: root164Tail28
def root164Tail26 : List Nat := 9697 :: root164Tail27
def root164Tail25 : List Nat := 9277 :: root164Tail26
def root164Tail24 : List Nat := 9199 :: root164Tail25
def root164Tail23 : List Nat := 8641 :: root164Tail24
def root164Tail22 : List Nat := 8017 :: root164Tail23
def root164Tail21 : List Nat := 7759 :: root164Tail22
def root164Tail20 : List Nat := 7549 :: root164Tail21
def root164Tail19 : List Nat := 7027 :: root164Tail20
def root164Tail18 : List Nat := 6163 :: root164Tail19
def root164Tail17 : List Nat := 6091 :: root164Tail18
def root164Tail16 : List Nat := 5881 :: root164Tail17
def root164Tail15 : List Nat := 4153 :: root164Tail16
def root164Tail14 : List Nat := 3727 :: root164Tail15
def root164Tail13 : List Nat := 3613 :: root164Tail14
def root164Tail12 : List Nat := 3571 :: root164Tail13
def root164Tail11 : List Nat := 3361 :: root164Tail12
def root164Tail10 : List Nat := 2803 :: root164Tail11
def root164Tail9 : List Nat := 2719 :: root164Tail10
def root164Tail8 : List Nat := 2689 :: root164Tail9
def root164Tail7 : List Nat := 2269 :: root164Tail8
def root164Tail6 : List Nat := 2029 :: root164Tail7
def root164Tail5 : List Nat := 2017 :: root164Tail6
def root164Tail4 : List Nat := 1861 :: root164Tail5
def root164Tail3 : List Nat := 1753 :: root164Tail4
def root164Tail2 : List Nat := 1609 :: root164Tail3
def root164Tail1 : List Nat := 1171 :: root164Tail2
def root164Tail0 : List Nat := 1009 :: root164Tail1
def root164Candidates0 : List Nat := [3613, 7027]
set_option maxRecDepth 8192 in
theorem root164Candidates0_checked : candidates edge 4 991 1009 root164Tail1 winner.bound = root164Candidates0 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
theorem root164Step0 : scan edge (search edge 3) 4 [991] 991 root164Tail0 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail1 winner := by
  rw [root164Tail0, scan, if_neg (by decide), root164Candidates0_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail1
    (if 3 ≤ root164Candidates0.length then search edge 3 [991, 1009] root164Candidates0 2000 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates1 : List Nat := [5881, 9199, 18493]
set_option maxRecDepth 8192 in
theorem root164Candidates1_checked : candidates edge 4 991 1171 root164Tail2 winner.bound = root164Candidates1 := by
  unfold candidates
  have row : edge 1171 = (SortedNatTree.contains FastRows.row1171) := by
    change Data.row1171.contains = (SortedNatTree.contains FastRows.row1171)
    exact FastRows.row1171_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root164Child1 : search edge 3 [991, 1171] root164Candidates1 2162 winner = winner := by decide
theorem root164Step1 : scan edge (search edge 3) 4 [991] 991 root164Tail1 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail2 winner := by
  rw [root164Tail1, scan, if_neg (by decide), root164Candidates1_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail2
    (if 3 ≤ root164Candidates1.length then search edge 3 [991, 1171] root164Candidates1 2162 winner else winner) = _
  rw [if_pos (by decide), root164Child1]
def root164Candidates2 : List Nat := [2017, 2803, 3613]
set_option maxRecDepth 8192 in
theorem root164Candidates2_checked : candidates edge 4 991 1609 root164Tail3 winner.bound = root164Candidates2 := by
  unfold candidates
  have row : edge 1609 = (SortedNatTree.contains FastRows.row1609) := by
    change Data.row1609.contains = (SortedNatTree.contains FastRows.row1609)
    exact FastRows.row1609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root164Child2 : search edge 3 [991, 1609] root164Candidates2 2600 winner = winner := by decide
theorem root164Step2 : scan edge (search edge 3) 4 [991] 991 root164Tail2 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail3 winner := by
  rw [root164Tail2, scan, if_neg (by decide), root164Candidates2_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail3
    (if 3 ≤ root164Candidates2.length then search edge 3 [991, 1609] root164Candidates2 2600 winner else winner) = _
  rw [if_pos (by decide), root164Child2]
def root164Candidates3 : List Nat := [7549, 11839]
set_option maxRecDepth 8192 in
theorem root164Candidates3_checked : candidates edge 4 991 1753 root164Tail4 winner.bound = root164Candidates3 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
theorem root164Step3 : scan edge (search edge 3) 4 [991] 991 root164Tail3 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail4 winner := by
  rw [root164Tail3, scan, if_neg (by decide), root164Candidates3_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail4
    (if 3 ≤ root164Candidates3.length then search edge 3 [991, 1753] root164Candidates3 2744 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates4 : List Nat := [5881]
set_option maxRecDepth 8192 in
theorem root164Candidates4_checked : candidates edge 4 991 1861 root164Tail5 winner.bound = root164Candidates4 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
theorem root164Step4 : scan edge (search edge 3) 4 [991] 991 root164Tail4 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail5 winner := by
  rw [root164Tail4, scan, if_neg (by decide), root164Candidates4_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail5
    (if 3 ≤ root164Candidates4.length then search edge 3 [991, 1861] root164Candidates4 2852 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates5 : List Nat := [2689]
set_option maxRecDepth 8192 in
theorem root164Candidates5_checked : candidates edge 4 991 2017 root164Tail6 winner.bound = root164Candidates5 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root164Step5 : scan edge (search edge 3) 4 [991] 991 root164Tail5 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail6 winner := by
  rw [root164Tail5, scan, if_neg (by decide), root164Candidates5_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail6
    (if 3 ≤ root164Candidates5.length then search edge 3 [991, 2017] root164Candidates5 3008 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates6 : List Nat := [7027, 12109]
set_option maxRecDepth 8192 in
theorem root164Candidates6_checked : candidates edge 4 991 2029 root164Tail7 winner.bound = root164Candidates6 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
theorem root164Step6 : scan edge (search edge 3) 4 [991] 991 root164Tail6 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail7 winner := by
  rw [root164Tail6, scan, if_neg (by decide), root164Candidates6_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail7
    (if 3 ≤ root164Candidates6.length then search edge 3 [991, 2029] root164Candidates6 3020 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates7 : List Nat := [3613, 7027, 9697]
set_option maxRecDepth 8192 in
theorem root164Candidates7_checked : candidates edge 4 991 2269 root164Tail8 winner.bound = root164Candidates7 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root164Child7 : search edge 3 [991, 2269] root164Candidates7 3260 winner = winner := by decide
theorem root164Step7 : scan edge (search edge 3) 4 [991] 991 root164Tail7 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail8 winner := by
  rw [root164Tail7, scan, if_neg (by decide), root164Candidates7_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail8
    (if 3 ≤ root164Candidates7.length then search edge 3 [991, 2269] root164Candidates7 3260 winner else winner) = _
  rw [if_pos (by decide), root164Child7]
def root164Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates8_checked : candidates edge 4 991 2689 root164Tail9 winner.bound = root164Candidates8 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root164Step8 : scan edge (search edge 3) 4 [991] 991 root164Tail8 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail9 winner := by
  rw [root164Tail8, scan, if_neg (by decide), root164Candidates8_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail9
    (if 3 ≤ root164Candidates8.length then search edge 3 [991, 2689] root164Candidates8 3680 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates9 : List Nat := [21961]
set_option maxRecDepth 8192 in
theorem root164Candidates9_checked : candidates edge 4 991 2719 root164Tail10 winner.bound = root164Candidates9 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root164Step9 : scan edge (search edge 3) 4 [991] 991 root164Tail9 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail10 winner := by
  rw [root164Tail9, scan, if_neg (by decide), root164Candidates9_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail10
    (if 3 ≤ root164Candidates9.length then search edge 3 [991, 2719] root164Candidates9 3710 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates10 : List Nat := [7549, 14341, 20809]
set_option maxRecDepth 8192 in
theorem root164Candidates10_checked : candidates edge 4 991 2803 root164Tail11 winner.bound = root164Candidates10 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root164Child10 : search edge 3 [991, 2803] root164Candidates10 3794 winner = winner := by decide
theorem root164Step10 : scan edge (search edge 3) 4 [991] 991 root164Tail10 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail11 winner := by
  rw [root164Tail10, scan, if_neg (by decide), root164Candidates10_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail11
    (if 3 ≤ root164Candidates10.length then search edge 3 [991, 2803] root164Candidates10 3794 winner else winner) = _
  rw [if_pos (by decide), root164Child10]
def root164Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates11_checked : candidates edge 4 991 3361 root164Tail12 winner.bound = root164Candidates11 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root164Step11 : scan edge (search edge 3) 4 [991] 991 root164Tail11 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail12 winner := by
  rw [root164Tail11, scan, if_neg (by decide), root164Candidates11_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail12
    (if 3 ≤ root164Candidates11.length then search edge 3 [991, 3361] root164Candidates11 4352 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates12_checked : candidates edge 4 991 3571 root164Tail13 winner.bound = root164Candidates12 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root164Step12 : scan edge (search edge 3) 4 [991] 991 root164Tail12 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail13 winner := by
  rw [root164Tail12, scan, if_neg (by decide), root164Candidates12_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail13
    (if 3 ≤ root164Candidates12.length then search edge 3 [991, 3571] root164Candidates12 4562 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates13_checked : candidates edge 4 991 3613 root164Tail14 winner.bound = root164Candidates13 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root164Step13 : scan edge (search edge 3) 4 [991] 991 root164Tail13 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail14 winner := by
  rw [root164Tail13, scan, if_neg (by decide), root164Candidates13_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail14
    (if 3 ≤ root164Candidates13.length then search edge 3 [991, 3613] root164Candidates13 4604 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates14 : List Nat := [16189]
set_option maxRecDepth 8192 in
theorem root164Candidates14_checked : candidates edge 4 991 3727 root164Tail15 winner.bound = root164Candidates14 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root164Step14 : scan edge (search edge 3) 4 [991] 991 root164Tail14 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail15 winner := by
  rw [root164Tail14, scan, if_neg (by decide), root164Candidates14_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail15
    (if 3 ≤ root164Candidates14.length then search edge 3 [991, 3727] root164Candidates14 4718 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates15 : List Nat := [20809]
set_option maxRecDepth 8192 in
theorem root164Candidates15_checked : candidates edge 4 991 4153 root164Tail16 winner.bound = root164Candidates15 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
theorem root164Step15 : scan edge (search edge 3) 4 [991] 991 root164Tail15 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail16 winner := by
  rw [root164Tail15, scan, if_neg (by decide), root164Candidates15_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail16
    (if 3 ≤ root164Candidates15.length then search edge 3 [991, 4153] root164Candidates15 5144 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates16_checked : candidates edge 4 991 5881 root164Tail17 winner.bound = root164Candidates16 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root164Step16 : scan edge (search edge 3) 4 [991] 991 root164Tail16 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail17 winner := by
  rw [root164Tail16, scan, if_neg (by decide), root164Candidates16_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail17
    (if 3 ≤ root164Candidates16.length then search edge 3 [991, 5881] root164Candidates16 6872 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates17_checked : candidates edge 4 991 6091 root164Tail18 winner.bound = root164Candidates17 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root164Step17 : scan edge (search edge 3) 4 [991] 991 root164Tail17 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail18 winner := by
  rw [root164Tail17, scan, if_neg (by decide), root164Candidates17_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail18
    (if 3 ≤ root164Candidates17.length then search edge 3 [991, 6091] root164Candidates17 7082 winner else winner) = _
  rw [if_neg (by decide)]
def root164Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root164Candidates18_checked : candidates edge 4 991 6163 root164Tail19 winner.bound = root164Candidates18 := by
  unfold candidates
  have row : edge 6163 = (SortedNatTree.contains FastRows.row6163) := by
    change Data.row6163.contains = (SortedNatTree.contains FastRows.row6163)
    exact FastRows.row6163_checked.symm
  rw [row]
  decide
theorem root164Step18 : scan edge (search edge 3) 4 [991] 991 root164Tail18 winner =
    scan edge (search edge 3) 4 [991] 991 root164Tail19 winner := by
  rw [root164Tail18, scan, if_neg (by decide), root164Candidates18_checked]
  change scan edge (search edge 3) 4 [991] 991 root164Tail19
    (if 3 ≤ root164Candidates18.length then search edge 3 [991, 6163] root164Candidates18 7154 winner else winner) = _
  rw [if_neg (by decide)]
theorem root164Scan19 : scan edge (search edge 3) 4 [991] 991 root164Tail19 winner = winner := by rw [root164Tail19, scan, if_pos (by decide)]
theorem root164Scan18 : scan edge (search edge 3) 4 [991] 991 root164Tail18 winner = winner :=
  (root164Step18).trans root164Scan19
theorem root164Scan17 : scan edge (search edge 3) 4 [991] 991 root164Tail17 winner = winner :=
  (root164Step17).trans root164Scan18
theorem root164Scan16 : scan edge (search edge 3) 4 [991] 991 root164Tail16 winner = winner :=
  (root164Step16).trans root164Scan17
theorem root164Scan15 : scan edge (search edge 3) 4 [991] 991 root164Tail15 winner = winner :=
  (root164Step15).trans root164Scan16
theorem root164Scan14 : scan edge (search edge 3) 4 [991] 991 root164Tail14 winner = winner :=
  (root164Step14).trans root164Scan15
theorem root164Scan13 : scan edge (search edge 3) 4 [991] 991 root164Tail13 winner = winner :=
  (root164Step13).trans root164Scan14
theorem root164Scan12 : scan edge (search edge 3) 4 [991] 991 root164Tail12 winner = winner :=
  (root164Step12).trans root164Scan13
theorem root164Scan11 : scan edge (search edge 3) 4 [991] 991 root164Tail11 winner = winner :=
  (root164Step11).trans root164Scan12
theorem root164Scan10 : scan edge (search edge 3) 4 [991] 991 root164Tail10 winner = winner :=
  (root164Step10).trans root164Scan11
theorem root164Scan9 : scan edge (search edge 3) 4 [991] 991 root164Tail9 winner = winner :=
  (root164Step9).trans root164Scan10
theorem root164Scan8 : scan edge (search edge 3) 4 [991] 991 root164Tail8 winner = winner :=
  (root164Step8).trans root164Scan9
theorem root164Scan7 : scan edge (search edge 3) 4 [991] 991 root164Tail7 winner = winner :=
  (root164Step7).trans root164Scan8
theorem root164Scan6 : scan edge (search edge 3) 4 [991] 991 root164Tail6 winner = winner :=
  (root164Step6).trans root164Scan7
theorem root164Scan5 : scan edge (search edge 3) 4 [991] 991 root164Tail5 winner = winner :=
  (root164Step5).trans root164Scan6
theorem root164Scan4 : scan edge (search edge 3) 4 [991] 991 root164Tail4 winner = winner :=
  (root164Step4).trans root164Scan5
theorem root164Scan3 : scan edge (search edge 3) 4 [991] 991 root164Tail3 winner = winner :=
  (root164Step3).trans root164Scan4
theorem root164Scan2 : scan edge (search edge 3) 4 [991] 991 root164Tail2 winner = winner :=
  (root164Step2).trans root164Scan3
theorem root164Scan1 : scan edge (search edge 3) 4 [991] 991 root164Tail1 winner = winner :=
  (root164Step1).trans root164Scan2
theorem root164Scan0 : scan edge (search edge 3) 4 [991] 991 root164Tail0 winner = winner :=
  (root164Step0).trans root164Scan1
set_option maxRecDepth 4096 in
theorem root164Branch : search edge 4 [991] root164Candidates 991 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [991] 991 root164Tail0 winner = winner
  exact root164Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
