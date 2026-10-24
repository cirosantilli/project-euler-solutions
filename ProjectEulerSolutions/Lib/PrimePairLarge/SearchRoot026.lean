import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root26Candidates : List Nat := Data.row109
set_option maxRecDepth 8192 in
theorem root26Candidates_checked : candidates edge 5 0 109 tail27 winner.bound = root26Candidates := by
  apply candidates_eq_row edge 5 0 109 tail27 root26Candidates winner.bound tail27_ordered
  · decide
  · decide
  · funext q
    rfl
def root26Tail82 : List Nat := []
def root26Tail81 : List Nat := 25657 :: root26Tail82
def root26Tail80 : List Nat := 25153 :: root26Tail81
def root26Tail79 : List Nat := 24877 :: root26Tail80
def root26Tail78 : List Nat := 24733 :: root26Tail79
def root26Tail77 : List Nat := 24691 :: root26Tail78
def root26Tail76 : List Nat := 23929 :: root26Tail77
def root26Tail75 : List Nat := 23761 :: root26Tail76
def root26Tail74 : List Nat := 23203 :: root26Tail75
def root26Tail73 : List Nat := 22699 :: root26Tail74
def root26Tail72 : List Nat := 22543 :: root26Tail73
def root26Tail71 : List Nat := 21961 :: root26Tail72
def root26Tail70 : List Nat := 21727 :: root26Tail71
def root26Tail69 : List Nat := 20887 :: root26Tail70
def root26Tail68 : List Nat := 20641 :: root26Tail69
def root26Tail67 : List Nat := 20089 :: root26Tail68
def root26Tail66 : List Nat := 20071 :: root26Tail67
def root26Tail65 : List Nat := 19759 :: root26Tail66
def root26Tail64 : List Nat := 19081 :: root26Tail65
def root26Tail63 : List Nat := 18637 :: root26Tail64
def root26Tail62 : List Nat := 18433 :: root26Tail63
def root26Tail61 : List Nat := 18289 :: root26Tail62
def root26Tail60 : List Nat := 16981 :: root26Tail61
def root26Tail59 : List Nat := 16189 :: root26Tail60
def root26Tail58 : List Nat := 16183 :: root26Tail59
def root26Tail57 : List Nat := 16063 :: root26Tail58
def root26Tail56 : List Nat := 15991 :: root26Tail57
def root26Tail55 : List Nat := 15493 :: root26Tail56
def root26Tail54 : List Nat := 15361 :: root26Tail55
def root26Tail53 : List Nat := 14923 :: root26Tail54
def root26Tail52 : List Nat := 14737 :: root26Tail53
def root26Tail51 : List Nat := 14401 :: root26Tail52
def root26Tail50 : List Nat := 14143 :: root26Tail51
def root26Tail49 : List Nat := 14083 :: root26Tail50
def root26Tail48 : List Nat := 13807 :: root26Tail49
def root26Tail47 : List Nat := 13597 :: root26Tail48
def root26Tail46 : List Nat := 13537 :: root26Tail47
def root26Tail45 : List Nat := 13411 :: root26Tail46
def root26Tail44 : List Nat := 12049 :: root26Tail45
def root26Tail43 : List Nat := 11779 :: root26Tail44
def root26Tail42 : List Nat := 11689 :: root26Tail43
def root26Tail41 : List Nat := 11119 :: root26Tail42
def root26Tail40 : List Nat := 9649 :: root26Tail41
def root26Tail39 : List Nat := 9619 :: root26Tail40
def root26Tail38 : List Nat := 9463 :: root26Tail39
def root26Tail37 : List Nat := 9421 :: root26Tail38
def root26Tail36 : List Nat := 9337 :: root26Tail37
def root26Tail35 : List Nat := 9199 :: root26Tail36
def root26Tail34 : List Nat := 8731 :: root26Tail35
def root26Tail33 : List Nat := 8443 :: root26Tail34
def root26Tail32 : List Nat := 8269 :: root26Tail33
def root26Tail31 : List Nat := 7933 :: root26Tail32
def root26Tail30 : List Nat := 7351 :: root26Tail31
def root26Tail29 : List Nat := 7321 :: root26Tail30
def root26Tail28 : List Nat := 7297 :: root26Tail29
def root26Tail27 : List Nat := 6673 :: root26Tail28
def root26Tail26 : List Nat := 5791 :: root26Tail27
def root26Tail25 : List Nat := 5503 :: root26Tail26
def root26Tail24 : List Nat := 4957 :: root26Tail25
def root26Tail23 : List Nat := 4801 :: root26Tail24
def root26Tail22 : List Nat := 4759 :: root26Tail23
def root26Tail21 : List Nat := 4567 :: root26Tail22
def root26Tail20 : List Nat := 4441 :: root26Tail21
def root26Tail19 : List Nat := 4129 :: root26Tail20
def root26Tail18 : List Nat := 3943 :: root26Tail19
def root26Tail17 : List Nat := 3529 :: root26Tail18
def root26Tail16 : List Nat := 3331 :: root26Tail17
def root26Tail15 : List Nat := 2803 :: root26Tail16
def root26Tail14 : List Nat := 2389 :: root26Tail15
def root26Tail13 : List Nat := 1777 :: root26Tail14
def root26Tail12 : List Nat := 1609 :: root26Tail13
def root26Tail11 : List Nat := 1459 :: root26Tail12
def root26Tail10 : List Nat := 1399 :: root26Tail11
def root26Tail9 : List Nat := 1381 :: root26Tail10
def root26Tail8 : List Nat := 1063 :: root26Tail9
def root26Tail7 : List Nat := 919 :: root26Tail8
def root26Tail6 : List Nat := 883 :: root26Tail7
def root26Tail5 : List Nat := 859 :: root26Tail6
def root26Tail4 : List Nat := 673 :: root26Tail5
def root26Tail3 : List Nat := 661 :: root26Tail4
def root26Tail2 : List Nat := 313 :: root26Tail3
def root26Tail1 : List Nat := 199 :: root26Tail2
def root26Tail0 : List Nat := 139 :: root26Tail1
def root26Candidates0 : List Nat := [661, 2803, 7933, 8269, 8731]
set_option maxRecDepth 8192 in
theorem root26Candidates0_checked : candidates edge 4 109 139 root26Tail1 winner.bound = root26Candidates0 := by
  unfold candidates
  have row : edge 139 = (SortedNatTree.contains FastRows.row139) := by
    change Data.row139.contains = (SortedNatTree.contains FastRows.row139)
    exact FastRows.row139_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child0 : search edge 3 [109, 139] root26Candidates0 248 winner = winner := by decide
theorem root26Step0 : scan edge (search edge 3) 4 [109] 109 root26Tail0 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail1 winner := by
  rw [root26Tail0, scan, if_neg (by decide), root26Candidates0_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail1
    (if 3 ≤ root26Candidates0.length then search edge 3 [109, 139] root26Candidates0 248 winner else winner) = _
  rw [if_pos (by decide), root26Child0]
def root26Candidates1 : List Nat := [673, 1609, 3529, 8443, 14737]
set_option maxRecDepth 8192 in
theorem root26Candidates1_checked : candidates edge 4 109 199 root26Tail2 winner.bound = root26Candidates1 := by
  unfold candidates
  have row : edge 199 = (SortedNatTree.contains FastRows.row199) := by
    change Data.row199.contains = (SortedNatTree.contains FastRows.row199)
    exact FastRows.row199_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child1 : search edge 3 [109, 199] root26Candidates1 308 winner = winner := by decide
theorem root26Step1 : scan edge (search edge 3) 4 [109] 109 root26Tail1 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail2 winner := by
  rw [root26Tail1, scan, if_neg (by decide), root26Candidates1_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail2
    (if 3 ≤ root26Candidates1.length then search edge 3 [109, 199] root26Candidates1 308 winner else winner) = _
  rw [if_pos (by decide), root26Child1]
def root26Candidates2 : List Nat := [12049, 16981, 24877]
set_option maxRecDepth 8192 in
theorem root26Candidates2_checked : candidates edge 4 109 313 root26Tail3 winner.bound = root26Candidates2 := by
  unfold candidates
  have row : edge 313 = (SortedNatTree.contains FastRows.row313) := by
    change Data.row313.contains = (SortedNatTree.contains FastRows.row313)
    exact FastRows.row313_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child2 : search edge 3 [109, 313] root26Candidates2 422 winner = winner := by decide
theorem root26Step2 : scan edge (search edge 3) 4 [109] 109 root26Tail2 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail3 winner := by
  rw [root26Tail2, scan, if_neg (by decide), root26Candidates2_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail3
    (if 3 ≤ root26Candidates2.length then search edge 3 [109, 313] root26Candidates2 422 winner else winner) = _
  rw [if_pos (by decide), root26Child2]
def root26Candidates3 : List Nat := [883, 1777, 3331, 4759, 5503, 20641]
set_option maxRecDepth 8192 in
theorem root26Candidates3_checked : candidates edge 4 109 661 root26Tail4 winner.bound = root26Candidates3 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child3 : search edge 3 [109, 661] root26Candidates3 770 winner = winner := by decide
theorem root26Step3 : scan edge (search edge 3) 4 [109] 109 root26Tail3 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail4 winner := by
  rw [root26Tail3, scan, if_neg (by decide), root26Candidates3_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail4
    (if 3 ≤ root26Candidates3.length then search edge 3 [109, 661] root26Candidates3 770 winner else winner) = _
  rw [if_pos (by decide), root26Child3]
def root26Candidates4 : List Nat := [11689, 13537, 20887]
set_option maxRecDepth 8192 in
theorem root26Candidates4_checked : candidates edge 4 109 673 root26Tail5 winner.bound = root26Candidates4 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child4 : search edge 3 [109, 673] root26Candidates4 782 winner = winner := by decide
theorem root26Step4 : scan edge (search edge 3) 4 [109] 109 root26Tail4 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail5 winner := by
  rw [root26Tail4, scan, if_neg (by decide), root26Candidates4_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail5
    (if 3 ≤ root26Candidates4.length then search edge 3 [109, 673] root26Candidates4 782 winner else winner) = _
  rw [if_pos (by decide), root26Child4]
def root26Candidates5 : List Nat := [919, 1381, 3943, 11689, 18433, 24691]
set_option maxRecDepth 8192 in
theorem root26Candidates5_checked : candidates edge 4 109 859 root26Tail6 winner.bound = root26Candidates5 := by
  unfold candidates
  have row : edge 859 = (SortedNatTree.contains FastRows.row859) := by
    change Data.row859.contains = (SortedNatTree.contains FastRows.row859)
    exact FastRows.row859_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child5 : search edge 3 [109, 859] root26Candidates5 968 winner = winner := by decide
theorem root26Step5 : scan edge (search edge 3) 4 [109] 109 root26Tail5 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail6 winner := by
  rw [root26Tail5, scan, if_neg (by decide), root26Candidates5_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail6
    (if 3 ≤ root26Candidates5.length then search edge 3 [109, 859] root26Candidates5 968 winner else winner) = _
  rw [if_pos (by decide), root26Child5]
def root26Candidates6 : List Nat := [9421, 11779, 13807, 22543]
set_option maxRecDepth 8192 in
theorem root26Candidates6_checked : candidates edge 4 109 883 root26Tail7 winner.bound = root26Candidates6 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child6 : search edge 3 [109, 883] root26Candidates6 992 winner = winner := by decide
theorem root26Step6 : scan edge (search edge 3) 4 [109] 109 root26Tail6 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail7 winner := by
  rw [root26Tail6, scan, if_neg (by decide), root26Candidates6_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail7
    (if 3 ≤ root26Candidates6.length then search edge 3 [109, 883] root26Candidates6 992 winner else winner) = _
  rw [if_pos (by decide), root26Child6]
def root26Candidates7 : List Nat := [4441, 4567, 9649, 13597]
set_option maxRecDepth 8192 in
theorem root26Candidates7_checked : candidates edge 4 109 919 root26Tail8 winner.bound = root26Candidates7 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child7 : search edge 3 [109, 919] root26Candidates7 1028 winner = winner := by decide
theorem root26Step7 : scan edge (search edge 3) 4 [109] 109 root26Tail7 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail8 winner := by
  rw [root26Tail7, scan, if_neg (by decide), root26Candidates7_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail8
    (if 3 ≤ root26Candidates7.length then search edge 3 [109, 919] root26Candidates7 1028 winner else winner) = _
  rw [if_pos (by decide), root26Child7]
def root26Candidates8 : List Nat := [1609, 9463]
set_option maxRecDepth 8192 in
theorem root26Candidates8_checked : candidates edge 4 109 1063 root26Tail9 winner.bound = root26Candidates8 := by
  unfold candidates
  have row : edge 1063 = (SortedNatTree.contains FastRows.row1063) := by
    change Data.row1063.contains = (SortedNatTree.contains FastRows.row1063)
    exact FastRows.row1063_checked.symm
  rw [row]
  decide
theorem root26Step8 : scan edge (search edge 3) 4 [109] 109 root26Tail8 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail9 winner := by
  rw [root26Tail8, scan, if_neg (by decide), root26Candidates8_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail9
    (if 3 ≤ root26Candidates8.length then search edge 3 [109, 1063] root26Candidates8 1172 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates9 : List Nat := [7933, 9199, 23203]
set_option maxRecDepth 8192 in
theorem root26Candidates9_checked : candidates edge 4 109 1381 root26Tail10 winner.bound = root26Candidates9 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child9 : search edge 3 [109, 1381] root26Candidates9 1490 winner = winner := by decide
theorem root26Step9 : scan edge (search edge 3) 4 [109] 109 root26Tail9 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail10 winner := by
  rw [root26Tail9, scan, if_neg (by decide), root26Candidates9_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail10
    (if 3 ≤ root26Candidates9.length then search edge 3 [109, 1381] root26Candidates9 1490 winner else winner) = _
  rw [if_pos (by decide), root26Child9]
def root26Candidates10 : List Nat := [4801, 9463, 14923]
set_option maxRecDepth 8192 in
theorem root26Candidates10_checked : candidates edge 4 109 1399 root26Tail11 winner.bound = root26Candidates10 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child10 : search edge 3 [109, 1399] root26Candidates10 1508 winner = winner := by decide
theorem root26Step10 : scan edge (search edge 3) 4 [109] 109 root26Tail10 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail11 winner := by
  rw [root26Tail10, scan, if_neg (by decide), root26Candidates10_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail11
    (if 3 ≤ root26Candidates10.length then search edge 3 [109, 1399] root26Candidates10 1508 winner else winner) = _
  rw [if_pos (by decide), root26Child10]
def root26Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root26Candidates11_checked : candidates edge 4 109 1459 root26Tail12 winner.bound = root26Candidates11 := by
  unfold candidates
  have row : edge 1459 = (SortedNatTree.contains FastRows.row1459) := by
    change Data.row1459.contains = (SortedNatTree.contains FastRows.row1459)
    exact FastRows.row1459_checked.symm
  rw [row]
  decide
theorem root26Step11 : scan edge (search edge 3) 4 [109] 109 root26Tail11 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail12 winner := by
  rw [root26Tail11, scan, if_neg (by decide), root26Candidates11_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail12
    (if 3 ≤ root26Candidates11.length then search edge 3 [109, 1459] root26Candidates11 1568 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates12 : List Nat := [2803, 4129, 20887, 23761]
set_option maxRecDepth 8192 in
theorem root26Candidates12_checked : candidates edge 4 109 1609 root26Tail13 winner.bound = root26Candidates12 := by
  unfold candidates
  have row : edge 1609 = (SortedNatTree.contains FastRows.row1609) := by
    change Data.row1609.contains = (SortedNatTree.contains FastRows.row1609)
    exact FastRows.row1609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child12 : search edge 3 [109, 1609] root26Candidates12 1718 winner = winner := by decide
theorem root26Step12 : scan edge (search edge 3) 4 [109] 109 root26Tail12 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail13 winner := by
  rw [root26Tail12, scan, if_neg (by decide), root26Candidates12_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail13
    (if 3 ≤ root26Candidates12.length then search edge 3 [109, 1609] root26Candidates12 1718 winner else winner) = _
  rw [if_pos (by decide), root26Child12]
def root26Candidates13 : List Nat := [15991, 16183]
set_option maxRecDepth 8192 in
theorem root26Candidates13_checked : candidates edge 4 109 1777 root26Tail14 winner.bound = root26Candidates13 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
theorem root26Step13 : scan edge (search edge 3) 4 [109] 109 root26Tail13 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail14 winner := by
  rw [root26Tail13, scan, if_neg (by decide), root26Candidates13_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail14
    (if 3 ≤ root26Candidates13.length then search edge 3 [109, 1777] root26Candidates13 1886 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates14 : List Nat := [4957, 7351, 9619, 16063]
set_option maxRecDepth 8192 in
theorem root26Candidates14_checked : candidates edge 4 109 2389 root26Tail15 winner.bound = root26Candidates14 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child14 : search edge 3 [109, 2389] root26Candidates14 2498 winner = winner := by decide
theorem root26Step14 : scan edge (search edge 3) 4 [109] 109 root26Tail14 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail15 winner := by
  rw [root26Tail14, scan, if_neg (by decide), root26Candidates14_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail15
    (if 3 ≤ root26Candidates14.length then search edge 3 [109, 2389] root26Candidates14 2498 winner else winner) = _
  rw [if_pos (by decide), root26Child14]
def root26Candidates15 : List Nat := [14143]
set_option maxRecDepth 8192 in
theorem root26Candidates15_checked : candidates edge 4 109 2803 root26Tail16 winner.bound = root26Candidates15 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root26Step15 : scan edge (search edge 3) 4 [109] 109 root26Tail15 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail16 winner := by
  rw [root26Tail15, scan, if_neg (by decide), root26Candidates15_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail16
    (if 3 ≤ root26Candidates15.length then search edge 3 [109, 2803] root26Candidates15 2912 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates16 : List Nat := [9421]
set_option maxRecDepth 8192 in
theorem root26Candidates16_checked : candidates edge 4 109 3331 root26Tail17 winner.bound = root26Candidates16 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root26Step16 : scan edge (search edge 3) 4 [109] 109 root26Tail16 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail17 winner := by
  rw [root26Tail16, scan, if_neg (by decide), root26Candidates16_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail17
    (if 3 ≤ root26Candidates16.length then search edge 3 [109, 3331] root26Candidates16 3440 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates17 : List Nat := [3943, 9619, 14143, 16981]
set_option maxRecDepth 8192 in
theorem root26Candidates17_checked : candidates edge 4 109 3529 root26Tail18 winner.bound = root26Candidates17 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child17 : search edge 3 [109, 3529] root26Candidates17 3638 winner = winner := by decide
theorem root26Step17 : scan edge (search edge 3) 4 [109] 109 root26Tail17 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail18 winner := by
  rw [root26Tail17, scan, if_neg (by decide), root26Candidates17_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail18
    (if 3 ≤ root26Candidates17.length then search edge 3 [109, 3529] root26Candidates17 3638 winner else winner) = _
  rw [if_pos (by decide), root26Child17]
def root26Candidates18 : List Nat := [14083, 16183, 21727]
set_option maxRecDepth 8192 in
theorem root26Candidates18_checked : candidates edge 4 109 3943 root26Tail19 winner.bound = root26Candidates18 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child18 : search edge 3 [109, 3943] root26Candidates18 4052 winner = winner := by decide
theorem root26Step18 : scan edge (search edge 3) 4 [109] 109 root26Tail18 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail19 winner := by
  rw [root26Tail18, scan, if_neg (by decide), root26Candidates18_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail19
    (if 3 ≤ root26Candidates18.length then search edge 3 [109, 3943] root26Candidates18 4052 winner else winner) = _
  rw [if_pos (by decide), root26Child18]
def root26Candidates19 : List Nat := [15991]
set_option maxRecDepth 8192 in
theorem root26Candidates19_checked : candidates edge 4 109 4129 root26Tail20 winner.bound = root26Candidates19 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root26Step19 : scan edge (search edge 3) 4 [109] 109 root26Tail19 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail20 winner := by
  rw [root26Tail19, scan, if_neg (by decide), root26Candidates19_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail20
    (if 3 ≤ root26Candidates19.length then search edge 3 [109, 4129] root26Candidates19 4238 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root26Candidates20_checked : candidates edge 4 109 4441 root26Tail21 winner.bound = root26Candidates20 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root26Step20 : scan edge (search edge 3) 4 [109] 109 root26Tail20 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail21 winner := by
  rw [root26Tail20, scan, if_neg (by decide), root26Candidates20_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail21
    (if 3 ≤ root26Candidates20.length then search edge 3 [109, 4441] root26Candidates20 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates21 : List Nat := [14737, 16189, 18433]
set_option maxRecDepth 8192 in
theorem root26Candidates21_checked : candidates edge 4 109 4567 root26Tail22 winner.bound = root26Candidates21 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root26Child21 : search edge 3 [109, 4567] root26Candidates21 4676 winner = winner := by decide
theorem root26Step21 : scan edge (search edge 3) 4 [109] 109 root26Tail21 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail22 winner := by
  rw [root26Tail21, scan, if_neg (by decide), root26Candidates21_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail22
    (if 3 ≤ root26Candidates21.length then search edge 3 [109, 4567] root26Candidates21 4676 winner else winner) = _
  rw [if_pos (by decide), root26Child21]
def root26Candidates22 : List Nat := [13597]
set_option maxRecDepth 8192 in
theorem root26Candidates22_checked : candidates edge 4 109 4759 root26Tail23 winner.bound = root26Candidates22 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root26Step22 : scan edge (search edge 3) 4 [109] 109 root26Tail22 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail23 winner := by
  rw [root26Tail22, scan, if_neg (by decide), root26Candidates22_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail23
    (if 3 ≤ root26Candidates22.length then search edge 3 [109, 4759] root26Candidates22 4868 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root26Candidates23_checked : candidates edge 4 109 4801 root26Tail24 winner.bound = root26Candidates23 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root26Step23 : scan edge (search edge 3) 4 [109] 109 root26Tail23 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail24 winner := by
  rw [root26Tail23, scan, if_neg (by decide), root26Candidates23_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail24
    (if 3 ≤ root26Candidates23.length then search edge 3 [109, 4801] root26Candidates23 4910 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates24 : List Nat := [15361]
set_option maxRecDepth 8192 in
theorem root26Candidates24_checked : candidates edge 4 109 4957 root26Tail25 winner.bound = root26Candidates24 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root26Step24 : scan edge (search edge 3) 4 [109] 109 root26Tail24 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail25 winner := by
  rw [root26Tail24, scan, if_neg (by decide), root26Candidates24_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail25
    (if 3 ≤ root26Candidates24.length then search edge 3 [109, 4957] root26Candidates24 5066 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root26Candidates25_checked : candidates edge 4 109 5503 root26Tail26 winner.bound = root26Candidates25 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root26Step25 : scan edge (search edge 3) 4 [109] 109 root26Tail25 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail26 winner := by
  rw [root26Tail25, scan, if_neg (by decide), root26Candidates25_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail26
    (if 3 ≤ root26Candidates25.length then search edge 3 [109, 5503] root26Candidates25 5612 winner else winner) = _
  rw [if_neg (by decide)]
def root26Candidates26 : List Nat := [16189, 20089]
set_option maxRecDepth 8192 in
theorem root26Candidates26_checked : candidates edge 4 109 5791 root26Tail27 winner.bound = root26Candidates26 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root26Step26 : scan edge (search edge 3) 4 [109] 109 root26Tail26 winner =
    scan edge (search edge 3) 4 [109] 109 root26Tail27 winner := by
  rw [root26Tail26, scan, if_neg (by decide), root26Candidates26_checked]
  change scan edge (search edge 3) 4 [109] 109 root26Tail27
    (if 3 ≤ root26Candidates26.length then search edge 3 [109, 5791] root26Candidates26 5900 winner else winner) = _
  rw [if_neg (by decide)]
theorem root26Scan27 : scan edge (search edge 3) 4 [109] 109 root26Tail27 winner = winner := by rw [root26Tail27, scan, if_pos (by decide)]
theorem root26Scan26 : scan edge (search edge 3) 4 [109] 109 root26Tail26 winner = winner :=
  (root26Step26).trans root26Scan27
theorem root26Scan25 : scan edge (search edge 3) 4 [109] 109 root26Tail25 winner = winner :=
  (root26Step25).trans root26Scan26
theorem root26Scan24 : scan edge (search edge 3) 4 [109] 109 root26Tail24 winner = winner :=
  (root26Step24).trans root26Scan25
theorem root26Scan23 : scan edge (search edge 3) 4 [109] 109 root26Tail23 winner = winner :=
  (root26Step23).trans root26Scan24
theorem root26Scan22 : scan edge (search edge 3) 4 [109] 109 root26Tail22 winner = winner :=
  (root26Step22).trans root26Scan23
theorem root26Scan21 : scan edge (search edge 3) 4 [109] 109 root26Tail21 winner = winner :=
  (root26Step21).trans root26Scan22
theorem root26Scan20 : scan edge (search edge 3) 4 [109] 109 root26Tail20 winner = winner :=
  (root26Step20).trans root26Scan21
theorem root26Scan19 : scan edge (search edge 3) 4 [109] 109 root26Tail19 winner = winner :=
  (root26Step19).trans root26Scan20
theorem root26Scan18 : scan edge (search edge 3) 4 [109] 109 root26Tail18 winner = winner :=
  (root26Step18).trans root26Scan19
theorem root26Scan17 : scan edge (search edge 3) 4 [109] 109 root26Tail17 winner = winner :=
  (root26Step17).trans root26Scan18
theorem root26Scan16 : scan edge (search edge 3) 4 [109] 109 root26Tail16 winner = winner :=
  (root26Step16).trans root26Scan17
theorem root26Scan15 : scan edge (search edge 3) 4 [109] 109 root26Tail15 winner = winner :=
  (root26Step15).trans root26Scan16
theorem root26Scan14 : scan edge (search edge 3) 4 [109] 109 root26Tail14 winner = winner :=
  (root26Step14).trans root26Scan15
theorem root26Scan13 : scan edge (search edge 3) 4 [109] 109 root26Tail13 winner = winner :=
  (root26Step13).trans root26Scan14
theorem root26Scan12 : scan edge (search edge 3) 4 [109] 109 root26Tail12 winner = winner :=
  (root26Step12).trans root26Scan13
theorem root26Scan11 : scan edge (search edge 3) 4 [109] 109 root26Tail11 winner = winner :=
  (root26Step11).trans root26Scan12
theorem root26Scan10 : scan edge (search edge 3) 4 [109] 109 root26Tail10 winner = winner :=
  (root26Step10).trans root26Scan11
theorem root26Scan9 : scan edge (search edge 3) 4 [109] 109 root26Tail9 winner = winner :=
  (root26Step9).trans root26Scan10
theorem root26Scan8 : scan edge (search edge 3) 4 [109] 109 root26Tail8 winner = winner :=
  (root26Step8).trans root26Scan9
theorem root26Scan7 : scan edge (search edge 3) 4 [109] 109 root26Tail7 winner = winner :=
  (root26Step7).trans root26Scan8
theorem root26Scan6 : scan edge (search edge 3) 4 [109] 109 root26Tail6 winner = winner :=
  (root26Step6).trans root26Scan7
theorem root26Scan5 : scan edge (search edge 3) 4 [109] 109 root26Tail5 winner = winner :=
  (root26Step5).trans root26Scan6
theorem root26Scan4 : scan edge (search edge 3) 4 [109] 109 root26Tail4 winner = winner :=
  (root26Step4).trans root26Scan5
theorem root26Scan3 : scan edge (search edge 3) 4 [109] 109 root26Tail3 winner = winner :=
  (root26Step3).trans root26Scan4
theorem root26Scan2 : scan edge (search edge 3) 4 [109] 109 root26Tail2 winner = winner :=
  (root26Step2).trans root26Scan3
theorem root26Scan1 : scan edge (search edge 3) 4 [109] 109 root26Tail1 winner = winner :=
  (root26Step1).trans root26Scan2
theorem root26Scan0 : scan edge (search edge 3) 4 [109] 109 root26Tail0 winner = winner :=
  (root26Step0).trans root26Scan1
set_option maxRecDepth 4096 in
theorem root26Branch : search edge 4 [109] root26Candidates 109 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [109] 109 root26Tail0 winner = winner
  exact root26Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
