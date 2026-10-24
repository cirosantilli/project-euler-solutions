import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root39Candidates : List Nat := Data.row181
set_option maxRecDepth 8192 in
theorem root39Candidates_checked : candidates edge 5 0 181 tail40 winner.bound = root39Candidates := by
  apply candidates_eq_row edge 5 0 181 tail40 root39Candidates winner.bound tail40_ordered
  · decide
  · decide
  · funext q
    rfl
def root39Tail75 : List Nat := []
def root39Tail74 : List Nat := 24763 :: root39Tail75
def root39Tail73 : List Nat := 24697 :: root39Tail74
def root39Tail72 : List Nat := 24109 :: root39Tail73
def root39Tail71 : List Nat := 24061 :: root39Tail72
def root39Tail70 : List Nat := 23143 :: root39Tail71
def root39Tail69 : List Nat := 22921 :: root39Tail70
def root39Tail68 : List Nat := 22543 :: root39Tail69
def root39Tail67 : List Nat := 21211 :: root39Tail68
def root39Tail66 : List Nat := 20899 :: root39Tail67
def root39Tail65 : List Nat := 19819 :: root39Tail66
def root39Tail64 : List Nat := 19489 :: root39Tail65
def root39Tail63 : List Nat := 19237 :: root39Tail64
def root39Tail62 : List Nat := 18553 :: root39Tail63
def root39Tail61 : List Nat := 18223 :: root39Tail62
def root39Tail60 : List Nat := 18211 :: root39Tail61
def root39Tail59 : List Nat := 18013 :: root39Tail60
def root39Tail58 : List Nat := 17977 :: root39Tail59
def root39Tail57 : List Nat := 17959 :: root39Tail58
def root39Tail56 : List Nat := 17923 :: root39Tail57
def root39Tail55 : List Nat := 17509 :: root39Tail56
def root39Tail54 : List Nat := 15991 :: root39Tail55
def root39Tail53 : List Nat := 15919 :: root39Tail54
def root39Tail52 : List Nat := 15739 :: root39Tail53
def root39Tail51 : List Nat := 15667 :: root39Tail52
def root39Tail50 : List Nat := 15583 :: root39Tail51
def root39Tail49 : List Nat := 15187 :: root39Tail50
def root39Tail48 : List Nat := 15121 :: root39Tail49
def root39Tail47 : List Nat := 14851 :: root39Tail48
def root39Tail46 : List Nat := 13807 :: root39Tail47
def root39Tail45 : List Nat := 12889 :: root39Tail46
def root39Tail44 : List Nat := 12763 :: root39Tail45
def root39Tail43 : List Nat := 12007 :: root39Tail44
def root39Tail42 : List Nat := 11923 :: root39Tail43
def root39Tail41 : List Nat := 11551 :: root39Tail42
def root39Tail40 : List Nat := 11329 :: root39Tail41
def root39Tail39 : List Nat := 11131 :: root39Tail40
def root39Tail38 : List Nat := 10957 :: root39Tail39
def root39Tail37 : List Nat := 10531 :: root39Tail38
def root39Tail36 : List Nat := 10333 :: root39Tail37
def root39Tail35 : List Nat := 9871 :: root39Tail36
def root39Tail34 : List Nat := 9679 :: root39Tail35
def root39Tail33 : List Nat := 9397 :: root39Tail34
def root39Tail32 : List Nat := 9283 :: root39Tail33
def root39Tail31 : List Nat := 9109 :: root39Tail32
def root39Tail30 : List Nat := 9007 :: root39Tail31
def root39Tail29 : List Nat := 8689 :: root39Tail30
def root39Tail28 : List Nat := 8647 :: root39Tail29
def root39Tail27 : List Nat := 8419 :: root39Tail28
def root39Tail26 : List Nat := 7687 :: root39Tail27
def root39Tail25 : List Nat := 7549 :: root39Tail26
def root39Tail24 : List Nat := 6337 :: root39Tail25
def root39Tail23 : List Nat := 6007 :: root39Tail24
def root39Tail22 : List Nat := 5839 :: root39Tail23
def root39Tail21 : List Nat := 5323 :: root39Tail22
def root39Tail20 : List Nat := 5101 :: root39Tail21
def root39Tail19 : List Nat := 4051 :: root39Tail20
def root39Tail18 : List Nat := 3673 :: root39Tail19
def root39Tail17 : List Nat := 3391 :: root39Tail18
def root39Tail16 : List Nat := 2917 :: root39Tail17
def root39Tail15 : List Nat := 2683 :: root39Tail16
def root39Tail14 : List Nat := 2311 :: root39Tail15
def root39Tail13 : List Nat := 2269 :: root39Tail14
def root39Tail12 : List Nat := 1759 :: root39Tail13
def root39Tail11 : List Nat := 1657 :: root39Tail12
def root39Tail10 : List Nat := 787 :: root39Tail11
def root39Tail9 : List Nat := 757 :: root39Tail10
def root39Tail8 : List Nat := 751 :: root39Tail9
def root39Tail7 : List Nat := 619 :: root39Tail8
def root39Tail6 : List Nat := 607 :: root39Tail7
def root39Tail5 : List Nat := 499 :: root39Tail6
def root39Tail4 : List Nat := 421 :: root39Tail5
def root39Tail3 : List Nat := 397 :: root39Tail4
def root39Tail2 : List Nat := 283 :: root39Tail3
def root39Tail1 : List Nat := 199 :: root39Tail2
def root39Tail0 : List Nat := 193 :: root39Tail1
def root39Candidates0 : List Nat := [283, 751, 9283, 15991, 19819]
set_option maxRecDepth 8192 in
theorem root39Candidates0_checked : candidates edge 4 181 193 root39Tail1 winner.bound = root39Candidates0 := by
  unfold candidates
  have row : edge 193 = (SortedNatTree.contains FastRows.row193) := by
    change Data.row193.contains = (SortedNatTree.contains FastRows.row193)
    exact FastRows.row193_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child0 : search edge 3 [181, 193] root39Candidates0 374 winner = winner := by decide
theorem root39Step0 : scan edge (search edge 3) 4 [181] 181 root39Tail0 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail1 winner := by
  rw [root39Tail0, scan, if_neg (by decide), root39Candidates0_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail1
    (if 3 ≤ root39Candidates0.length then search edge 3 [181, 193] root39Candidates0 374 winner else winner) = _
  rw [if_pos (by decide), root39Child0]
def root39Candidates1 : List Nat := [751, 2917, 18013]
set_option maxRecDepth 8192 in
theorem root39Candidates1_checked : candidates edge 4 181 199 root39Tail2 winner.bound = root39Candidates1 := by
  unfold candidates
  have row : edge 199 = (SortedNatTree.contains FastRows.row199) := by
    change Data.row199.contains = (SortedNatTree.contains FastRows.row199)
    exact FastRows.row199_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child1 : search edge 3 [181, 199] root39Candidates1 380 winner = winner := by decide
theorem root39Step1 : scan edge (search edge 3) 4 [181] 181 root39Tail1 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail2 winner := by
  rw [root39Tail1, scan, if_neg (by decide), root39Candidates1_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail2
    (if 3 ≤ root39Candidates1.length then search edge 3 [181, 199] root39Candidates1 380 winner else winner) = _
  rw [if_pos (by decide), root39Child1]
def root39Candidates2 : List Nat := [397, 1657, 18211, 20899]
set_option maxRecDepth 8192 in
theorem root39Candidates2_checked : candidates edge 4 181 283 root39Tail3 winner.bound = root39Candidates2 := by
  unfold candidates
  have row : edge 283 = (SortedNatTree.contains FastRows.row283) := by
    change Data.row283.contains = (SortedNatTree.contains FastRows.row283)
    exact FastRows.row283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child2 : search edge 3 [181, 283] root39Candidates2 464 winner = winner := by decide
theorem root39Step2 : scan edge (search edge 3) 4 [181] 181 root39Tail2 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail3 winner := by
  rw [root39Tail2, scan, if_neg (by decide), root39Candidates2_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail3
    (if 3 ≤ root39Candidates2.length then search edge 3 [181, 283] root39Candidates2 464 winner else winner) = _
  rw [if_pos (by decide), root39Child2]
def root39Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root39Candidates3_checked : candidates edge 4 181 397 root39Tail4 winner.bound = root39Candidates3 := by
  unfold candidates
  have row : edge 397 = (SortedNatTree.contains FastRows.row397) := by
    change Data.row397.contains = (SortedNatTree.contains FastRows.row397)
    exact FastRows.row397_checked.symm
  rw [row]
  decide
theorem root39Step3 : scan edge (search edge 3) 4 [181] 181 root39Tail3 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail4 winner := by
  rw [root39Tail3, scan, if_neg (by decide), root39Candidates3_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail4
    (if 3 ≤ root39Candidates3.length then search edge 3 [181, 397] root39Candidates3 578 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates4 : List Nat := [607, 1657]
set_option maxRecDepth 8192 in
theorem root39Candidates4_checked : candidates edge 4 181 421 root39Tail5 winner.bound = root39Candidates4 := by
  unfold candidates
  have row : edge 421 = (SortedNatTree.contains FastRows.row421) := by
    change Data.row421.contains = (SortedNatTree.contains FastRows.row421)
    exact FastRows.row421_checked.symm
  rw [row]
  decide
theorem root39Step4 : scan edge (search edge 3) 4 [181] 181 root39Tail4 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail5 winner := by
  rw [root39Tail4, scan, if_neg (by decide), root39Candidates4_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail5
    (if 3 ≤ root39Candidates4.length then search edge 3 [181, 421] root39Candidates4 602 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates5 : List Nat := [4051, 7687]
set_option maxRecDepth 8192 in
theorem root39Candidates5_checked : candidates edge 4 181 499 root39Tail6 winner.bound = root39Candidates5 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
theorem root39Step5 : scan edge (search edge 3) 4 [181] 181 root39Tail5 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail6 winner := by
  rw [root39Tail5, scan, if_neg (by decide), root39Candidates5_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail6
    (if 3 ≤ root39Candidates5.length then search edge 3 [181, 499] root39Candidates5 680 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates6 : List Nat := [619, 6337, 10333, 11923, 12889, 15121, 19819]
set_option maxRecDepth 8192 in
theorem root39Candidates6_checked : candidates edge 4 181 607 root39Tail7 winner.bound = root39Candidates6 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child6 : search edge 3 [181, 607] root39Candidates6 788 winner = winner := by decide
theorem root39Step6 : scan edge (search edge 3) 4 [181] 181 root39Tail6 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail7 winner := by
  rw [root39Tail6, scan, if_neg (by decide), root39Candidates6_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail7
    (if 3 ≤ root39Candidates6.length then search edge 3 [181, 607] root39Candidates6 788 winner else winner) = _
  rw [if_pos (by decide), root39Child6]
def root39Candidates7 : List Nat := [7687, 9007]
set_option maxRecDepth 8192 in
theorem root39Candidates7_checked : candidates edge 4 181 619 root39Tail8 winner.bound = root39Candidates7 := by
  unfold candidates
  have row : edge 619 = (SortedNatTree.contains FastRows.row619) := by
    change Data.row619.contains = (SortedNatTree.contains FastRows.row619)
    exact FastRows.row619_checked.symm
  rw [row]
  decide
theorem root39Step7 : scan edge (search edge 3) 4 [181] 181 root39Tail7 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail8 winner := by
  rw [root39Tail7, scan, if_neg (by decide), root39Candidates7_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail8
    (if 3 ≤ root39Candidates7.length then search edge 3 [181, 619] root39Candidates7 800 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates8 : List Nat := [787, 10531, 15919, 18553]
set_option maxRecDepth 8192 in
theorem root39Candidates8_checked : candidates edge 4 181 751 root39Tail9 winner.bound = root39Candidates8 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child8 : search edge 3 [181, 751] root39Candidates8 932 winner = winner := by decide
theorem root39Step8 : scan edge (search edge 3) 4 [181] 181 root39Tail8 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail9 winner := by
  rw [root39Tail8, scan, if_neg (by decide), root39Candidates8_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail9
    (if 3 ≤ root39Candidates8.length then search edge 3 [181, 751] root39Candidates8 932 winner else winner) = _
  rw [if_pos (by decide), root39Child8]
def root39Candidates9 : List Nat := [9109, 22921]
set_option maxRecDepth 8192 in
theorem root39Candidates9_checked : candidates edge 4 181 757 root39Tail10 winner.bound = root39Candidates9 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
theorem root39Step9 : scan edge (search edge 3) 4 [181] 181 root39Tail9 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail10 winner := by
  rw [root39Tail9, scan, if_neg (by decide), root39Candidates9_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail10
    (if 3 ≤ root39Candidates9.length then search edge 3 [181, 757] root39Candidates9 938 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates10 : List Nat := [6007, 15583, 19237]
set_option maxRecDepth 8192 in
theorem root39Candidates10_checked : candidates edge 4 181 787 root39Tail11 winner.bound = root39Candidates10 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child10 : search edge 3 [181, 787] root39Candidates10 968 winner = winner := by decide
theorem root39Step10 : scan edge (search edge 3) 4 [181] 181 root39Tail10 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail11 winner := by
  rw [root39Tail10, scan, if_neg (by decide), root39Candidates10_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail11
    (if 3 ≤ root39Candidates10.length then search edge 3 [181, 787] root39Candidates10 968 winner else winner) = _
  rw [if_pos (by decide), root39Child10]
def root39Candidates11 : List Nat := [2683, 2917, 5101, 8647]
set_option maxRecDepth 8192 in
theorem root39Candidates11_checked : candidates edge 4 181 1657 root39Tail12 winner.bound = root39Candidates11 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child11 : search edge 3 [181, 1657] root39Candidates11 1838 winner = winner := by decide
theorem root39Step11 : scan edge (search edge 3) 4 [181] 181 root39Tail11 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail12 winner := by
  rw [root39Tail11, scan, if_neg (by decide), root39Candidates11_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail12
    (if 3 ≤ root39Candidates11.length then search edge 3 [181, 1657] root39Candidates11 1838 winner else winner) = _
  rw [if_pos (by decide), root39Child11]
def root39Candidates12 : List Nat := [15121]
set_option maxRecDepth 8192 in
theorem root39Candidates12_checked : candidates edge 4 181 1759 root39Tail13 winner.bound = root39Candidates12 := by
  unfold candidates
  have row : edge 1759 = (SortedNatTree.contains FastRows.row1759) := by
    change Data.row1759.contains = (SortedNatTree.contains FastRows.row1759)
    exact FastRows.row1759_checked.symm
  rw [row]
  decide
theorem root39Step12 : scan edge (search edge 3) 4 [181] 181 root39Tail12 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail13 winner := by
  rw [root39Tail12, scan, if_neg (by decide), root39Candidates12_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail13
    (if 3 ≤ root39Candidates12.length then search edge 3 [181, 1759] root39Candidates12 1940 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates13 : List Nat := [9109, 11131]
set_option maxRecDepth 8192 in
theorem root39Candidates13_checked : candidates edge 4 181 2269 root39Tail14 winner.bound = root39Candidates13 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
theorem root39Step13 : scan edge (search edge 3) 4 [181] 181 root39Tail13 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail14 winner := by
  rw [root39Tail13, scan, if_neg (by decide), root39Candidates13_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail14
    (if 3 ≤ root39Candidates13.length then search edge 3 [181, 2269] root39Candidates13 2450 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates14 : List Nat := [17977]
set_option maxRecDepth 8192 in
theorem root39Candidates14_checked : candidates edge 4 181 2311 root39Tail15 winner.bound = root39Candidates14 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root39Step14 : scan edge (search edge 3) 4 [181] 181 root39Tail14 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail15 winner := by
  rw [root39Tail14, scan, if_neg (by decide), root39Candidates14_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail15
    (if 3 ≤ root39Candidates14.length then search edge 3 [181, 2311] root39Candidates14 2492 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates15 : List Nat := [19489]
set_option maxRecDepth 8192 in
theorem root39Candidates15_checked : candidates edge 4 181 2683 root39Tail16 winner.bound = root39Candidates15 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root39Step15 : scan edge (search edge 3) 4 [181] 181 root39Tail15 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail16 winner := by
  rw [root39Tail15, scan, if_neg (by decide), root39Candidates15_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail16
    (if 3 ≤ root39Candidates15.length then search edge 3 [181, 2683] root39Candidates15 2864 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates16 : List Nat := [7549]
set_option maxRecDepth 8192 in
theorem root39Candidates16_checked : candidates edge 4 181 2917 root39Tail17 winner.bound = root39Candidates16 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root39Step16 : scan edge (search edge 3) 4 [181] 181 root39Tail16 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail17 winner := by
  rw [root39Tail16, scan, if_neg (by decide), root39Candidates16_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail17
    (if 3 ≤ root39Candidates16.length then search edge 3 [181, 2917] root39Candidates16 3098 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates17 : List Nat := [10531, 15739]
set_option maxRecDepth 8192 in
theorem root39Candidates17_checked : candidates edge 4 181 3391 root39Tail18 winner.bound = root39Candidates17 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
theorem root39Step17 : scan edge (search edge 3) 4 [181] 181 root39Tail17 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail18 winner := by
  rw [root39Tail17, scan, if_neg (by decide), root39Candidates17_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail18
    (if 3 ≤ root39Candidates17.length then search edge 3 [181, 3391] root39Candidates17 3572 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root39Candidates18_checked : candidates edge 4 181 3673 root39Tail19 winner.bound = root39Candidates18 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root39Step18 : scan edge (search edge 3) 4 [181] 181 root39Tail18 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail19 winner := by
  rw [root39Tail18, scan, if_neg (by decide), root39Candidates18_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail19
    (if 3 ≤ root39Candidates18.length then search edge 3 [181, 3673] root39Candidates18 3854 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates19 : List Nat := [7549, 8689, 10957]
set_option maxRecDepth 8192 in
theorem root39Candidates19_checked : candidates edge 4 181 4051 root39Tail20 winner.bound = root39Candidates19 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child19 : search edge 3 [181, 4051] root39Candidates19 4232 winner = winner := by decide
theorem root39Step19 : scan edge (search edge 3) 4 [181] 181 root39Tail19 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail20 winner := by
  rw [root39Tail19, scan, if_neg (by decide), root39Candidates19_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail20
    (if 3 ≤ root39Candidates19.length then search edge 3 [181, 4051] root39Candidates19 4232 winner else winner) = _
  rw [if_pos (by decide), root39Child19]
def root39Candidates20 : List Nat := [18013]
set_option maxRecDepth 8192 in
theorem root39Candidates20_checked : candidates edge 4 181 5101 root39Tail21 winner.bound = root39Candidates20 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root39Step20 : scan edge (search edge 3) 4 [181] 181 root39Tail20 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail21 winner := by
  rw [root39Tail20, scan, if_neg (by decide), root39Candidates20_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail21
    (if 3 ≤ root39Candidates20.length then search edge 3 [181, 5101] root39Candidates20 5282 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates21 : List Nat := [12889]
set_option maxRecDepth 8192 in
theorem root39Candidates21_checked : candidates edge 4 181 5323 root39Tail22 winner.bound = root39Candidates21 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
theorem root39Step21 : scan edge (search edge 3) 4 [181] 181 root39Tail21 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail22 winner := by
  rw [root39Tail21, scan, if_neg (by decide), root39Candidates21_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail22
    (if 3 ≤ root39Candidates21.length then search edge 3 [181, 5323] root39Candidates21 5504 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates22 : List Nat := [10333, 15121, 15187, 15583]
set_option maxRecDepth 8192 in
theorem root39Candidates22_checked : candidates edge 4 181 5839 root39Tail23 winner.bound = root39Candidates22 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root39Child22 : search edge 3 [181, 5839] root39Candidates22 6020 winner = winner := by decide
theorem root39Step22 : scan edge (search edge 3) 4 [181] 181 root39Tail22 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail23 winner := by
  rw [root39Tail22, scan, if_neg (by decide), root39Candidates22_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail23
    (if 3 ≤ root39Candidates22.length then search edge 3 [181, 5839] root39Candidates22 6020 winner else winner) = _
  rw [if_pos (by decide), root39Child22]
def root39Candidates23 : List Nat := [15187, 15991]
set_option maxRecDepth 8192 in
theorem root39Candidates23_checked : candidates edge 4 181 6007 root39Tail24 winner.bound = root39Candidates23 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root39Step23 : scan edge (search edge 3) 4 [181] 181 root39Tail23 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail24 winner := by
  rw [root39Tail23, scan, if_neg (by decide), root39Candidates23_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail24
    (if 3 ≤ root39Candidates23.length then search edge 3 [181, 6007] root39Candidates23 6188 winner else winner) = _
  rw [if_neg (by decide)]
def root39Candidates24 : List Nat := [15583, 19237]
set_option maxRecDepth 8192 in
theorem root39Candidates24_checked : candidates edge 4 181 6337 root39Tail25 winner.bound = root39Candidates24 := by
  unfold candidates
  have row : edge 6337 = (SortedNatTree.contains FastRows.row6337) := by
    change Data.row6337.contains = (SortedNatTree.contains FastRows.row6337)
    exact FastRows.row6337_checked.symm
  rw [row]
  decide
theorem root39Step24 : scan edge (search edge 3) 4 [181] 181 root39Tail24 winner =
    scan edge (search edge 3) 4 [181] 181 root39Tail25 winner := by
  rw [root39Tail24, scan, if_neg (by decide), root39Candidates24_checked]
  change scan edge (search edge 3) 4 [181] 181 root39Tail25
    (if 3 ≤ root39Candidates24.length then search edge 3 [181, 6337] root39Candidates24 6518 winner else winner) = _
  rw [if_neg (by decide)]
theorem root39Scan25 : scan edge (search edge 3) 4 [181] 181 root39Tail25 winner = winner := by rw [root39Tail25, scan, if_pos (by decide)]
theorem root39Scan24 : scan edge (search edge 3) 4 [181] 181 root39Tail24 winner = winner :=
  (root39Step24).trans root39Scan25
theorem root39Scan23 : scan edge (search edge 3) 4 [181] 181 root39Tail23 winner = winner :=
  (root39Step23).trans root39Scan24
theorem root39Scan22 : scan edge (search edge 3) 4 [181] 181 root39Tail22 winner = winner :=
  (root39Step22).trans root39Scan23
theorem root39Scan21 : scan edge (search edge 3) 4 [181] 181 root39Tail21 winner = winner :=
  (root39Step21).trans root39Scan22
theorem root39Scan20 : scan edge (search edge 3) 4 [181] 181 root39Tail20 winner = winner :=
  (root39Step20).trans root39Scan21
theorem root39Scan19 : scan edge (search edge 3) 4 [181] 181 root39Tail19 winner = winner :=
  (root39Step19).trans root39Scan20
theorem root39Scan18 : scan edge (search edge 3) 4 [181] 181 root39Tail18 winner = winner :=
  (root39Step18).trans root39Scan19
theorem root39Scan17 : scan edge (search edge 3) 4 [181] 181 root39Tail17 winner = winner :=
  (root39Step17).trans root39Scan18
theorem root39Scan16 : scan edge (search edge 3) 4 [181] 181 root39Tail16 winner = winner :=
  (root39Step16).trans root39Scan17
theorem root39Scan15 : scan edge (search edge 3) 4 [181] 181 root39Tail15 winner = winner :=
  (root39Step15).trans root39Scan16
theorem root39Scan14 : scan edge (search edge 3) 4 [181] 181 root39Tail14 winner = winner :=
  (root39Step14).trans root39Scan15
theorem root39Scan13 : scan edge (search edge 3) 4 [181] 181 root39Tail13 winner = winner :=
  (root39Step13).trans root39Scan14
theorem root39Scan12 : scan edge (search edge 3) 4 [181] 181 root39Tail12 winner = winner :=
  (root39Step12).trans root39Scan13
theorem root39Scan11 : scan edge (search edge 3) 4 [181] 181 root39Tail11 winner = winner :=
  (root39Step11).trans root39Scan12
theorem root39Scan10 : scan edge (search edge 3) 4 [181] 181 root39Tail10 winner = winner :=
  (root39Step10).trans root39Scan11
theorem root39Scan9 : scan edge (search edge 3) 4 [181] 181 root39Tail9 winner = winner :=
  (root39Step9).trans root39Scan10
theorem root39Scan8 : scan edge (search edge 3) 4 [181] 181 root39Tail8 winner = winner :=
  (root39Step8).trans root39Scan9
theorem root39Scan7 : scan edge (search edge 3) 4 [181] 181 root39Tail7 winner = winner :=
  (root39Step7).trans root39Scan8
theorem root39Scan6 : scan edge (search edge 3) 4 [181] 181 root39Tail6 winner = winner :=
  (root39Step6).trans root39Scan7
theorem root39Scan5 : scan edge (search edge 3) 4 [181] 181 root39Tail5 winner = winner :=
  (root39Step5).trans root39Scan6
theorem root39Scan4 : scan edge (search edge 3) 4 [181] 181 root39Tail4 winner = winner :=
  (root39Step4).trans root39Scan5
theorem root39Scan3 : scan edge (search edge 3) 4 [181] 181 root39Tail3 winner = winner :=
  (root39Step3).trans root39Scan4
theorem root39Scan2 : scan edge (search edge 3) 4 [181] 181 root39Tail2 winner = winner :=
  (root39Step2).trans root39Scan3
theorem root39Scan1 : scan edge (search edge 3) 4 [181] 181 root39Tail1 winner = winner :=
  (root39Step1).trans root39Scan2
theorem root39Scan0 : scan edge (search edge 3) 4 [181] 181 root39Tail0 winner = winner :=
  (root39Step0).trans root39Scan1
set_option maxRecDepth 4096 in
theorem root39Branch : search edge 4 [181] root39Candidates 181 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [181] 181 root39Tail0 winner = winner
  exact root39Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
