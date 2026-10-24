import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root36Candidates : List Nat := Data.row167
set_option maxRecDepth 8192 in
theorem root36Candidates_checked : candidates edge 5 0 167 tail37 winner.bound = root36Candidates := by
  apply candidates_eq_row edge 5 0 167 tail37 root36Candidates winner.bound tail37_ordered
  · decide
  · decide
  · funext q
    rfl
def root36Tail75 : List Nat := []
def root36Tail74 : List Nat := 25799 :: root36Tail75
def root36Tail73 : List Nat := 25349 :: root36Tail74
def root36Tail72 : List Nat := 25343 :: root36Tail73
def root36Tail71 : List Nat := 24977 :: root36Tail72
def root36Tail70 : List Nat := 24551 :: root36Tail71
def root36Tail69 : List Nat := 23873 :: root36Tail70
def root36Tail68 : List Nat := 23159 :: root36Tail69
def root36Tail67 : List Nat := 22961 :: root36Tail68
def root36Tail66 : List Nat := 22697 :: root36Tail67
def root36Tail65 : List Nat := 22481 :: root36Tail66
def root36Tail64 : List Nat := 21911 :: root36Tail65
def root36Tail63 : List Nat := 21611 :: root36Tail64
def root36Tail62 : List Nat := 20477 :: root36Tail63
def root36Tail61 : List Nat := 20441 :: root36Tail62
def root36Tail60 : List Nat := 20063 :: root36Tail61
def root36Tail59 : List Nat := 19763 :: root36Tail60
def root36Tail58 : List Nat := 19709 :: root36Tail59
def root36Tail57 : List Nat := 19559 :: root36Tail58
def root36Tail56 : List Nat := 19427 :: root36Tail57
def root36Tail55 : List Nat := 19163 :: root36Tail56
def root36Tail54 : List Nat := 18839 :: root36Tail55
def root36Tail53 : List Nat := 18743 :: root36Tail54
def root36Tail52 : List Nat := 18371 :: root36Tail53
def root36Tail51 : List Nat := 17957 :: root36Tail52
def root36Tail50 : List Nat := 17231 :: root36Tail51
def root36Tail49 : List Nat := 17027 :: root36Tail50
def root36Tail48 : List Nat := 16943 :: root36Tail49
def root36Tail47 : List Nat := 16691 :: root36Tail48
def root36Tail46 : List Nat := 16481 :: root36Tail47
def root36Tail45 : List Nat := 15053 :: root36Tail46
def root36Tail44 : List Nat := 14717 :: root36Tail45
def root36Tail43 : List Nat := 14669 :: root36Tail44
def root36Tail42 : List Nat := 14303 :: root36Tail43
def root36Tail41 : List Nat := 13457 :: root36Tail42
def root36Tail40 : List Nat := 12539 :: root36Tail41
def root36Tail39 : List Nat := 12107 :: root36Tail40
def root36Tail38 : List Nat := 10973 :: root36Tail39
def root36Tail37 : List Nat := 10613 :: root36Tail38
def root36Tail36 : List Nat := 10559 :: root36Tail37
def root36Tail35 : List Nat := 10487 :: root36Tail36
def root36Tail34 : List Nat := 10181 :: root36Tail35
def root36Tail33 : List Nat := 9857 :: root36Tail34
def root36Tail32 : List Nat := 9521 :: root36Tail33
def root36Tail31 : List Nat := 8627 :: root36Tail32
def root36Tail30 : List Nat := 8543 :: root36Tail31
def root36Tail29 : List Nat := 8429 :: root36Tail30
def root36Tail28 : List Nat := 8423 :: root36Tail29
def root36Tail27 : List Nat := 8009 :: root36Tail28
def root36Tail26 : List Nat := 7877 :: root36Tail27
def root36Tail25 : List Nat := 7673 :: root36Tail26
def root36Tail24 : List Nat := 7451 :: root36Tail25
def root36Tail23 : List Nat := 7253 :: root36Tail24
def root36Tail22 : List Nat := 6947 :: root36Tail23
def root36Tail21 : List Nat := 6827 :: root36Tail22
def root36Tail20 : List Nat := 6053 :: root36Tail21
def root36Tail19 : List Nat := 5981 :: root36Tail20
def root36Tail18 : List Nat := 5441 :: root36Tail19
def root36Tail17 : List Nat := 4733 :: root36Tail18
def root36Tail16 : List Nat := 4271 :: root36Tail17
def root36Tail15 : List Nat := 3719 :: root36Tail16
def root36Tail14 : List Nat := 3461 :: root36Tail15
def root36Tail13 : List Nat := 3011 :: root36Tail14
def root36Tail12 : List Nat := 2999 :: root36Tail13
def root36Tail11 : List Nat := 2963 :: root36Tail12
def root36Tail10 : List Nat := 2441 :: root36Tail11
def root36Tail9 : List Nat := 2129 :: root36Tail10
def root36Tail8 : List Nat := 2087 :: root36Tail9
def root36Tail7 : List Nat := 2063 :: root36Tail8
def root36Tail6 : List Nat := 911 :: root36Tail7
def root36Tail5 : List Nat := 677 :: root36Tail6
def root36Tail4 : List Nat := 641 :: root36Tail5
def root36Tail3 : List Nat := 521 :: root36Tail4
def root36Tail2 : List Nat := 491 :: root36Tail3
def root36Tail1 : List Nat := 443 :: root36Tail2
def root36Tail0 : List Nat := 269 :: root36Tail1
def root36Candidates0 : List Nat := [4733, 6053]
set_option maxRecDepth 8192 in
theorem root36Candidates0_checked : candidates edge 4 167 269 root36Tail1 winner.bound = root36Candidates0 := by
  unfold candidates
  have row : edge 269 = (SortedNatTree.contains FastRows.row269) := by
    change Data.row269.contains = (SortedNatTree.contains FastRows.row269)
    exact FastRows.row269_checked.symm
  rw [row]
  decide
theorem root36Step0 : scan edge (search edge 3) 4 [167] 167 root36Tail0 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail1 winner := by
  rw [root36Tail0, scan, if_neg (by decide), root36Candidates0_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail1
    (if 3 ≤ root36Candidates0.length then search edge 3 [167, 269] root36Candidates0 436 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates1 : List Nat := [9857, 10181, 15053, 19559, 20063]
set_option maxRecDepth 8192 in
theorem root36Candidates1_checked : candidates edge 4 167 443 root36Tail2 winner.bound = root36Candidates1 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root36Child1 : search edge 3 [167, 443] root36Candidates1 610 winner = winner := by decide
theorem root36Step1 : scan edge (search edge 3) 4 [167] 167 root36Tail1 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail2 winner := by
  rw [root36Tail1, scan, if_neg (by decide), root36Candidates1_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail2
    (if 3 ≤ root36Candidates1.length then search edge 3 [167, 443] root36Candidates1 610 winner else winner) = _
  rw [if_pos (by decide), root36Child1]
def root36Candidates2 : List Nat := [6053, 10559, 19163]
set_option maxRecDepth 8192 in
theorem root36Candidates2_checked : candidates edge 4 167 491 root36Tail3 winner.bound = root36Candidates2 := by
  unfold candidates
  have row : edge 491 = (SortedNatTree.contains FastRows.row491) := by
    change Data.row491.contains = (SortedNatTree.contains FastRows.row491)
    exact FastRows.row491_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root36Child2 : search edge 3 [167, 491] root36Candidates2 658 winner = winner := by decide
theorem root36Step2 : scan edge (search edge 3) 4 [167] 167 root36Tail2 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail3 winner := by
  rw [root36Tail2, scan, if_neg (by decide), root36Candidates2_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail3
    (if 3 ≤ root36Candidates2.length then search edge 3 [167, 491] root36Candidates2 658 winner else winner) = _
  rw [if_pos (by decide), root36Child2]
def root36Candidates3 : List Nat := [641, 6053, 7673, 22481]
set_option maxRecDepth 8192 in
theorem root36Candidates3_checked : candidates edge 4 167 521 root36Tail4 winner.bound = root36Candidates3 := by
  unfold candidates
  have row : edge 521 = (SortedNatTree.contains FastRows.row521) := by
    change Data.row521.contains = (SortedNatTree.contains FastRows.row521)
    exact FastRows.row521_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root36Child3 : search edge 3 [167, 521] root36Candidates3 688 winner = winner := by decide
theorem root36Step3 : scan edge (search edge 3) 4 [167] 167 root36Tail3 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail4 winner := by
  rw [root36Tail3, scan, if_neg (by decide), root36Candidates3_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail4
    (if 3 ≤ root36Candidates3.length then search edge 3 [167, 521] root36Candidates3 688 winner else winner) = _
  rw [if_pos (by decide), root36Child3]
def root36Candidates4 : List Nat := [5441, 8543, 20477]
set_option maxRecDepth 8192 in
theorem root36Candidates4_checked : candidates edge 4 167 641 root36Tail5 winner.bound = root36Candidates4 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root36Child4 : search edge 3 [167, 641] root36Candidates4 808 winner = winner := by decide
theorem root36Step4 : scan edge (search edge 3) 4 [167] 167 root36Tail4 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail5 winner := by
  rw [root36Tail4, scan, if_neg (by decide), root36Candidates4_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail5
    (if 3 ≤ root36Candidates4.length then search edge 3 [167, 641] root36Candidates4 808 winner else winner) = _
  rw [if_pos (by decide), root36Child4]
def root36Candidates5 : List Nat := [7451, 22961]
set_option maxRecDepth 8192 in
theorem root36Candidates5_checked : candidates edge 4 167 677 root36Tail6 winner.bound = root36Candidates5 := by
  unfold candidates
  have row : edge 677 = (SortedNatTree.contains FastRows.row677) := by
    change Data.row677.contains = (SortedNatTree.contains FastRows.row677)
    exact FastRows.row677_checked.symm
  rw [row]
  decide
theorem root36Step5 : scan edge (search edge 3) 4 [167] 167 root36Tail5 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail6 winner := by
  rw [root36Tail5, scan, if_neg (by decide), root36Candidates5_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail6
    (if 3 ≤ root36Candidates5.length then search edge 3 [167, 677] root36Candidates5 844 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates6 : List Nat := [17231]
set_option maxRecDepth 8192 in
theorem root36Candidates6_checked : candidates edge 4 167 911 root36Tail7 winner.bound = root36Candidates6 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root36Step6 : scan edge (search edge 3) 4 [167] 167 root36Tail6 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail7 winner := by
  rw [root36Tail6, scan, if_neg (by decide), root36Candidates6_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail7
    (if 3 ≤ root36Candidates6.length then search edge 3 [167, 911] root36Candidates6 1078 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates7 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root36Candidates7_checked : candidates edge 4 167 2063 root36Tail8 winner.bound = root36Candidates7 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root36Step7 : scan edge (search edge 3) 4 [167] 167 root36Tail7 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail8 winner := by
  rw [root36Tail7, scan, if_neg (by decide), root36Candidates7_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail8
    (if 3 ≤ root36Candidates7.length then search edge 3 [167, 2063] root36Candidates7 2230 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates8 : List Nat := [17231]
set_option maxRecDepth 8192 in
theorem root36Candidates8_checked : candidates edge 4 167 2087 root36Tail9 winner.bound = root36Candidates8 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root36Step8 : scan edge (search edge 3) 4 [167] 167 root36Tail8 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail9 winner := by
  rw [root36Tail8, scan, if_neg (by decide), root36Candidates8_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail9
    (if 3 ≤ root36Candidates8.length then search edge 3 [167, 2087] root36Candidates8 2254 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates9 : List Nat := [4271, 16481]
set_option maxRecDepth 8192 in
theorem root36Candidates9_checked : candidates edge 4 167 2129 root36Tail10 winner.bound = root36Candidates9 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root36Step9 : scan edge (search edge 3) 4 [167] 167 root36Tail9 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail10 winner := by
  rw [root36Tail9, scan, if_neg (by decide), root36Candidates9_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail10
    (if 3 ≤ root36Candidates9.length then search edge 3 [167, 2129] root36Candidates9 2296 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates10 : List Nat := [8423, 20477]
set_option maxRecDepth 8192 in
theorem root36Candidates10_checked : candidates edge 4 167 2441 root36Tail11 winner.bound = root36Candidates10 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
theorem root36Step10 : scan edge (search edge 3) 4 [167] 167 root36Tail10 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail11 winner := by
  rw [root36Tail10, scan, if_neg (by decide), root36Candidates10_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail11
    (if 3 ≤ root36Candidates10.length then search edge 3 [167, 2441] root36Candidates10 2608 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates11 : List Nat := [14303, 19427]
set_option maxRecDepth 8192 in
theorem root36Candidates11_checked : candidates edge 4 167 2963 root36Tail12 winner.bound = root36Candidates11 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root36Step11 : scan edge (search edge 3) 4 [167] 167 root36Tail11 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail12 winner := by
  rw [root36Tail11, scan, if_neg (by decide), root36Candidates11_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail12
    (if 3 ≤ root36Candidates11.length then search edge 3 [167, 2963] root36Candidates11 3130 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates12 : List Nat := [14669]
set_option maxRecDepth 8192 in
theorem root36Candidates12_checked : candidates edge 4 167 2999 root36Tail13 winner.bound = root36Candidates12 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root36Step12 : scan edge (search edge 3) 4 [167] 167 root36Tail12 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail13 winner := by
  rw [root36Tail12, scan, if_neg (by decide), root36Candidates12_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail13
    (if 3 ≤ root36Candidates12.length then search edge 3 [167, 2999] root36Candidates12 3166 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates13 : List Nat := [8009]
set_option maxRecDepth 8192 in
theorem root36Candidates13_checked : candidates edge 4 167 3011 root36Tail14 winner.bound = root36Candidates13 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root36Step13 : scan edge (search edge 3) 4 [167] 167 root36Tail13 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail14 winner := by
  rw [root36Tail13, scan, if_neg (by decide), root36Candidates13_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail14
    (if 3 ≤ root36Candidates13.length then search edge 3 [167, 3011] root36Candidates13 3178 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates14 : List Nat := [8543, 10559, 18371]
set_option maxRecDepth 8192 in
theorem root36Candidates14_checked : candidates edge 4 167 3461 root36Tail15 winner.bound = root36Candidates14 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root36Child14 : search edge 3 [167, 3461] root36Candidates14 3628 winner = winner := by decide
theorem root36Step14 : scan edge (search edge 3) 4 [167] 167 root36Tail14 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail15 winner := by
  rw [root36Tail14, scan, if_neg (by decide), root36Candidates14_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail15
    (if 3 ≤ root36Candidates14.length then search edge 3 [167, 3461] root36Candidates14 3628 winner else winner) = _
  rw [if_pos (by decide), root36Child14]
def root36Candidates15 : List Nat := [4271]
set_option maxRecDepth 8192 in
theorem root36Candidates15_checked : candidates edge 4 167 3719 root36Tail16 winner.bound = root36Candidates15 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root36Step15 : scan edge (search edge 3) 4 [167] 167 root36Tail15 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail16 winner := by
  rw [root36Tail15, scan, if_neg (by decide), root36Candidates15_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail16
    (if 3 ≤ root36Candidates15.length then search edge 3 [167, 3719] root36Candidates15 3886 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates16 : List Nat := [16691, 20063]
set_option maxRecDepth 8192 in
theorem root36Candidates16_checked : candidates edge 4 167 4271 root36Tail17 winner.bound = root36Candidates16 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root36Step16 : scan edge (search edge 3) 4 [167] 167 root36Tail16 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail17 winner := by
  rw [root36Tail16, scan, if_neg (by decide), root36Candidates16_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail17
    (if 3 ≤ root36Candidates16.length then search edge 3 [167, 4271] root36Candidates16 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root36Candidates17_checked : candidates edge 4 167 4733 root36Tail18 winner.bound = root36Candidates17 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root36Step17 : scan edge (search edge 3) 4 [167] 167 root36Tail17 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail18 winner := by
  rw [root36Tail17, scan, if_neg (by decide), root36Candidates17_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail18
    (if 3 ≤ root36Candidates17.length then search edge 3 [167, 4733] root36Candidates17 4900 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates18 : List Nat := [12539]
set_option maxRecDepth 8192 in
theorem root36Candidates18_checked : candidates edge 4 167 5441 root36Tail19 winner.bound = root36Candidates18 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root36Step18 : scan edge (search edge 3) 4 [167] 167 root36Tail18 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail19 winner := by
  rw [root36Tail18, scan, if_neg (by decide), root36Candidates18_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail19
    (if 3 ≤ root36Candidates18.length then search edge 3 [167, 5441] root36Candidates18 5608 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates19 : List Nat := [8543, 14669]
set_option maxRecDepth 8192 in
theorem root36Candidates19_checked : candidates edge 4 167 5981 root36Tail20 winner.bound = root36Candidates19 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root36Step19 : scan edge (search edge 3) 4 [167] 167 root36Tail19 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail20 winner := by
  rw [root36Tail19, scan, if_neg (by decide), root36Candidates19_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail20
    (if 3 ≤ root36Candidates19.length then search edge 3 [167, 5981] root36Candidates19 6148 winner else winner) = _
  rw [if_neg (by decide)]
def root36Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root36Candidates20_checked : candidates edge 4 167 6053 root36Tail21 winner.bound = root36Candidates20 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root36Step20 : scan edge (search edge 3) 4 [167] 167 root36Tail20 winner =
    scan edge (search edge 3) 4 [167] 167 root36Tail21 winner := by
  rw [root36Tail20, scan, if_neg (by decide), root36Candidates20_checked]
  change scan edge (search edge 3) 4 [167] 167 root36Tail21
    (if 3 ≤ root36Candidates20.length then search edge 3 [167, 6053] root36Candidates20 6220 winner else winner) = _
  rw [if_neg (by decide)]
theorem root36Scan21 : scan edge (search edge 3) 4 [167] 167 root36Tail21 winner = winner := by rw [root36Tail21, scan, if_pos (by decide)]
theorem root36Scan20 : scan edge (search edge 3) 4 [167] 167 root36Tail20 winner = winner :=
  (root36Step20).trans root36Scan21
theorem root36Scan19 : scan edge (search edge 3) 4 [167] 167 root36Tail19 winner = winner :=
  (root36Step19).trans root36Scan20
theorem root36Scan18 : scan edge (search edge 3) 4 [167] 167 root36Tail18 winner = winner :=
  (root36Step18).trans root36Scan19
theorem root36Scan17 : scan edge (search edge 3) 4 [167] 167 root36Tail17 winner = winner :=
  (root36Step17).trans root36Scan18
theorem root36Scan16 : scan edge (search edge 3) 4 [167] 167 root36Tail16 winner = winner :=
  (root36Step16).trans root36Scan17
theorem root36Scan15 : scan edge (search edge 3) 4 [167] 167 root36Tail15 winner = winner :=
  (root36Step15).trans root36Scan16
theorem root36Scan14 : scan edge (search edge 3) 4 [167] 167 root36Tail14 winner = winner :=
  (root36Step14).trans root36Scan15
theorem root36Scan13 : scan edge (search edge 3) 4 [167] 167 root36Tail13 winner = winner :=
  (root36Step13).trans root36Scan14
theorem root36Scan12 : scan edge (search edge 3) 4 [167] 167 root36Tail12 winner = winner :=
  (root36Step12).trans root36Scan13
theorem root36Scan11 : scan edge (search edge 3) 4 [167] 167 root36Tail11 winner = winner :=
  (root36Step11).trans root36Scan12
theorem root36Scan10 : scan edge (search edge 3) 4 [167] 167 root36Tail10 winner = winner :=
  (root36Step10).trans root36Scan11
theorem root36Scan9 : scan edge (search edge 3) 4 [167] 167 root36Tail9 winner = winner :=
  (root36Step9).trans root36Scan10
theorem root36Scan8 : scan edge (search edge 3) 4 [167] 167 root36Tail8 winner = winner :=
  (root36Step8).trans root36Scan9
theorem root36Scan7 : scan edge (search edge 3) 4 [167] 167 root36Tail7 winner = winner :=
  (root36Step7).trans root36Scan8
theorem root36Scan6 : scan edge (search edge 3) 4 [167] 167 root36Tail6 winner = winner :=
  (root36Step6).trans root36Scan7
theorem root36Scan5 : scan edge (search edge 3) 4 [167] 167 root36Tail5 winner = winner :=
  (root36Step5).trans root36Scan6
theorem root36Scan4 : scan edge (search edge 3) 4 [167] 167 root36Tail4 winner = winner :=
  (root36Step4).trans root36Scan5
theorem root36Scan3 : scan edge (search edge 3) 4 [167] 167 root36Tail3 winner = winner :=
  (root36Step3).trans root36Scan4
theorem root36Scan2 : scan edge (search edge 3) 4 [167] 167 root36Tail2 winner = winner :=
  (root36Step2).trans root36Scan3
theorem root36Scan1 : scan edge (search edge 3) 4 [167] 167 root36Tail1 winner = winner :=
  (root36Step1).trans root36Scan2
theorem root36Scan0 : scan edge (search edge 3) 4 [167] 167 root36Tail0 winner = winner :=
  (root36Step0).trans root36Scan1
set_option maxRecDepth 4096 in
theorem root36Branch : search edge 4 [167] root36Candidates 167 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [167] 167 root36Tail0 winner = winner
  exact root36Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
