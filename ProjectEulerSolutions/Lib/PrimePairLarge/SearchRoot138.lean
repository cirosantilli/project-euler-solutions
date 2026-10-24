import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root138Candidates : List Nat := Data.row811
set_option maxRecDepth 8192 in
theorem root138Candidates_checked : candidates edge 5 0 811 tail139 winner.bound = root138Candidates := by
  apply candidates_eq_row edge 5 0 811 tail139 root138Candidates winner.bound tail139_ordered
  · decide
  · decide
  · funext q
    rfl
def root138Tail61 : List Nat := []
def root138Tail60 : List Nat := 24481 :: root138Tail61
def root138Tail59 : List Nat := 24181 :: root138Tail60
def root138Tail58 : List Nat := 23143 :: root138Tail59
def root138Tail57 : List Nat := 22159 :: root138Tail58
def root138Tail56 : List Nat := 20323 :: root138Tail57
def root138Tail55 : List Nat := 20071 :: root138Tail56
def root138Tail54 : List Nat := 19993 :: root138Tail55
def root138Tail53 : List Nat := 19237 :: root138Tail54
def root138Tail52 : List Nat := 19009 :: root138Tail53
def root138Tail51 : List Nat := 18637 :: root138Tail52
def root138Tail50 : List Nat := 18397 :: root138Tail51
def root138Tail49 : List Nat := 18229 :: root138Tail50
def root138Tail48 : List Nat := 17977 :: root138Tail49
def root138Tail47 : List Nat := 17923 :: root138Tail48
def root138Tail46 : List Nat := 17551 :: root138Tail47
def root138Tail45 : List Nat := 17383 :: root138Tail46
def root138Tail44 : List Nat := 16249 :: root138Tail45
def root138Tail43 : List Nat := 15901 :: root138Tail44
def root138Tail42 : List Nat := 15439 :: root138Tail43
def root138Tail41 : List Nat := 15277 :: root138Tail42
def root138Tail40 : List Nat := 15271 :: root138Tail41
def root138Tail39 : List Nat := 15193 :: root138Tail40
def root138Tail38 : List Nat := 15061 :: root138Tail39
def root138Tail37 : List Nat := 13567 :: root138Tail38
def root138Tail36 : List Nat := 13339 :: root138Tail37
def root138Tail35 : List Nat := 12841 :: root138Tail36
def root138Tail34 : List Nat := 12799 :: root138Tail35
def root138Tail33 : List Nat := 12289 :: root138Tail34
def root138Tail32 : List Nat := 11617 :: root138Tail33
def root138Tail31 : List Nat := 11257 :: root138Tail32
def root138Tail30 : List Nat := 10567 :: root138Tail31
def root138Tail29 : List Nat := 9679 :: root138Tail30
def root138Tail28 : List Nat := 8233 :: root138Tail29
def root138Tail27 : List Nat := 8191 :: root138Tail28
def root138Tail26 : List Nat := 8179 :: root138Tail27
def root138Tail25 : List Nat := 8101 :: root138Tail26
def root138Tail24 : List Nat := 7561 :: root138Tail25
def root138Tail23 : List Nat := 6967 :: root138Tail24
def root138Tail22 : List Nat := 6547 :: root138Tail23
def root138Tail21 : List Nat := 6469 :: root138Tail22
def root138Tail20 : List Nat := 6133 :: root138Tail21
def root138Tail19 : List Nat := 5791 :: root138Tail20
def root138Tail18 : List Nat := 5449 :: root138Tail19
def root138Tail17 : List Nat := 5119 :: root138Tail18
def root138Tail16 : List Nat := 4861 :: root138Tail17
def root138Tail15 : List Nat := 4201 :: root138Tail16
def root138Tail14 : List Nat := 3967 :: root138Tail15
def root138Tail13 : List Nat := 3691 :: root138Tail14
def root138Tail12 : List Nat := 3643 :: root138Tail13
def root138Tail11 : List Nat := 3631 :: root138Tail12
def root138Tail10 : List Nat := 3109 :: root138Tail11
def root138Tail9 : List Nat := 2011 :: root138Tail10
def root138Tail8 : List Nat := 1723 :: root138Tail9
def root138Tail7 : List Nat := 1657 :: root138Tail8
def root138Tail6 : List Nat := 1531 :: root138Tail7
def root138Tail5 : List Nat := 1459 :: root138Tail6
def root138Tail4 : List Nat := 1447 :: root138Tail5
def root138Tail3 : List Nat := 1171 :: root138Tail4
def root138Tail2 : List Nat := 997 :: root138Tail3
def root138Tail1 : List Nat := 991 :: root138Tail2
def root138Tail0 : List Nat := 919 :: root138Tail1
def root138Candidates0 : List Nat := [3643, 24181]
set_option maxRecDepth 8192 in
theorem root138Candidates0_checked : candidates edge 4 811 919 root138Tail1 winner.bound = root138Candidates0 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
theorem root138Step0 : scan edge (search edge 3) 4 [811] 811 root138Tail0 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail1 winner := by
  rw [root138Tail0, scan, if_neg (by decide), root138Candidates0_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail1
    (if 3 ≤ root138Candidates0.length then search edge 3 [811, 919] root138Candidates0 1730 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates1 : List Nat := [1171]
set_option maxRecDepth 8192 in
theorem root138Candidates1_checked : candidates edge 4 811 991 root138Tail2 winner.bound = root138Candidates1 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
theorem root138Step1 : scan edge (search edge 3) 4 [811] 811 root138Tail1 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail2 winner := by
  rw [root138Tail1, scan, if_neg (by decide), root138Candidates1_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail2
    (if 3 ≤ root138Candidates1.length then search edge 3 [811, 991] root138Candidates1 1802 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates2 : List Nat := [15901]
set_option maxRecDepth 8192 in
theorem root138Candidates2_checked : candidates edge 4 811 997 root138Tail3 winner.bound = root138Candidates2 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
theorem root138Step2 : scan edge (search edge 3) 4 [811] 811 root138Tail2 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail3 winner := by
  rw [root138Tail2, scan, if_neg (by decide), root138Candidates2_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail3
    (if 3 ≤ root138Candidates2.length then search edge 3 [811, 997] root138Candidates2 1808 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates3 : List Nat := [2011]
set_option maxRecDepth 8192 in
theorem root138Candidates3_checked : candidates edge 4 811 1171 root138Tail4 winner.bound = root138Candidates3 := by
  unfold candidates
  have row : edge 1171 = (SortedNatTree.contains FastRows.row1171) := by
    change Data.row1171.contains = (SortedNatTree.contains FastRows.row1171)
    exact FastRows.row1171_checked.symm
  rw [row]
  decide
theorem root138Step3 : scan edge (search edge 3) 4 [811] 811 root138Tail3 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail4 winner := by
  rw [root138Tail3, scan, if_neg (by decide), root138Candidates3_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail4
    (if 3 ≤ root138Candidates3.length then search edge 3 [811, 1171] root138Candidates3 1982 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates4 : List Nat := [2011, 6967]
set_option maxRecDepth 8192 in
theorem root138Candidates4_checked : candidates edge 4 811 1447 root138Tail5 winner.bound = root138Candidates4 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
theorem root138Step4 : scan edge (search edge 3) 4 [811] 811 root138Tail4 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail5 winner := by
  rw [root138Tail4, scan, if_neg (by decide), root138Candidates4_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail5
    (if 3 ≤ root138Candidates4.length then search edge 3 [811, 1447] root138Candidates4 2258 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates5 : List Nat := [1657, 15277]
set_option maxRecDepth 8192 in
theorem root138Candidates5_checked : candidates edge 4 811 1459 root138Tail6 winner.bound = root138Candidates5 := by
  unfold candidates
  have row : edge 1459 = (SortedNatTree.contains FastRows.row1459) := by
    change Data.row1459.contains = (SortedNatTree.contains FastRows.row1459)
    exact FastRows.row1459_checked.symm
  rw [row]
  decide
theorem root138Step5 : scan edge (search edge 3) 4 [811] 811 root138Tail5 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail6 winner := by
  rw [root138Tail5, scan, if_neg (by decide), root138Candidates5_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail6
    (if 3 ≤ root138Candidates5.length then search edge 3 [811, 1459] root138Candidates5 2270 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates6 : List Nat := [5119, 17383, 20323, 23143]
set_option maxRecDepth 8192 in
theorem root138Candidates6_checked : candidates edge 4 811 1531 root138Tail7 winner.bound = root138Candidates6 := by
  unfold candidates
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root138Child6 : search edge 3 [811, 1531] root138Candidates6 2342 winner = winner := by decide
theorem root138Step6 : scan edge (search edge 3) 4 [811] 811 root138Tail6 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail7 winner := by
  rw [root138Tail6, scan, if_neg (by decide), root138Candidates6_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail7
    (if 3 ≤ root138Candidates6.length then search edge 3 [811, 1531] root138Candidates6 2342 winner else winner) = _
  rw [if_pos (by decide), root138Child6]
def root138Candidates7 : List Nat := [3967]
set_option maxRecDepth 8192 in
theorem root138Candidates7_checked : candidates edge 4 811 1657 root138Tail8 winner.bound = root138Candidates7 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
theorem root138Step7 : scan edge (search edge 3) 4 [811] 811 root138Tail7 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail8 winner := by
  rw [root138Tail7, scan, if_neg (by decide), root138Candidates7_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail8
    (if 3 ≤ root138Candidates7.length then search edge 3 [811, 1657] root138Candidates7 2468 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates8 : List Nat := [6547, 8233, 15271, 18637]
set_option maxRecDepth 8192 in
theorem root138Candidates8_checked : candidates edge 4 811 1723 root138Tail9 winner.bound = root138Candidates8 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root138Child8 : search edge 3 [811, 1723] root138Candidates8 2534 winner = winner := by decide
theorem root138Step8 : scan edge (search edge 3) 4 [811] 811 root138Tail8 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail9 winner := by
  rw [root138Tail8, scan, if_neg (by decide), root138Candidates8_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail9
    (if 3 ≤ root138Candidates8.length then search edge 3 [811, 1723] root138Candidates8 2534 winner else winner) = _
  rw [if_pos (by decide), root138Child8]
def root138Candidates9 : List Nat := [20323]
set_option maxRecDepth 8192 in
theorem root138Candidates9_checked : candidates edge 4 811 2011 root138Tail10 winner.bound = root138Candidates9 := by
  unfold candidates
  have row : edge 2011 = (SortedNatTree.contains FastRows.row2011) := by
    change Data.row2011.contains = (SortedNatTree.contains FastRows.row2011)
    exact FastRows.row2011_checked.symm
  rw [row]
  decide
theorem root138Step9 : scan edge (search edge 3) 4 [811] 811 root138Tail9 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail10 winner := by
  rw [root138Tail9, scan, if_neg (by decide), root138Candidates9_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail10
    (if 3 ≤ root138Candidates9.length then search edge 3 [811, 2011] root138Candidates9 2822 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates10 : List Nat := [5791, 15271]
set_option maxRecDepth 8192 in
theorem root138Candidates10_checked : candidates edge 4 811 3109 root138Tail11 winner.bound = root138Candidates10 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root138Step10 : scan edge (search edge 3) 4 [811] 811 root138Tail10 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail11 winner := by
  rw [root138Tail10, scan, if_neg (by decide), root138Candidates10_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail11
    (if 3 ≤ root138Candidates10.length then search edge 3 [811, 3109] root138Candidates10 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates11 : List Nat := [5119, 6547, 18397]
set_option maxRecDepth 8192 in
theorem root138Candidates11_checked : candidates edge 4 811 3631 root138Tail12 winner.bound = root138Candidates11 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root138Child11 : search edge 3 [811, 3631] root138Candidates11 4442 winner = winner := by decide
theorem root138Step11 : scan edge (search edge 3) 4 [811] 811 root138Tail11 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail12 winner := by
  rw [root138Tail11, scan, if_neg (by decide), root138Candidates11_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail12
    (if 3 ≤ root138Candidates11.length then search edge 3 [811, 3631] root138Candidates11 4442 winner else winner) = _
  rw [if_pos (by decide), root138Child11]
def root138Candidates12 : List Nat := [5449]
set_option maxRecDepth 8192 in
theorem root138Candidates12_checked : candidates edge 4 811 3643 root138Tail13 winner.bound = root138Candidates12 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root138Step12 : scan edge (search edge 3) 4 [811] 811 root138Tail12 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail13 winner := by
  rw [root138Tail12, scan, if_neg (by decide), root138Candidates12_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail13
    (if 3 ≤ root138Candidates12.length then search edge 3 [811, 3643] root138Candidates12 4454 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates13 : List Nat := [7561]
set_option maxRecDepth 8192 in
theorem root138Candidates13_checked : candidates edge 4 811 3691 root138Tail14 winner.bound = root138Candidates13 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
theorem root138Step13 : scan edge (search edge 3) 4 [811] 811 root138Tail13 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail14 winner := by
  rw [root138Tail13, scan, if_neg (by decide), root138Candidates13_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail14
    (if 3 ≤ root138Candidates13.length then search edge 3 [811, 3691] root138Candidates13 4502 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates14 : List Nat := [5791]
set_option maxRecDepth 8192 in
theorem root138Candidates14_checked : candidates edge 4 811 3967 root138Tail15 winner.bound = root138Candidates14 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root138Step14 : scan edge (search edge 3) 4 [811] 811 root138Tail14 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail15 winner := by
  rw [root138Tail14, scan, if_neg (by decide), root138Candidates14_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail15
    (if 3 ≤ root138Candidates14.length then search edge 3 [811, 3967] root138Candidates14 4778 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates15 : List Nat := [19237]
set_option maxRecDepth 8192 in
theorem root138Candidates15_checked : candidates edge 4 811 4201 root138Tail16 winner.bound = root138Candidates15 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root138Step15 : scan edge (search edge 3) 4 [811] 811 root138Tail15 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail16 winner := by
  rw [root138Tail15, scan, if_neg (by decide), root138Candidates15_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail16
    (if 3 ≤ root138Candidates15.length then search edge 3 [811, 4201] root138Candidates15 5012 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates16 : List Nat := [12799, 15439, 18637]
set_option maxRecDepth 8192 in
theorem root138Candidates16_checked : candidates edge 4 811 4861 root138Tail17 winner.bound = root138Candidates16 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root138Child16 : search edge 3 [811, 4861] root138Candidates16 5672 winner = winner := by decide
theorem root138Step16 : scan edge (search edge 3) 4 [811] 811 root138Tail16 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail17 winner := by
  rw [root138Tail16, scan, if_neg (by decide), root138Candidates16_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail17
    (if 3 ≤ root138Candidates16.length then search edge 3 [811, 4861] root138Candidates16 5672 winner else winner) = _
  rw [if_pos (by decide), root138Child16]
def root138Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root138Candidates17_checked : candidates edge 4 811 5119 root138Tail18 winner.bound = root138Candidates17 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root138Step17 : scan edge (search edge 3) 4 [811] 811 root138Tail17 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail18 winner := by
  rw [root138Tail17, scan, if_neg (by decide), root138Candidates17_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail18
    (if 3 ≤ root138Candidates17.length then search edge 3 [811, 5119] root138Candidates17 5930 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates18 : List Nat := [15901]
set_option maxRecDepth 8192 in
theorem root138Candidates18_checked : candidates edge 4 811 5449 root138Tail19 winner.bound = root138Candidates18 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root138Step18 : scan edge (search edge 3) 4 [811] 811 root138Tail18 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail19 winner := by
  rw [root138Tail18, scan, if_neg (by decide), root138Candidates18_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail19
    (if 3 ≤ root138Candidates18.length then search edge 3 [811, 5449] root138Candidates18 6260 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root138Candidates19_checked : candidates edge 4 811 5791 root138Tail20 winner.bound = root138Candidates19 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root138Step19 : scan edge (search edge 3) 4 [811] 811 root138Tail19 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail20 winner := by
  rw [root138Tail19, scan, if_neg (by decide), root138Candidates19_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail20
    (if 3 ≤ root138Candidates19.length then search edge 3 [811, 5791] root138Candidates19 6602 winner else winner) = _
  rw [if_neg (by decide)]
def root138Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root138Candidates20_checked : candidates edge 4 811 6133 root138Tail21 winner.bound = root138Candidates20 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root138Step20 : scan edge (search edge 3) 4 [811] 811 root138Tail20 winner =
    scan edge (search edge 3) 4 [811] 811 root138Tail21 winner := by
  rw [root138Tail20, scan, if_neg (by decide), root138Candidates20_checked]
  change scan edge (search edge 3) 4 [811] 811 root138Tail21
    (if 3 ≤ root138Candidates20.length then search edge 3 [811, 6133] root138Candidates20 6944 winner else winner) = _
  rw [if_neg (by decide)]
theorem root138Scan21 : scan edge (search edge 3) 4 [811] 811 root138Tail21 winner = winner := by rw [root138Tail21, scan, if_pos (by decide)]
theorem root138Scan20 : scan edge (search edge 3) 4 [811] 811 root138Tail20 winner = winner :=
  (root138Step20).trans root138Scan21
theorem root138Scan19 : scan edge (search edge 3) 4 [811] 811 root138Tail19 winner = winner :=
  (root138Step19).trans root138Scan20
theorem root138Scan18 : scan edge (search edge 3) 4 [811] 811 root138Tail18 winner = winner :=
  (root138Step18).trans root138Scan19
theorem root138Scan17 : scan edge (search edge 3) 4 [811] 811 root138Tail17 winner = winner :=
  (root138Step17).trans root138Scan18
theorem root138Scan16 : scan edge (search edge 3) 4 [811] 811 root138Tail16 winner = winner :=
  (root138Step16).trans root138Scan17
theorem root138Scan15 : scan edge (search edge 3) 4 [811] 811 root138Tail15 winner = winner :=
  (root138Step15).trans root138Scan16
theorem root138Scan14 : scan edge (search edge 3) 4 [811] 811 root138Tail14 winner = winner :=
  (root138Step14).trans root138Scan15
theorem root138Scan13 : scan edge (search edge 3) 4 [811] 811 root138Tail13 winner = winner :=
  (root138Step13).trans root138Scan14
theorem root138Scan12 : scan edge (search edge 3) 4 [811] 811 root138Tail12 winner = winner :=
  (root138Step12).trans root138Scan13
theorem root138Scan11 : scan edge (search edge 3) 4 [811] 811 root138Tail11 winner = winner :=
  (root138Step11).trans root138Scan12
theorem root138Scan10 : scan edge (search edge 3) 4 [811] 811 root138Tail10 winner = winner :=
  (root138Step10).trans root138Scan11
theorem root138Scan9 : scan edge (search edge 3) 4 [811] 811 root138Tail9 winner = winner :=
  (root138Step9).trans root138Scan10
theorem root138Scan8 : scan edge (search edge 3) 4 [811] 811 root138Tail8 winner = winner :=
  (root138Step8).trans root138Scan9
theorem root138Scan7 : scan edge (search edge 3) 4 [811] 811 root138Tail7 winner = winner :=
  (root138Step7).trans root138Scan8
theorem root138Scan6 : scan edge (search edge 3) 4 [811] 811 root138Tail6 winner = winner :=
  (root138Step6).trans root138Scan7
theorem root138Scan5 : scan edge (search edge 3) 4 [811] 811 root138Tail5 winner = winner :=
  (root138Step5).trans root138Scan6
theorem root138Scan4 : scan edge (search edge 3) 4 [811] 811 root138Tail4 winner = winner :=
  (root138Step4).trans root138Scan5
theorem root138Scan3 : scan edge (search edge 3) 4 [811] 811 root138Tail3 winner = winner :=
  (root138Step3).trans root138Scan4
theorem root138Scan2 : scan edge (search edge 3) 4 [811] 811 root138Tail2 winner = winner :=
  (root138Step2).trans root138Scan3
theorem root138Scan1 : scan edge (search edge 3) 4 [811] 811 root138Tail1 winner = winner :=
  (root138Step1).trans root138Scan2
theorem root138Scan0 : scan edge (search edge 3) 4 [811] 811 root138Tail0 winner = winner :=
  (root138Step0).trans root138Scan1
set_option maxRecDepth 4096 in
theorem root138Branch : search edge 4 [811] root138Candidates 811 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [811] 811 root138Tail0 winner = winner
  exact root138Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
