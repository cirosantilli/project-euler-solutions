import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root268Candidates : List Nat := Data.row1741
set_option maxRecDepth 8192 in
theorem root268Candidates_checked : candidates edge 5 0 1741 tail269 winner.bound = root268Candidates := by
  apply candidates_eq_row edge 5 0 1741 tail269 root268Candidates winner.bound tail269_ordered
  · decide
  · decide
  · funext q
    rfl
def root268Tail52 : List Nat := []
def root268Tail51 : List Nat := 24169 :: root268Tail52
def root268Tail50 : List Nat := 24103 :: root268Tail51
def root268Tail49 : List Nat := 24061 :: root268Tail50
def root268Tail48 : List Nat := 23857 :: root268Tail49
def root268Tail47 : List Nat := 23623 :: root268Tail48
def root268Tail46 : List Nat := 21169 :: root268Tail47
def root268Tail45 : List Nat := 20641 :: root268Tail46
def root268Tail44 : List Nat := 20593 :: root268Tail45
def root268Tail43 : List Nat := 19927 :: root268Tail44
def root268Tail42 : List Nat := 18121 :: root268Tail43
def root268Tail41 : List Nat := 17737 :: root268Tail42
def root268Tail40 : List Nat := 17467 :: root268Tail41
def root268Tail39 : List Nat := 15919 :: root268Tail40
def root268Tail38 : List Nat := 14887 :: root268Tail39
def root268Tail37 : List Nat := 14653 :: root268Tail38
def root268Tail36 : List Nat := 14029 :: root268Tail37
def root268Tail35 : List Nat := 13309 :: root268Tail36
def root268Tail34 : List Nat := 12973 :: root268Tail35
def root268Tail33 : List Nat := 11887 :: root268Tail34
def root268Tail32 : List Nat := 10837 :: root268Tail33
def root268Tail31 : List Nat := 10159 :: root268Tail32
def root268Tail30 : List Nat := 9091 :: root268Tail31
def root268Tail29 : List Nat := 9007 :: root268Tail30
def root268Tail28 : List Nat := 8893 :: root268Tail29
def root268Tail27 : List Nat := 8539 :: root268Tail28
def root268Tail26 : List Nat := 8419 :: root268Tail27
def root268Tail25 : List Nat := 8287 :: root268Tail26
def root268Tail24 : List Nat := 7993 :: root268Tail25
def root268Tail23 : List Nat := 7573 :: root268Tail24
def root268Tail22 : List Nat := 7489 :: root268Tail23
def root268Tail21 : List Nat := 6397 :: root268Tail22
def root268Tail20 : List Nat := 5581 :: root268Tail21
def root268Tail19 : List Nat := 5557 :: root268Tail20
def root268Tail18 : List Nat := 5479 :: root268Tail19
def root268Tail17 : List Nat := 5227 :: root268Tail18
def root268Tail16 : List Nat := 4003 :: root268Tail17
def root268Tail15 : List Nat := 3919 :: root268Tail16
def root268Tail14 : List Nat := 3793 :: root268Tail15
def root268Tail13 : List Nat := 3607 :: root268Tail14
def root268Tail12 : List Nat := 3373 :: root268Tail13
def root268Tail11 : List Nat := 3313 :: root268Tail12
def root268Tail10 : List Nat := 3307 :: root268Tail11
def root268Tail9 : List Nat := 2971 :: root268Tail10
def root268Tail8 : List Nat := 2953 :: root268Tail9
def root268Tail7 : List Nat := 2887 :: root268Tail8
def root268Tail6 : List Nat := 2851 :: root268Tail7
def root268Tail5 : List Nat := 2347 :: root268Tail6
def root268Tail4 : List Nat := 2287 :: root268Tail5
def root268Tail3 : List Nat := 2281 :: root268Tail4
def root268Tail2 : List Nat := 2203 :: root268Tail3
def root268Tail1 : List Nat := 2053 :: root268Tail2
def root268Tail0 : List Nat := 1753 :: root268Tail1
def root268Candidates0 : List Nat := [8893, 9007, 9091, 10159, 13309]
set_option maxRecDepth 8192 in
theorem root268Candidates0_checked : candidates edge 4 1741 1753 root268Tail1 winner.bound = root268Candidates0 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root268Child0 : search edge 3 [1741, 1753] root268Candidates0 3494 winner = winner := by decide
theorem root268Step0 : scan edge (search edge 3) 4 [1741] 1741 root268Tail0 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail1 winner := by
  rw [root268Tail0, scan, if_neg (by decide), root268Candidates0_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail1
    (if 3 ≤ root268Candidates0.length then search edge 3 [1741, 1753] root268Candidates0 3494 winner else winner) = _
  rw [if_pos (by decide), root268Child0]
def root268Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root268Candidates1_checked : candidates edge 4 1741 2053 root268Tail2 winner.bound = root268Candidates1 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root268Step1 : scan edge (search edge 3) 4 [1741] 1741 root268Tail1 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail2 winner := by
  rw [root268Tail1, scan, if_neg (by decide), root268Candidates1_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail2
    (if 3 ≤ root268Candidates1.length then search edge 3 [1741, 2053] root268Candidates1 3794 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates2 : List Nat := [2887, 5557]
set_option maxRecDepth 8192 in
theorem root268Candidates2_checked : candidates edge 4 1741 2203 root268Tail3 winner.bound = root268Candidates2 := by
  unfold candidates
  have row : edge 2203 = (SortedNatTree.contains FastRows.row2203) := by
    change Data.row2203.contains = (SortedNatTree.contains FastRows.row2203)
    exact FastRows.row2203_checked.symm
  rw [row]
  decide
theorem root268Step2 : scan edge (search edge 3) 4 [1741] 1741 root268Tail2 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail3 winner := by
  rw [root268Tail2, scan, if_neg (by decide), root268Candidates2_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail3
    (if 3 ≤ root268Candidates2.length then search edge 3 [1741, 2203] root268Candidates2 3944 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates3 : List Nat := [2887, 9091]
set_option maxRecDepth 8192 in
theorem root268Candidates3_checked : candidates edge 4 1741 2281 root268Tail4 winner.bound = root268Candidates3 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root268Step3 : scan edge (search edge 3) 4 [1741] 1741 root268Tail3 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail4 winner := by
  rw [root268Tail3, scan, if_neg (by decide), root268Candidates3_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail4
    (if 3 ≤ root268Candidates3.length then search edge 3 [1741, 2281] root268Candidates3 4022 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates4 : List Nat := [3607, 9007, 20641]
set_option maxRecDepth 8192 in
theorem root268Candidates4_checked : candidates edge 4 1741 2287 root268Tail5 winner.bound = root268Candidates4 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root268Child4 : search edge 3 [1741, 2287] root268Candidates4 4028 winner = winner := by decide
theorem root268Step4 : scan edge (search edge 3) 4 [1741] 1741 root268Tail4 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail5 winner := by
  rw [root268Tail4, scan, if_neg (by decide), root268Candidates4_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail5
    (if 3 ≤ root268Candidates4.length then search edge 3 [1741, 2287] root268Candidates4 4028 winner else winner) = _
  rw [if_pos (by decide), root268Child4]
def root268Candidates5 : List Nat := [2851]
set_option maxRecDepth 8192 in
theorem root268Candidates5_checked : candidates edge 4 1741 2347 root268Tail6 winner.bound = root268Candidates5 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
theorem root268Step5 : scan edge (search edge 3) 4 [1741] 1741 root268Tail5 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail6 winner := by
  rw [root268Tail5, scan, if_neg (by decide), root268Candidates5_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail6
    (if 3 ≤ root268Candidates5.length then search edge 3 [1741, 2347] root268Candidates5 4088 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates6 : List Nat := [3313, 14887]
set_option maxRecDepth 8192 in
theorem root268Candidates6_checked : candidates edge 4 1741 2851 root268Tail7 winner.bound = root268Candidates6 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root268Step6 : scan edge (search edge 3) 4 [1741] 1741 root268Tail6 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail7 winner := by
  rw [root268Tail6, scan, if_neg (by decide), root268Candidates6_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail7
    (if 3 ≤ root268Candidates6.length then search edge 3 [1741, 2851] root268Candidates6 4592 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates7 : List Nat := [7573, 12973]
set_option maxRecDepth 8192 in
theorem root268Candidates7_checked : candidates edge 4 1741 2887 root268Tail8 winner.bound = root268Candidates7 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
theorem root268Step7 : scan edge (search edge 3) 4 [1741] 1741 root268Tail7 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail8 winner := by
  rw [root268Tail7, scan, if_neg (by decide), root268Candidates7_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail8
    (if 3 ≤ root268Candidates7.length then search edge 3 [1741, 2887] root268Candidates7 4628 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates8 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root268Candidates8_checked : candidates edge 4 1741 2953 root268Tail9 winner.bound = root268Candidates8 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
theorem root268Step8 : scan edge (search edge 3) 4 [1741] 1741 root268Tail8 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail9 winner := by
  rw [root268Tail8, scan, if_neg (by decide), root268Candidates8_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail9
    (if 3 ≤ root268Candidates8.length then search edge 3 [1741, 2953] root268Candidates8 4694 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates9 : List Nat := [5557, 7573, 18121]
set_option maxRecDepth 8192 in
theorem root268Candidates9_checked : candidates edge 4 1741 2971 root268Tail10 winner.bound = root268Candidates9 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root268Child9 : search edge 3 [1741, 2971] root268Candidates9 4712 winner = winner := by decide
theorem root268Step9 : scan edge (search edge 3) 4 [1741] 1741 root268Tail9 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail10 winner := by
  rw [root268Tail9, scan, if_neg (by decide), root268Candidates9_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail10
    (if 3 ≤ root268Candidates9.length then search edge 3 [1741, 2971] root268Candidates9 4712 winner else winner) = _
  rw [if_pos (by decide), root268Child9]
def root268Candidates10 : List Nat := [9091]
set_option maxRecDepth 8192 in
theorem root268Candidates10_checked : candidates edge 4 1741 3307 root268Tail11 winner.bound = root268Candidates10 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root268Step10 : scan edge (search edge 3) 4 [1741] 1741 root268Tail10 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail11 winner := by
  rw [root268Tail10, scan, if_neg (by decide), root268Candidates10_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail11
    (if 3 ≤ root268Candidates10.length then search edge 3 [1741, 3307] root268Candidates10 5048 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root268Candidates11_checked : candidates edge 4 1741 3313 root268Tail12 winner.bound = root268Candidates11 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root268Step11 : scan edge (search edge 3) 4 [1741] 1741 root268Tail11 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail12 winner := by
  rw [root268Tail11, scan, if_neg (by decide), root268Candidates11_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail12
    (if 3 ≤ root268Candidates11.length then search edge 3 [1741, 3313] root268Candidates11 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates12 : List Nat := [5557]
set_option maxRecDepth 8192 in
theorem root268Candidates12_checked : candidates edge 4 1741 3373 root268Tail13 winner.bound = root268Candidates12 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root268Step12 : scan edge (search edge 3) 4 [1741] 1741 root268Tail12 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail13 winner := by
  rw [root268Tail12, scan, if_neg (by decide), root268Candidates12_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail13
    (if 3 ≤ root268Candidates12.length then search edge 3 [1741, 3373] root268Candidates12 5114 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root268Candidates13_checked : candidates edge 4 1741 3607 root268Tail14 winner.bound = root268Candidates13 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root268Step13 : scan edge (search edge 3) 4 [1741] 1741 root268Tail13 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail14 winner := by
  rw [root268Tail13, scan, if_neg (by decide), root268Candidates13_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail14
    (if 3 ≤ root268Candidates13.length then search edge 3 [1741, 3607] root268Candidates13 5348 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates14 : List Nat := [9007]
set_option maxRecDepth 8192 in
theorem root268Candidates14_checked : candidates edge 4 1741 3793 root268Tail15 winner.bound = root268Candidates14 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
theorem root268Step14 : scan edge (search edge 3) 4 [1741] 1741 root268Tail14 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail15 winner := by
  rw [root268Tail14, scan, if_neg (by decide), root268Candidates14_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail15
    (if 3 ≤ root268Candidates14.length then search edge 3 [1741, 3793] root268Candidates14 5534 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates15 : List Nat := [6397, 7489]
set_option maxRecDepth 8192 in
theorem root268Candidates15_checked : candidates edge 4 1741 3919 root268Tail16 winner.bound = root268Candidates15 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root268Step15 : scan edge (search edge 3) 4 [1741] 1741 root268Tail15 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail16 winner := by
  rw [root268Tail15, scan, if_neg (by decide), root268Candidates15_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail16
    (if 3 ≤ root268Candidates15.length then search edge 3 [1741, 3919] root268Candidates15 5660 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates16 : List Nat := [8539]
set_option maxRecDepth 8192 in
theorem root268Candidates16_checked : candidates edge 4 1741 4003 root268Tail17 winner.bound = root268Candidates16 := by
  unfold candidates
  have row : edge 4003 = (SortedNatTree.contains FastRows.row4003) := by
    change Data.row4003.contains = (SortedNatTree.contains FastRows.row4003)
    exact FastRows.row4003_checked.symm
  rw [row]
  decide
theorem root268Step16 : scan edge (search edge 3) 4 [1741] 1741 root268Tail16 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail17 winner := by
  rw [root268Tail16, scan, if_neg (by decide), root268Candidates16_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail17
    (if 3 ≤ root268Candidates16.length then search edge 3 [1741, 4003] root268Candidates16 5744 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates17 : List Nat := [14029]
set_option maxRecDepth 8192 in
theorem root268Candidates17_checked : candidates edge 4 1741 5227 root268Tail18 winner.bound = root268Candidates17 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root268Step17 : scan edge (search edge 3) 4 [1741] 1741 root268Tail17 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail18 winner := by
  rw [root268Tail17, scan, if_neg (by decide), root268Candidates17_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail18
    (if 3 ≤ root268Candidates17.length then search edge 3 [1741, 5227] root268Candidates17 6968 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates18 : List Nat := [5581]
set_option maxRecDepth 8192 in
theorem root268Candidates18_checked : candidates edge 4 1741 5479 root268Tail19 winner.bound = root268Candidates18 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root268Step18 : scan edge (search edge 3) 4 [1741] 1741 root268Tail18 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail19 winner := by
  rw [root268Tail18, scan, if_neg (by decide), root268Candidates18_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail19
    (if 3 ≤ root268Candidates18.length then search edge 3 [1741, 5479] root268Candidates18 7220 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root268Candidates19_checked : candidates edge 4 1741 5557 root268Tail20 winner.bound = root268Candidates19 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root268Step19 : scan edge (search edge 3) 4 [1741] 1741 root268Tail19 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail20 winner := by
  rw [root268Tail19, scan, if_neg (by decide), root268Candidates19_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail20
    (if 3 ≤ root268Candidates19.length then search edge 3 [1741, 5557] root268Candidates19 7298 winner else winner) = _
  rw [if_neg (by decide)]
def root268Candidates20 : List Nat := [17467]
set_option maxRecDepth 8192 in
theorem root268Candidates20_checked : candidates edge 4 1741 5581 root268Tail21 winner.bound = root268Candidates20 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root268Step20 : scan edge (search edge 3) 4 [1741] 1741 root268Tail20 winner =
    scan edge (search edge 3) 4 [1741] 1741 root268Tail21 winner := by
  rw [root268Tail20, scan, if_neg (by decide), root268Candidates20_checked]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail21
    (if 3 ≤ root268Candidates20.length then search edge 3 [1741, 5581] root268Candidates20 7322 winner else winner) = _
  rw [if_neg (by decide)]
theorem root268Scan21 : scan edge (search edge 3) 4 [1741] 1741 root268Tail21 winner = winner := by rw [root268Tail21, scan, if_pos (by decide)]
theorem root268Scan20 : scan edge (search edge 3) 4 [1741] 1741 root268Tail20 winner = winner :=
  (root268Step20).trans root268Scan21
theorem root268Scan19 : scan edge (search edge 3) 4 [1741] 1741 root268Tail19 winner = winner :=
  (root268Step19).trans root268Scan20
theorem root268Scan18 : scan edge (search edge 3) 4 [1741] 1741 root268Tail18 winner = winner :=
  (root268Step18).trans root268Scan19
theorem root268Scan17 : scan edge (search edge 3) 4 [1741] 1741 root268Tail17 winner = winner :=
  (root268Step17).trans root268Scan18
theorem root268Scan16 : scan edge (search edge 3) 4 [1741] 1741 root268Tail16 winner = winner :=
  (root268Step16).trans root268Scan17
theorem root268Scan15 : scan edge (search edge 3) 4 [1741] 1741 root268Tail15 winner = winner :=
  (root268Step15).trans root268Scan16
theorem root268Scan14 : scan edge (search edge 3) 4 [1741] 1741 root268Tail14 winner = winner :=
  (root268Step14).trans root268Scan15
theorem root268Scan13 : scan edge (search edge 3) 4 [1741] 1741 root268Tail13 winner = winner :=
  (root268Step13).trans root268Scan14
theorem root268Scan12 : scan edge (search edge 3) 4 [1741] 1741 root268Tail12 winner = winner :=
  (root268Step12).trans root268Scan13
theorem root268Scan11 : scan edge (search edge 3) 4 [1741] 1741 root268Tail11 winner = winner :=
  (root268Step11).trans root268Scan12
theorem root268Scan10 : scan edge (search edge 3) 4 [1741] 1741 root268Tail10 winner = winner :=
  (root268Step10).trans root268Scan11
theorem root268Scan9 : scan edge (search edge 3) 4 [1741] 1741 root268Tail9 winner = winner :=
  (root268Step9).trans root268Scan10
theorem root268Scan8 : scan edge (search edge 3) 4 [1741] 1741 root268Tail8 winner = winner :=
  (root268Step8).trans root268Scan9
theorem root268Scan7 : scan edge (search edge 3) 4 [1741] 1741 root268Tail7 winner = winner :=
  (root268Step7).trans root268Scan8
theorem root268Scan6 : scan edge (search edge 3) 4 [1741] 1741 root268Tail6 winner = winner :=
  (root268Step6).trans root268Scan7
theorem root268Scan5 : scan edge (search edge 3) 4 [1741] 1741 root268Tail5 winner = winner :=
  (root268Step5).trans root268Scan6
theorem root268Scan4 : scan edge (search edge 3) 4 [1741] 1741 root268Tail4 winner = winner :=
  (root268Step4).trans root268Scan5
theorem root268Scan3 : scan edge (search edge 3) 4 [1741] 1741 root268Tail3 winner = winner :=
  (root268Step3).trans root268Scan4
theorem root268Scan2 : scan edge (search edge 3) 4 [1741] 1741 root268Tail2 winner = winner :=
  (root268Step2).trans root268Scan3
theorem root268Scan1 : scan edge (search edge 3) 4 [1741] 1741 root268Tail1 winner = winner :=
  (root268Step1).trans root268Scan2
theorem root268Scan0 : scan edge (search edge 3) 4 [1741] 1741 root268Tail0 winner = winner :=
  (root268Step0).trans root268Scan1
set_option maxRecDepth 4096 in
theorem root268Branch : search edge 4 [1741] root268Candidates 1741 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1741] 1741 root268Tail0 winner = winner
  exact root268Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
