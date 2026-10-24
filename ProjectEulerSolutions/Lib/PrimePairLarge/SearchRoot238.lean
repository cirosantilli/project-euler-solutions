import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root238Candidates : List Nat := Data.row1523
set_option maxRecDepth 8192 in
theorem root238Candidates_checked : candidates edge 5 0 1523 tail239 winner.bound = root238Candidates := by
  apply candidates_eq_row edge 5 0 1523 tail239 root238Candidates winner.bound tail239_ordered
  · decide
  · decide
  · funext q
    rfl
def root238Tail52 : List Nat := []
def root238Tail51 : List Nat := 24413 :: root238Tail52
def root238Tail50 : List Nat := 23669 :: root238Tail51
def root238Tail49 : List Nat := 23357 :: root238Tail50
def root238Tail48 : List Nat := 22571 :: root238Tail49
def root238Tail47 : List Nat := 22409 :: root238Tail48
def root238Tail46 : List Nat := 22283 :: root238Tail47
def root238Tail45 : List Nat := 22067 :: root238Tail46
def root238Tail44 : List Nat := 22013 :: root238Tail45
def root238Tail43 : List Nat := 21737 :: root238Tail44
def root238Tail42 : List Nat := 21317 :: root238Tail43
def root238Tail41 : List Nat := 19763 :: root238Tail42
def root238Tail40 : List Nat := 19577 :: root238Tail41
def root238Tail39 : List Nat := 19163 :: root238Tail40
def root238Tail38 : List Nat := 16811 :: root238Tail39
def root238Tail37 : List Nat := 16067 :: root238Tail38
def root238Tail36 : List Nat := 15797 :: root238Tail37
def root238Tail35 : List Nat := 13331 :: root238Tail36
def root238Tail34 : List Nat := 11807 :: root238Tail35
def root238Tail33 : List Nat := 11399 :: root238Tail34
def root238Tail32 : List Nat := 9929 :: root238Tail33
def root238Tail31 : List Nat := 9677 :: root238Tail32
def root238Tail30 : List Nat := 9551 :: root238Tail31
def root238Tail29 : List Nat := 9377 :: root238Tail30
def root238Tail28 : List Nat := 7907 :: root238Tail29
def root238Tail27 : List Nat := 7583 :: root238Tail28
def root238Tail26 : List Nat := 7043 :: root238Tail27
def root238Tail25 : List Nat := 6827 :: root238Tail26
def root238Tail24 : List Nat := 6311 :: root238Tail25
def root238Tail23 : List Nat := 5813 :: root238Tail24
def root238Tail22 : List Nat := 5573 :: root238Tail23
def root238Tail21 : List Nat := 5483 :: root238Tail22
def root238Tail20 : List Nat := 5309 :: root238Tail21
def root238Tail19 : List Nat := 5303 :: root238Tail20
def root238Tail18 : List Nat := 5279 :: root238Tail19
def root238Tail17 : List Nat := 4679 :: root238Tail18
def root238Tail16 : List Nat := 4517 :: root238Tail17
def root238Tail15 : List Nat := 4253 :: root238Tail16
def root238Tail14 : List Nat := 4139 :: root238Tail15
def root238Tail13 : List Nat := 4013 :: root238Tail14
def root238Tail12 : List Nat := 3929 :: root238Tail13
def root238Tail11 : List Nat := 3833 :: root238Tail12
def root238Tail10 : List Nat := 3761 :: root238Tail11
def root238Tail9 : List Nat := 3461 :: root238Tail10
def root238Tail8 : List Nat := 3299 :: root238Tail9
def root238Tail7 : List Nat := 3203 :: root238Tail8
def root238Tail6 : List Nat := 2999 :: root238Tail7
def root238Tail5 : List Nat := 2957 :: root238Tail6
def root238Tail4 : List Nat := 2909 :: root238Tail5
def root238Tail3 : List Nat := 2579 :: root238Tail4
def root238Tail2 : List Nat := 2447 :: root238Tail3
def root238Tail1 : List Nat := 2111 :: root238Tail2
def root238Tail0 : List Nat := 1913 :: root238Tail1
def root238Candidates0 : List Nat := [3203, 5483, 7583]
set_option maxRecDepth 8192 in
theorem root238Candidates0_checked : candidates edge 4 1523 1913 root238Tail1 winner.bound = root238Candidates0 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root238Child0 : search edge 3 [1523, 1913] root238Candidates0 3436 winner = winner := by decide
theorem root238Step0 : scan edge (search edge 3) 4 [1523] 1523 root238Tail0 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail1 winner := by
  rw [root238Tail0, scan, if_neg (by decide), root238Candidates0_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail1
    (if 3 ≤ root238Candidates0.length then search edge 3 [1523, 1913] root238Candidates0 3436 winner else winner) = _
  rw [if_pos (by decide), root238Child0]
def root238Candidates1 : List Nat := [2999, 11399, 22283]
set_option maxRecDepth 8192 in
theorem root238Candidates1_checked : candidates edge 4 1523 2111 root238Tail2 winner.bound = root238Candidates1 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root238Child1 : search edge 3 [1523, 2111] root238Candidates1 3634 winner = winner := by decide
theorem root238Step1 : scan edge (search edge 3) 4 [1523] 1523 root238Tail1 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail2 winner := by
  rw [root238Tail1, scan, if_neg (by decide), root238Candidates1_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail2
    (if 3 ≤ root238Candidates1.length then search edge 3 [1523, 2111] root238Candidates1 3634 winner else winner) = _
  rw [if_pos (by decide), root238Child1]
def root238Candidates2 : List Nat := [5813, 7907]
set_option maxRecDepth 8192 in
theorem root238Candidates2_checked : candidates edge 4 1523 2447 root238Tail3 winner.bound = root238Candidates2 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root238Step2 : scan edge (search edge 3) 4 [1523] 1523 root238Tail2 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail3 winner := by
  rw [root238Tail2, scan, if_neg (by decide), root238Candidates2_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail3
    (if 3 ≤ root238Candidates2.length then search edge 3 [1523, 2447] root238Candidates2 3970 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates3_checked : candidates edge 4 1523 2579 root238Tail4 winner.bound = root238Candidates3 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root238Step3 : scan edge (search edge 3) 4 [1523] 1523 root238Tail3 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail4 winner := by
  rw [root238Tail3, scan, if_neg (by decide), root238Candidates3_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail4
    (if 3 ≤ root238Candidates3.length then search edge 3 [1523, 2579] root238Candidates3 4102 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates4 : List Nat := [4253, 5813, 15797]
set_option maxRecDepth 8192 in
theorem root238Candidates4_checked : candidates edge 4 1523 2909 root238Tail5 winner.bound = root238Candidates4 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root238Child4 : search edge 3 [1523, 2909] root238Candidates4 4432 winner = winner := by decide
theorem root238Step4 : scan edge (search edge 3) 4 [1523] 1523 root238Tail4 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail5 winner := by
  rw [root238Tail4, scan, if_neg (by decide), root238Candidates4_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail5
    (if 3 ≤ root238Candidates4.length then search edge 3 [1523, 2909] root238Candidates4 4432 winner else winner) = _
  rw [if_pos (by decide), root238Child4]
def root238Candidates5 : List Nat := [9377, 9551]
set_option maxRecDepth 8192 in
theorem root238Candidates5_checked : candidates edge 4 1523 2957 root238Tail6 winner.bound = root238Candidates5 := by
  unfold candidates
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row]
  decide
theorem root238Step5 : scan edge (search edge 3) 4 [1523] 1523 root238Tail5 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail6 winner := by
  rw [root238Tail5, scan, if_neg (by decide), root238Candidates5_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail6
    (if 3 ≤ root238Candidates5.length then search edge 3 [1523, 2957] root238Candidates5 4480 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates6 : List Nat := [6311, 7907]
set_option maxRecDepth 8192 in
theorem root238Candidates6_checked : candidates edge 4 1523 2999 root238Tail7 winner.bound = root238Candidates6 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root238Step6 : scan edge (search edge 3) 4 [1523] 1523 root238Tail6 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail7 winner := by
  rw [root238Tail6, scan, if_neg (by decide), root238Candidates6_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail7
    (if 3 ≤ root238Candidates6.length then search edge 3 [1523, 2999] root238Candidates6 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates7 : List Nat := [5279, 6827, 19163]
set_option maxRecDepth 8192 in
theorem root238Candidates7_checked : candidates edge 4 1523 3203 root238Tail8 winner.bound = root238Candidates7 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root238Child7 : search edge 3 [1523, 3203] root238Candidates7 4726 winner = winner := by decide
theorem root238Step7 : scan edge (search edge 3) 4 [1523] 1523 root238Tail7 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail8 winner := by
  rw [root238Tail7, scan, if_neg (by decide), root238Candidates7_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail8
    (if 3 ≤ root238Candidates7.length then search edge 3 [1523, 3203] root238Candidates7 4726 winner else winner) = _
  rw [if_pos (by decide), root238Child7]
def root238Candidates8 : List Nat := [7043]
set_option maxRecDepth 8192 in
theorem root238Candidates8_checked : candidates edge 4 1523 3299 root238Tail9 winner.bound = root238Candidates8 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root238Step8 : scan edge (search edge 3) 4 [1523] 1523 root238Tail8 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail9 winner := by
  rw [root238Tail8, scan, if_neg (by decide), root238Candidates8_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail9
    (if 3 ≤ root238Candidates8.length then search edge 3 [1523, 3299] root238Candidates8 4822 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates9_checked : candidates edge 4 1523 3461 root238Tail10 winner.bound = root238Candidates9 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root238Step9 : scan edge (search edge 3) 4 [1523] 1523 root238Tail9 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail10 winner := by
  rw [root238Tail9, scan, if_neg (by decide), root238Candidates9_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail10
    (if 3 ≤ root238Candidates9.length then search edge 3 [1523, 3461] root238Candidates9 4984 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates10_checked : candidates edge 4 1523 3761 root238Tail11 winner.bound = root238Candidates10 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root238Step10 : scan edge (search edge 3) 4 [1523] 1523 root238Tail10 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail11 winner := by
  rw [root238Tail10, scan, if_neg (by decide), root238Candidates10_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail11
    (if 3 ≤ root238Candidates10.length then search edge 3 [1523, 3761] root238Candidates10 5284 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates11 : List Nat := [4253, 5813, 6827]
set_option maxRecDepth 8192 in
theorem root238Candidates11_checked : candidates edge 4 1523 3833 root238Tail12 winner.bound = root238Candidates11 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root238Child11 : search edge 3 [1523, 3833] root238Candidates11 5356 winner = winner := by decide
theorem root238Step11 : scan edge (search edge 3) 4 [1523] 1523 root238Tail11 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail12 winner := by
  rw [root238Tail11, scan, if_neg (by decide), root238Candidates11_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail12
    (if 3 ≤ root238Candidates11.length then search edge 3 [1523, 3833] root238Candidates11 5356 winner else winner) = _
  rw [if_pos (by decide), root238Child11]
def root238Candidates12 : List Nat := [7043]
set_option maxRecDepth 8192 in
theorem root238Candidates12_checked : candidates edge 4 1523 3929 root238Tail13 winner.bound = root238Candidates12 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root238Step12 : scan edge (search edge 3) 4 [1523] 1523 root238Tail12 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail13 winner := by
  rw [root238Tail12, scan, if_neg (by decide), root238Candidates12_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail13
    (if 3 ≤ root238Candidates12.length then search edge 3 [1523, 3929] root238Candidates12 5452 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates13 : List Nat := [5573]
set_option maxRecDepth 8192 in
theorem root238Candidates13_checked : candidates edge 4 1523 4013 root238Tail14 winner.bound = root238Candidates13 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root238Step13 : scan edge (search edge 3) 4 [1523] 1523 root238Tail13 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail14 winner := by
  rw [root238Tail13, scan, if_neg (by decide), root238Candidates13_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail14
    (if 3 ≤ root238Candidates13.length then search edge 3 [1523, 4013] root238Candidates13 5536 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates14_checked : candidates edge 4 1523 4139 root238Tail15 winner.bound = root238Candidates14 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root238Step14 : scan edge (search edge 3) 4 [1523] 1523 root238Tail14 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail15 winner := by
  rw [root238Tail14, scan, if_neg (by decide), root238Candidates14_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail15
    (if 3 ≤ root238Candidates14.length then search edge 3 [1523, 4139] root238Candidates14 5662 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates15_checked : candidates edge 4 1523 4253 root238Tail16 winner.bound = root238Candidates15 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root238Step15 : scan edge (search edge 3) 4 [1523] 1523 root238Tail15 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail16 winner := by
  rw [root238Tail15, scan, if_neg (by decide), root238Candidates15_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail16
    (if 3 ≤ root238Candidates15.length then search edge 3 [1523, 4253] root238Candidates15 5776 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates16 : List Nat := [5279, 9377]
set_option maxRecDepth 8192 in
theorem root238Candidates16_checked : candidates edge 4 1523 4517 root238Tail17 winner.bound = root238Candidates16 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root238Step16 : scan edge (search edge 3) 4 [1523] 1523 root238Tail16 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail17 winner := by
  rw [root238Tail16, scan, if_neg (by decide), root238Candidates16_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail17
    (if 3 ≤ root238Candidates16.length then search edge 3 [1523, 4517] root238Candidates16 6040 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates17_checked : candidates edge 4 1523 4679 root238Tail18 winner.bound = root238Candidates17 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root238Step17 : scan edge (search edge 3) 4 [1523] 1523 root238Tail17 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail18 winner := by
  rw [root238Tail17, scan, if_neg (by decide), root238Candidates17_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail18
    (if 3 ≤ root238Candidates17.length then search edge 3 [1523, 4679] root238Candidates17 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates18 : List Nat := [6827]
set_option maxRecDepth 8192 in
theorem root238Candidates18_checked : candidates edge 4 1523 5279 root238Tail19 winner.bound = root238Candidates18 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root238Step18 : scan edge (search edge 3) 4 [1523] 1523 root238Tail18 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail19 winner := by
  rw [root238Tail18, scan, if_neg (by decide), root238Candidates18_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail19
    (if 3 ≤ root238Candidates18.length then search edge 3 [1523, 5279] root238Candidates18 6802 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates19 : List Nat := [16067]
set_option maxRecDepth 8192 in
theorem root238Candidates19_checked : candidates edge 4 1523 5303 root238Tail20 winner.bound = root238Candidates19 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root238Step19 : scan edge (search edge 3) 4 [1523] 1523 root238Tail19 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail20 winner := by
  rw [root238Tail19, scan, if_neg (by decide), root238Candidates19_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail20
    (if 3 ≤ root238Candidates19.length then search edge 3 [1523, 5303] root238Candidates19 6826 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates20_checked : candidates edge 4 1523 5309 root238Tail21 winner.bound = root238Candidates20 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root238Step20 : scan edge (search edge 3) 4 [1523] 1523 root238Tail20 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail21 winner := by
  rw [root238Tail20, scan, if_neg (by decide), root238Candidates20_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail21
    (if 3 ≤ root238Candidates20.length then search edge 3 [1523, 5309] root238Candidates20 6832 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates21 : List Nat := [15797]
set_option maxRecDepth 8192 in
theorem root238Candidates21_checked : candidates edge 4 1523 5483 root238Tail22 winner.bound = root238Candidates21 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root238Step21 : scan edge (search edge 3) 4 [1523] 1523 root238Tail21 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail22 winner := by
  rw [root238Tail21, scan, if_neg (by decide), root238Candidates21_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail22
    (if 3 ≤ root238Candidates21.length then search edge 3 [1523, 5483] root238Candidates21 7006 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root238Candidates22_checked : candidates edge 4 1523 5573 root238Tail23 winner.bound = root238Candidates22 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root238Step22 : scan edge (search edge 3) 4 [1523] 1523 root238Tail22 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail23 winner := by
  rw [root238Tail22, scan, if_neg (by decide), root238Candidates22_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail23
    (if 3 ≤ root238Candidates22.length then search edge 3 [1523, 5573] root238Candidates22 7096 winner else winner) = _
  rw [if_neg (by decide)]
def root238Candidates23 : List Nat := [7907]
set_option maxRecDepth 8192 in
theorem root238Candidates23_checked : candidates edge 4 1523 5813 root238Tail24 winner.bound = root238Candidates23 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root238Step23 : scan edge (search edge 3) 4 [1523] 1523 root238Tail23 winner =
    scan edge (search edge 3) 4 [1523] 1523 root238Tail24 winner := by
  rw [root238Tail23, scan, if_neg (by decide), root238Candidates23_checked]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail24
    (if 3 ≤ root238Candidates23.length then search edge 3 [1523, 5813] root238Candidates23 7336 winner else winner) = _
  rw [if_neg (by decide)]
theorem root238Scan24 : scan edge (search edge 3) 4 [1523] 1523 root238Tail24 winner = winner := by rw [root238Tail24, scan, if_pos (by decide)]
theorem root238Scan23 : scan edge (search edge 3) 4 [1523] 1523 root238Tail23 winner = winner :=
  (root238Step23).trans root238Scan24
theorem root238Scan22 : scan edge (search edge 3) 4 [1523] 1523 root238Tail22 winner = winner :=
  (root238Step22).trans root238Scan23
theorem root238Scan21 : scan edge (search edge 3) 4 [1523] 1523 root238Tail21 winner = winner :=
  (root238Step21).trans root238Scan22
theorem root238Scan20 : scan edge (search edge 3) 4 [1523] 1523 root238Tail20 winner = winner :=
  (root238Step20).trans root238Scan21
theorem root238Scan19 : scan edge (search edge 3) 4 [1523] 1523 root238Tail19 winner = winner :=
  (root238Step19).trans root238Scan20
theorem root238Scan18 : scan edge (search edge 3) 4 [1523] 1523 root238Tail18 winner = winner :=
  (root238Step18).trans root238Scan19
theorem root238Scan17 : scan edge (search edge 3) 4 [1523] 1523 root238Tail17 winner = winner :=
  (root238Step17).trans root238Scan18
theorem root238Scan16 : scan edge (search edge 3) 4 [1523] 1523 root238Tail16 winner = winner :=
  (root238Step16).trans root238Scan17
theorem root238Scan15 : scan edge (search edge 3) 4 [1523] 1523 root238Tail15 winner = winner :=
  (root238Step15).trans root238Scan16
theorem root238Scan14 : scan edge (search edge 3) 4 [1523] 1523 root238Tail14 winner = winner :=
  (root238Step14).trans root238Scan15
theorem root238Scan13 : scan edge (search edge 3) 4 [1523] 1523 root238Tail13 winner = winner :=
  (root238Step13).trans root238Scan14
theorem root238Scan12 : scan edge (search edge 3) 4 [1523] 1523 root238Tail12 winner = winner :=
  (root238Step12).trans root238Scan13
theorem root238Scan11 : scan edge (search edge 3) 4 [1523] 1523 root238Tail11 winner = winner :=
  (root238Step11).trans root238Scan12
theorem root238Scan10 : scan edge (search edge 3) 4 [1523] 1523 root238Tail10 winner = winner :=
  (root238Step10).trans root238Scan11
theorem root238Scan9 : scan edge (search edge 3) 4 [1523] 1523 root238Tail9 winner = winner :=
  (root238Step9).trans root238Scan10
theorem root238Scan8 : scan edge (search edge 3) 4 [1523] 1523 root238Tail8 winner = winner :=
  (root238Step8).trans root238Scan9
theorem root238Scan7 : scan edge (search edge 3) 4 [1523] 1523 root238Tail7 winner = winner :=
  (root238Step7).trans root238Scan8
theorem root238Scan6 : scan edge (search edge 3) 4 [1523] 1523 root238Tail6 winner = winner :=
  (root238Step6).trans root238Scan7
theorem root238Scan5 : scan edge (search edge 3) 4 [1523] 1523 root238Tail5 winner = winner :=
  (root238Step5).trans root238Scan6
theorem root238Scan4 : scan edge (search edge 3) 4 [1523] 1523 root238Tail4 winner = winner :=
  (root238Step4).trans root238Scan5
theorem root238Scan3 : scan edge (search edge 3) 4 [1523] 1523 root238Tail3 winner = winner :=
  (root238Step3).trans root238Scan4
theorem root238Scan2 : scan edge (search edge 3) 4 [1523] 1523 root238Tail2 winner = winner :=
  (root238Step2).trans root238Scan3
theorem root238Scan1 : scan edge (search edge 3) 4 [1523] 1523 root238Tail1 winner = winner :=
  (root238Step1).trans root238Scan2
theorem root238Scan0 : scan edge (search edge 3) 4 [1523] 1523 root238Tail0 winner = winner :=
  (root238Step0).trans root238Scan1
set_option maxRecDepth 4096 in
theorem root238Branch : search edge 4 [1523] root238Candidates 1523 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1523] 1523 root238Tail0 winner = winner
  exact root238Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
