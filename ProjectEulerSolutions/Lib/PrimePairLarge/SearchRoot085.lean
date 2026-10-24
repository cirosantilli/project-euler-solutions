import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root85Candidates : List Nat := Data.row457
set_option maxRecDepth 8192 in
theorem root85Candidates_checked : candidates edge 5 0 457 tail86 winner.bound = root85Candidates := by
  apply candidates_eq_row edge 5 0 457 tail86 root85Candidates winner.bound tail86_ordered
  · decide
  · decide
  · funext q
    rfl
def root85Tail67 : List Nat := []
def root85Tail66 : List Nat := 25453 :: root85Tail67
def root85Tail65 : List Nat := 24919 :: root85Tail66
def root85Tail64 : List Nat := 24571 :: root85Tail65
def root85Tail63 : List Nat := 24379 :: root85Tail64
def root85Tail62 : List Nat := 23827 :: root85Tail63
def root85Tail61 : List Nat := 23131 :: root85Tail62
def root85Tail60 : List Nat := 22447 :: root85Tail61
def root85Tail59 : List Nat := 22291 :: root85Tail60
def root85Tail58 : List Nat := 21859 :: root85Tail59
def root85Tail57 : List Nat := 20887 :: root85Tail58
def root85Tail56 : List Nat := 20749 :: root85Tail57
def root85Tail55 : List Nat := 20611 :: root85Tail56
def root85Tail54 : List Nat := 20161 :: root85Tail55
def root85Tail53 : List Nat := 20149 :: root85Tail54
def root85Tail52 : List Nat := 20107 :: root85Tail53
def root85Tail51 : List Nat := 19447 :: root85Tail52
def root85Tail50 : List Nat := 19081 :: root85Tail51
def root85Tail49 : List Nat := 18679 :: root85Tail50
def root85Tail48 : List Nat := 18493 :: root85Tail49
def root85Tail47 : List Nat := 18217 :: root85Tail48
def root85Tail46 : List Nat := 17293 :: root85Tail47
def root85Tail45 : List Nat := 16843 :: root85Tail46
def root85Tail44 : List Nat := 16729 :: root85Tail45
def root85Tail43 : List Nat := 16339 :: root85Tail44
def root85Tail42 : List Nat := 15991 :: root85Tail43
def root85Tail41 : List Nat := 15889 :: root85Tail42
def root85Tail40 : List Nat := 14827 :: root85Tail41
def root85Tail39 : List Nat := 14407 :: root85Tail40
def root85Tail38 : List Nat := 14293 :: root85Tail39
def root85Tail37 : List Nat := 14281 :: root85Tail38
def root85Tail36 : List Nat := 14083 :: root85Tail37
def root85Tail35 : List Nat := 13879 :: root85Tail36
def root85Tail34 : List Nat := 13597 :: root85Tail35
def root85Tail33 : List Nat := 13177 :: root85Tail34
def root85Tail32 : List Nat := 11593 :: root85Tail33
def root85Tail31 : List Nat := 11443 :: root85Tail32
def root85Tail30 : List Nat := 11437 :: root85Tail31
def root85Tail29 : List Nat := 10303 :: root85Tail30
def root85Tail28 : List Nat := 9901 :: root85Tail29
def root85Tail27 : List Nat := 9343 :: root85Tail28
def root85Tail26 : List Nat := 9007 :: root85Tail27
def root85Tail25 : List Nat := 8923 :: root85Tail26
def root85Tail24 : List Nat := 7669 :: root85Tail25
def root85Tail23 : List Nat := 7411 :: root85Tail24
def root85Tail22 : List Nat := 7297 :: root85Tail23
def root85Tail21 : List Nat := 7039 :: root85Tail22
def root85Tail20 : List Nat := 6997 :: root85Tail21
def root85Tail19 : List Nat := 5647 :: root85Tail20
def root85Tail18 : List Nat := 5503 :: root85Tail19
def root85Tail17 : List Nat := 5281 :: root85Tail18
def root85Tail16 : List Nat := 5107 :: root85Tail17
def root85Tail15 : List Nat := 4987 :: root85Tail16
def root85Tail14 : List Nat := 4441 :: root85Tail15
def root85Tail13 : List Nat := 3931 :: root85Tail14
def root85Tail12 : List Nat := 3631 :: root85Tail13
def root85Tail11 : List Nat := 3571 :: root85Tail12
def root85Tail10 : List Nat := 2917 :: root85Tail11
def root85Tail9 : List Nat := 2707 :: root85Tail10
def root85Tail8 : List Nat := 2551 :: root85Tail9
def root85Tail7 : List Nat := 2383 :: root85Tail8
def root85Tail6 : List Nat := 2131 :: root85Tail7
def root85Tail5 : List Nat := 1597 :: root85Tail6
def root85Tail4 : List Nat := 1033 :: root85Tail5
def root85Tail3 : List Nat := 829 :: root85Tail4
def root85Tail2 : List Nat := 757 :: root85Tail3
def root85Tail1 : List Nat := 673 :: root85Tail2
def root85Tail0 : List Nat := 643 :: root85Tail1
def root85Candidates0 : List Nat := [2383, 2917, 8923, 9007]
set_option maxRecDepth 8192 in
theorem root85Candidates0_checked : candidates edge 4 457 643 root85Tail1 winner.bound = root85Candidates0 := by
  unfold candidates
  have row : edge 643 = (SortedNatTree.contains FastRows.row643) := by
    change Data.row643.contains = (SortedNatTree.contains FastRows.row643)
    exact FastRows.row643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child0 : search edge 3 [457, 643] root85Candidates0 1100 winner = winner := by decide
theorem root85Step0 : scan edge (search edge 3) 4 [457] 457 root85Tail0 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail1 winner := by
  rw [root85Tail0, scan, if_neg (by decide), root85Candidates0_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail1
    (if 3 ≤ root85Candidates0.length then search edge 3 [457, 643] root85Candidates0 1100 winner else winner) = _
  rw [if_pos (by decide), root85Child0]
def root85Candidates1 : List Nat := [13177, 20887, 23827]
set_option maxRecDepth 8192 in
theorem root85Candidates1_checked : candidates edge 4 457 673 root85Tail2 winner.bound = root85Candidates1 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child1 : search edge 3 [457, 673] root85Candidates1 1130 winner = winner := by decide
theorem root85Step1 : scan edge (search edge 3) 4 [457] 457 root85Tail1 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail2 winner := by
  rw [root85Tail1, scan, if_neg (by decide), root85Candidates1_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail2
    (if 3 ≤ root85Candidates1.length then search edge 3 [457, 673] root85Candidates1 1130 winner else winner) = _
  rw [if_pos (by decide), root85Child1]
def root85Candidates2 : List Nat := [829, 3931, 11437]
set_option maxRecDepth 8192 in
theorem root85Candidates2_checked : candidates edge 4 457 757 root85Tail3 winner.bound = root85Candidates2 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child2 : search edge 3 [457, 757] root85Candidates2 1214 winner = winner := by decide
theorem root85Step2 : scan edge (search edge 3) 4 [457] 457 root85Tail2 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail3 winner := by
  rw [root85Tail2, scan, if_neg (by decide), root85Candidates2_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail3
    (if 3 ≤ root85Candidates2.length then search edge 3 [457, 757] root85Candidates2 1214 winner else winner) = _
  rw [if_pos (by decide), root85Child2]
def root85Candidates3 : List Nat := [1033, 5107, 8923, 14827]
set_option maxRecDepth 8192 in
theorem root85Candidates3_checked : candidates edge 4 457 829 root85Tail4 winner.bound = root85Candidates3 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child3 : search edge 3 [457, 829] root85Candidates3 1286 winner = winner := by decide
theorem root85Step3 : scan edge (search edge 3) 4 [457] 457 root85Tail3 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail4 winner := by
  rw [root85Tail3, scan, if_neg (by decide), root85Candidates3_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail4
    (if 3 ≤ root85Candidates3.length then search edge 3 [457, 829] root85Candidates3 1286 winner else winner) = _
  rw [if_pos (by decide), root85Child3]
def root85Candidates4 : List Nat := [2131, 22291]
set_option maxRecDepth 8192 in
theorem root85Candidates4_checked : candidates edge 4 457 1033 root85Tail5 winner.bound = root85Candidates4 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
theorem root85Step4 : scan edge (search edge 3) 4 [457] 457 root85Tail4 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail5 winner := by
  rw [root85Tail4, scan, if_neg (by decide), root85Candidates4_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail5
    (if 3 ≤ root85Candidates4.length then search edge 3 [457, 1033] root85Candidates4 1490 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates5 : List Nat := [5503, 19447, 20149]
set_option maxRecDepth 8192 in
theorem root85Candidates5_checked : candidates edge 4 457 1597 root85Tail6 winner.bound = root85Candidates5 := by
  unfold candidates
  have row : edge 1597 = (SortedNatTree.contains FastRows.row1597) := by
    change Data.row1597.contains = (SortedNatTree.contains FastRows.row1597)
    exact FastRows.row1597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child5 : search edge 3 [457, 1597] root85Candidates5 2054 winner = winner := by decide
theorem root85Step5 : scan edge (search edge 3) 4 [457] 457 root85Tail5 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail6 winner := by
  rw [root85Tail5, scan, if_neg (by decide), root85Candidates5_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail6
    (if 3 ≤ root85Candidates5.length then search edge 3 [457, 1597] root85Candidates5 2054 winner else winner) = _
  rw [if_pos (by decide), root85Child5]
def root85Candidates6 : List Nat := [7669]
set_option maxRecDepth 8192 in
theorem root85Candidates6_checked : candidates edge 4 457 2131 root85Tail7 winner.bound = root85Candidates6 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root85Step6 : scan edge (search edge 3) 4 [457] 457 root85Tail6 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail7 winner := by
  rw [root85Tail6, scan, if_neg (by decide), root85Candidates6_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail7
    (if 3 ≤ root85Candidates6.length then search edge 3 [457, 2131] root85Candidates6 2588 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root85Candidates7_checked : candidates edge 4 457 2383 root85Tail8 winner.bound = root85Candidates7 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root85Step7 : scan edge (search edge 3) 4 [457] 457 root85Tail7 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail8 winner := by
  rw [root85Tail7, scan, if_neg (by decide), root85Candidates7_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail8
    (if 3 ≤ root85Candidates7.length then search edge 3 [457, 2383] root85Candidates7 2840 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates8 : List Nat := [22447]
set_option maxRecDepth 8192 in
theorem root85Candidates8_checked : candidates edge 4 457 2551 root85Tail9 winner.bound = root85Candidates8 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root85Step8 : scan edge (search edge 3) 4 [457] 457 root85Tail8 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail9 winner := by
  rw [root85Tail8, scan, if_neg (by decide), root85Candidates8_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail9
    (if 3 ≤ root85Candidates8.length then search edge 3 [457, 2551] root85Candidates8 3008 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates9 : List Nat := [11443, 14281, 18493]
set_option maxRecDepth 8192 in
theorem root85Candidates9_checked : candidates edge 4 457 2707 root85Tail10 winner.bound = root85Candidates9 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child9 : search edge 3 [457, 2707] root85Candidates9 3164 winner = winner := by decide
theorem root85Step9 : scan edge (search edge 3) 4 [457] 457 root85Tail9 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail10 winner := by
  rw [root85Tail9, scan, if_neg (by decide), root85Candidates9_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail10
    (if 3 ≤ root85Candidates9.length then search edge 3 [457, 2707] root85Candidates9 3164 winner else winner) = _
  rw [if_pos (by decide), root85Child9]
def root85Candidates10 : List Nat := [3631, 11443, 19081, 20749]
set_option maxRecDepth 8192 in
theorem root85Candidates10_checked : candidates edge 4 457 2917 root85Tail11 winner.bound = root85Candidates10 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root85Child10 : search edge 3 [457, 2917] root85Candidates10 3374 winner = winner := by decide
theorem root85Step10 : scan edge (search edge 3) 4 [457] 457 root85Tail10 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail11 winner := by
  rw [root85Tail10, scan, if_neg (by decide), root85Candidates10_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail11
    (if 3 ≤ root85Candidates10.length then search edge 3 [457, 2917] root85Candidates10 3374 winner else winner) = _
  rw [if_pos (by decide), root85Child10]
def root85Candidates11 : List Nat := [20107]
set_option maxRecDepth 8192 in
theorem root85Candidates11_checked : candidates edge 4 457 3571 root85Tail12 winner.bound = root85Candidates11 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root85Step11 : scan edge (search edge 3) 4 [457] 457 root85Tail11 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail12 winner := by
  rw [root85Tail11, scan, if_neg (by decide), root85Candidates11_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail12
    (if 3 ≤ root85Candidates11.length then search edge 3 [457, 3571] root85Candidates11 4028 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates12 : List Nat := [8923, 14083]
set_option maxRecDepth 8192 in
theorem root85Candidates12_checked : candidates edge 4 457 3631 root85Tail13 winner.bound = root85Candidates12 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root85Step12 : scan edge (search edge 3) 4 [457] 457 root85Tail12 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail13 winner := by
  rw [root85Tail12, scan, if_neg (by decide), root85Candidates12_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail13
    (if 3 ≤ root85Candidates12.length then search edge 3 [457, 3631] root85Candidates12 4088 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates13 : List Nat := [20149]
set_option maxRecDepth 8192 in
theorem root85Candidates13_checked : candidates edge 4 457 3931 root85Tail14 winner.bound = root85Candidates13 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root85Step13 : scan edge (search edge 3) 4 [457] 457 root85Tail13 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail14 winner := by
  rw [root85Tail13, scan, if_neg (by decide), root85Candidates13_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail14
    (if 3 ≤ root85Candidates13.length then search edge 3 [457, 3931] root85Candidates13 4388 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root85Candidates14_checked : candidates edge 4 457 4441 root85Tail15 winner.bound = root85Candidates14 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root85Step14 : scan edge (search edge 3) 4 [457] 457 root85Tail14 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail15 winner := by
  rw [root85Tail14, scan, if_neg (by decide), root85Candidates14_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail15
    (if 3 ≤ root85Candidates14.length then search edge 3 [457, 4441] root85Candidates14 4898 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates15 : List Nat := [15991, 18217]
set_option maxRecDepth 8192 in
theorem root85Candidates15_checked : candidates edge 4 457 4987 root85Tail16 winner.bound = root85Candidates15 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root85Step15 : scan edge (search edge 3) 4 [457] 457 root85Tail15 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail16 winner := by
  rw [root85Tail15, scan, if_neg (by decide), root85Candidates15_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail16
    (if 3 ≤ root85Candidates15.length then search edge 3 [457, 4987] root85Candidates15 5444 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates16 : List Nat := [9901]
set_option maxRecDepth 8192 in
theorem root85Candidates16_checked : candidates edge 4 457 5107 root85Tail17 winner.bound = root85Candidates16 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root85Step16 : scan edge (search edge 3) 4 [457] 457 root85Tail16 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail17 winner := by
  rw [root85Tail16, scan, if_neg (by decide), root85Candidates16_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail17
    (if 3 ≤ root85Candidates16.length then search edge 3 [457, 5107] root85Candidates16 5564 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root85Candidates17_checked : candidates edge 4 457 5281 root85Tail18 winner.bound = root85Candidates17 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root85Step17 : scan edge (search edge 3) 4 [457] 457 root85Tail17 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail18 winner := by
  rw [root85Tail17, scan, if_neg (by decide), root85Candidates17_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail18
    (if 3 ≤ root85Candidates17.length then search edge 3 [457, 5281] root85Candidates17 5738 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root85Candidates18_checked : candidates edge 4 457 5503 root85Tail19 winner.bound = root85Candidates18 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root85Step18 : scan edge (search edge 3) 4 [457] 457 root85Tail18 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail19 winner := by
  rw [root85Tail18, scan, if_neg (by decide), root85Candidates18_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail19
    (if 3 ≤ root85Candidates18.length then search edge 3 [457, 5503] root85Candidates18 5960 winner else winner) = _
  rw [if_neg (by decide)]
def root85Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root85Candidates19_checked : candidates edge 4 457 5647 root85Tail20 winner.bound = root85Candidates19 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root85Step19 : scan edge (search edge 3) 4 [457] 457 root85Tail19 winner =
    scan edge (search edge 3) 4 [457] 457 root85Tail20 winner := by
  rw [root85Tail19, scan, if_neg (by decide), root85Candidates19_checked]
  change scan edge (search edge 3) 4 [457] 457 root85Tail20
    (if 3 ≤ root85Candidates19.length then search edge 3 [457, 5647] root85Candidates19 6104 winner else winner) = _
  rw [if_neg (by decide)]
theorem root85Scan20 : scan edge (search edge 3) 4 [457] 457 root85Tail20 winner = winner := by rw [root85Tail20, scan, if_pos (by decide)]
theorem root85Scan19 : scan edge (search edge 3) 4 [457] 457 root85Tail19 winner = winner :=
  (root85Step19).trans root85Scan20
theorem root85Scan18 : scan edge (search edge 3) 4 [457] 457 root85Tail18 winner = winner :=
  (root85Step18).trans root85Scan19
theorem root85Scan17 : scan edge (search edge 3) 4 [457] 457 root85Tail17 winner = winner :=
  (root85Step17).trans root85Scan18
theorem root85Scan16 : scan edge (search edge 3) 4 [457] 457 root85Tail16 winner = winner :=
  (root85Step16).trans root85Scan17
theorem root85Scan15 : scan edge (search edge 3) 4 [457] 457 root85Tail15 winner = winner :=
  (root85Step15).trans root85Scan16
theorem root85Scan14 : scan edge (search edge 3) 4 [457] 457 root85Tail14 winner = winner :=
  (root85Step14).trans root85Scan15
theorem root85Scan13 : scan edge (search edge 3) 4 [457] 457 root85Tail13 winner = winner :=
  (root85Step13).trans root85Scan14
theorem root85Scan12 : scan edge (search edge 3) 4 [457] 457 root85Tail12 winner = winner :=
  (root85Step12).trans root85Scan13
theorem root85Scan11 : scan edge (search edge 3) 4 [457] 457 root85Tail11 winner = winner :=
  (root85Step11).trans root85Scan12
theorem root85Scan10 : scan edge (search edge 3) 4 [457] 457 root85Tail10 winner = winner :=
  (root85Step10).trans root85Scan11
theorem root85Scan9 : scan edge (search edge 3) 4 [457] 457 root85Tail9 winner = winner :=
  (root85Step9).trans root85Scan10
theorem root85Scan8 : scan edge (search edge 3) 4 [457] 457 root85Tail8 winner = winner :=
  (root85Step8).trans root85Scan9
theorem root85Scan7 : scan edge (search edge 3) 4 [457] 457 root85Tail7 winner = winner :=
  (root85Step7).trans root85Scan8
theorem root85Scan6 : scan edge (search edge 3) 4 [457] 457 root85Tail6 winner = winner :=
  (root85Step6).trans root85Scan7
theorem root85Scan5 : scan edge (search edge 3) 4 [457] 457 root85Tail5 winner = winner :=
  (root85Step5).trans root85Scan6
theorem root85Scan4 : scan edge (search edge 3) 4 [457] 457 root85Tail4 winner = winner :=
  (root85Step4).trans root85Scan5
theorem root85Scan3 : scan edge (search edge 3) 4 [457] 457 root85Tail3 winner = winner :=
  (root85Step3).trans root85Scan4
theorem root85Scan2 : scan edge (search edge 3) 4 [457] 457 root85Tail2 winner = winner :=
  (root85Step2).trans root85Scan3
theorem root85Scan1 : scan edge (search edge 3) 4 [457] 457 root85Tail1 winner = winner :=
  (root85Step1).trans root85Scan2
theorem root85Scan0 : scan edge (search edge 3) 4 [457] 457 root85Tail0 winner = winner :=
  (root85Step0).trans root85Scan1
set_option maxRecDepth 4096 in
theorem root85Branch : search edge 4 [457] root85Candidates 457 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [457] 457 root85Tail0 winner = winner
  exact root85Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
