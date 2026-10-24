import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot018Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root18Candidates0 : List Nat := [1933, 9859, 19381, 23497]
set_option maxRecDepth 8192 in
theorem root18Candidates0_checked : candidates edge 4 73 277 root18Tail1 winner.bound = root18Candidates0 := by
  unfold candidates
  have row : edge 277 = (SortedNatTree.contains FastRows.row277) := by
    change Data.row277.contains = (SortedNatTree.contains FastRows.row277)
    exact FastRows.row277_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child0 : search edge 3 [73, 277] root18Candidates0 350 winner = winner := by decide
theorem root18Step0 : scan edge (search edge 3) 4 [73] 73 root18Tail0 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail1 winner := by
  rw [root18Tail0, scan, if_neg (by decide), root18Candidates0_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail1
    (if 3 ≤ root18Candidates0.length then search edge 3 [73, 277] root18Candidates0 350 winner else winner) = _
  rw [if_pos (by decide), root18Child0]
def root18Candidates1 : List Nat := [643, 823, 4549, 5443, 18757, 19597, 22669]
set_option maxRecDepth 8192 in
theorem root18Candidates1_checked : candidates edge 4 73 547 root18Tail2 winner.bound = root18Candidates1 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child1 : search edge 3 [73, 547] root18Candidates1 620 winner = winner := by decide
theorem root18Step1 : scan edge (search edge 3) 4 [73] 73 root18Tail1 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail2 winner := by
  rw [root18Tail1, scan, if_neg (by decide), root18Candidates1_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail2
    (if 3 ≤ root18Candidates1.length then search edge 3 [73, 547] root18Candidates1 620 winner else winner) = _
  rw [if_pos (by decide), root18Child1]
def root18Candidates2 : List Nat := [1933, 4759, 20359]
set_option maxRecDepth 8192 in
theorem root18Candidates2_checked : candidates edge 4 73 571 root18Tail3 winner.bound = root18Candidates2 := by
  unfold candidates
  have row : edge 571 = (SortedNatTree.contains FastRows.row571) := by
    change Data.row571.contains = (SortedNatTree.contains FastRows.row571)
    exact FastRows.row571_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child2 : search edge 3 [73, 571] root18Candidates2 644 winner = winner := by decide
theorem root18Step2 : scan edge (search edge 3) 4 [73] 73 root18Tail2 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail3 winner := by
  rw [root18Tail2, scan, if_neg (by decide), root18Candidates2_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail3
    (if 3 ≤ root18Candidates2.length then search edge 3 [73, 571] root18Candidates2 644 winner else winner) = _
  rw [if_pos (by decide), root18Child2]
def root18Candidates3 : List Nat := [2467, 5107, 6121, 18523, 25243]
set_option maxRecDepth 8192 in
theorem root18Candidates3_checked : candidates edge 4 73 607 root18Tail4 winner.bound = root18Candidates3 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child3 : search edge 3 [73, 607] root18Candidates3 680 winner = winner := by decide
theorem root18Step3 : scan edge (search edge 3) 4 [73] 73 root18Tail3 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail4 winner := by
  rw [root18Tail3, scan, if_neg (by decide), root18Candidates3_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail4
    (if 3 ≤ root18Candidates3.length then search edge 3 [73, 607] root18Candidates3 680 winner else winner) = _
  rw [if_pos (by decide), root18Child3]
def root18Candidates4 : List Nat := [9397, 15313, 15739, 18757, 21529, 21757, 24133, 25243]
set_option maxRecDepth 8192 in
theorem root18Candidates4_checked : candidates edge 4 73 643 root18Tail5 winner.bound = root18Candidates4 := by
  unfold candidates
  have row : edge 643 = (SortedNatTree.contains FastRows.row643) := by
    change Data.row643.contains = (SortedNatTree.contains FastRows.row643)
    exact FastRows.row643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child4 : search edge 3 [73, 643] root18Candidates4 716 winner = winner := by decide
theorem root18Step4 : scan edge (search edge 3) 4 [73] 73 root18Tail4 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail5 winner := by
  rw [root18Tail4, scan, if_neg (by decide), root18Candidates4_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail5
    (if 3 ≤ root18Candidates4.length then search edge 3 [73, 643] root18Candidates4 716 winner else winner) = _
  rw [if_pos (by decide), root18Child4]
def root18Candidates5 : List Nat := [4549, 10651, 11497]
set_option maxRecDepth 8192 in
theorem root18Candidates5_checked : candidates edge 4 73 757 root18Tail6 winner.bound = root18Candidates5 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child5 : search edge 3 [73, 757] root18Candidates5 830 winner = winner := by decide
theorem root18Step5 : scan edge (search edge 3) 4 [73] 73 root18Tail5 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail6 winner := by
  rw [root18Tail5, scan, if_neg (by decide), root18Candidates5_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail6
    (if 3 ≤ root18Candidates5.length then search edge 3 [73, 757] root18Candidates5 830 winner else winner) = _
  rw [if_pos (by decide), root18Child5]
def root18Candidates6 : List Nat := [1933]
set_option maxRecDepth 8192 in
theorem root18Candidates6_checked : candidates edge 4 73 823 root18Tail7 winner.bound = root18Candidates6 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
theorem root18Step6 : scan edge (search edge 3) 4 [73] 73 root18Tail6 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail7 winner := by
  rw [root18Tail6, scan, if_neg (by decide), root18Candidates6_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail7
    (if 3 ≤ root18Candidates6.length then search edge 3 [73, 823] root18Candidates6 896 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates7 : List Nat := [3823, 7591, 9601, 9859, 16567, 23497]
set_option maxRecDepth 8192 in
theorem root18Candidates7_checked : candidates edge 4 73 1117 root18Tail8 winner.bound = root18Candidates7 := by
  unfold candidates
  have row : edge 1117 = (SortedNatTree.contains FastRows.row1117) := by
    change Data.row1117.contains = (SortedNatTree.contains FastRows.row1117)
    exact FastRows.row1117_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child7 : search edge 3 [73, 1117] root18Candidates7 1190 winner = winner := by decide
theorem root18Step7 : scan edge (search edge 3) 4 [73] 73 root18Tail7 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail8 winner := by
  rw [root18Tail7, scan, if_neg (by decide), root18Candidates7_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail8
    (if 3 ≤ root18Candidates7.length then search edge 3 [73, 1117] root18Candidates7 1190 winner else winner) = _
  rw [if_pos (by decide), root18Child7]
def root18Candidates8 : List Nat := [18523]
set_option maxRecDepth 8192 in
theorem root18Candidates8_checked : candidates edge 4 73 1447 root18Tail9 winner.bound = root18Candidates8 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
theorem root18Step8 : scan edge (search edge 3) 4 [73] 73 root18Tail8 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail9 winner := by
  rw [root18Tail8, scan, if_neg (by decide), root18Candidates8_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail9
    (if 3 ≤ root18Candidates8.length then search edge 3 [73, 1447] root18Candidates8 1520 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates9 : List Nat := [2617, 3823, 9631, 9829, 18523]
set_option maxRecDepth 8192 in
theorem root18Candidates9_checked : candidates edge 4 73 1597 root18Tail10 winner.bound = root18Candidates9 := by
  unfold candidates
  have row : edge 1597 = (SortedNatTree.contains FastRows.row1597) := by
    change Data.row1597.contains = (SortedNatTree.contains FastRows.row1597)
    exact FastRows.row1597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child9 : search edge 3 [73, 1597] root18Candidates9 1670 winner = winner := by decide
theorem root18Step9 : scan edge (search edge 3) 4 [73] 73 root18Tail9 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail10 winner := by
  rw [root18Tail9, scan, if_neg (by decide), root18Candidates9_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail10
    (if 3 ≤ root18Candidates9.length then search edge 3 [73, 1597] root18Candidates9 1670 winner else winner) = _
  rw [if_pos (by decide), root18Child9]
def root18Candidates10 : List Nat := [2293, 8677, 18523]
set_option maxRecDepth 8192 in
theorem root18Candidates10_checked : candidates edge 4 73 1933 root18Tail11 winner.bound = root18Candidates10 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child10 : search edge 3 [73, 1933] root18Candidates10 2006 winner = winner := by decide
theorem root18Step10 : scan edge (search edge 3) 4 [73] 73 root18Tail10 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail11 winner := by
  rw [root18Tail10, scan, if_neg (by decide), root18Candidates10_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail11
    (if 3 ≤ root18Candidates10.length then search edge 3 [73, 1933] root18Candidates10 2006 winner else winner) = _
  rw [if_pos (by decide), root18Child10]
def root18Candidates11 : List Nat := [6121, 7309, 9859, 11287, 11701, 17977]
set_option maxRecDepth 8192 in
theorem root18Candidates11_checked : candidates edge 4 73 2029 root18Tail12 winner.bound = root18Candidates11 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child11 : search edge 3 [73, 2029] root18Candidates11 2102 winner = winner := by decide
theorem root18Step11 : scan edge (search edge 3) 4 [73] 73 root18Tail11 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail12 winner := by
  rw [root18Tail11, scan, if_neg (by decide), root18Candidates11_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail12
    (if 3 ≤ root18Candidates11.length then search edge 3 [73, 2029] root18Candidates11 2102 winner else winner) = _
  rw [if_pos (by decide), root18Child11]
def root18Candidates12 : List Nat := [6277, 6793, 7537]
set_option maxRecDepth 8192 in
theorem root18Candidates12_checked : candidates edge 4 73 2287 root18Tail13 winner.bound = root18Candidates12 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child12 : search edge 3 [73, 2287] root18Candidates12 2360 winner = winner := by decide
theorem root18Step12 : scan edge (search edge 3) 4 [73] 73 root18Tail12 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail13 winner := by
  rw [root18Tail12, scan, if_neg (by decide), root18Candidates12_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail13
    (if 3 ≤ root18Candidates12.length then search edge 3 [73, 2287] root18Candidates12 2360 winner else winner) = _
  rw [if_pos (by decide), root18Child12]
def root18Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root18Candidates13_checked : candidates edge 4 73 2293 root18Tail14 winner.bound = root18Candidates13 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root18Step13 : scan edge (search edge 3) 4 [73] 73 root18Tail13 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail14 winner := by
  rw [root18Tail13, scan, if_neg (by decide), root18Candidates13_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail14
    (if 3 ≤ root18Candidates13.length then search edge 3 [73, 2293] root18Candidates13 2366 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates14 : List Nat := [6277, 9601]
set_option maxRecDepth 8192 in
theorem root18Candidates14_checked : candidates edge 4 73 2467 root18Tail15 winner.bound = root18Candidates14 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root18Step14 : scan edge (search edge 3) 4 [73] 73 root18Tail14 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail15 winner := by
  rw [root18Tail14, scan, if_neg (by decide), root18Candidates14_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail15
    (if 3 ≤ root18Candidates14.length then search edge 3 [73, 2467] root18Candidates14 2540 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates15 : List Nat := [4759, 11467, 21757]
set_option maxRecDepth 8192 in
theorem root18Candidates15_checked : candidates edge 4 73 2521 root18Tail16 winner.bound = root18Candidates15 := by
  unfold candidates
  have row : edge 2521 = (SortedNatTree.contains FastRows.row2521) := by
    change Data.row2521.contains = (SortedNatTree.contains FastRows.row2521)
    exact FastRows.row2521_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child15 : search edge 3 [73, 2521] root18Candidates15 2594 winner = winner := by decide
theorem root18Step15 : scan edge (search edge 3) 4 [73] 73 root18Tail15 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail16 winner := by
  rw [root18Tail15, scan, if_neg (by decide), root18Candidates15_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail16
    (if 3 ≤ root18Candidates15.length then search edge 3 [73, 2521] root18Candidates15 2594 winner else winner) = _
  rw [if_pos (by decide), root18Child15]
def root18Candidates16 : List Nat := [10651, 20359]
set_option maxRecDepth 8192 in
theorem root18Candidates16_checked : candidates edge 4 73 2617 root18Tail17 winner.bound = root18Candidates16 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
theorem root18Step16 : scan edge (search edge 3) 4 [73] 73 root18Tail16 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail17 winner := by
  rw [root18Tail16, scan, if_neg (by decide), root18Candidates16_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail17
    (if 3 ≤ root18Candidates16.length then search edge 3 [73, 2617] root18Candidates16 2690 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates17 : List Nat := [6367, 18379]
set_option maxRecDepth 8192 in
theorem root18Candidates17_checked : candidates edge 4 73 2689 root18Tail18 winner.bound = root18Candidates17 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root18Step17 : scan edge (search edge 3) 4 [73] 73 root18Tail17 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail18 winner := by
  rw [root18Tail17, scan, if_neg (by decide), root18Candidates17_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail18
    (if 3 ≤ root18Candidates17.length then search edge 3 [73, 2689] root18Candidates17 2762 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates18 : List Nat := [11467]
set_option maxRecDepth 8192 in
theorem root18Candidates18_checked : candidates edge 4 73 2749 root18Tail19 winner.bound = root18Candidates18 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root18Step18 : scan edge (search edge 3) 4 [73] 73 root18Tail18 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail19 winner := by
  rw [root18Tail18, scan, if_neg (by decide), root18Candidates18_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail19
    (if 3 ≤ root18Candidates18.length then search edge 3 [73, 2749] root18Candidates18 2822 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates19 : List Nat := [3433, 5107, 5443, 15739]
set_option maxRecDepth 8192 in
theorem root18Candidates19_checked : candidates edge 4 73 3391 root18Tail20 winner.bound = root18Candidates19 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child19 : search edge 3 [73, 3391] root18Candidates19 3464 winner = winner := by decide
theorem root18Step19 : scan edge (search edge 3) 4 [73] 73 root18Tail19 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail20 winner := by
  rw [root18Tail19, scan, if_neg (by decide), root18Candidates19_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail20
    (if 3 ≤ root18Candidates19.length then search edge 3 [73, 3391] root18Candidates19 3464 winner else winner) = _
  rw [if_pos (by decide), root18Child19]
def root18Candidates20 : List Nat := [7129, 7207]
set_option maxRecDepth 8192 in
theorem root18Candidates20_checked : candidates edge 4 73 3433 root18Tail21 winner.bound = root18Candidates20 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root18Step20 : scan edge (search edge 3) 4 [73] 73 root18Tail20 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail21 winner := by
  rw [root18Tail20, scan, if_neg (by decide), root18Candidates20_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail21
    (if 3 ≤ root18Candidates20.length then search edge 3 [73, 3433] root18Candidates20 3506 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates21 : List Nat := [4177, 5659, 7159, 9433, 11149, 21529]
set_option maxRecDepth 8192 in
theorem root18Candidates21_checked : candidates edge 4 73 3517 root18Tail22 winner.bound = root18Candidates21 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child21 : search edge 3 [73, 3517] root18Candidates21 3590 winner = winner := by decide
theorem root18Step21 : scan edge (search edge 3) 4 [73] 73 root18Tail21 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail22 winner := by
  rw [root18Tail21, scan, if_neg (by decide), root18Candidates21_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail22
    (if 3 ≤ root18Candidates21.length then search edge 3 [73, 3517] root18Candidates21 3590 winner else winner) = _
  rw [if_pos (by decide), root18Child21]
def root18Candidates22 : List Nat := [4549, 4759]
set_option maxRecDepth 8192 in
theorem root18Candidates22_checked : candidates edge 4 73 3823 root18Tail23 winner.bound = root18Candidates22 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root18Step22 : scan edge (search edge 3) 4 [73] 73 root18Tail22 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail23 winner := by
  rw [root18Tail22, scan, if_neg (by decide), root18Candidates22_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail23
    (if 3 ≤ root18Candidates22.length then search edge 3 [73, 3823] root18Candidates22 3896 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates23 : List Nat := [5659, 6367, 16567]
set_option maxRecDepth 8192 in
theorem root18Candidates23_checked : candidates edge 4 73 3847 root18Tail24 winner.bound = root18Candidates23 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child23 : search edge 3 [73, 3847] root18Candidates23 3920 winner = winner := by decide
theorem root18Step23 : scan edge (search edge 3) 4 [73] 73 root18Tail23 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail24 winner := by
  rw [root18Tail23, scan, if_neg (by decide), root18Candidates23_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail24
    (if 3 ≤ root18Candidates23.length then search edge 3 [73, 3847] root18Candidates23 3920 winner else winner) = _
  rw [if_pos (by decide), root18Child23]
def root18Candidates24 : List Nat := [7537, 13693]
set_option maxRecDepth 8192 in
theorem root18Candidates24_checked : candidates edge 4 73 4177 root18Tail25 winner.bound = root18Candidates24 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root18Step24 : scan edge (search edge 3) 4 [73] 73 root18Tail24 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail25 winner := by
  rw [root18Tail24, scan, if_neg (by decide), root18Candidates24_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail25
    (if 3 ≤ root18Candidates24.length then search edge 3 [73, 4177] root18Candidates24 4250 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates25 : List Nat := [9631]
set_option maxRecDepth 8192 in
theorem root18Candidates25_checked : candidates edge 4 73 4549 root18Tail26 winner.bound = root18Candidates25 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root18Step25 : scan edge (search edge 3) 4 [73] 73 root18Tail25 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail26 winner := by
  rw [root18Tail25, scan, if_neg (by decide), root18Candidates25_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail26
    (if 3 ≤ root18Candidates25.length then search edge 3 [73, 4549] root18Candidates25 4622 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates26 : List Nat := [9601, 9787]
set_option maxRecDepth 8192 in
theorem root18Candidates26_checked : candidates edge 4 73 4663 root18Tail27 winner.bound = root18Candidates26 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root18Step26 : scan edge (search edge 3) 4 [73] 73 root18Tail26 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail27 winner := by
  rw [root18Tail26, scan, if_neg (by decide), root18Candidates26_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail27
    (if 3 ≤ root18Candidates26.length then search edge 3 [73, 4663] root18Candidates26 4736 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates27 : List Nat := [6823]
set_option maxRecDepth 8192 in
theorem root18Candidates27_checked : candidates edge 4 73 4759 root18Tail28 winner.bound = root18Candidates27 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root18Step27 : scan edge (search edge 3) 4 [73] 73 root18Tail27 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail28 winner := by
  rw [root18Tail27, scan, if_neg (by decide), root18Candidates27_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail28
    (if 3 ≤ root18Candidates27.length then search edge 3 [73, 4759] root18Candidates27 4832 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates28 : List Nat := [7351]
set_option maxRecDepth 8192 in
theorem root18Candidates28_checked : candidates edge 4 73 4813 root18Tail29 winner.bound = root18Candidates28 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root18Step28 : scan edge (search edge 3) 4 [73] 73 root18Tail28 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail29 winner := by
  rw [root18Tail28, scan, if_neg (by decide), root18Candidates28_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail29
    (if 3 ≤ root18Candidates28.length then search edge 3 [73, 4813] root18Candidates28 4886 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates29 : List Nat := [9787, 18523]
set_option maxRecDepth 8192 in
theorem root18Candidates29_checked : candidates edge 4 73 4957 root18Tail30 winner.bound = root18Candidates29 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root18Step29 : scan edge (search edge 3) 4 [73] 73 root18Tail29 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail30 winner := by
  rw [root18Tail29, scan, if_neg (by decide), root18Candidates29_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail30
    (if 3 ≤ root18Candidates29.length then search edge 3 [73, 4957] root18Candidates29 5030 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates30 : List Nat := [6793, 18379]
set_option maxRecDepth 8192 in
theorem root18Candidates30_checked : candidates edge 4 73 5107 root18Tail31 winner.bound = root18Candidates30 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root18Step30 : scan edge (search edge 3) 4 [73] 73 root18Tail30 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail31 winner := by
  rw [root18Tail30, scan, if_neg (by decide), root18Candidates30_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail31
    (if 3 ≤ root18Candidates30.length then search edge 3 [73, 5107] root18Candidates30 5180 winner else winner) = _
  rw [if_neg (by decide)]
def root18Candidates31 : List Nat := [7351, 12037, 13513]
set_option maxRecDepth 8192 in
theorem root18Candidates31_checked : candidates edge 4 73 5443 root18Tail32 winner.bound = root18Candidates31 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root18Child31 : search edge 3 [73, 5443] root18Candidates31 5516 winner = winner := by decide
theorem root18Step31 : scan edge (search edge 3) 4 [73] 73 root18Tail31 winner =
    scan edge (search edge 3) 4 [73] 73 root18Tail32 winner := by
  rw [root18Tail31, scan, if_neg (by decide), root18Candidates31_checked]
  change scan edge (search edge 3) 4 [73] 73 root18Tail32
    (if 3 ≤ root18Candidates31.length then search edge 3 [73, 5443] root18Candidates31 5516 winner else winner) = _
  rw [if_pos (by decide), root18Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
