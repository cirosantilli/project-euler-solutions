import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root139Candidates : List Nat := Data.row821
set_option maxRecDepth 8192 in
theorem root139Candidates_checked : candidates edge 5 0 821 tail140 winner.bound = root139Candidates := by
  apply candidates_eq_row edge 5 0 821 tail140 root139Candidates winner.bound tail140_ordered
  · decide
  · decide
  · funext q
    rfl
def root139Tail62 : List Nat := []
def root139Tail61 : List Nat := 25031 :: root139Tail62
def root139Tail60 : List Nat := 24989 :: root139Tail61
def root139Tail59 : List Nat := 24473 :: root139Tail60
def root139Tail58 : List Nat := 24359 :: root139Tail59
def root139Tail57 : List Nat := 23687 :: root139Tail58
def root139Tail56 : List Nat := 23633 :: root139Tail57
def root139Tail55 : List Nat := 23057 :: root139Tail56
def root139Tail54 : List Nat := 22961 :: root139Tail55
def root139Tail53 : List Nat := 22727 :: root139Tail54
def root139Tail52 : List Nat := 22571 :: root139Tail53
def root139Tail51 : List Nat := 22067 :: root139Tail52
def root139Tail50 : List Nat := 21503 :: root139Tail51
def root139Tail49 : List Nat := 20177 :: root139Tail50
def root139Tail48 : List Nat := 19841 :: root139Tail49
def root139Tail47 : List Nat := 19727 :: root139Tail48
def root139Tail46 : List Nat := 18899 :: root139Tail47
def root139Tail45 : List Nat := 18521 :: root139Tail46
def root139Tail44 : List Nat := 18077 :: root139Tail45
def root139Tail43 : List Nat := 17807 :: root139Tail44
def root139Tail42 : List Nat := 16901 :: root139Tail43
def root139Tail41 : List Nat := 16703 :: root139Tail42
def root139Tail40 : List Nat := 16253 :: root139Tail41
def root139Tail39 : List Nat := 15923 :: root139Tail40
def root139Tail38 : List Nat := 15581 :: root139Tail39
def root139Tail37 : List Nat := 15053 :: root139Tail38
def root139Tail36 : List Nat := 14957 :: root139Tail37
def root139Tail35 : List Nat := 14321 :: root139Tail36
def root139Tail34 : List Nat := 13679 :: root139Tail35
def root139Tail33 : List Nat := 12689 :: root139Tail34
def root139Tail32 : List Nat := 12347 :: root139Tail33
def root139Tail31 : List Nat := 12197 :: root139Tail32
def root139Tail30 : List Nat := 12101 :: root139Tail31
def root139Tail29 : List Nat := 11423 :: root139Tail30
def root139Tail28 : List Nat := 11261 :: root139Tail29
def root139Tail27 : List Nat := 10169 :: root139Tail28
def root139Tail26 : List Nat := 8831 :: root139Tail27
def root139Tail25 : List Nat := 8753 :: root139Tail26
def root139Tail24 : List Nat := 8681 :: root139Tail25
def root139Tail23 : List Nat := 8669 :: root139Tail24
def root139Tail22 : List Nat := 8663 :: root139Tail23
def root139Tail21 : List Nat := 8597 :: root139Tail22
def root139Tail20 : List Nat := 8447 :: root139Tail21
def root139Tail19 : List Nat := 7523 :: root139Tail20
def root139Tail18 : List Nat := 6737 :: root139Tail19
def root139Tail17 : List Nat := 6521 :: root139Tail18
def root139Tail16 : List Nat := 6317 :: root139Tail17
def root139Tail15 : List Nat := 5849 :: root139Tail16
def root139Tail14 : List Nat := 5843 :: root139Tail15
def root139Tail13 : List Nat := 5783 :: root139Tail14
def root139Tail12 : List Nat := 4691 :: root139Tail13
def root139Tail11 : List Nat := 3581 :: root139Tail12
def root139Tail10 : List Nat := 2969 :: root139Tail11
def root139Tail9 : List Nat := 2591 :: root139Tail10
def root139Tail8 : List Nat := 2549 :: root139Tail9
def root139Tail7 : List Nat := 1949 :: root139Tail8
def root139Tail6 : List Nat := 1607 :: root139Tail7
def root139Tail5 : List Nat := 1433 :: root139Tail6
def root139Tail4 : List Nat := 1163 :: root139Tail5
def root139Tail3 : List Nat := 1109 :: root139Tail4
def root139Tail2 : List Nat := 971 :: root139Tail3
def root139Tail1 : List Nat := 857 :: root139Tail2
def root139Tail0 : List Nat := 827 :: root139Tail1
def root139Candidates0 : List Nat := [857, 8663, 22571]
set_option maxRecDepth 8192 in
theorem root139Candidates0_checked : candidates edge 4 821 827 root139Tail1 winner.bound = root139Candidates0 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root139Child0 : search edge 3 [821, 827] root139Candidates0 1648 winner = winner := by decide
theorem root139Step0 : scan edge (search edge 3) 4 [821] 821 root139Tail0 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail1 winner := by
  rw [root139Tail0, scan, if_neg (by decide), root139Candidates0_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail1
    (if 3 ≤ root139Candidates0.length then search edge 3 [821, 827] root139Candidates0 1648 winner else winner) = _
  rw [if_pos (by decide), root139Child0]
def root139Candidates1 : List Nat := [8681]
set_option maxRecDepth 8192 in
theorem root139Candidates1_checked : candidates edge 4 821 857 root139Tail2 winner.bound = root139Candidates1 := by
  unfold candidates
  have row : edge 857 = (SortedNatTree.contains FastRows.row857) := by
    change Data.row857.contains = (SortedNatTree.contains FastRows.row857)
    exact FastRows.row857_checked.symm
  rw [row]
  decide
theorem root139Step1 : scan edge (search edge 3) 4 [821] 821 root139Tail1 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail2 winner := by
  rw [root139Tail1, scan, if_neg (by decide), root139Candidates1_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail2
    (if 3 ≤ root139Candidates1.length then search edge 3 [821, 857] root139Candidates1 1678 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates2 : List Nat := [8831, 14321, 21503, 23633]
set_option maxRecDepth 8192 in
theorem root139Candidates2_checked : candidates edge 4 821 971 root139Tail3 winner.bound = root139Candidates2 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root139Child2 : search edge 3 [821, 971] root139Candidates2 1792 winner = winner := by decide
theorem root139Step2 : scan edge (search edge 3) 4 [821] 821 root139Tail2 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail3 winner := by
  rw [root139Tail2, scan, if_neg (by decide), root139Candidates2_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail3
    (if 3 ≤ root139Candidates2.length then search edge 3 [821, 971] root139Candidates2 1792 winner else winner) = _
  rw [if_pos (by decide), root139Child2]
def root139Candidates3 : List Nat := [6521, 8663, 14957]
set_option maxRecDepth 8192 in
theorem root139Candidates3_checked : candidates edge 4 821 1109 root139Tail4 winner.bound = root139Candidates3 := by
  unfold candidates
  have row : edge 1109 = (SortedNatTree.contains FastRows.row1109) := by
    change Data.row1109.contains = (SortedNatTree.contains FastRows.row1109)
    exact FastRows.row1109_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root139Child3 : search edge 3 [821, 1109] root139Candidates3 1930 winner = winner := by decide
theorem root139Step3 : scan edge (search edge 3) 4 [821] 821 root139Tail3 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail4 winner := by
  rw [root139Tail3, scan, if_neg (by decide), root139Candidates3_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail4
    (if 3 ≤ root139Candidates3.length then search edge 3 [821, 1109] root139Candidates3 1930 winner else winner) = _
  rw [if_pos (by decide), root139Child3]
def root139Candidates4 : List Nat := [5849, 12197, 12347, 15053]
set_option maxRecDepth 8192 in
theorem root139Candidates4_checked : candidates edge 4 821 1163 root139Tail5 winner.bound = root139Candidates4 := by
  unfold candidates
  have row : edge 1163 = (SortedNatTree.contains FastRows.row1163) := by
    change Data.row1163.contains = (SortedNatTree.contains FastRows.row1163)
    exact FastRows.row1163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root139Child4 : search edge 3 [821, 1163] root139Candidates4 1984 winner = winner := by decide
theorem root139Step4 : scan edge (search edge 3) 4 [821] 821 root139Tail4 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail5 winner := by
  rw [root139Tail4, scan, if_neg (by decide), root139Candidates4_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail5
    (if 3 ≤ root139Candidates4.length then search edge 3 [821, 1163] root139Candidates4 1984 winner else winner) = _
  rw [if_pos (by decide), root139Child4]
def root139Candidates5 : List Nat := [6737]
set_option maxRecDepth 8192 in
theorem root139Candidates5_checked : candidates edge 4 821 1433 root139Tail6 winner.bound = root139Candidates5 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
theorem root139Step5 : scan edge (search edge 3) 4 [821] 821 root139Tail5 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail6 winner := by
  rw [root139Tail5, scan, if_neg (by decide), root139Candidates5_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail6
    (if 3 ≤ root139Candidates5.length then search edge 3 [821, 1433] root139Candidates5 2254 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates6 : List Nat := [2549, 5849, 16901]
set_option maxRecDepth 8192 in
theorem root139Candidates6_checked : candidates edge 4 821 1607 root139Tail7 winner.bound = root139Candidates6 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root139Child6 : search edge 3 [821, 1607] root139Candidates6 2428 winner = winner := by decide
theorem root139Step6 : scan edge (search edge 3) 4 [821] 821 root139Tail6 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail7 winner := by
  rw [root139Tail6, scan, if_neg (by decide), root139Candidates6_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail7
    (if 3 ≤ root139Candidates6.length then search edge 3 [821, 1607] root139Candidates6 2428 winner else winner) = _
  rw [if_pos (by decide), root139Child6]
def root139Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root139Candidates7_checked : candidates edge 4 821 1949 root139Tail8 winner.bound = root139Candidates7 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root139Step7 : scan edge (search edge 3) 4 [821] 821 root139Tail7 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail8 winner := by
  rw [root139Tail7, scan, if_neg (by decide), root139Candidates7_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail8
    (if 3 ≤ root139Candidates7.length then search edge 3 [821, 1949] root139Candidates7 2770 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates8 : List Nat := [18899]
set_option maxRecDepth 8192 in
theorem root139Candidates8_checked : candidates edge 4 821 2549 root139Tail9 winner.bound = root139Candidates8 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root139Step8 : scan edge (search edge 3) 4 [821] 821 root139Tail8 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail9 winner := by
  rw [root139Tail8, scan, if_neg (by decide), root139Candidates8_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail9
    (if 3 ≤ root139Candidates8.length then search edge 3 [821, 2549] root139Candidates8 3370 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates9 : List Nat := [5849]
set_option maxRecDepth 8192 in
theorem root139Candidates9_checked : candidates edge 4 821 2591 root139Tail10 winner.bound = root139Candidates9 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root139Step9 : scan edge (search edge 3) 4 [821] 821 root139Tail9 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail10 winner := by
  rw [root139Tail9, scan, if_neg (by decide), root139Candidates9_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail10
    (if 3 ≤ root139Candidates9.length then search edge 3 [821, 2591] root139Candidates9 3412 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates10 : List Nat := [8597, 12197]
set_option maxRecDepth 8192 in
theorem root139Candidates10_checked : candidates edge 4 821 2969 root139Tail11 winner.bound = root139Candidates10 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root139Step10 : scan edge (search edge 3) 4 [821] 821 root139Tail10 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail11 winner := by
  rw [root139Tail10, scan, if_neg (by decide), root139Candidates10_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail11
    (if 3 ≤ root139Candidates10.length then search edge 3 [821, 2969] root139Candidates10 3790 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates11 : List Nat := [5849]
set_option maxRecDepth 8192 in
theorem root139Candidates11_checked : candidates edge 4 821 3581 root139Tail12 winner.bound = root139Candidates11 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root139Step11 : scan edge (search edge 3) 4 [821] 821 root139Tail11 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail12 winner := by
  rw [root139Tail11, scan, if_neg (by decide), root139Candidates11_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail12
    (if 3 ≤ root139Candidates11.length then search edge 3 [821, 3581] root139Candidates11 4402 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates12 : List Nat := [6521, 19841]
set_option maxRecDepth 8192 in
theorem root139Candidates12_checked : candidates edge 4 821 4691 root139Tail13 winner.bound = root139Candidates12 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root139Step12 : scan edge (search edge 3) 4 [821] 821 root139Tail12 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail13 winner := by
  rw [root139Tail12, scan, if_neg (by decide), root139Candidates12_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail13
    (if 3 ≤ root139Candidates12.length then search edge 3 [821, 4691] root139Candidates12 5512 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates13 : List Nat := [18077]
set_option maxRecDepth 8192 in
theorem root139Candidates13_checked : candidates edge 4 821 5783 root139Tail14 winner.bound = root139Candidates13 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root139Step13 : scan edge (search edge 3) 4 [821] 821 root139Tail13 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail14 winner := by
  rw [root139Tail13, scan, if_neg (by decide), root139Candidates13_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail14
    (if 3 ≤ root139Candidates13.length then search edge 3 [821, 5783] root139Candidates13 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root139Candidates14_checked : candidates edge 4 821 5843 root139Tail15 winner.bound = root139Candidates14 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root139Step14 : scan edge (search edge 3) 4 [821] 821 root139Tail14 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail15 winner := by
  rw [root139Tail14, scan, if_neg (by decide), root139Candidates14_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail15
    (if 3 ≤ root139Candidates14.length then search edge 3 [821, 5843] root139Candidates14 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root139Candidates15 : List Nat := [8681, 8753]
set_option maxRecDepth 8192 in
theorem root139Candidates15_checked : candidates edge 4 821 5849 root139Tail16 winner.bound = root139Candidates15 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root139Step15 : scan edge (search edge 3) 4 [821] 821 root139Tail15 winner =
    scan edge (search edge 3) 4 [821] 821 root139Tail16 winner := by
  rw [root139Tail15, scan, if_neg (by decide), root139Candidates15_checked]
  change scan edge (search edge 3) 4 [821] 821 root139Tail16
    (if 3 ≤ root139Candidates15.length then search edge 3 [821, 5849] root139Candidates15 6670 winner else winner) = _
  rw [if_neg (by decide)]
theorem root139Scan16 : scan edge (search edge 3) 4 [821] 821 root139Tail16 winner = winner := by rw [root139Tail16, scan, if_pos (by decide)]
theorem root139Scan15 : scan edge (search edge 3) 4 [821] 821 root139Tail15 winner = winner :=
  (root139Step15).trans root139Scan16
theorem root139Scan14 : scan edge (search edge 3) 4 [821] 821 root139Tail14 winner = winner :=
  (root139Step14).trans root139Scan15
theorem root139Scan13 : scan edge (search edge 3) 4 [821] 821 root139Tail13 winner = winner :=
  (root139Step13).trans root139Scan14
theorem root139Scan12 : scan edge (search edge 3) 4 [821] 821 root139Tail12 winner = winner :=
  (root139Step12).trans root139Scan13
theorem root139Scan11 : scan edge (search edge 3) 4 [821] 821 root139Tail11 winner = winner :=
  (root139Step11).trans root139Scan12
theorem root139Scan10 : scan edge (search edge 3) 4 [821] 821 root139Tail10 winner = winner :=
  (root139Step10).trans root139Scan11
theorem root139Scan9 : scan edge (search edge 3) 4 [821] 821 root139Tail9 winner = winner :=
  (root139Step9).trans root139Scan10
theorem root139Scan8 : scan edge (search edge 3) 4 [821] 821 root139Tail8 winner = winner :=
  (root139Step8).trans root139Scan9
theorem root139Scan7 : scan edge (search edge 3) 4 [821] 821 root139Tail7 winner = winner :=
  (root139Step7).trans root139Scan8
theorem root139Scan6 : scan edge (search edge 3) 4 [821] 821 root139Tail6 winner = winner :=
  (root139Step6).trans root139Scan7
theorem root139Scan5 : scan edge (search edge 3) 4 [821] 821 root139Tail5 winner = winner :=
  (root139Step5).trans root139Scan6
theorem root139Scan4 : scan edge (search edge 3) 4 [821] 821 root139Tail4 winner = winner :=
  (root139Step4).trans root139Scan5
theorem root139Scan3 : scan edge (search edge 3) 4 [821] 821 root139Tail3 winner = winner :=
  (root139Step3).trans root139Scan4
theorem root139Scan2 : scan edge (search edge 3) 4 [821] 821 root139Tail2 winner = winner :=
  (root139Step2).trans root139Scan3
theorem root139Scan1 : scan edge (search edge 3) 4 [821] 821 root139Tail1 winner = winner :=
  (root139Step1).trans root139Scan2
theorem root139Scan0 : scan edge (search edge 3) 4 [821] 821 root139Tail0 winner = winner :=
  (root139Step0).trans root139Scan1
set_option maxRecDepth 4096 in
theorem root139Branch : search edge 4 [821] root139Candidates 821 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [821] 821 root139Tail0 winner = winner
  exact root139Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
