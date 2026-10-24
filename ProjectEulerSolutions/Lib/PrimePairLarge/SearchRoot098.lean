import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root98Candidates : List Nat := Data.row547
set_option maxRecDepth 8192 in
theorem root98Candidates_checked : candidates edge 5 0 547 tail99 winner.bound = root98Candidates := by
  apply candidates_eq_row edge 5 0 547 tail99 root98Candidates winner.bound tail99_ordered
  · decide
  · decide
  · funext q
    rfl
def root98Tail72 : List Nat := []
def root98Tail71 : List Nat := 23761 :: root98Tail72
def root98Tail70 : List Nat := 23593 :: root98Tail71
def root98Tail69 : List Nat := 23131 :: root98Tail70
def root98Tail68 : List Nat := 22669 :: root98Tail69
def root98Tail67 : List Nat := 22651 :: root98Tail68
def root98Tail66 : List Nat := 22573 :: root98Tail67
def root98Tail65 : List Nat := 22189 :: root98Tail66
def root98Tail64 : List Nat := 22039 :: root98Tail65
def root98Tail63 : List Nat := 22027 :: root98Tail64
def root98Tail62 : List Nat := 21859 :: root98Tail63
def root98Tail61 : List Nat := 21649 :: root98Tail62
def root98Tail60 : List Nat := 21517 :: root98Tail61
def root98Tail59 : List Nat := 21313 :: root98Tail60
def root98Tail58 : List Nat := 20947 :: root98Tail59
def root98Tail57 : List Nat := 20719 :: root98Tail58
def root98Tail56 : List Nat := 20509 :: root98Tail57
def root98Tail55 : List Nat := 20431 :: root98Tail56
def root98Tail54 : List Nat := 20011 :: root98Tail55
def root98Tail53 : List Nat := 19597 :: root98Tail54
def root98Tail52 : List Nat := 19081 :: root98Tail53
def root98Tail51 : List Nat := 18913 :: root98Tail52
def root98Tail50 : List Nat := 18757 :: root98Tail51
def root98Tail49 : List Nat := 18553 :: root98Tail50
def root98Tail48 : List Nat := 18457 :: root98Tail49
def root98Tail47 : List Nat := 18451 :: root98Tail48
def root98Tail46 : List Nat := 18289 :: root98Tail47
def root98Tail45 : List Nat := 17293 :: root98Tail46
def root98Tail44 : List Nat := 17167 :: root98Tail45
def root98Tail43 : List Nat := 16231 :: root98Tail44
def root98Tail42 : List Nat := 15901 :: root98Tail43
def root98Tail41 : List Nat := 15559 :: root98Tail42
def root98Tail40 : List Nat := 15319 :: root98Tail41
def root98Tail39 : List Nat := 15187 :: root98Tail40
def root98Tail38 : List Nat := 15013 :: root98Tail39
def root98Tail37 : List Nat := 14683 :: root98Tail38
def root98Tail36 : List Nat := 14563 :: root98Tail37
def root98Tail35 : List Nat := 14389 :: root98Tail36
def root98Tail34 : List Nat := 13627 :: root98Tail35
def root98Tail33 : List Nat := 13093 :: root98Tail34
def root98Tail32 : List Nat := 11827 :: root98Tail33
def root98Tail31 : List Nat := 11689 :: root98Tail32
def root98Tail30 : List Nat := 10891 :: root98Tail31
def root98Tail29 : List Nat := 10321 :: root98Tail30
def root98Tail28 : List Nat := 10111 :: root98Tail29
def root98Tail27 : List Nat := 9319 :: root98Tail28
def root98Tail26 : List Nat := 9013 :: root98Tail27
def root98Tail25 : List Nat := 8713 :: root98Tail26
def root98Tail24 : List Nat := 8311 :: root98Tail25
def root98Tail23 : List Nat := 7867 :: root98Tail24
def root98Tail22 : List Nat := 7237 :: root98Tail23
def root98Tail21 : List Nat := 6673 :: root98Tail22
def root98Tail20 : List Nat := 6451 :: root98Tail21
def root98Tail19 : List Nat := 6073 :: root98Tail20
def root98Tail18 : List Nat := 5557 :: root98Tail19
def root98Tail17 : List Nat := 5443 :: root98Tail18
def root98Tail16 : List Nat := 4549 :: root98Tail17
def root98Tail15 : List Nat := 4519 :: root98Tail16
def root98Tail14 : List Nat := 3271 :: root98Tail15
def root98Tail13 : List Nat := 3229 :: root98Tail14
def root98Tail12 : List Nat := 3001 :: root98Tail13
def root98Tail11 : List Nat := 2473 :: root98Tail12
def root98Tail10 : List Nat := 1699 :: root98Tail11
def root98Tail9 : List Nat := 1663 :: root98Tail10
def root98Tail8 : List Nat := 1237 :: root98Tail9
def root98Tail7 : List Nat := 853 :: root98Tail8
def root98Tail6 : List Nat := 823 :: root98Tail7
def root98Tail5 : List Nat := 787 :: root98Tail6
def root98Tail4 : List Nat := 769 :: root98Tail5
def root98Tail3 : List Nat := 709 :: root98Tail4
def root98Tail2 : List Nat := 661 :: root98Tail3
def root98Tail1 : List Nat := 643 :: root98Tail2
def root98Tail0 : List Nat := 577 :: root98Tail1
def root98Candidates0 : List Nat := [20431]
set_option maxRecDepth 8192 in
theorem root98Candidates0_checked : candidates edge 4 547 577 root98Tail1 winner.bound = root98Candidates0 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
theorem root98Step0 : scan edge (search edge 3) 4 [547] 547 root98Tail0 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail1 winner := by
  rw [root98Tail0, scan, if_neg (by decide), root98Candidates0_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail1
    (if 3 ≤ root98Candidates0.length then search edge 3 [547, 577] root98Candidates0 1124 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates1 : List Nat := [10111, 18757]
set_option maxRecDepth 8192 in
theorem root98Candidates1_checked : candidates edge 4 547 643 root98Tail2 winner.bound = root98Candidates1 := by
  unfold candidates
  have row : edge 643 = (SortedNatTree.contains FastRows.row643) := by
    change Data.row643.contains = (SortedNatTree.contains FastRows.row643)
    exact FastRows.row643_checked.symm
  rw [row]
  decide
theorem root98Step1 : scan edge (search edge 3) 4 [547] 547 root98Tail1 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail2 winner := by
  rw [root98Tail1, scan, if_neg (by decide), root98Candidates1_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail2
    (if 3 ≤ root98Candidates1.length then search edge 3 [547, 643] root98Candidates1 1190 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates2 : List Nat := [769, 1237, 20011]
set_option maxRecDepth 8192 in
theorem root98Candidates2_checked : candidates edge 4 547 661 root98Tail3 winner.bound = root98Candidates2 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child2 : search edge 3 [547, 661] root98Candidates2 1208 winner = winner := by decide
theorem root98Step2 : scan edge (search edge 3) 4 [547] 547 root98Tail2 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail3 winner := by
  rw [root98Tail2, scan, if_neg (by decide), root98Candidates2_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail3
    (if 3 ≤ root98Candidates2.length then search edge 3 [547, 661] root98Candidates2 1208 winner else winner) = _
  rw [if_pos (by decide), root98Child2]
def root98Candidates3 : List Nat := [823, 1663, 4519, 8713, 10891, 22039, 22669]
set_option maxRecDepth 8192 in
theorem root98Candidates3_checked : candidates edge 4 547 709 root98Tail4 winner.bound = root98Candidates3 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child3 : search edge 3 [547, 709] root98Candidates3 1256 winner = winner := by decide
theorem root98Step3 : scan edge (search edge 3) 4 [547] 547 root98Tail3 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail4 winner := by
  rw [root98Tail3, scan, if_neg (by decide), root98Candidates3_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail4
    (if 3 ≤ root98Candidates3.length then search edge 3 [547, 709] root98Candidates3 1256 winner else winner) = _
  rw [if_pos (by decide), root98Child3]
def root98Candidates4 : List Nat := [23131]
set_option maxRecDepth 8192 in
theorem root98Candidates4_checked : candidates edge 4 547 769 root98Tail5 winner.bound = root98Candidates4 := by
  unfold candidates
  have row : edge 769 = (SortedNatTree.contains FastRows.row769) := by
    change Data.row769.contains = (SortedNatTree.contains FastRows.row769)
    exact FastRows.row769_checked.symm
  rw [row]
  decide
theorem root98Step4 : scan edge (search edge 3) 4 [547] 547 root98Tail4 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail5 winner := by
  rw [root98Tail4, scan, if_neg (by decide), root98Candidates4_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail5
    (if 3 ≤ root98Candidates4.length then search edge 3 [547, 769] root98Candidates4 1316 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates5 : List Nat := [823, 20011, 20431]
set_option maxRecDepth 8192 in
theorem root98Candidates5_checked : candidates edge 4 547 787 root98Tail6 winner.bound = root98Candidates5 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child5 : search edge 3 [547, 787] root98Candidates5 1334 winner = winner := by decide
theorem root98Step5 : scan edge (search edge 3) 4 [547] 547 root98Tail5 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail6 winner := by
  rw [root98Tail5, scan, if_neg (by decide), root98Candidates5_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail6
    (if 3 ≤ root98Candidates5.length then search edge 3 [547, 787] root98Candidates5 1334 winner else winner) = _
  rw [if_pos (by decide), root98Child5]
def root98Candidates6 : List Nat := [7237, 15319, 15559]
set_option maxRecDepth 8192 in
theorem root98Candidates6_checked : candidates edge 4 547 823 root98Tail7 winner.bound = root98Candidates6 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child6 : search edge 3 [547, 823] root98Candidates6 1370 winner = winner := by decide
theorem root98Step6 : scan edge (search edge 3) 4 [547] 547 root98Tail6 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail7 winner := by
  rw [root98Tail6, scan, if_neg (by decide), root98Candidates6_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail7
    (if 3 ≤ root98Candidates6.length then search edge 3 [547, 823] root98Candidates6 1370 winner else winner) = _
  rw [if_pos (by decide), root98Child6]
def root98Candidates7 : List Nat := [3001, 23593]
set_option maxRecDepth 8192 in
theorem root98Candidates7_checked : candidates edge 4 547 853 root98Tail8 winner.bound = root98Candidates7 := by
  unfold candidates
  have row : edge 853 = (SortedNatTree.contains FastRows.row853) := by
    change Data.row853.contains = (SortedNatTree.contains FastRows.row853)
    exact FastRows.row853_checked.symm
  rw [row]
  decide
theorem root98Step7 : scan edge (search edge 3) 4 [547] 547 root98Tail7 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail8 winner := by
  rw [root98Tail7, scan, if_neg (by decide), root98Candidates7_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail8
    (if 3 ≤ root98Candidates7.length then search edge 3 [547, 853] root98Candidates7 1400 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates8 : List Nat := [5557, 6673, 8713, 20011]
set_option maxRecDepth 8192 in
theorem root98Candidates8_checked : candidates edge 4 547 1237 root98Tail9 winner.bound = root98Candidates8 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child8 : search edge 3 [547, 1237] root98Candidates8 1784 winner = winner := by decide
theorem root98Step8 : scan edge (search edge 3) 4 [547] 547 root98Tail8 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail9 winner := by
  rw [root98Tail8, scan, if_neg (by decide), root98Candidates8_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail9
    (if 3 ≤ root98Candidates8.length then search edge 3 [547, 1237] root98Candidates8 1784 winner else winner) = _
  rw [if_pos (by decide), root98Child8]
def root98Candidates9 : List Nat := [3229, 15901, 16231]
set_option maxRecDepth 8192 in
theorem root98Candidates9_checked : candidates edge 4 547 1663 root98Tail10 winner.bound = root98Candidates9 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child9 : search edge 3 [547, 1663] root98Candidates9 2210 winner = winner := by decide
theorem root98Step9 : scan edge (search edge 3) 4 [547] 547 root98Tail9 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail10 winner := by
  rw [root98Tail9, scan, if_neg (by decide), root98Candidates9_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail10
    (if 3 ≤ root98Candidates9.length then search edge 3 [547, 1663] root98Candidates9 2210 winner else winner) = _
  rw [if_pos (by decide), root98Child9]
def root98Candidates10 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root98Candidates10_checked : candidates edge 4 547 1699 root98Tail11 winner.bound = root98Candidates10 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
theorem root98Step10 : scan edge (search edge 3) 4 [547] 547 root98Tail10 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail11 winner := by
  rw [root98Tail10, scan, if_neg (by decide), root98Candidates10_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail11
    (if 3 ≤ root98Candidates10.length then search edge 3 [547, 1699] root98Candidates10 2246 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates11 : List Nat := [3001, 11689, 20509]
set_option maxRecDepth 8192 in
theorem root98Candidates11_checked : candidates edge 4 547 2473 root98Tail12 winner.bound = root98Candidates11 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child11 : search edge 3 [547, 2473] root98Candidates11 3020 winner = winner := by decide
theorem root98Step11 : scan edge (search edge 3) 4 [547] 547 root98Tail11 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail12 winner := by
  rw [root98Tail11, scan, if_neg (by decide), root98Candidates11_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail12
    (if 3 ≤ root98Candidates11.length then search edge 3 [547, 2473] root98Candidates11 3020 winner else winner) = _
  rw [if_pos (by decide), root98Child11]
def root98Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root98Candidates12_checked : candidates edge 4 547 3001 root98Tail13 winner.bound = root98Candidates12 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
theorem root98Step12 : scan edge (search edge 3) 4 [547] 547 root98Tail12 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail13 winner := by
  rw [root98Tail12, scan, if_neg (by decide), root98Candidates12_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail13
    (if 3 ≤ root98Candidates12.length then search edge 3 [547, 3001] root98Candidates12 3548 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates13 : List Nat := [6673, 21649]
set_option maxRecDepth 8192 in
theorem root98Candidates13_checked : candidates edge 4 547 3229 root98Tail14 winner.bound = root98Candidates13 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root98Step13 : scan edge (search edge 3) 4 [547] 547 root98Tail13 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail14 winner := by
  rw [root98Tail13, scan, if_neg (by decide), root98Candidates13_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail14
    (if 3 ≤ root98Candidates13.length then search edge 3 [547, 3229] root98Candidates13 3776 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates14 : List Nat := [9013]
set_option maxRecDepth 8192 in
theorem root98Candidates14_checked : candidates edge 4 547 3271 root98Tail15 winner.bound = root98Candidates14 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root98Step14 : scan edge (search edge 3) 4 [547] 547 root98Tail14 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail15 winner := by
  rw [root98Tail14, scan, if_neg (by decide), root98Candidates14_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail15
    (if 3 ≤ root98Candidates14.length then search edge 3 [547, 3271] root98Candidates14 3818 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates15 : List Nat := [9013, 10321, 13093, 15319, 18289]
set_option maxRecDepth 8192 in
theorem root98Candidates15_checked : candidates edge 4 547 4519 root98Tail16 winner.bound = root98Candidates15 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root98Child15 : search edge 3 [547, 4519] root98Candidates15 5066 winner = winner := by decide
theorem root98Step15 : scan edge (search edge 3) 4 [547] 547 root98Tail15 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail16 winner := by
  rw [root98Tail15, scan, if_neg (by decide), root98Candidates15_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail16
    (if 3 ≤ root98Candidates15.length then search edge 3 [547, 4519] root98Candidates15 5066 winner else winner) = _
  rw [if_pos (by decide), root98Child15]
def root98Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root98Candidates16_checked : candidates edge 4 547 4549 root98Tail17 winner.bound = root98Candidates16 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root98Step16 : scan edge (search edge 3) 4 [547] 547 root98Tail16 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail17 winner := by
  rw [root98Tail16, scan, if_neg (by decide), root98Candidates16_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail17
    (if 3 ≤ root98Candidates16.length then search edge 3 [547, 4549] root98Candidates16 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates17 : List Nat := [15559]
set_option maxRecDepth 8192 in
theorem root98Candidates17_checked : candidates edge 4 547 5443 root98Tail18 winner.bound = root98Candidates17 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root98Step17 : scan edge (search edge 3) 4 [547] 547 root98Tail17 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail18 winner := by
  rw [root98Tail17, scan, if_neg (by decide), root98Candidates17_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail18
    (if 3 ≤ root98Candidates17.length then search edge 3 [547, 5443] root98Candidates17 5990 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root98Candidates18_checked : candidates edge 4 547 5557 root98Tail19 winner.bound = root98Candidates18 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root98Step18 : scan edge (search edge 3) 4 [547] 547 root98Tail18 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail19 winner := by
  rw [root98Tail18, scan, if_neg (by decide), root98Candidates18_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail19
    (if 3 ≤ root98Candidates18.length then search edge 3 [547, 5557] root98Candidates18 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root98Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root98Candidates19_checked : candidates edge 4 547 6073 root98Tail20 winner.bound = root98Candidates19 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root98Step19 : scan edge (search edge 3) 4 [547] 547 root98Tail19 winner =
    scan edge (search edge 3) 4 [547] 547 root98Tail20 winner := by
  rw [root98Tail19, scan, if_neg (by decide), root98Candidates19_checked]
  change scan edge (search edge 3) 4 [547] 547 root98Tail20
    (if 3 ≤ root98Candidates19.length then search edge 3 [547, 6073] root98Candidates19 6620 winner else winner) = _
  rw [if_neg (by decide)]
theorem root98Scan20 : scan edge (search edge 3) 4 [547] 547 root98Tail20 winner = winner := by rw [root98Tail20, scan, if_pos (by decide)]
theorem root98Scan19 : scan edge (search edge 3) 4 [547] 547 root98Tail19 winner = winner :=
  (root98Step19).trans root98Scan20
theorem root98Scan18 : scan edge (search edge 3) 4 [547] 547 root98Tail18 winner = winner :=
  (root98Step18).trans root98Scan19
theorem root98Scan17 : scan edge (search edge 3) 4 [547] 547 root98Tail17 winner = winner :=
  (root98Step17).trans root98Scan18
theorem root98Scan16 : scan edge (search edge 3) 4 [547] 547 root98Tail16 winner = winner :=
  (root98Step16).trans root98Scan17
theorem root98Scan15 : scan edge (search edge 3) 4 [547] 547 root98Tail15 winner = winner :=
  (root98Step15).trans root98Scan16
theorem root98Scan14 : scan edge (search edge 3) 4 [547] 547 root98Tail14 winner = winner :=
  (root98Step14).trans root98Scan15
theorem root98Scan13 : scan edge (search edge 3) 4 [547] 547 root98Tail13 winner = winner :=
  (root98Step13).trans root98Scan14
theorem root98Scan12 : scan edge (search edge 3) 4 [547] 547 root98Tail12 winner = winner :=
  (root98Step12).trans root98Scan13
theorem root98Scan11 : scan edge (search edge 3) 4 [547] 547 root98Tail11 winner = winner :=
  (root98Step11).trans root98Scan12
theorem root98Scan10 : scan edge (search edge 3) 4 [547] 547 root98Tail10 winner = winner :=
  (root98Step10).trans root98Scan11
theorem root98Scan9 : scan edge (search edge 3) 4 [547] 547 root98Tail9 winner = winner :=
  (root98Step9).trans root98Scan10
theorem root98Scan8 : scan edge (search edge 3) 4 [547] 547 root98Tail8 winner = winner :=
  (root98Step8).trans root98Scan9
theorem root98Scan7 : scan edge (search edge 3) 4 [547] 547 root98Tail7 winner = winner :=
  (root98Step7).trans root98Scan8
theorem root98Scan6 : scan edge (search edge 3) 4 [547] 547 root98Tail6 winner = winner :=
  (root98Step6).trans root98Scan7
theorem root98Scan5 : scan edge (search edge 3) 4 [547] 547 root98Tail5 winner = winner :=
  (root98Step5).trans root98Scan6
theorem root98Scan4 : scan edge (search edge 3) 4 [547] 547 root98Tail4 winner = winner :=
  (root98Step4).trans root98Scan5
theorem root98Scan3 : scan edge (search edge 3) 4 [547] 547 root98Tail3 winner = winner :=
  (root98Step3).trans root98Scan4
theorem root98Scan2 : scan edge (search edge 3) 4 [547] 547 root98Tail2 winner = winner :=
  (root98Step2).trans root98Scan3
theorem root98Scan1 : scan edge (search edge 3) 4 [547] 547 root98Tail1 winner = winner :=
  (root98Step1).trans root98Scan2
theorem root98Scan0 : scan edge (search edge 3) 4 [547] 547 root98Tail0 winner = winner :=
  (root98Step0).trans root98Scan1
set_option maxRecDepth 4096 in
theorem root98Branch : search edge 4 [547] root98Candidates 547 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [547] 547 root98Tail0 winner = winner
  exact root98Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
