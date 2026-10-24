import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root35Candidates : List Nat := Data.row163
set_option maxRecDepth 8192 in
theorem root35Candidates_checked : candidates edge 5 0 163 tail36 winner.bound = root35Candidates := by
  apply candidates_eq_row edge 5 0 163 tail36 root35Candidates winner.bound tail36_ordered
  · decide
  · decide
  · funext q
    rfl
def root35Tail67 : List Nat := []
def root35Tail66 : List Nat := 25303 :: root35Tail67
def root35Tail65 : List Nat := 24877 :: root35Tail66
def root35Tail64 : List Nat := 24793 :: root35Tail65
def root35Tail63 : List Nat := 24151 :: root35Tail64
def root35Tail62 : List Nat := 23971 :: root35Tail63
def root35Tail61 : List Nat := 23917 :: root35Tail62
def root35Tail60 : List Nat := 23677 :: root35Tail61
def root35Tail59 : List Nat := 22303 :: root35Tail60
def root35Tail58 : List Nat := 21787 :: root35Tail59
def root35Tail57 : List Nat := 20809 :: root35Tail58
def root35Tail56 : List Nat := 20023 :: root35Tail57
def root35Tail55 : List Nat := 19867 :: root35Tail56
def root35Tail54 : List Nat := 19801 :: root35Tail55
def root35Tail53 : List Nat := 19759 :: root35Tail54
def root35Tail52 : List Nat := 19417 :: root35Tail53
def root35Tail51 : List Nat := 18553 :: root35Tail52
def root35Tail50 : List Nat := 17839 :: root35Tail51
def root35Tail49 : List Nat := 17293 :: root35Tail50
def root35Tail48 : List Nat := 16903 :: root35Tail49
def root35Tail47 : List Nat := 16759 :: root35Tail48
def root35Tail46 : List Nat := 16747 :: root35Tail47
def root35Tail45 : List Nat := 16411 :: root35Tail46
def root35Tail44 : List Nat := 16369 :: root35Tail45
def root35Tail43 : List Nat := 16087 :: root35Tail44
def root35Tail42 : List Nat := 15973 :: root35Tail43
def root35Tail41 : List Nat := 15601 :: root35Tail42
def root35Tail40 : List Nat := 13903 :: root35Tail41
def root35Tail39 : List Nat := 13681 :: root35Tail40
def root35Tail38 : List Nat := 12739 :: root35Tail39
def root35Tail37 : List Nat := 12421 :: root35Tail38
def root35Tail36 : List Nat := 12391 :: root35Tail37
def root35Tail35 : List Nat := 11821 :: root35Tail36
def root35Tail34 : List Nat := 11731 :: root35Tail35
def root35Tail33 : List Nat := 11719 :: root35Tail34
def root35Tail32 : List Nat := 11149 :: root35Tail33
def root35Tail31 : List Nat := 10891 :: root35Tail32
def root35Tail30 : List Nat := 9829 :: root35Tail31
def root35Tail29 : List Nat := 9199 :: root35Tail30
def root35Tail28 : List Nat := 8929 :: root35Tail29
def root35Tail27 : List Nat := 8719 :: root35Tail28
def root35Tail26 : List Nat := 8599 :: root35Tail27
def root35Tail25 : List Nat := 7927 :: root35Tail26
def root35Tail24 : List Nat := 7243 :: root35Tail25
def root35Tail23 : List Nat := 6997 :: root35Tail24
def root35Tail22 : List Nat := 6823 :: root35Tail23
def root35Tail21 : List Nat := 6781 :: root35Tail22
def root35Tail20 : List Nat := 6577 :: root35Tail21
def root35Tail19 : List Nat := 6343 :: root35Tail20
def root35Tail18 : List Nat := 5563 :: root35Tail19
def root35Tail17 : List Nat := 4597 :: root35Tail18
def root35Tail16 : List Nat := 4441 :: root35Tail17
def root35Tail15 : List Nat := 4177 :: root35Tail16
def root35Tail14 : List Nat := 3823 :: root35Tail15
def root35Tail13 : List Nat := 2647 :: root35Tail14
def root35Tail12 : List Nat := 2383 :: root35Tail13
def root35Tail11 : List Nat := 2311 :: root35Tail12
def root35Tail10 : List Nat := 1783 :: root35Tail11
def root35Tail9 : List Nat := 1741 :: root35Tail10
def root35Tail8 : List Nat := 1579 :: root35Tail9
def root35Tail7 : List Nat := 997 :: root35Tail8
def root35Tail6 : List Nat := 883 :: root35Tail7
def root35Tail5 : List Nat := 811 :: root35Tail6
def root35Tail4 : List Nat := 613 :: root35Tail5
def root35Tail3 : List Nat := 409 :: root35Tail4
def root35Tail2 : List Nat := 367 :: root35Tail3
def root35Tail1 : List Nat := 307 :: root35Tail2
def root35Tail0 : List Nat := 193 :: root35Tail1
def root35Candidates0 : List Nat := [811, 883, 21787, 23677]
set_option maxRecDepth 8192 in
theorem root35Candidates0_checked : candidates edge 4 163 193 root35Tail1 winner.bound = root35Candidates0 := by
  unfold candidates
  have row : edge 193 = (SortedNatTree.contains FastRows.row193) := by
    change Data.row193.contains = (SortedNatTree.contains FastRows.row193)
    exact FastRows.row193_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child0 : search edge 3 [163, 193] root35Candidates0 356 winner = winner := by decide
theorem root35Step0 : scan edge (search edge 3) 4 [163] 163 root35Tail0 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail1 winner := by
  rw [root35Tail0, scan, if_neg (by decide), root35Candidates0_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail1
    (if 3 ≤ root35Candidates0.length then search edge 3 [163, 193] root35Candidates0 356 winner else winner) = _
  rw [if_pos (by decide), root35Child0]
def root35Candidates1 : List Nat := [367, 7927, 20023, 21787, 23971]
set_option maxRecDepth 8192 in
theorem root35Candidates1_checked : candidates edge 4 163 307 root35Tail2 winner.bound = root35Candidates1 := by
  unfold candidates
  have row : edge 307 = (SortedNatTree.contains FastRows.row307) := by
    change Data.row307.contains = (SortedNatTree.contains FastRows.row307)
    exact FastRows.row307_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child1 : search edge 3 [163, 307] root35Candidates1 470 winner = winner := by decide
theorem root35Step1 : scan edge (search edge 3) 4 [163] 163 root35Tail1 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail2 winner := by
  rw [root35Tail1, scan, if_neg (by decide), root35Candidates1_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail2
    (if 3 ≤ root35Candidates1.length then search edge 3 [163, 307] root35Candidates1 470 winner else winner) = _
  rw [if_pos (by decide), root35Child1]
def root35Candidates2 : List Nat := [613, 4597, 5563, 8929, 19759, 25303]
set_option maxRecDepth 8192 in
theorem root35Candidates2_checked : candidates edge 4 163 367 root35Tail3 winner.bound = root35Candidates2 := by
  unfold candidates
  have row : edge 367 = (SortedNatTree.contains FastRows.row367) := by
    change Data.row367.contains = (SortedNatTree.contains FastRows.row367)
    exact FastRows.row367_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child2 : search edge 3 [163, 367] root35Candidates2 530 winner = winner := by decide
theorem root35Step2 : scan edge (search edge 3) 4 [163] 163 root35Tail2 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail3 winner := by
  rw [root35Tail2, scan, if_neg (by decide), root35Candidates2_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail3
    (if 3 ≤ root35Candidates2.length then search edge 3 [163, 367] root35Candidates2 530 winner else winner) = _
  rw [if_pos (by decide), root35Child2]
def root35Candidates3 : List Nat := [1741, 22303]
set_option maxRecDepth 8192 in
theorem root35Candidates3_checked : candidates edge 4 163 409 root35Tail4 winner.bound = root35Candidates3 := by
  unfold candidates
  have row : edge 409 = (SortedNatTree.contains FastRows.row409) := by
    change Data.row409.contains = (SortedNatTree.contains FastRows.row409)
    exact FastRows.row409_checked.symm
  rw [row]
  decide
theorem root35Step3 : scan edge (search edge 3) 4 [163] 163 root35Tail3 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail4 winner := by
  rw [root35Tail3, scan, if_neg (by decide), root35Candidates3_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail4
    (if 3 ≤ root35Candidates3.length then search edge 3 [163, 409] root35Candidates3 572 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates4 : List Nat := [883, 15973, 16747, 19759]
set_option maxRecDepth 8192 in
theorem root35Candidates4_checked : candidates edge 4 163 613 root35Tail5 winner.bound = root35Candidates4 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child4 : search edge 3 [163, 613] root35Candidates4 776 winner = winner := by decide
theorem root35Step4 : scan edge (search edge 3) 4 [163] 163 root35Tail4 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail5 winner := by
  rw [root35Tail4, scan, if_neg (by decide), root35Candidates4_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail5
    (if 3 ≤ root35Candidates4.length then search edge 3 [163, 613] root35Candidates4 776 winner else winner) = _
  rw [if_pos (by decide), root35Child4]
def root35Candidates5 : List Nat := [997]
set_option maxRecDepth 8192 in
theorem root35Candidates5_checked : candidates edge 4 163 811 root35Tail6 winner.bound = root35Candidates5 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
theorem root35Step5 : scan edge (search edge 3) 4 [163] 163 root35Tail5 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail6 winner := by
  rw [root35Tail5, scan, if_neg (by decide), root35Candidates5_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail6
    (if 3 ≤ root35Candidates5.length then search edge 3 [163, 811] root35Candidates5 974 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates6 : List Nat := [1783, 15973, 17293, 19801]
set_option maxRecDepth 8192 in
theorem root35Candidates6_checked : candidates edge 4 163 883 root35Tail7 winner.bound = root35Candidates6 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child6 : search edge 3 [163, 883] root35Candidates6 1046 winner = winner := by decide
theorem root35Step6 : scan edge (search edge 3) 4 [163] 163 root35Tail6 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail7 winner := by
  rw [root35Tail6, scan, if_neg (by decide), root35Candidates6_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail7
    (if 3 ≤ root35Candidates6.length then search edge 3 [163, 883] root35Candidates6 1046 winner else winner) = _
  rw [if_pos (by decide), root35Child6]
def root35Candidates7 : List Nat := [4177, 16087]
set_option maxRecDepth 8192 in
theorem root35Candidates7_checked : candidates edge 4 163 997 root35Tail8 winner.bound = root35Candidates7 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
theorem root35Step7 : scan edge (search edge 3) 4 [163] 163 root35Tail7 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail8 winner := by
  rw [root35Tail7, scan, if_neg (by decide), root35Candidates7_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail8
    (if 3 ≤ root35Candidates7.length then search edge 3 [163, 997] root35Candidates7 1160 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root35Candidates8_checked : candidates edge 4 163 1579 root35Tail9 winner.bound = root35Candidates8 := by
  unfold candidates
  have row : edge 1579 = (SortedNatTree.contains FastRows.row1579) := by
    change Data.row1579.contains = (SortedNatTree.contains FastRows.row1579)
    exact FastRows.row1579_checked.symm
  rw [row]
  decide
theorem root35Step8 : scan edge (search edge 3) 4 [163] 163 root35Tail8 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail9 winner := by
  rw [root35Tail8, scan, if_neg (by decide), root35Candidates8_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail9
    (if 3 ≤ root35Candidates8.length then search edge 3 [163, 1579] root35Candidates8 1742 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root35Candidates9_checked : candidates edge 4 163 1741 root35Tail10 winner.bound = root35Candidates9 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
theorem root35Step9 : scan edge (search edge 3) 4 [163] 163 root35Tail9 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail10 winner := by
  rw [root35Tail9, scan, if_neg (by decide), root35Candidates9_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail10
    (if 3 ≤ root35Candidates9.length then search edge 3 [163, 1741] root35Candidates9 1904 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates10 : List Nat := [6823, 11719, 12391, 16903]
set_option maxRecDepth 8192 in
theorem root35Candidates10_checked : candidates edge 4 163 1783 root35Tail11 winner.bound = root35Candidates10 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child10 : search edge 3 [163, 1783] root35Candidates10 1946 winner = winner := by decide
theorem root35Step10 : scan edge (search edge 3) 4 [163] 163 root35Tail10 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail11 winner := by
  rw [root35Tail10, scan, if_neg (by decide), root35Candidates10_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail11
    (if 3 ≤ root35Candidates10.length then search edge 3 [163, 1783] root35Candidates10 1946 winner else winner) = _
  rw [if_pos (by decide), root35Child10]
def root35Candidates11 : List Nat := [2647, 19417, 20023]
set_option maxRecDepth 8192 in
theorem root35Candidates11_checked : candidates edge 4 163 2311 root35Tail12 winner.bound = root35Candidates11 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child11 : search edge 3 [163, 2311] root35Candidates11 2474 winner = winner := by decide
theorem root35Step11 : scan edge (search edge 3) 4 [163] 163 root35Tail11 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail12 winner := by
  rw [root35Tail11, scan, if_neg (by decide), root35Candidates11_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail12
    (if 3 ≤ root35Candidates11.length then search edge 3 [163, 2311] root35Candidates11 2474 winner else winner) = _
  rw [if_pos (by decide), root35Child11]
def root35Candidates12 : List Nat := [8599, 13681, 16369, 19759]
set_option maxRecDepth 8192 in
theorem root35Candidates12_checked : candidates edge 4 163 2383 root35Tail13 winner.bound = root35Candidates12 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child12 : search edge 3 [163, 2383] root35Candidates12 2546 winner = winner := by decide
theorem root35Step12 : scan edge (search edge 3) 4 [163] 163 root35Tail12 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail13 winner := by
  rw [root35Tail12, scan, if_neg (by decide), root35Candidates12_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail13
    (if 3 ≤ root35Candidates12.length then search edge 3 [163, 2383] root35Candidates12 2546 winner else winner) = _
  rw [if_pos (by decide), root35Child12]
def root35Candidates13 : List Nat := [6577, 11731, 12391, 12739, 17293]
set_option maxRecDepth 8192 in
theorem root35Candidates13_checked : candidates edge 4 163 2647 root35Tail14 winner.bound = root35Candidates13 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root35Child13 : search edge 3 [163, 2647] root35Candidates13 2810 winner = winner := by decide
theorem root35Step13 : scan edge (search edge 3) 4 [163] 163 root35Tail13 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail14 winner := by
  rw [root35Tail13, scan, if_neg (by decide), root35Candidates13_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail14
    (if 3 ≤ root35Candidates13.length then search edge 3 [163, 2647] root35Candidates13 2810 winner else winner) = _
  rw [if_pos (by decide), root35Child13]
def root35Candidates14 : List Nat := [16369]
set_option maxRecDepth 8192 in
theorem root35Candidates14_checked : candidates edge 4 163 3823 root35Tail15 winner.bound = root35Candidates14 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root35Step14 : scan edge (search edge 3) 4 [163] 163 root35Tail14 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail15 winner := by
  rw [root35Tail14, scan, if_neg (by decide), root35Candidates14_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail15
    (if 3 ≤ root35Candidates14.length then search edge 3 [163, 3823] root35Candidates14 3986 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates15 : List Nat := [6781]
set_option maxRecDepth 8192 in
theorem root35Candidates15_checked : candidates edge 4 163 4177 root35Tail16 winner.bound = root35Candidates15 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root35Step15 : scan edge (search edge 3) 4 [163] 163 root35Tail15 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail16 winner := by
  rw [root35Tail15, scan, if_neg (by decide), root35Candidates15_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail16
    (if 3 ≤ root35Candidates15.length then search edge 3 [163, 4177] root35Candidates15 4340 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root35Candidates16_checked : candidates edge 4 163 4441 root35Tail17 winner.bound = root35Candidates16 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root35Step16 : scan edge (search edge 3) 4 [163] 163 root35Tail16 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail17 winner := by
  rw [root35Tail16, scan, if_neg (by decide), root35Candidates16_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail17
    (if 3 ≤ root35Candidates16.length then search edge 3 [163, 4441] root35Candidates16 4604 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates17 : List Nat := [5563, 19417]
set_option maxRecDepth 8192 in
theorem root35Candidates17_checked : candidates edge 4 163 4597 root35Tail18 winner.bound = root35Candidates17 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root35Step17 : scan edge (search edge 3) 4 [163] 163 root35Tail17 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail18 winner := by
  rw [root35Tail17, scan, if_neg (by decide), root35Candidates17_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail18
    (if 3 ≤ root35Candidates17.length then search edge 3 [163, 4597] root35Candidates17 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates18 : List Nat := [6781]
set_option maxRecDepth 8192 in
theorem root35Candidates18_checked : candidates edge 4 163 5563 root35Tail19 winner.bound = root35Candidates18 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root35Step18 : scan edge (search edge 3) 4 [163] 163 root35Tail18 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail19 winner := by
  rw [root35Tail18, scan, if_neg (by decide), root35Candidates18_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail19
    (if 3 ≤ root35Candidates18.length then search edge 3 [163, 5563] root35Candidates18 5726 winner else winner) = _
  rw [if_neg (by decide)]
def root35Candidates19 : List Nat := [17839]
set_option maxRecDepth 8192 in
theorem root35Candidates19_checked : candidates edge 4 163 6343 root35Tail20 winner.bound = root35Candidates19 := by
  unfold candidates
  have row : edge 6343 = (SortedNatTree.contains FastRows.row6343) := by
    change Data.row6343.contains = (SortedNatTree.contains FastRows.row6343)
    exact FastRows.row6343_checked.symm
  rw [row]
  decide
theorem root35Step19 : scan edge (search edge 3) 4 [163] 163 root35Tail19 winner =
    scan edge (search edge 3) 4 [163] 163 root35Tail20 winner := by
  rw [root35Tail19, scan, if_neg (by decide), root35Candidates19_checked]
  change scan edge (search edge 3) 4 [163] 163 root35Tail20
    (if 3 ≤ root35Candidates19.length then search edge 3 [163, 6343] root35Candidates19 6506 winner else winner) = _
  rw [if_neg (by decide)]
theorem root35Scan20 : scan edge (search edge 3) 4 [163] 163 root35Tail20 winner = winner := by rw [root35Tail20, scan, if_pos (by decide)]
theorem root35Scan19 : scan edge (search edge 3) 4 [163] 163 root35Tail19 winner = winner :=
  (root35Step19).trans root35Scan20
theorem root35Scan18 : scan edge (search edge 3) 4 [163] 163 root35Tail18 winner = winner :=
  (root35Step18).trans root35Scan19
theorem root35Scan17 : scan edge (search edge 3) 4 [163] 163 root35Tail17 winner = winner :=
  (root35Step17).trans root35Scan18
theorem root35Scan16 : scan edge (search edge 3) 4 [163] 163 root35Tail16 winner = winner :=
  (root35Step16).trans root35Scan17
theorem root35Scan15 : scan edge (search edge 3) 4 [163] 163 root35Tail15 winner = winner :=
  (root35Step15).trans root35Scan16
theorem root35Scan14 : scan edge (search edge 3) 4 [163] 163 root35Tail14 winner = winner :=
  (root35Step14).trans root35Scan15
theorem root35Scan13 : scan edge (search edge 3) 4 [163] 163 root35Tail13 winner = winner :=
  (root35Step13).trans root35Scan14
theorem root35Scan12 : scan edge (search edge 3) 4 [163] 163 root35Tail12 winner = winner :=
  (root35Step12).trans root35Scan13
theorem root35Scan11 : scan edge (search edge 3) 4 [163] 163 root35Tail11 winner = winner :=
  (root35Step11).trans root35Scan12
theorem root35Scan10 : scan edge (search edge 3) 4 [163] 163 root35Tail10 winner = winner :=
  (root35Step10).trans root35Scan11
theorem root35Scan9 : scan edge (search edge 3) 4 [163] 163 root35Tail9 winner = winner :=
  (root35Step9).trans root35Scan10
theorem root35Scan8 : scan edge (search edge 3) 4 [163] 163 root35Tail8 winner = winner :=
  (root35Step8).trans root35Scan9
theorem root35Scan7 : scan edge (search edge 3) 4 [163] 163 root35Tail7 winner = winner :=
  (root35Step7).trans root35Scan8
theorem root35Scan6 : scan edge (search edge 3) 4 [163] 163 root35Tail6 winner = winner :=
  (root35Step6).trans root35Scan7
theorem root35Scan5 : scan edge (search edge 3) 4 [163] 163 root35Tail5 winner = winner :=
  (root35Step5).trans root35Scan6
theorem root35Scan4 : scan edge (search edge 3) 4 [163] 163 root35Tail4 winner = winner :=
  (root35Step4).trans root35Scan5
theorem root35Scan3 : scan edge (search edge 3) 4 [163] 163 root35Tail3 winner = winner :=
  (root35Step3).trans root35Scan4
theorem root35Scan2 : scan edge (search edge 3) 4 [163] 163 root35Tail2 winner = winner :=
  (root35Step2).trans root35Scan3
theorem root35Scan1 : scan edge (search edge 3) 4 [163] 163 root35Tail1 winner = winner :=
  (root35Step1).trans root35Scan2
theorem root35Scan0 : scan edge (search edge 3) 4 [163] 163 root35Tail0 winner = winner :=
  (root35Step0).trans root35Scan1
set_option maxRecDepth 4096 in
theorem root35Branch : search edge 4 [163] root35Candidates 163 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [163] 163 root35Tail0 winner = winner
  exact root35Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
