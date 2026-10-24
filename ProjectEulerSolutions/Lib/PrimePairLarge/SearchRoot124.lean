import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root124Candidates : List Nat := Data.row709
set_option maxRecDepth 8192 in
theorem root124Candidates_checked : candidates edge 5 0 709 tail125 winner.bound = root124Candidates := by
  apply candidates_eq_row edge 5 0 709 tail125 root124Candidates winner.bound tail125_ordered
  · decide
  · decide
  · funext q
    rfl
def root124Tail61 : List Nat := []
def root124Tail60 : List Nat := 24043 :: root124Tail61
def root124Tail59 : List Nat := 23971 :: root124Tail60
def root124Tail58 : List Nat := 23887 :: root124Tail59
def root124Tail57 : List Nat := 23059 :: root124Tail58
def root124Tail56 : List Nat := 22807 :: root124Tail57
def root124Tail55 : List Nat := 22669 :: root124Tail56
def root124Tail54 : List Nat := 22051 :: root124Tail55
def root124Tail53 : List Nat := 22039 :: root124Tail54
def root124Tail52 : List Nat := 20173 :: root124Tail53
def root124Tail51 : List Nat := 20089 :: root124Tail52
def root124Tail50 : List Nat := 19867 :: root124Tail51
def root124Tail49 : List Nat := 19753 :: root124Tail50
def root124Tail48 : List Nat := 19687 :: root124Tail49
def root124Tail47 : List Nat := 19531 :: root124Tail48
def root124Tail46 : List Nat := 19501 :: root124Tail47
def root124Tail45 : List Nat := 19489 :: root124Tail46
def root124Tail44 : List Nat := 18121 :: root124Tail45
def root124Tail43 : List Nat := 17923 :: root124Tail44
def root124Tail42 : List Nat := 17911 :: root124Tail43
def root124Tail41 : List Nat := 17881 :: root124Tail42
def root124Tail40 : List Nat := 17827 :: root124Tail41
def root124Tail39 : List Nat := 14197 :: root124Tail40
def root124Tail38 : List Nat := 12451 :: root124Tail39
def root124Tail37 : List Nat := 12421 :: root124Tail38
def root124Tail36 : List Nat := 11779 :: root124Tail37
def root124Tail35 : List Nat := 11677 :: root124Tail36
def root124Tail34 : List Nat := 11497 :: root124Tail35
def root124Tail33 : List Nat := 10891 :: root124Tail34
def root124Tail32 : List Nat := 10729 :: root124Tail33
def root124Tail31 : List Nat := 10627 :: root124Tail32
def root124Tail30 : List Nat := 10513 :: root124Tail31
def root124Tail29 : List Nat := 10429 :: root124Tail30
def root124Tail28 : List Nat := 9811 :: root124Tail29
def root124Tail27 : List Nat := 9769 :: root124Tail28
def root124Tail26 : List Nat := 9007 :: root124Tail27
def root124Tail25 : List Nat := 8737 :: root124Tail26
def root124Tail24 : List Nat := 8713 :: root124Tail25
def root124Tail23 : List Nat := 8263 :: root124Tail24
def root124Tail22 : List Nat := 8209 :: root124Tail23
def root124Tail21 : List Nat := 7459 :: root124Tail22
def root124Tail20 : List Nat := 7213 :: root124Tail21
def root124Tail19 : List Nat := 6079 :: root124Tail20
def root124Tail18 : List Nat := 5521 :: root124Tail19
def root124Tail17 : List Nat := 4597 :: root124Tail18
def root124Tail16 : List Nat := 4519 :: root124Tail17
def root124Tail15 : List Nat := 4177 :: root124Tail16
def root124Tail14 : List Nat := 3769 :: root124Tail15
def root124Tail13 : List Nat := 3469 :: root124Tail14
def root124Tail12 : List Nat := 3181 :: root124Tail13
def root124Tail11 : List Nat := 3109 :: root124Tail12
def root124Tail10 : List Nat := 3067 :: root124Tail11
def root124Tail9 : List Nat := 2749 :: root124Tail10
def root124Tail8 : List Nat := 2731 :: root124Tail9
def root124Tail7 : List Nat := 2647 :: root124Tail8
def root124Tail6 : List Nat := 1951 :: root124Tail7
def root124Tail5 : List Nat := 1933 :: root124Tail6
def root124Tail4 : List Nat := 1663 :: root124Tail5
def root124Tail3 : List Nat := 1543 :: root124Tail4
def root124Tail2 : List Nat := 1471 :: root124Tail3
def root124Tail1 : List Nat := 967 :: root124Tail2
def root124Tail0 : List Nat := 823 :: root124Tail1
def root124Candidates0 : List Nat := [967, 1933, 19753]
set_option maxRecDepth 8192 in
theorem root124Candidates0_checked : candidates edge 4 709 823 root124Tail1 winner.bound = root124Candidates0 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root124Child0 : search edge 3 [709, 823] root124Candidates0 1532 winner = winner := by decide
theorem root124Step0 : scan edge (search edge 3) 4 [709] 709 root124Tail0 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail1 winner := by
  rw [root124Tail0, scan, if_neg (by decide), root124Candidates0_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail1
    (if 3 ≤ root124Candidates0.length then search edge 3 [709, 823] root124Candidates0 1532 winner else winner) = _
  rw [if_pos (by decide), root124Child0]
def root124Candidates1 : List Nat := [1663, 1951, 19501]
set_option maxRecDepth 8192 in
theorem root124Candidates1_checked : candidates edge 4 709 967 root124Tail2 winner.bound = root124Candidates1 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root124Child1 : search edge 3 [709, 967] root124Candidates1 1676 winner = winner := by decide
theorem root124Step1 : scan edge (search edge 3) 4 [709] 709 root124Tail1 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail2 winner := by
  rw [root124Tail1, scan, if_neg (by decide), root124Candidates1_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail2
    (if 3 ≤ root124Candidates1.length then search edge 3 [709, 967] root124Candidates1 1676 winner else winner) = _
  rw [if_pos (by decide), root124Child1]
def root124Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root124Candidates2_checked : candidates edge 4 709 1471 root124Tail3 winner.bound = root124Candidates2 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
theorem root124Step2 : scan edge (search edge 3) 4 [709] 709 root124Tail2 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail3 winner := by
  rw [root124Tail2, scan, if_neg (by decide), root124Candidates2_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail3
    (if 3 ≤ root124Candidates2.length then search edge 3 [709, 1471] root124Candidates2 2180 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root124Candidates3_checked : candidates edge 4 709 1543 root124Tail4 winner.bound = root124Candidates3 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
theorem root124Step3 : scan edge (search edge 3) 4 [709] 709 root124Tail3 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail4 winner := by
  rw [root124Tail3, scan, if_neg (by decide), root124Candidates3_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail4
    (if 3 ≤ root124Candidates3.length then search edge 3 [709, 1543] root124Candidates3 2252 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates4 : List Nat := [9769]
set_option maxRecDepth 8192 in
theorem root124Candidates4_checked : candidates edge 4 709 1663 root124Tail5 winner.bound = root124Candidates4 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
theorem root124Step4 : scan edge (search edge 3) 4 [709] 709 root124Tail4 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail5 winner := by
  rw [root124Tail4, scan, if_neg (by decide), root124Candidates4_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail5
    (if 3 ≤ root124Candidates4.length then search edge 3 [709, 1663] root124Candidates4 2372 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates5 : List Nat := [10627]
set_option maxRecDepth 8192 in
theorem root124Candidates5_checked : candidates edge 4 709 1933 root124Tail6 winner.bound = root124Candidates5 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root124Step5 : scan edge (search edge 3) 4 [709] 709 root124Tail5 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail6 winner := by
  rw [root124Tail5, scan, if_neg (by decide), root124Candidates5_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail6
    (if 3 ≤ root124Candidates5.length then search edge 3 [709, 1933] root124Candidates5 2642 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates6 : List Nat := [10429]
set_option maxRecDepth 8192 in
theorem root124Candidates6_checked : candidates edge 4 709 1951 root124Tail7 winner.bound = root124Candidates6 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
theorem root124Step6 : scan edge (search edge 3) 4 [709] 709 root124Tail6 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail7 winner := by
  rw [root124Tail6, scan, if_neg (by decide), root124Candidates6_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail7
    (if 3 ≤ root124Candidates6.length then search edge 3 [709, 1951] root124Candidates6 2660 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates7 : List Nat := [8737]
set_option maxRecDepth 8192 in
theorem root124Candidates7_checked : candidates edge 4 709 2647 root124Tail8 winner.bound = root124Candidates7 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root124Step7 : scan edge (search edge 3) 4 [709] 709 root124Tail7 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail8 winner := by
  rw [root124Tail7, scan, if_neg (by decide), root124Candidates7_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail8
    (if 3 ≤ root124Candidates7.length then search edge 3 [709, 2647] root124Candidates7 3356 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates8 : List Nat := [8737, 17881]
set_option maxRecDepth 8192 in
theorem root124Candidates8_checked : candidates edge 4 709 2731 root124Tail9 winner.bound = root124Candidates8 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
theorem root124Step8 : scan edge (search edge 3) 4 [709] 709 root124Tail8 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail9 winner := by
  rw [root124Tail8, scan, if_neg (by decide), root124Candidates8_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail9
    (if 3 ≤ root124Candidates8.length then search edge 3 [709, 2731] root124Candidates8 3440 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates9 : List Nat := [22051]
set_option maxRecDepth 8192 in
theorem root124Candidates9_checked : candidates edge 4 709 2749 root124Tail10 winner.bound = root124Candidates9 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root124Step9 : scan edge (search edge 3) 4 [709] 709 root124Tail9 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail10 winner := by
  rw [root124Tail9, scan, if_neg (by decide), root124Candidates9_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail10
    (if 3 ≤ root124Candidates9.length then search edge 3 [709, 2749] root124Candidates9 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates10 : List Nat := [7459]
set_option maxRecDepth 8192 in
theorem root124Candidates10_checked : candidates edge 4 709 3067 root124Tail11 winner.bound = root124Candidates10 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root124Step10 : scan edge (search edge 3) 4 [709] 709 root124Tail10 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail11 winner := by
  rw [root124Tail10, scan, if_neg (by decide), root124Candidates10_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail11
    (if 3 ≤ root124Candidates10.length then search edge 3 [709, 3067] root124Candidates10 3776 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root124Candidates11_checked : candidates edge 4 709 3109 root124Tail12 winner.bound = root124Candidates11 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root124Step11 : scan edge (search edge 3) 4 [709] 709 root124Tail11 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail12 winner := by
  rw [root124Tail11, scan, if_neg (by decide), root124Candidates11_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail12
    (if 3 ≤ root124Candidates11.length then search edge 3 [709, 3109] root124Candidates11 3818 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates12 : List Nat := [10729, 19687]
set_option maxRecDepth 8192 in
theorem root124Candidates12_checked : candidates edge 4 709 3181 root124Tail13 winner.bound = root124Candidates12 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
theorem root124Step12 : scan edge (search edge 3) 4 [709] 709 root124Tail12 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail13 winner := by
  rw [root124Tail12, scan, if_neg (by decide), root124Candidates12_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail13
    (if 3 ≤ root124Candidates12.length then search edge 3 [709, 3181] root124Candidates12 3890 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root124Candidates13_checked : candidates edge 4 709 3469 root124Tail14 winner.bound = root124Candidates13 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root124Step13 : scan edge (search edge 3) 4 [709] 709 root124Tail13 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail14 winner := by
  rw [root124Tail13, scan, if_neg (by decide), root124Candidates13_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail14
    (if 3 ≤ root124Candidates13.length then search edge 3 [709, 3469] root124Candidates13 4178 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates14 : List Nat := [4597, 19687]
set_option maxRecDepth 8192 in
theorem root124Candidates14_checked : candidates edge 4 709 3769 root124Tail15 winner.bound = root124Candidates14 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root124Step14 : scan edge (search edge 3) 4 [709] 709 root124Tail14 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail15 winner := by
  rw [root124Tail14, scan, if_neg (by decide), root124Candidates14_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail15
    (if 3 ≤ root124Candidates14.length then search edge 3 [709, 3769] root124Candidates14 4478 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root124Candidates15_checked : candidates edge 4 709 4177 root124Tail16 winner.bound = root124Candidates15 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root124Step15 : scan edge (search edge 3) 4 [709] 709 root124Tail15 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail16 winner := by
  rw [root124Tail15, scan, if_neg (by decide), root124Candidates15_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail16
    (if 3 ≤ root124Candidates15.length then search edge 3 [709, 4177] root124Candidates15 4886 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates16 : List Nat := [8737]
set_option maxRecDepth 8192 in
theorem root124Candidates16_checked : candidates edge 4 709 4519 root124Tail17 winner.bound = root124Candidates16 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root124Step16 : scan edge (search edge 3) 4 [709] 709 root124Tail16 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail17 winner := by
  rw [root124Tail16, scan, if_neg (by decide), root124Candidates16_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail17
    (if 3 ≤ root124Candidates16.length then search edge 3 [709, 4519] root124Candidates16 5228 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates17 : List Nat := [6079, 7213, 8263, 11779, 14197, 20089]
set_option maxRecDepth 8192 in
theorem root124Candidates17_checked : candidates edge 4 709 4597 root124Tail18 winner.bound = root124Candidates17 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root124Child17 : search edge 3 [709, 4597] root124Candidates17 5306 winner = winner := by decide
theorem root124Step17 : scan edge (search edge 3) 4 [709] 709 root124Tail17 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail18 winner := by
  rw [root124Tail17, scan, if_neg (by decide), root124Candidates17_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail18
    (if 3 ≤ root124Candidates17.length then search edge 3 [709, 4597] root124Candidates17 5306 winner else winner) = _
  rw [if_pos (by decide), root124Child17]
def root124Candidates18 : List Nat := [19753]
set_option maxRecDepth 8192 in
theorem root124Candidates18_checked : candidates edge 4 709 5521 root124Tail19 winner.bound = root124Candidates18 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root124Step18 : scan edge (search edge 3) 4 [709] 709 root124Tail18 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail19 winner := by
  rw [root124Tail18, scan, if_neg (by decide), root124Candidates18_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail19
    (if 3 ≤ root124Candidates18.length then search edge 3 [709, 5521] root124Candidates18 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root124Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root124Candidates19_checked : candidates edge 4 709 6079 root124Tail20 winner.bound = root124Candidates19 := by
  unfold candidates
  have row : edge 6079 = (SortedNatTree.contains FastRows.row6079) := by
    change Data.row6079.contains = (SortedNatTree.contains FastRows.row6079)
    exact FastRows.row6079_checked.symm
  rw [row]
  decide
theorem root124Step19 : scan edge (search edge 3) 4 [709] 709 root124Tail19 winner =
    scan edge (search edge 3) 4 [709] 709 root124Tail20 winner := by
  rw [root124Tail19, scan, if_neg (by decide), root124Candidates19_checked]
  change scan edge (search edge 3) 4 [709] 709 root124Tail20
    (if 3 ≤ root124Candidates19.length then search edge 3 [709, 6079] root124Candidates19 6788 winner else winner) = _
  rw [if_neg (by decide)]
theorem root124Scan20 : scan edge (search edge 3) 4 [709] 709 root124Tail20 winner = winner := by rw [root124Tail20, scan, if_pos (by decide)]
theorem root124Scan19 : scan edge (search edge 3) 4 [709] 709 root124Tail19 winner = winner :=
  (root124Step19).trans root124Scan20
theorem root124Scan18 : scan edge (search edge 3) 4 [709] 709 root124Tail18 winner = winner :=
  (root124Step18).trans root124Scan19
theorem root124Scan17 : scan edge (search edge 3) 4 [709] 709 root124Tail17 winner = winner :=
  (root124Step17).trans root124Scan18
theorem root124Scan16 : scan edge (search edge 3) 4 [709] 709 root124Tail16 winner = winner :=
  (root124Step16).trans root124Scan17
theorem root124Scan15 : scan edge (search edge 3) 4 [709] 709 root124Tail15 winner = winner :=
  (root124Step15).trans root124Scan16
theorem root124Scan14 : scan edge (search edge 3) 4 [709] 709 root124Tail14 winner = winner :=
  (root124Step14).trans root124Scan15
theorem root124Scan13 : scan edge (search edge 3) 4 [709] 709 root124Tail13 winner = winner :=
  (root124Step13).trans root124Scan14
theorem root124Scan12 : scan edge (search edge 3) 4 [709] 709 root124Tail12 winner = winner :=
  (root124Step12).trans root124Scan13
theorem root124Scan11 : scan edge (search edge 3) 4 [709] 709 root124Tail11 winner = winner :=
  (root124Step11).trans root124Scan12
theorem root124Scan10 : scan edge (search edge 3) 4 [709] 709 root124Tail10 winner = winner :=
  (root124Step10).trans root124Scan11
theorem root124Scan9 : scan edge (search edge 3) 4 [709] 709 root124Tail9 winner = winner :=
  (root124Step9).trans root124Scan10
theorem root124Scan8 : scan edge (search edge 3) 4 [709] 709 root124Tail8 winner = winner :=
  (root124Step8).trans root124Scan9
theorem root124Scan7 : scan edge (search edge 3) 4 [709] 709 root124Tail7 winner = winner :=
  (root124Step7).trans root124Scan8
theorem root124Scan6 : scan edge (search edge 3) 4 [709] 709 root124Tail6 winner = winner :=
  (root124Step6).trans root124Scan7
theorem root124Scan5 : scan edge (search edge 3) 4 [709] 709 root124Tail5 winner = winner :=
  (root124Step5).trans root124Scan6
theorem root124Scan4 : scan edge (search edge 3) 4 [709] 709 root124Tail4 winner = winner :=
  (root124Step4).trans root124Scan5
theorem root124Scan3 : scan edge (search edge 3) 4 [709] 709 root124Tail3 winner = winner :=
  (root124Step3).trans root124Scan4
theorem root124Scan2 : scan edge (search edge 3) 4 [709] 709 root124Tail2 winner = winner :=
  (root124Step2).trans root124Scan3
theorem root124Scan1 : scan edge (search edge 3) 4 [709] 709 root124Tail1 winner = winner :=
  (root124Step1).trans root124Scan2
theorem root124Scan0 : scan edge (search edge 3) 4 [709] 709 root124Tail0 winner = winner :=
  (root124Step0).trans root124Scan1
set_option maxRecDepth 4096 in
theorem root124Branch : search edge 4 [709] root124Candidates 709 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [709] 709 root124Tail0 winner = winner
  exact root124Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
