import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root60Candidates : List Nat := Data.row307
set_option maxRecDepth 8192 in
theorem root60Candidates_checked : candidates edge 5 0 307 tail61 winner.bound = root60Candidates := by
  apply candidates_eq_row edge 5 0 307 tail61 root60Candidates winner.bound tail61_ordered
  · decide
  · decide
  · funext q
    rfl
def root60Tail65 : List Nat := []
def root60Tail64 : List Nat := 25411 :: root60Tail65
def root60Tail63 : List Nat := 24103 :: root60Tail64
def root60Tail62 : List Nat := 23977 :: root60Tail63
def root60Tail61 : List Nat := 23971 :: root60Tail62
def root60Tail60 : List Nat := 23761 :: root60Tail61
def root60Tail59 : List Nat := 23059 :: root60Tail60
def root60Tail58 : List Nat := 22171 :: root60Tail59
def root60Tail57 : List Nat := 22093 :: root60Tail58
def root60Tail56 : List Nat := 22051 :: root60Tail57
def root60Tail55 : List Nat := 21787 :: root60Tail56
def root60Tail54 : List Nat := 20983 :: root60Tail55
def root60Tail53 : List Nat := 20959 :: root60Tail54
def root60Tail52 : List Nat := 20023 :: root60Tail53
def root60Tail51 : List Nat := 19477 :: root60Tail52
def root60Tail50 : List Nat := 18313 :: root60Tail51
def root60Tail49 : List Nat := 18223 :: root60Tail50
def root60Tail48 : List Nat := 18097 :: root60Tail49
def root60Tail47 : List Nat := 17923 :: root60Tail48
def root60Tail46 : List Nat := 17257 :: root60Tail47
def root60Tail45 : List Nat := 15877 :: root60Tail46
def root60Tail44 : List Nat := 15859 :: root60Tail45
def root60Tail43 : List Nat := 14533 :: root60Tail44
def root60Tail42 : List Nat := 14323 :: root60Tail43
def root60Tail41 : List Nat := 13297 :: root60Tail42
def root60Tail40 : List Nat := 12853 :: root60Tail41
def root60Tail39 : List Nat := 12601 :: root60Tail40
def root60Tail38 : List Nat := 12589 :: root60Tail39
def root60Tail37 : List Nat := 12547 :: root60Tail38
def root60Tail36 : List Nat := 12373 :: root60Tail37
def root60Tail35 : List Nat := 12211 :: root60Tail36
def root60Tail34 : List Nat := 11491 :: root60Tail35
def root60Tail33 : List Nat := 10861 :: root60Tail34
def root60Tail32 : List Nat := 10627 :: root60Tail33
def root60Tail31 : List Nat := 10453 :: root60Tail32
def root60Tail30 : List Nat := 9871 :: root60Tail31
def root60Tail29 : List Nat := 9649 :: root60Tail30
def root60Tail28 : List Nat := 9619 :: root60Tail29
def root60Tail27 : List Nat := 9319 :: root60Tail28
def root60Tail26 : List Nat := 8839 :: root60Tail27
def root60Tail25 : List Nat := 8563 :: root60Tail26
def root60Tail24 : List Nat := 8059 :: root60Tail25
def root60Tail23 : List Nat := 7927 :: root60Tail24
def root60Tail22 : List Nat := 7681 :: root60Tail23
def root60Tail21 : List Nat := 7159 :: root60Tail22
def root60Tail20 : List Nat := 6679 :: root60Tail21
def root60Tail19 : List Nat := 6373 :: root60Tail20
def root60Tail18 : List Nat := 5857 :: root60Tail19
def root60Tail17 : List Nat := 5209 :: root60Tail18
def root60Tail16 : List Nat := 5119 :: root60Tail17
def root60Tail15 : List Nat := 4243 :: root60Tail16
def root60Tail14 : List Nat := 3943 :: root60Tail15
def root60Tail13 : List Nat := 3547 :: root60Tail14
def root60Tail12 : List Nat := 2467 :: root60Tail13
def root60Tail11 : List Nat := 2269 :: root60Tail12
def root60Tail10 : List Nat := 2137 :: root60Tail11
def root60Tail9 : List Nat := 1993 :: root60Tail10
def root60Tail8 : List Nat := 1381 :: root60Tail9
def root60Tail7 : List Nat := 1279 :: root60Tail8
def root60Tail6 : List Nat := 1093 :: root60Tail7
def root60Tail5 : List Nat := 1069 :: root60Tail6
def root60Tail4 : List Nat := 733 :: root60Tail5
def root60Tail3 : List Nat := 631 :: root60Tail4
def root60Tail2 : List Nat := 577 :: root60Tail3
def root60Tail1 : List Nat := 523 :: root60Tail2
def root60Tail0 : List Nat := 367 :: root60Tail1
def root60Candidates0 : List Nat := [8839, 10627, 13297, 15859, 23059]
set_option maxRecDepth 8192 in
theorem root60Candidates0_checked : candidates edge 4 307 367 root60Tail1 winner.bound = root60Candidates0 := by
  unfold candidates
  have row : edge 367 = (SortedNatTree.contains FastRows.row367) := by
    change Data.row367.contains = (SortedNatTree.contains FastRows.row367)
    exact FastRows.row367_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root60Child0 : search edge 3 [307, 367] root60Candidates0 674 winner = winner := by decide
theorem root60Step0 : scan edge (search edge 3) 4 [307] 307 root60Tail0 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail1 winner := by
  rw [root60Tail0, scan, if_neg (by decide), root60Candidates0_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail1
    (if 3 ≤ root60Candidates0.length then search edge 3 [307, 367] root60Candidates0 674 winner else winner) = _
  rw [if_pos (by decide), root60Child0]
def root60Candidates1 : List Nat := [577]
set_option maxRecDepth 8192 in
theorem root60Candidates1_checked : candidates edge 4 307 523 root60Tail2 winner.bound = root60Candidates1 := by
  unfold candidates
  have row : edge 523 = (SortedNatTree.contains FastRows.row523) := by
    change Data.row523.contains = (SortedNatTree.contains FastRows.row523)
    exact FastRows.row523_checked.symm
  rw [row]
  decide
theorem root60Step1 : scan edge (search edge 3) 4 [307] 307 root60Tail1 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail2 winner := by
  rw [root60Tail1, scan, if_neg (by decide), root60Candidates1_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail2
    (if 3 ≤ root60Candidates1.length then search edge 3 [307, 523] root60Candidates1 830 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates2 : List Nat := [14323, 18097]
set_option maxRecDepth 8192 in
theorem root60Candidates2_checked : candidates edge 4 307 577 root60Tail3 winner.bound = root60Candidates2 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
theorem root60Step2 : scan edge (search edge 3) 4 [307] 307 root60Tail2 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail3 winner := by
  rw [root60Tail2, scan, if_neg (by decide), root60Candidates2_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail3
    (if 3 ≤ root60Candidates2.length then search edge 3 [307, 577] root60Candidates2 884 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates3 : List Nat := [7681]
set_option maxRecDepth 8192 in
theorem root60Candidates3_checked : candidates edge 4 307 631 root60Tail4 winner.bound = root60Candidates3 := by
  unfold candidates
  have row : edge 631 = (SortedNatTree.contains FastRows.row631) := by
    change Data.row631.contains = (SortedNatTree.contains FastRows.row631)
    exact FastRows.row631_checked.symm
  rw [row]
  decide
theorem root60Step3 : scan edge (search edge 3) 4 [307] 307 root60Tail3 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail4 winner := by
  rw [root60Tail3, scan, if_neg (by decide), root60Candidates3_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail4
    (if 3 ≤ root60Candidates3.length then search edge 3 [307, 631] root60Candidates3 938 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates4 : List Nat := [24103]
set_option maxRecDepth 8192 in
theorem root60Candidates4_checked : candidates edge 4 307 733 root60Tail5 winner.bound = root60Candidates4 := by
  unfold candidates
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row]
  decide
theorem root60Step4 : scan edge (search edge 3) 4 [307] 307 root60Tail4 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail5 winner := by
  rw [root60Tail4, scan, if_neg (by decide), root60Candidates4_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail5
    (if 3 ≤ root60Candidates4.length then search edge 3 [307, 733] root60Candidates4 1040 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates5 : List Nat := [12853]
set_option maxRecDepth 8192 in
theorem root60Candidates5_checked : candidates edge 4 307 1069 root60Tail6 winner.bound = root60Candidates5 := by
  unfold candidates
  have row : edge 1069 = (SortedNatTree.contains FastRows.row1069) := by
    change Data.row1069.contains = (SortedNatTree.contains FastRows.row1069)
    exact FastRows.row1069_checked.symm
  rw [row]
  decide
theorem root60Step5 : scan edge (search edge 3) 4 [307] 307 root60Tail5 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail6 winner := by
  rw [root60Tail5, scan, if_neg (by decide), root60Candidates5_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail6
    (if 3 ≤ root60Candidates5.length then search edge 3 [307, 1069] root60Candidates5 1376 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates6 : List Nat := [4243, 23971]
set_option maxRecDepth 8192 in
theorem root60Candidates6_checked : candidates edge 4 307 1093 root60Tail7 winner.bound = root60Candidates6 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
theorem root60Step6 : scan edge (search edge 3) 4 [307] 307 root60Tail6 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail7 winner := by
  rw [root60Tail6, scan, if_neg (by decide), root60Candidates6_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail7
    (if 3 ≤ root60Candidates6.length then search edge 3 [307, 1093] root60Candidates6 1400 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates7 : List Nat := [14533]
set_option maxRecDepth 8192 in
theorem root60Candidates7_checked : candidates edge 4 307 1279 root60Tail8 winner.bound = root60Candidates7 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
theorem root60Step7 : scan edge (search edge 3) 4 [307] 307 root60Tail7 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail8 winner := by
  rw [root60Tail7, scan, if_neg (by decide), root60Candidates7_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail8
    (if 3 ≤ root60Candidates7.length then search edge 3 [307, 1279] root60Candidates7 1586 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates8 : List Nat := [10627]
set_option maxRecDepth 8192 in
theorem root60Candidates8_checked : candidates edge 4 307 1381 root60Tail9 winner.bound = root60Candidates8 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root60Step8 : scan edge (search edge 3) 4 [307] 307 root60Tail8 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail9 winner := by
  rw [root60Tail8, scan, if_neg (by decide), root60Candidates8_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail9
    (if 3 ≤ root60Candidates8.length then search edge 3 [307, 1381] root60Candidates8 1688 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root60Candidates9_checked : candidates edge 4 307 1993 root60Tail10 winner.bound = root60Candidates9 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
theorem root60Step9 : scan edge (search edge 3) 4 [307] 307 root60Tail9 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail10 winner := by
  rw [root60Tail9, scan, if_neg (by decide), root60Candidates9_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail10
    (if 3 ≤ root60Candidates9.length then search edge 3 [307, 1993] root60Candidates9 2300 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates10 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root60Candidates10_checked : candidates edge 4 307 2137 root60Tail11 winner.bound = root60Candidates10 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
theorem root60Step10 : scan edge (search edge 3) 4 [307] 307 root60Tail10 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail11 winner := by
  rw [root60Tail10, scan, if_neg (by decide), root60Candidates10_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail11
    (if 3 ≤ root60Candidates10.length then search edge 3 [307, 2137] root60Candidates10 2444 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates11 : List Nat := [8563]
set_option maxRecDepth 8192 in
theorem root60Candidates11_checked : candidates edge 4 307 2269 root60Tail12 winner.bound = root60Candidates11 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
theorem root60Step11 : scan edge (search edge 3) 4 [307] 307 root60Tail11 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail12 winner := by
  rw [root60Tail11, scan, if_neg (by decide), root60Candidates11_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail12
    (if 3 ≤ root60Candidates11.length then search edge 3 [307, 2269] root60Candidates11 2576 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates12 : List Nat := [10861]
set_option maxRecDepth 8192 in
theorem root60Candidates12_checked : candidates edge 4 307 2467 root60Tail13 winner.bound = root60Candidates12 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root60Step12 : scan edge (search edge 3) 4 [307] 307 root60Tail12 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail13 winner := by
  rw [root60Tail12, scan, if_neg (by decide), root60Candidates12_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail13
    (if 3 ≤ root60Candidates12.length then search edge 3 [307, 2467] root60Candidates12 2774 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates13 : List Nat := [9319]
set_option maxRecDepth 8192 in
theorem root60Candidates13_checked : candidates edge 4 307 3547 root60Tail14 winner.bound = root60Candidates13 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root60Step13 : scan edge (search edge 3) 4 [307] 307 root60Tail13 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail14 winner := by
  rw [root60Tail13, scan, if_neg (by decide), root60Candidates13_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail14
    (if 3 ≤ root60Candidates13.length then search edge 3 [307, 3547] root60Candidates13 3854 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates14 : List Nat := [8059, 8563, 10627, 14533]
set_option maxRecDepth 8192 in
theorem root60Candidates14_checked : candidates edge 4 307 3943 root60Tail15 winner.bound = root60Candidates14 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root60Child14 : search edge 3 [307, 3943] root60Candidates14 4250 winner = winner := by decide
theorem root60Step14 : scan edge (search edge 3) 4 [307] 307 root60Tail14 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail15 winner := by
  rw [root60Tail14, scan, if_neg (by decide), root60Candidates14_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail15
    (if 3 ≤ root60Candidates14.length then search edge 3 [307, 3943] root60Candidates14 4250 winner else winner) = _
  rw [if_pos (by decide), root60Child14]
def root60Candidates15 : List Nat := [11491]
set_option maxRecDepth 8192 in
theorem root60Candidates15_checked : candidates edge 4 307 4243 root60Tail16 winner.bound = root60Candidates15 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root60Step15 : scan edge (search edge 3) 4 [307] 307 root60Tail15 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail16 winner := by
  rw [root60Tail15, scan, if_neg (by decide), root60Candidates15_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail16
    (if 3 ≤ root60Candidates15.length then search edge 3 [307, 4243] root60Candidates15 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates16 : List Nat := [12601, 12853]
set_option maxRecDepth 8192 in
theorem root60Candidates16_checked : candidates edge 4 307 5119 root60Tail17 winner.bound = root60Candidates16 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root60Step16 : scan edge (search edge 3) 4 [307] 307 root60Tail16 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail17 winner := by
  rw [root60Tail16, scan, if_neg (by decide), root60Candidates16_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail17
    (if 3 ≤ root60Candidates16.length then search edge 3 [307, 5119] root60Candidates16 5426 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates17 : List Nat := [14533]
set_option maxRecDepth 8192 in
theorem root60Candidates17_checked : candidates edge 4 307 5209 root60Tail18 winner.bound = root60Candidates17 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root60Step17 : scan edge (search edge 3) 4 [307] 307 root60Tail17 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail18 winner := by
  rw [root60Tail17, scan, if_neg (by decide), root60Candidates17_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail18
    (if 3 ≤ root60Candidates17.length then search edge 3 [307, 5209] root60Candidates17 5516 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates18 : List Nat := [8059]
set_option maxRecDepth 8192 in
theorem root60Candidates18_checked : candidates edge 4 307 5857 root60Tail19 winner.bound = root60Candidates18 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root60Step18 : scan edge (search edge 3) 4 [307] 307 root60Tail18 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail19 winner := by
  rw [root60Tail18, scan, if_neg (by decide), root60Candidates18_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail19
    (if 3 ≤ root60Candidates18.length then search edge 3 [307, 5857] root60Candidates18 6164 winner else winner) = _
  rw [if_neg (by decide)]
def root60Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root60Candidates19_checked : candidates edge 4 307 6373 root60Tail20 winner.bound = root60Candidates19 := by
  unfold candidates
  have row : edge 6373 = (SortedNatTree.contains FastRows.row6373) := by
    change Data.row6373.contains = (SortedNatTree.contains FastRows.row6373)
    exact FastRows.row6373_checked.symm
  rw [row]
  decide
theorem root60Step19 : scan edge (search edge 3) 4 [307] 307 root60Tail19 winner =
    scan edge (search edge 3) 4 [307] 307 root60Tail20 winner := by
  rw [root60Tail19, scan, if_neg (by decide), root60Candidates19_checked]
  change scan edge (search edge 3) 4 [307] 307 root60Tail20
    (if 3 ≤ root60Candidates19.length then search edge 3 [307, 6373] root60Candidates19 6680 winner else winner) = _
  rw [if_neg (by decide)]
theorem root60Scan20 : scan edge (search edge 3) 4 [307] 307 root60Tail20 winner = winner := by rw [root60Tail20, scan, if_pos (by decide)]
theorem root60Scan19 : scan edge (search edge 3) 4 [307] 307 root60Tail19 winner = winner :=
  (root60Step19).trans root60Scan20
theorem root60Scan18 : scan edge (search edge 3) 4 [307] 307 root60Tail18 winner = winner :=
  (root60Step18).trans root60Scan19
theorem root60Scan17 : scan edge (search edge 3) 4 [307] 307 root60Tail17 winner = winner :=
  (root60Step17).trans root60Scan18
theorem root60Scan16 : scan edge (search edge 3) 4 [307] 307 root60Tail16 winner = winner :=
  (root60Step16).trans root60Scan17
theorem root60Scan15 : scan edge (search edge 3) 4 [307] 307 root60Tail15 winner = winner :=
  (root60Step15).trans root60Scan16
theorem root60Scan14 : scan edge (search edge 3) 4 [307] 307 root60Tail14 winner = winner :=
  (root60Step14).trans root60Scan15
theorem root60Scan13 : scan edge (search edge 3) 4 [307] 307 root60Tail13 winner = winner :=
  (root60Step13).trans root60Scan14
theorem root60Scan12 : scan edge (search edge 3) 4 [307] 307 root60Tail12 winner = winner :=
  (root60Step12).trans root60Scan13
theorem root60Scan11 : scan edge (search edge 3) 4 [307] 307 root60Tail11 winner = winner :=
  (root60Step11).trans root60Scan12
theorem root60Scan10 : scan edge (search edge 3) 4 [307] 307 root60Tail10 winner = winner :=
  (root60Step10).trans root60Scan11
theorem root60Scan9 : scan edge (search edge 3) 4 [307] 307 root60Tail9 winner = winner :=
  (root60Step9).trans root60Scan10
theorem root60Scan8 : scan edge (search edge 3) 4 [307] 307 root60Tail8 winner = winner :=
  (root60Step8).trans root60Scan9
theorem root60Scan7 : scan edge (search edge 3) 4 [307] 307 root60Tail7 winner = winner :=
  (root60Step7).trans root60Scan8
theorem root60Scan6 : scan edge (search edge 3) 4 [307] 307 root60Tail6 winner = winner :=
  (root60Step6).trans root60Scan7
theorem root60Scan5 : scan edge (search edge 3) 4 [307] 307 root60Tail5 winner = winner :=
  (root60Step5).trans root60Scan6
theorem root60Scan4 : scan edge (search edge 3) 4 [307] 307 root60Tail4 winner = winner :=
  (root60Step4).trans root60Scan5
theorem root60Scan3 : scan edge (search edge 3) 4 [307] 307 root60Tail3 winner = winner :=
  (root60Step3).trans root60Scan4
theorem root60Scan2 : scan edge (search edge 3) 4 [307] 307 root60Tail2 winner = winner :=
  (root60Step2).trans root60Scan3
theorem root60Scan1 : scan edge (search edge 3) 4 [307] 307 root60Tail1 winner = winner :=
  (root60Step1).trans root60Scan2
theorem root60Scan0 : scan edge (search edge 3) 4 [307] 307 root60Tail0 winner = winner :=
  (root60Step0).trans root60Scan1
set_option maxRecDepth 4096 in
theorem root60Branch : search edge 4 [307] root60Candidates 307 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [307] 307 root60Tail0 winner = winner
  exact root60Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
