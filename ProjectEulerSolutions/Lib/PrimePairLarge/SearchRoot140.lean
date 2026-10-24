import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root140Candidates : List Nat := Data.row823
set_option maxRecDepth 8192 in
theorem root140Candidates_checked : candidates edge 5 0 823 tail141 winner.bound = root140Candidates := by
  apply candidates_eq_row edge 5 0 823 tail141 root140Candidates winner.bound tail141_ordered
  · decide
  · decide
  · funext q
    rfl
def root140Tail65 : List Nat := []
def root140Tail64 : List Nat := 25147 :: root140Tail65
def root140Tail63 : List Nat := 24793 :: root140Tail64
def root140Tail62 : List Nat := 24547 :: root140Tail63
def root140Tail61 : List Nat := 24373 :: root140Tail62
def root140Tail60 : List Nat := 22111 :: root140Tail61
def root140Tail59 : List Nat := 21247 :: root140Tail60
def root140Tail58 : List Nat := 21013 :: root140Tail59
def root140Tail57 : List Nat := 20599 :: root140Tail58
def root140Tail56 : List Nat := 20551 :: root140Tail57
def root140Tail55 : List Nat := 20107 :: root140Tail56
def root140Tail54 : List Nat := 20047 :: root140Tail55
def root140Tail53 : List Nat := 19753 :: root140Tail54
def root140Tail52 : List Nat := 19213 :: root140Tail53
def root140Tail51 : List Nat := 18493 :: root140Tail52
def root140Tail50 : List Nat := 18427 :: root140Tail51
def root140Tail49 : List Nat := 17659 :: root140Tail50
def root140Tail48 : List Nat := 17377 :: root140Tail49
def root140Tail47 : List Nat := 16963 :: root140Tail48
def root140Tail46 : List Nat := 15559 :: root140Tail47
def root140Tail45 : List Nat := 15319 :: root140Tail46
def root140Tail44 : List Nat := 14653 :: root140Tail45
def root140Tail43 : List Nat := 14461 :: root140Tail44
def root140Tail42 : List Nat := 13963 :: root140Tail43
def root140Tail41 : List Nat := 12889 :: root140Tail42
def root140Tail40 : List Nat := 12619 :: root140Tail41
def root140Tail39 : List Nat := 12049 :: root140Tail40
def root140Tail38 : List Nat := 10909 :: root140Tail39
def root140Tail37 : List Nat := 10663 :: root140Tail38
def root140Tail36 : List Nat := 10477 :: root140Tail37
def root140Tail35 : List Nat := 10303 :: root140Tail36
def root140Tail34 : List Nat := 10099 :: root140Tail35
def root140Tail33 : List Nat := 10093 :: root140Tail34
def root140Tail32 : List Nat := 9967 :: root140Tail33
def root140Tail31 : List Nat := 9421 :: root140Tail32
def root140Tail30 : List Nat := 9199 :: root140Tail31
def root140Tail29 : List Nat := 9151 :: root140Tail30
def root140Tail28 : List Nat := 8101 :: root140Tail29
def root140Tail27 : List Nat := 7699 :: root140Tail28
def root140Tail26 : List Nat := 7237 :: root140Tail27
def root140Tail25 : List Nat := 6733 :: root140Tail26
def root140Tail24 : List Nat := 6337 :: root140Tail25
def root140Tail23 : List Nat := 5881 :: root140Tail24
def root140Tail22 : List Nat := 5827 :: root140Tail23
def root140Tail21 : List Nat := 5323 :: root140Tail22
def root140Tail20 : List Nat := 5281 :: root140Tail21
def root140Tail19 : List Nat := 5209 :: root140Tail20
def root140Tail18 : List Nat := 4951 :: root140Tail19
def root140Tail17 : List Nat := 4909 :: root140Tail18
def root140Tail16 : List Nat := 4789 :: root140Tail17
def root140Tail15 : List Nat := 4243 :: root140Tail16
def root140Tail14 : List Nat := 3541 :: root140Tail15
def root140Tail13 : List Nat := 2953 :: root140Tail14
def root140Tail12 : List Nat := 2557 :: root140Tail13
def root140Tail11 : List Nat := 2347 :: root140Tail12
def root140Tail10 : List Nat := 2311 :: root140Tail11
def root140Tail9 : List Nat := 2281 :: root140Tail10
def root140Tail8 : List Nat := 2179 :: root140Tail9
def root140Tail7 : List Nat := 2113 :: root140Tail8
def root140Tail6 : List Nat := 2017 :: root140Tail7
def root140Tail5 : List Nat := 1933 :: root140Tail6
def root140Tail4 : List Nat := 1777 :: root140Tail5
def root140Tail3 : List Nat := 1459 :: root140Tail4
def root140Tail2 : List Nat := 1303 :: root140Tail3
def root140Tail1 : List Nat := 967 :: root140Tail2
def root140Tail0 : List Nat := 877 :: root140Tail1
def root140Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates0_checked : candidates edge 4 823 877 root140Tail1 winner.bound = root140Candidates0 := by
  unfold candidates
  have row : edge 877 = (SortedNatTree.contains FastRows.row877) := by
    change Data.row877.contains = (SortedNatTree.contains FastRows.row877)
    exact FastRows.row877_checked.symm
  rw [row]
  decide
theorem root140Step0 : scan edge (search edge 3) 4 [823] 823 root140Tail0 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail1 winner := by
  rw [root140Tail0, scan, if_neg (by decide), root140Candidates0_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail1
    (if 3 ≤ root140Candidates0.length then search edge 3 [823, 877] root140Candidates0 1700 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates1 : List Nat := [2557, 10303, 19213, 22111]
set_option maxRecDepth 8192 in
theorem root140Candidates1_checked : candidates edge 4 823 967 root140Tail2 winner.bound = root140Candidates1 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root140Child1 : search edge 3 [823, 967] root140Candidates1 1790 winner = winner := by decide
theorem root140Step1 : scan edge (search edge 3) 4 [823] 823 root140Tail1 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail2 winner := by
  rw [root140Tail1, scan, if_neg (by decide), root140Candidates1_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail2
    (if 3 ≤ root140Candidates1.length then search edge 3 [823, 967] root140Candidates1 1790 winner else winner) = _
  rw [if_pos (by decide), root140Child1]
def root140Candidates2 : List Nat := [2953, 8101, 18427]
set_option maxRecDepth 8192 in
theorem root140Candidates2_checked : candidates edge 4 823 1303 root140Tail3 winner.bound = root140Candidates2 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root140Child2 : search edge 3 [823, 1303] root140Candidates2 2126 winner = winner := by decide
theorem root140Step2 : scan edge (search edge 3) 4 [823] 823 root140Tail2 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail3 winner := by
  rw [root140Tail2, scan, if_neg (by decide), root140Candidates2_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail3
    (if 3 ≤ root140Candidates2.length then search edge 3 [823, 1303] root140Candidates2 2126 winner else winner) = _
  rw [if_pos (by decide), root140Child2]
def root140Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates3_checked : candidates edge 4 823 1459 root140Tail4 winner.bound = root140Candidates3 := by
  unfold candidates
  have row : edge 1459 = (SortedNatTree.contains FastRows.row1459) := by
    change Data.row1459.contains = (SortedNatTree.contains FastRows.row1459)
    exact FastRows.row1459_checked.symm
  rw [row]
  decide
theorem root140Step3 : scan edge (search edge 3) 4 [823] 823 root140Tail3 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail4 winner := by
  rw [root140Tail3, scan, if_neg (by decide), root140Candidates3_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail4
    (if 3 ≤ root140Candidates3.length then search edge 3 [823, 1459] root140Candidates3 2282 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates4 : List Nat := [2017]
set_option maxRecDepth 8192 in
theorem root140Candidates4_checked : candidates edge 4 823 1777 root140Tail5 winner.bound = root140Candidates4 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
theorem root140Step4 : scan edge (search edge 3) 4 [823] 823 root140Tail4 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail5 winner := by
  rw [root140Tail4, scan, if_neg (by decide), root140Candidates4_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail5
    (if 3 ≤ root140Candidates4.length then search edge 3 [823, 1777] root140Candidates4 2600 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates5_checked : candidates edge 4 823 1933 root140Tail6 winner.bound = root140Candidates5 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root140Step5 : scan edge (search edge 3) 4 [823] 823 root140Tail5 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail6 winner := by
  rw [root140Tail5, scan, if_neg (by decide), root140Candidates5_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail6
    (if 3 ≤ root140Candidates5.length then search edge 3 [823, 1933] root140Candidates5 2756 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates6 : List Nat := [7237, 19753, 20047]
set_option maxRecDepth 8192 in
theorem root140Candidates6_checked : candidates edge 4 823 2017 root140Tail7 winner.bound = root140Candidates6 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root140Child6 : search edge 3 [823, 2017] root140Candidates6 2840 winner = winner := by decide
theorem root140Step6 : scan edge (search edge 3) 4 [823] 823 root140Tail6 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail7 winner := by
  rw [root140Tail6, scan, if_neg (by decide), root140Candidates6_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail7
    (if 3 ≤ root140Candidates6.length then search edge 3 [823, 2017] root140Candidates6 2840 winner else winner) = _
  rw [if_pos (by decide), root140Child6]
def root140Candidates7 : List Nat := [14461]
set_option maxRecDepth 8192 in
theorem root140Candidates7_checked : candidates edge 4 823 2113 root140Tail8 winner.bound = root140Candidates7 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root140Step7 : scan edge (search edge 3) 4 [823] 823 root140Tail7 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail8 winner := by
  rw [root140Tail7, scan, if_neg (by decide), root140Candidates7_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail8
    (if 3 ≤ root140Candidates7.length then search edge 3 [823, 2113] root140Candidates7 2936 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates8 : List Nat := [15559]
set_option maxRecDepth 8192 in
theorem root140Candidates8_checked : candidates edge 4 823 2179 root140Tail9 winner.bound = root140Candidates8 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
theorem root140Step8 : scan edge (search edge 3) 4 [823] 823 root140Tail8 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail9 winner := by
  rw [root140Tail8, scan, if_neg (by decide), root140Candidates8_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail9
    (if 3 ≤ root140Candidates8.length then search edge 3 [823, 2179] root140Candidates8 3002 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates9_checked : candidates edge 4 823 2281 root140Tail10 winner.bound = root140Candidates9 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root140Step9 : scan edge (search edge 3) 4 [823] 823 root140Tail9 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail10 winner := by
  rw [root140Tail9, scan, if_neg (by decide), root140Candidates9_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail10
    (if 3 ≤ root140Candidates9.length then search edge 3 [823, 2281] root140Candidates9 3104 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates10_checked : candidates edge 4 823 2311 root140Tail11 winner.bound = root140Candidates10 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root140Step10 : scan edge (search edge 3) 4 [823] 823 root140Tail10 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail11 winner := by
  rw [root140Tail10, scan, if_neg (by decide), root140Candidates10_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail11
    (if 3 ≤ root140Candidates10.length then search edge 3 [823, 2311] root140Candidates10 3134 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates11 : List Nat := [9151, 19753, 21247]
set_option maxRecDepth 8192 in
theorem root140Candidates11_checked : candidates edge 4 823 2347 root140Tail12 winner.bound = root140Candidates11 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root140Child11 : search edge 3 [823, 2347] root140Candidates11 3170 winner = winner := by decide
theorem root140Step11 : scan edge (search edge 3) 4 [823] 823 root140Tail11 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail12 winner := by
  rw [root140Tail11, scan, if_neg (by decide), root140Candidates11_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail12
    (if 3 ≤ root140Candidates11.length then search edge 3 [823, 2347] root140Candidates11 3170 winner else winner) = _
  rw [if_pos (by decide), root140Child11]
def root140Candidates12 : List Nat := [4243, 12049]
set_option maxRecDepth 8192 in
theorem root140Candidates12_checked : candidates edge 4 823 2557 root140Tail13 winner.bound = root140Candidates12 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
theorem root140Step12 : scan edge (search edge 3) 4 [823] 823 root140Tail12 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail13 winner := by
  rw [root140Tail12, scan, if_neg (by decide), root140Candidates12_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail13
    (if 3 ≤ root140Candidates12.length then search edge 3 [823, 2557] root140Candidates12 3380 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates13 : List Nat := [6733]
set_option maxRecDepth 8192 in
theorem root140Candidates13_checked : candidates edge 4 823 2953 root140Tail14 winner.bound = root140Candidates13 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
theorem root140Step13 : scan edge (search edge 3) 4 [823] 823 root140Tail13 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail14 winner := by
  rw [root140Tail13, scan, if_neg (by decide), root140Candidates13_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail14
    (if 3 ≤ root140Candidates13.length then search edge 3 [823, 2953] root140Candidates13 3776 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates14_checked : candidates edge 4 823 3541 root140Tail15 winner.bound = root140Candidates14 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root140Step14 : scan edge (search edge 3) 4 [823] 823 root140Tail14 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail15 winner := by
  rw [root140Tail14, scan, if_neg (by decide), root140Candidates14_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail15
    (if 3 ≤ root140Candidates14.length then search edge 3 [823, 3541] root140Candidates14 4364 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates15 : List Nat := [7699, 14461]
set_option maxRecDepth 8192 in
theorem root140Candidates15_checked : candidates edge 4 823 4243 root140Tail16 winner.bound = root140Candidates15 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root140Step15 : scan edge (search edge 3) 4 [823] 823 root140Tail15 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail16 winner := by
  rw [root140Tail15, scan, if_neg (by decide), root140Candidates15_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail16
    (if 3 ≤ root140Candidates15.length then search edge 3 [823, 4243] root140Candidates15 5066 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates16_checked : candidates edge 4 823 4789 root140Tail17 winner.bound = root140Candidates16 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root140Step16 : scan edge (search edge 3) 4 [823] 823 root140Tail16 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail17 winner := by
  rw [root140Tail16, scan, if_neg (by decide), root140Candidates16_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail17
    (if 3 ≤ root140Candidates16.length then search edge 3 [823, 4789] root140Candidates16 5612 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates17 : List Nat := [10477]
set_option maxRecDepth 8192 in
theorem root140Candidates17_checked : candidates edge 4 823 4909 root140Tail18 winner.bound = root140Candidates17 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root140Step17 : scan edge (search edge 3) 4 [823] 823 root140Tail17 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail18 winner := by
  rw [root140Tail17, scan, if_neg (by decide), root140Candidates17_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail18
    (if 3 ≤ root140Candidates17.length then search edge 3 [823, 4909] root140Candidates17 5732 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates18 : List Nat := [13963]
set_option maxRecDepth 8192 in
theorem root140Candidates18_checked : candidates edge 4 823 4951 root140Tail19 winner.bound = root140Candidates18 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root140Step18 : scan edge (search edge 3) 4 [823] 823 root140Tail18 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail19 winner := by
  rw [root140Tail18, scan, if_neg (by decide), root140Candidates18_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail19
    (if 3 ≤ root140Candidates18.length then search edge 3 [823, 4951] root140Candidates18 5774 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates19 : List Nat := [10303, 18427]
set_option maxRecDepth 8192 in
theorem root140Candidates19_checked : candidates edge 4 823 5209 root140Tail20 winner.bound = root140Candidates19 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root140Step19 : scan edge (search edge 3) 4 [823] 823 root140Tail19 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail20 winner := by
  rw [root140Tail19, scan, if_neg (by decide), root140Candidates19_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail20
    (if 3 ≤ root140Candidates19.length then search edge 3 [823, 5209] root140Candidates19 6032 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates20_checked : candidates edge 4 823 5281 root140Tail21 winner.bound = root140Candidates20 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root140Step20 : scan edge (search edge 3) 4 [823] 823 root140Tail20 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail21 winner := by
  rw [root140Tail20, scan, if_neg (by decide), root140Candidates20_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail21
    (if 3 ≤ root140Candidates20.length then search edge 3 [823, 5281] root140Candidates20 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates21 : List Nat := [7237, 12889, 18427, 19213]
set_option maxRecDepth 8192 in
theorem root140Candidates21_checked : candidates edge 4 823 5323 root140Tail22 winner.bound = root140Candidates21 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root140Child21 : search edge 3 [823, 5323] root140Candidates21 6146 winner = winner := by decide
theorem root140Step21 : scan edge (search edge 3) 4 [823] 823 root140Tail21 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail22 winner := by
  rw [root140Tail21, scan, if_neg (by decide), root140Candidates21_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail22
    (if 3 ≤ root140Candidates21.length then search edge 3 [823, 5323] root140Candidates21 6146 winner else winner) = _
  rw [if_pos (by decide), root140Child21]
def root140Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates22_checked : candidates edge 4 823 5827 root140Tail23 winner.bound = root140Candidates22 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root140Step22 : scan edge (search edge 3) 4 [823] 823 root140Tail22 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail23 winner := by
  rw [root140Tail22, scan, if_neg (by decide), root140Candidates22_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail23
    (if 3 ≤ root140Candidates22.length then search edge 3 [823, 5827] root140Candidates22 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root140Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root140Candidates23_checked : candidates edge 4 823 5881 root140Tail24 winner.bound = root140Candidates23 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root140Step23 : scan edge (search edge 3) 4 [823] 823 root140Tail23 winner =
    scan edge (search edge 3) 4 [823] 823 root140Tail24 winner := by
  rw [root140Tail23, scan, if_neg (by decide), root140Candidates23_checked]
  change scan edge (search edge 3) 4 [823] 823 root140Tail24
    (if 3 ≤ root140Candidates23.length then search edge 3 [823, 5881] root140Candidates23 6704 winner else winner) = _
  rw [if_neg (by decide)]
theorem root140Scan24 : scan edge (search edge 3) 4 [823] 823 root140Tail24 winner = winner := by rw [root140Tail24, scan, if_pos (by decide)]
theorem root140Scan23 : scan edge (search edge 3) 4 [823] 823 root140Tail23 winner = winner :=
  (root140Step23).trans root140Scan24
theorem root140Scan22 : scan edge (search edge 3) 4 [823] 823 root140Tail22 winner = winner :=
  (root140Step22).trans root140Scan23
theorem root140Scan21 : scan edge (search edge 3) 4 [823] 823 root140Tail21 winner = winner :=
  (root140Step21).trans root140Scan22
theorem root140Scan20 : scan edge (search edge 3) 4 [823] 823 root140Tail20 winner = winner :=
  (root140Step20).trans root140Scan21
theorem root140Scan19 : scan edge (search edge 3) 4 [823] 823 root140Tail19 winner = winner :=
  (root140Step19).trans root140Scan20
theorem root140Scan18 : scan edge (search edge 3) 4 [823] 823 root140Tail18 winner = winner :=
  (root140Step18).trans root140Scan19
theorem root140Scan17 : scan edge (search edge 3) 4 [823] 823 root140Tail17 winner = winner :=
  (root140Step17).trans root140Scan18
theorem root140Scan16 : scan edge (search edge 3) 4 [823] 823 root140Tail16 winner = winner :=
  (root140Step16).trans root140Scan17
theorem root140Scan15 : scan edge (search edge 3) 4 [823] 823 root140Tail15 winner = winner :=
  (root140Step15).trans root140Scan16
theorem root140Scan14 : scan edge (search edge 3) 4 [823] 823 root140Tail14 winner = winner :=
  (root140Step14).trans root140Scan15
theorem root140Scan13 : scan edge (search edge 3) 4 [823] 823 root140Tail13 winner = winner :=
  (root140Step13).trans root140Scan14
theorem root140Scan12 : scan edge (search edge 3) 4 [823] 823 root140Tail12 winner = winner :=
  (root140Step12).trans root140Scan13
theorem root140Scan11 : scan edge (search edge 3) 4 [823] 823 root140Tail11 winner = winner :=
  (root140Step11).trans root140Scan12
theorem root140Scan10 : scan edge (search edge 3) 4 [823] 823 root140Tail10 winner = winner :=
  (root140Step10).trans root140Scan11
theorem root140Scan9 : scan edge (search edge 3) 4 [823] 823 root140Tail9 winner = winner :=
  (root140Step9).trans root140Scan10
theorem root140Scan8 : scan edge (search edge 3) 4 [823] 823 root140Tail8 winner = winner :=
  (root140Step8).trans root140Scan9
theorem root140Scan7 : scan edge (search edge 3) 4 [823] 823 root140Tail7 winner = winner :=
  (root140Step7).trans root140Scan8
theorem root140Scan6 : scan edge (search edge 3) 4 [823] 823 root140Tail6 winner = winner :=
  (root140Step6).trans root140Scan7
theorem root140Scan5 : scan edge (search edge 3) 4 [823] 823 root140Tail5 winner = winner :=
  (root140Step5).trans root140Scan6
theorem root140Scan4 : scan edge (search edge 3) 4 [823] 823 root140Tail4 winner = winner :=
  (root140Step4).trans root140Scan5
theorem root140Scan3 : scan edge (search edge 3) 4 [823] 823 root140Tail3 winner = winner :=
  (root140Step3).trans root140Scan4
theorem root140Scan2 : scan edge (search edge 3) 4 [823] 823 root140Tail2 winner = winner :=
  (root140Step2).trans root140Scan3
theorem root140Scan1 : scan edge (search edge 3) 4 [823] 823 root140Tail1 winner = winner :=
  (root140Step1).trans root140Scan2
theorem root140Scan0 : scan edge (search edge 3) 4 [823] 823 root140Tail0 winner = winner :=
  (root140Step0).trans root140Scan1
set_option maxRecDepth 4096 in
theorem root140Branch : search edge 4 [823] root140Candidates 823 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [823] 823 root140Tail0 winner = winner
  exact root140Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
