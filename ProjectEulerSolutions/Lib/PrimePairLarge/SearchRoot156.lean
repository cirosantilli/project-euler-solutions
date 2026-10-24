import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root156Candidates : List Nat := Data.row937
set_option maxRecDepth 8192 in
theorem root156Candidates_checked : candidates edge 5 0 937 tail157 winner.bound = root156Candidates := by
  apply candidates_eq_row edge 5 0 937 tail157 root156Candidates winner.bound tail157_ordered
  · decide
  · decide
  · funext q
    rfl
def root156Tail58 : List Nat := []
def root156Tail57 : List Nat := 24097 :: root156Tail58
def root156Tail56 : List Nat := 23899 :: root156Tail57
def root156Tail55 : List Nat := 23887 :: root156Tail56
def root156Tail54 : List Nat := 23857 :: root156Tail55
def root156Tail53 : List Nat := 23767 :: root156Tail54
def root156Tail52 : List Nat := 23719 :: root156Tail53
def root156Tail51 : List Nat := 23671 :: root156Tail52
def root156Tail50 : List Nat := 22807 :: root156Tail51
def root156Tail49 : List Nat := 22039 :: root156Tail50
def root156Tail48 : List Nat := 21499 :: root156Tail49
def root156Tail47 : List Nat := 21247 :: root156Tail48
def root156Tail46 : List Nat := 21169 :: root156Tail47
def root156Tail45 : List Nat := 20479 :: root156Tail46
def root156Tail44 : List Nat := 19183 :: root156Tail45
def root156Tail43 : List Nat := 19051 :: root156Tail44
def root156Tail42 : List Nat := 18523 :: root156Tail43
def root156Tail41 : List Nat := 17581 :: root156Tail42
def root156Tail40 : List Nat := 16741 :: root156Tail41
def root156Tail39 : List Nat := 16339 :: root156Tail40
def root156Tail38 : List Nat := 16033 :: root156Tail39
def root156Tail37 : List Nat := 15451 :: root156Tail38
def root156Tail36 : List Nat := 15241 :: root156Tail37
def root156Tail35 : List Nat := 15031 :: root156Tail36
def root156Tail34 : List Nat := 14737 :: root156Tail35
def root156Tail33 : List Nat := 14389 :: root156Tail34
def root156Tail32 : List Nat := 14347 :: root156Tail33
def root156Tail31 : List Nat := 14143 :: root156Tail32
def root156Tail30 : List Nat := 13177 :: root156Tail31
def root156Tail29 : List Nat := 13171 :: root156Tail30
def root156Tail28 : List Nat := 13099 :: root156Tail29
def root156Tail27 : List Nat := 12973 :: root156Tail28
def root156Tail26 : List Nat := 10753 :: root156Tail27
def root156Tail25 : List Nat := 10651 :: root156Tail26
def root156Tail24 : List Nat := 10321 :: root156Tail25
def root156Tail23 : List Nat := 9679 :: root156Tail24
def root156Tail22 : List Nat := 9199 :: root156Tail23
def root156Tail21 : List Nat := 8233 :: root156Tail22
def root156Tail20 : List Nat := 8059 :: root156Tail21
def root156Tail19 : List Nat := 7213 :: root156Tail20
def root156Tail18 : List Nat := 5827 :: root156Tail19
def root156Tail17 : List Nat := 5791 :: root156Tail18
def root156Tail16 : List Nat := 5437 :: root156Tail17
def root156Tail15 : List Nat := 5407 :: root156Tail16
def root156Tail14 : List Nat := 5281 :: root156Tail15
def root156Tail13 : List Nat := 5059 :: root156Tail14
def root156Tail12 : List Nat := 4273 :: root156Tail13
def root156Tail11 : List Nat := 4159 :: root156Tail12
def root156Tail10 : List Nat := 4093 :: root156Tail11
def root156Tail9 : List Nat := 3943 :: root156Tail10
def root156Tail8 : List Nat := 3853 :: root156Tail9
def root156Tail7 : List Nat := 3319 :: root156Tail8
def root156Tail6 : List Nat := 3109 :: root156Tail7
def root156Tail5 : List Nat := 2791 :: root156Tail6
def root156Tail4 : List Nat := 2557 :: root156Tail5
def root156Tail3 : List Nat := 2131 :: root156Tail4
def root156Tail2 : List Nat := 2053 :: root156Tail3
def root156Tail1 : List Nat := 1783 :: root156Tail2
def root156Tail0 : List Nat := 1489 :: root156Tail1
def root156Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root156Candidates0_checked : candidates edge 4 937 1489 root156Tail1 winner.bound = root156Candidates0 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
theorem root156Step0 : scan edge (search edge 3) 4 [937] 937 root156Tail0 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail1 winner := by
  rw [root156Tail0, scan, if_neg (by decide), root156Candidates0_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail1
    (if 3 ≤ root156Candidates0.length then search edge 3 [937, 1489] root156Candidates0 2426 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates1 : List Nat := [2053, 2791, 4273]
set_option maxRecDepth 8192 in
theorem root156Candidates1_checked : candidates edge 4 937 1783 root156Tail2 winner.bound = root156Candidates1 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root156Child1 : search edge 3 [937, 1783] root156Candidates1 2720 winner = winner := by decide
theorem root156Step1 : scan edge (search edge 3) 4 [937] 937 root156Tail1 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail2 winner := by
  rw [root156Tail1, scan, if_neg (by decide), root156Candidates1_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail2
    (if 3 ≤ root156Candidates1.length then search edge 3 [937, 1783] root156Candidates1 2720 winner else winner) = _
  rw [if_pos (by decide), root156Child1]
def root156Candidates2 : List Nat := [18523]
set_option maxRecDepth 8192 in
theorem root156Candidates2_checked : candidates edge 4 937 2053 root156Tail3 winner.bound = root156Candidates2 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root156Step2 : scan edge (search edge 3) 4 [937] 937 root156Tail2 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail3 winner := by
  rw [root156Tail2, scan, if_neg (by decide), root156Candidates2_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail3
    (if 3 ≤ root156Candidates2.length then search edge 3 [937, 2053] root156Candidates2 2990 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root156Candidates3_checked : candidates edge 4 937 2131 root156Tail4 winner.bound = root156Candidates3 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root156Step3 : scan edge (search edge 3) 4 [937] 937 root156Tail3 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail4 winner := by
  rw [root156Tail3, scan, if_neg (by decide), root156Candidates3_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail4
    (if 3 ≤ root156Candidates3.length then search edge 3 [937, 2131] root156Candidates3 3068 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates4 : List Nat := [12973]
set_option maxRecDepth 8192 in
theorem root156Candidates4_checked : candidates edge 4 937 2557 root156Tail5 winner.bound = root156Candidates4 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
theorem root156Step4 : scan edge (search edge 3) 4 [937] 937 root156Tail4 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail5 winner := by
  rw [root156Tail4, scan, if_neg (by decide), root156Candidates4_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail5
    (if 3 ≤ root156Candidates4.length then search edge 3 [937, 2557] root156Candidates4 3494 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates5 : List Nat := [5281, 18523]
set_option maxRecDepth 8192 in
theorem root156Candidates5_checked : candidates edge 4 937 2791 root156Tail6 winner.bound = root156Candidates5 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root156Step5 : scan edge (search edge 3) 4 [937] 937 root156Tail5 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail6 winner := by
  rw [root156Tail5, scan, if_neg (by decide), root156Candidates5_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail6
    (if 3 ≤ root156Candidates5.length then search edge 3 [937, 2791] root156Candidates5 3728 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates6 : List Nat := [5791]
set_option maxRecDepth 8192 in
theorem root156Candidates6_checked : candidates edge 4 937 3109 root156Tail7 winner.bound = root156Candidates6 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root156Step6 : scan edge (search edge 3) 4 [937] 937 root156Tail6 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail7 winner := by
  rw [root156Tail6, scan, if_neg (by decide), root156Candidates6_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail7
    (if 3 ≤ root156Candidates6.length then search edge 3 [937, 3109] root156Candidates6 4046 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root156Candidates7_checked : candidates edge 4 937 3319 root156Tail8 winner.bound = root156Candidates7 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
theorem root156Step7 : scan edge (search edge 3) 4 [937] 937 root156Tail7 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail8 winner := by
  rw [root156Tail7, scan, if_neg (by decide), root156Candidates7_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail8
    (if 3 ≤ root156Candidates7.length then search edge 3 [937, 3319] root156Candidates7 4256 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root156Candidates8_checked : candidates edge 4 937 3853 root156Tail9 winner.bound = root156Candidates8 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root156Step8 : scan edge (search edge 3) 4 [937] 937 root156Tail8 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail9 winner := by
  rw [root156Tail8, scan, if_neg (by decide), root156Candidates8_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail9
    (if 3 ≤ root156Candidates8.length then search edge 3 [937, 3853] root156Candidates8 4790 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates9 : List Nat := [8059, 18523]
set_option maxRecDepth 8192 in
theorem root156Candidates9_checked : candidates edge 4 937 3943 root156Tail10 winner.bound = root156Candidates9 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root156Step9 : scan edge (search edge 3) 4 [937] 937 root156Tail9 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail10 winner := by
  rw [root156Tail9, scan, if_neg (by decide), root156Candidates9_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail10
    (if 3 ≤ root156Candidates9.length then search edge 3 [937, 3943] root156Candidates9 4880 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root156Candidates10_checked : candidates edge 4 937 4093 root156Tail11 winner.bound = root156Candidates10 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root156Step10 : scan edge (search edge 3) 4 [937] 937 root156Tail10 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail11 winner := by
  rw [root156Tail10, scan, if_neg (by decide), root156Candidates10_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail11
    (if 3 ≤ root156Candidates10.length then search edge 3 [937, 4093] root156Candidates10 5030 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates11 : List Nat := [8233]
set_option maxRecDepth 8192 in
theorem root156Candidates11_checked : candidates edge 4 937 4159 root156Tail12 winner.bound = root156Candidates11 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root156Step11 : scan edge (search edge 3) 4 [937] 937 root156Tail11 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail12 winner := by
  rw [root156Tail11, scan, if_neg (by decide), root156Candidates11_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail12
    (if 3 ≤ root156Candidates11.length then search edge 3 [937, 4159] root156Candidates11 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates12 : List Nat := [13171]
set_option maxRecDepth 8192 in
theorem root156Candidates12_checked : candidates edge 4 937 4273 root156Tail13 winner.bound = root156Candidates12 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root156Step12 : scan edge (search edge 3) 4 [937] 937 root156Tail12 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail13 winner := by
  rw [root156Tail12, scan, if_neg (by decide), root156Candidates12_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail13
    (if 3 ≤ root156Candidates12.length then search edge 3 [937, 4273] root156Candidates12 5210 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates13 : List Nat := [7213]
set_option maxRecDepth 8192 in
theorem root156Candidates13_checked : candidates edge 4 937 5059 root156Tail14 winner.bound = root156Candidates13 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root156Step13 : scan edge (search edge 3) 4 [937] 937 root156Tail13 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail14 winner := by
  rw [root156Tail13, scan, if_neg (by decide), root156Candidates13_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail14
    (if 3 ≤ root156Candidates13.length then search edge 3 [937, 5059] root156Candidates13 5996 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates14 : List Nat := [5437]
set_option maxRecDepth 8192 in
theorem root156Candidates14_checked : candidates edge 4 937 5281 root156Tail15 winner.bound = root156Candidates14 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root156Step14 : scan edge (search edge 3) 4 [937] 937 root156Tail14 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail15 winner := by
  rw [root156Tail14, scan, if_neg (by decide), root156Candidates14_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail15
    (if 3 ≤ root156Candidates14.length then search edge 3 [937, 5281] root156Candidates14 6218 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates15 : List Nat := [9679]
set_option maxRecDepth 8192 in
theorem root156Candidates15_checked : candidates edge 4 937 5407 root156Tail16 winner.bound = root156Candidates15 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root156Step15 : scan edge (search edge 3) 4 [937] 937 root156Tail15 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail16 winner := by
  rw [root156Tail15, scan, if_neg (by decide), root156Candidates15_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail16
    (if 3 ≤ root156Candidates15.length then search edge 3 [937, 5407] root156Candidates15 6344 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates16 : List Nat := [10753]
set_option maxRecDepth 8192 in
theorem root156Candidates16_checked : candidates edge 4 937 5437 root156Tail17 winner.bound = root156Candidates16 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root156Step16 : scan edge (search edge 3) 4 [937] 937 root156Tail16 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail17 winner := by
  rw [root156Tail16, scan, if_neg (by decide), root156Candidates16_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail17
    (if 3 ≤ root156Candidates16.length then search edge 3 [937, 5437] root156Candidates16 6374 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates17 : List Nat := [15451]
set_option maxRecDepth 8192 in
theorem root156Candidates17_checked : candidates edge 4 937 5791 root156Tail18 winner.bound = root156Candidates17 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root156Step17 : scan edge (search edge 3) 4 [937] 937 root156Tail17 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail18 winner := by
  rw [root156Tail17, scan, if_neg (by decide), root156Candidates17_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail18
    (if 3 ≤ root156Candidates17.length then search edge 3 [937, 5791] root156Candidates17 6728 winner else winner) = _
  rw [if_neg (by decide)]
def root156Candidates18 : List Nat := [7213, 9679]
set_option maxRecDepth 8192 in
theorem root156Candidates18_checked : candidates edge 4 937 5827 root156Tail19 winner.bound = root156Candidates18 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root156Step18 : scan edge (search edge 3) 4 [937] 937 root156Tail18 winner =
    scan edge (search edge 3) 4 [937] 937 root156Tail19 winner := by
  rw [root156Tail18, scan, if_neg (by decide), root156Candidates18_checked]
  change scan edge (search edge 3) 4 [937] 937 root156Tail19
    (if 3 ≤ root156Candidates18.length then search edge 3 [937, 5827] root156Candidates18 6764 winner else winner) = _
  rw [if_neg (by decide)]
theorem root156Scan19 : scan edge (search edge 3) 4 [937] 937 root156Tail19 winner = winner := by rw [root156Tail19, scan, if_pos (by decide)]
theorem root156Scan18 : scan edge (search edge 3) 4 [937] 937 root156Tail18 winner = winner :=
  (root156Step18).trans root156Scan19
theorem root156Scan17 : scan edge (search edge 3) 4 [937] 937 root156Tail17 winner = winner :=
  (root156Step17).trans root156Scan18
theorem root156Scan16 : scan edge (search edge 3) 4 [937] 937 root156Tail16 winner = winner :=
  (root156Step16).trans root156Scan17
theorem root156Scan15 : scan edge (search edge 3) 4 [937] 937 root156Tail15 winner = winner :=
  (root156Step15).trans root156Scan16
theorem root156Scan14 : scan edge (search edge 3) 4 [937] 937 root156Tail14 winner = winner :=
  (root156Step14).trans root156Scan15
theorem root156Scan13 : scan edge (search edge 3) 4 [937] 937 root156Tail13 winner = winner :=
  (root156Step13).trans root156Scan14
theorem root156Scan12 : scan edge (search edge 3) 4 [937] 937 root156Tail12 winner = winner :=
  (root156Step12).trans root156Scan13
theorem root156Scan11 : scan edge (search edge 3) 4 [937] 937 root156Tail11 winner = winner :=
  (root156Step11).trans root156Scan12
theorem root156Scan10 : scan edge (search edge 3) 4 [937] 937 root156Tail10 winner = winner :=
  (root156Step10).trans root156Scan11
theorem root156Scan9 : scan edge (search edge 3) 4 [937] 937 root156Tail9 winner = winner :=
  (root156Step9).trans root156Scan10
theorem root156Scan8 : scan edge (search edge 3) 4 [937] 937 root156Tail8 winner = winner :=
  (root156Step8).trans root156Scan9
theorem root156Scan7 : scan edge (search edge 3) 4 [937] 937 root156Tail7 winner = winner :=
  (root156Step7).trans root156Scan8
theorem root156Scan6 : scan edge (search edge 3) 4 [937] 937 root156Tail6 winner = winner :=
  (root156Step6).trans root156Scan7
theorem root156Scan5 : scan edge (search edge 3) 4 [937] 937 root156Tail5 winner = winner :=
  (root156Step5).trans root156Scan6
theorem root156Scan4 : scan edge (search edge 3) 4 [937] 937 root156Tail4 winner = winner :=
  (root156Step4).trans root156Scan5
theorem root156Scan3 : scan edge (search edge 3) 4 [937] 937 root156Tail3 winner = winner :=
  (root156Step3).trans root156Scan4
theorem root156Scan2 : scan edge (search edge 3) 4 [937] 937 root156Tail2 winner = winner :=
  (root156Step2).trans root156Scan3
theorem root156Scan1 : scan edge (search edge 3) 4 [937] 937 root156Tail1 winner = winner :=
  (root156Step1).trans root156Scan2
theorem root156Scan0 : scan edge (search edge 3) 4 [937] 937 root156Tail0 winner = winner :=
  (root156Step0).trans root156Scan1
set_option maxRecDepth 4096 in
theorem root156Branch : search edge 4 [937] root156Candidates 937 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [937] 937 root156Tail0 winner = winner
  exact root156Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
