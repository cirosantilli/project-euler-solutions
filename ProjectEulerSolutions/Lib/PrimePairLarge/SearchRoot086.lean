import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root86Candidates : List Nat := Data.row461
set_option maxRecDepth 8192 in
theorem root86Candidates_checked : candidates edge 5 0 461 tail87 winner.bound = root86Candidates := by
  apply candidates_eq_row edge 5 0 461 tail87 root86Candidates winner.bound tail87_ordered
  · decide
  · decide
  · funext q
    rfl
def root86Tail69 : List Nat := []
def root86Tail68 : List Nat := 25523 :: root86Tail69
def root86Tail67 : List Nat := 25169 :: root86Tail68
def root86Tail66 : List Nat := 25127 :: root86Tail67
def root86Tail65 : List Nat := 24473 :: root86Tail66
def root86Tail64 : List Nat := 23879 :: root86Tail65
def root86Tail63 : List Nat := 23633 :: root86Tail64
def root86Tail62 : List Nat := 23447 :: root86Tail63
def root86Tail61 : List Nat := 22697 :: root86Tail62
def root86Tail60 : List Nat := 22409 :: root86Tail61
def root86Tail59 : List Nat := 21821 :: root86Tail60
def root86Tail58 : List Nat := 21767 :: root86Tail59
def root86Tail57 : List Nat := 20849 :: root86Tail58
def root86Tail56 : List Nat := 20717 :: root86Tail57
def root86Tail55 : List Nat := 20549 :: root86Tail56
def root86Tail54 : List Nat := 20183 :: root86Tail55
def root86Tail53 : List Nat := 20051 :: root86Tail54
def root86Tail52 : List Nat := 19937 :: root86Tail53
def root86Tail51 : List Nat := 19841 :: root86Tail52
def root86Tail50 : List Nat := 19559 :: root86Tail51
def root86Tail49 : List Nat := 19553 :: root86Tail50
def root86Tail48 : List Nat := 19457 :: root86Tail49
def root86Tail47 : List Nat := 18587 :: root86Tail48
def root86Tail46 : List Nat := 18503 :: root86Tail47
def root86Tail45 : List Nat := 18341 :: root86Tail46
def root86Tail44 : List Nat := 18251 :: root86Tail45
def root86Tail43 : List Nat := 18077 :: root86Tail44
def root86Tail42 : List Nat := 17483 :: root86Tail43
def root86Tail41 : List Nat := 17417 :: root86Tail42
def root86Tail40 : List Nat := 17189 :: root86Tail41
def root86Tail39 : List Nat := 17021 :: root86Tail40
def root86Tail38 : List Nat := 15269 :: root86Tail39
def root86Tail37 : List Nat := 15149 :: root86Tail38
def root86Tail36 : List Nat := 13829 :: root86Tail37
def root86Tail35 : List Nat := 13037 :: root86Tail36
def root86Tail34 : List Nat := 12821 :: root86Tail35
def root86Tail33 : List Nat := 12611 :: root86Tail34
def root86Tail32 : List Nat := 12527 :: root86Tail33
def root86Tail31 : List Nat := 12071 :: root86Tail32
def root86Tail30 : List Nat := 11621 :: root86Tail31
def root86Tail29 : List Nat := 10859 :: root86Tail30
def root86Tail28 : List Nat := 10781 :: root86Tail29
def root86Tail27 : List Nat := 10487 :: root86Tail28
def root86Tail26 : List Nat := 9749 :: root86Tail27
def root86Tail25 : List Nat := 9623 :: root86Tail26
def root86Tail24 : List Nat := 9521 :: root86Tail25
def root86Tail23 : List Nat := 8753 :: root86Tail24
def root86Tail22 : List Nat := 7451 :: root86Tail23
def root86Tail21 : List Nat := 7187 :: root86Tail22
def root86Tail20 : List Nat := 6977 :: root86Tail21
def root86Tail19 : List Nat := 6473 :: root86Tail20
def root86Tail18 : List Nat := 5783 :: root86Tail19
def root86Tail17 : List Nat := 4937 :: root86Tail18
def root86Tail16 : List Nat := 4637 :: root86Tail17
def root86Tail15 : List Nat := 4271 :: root86Tail16
def root86Tail14 : List Nat := 4217 :: root86Tail15
def root86Tail13 : List Nat := 3881 :: root86Tail14
def root86Tail12 : List Nat := 3449 :: root86Tail13
def root86Tail11 : List Nat := 2549 :: root86Tail12
def root86Tail10 : List Nat := 2339 :: root86Tail11
def root86Tail9 : List Nat := 1697 :: root86Tail10
def root86Tail8 : List Nat := 1571 :: root86Tail9
def root86Tail7 : List Nat := 1487 :: root86Tail8
def root86Tail6 : List Nat := 1307 :: root86Tail7
def root86Tail5 : List Nat := 1181 :: root86Tail6
def root86Tail4 : List Nat := 983 :: root86Tail5
def root86Tail3 : List Nat := 677 :: root86Tail4
def root86Tail2 : List Nat := 653 :: root86Tail3
def root86Tail1 : List Nat := 569 :: root86Tail2
def root86Tail0 : List Nat := 479 :: root86Tail1
def root86Candidates0 : List Nat := [569, 1307, 10859, 17417, 20183]
set_option maxRecDepth 8192 in
theorem root86Candidates0_checked : candidates edge 4 461 479 root86Tail1 winner.bound = root86Candidates0 := by
  unfold candidates
  have row : edge 479 = (SortedNatTree.contains FastRows.row479) := by
    change Data.row479.contains = (SortedNatTree.contains FastRows.row479)
    exact FastRows.row479_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root86Child0 : search edge 3 [461, 479] root86Candidates0 940 winner = winner := by decide
theorem root86Step0 : scan edge (search edge 3) 4 [461] 461 root86Tail0 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail1 winner := by
  rw [root86Tail0, scan, if_neg (by decide), root86Candidates0_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail1
    (if 3 ≤ root86Candidates0.length then search edge 3 [461, 479] root86Candidates0 940 winner else winner) = _
  rw [if_pos (by decide), root86Child0]
def root86Candidates1 : List Nat := [1487]
set_option maxRecDepth 8192 in
theorem root86Candidates1_checked : candidates edge 4 461 569 root86Tail2 winner.bound = root86Candidates1 := by
  unfold candidates
  have row : edge 569 = (SortedNatTree.contains FastRows.row569) := by
    change Data.row569.contains = (SortedNatTree.contains FastRows.row569)
    exact FastRows.row569_checked.symm
  rw [row]
  decide
theorem root86Step1 : scan edge (search edge 3) 4 [461] 461 root86Tail1 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail2 winner := by
  rw [root86Tail1, scan, if_neg (by decide), root86Candidates1_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail2
    (if 3 ≤ root86Candidates1.length then search edge 3 [461, 569] root86Candidates1 1030 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates2 : List Nat := [15149]
set_option maxRecDepth 8192 in
theorem root86Candidates2_checked : candidates edge 4 461 653 root86Tail3 winner.bound = root86Candidates2 := by
  unfold candidates
  have row : edge 653 = (SortedNatTree.contains FastRows.row653) := by
    change Data.row653.contains = (SortedNatTree.contains FastRows.row653)
    exact FastRows.row653_checked.symm
  rw [row]
  decide
theorem root86Step2 : scan edge (search edge 3) 4 [461] 461 root86Tail2 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail3 winner := by
  rw [root86Tail2, scan, if_neg (by decide), root86Candidates2_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail3
    (if 3 ≤ root86Candidates2.length then search edge 3 [461, 653] root86Candidates2 1114 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates3 : List Nat := [7451, 12527, 13829, 17189]
set_option maxRecDepth 8192 in
theorem root86Candidates3_checked : candidates edge 4 461 677 root86Tail4 winner.bound = root86Candidates3 := by
  unfold candidates
  have row : edge 677 = (SortedNatTree.contains FastRows.row677) := by
    change Data.row677.contains = (SortedNatTree.contains FastRows.row677)
    exact FastRows.row677_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root86Child3 : search edge 3 [461, 677] root86Candidates3 1138 winner = winner := by decide
theorem root86Step3 : scan edge (search edge 3) 4 [461] 461 root86Tail3 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail4 winner := by
  rw [root86Tail3, scan, if_neg (by decide), root86Candidates3_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail4
    (if 3 ≤ root86Candidates3.length then search edge 3 [461, 677] root86Candidates3 1138 winner else winner) = _
  rw [if_pos (by decide), root86Child3]
def root86Candidates4 : List Nat := [10781, 22409]
set_option maxRecDepth 8192 in
theorem root86Candidates4_checked : candidates edge 4 461 983 root86Tail5 winner.bound = root86Candidates4 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root86Step4 : scan edge (search edge 3) 4 [461] 461 root86Tail4 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail5 winner := by
  rw [root86Tail4, scan, if_neg (by decide), root86Candidates4_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail5
    (if 3 ≤ root86Candidates4.length then search edge 3 [461, 983] root86Candidates4 1444 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates5 : List Nat := [6977]
set_option maxRecDepth 8192 in
theorem root86Candidates5_checked : candidates edge 4 461 1181 root86Tail6 winner.bound = root86Candidates5 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
theorem root86Step5 : scan edge (search edge 3) 4 [461] 461 root86Tail5 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail6 winner := by
  rw [root86Tail5, scan, if_neg (by decide), root86Candidates5_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail6
    (if 3 ≤ root86Candidates5.length then search edge 3 [461, 1181] root86Candidates5 1642 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root86Candidates6_checked : candidates edge 4 461 1307 root86Tail7 winner.bound = root86Candidates6 := by
  unfold candidates
  have row : edge 1307 = (SortedNatTree.contains FastRows.row1307) := by
    change Data.row1307.contains = (SortedNatTree.contains FastRows.row1307)
    exact FastRows.row1307_checked.symm
  rw [row]
  decide
theorem root86Step6 : scan edge (search edge 3) 4 [461] 461 root86Tail6 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail7 winner := by
  rw [root86Tail6, scan, if_neg (by decide), root86Candidates6_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail7
    (if 3 ≤ root86Candidates6.length then search edge 3 [461, 1307] root86Candidates6 1768 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates7 : List Nat := [6473, 20717]
set_option maxRecDepth 8192 in
theorem root86Candidates7_checked : candidates edge 4 461 1487 root86Tail8 winner.bound = root86Candidates7 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
theorem root86Step7 : scan edge (search edge 3) 4 [461] 461 root86Tail7 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail8 winner := by
  rw [root86Tail7, scan, if_neg (by decide), root86Candidates7_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail8
    (if 3 ≤ root86Candidates7.length then search edge 3 [461, 1487] root86Candidates7 1948 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root86Candidates8_checked : candidates edge 4 461 1571 root86Tail9 winner.bound = root86Candidates8 := by
  unfold candidates
  have row : edge 1571 = (SortedNatTree.contains FastRows.row1571) := by
    change Data.row1571.contains = (SortedNatTree.contains FastRows.row1571)
    exact FastRows.row1571_checked.symm
  rw [row]
  decide
theorem root86Step8 : scan edge (search edge 3) 4 [461] 461 root86Tail8 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail9 winner := by
  rw [root86Tail8, scan, if_neg (by decide), root86Candidates8_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail9
    (if 3 ≤ root86Candidates8.length then search edge 3 [461, 1571] root86Candidates8 2032 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates9 : List Nat := [12527]
set_option maxRecDepth 8192 in
theorem root86Candidates9_checked : candidates edge 4 461 1697 root86Tail10 winner.bound = root86Candidates9 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root86Step9 : scan edge (search edge 3) 4 [461] 461 root86Tail9 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail10 winner := by
  rw [root86Tail9, scan, if_neg (by decide), root86Candidates9_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail10
    (if 3 ≤ root86Candidates9.length then search edge 3 [461, 1697] root86Candidates9 2158 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root86Candidates10_checked : candidates edge 4 461 2339 root86Tail11 winner.bound = root86Candidates10 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root86Step10 : scan edge (search edge 3) 4 [461] 461 root86Tail10 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail11 winner := by
  rw [root86Tail10, scan, if_neg (by decide), root86Candidates10_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail11
    (if 3 ≤ root86Candidates10.length then search edge 3 [461, 2339] root86Candidates10 2800 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root86Candidates11_checked : candidates edge 4 461 2549 root86Tail12 winner.bound = root86Candidates11 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root86Step11 : scan edge (search edge 3) 4 [461] 461 root86Tail11 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail12 winner := by
  rw [root86Tail11, scan, if_neg (by decide), root86Candidates11_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail12
    (if 3 ≤ root86Candidates11.length then search edge 3 [461, 2549] root86Candidates11 3010 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root86Candidates12_checked : candidates edge 4 461 3449 root86Tail13 winner.bound = root86Candidates12 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root86Step12 : scan edge (search edge 3) 4 [461] 461 root86Tail12 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail13 winner := by
  rw [root86Tail12, scan, if_neg (by decide), root86Candidates12_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail13
    (if 3 ≤ root86Candidates12.length then search edge 3 [461, 3449] root86Candidates12 3910 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates13 : List Nat := [20549]
set_option maxRecDepth 8192 in
theorem root86Candidates13_checked : candidates edge 4 461 3881 root86Tail14 winner.bound = root86Candidates13 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root86Step13 : scan edge (search edge 3) 4 [461] 461 root86Tail13 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail14 winner := by
  rw [root86Tail13, scan, if_neg (by decide), root86Candidates13_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail14
    (if 3 ≤ root86Candidates13.length then search edge 3 [461, 3881] root86Candidates13 4342 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates14 : List Nat := [20549]
set_option maxRecDepth 8192 in
theorem root86Candidates14_checked : candidates edge 4 461 4217 root86Tail15 winner.bound = root86Candidates14 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root86Step14 : scan edge (search edge 3) 4 [461] 461 root86Tail14 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail15 winner := by
  rw [root86Tail14, scan, if_neg (by decide), root86Candidates14_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail15
    (if 3 ≤ root86Candidates14.length then search edge 3 [461, 4217] root86Candidates14 4678 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates15 : List Nat := [17021, 17189, 20549]
set_option maxRecDepth 8192 in
theorem root86Candidates15_checked : candidates edge 4 461 4271 root86Tail16 winner.bound = root86Candidates15 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root86Child15 : search edge 3 [461, 4271] root86Candidates15 4732 winner = winner := by decide
theorem root86Step15 : scan edge (search edge 3) 4 [461] 461 root86Tail15 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail16 winner := by
  rw [root86Tail15, scan, if_neg (by decide), root86Candidates15_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail16
    (if 3 ≤ root86Candidates15.length then search edge 3 [461, 4271] root86Candidates15 4732 winner else winner) = _
  rw [if_pos (by decide), root86Child15]
def root86Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root86Candidates16_checked : candidates edge 4 461 4637 root86Tail17 winner.bound = root86Candidates16 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root86Step16 : scan edge (search edge 3) 4 [461] 461 root86Tail16 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail17 winner := by
  rw [root86Tail16, scan, if_neg (by decide), root86Candidates16_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail17
    (if 3 ≤ root86Candidates16.length then search edge 3 [461, 4637] root86Candidates16 5098 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates17 : List Nat := [9521]
set_option maxRecDepth 8192 in
theorem root86Candidates17_checked : candidates edge 4 461 4937 root86Tail18 winner.bound = root86Candidates17 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root86Step17 : scan edge (search edge 3) 4 [461] 461 root86Tail17 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail18 winner := by
  rw [root86Tail17, scan, if_neg (by decide), root86Candidates17_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail18
    (if 3 ≤ root86Candidates17.length then search edge 3 [461, 4937] root86Candidates17 5398 winner else winner) = _
  rw [if_neg (by decide)]
def root86Candidates18 : List Nat := [18077]
set_option maxRecDepth 8192 in
theorem root86Candidates18_checked : candidates edge 4 461 5783 root86Tail19 winner.bound = root86Candidates18 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root86Step18 : scan edge (search edge 3) 4 [461] 461 root86Tail18 winner =
    scan edge (search edge 3) 4 [461] 461 root86Tail19 winner := by
  rw [root86Tail18, scan, if_neg (by decide), root86Candidates18_checked]
  change scan edge (search edge 3) 4 [461] 461 root86Tail19
    (if 3 ≤ root86Candidates18.length then search edge 3 [461, 5783] root86Candidates18 6244 winner else winner) = _
  rw [if_neg (by decide)]
theorem root86Scan19 : scan edge (search edge 3) 4 [461] 461 root86Tail19 winner = winner := by rw [root86Tail19, scan, if_pos (by decide)]
theorem root86Scan18 : scan edge (search edge 3) 4 [461] 461 root86Tail18 winner = winner :=
  (root86Step18).trans root86Scan19
theorem root86Scan17 : scan edge (search edge 3) 4 [461] 461 root86Tail17 winner = winner :=
  (root86Step17).trans root86Scan18
theorem root86Scan16 : scan edge (search edge 3) 4 [461] 461 root86Tail16 winner = winner :=
  (root86Step16).trans root86Scan17
theorem root86Scan15 : scan edge (search edge 3) 4 [461] 461 root86Tail15 winner = winner :=
  (root86Step15).trans root86Scan16
theorem root86Scan14 : scan edge (search edge 3) 4 [461] 461 root86Tail14 winner = winner :=
  (root86Step14).trans root86Scan15
theorem root86Scan13 : scan edge (search edge 3) 4 [461] 461 root86Tail13 winner = winner :=
  (root86Step13).trans root86Scan14
theorem root86Scan12 : scan edge (search edge 3) 4 [461] 461 root86Tail12 winner = winner :=
  (root86Step12).trans root86Scan13
theorem root86Scan11 : scan edge (search edge 3) 4 [461] 461 root86Tail11 winner = winner :=
  (root86Step11).trans root86Scan12
theorem root86Scan10 : scan edge (search edge 3) 4 [461] 461 root86Tail10 winner = winner :=
  (root86Step10).trans root86Scan11
theorem root86Scan9 : scan edge (search edge 3) 4 [461] 461 root86Tail9 winner = winner :=
  (root86Step9).trans root86Scan10
theorem root86Scan8 : scan edge (search edge 3) 4 [461] 461 root86Tail8 winner = winner :=
  (root86Step8).trans root86Scan9
theorem root86Scan7 : scan edge (search edge 3) 4 [461] 461 root86Tail7 winner = winner :=
  (root86Step7).trans root86Scan8
theorem root86Scan6 : scan edge (search edge 3) 4 [461] 461 root86Tail6 winner = winner :=
  (root86Step6).trans root86Scan7
theorem root86Scan5 : scan edge (search edge 3) 4 [461] 461 root86Tail5 winner = winner :=
  (root86Step5).trans root86Scan6
theorem root86Scan4 : scan edge (search edge 3) 4 [461] 461 root86Tail4 winner = winner :=
  (root86Step4).trans root86Scan5
theorem root86Scan3 : scan edge (search edge 3) 4 [461] 461 root86Tail3 winner = winner :=
  (root86Step3).trans root86Scan4
theorem root86Scan2 : scan edge (search edge 3) 4 [461] 461 root86Tail2 winner = winner :=
  (root86Step2).trans root86Scan3
theorem root86Scan1 : scan edge (search edge 3) 4 [461] 461 root86Tail1 winner = winner :=
  (root86Step1).trans root86Scan2
theorem root86Scan0 : scan edge (search edge 3) 4 [461] 461 root86Tail0 winner = winner :=
  (root86Step0).trans root86Scan1
set_option maxRecDepth 4096 in
theorem root86Branch : search edge 4 [461] root86Candidates 461 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [461] 461 root86Tail0 winner = winner
  exact root86Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
