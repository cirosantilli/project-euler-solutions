import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root27Candidates : List Nat := Data.row113
set_option maxRecDepth 8192 in
theorem root27Candidates_checked : candidates edge 5 0 113 tail28 winner.bound = root27Candidates := by
  apply candidates_eq_row edge 5 0 113 tail28 root27Candidates winner.bound tail28_ordered
  · decide
  · decide
  · funext q
    rfl
def root27Tail67 : List Nat := []
def root27Tail66 : List Nat := 25841 :: root27Tail67
def root27Tail65 : List Nat := 25793 :: root27Tail66
def root27Tail64 : List Nat := 24749 :: root27Tail65
def root27Tail63 : List Nat := 24623 :: root27Tail64
def root27Tail62 : List Nat := 24419 :: root27Tail63
def root27Tail61 : List Nat := 24281 :: root27Tail62
def root27Tail60 : List Nat := 24113 :: root27Tail61
def root27Tail59 : List Nat := 24029 :: root27Tail60
def root27Tail58 : List Nat := 23063 :: root27Tail59
def root27Tail57 : List Nat := 23057 :: root27Tail58
def root27Tail56 : List Nat := 23027 :: root27Tail57
def root27Tail55 : List Nat := 22769 :: root27Tail56
def root27Tail54 : List Nat := 22349 :: root27Tail55
def root27Tail53 : List Nat := 22193 :: root27Tail54
def root27Tail52 : List Nat := 19697 :: root27Tail53
def root27Tail51 : List Nat := 19211 :: root27Tail52
def root27Tail50 : List Nat := 19037 :: root27Tail51
def root27Tail49 : List Nat := 19031 :: root27Tail50
def root27Tail48 : List Nat := 18701 :: root27Tail49
def root27Tail47 : List Nat := 18401 :: root27Tail48
def root27Tail46 : List Nat := 17477 :: root27Tail47
def root27Tail45 : List Nat := 16187 :: root27Tail46
def root27Tail44 : List Nat := 16127 :: root27Tail45
def root27Tail43 : List Nat := 15809 :: root27Tail44
def root27Tail42 : List Nat := 15641 :: root27Tail43
def root27Tail41 : List Nat := 15581 :: root27Tail42
def root27Tail40 : List Nat := 14081 :: root27Tail41
def root27Tail39 : List Nat := 13877 :: root27Tail40
def root27Tail38 : List Nat := 13697 :: root27Tail39
def root27Tail37 : List Nat := 13691 :: root27Tail38
def root27Tail36 : List Nat := 11939 :: root27Tail37
def root27Tail35 : List Nat := 10883 :: root27Tail36
def root27Tail34 : List Nat := 10709 :: root27Tail35
def root27Tail33 : List Nat := 10433 :: root27Tail34
def root27Tail32 : List Nat := 10223 :: root27Tail33
def root27Tail31 : List Nat := 9851 :: root27Tail32
def root27Tail30 : List Nat := 9461 :: root27Tail31
def root27Tail29 : List Nat := 8999 :: root27Tail30
def root27Tail28 : List Nat := 8237 :: root27Tail29
def root27Tail27 : List Nat := 7457 :: root27Tail28
def root27Tail26 : List Nat := 6869 :: root27Tail27
def root27Tail25 : List Nat := 6719 :: root27Tail26
def root27Tail24 : List Nat := 6659 :: root27Tail25
def root27Tail23 : List Nat := 6389 :: root27Tail24
def root27Tail22 : List Nat := 5711 :: root27Tail23
def root27Tail21 : List Nat := 5333 :: root27Tail22
def root27Tail20 : List Nat := 4517 :: root27Tail21
def root27Tail19 : List Nat := 4421 :: root27Tail20
def root27Tail18 : List Nat := 3947 :: root27Tail19
def root27Tail17 : List Nat := 3659 :: root27Tail18
def root27Tail16 : List Nat := 3533 :: root27Tail17
def root27Tail15 : List Nat := 3359 :: root27Tail16
def root27Tail14 : List Nat := 3191 :: root27Tail15
def root27Tail13 : List Nat := 2861 :: root27Tail14
def root27Tail12 : List Nat := 2309 :: root27Tail13
def root27Tail11 : List Nat := 1973 :: root27Tail12
def root27Tail10 : List Nat := 1259 :: root27Tail11
def root27Tail9 : List Nat := 1217 :: root27Tail10
def root27Tail8 : List Nat := 983 :: root27Tail9
def root27Tail7 : List Nat := 761 :: root27Tail8
def root27Tail6 : List Nat := 647 :: root27Tail7
def root27Tail5 : List Nat := 383 :: root27Tail6
def root27Tail4 : List Nat := 233 :: root27Tail5
def root27Tail3 : List Nat := 227 :: root27Tail4
def root27Tail2 : List Nat := 167 :: root27Tail3
def root27Tail1 : List Nat := 149 :: root27Tail2
def root27Tail0 : List Nat := 131 :: root27Tail1
def root27Candidates0 : List Nat := [1217]
set_option maxRecDepth 8192 in
theorem root27Candidates0_checked : candidates edge 4 113 131 root27Tail1 winner.bound = root27Candidates0 := by
  unfold candidates
  have row : edge 131 = (SortedNatTree.contains FastRows.row131) := by
    change Data.row131.contains = (SortedNatTree.contains FastRows.row131)
    exact FastRows.row131_checked.symm
  rw [row]
  decide
theorem root27Step0 : scan edge (search edge 3) 4 [113] 113 root27Tail0 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail1 winner := by
  rw [root27Tail0, scan, if_neg (by decide), root27Candidates0_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail1
    (if 3 ≤ root27Candidates0.length then search edge 3 [113, 131] root27Candidates0 244 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates1 : List Nat := [19211]
set_option maxRecDepth 8192 in
theorem root27Candidates1_checked : candidates edge 4 113 149 root27Tail2 winner.bound = root27Candidates1 := by
  unfold candidates
  have row : edge 149 = (SortedNatTree.contains FastRows.row149) := by
    change Data.row149.contains = (SortedNatTree.contains FastRows.row149)
    exact FastRows.row149_checked.symm
  rw [row]
  decide
theorem root27Step1 : scan edge (search edge 3) 4 [113] 113 root27Tail1 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail2 winner := by
  rw [root27Tail1, scan, if_neg (by decide), root27Candidates1_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail2
    (if 3 ≤ root27Candidates1.length then search edge 3 [113, 149] root27Candidates1 262 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root27Candidates2_checked : candidates edge 4 113 167 root27Tail3 winner.bound = root27Candidates2 := by
  unfold candidates
  have row : edge 167 = (SortedNatTree.contains FastRows.row167) := by
    change Data.row167.contains = (SortedNatTree.contains FastRows.row167)
    exact FastRows.row167_checked.symm
  rw [row]
  decide
theorem root27Step2 : scan edge (search edge 3) 4 [113] 113 root27Tail2 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail3 winner := by
  rw [root27Tail2, scan, if_neg (by decide), root27Candidates2_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail3
    (if 3 ≤ root27Candidates2.length then search edge 3 [113, 167] root27Candidates2 280 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates3 : List Nat := [4421, 5333, 24749]
set_option maxRecDepth 8192 in
theorem root27Candidates3_checked : candidates edge 4 113 227 root27Tail4 winner.bound = root27Candidates3 := by
  unfold candidates
  have row : edge 227 = (SortedNatTree.contains FastRows.row227) := by
    change Data.row227.contains = (SortedNatTree.contains FastRows.row227)
    exact FastRows.row227_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root27Child3 : search edge 3 [113, 227] root27Candidates3 340 winner = winner := by decide
theorem root27Step3 : scan edge (search edge 3) 4 [113] 113 root27Tail3 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail4 winner := by
  rw [root27Tail3, scan, if_neg (by decide), root27Candidates3_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail4
    (if 3 ≤ root27Candidates3.length then search edge 3 [113, 227] root27Candidates3 340 winner else winner) = _
  rw [if_pos (by decide), root27Child3]
def root27Candidates4 : List Nat := [983, 24029]
set_option maxRecDepth 8192 in
theorem root27Candidates4_checked : candidates edge 4 113 233 root27Tail5 winner.bound = root27Candidates4 := by
  unfold candidates
  have row : edge 233 = (SortedNatTree.contains FastRows.row233) := by
    change Data.row233.contains = (SortedNatTree.contains FastRows.row233)
    exact FastRows.row233_checked.symm
  rw [row]
  decide
theorem root27Step4 : scan edge (search edge 3) 4 [113] 113 root27Tail4 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail5 winner := by
  rw [root27Tail4, scan, if_neg (by decide), root27Candidates4_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail5
    (if 3 ≤ root27Candidates4.length then search edge 3 [113, 233] root27Candidates4 346 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates5 : List Nat := [3359]
set_option maxRecDepth 8192 in
theorem root27Candidates5_checked : candidates edge 4 113 383 root27Tail6 winner.bound = root27Candidates5 := by
  unfold candidates
  have row : edge 383 = (SortedNatTree.contains FastRows.row383) := by
    change Data.row383.contains = (SortedNatTree.contains FastRows.row383)
    exact FastRows.row383_checked.symm
  rw [row]
  decide
theorem root27Step5 : scan edge (search edge 3) 4 [113] 113 root27Tail5 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail6 winner := by
  rw [root27Tail5, scan, if_neg (by decide), root27Candidates5_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail6
    (if 3 ≤ root27Candidates5.length then search edge 3 [113, 383] root27Candidates5 496 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates6 : List Nat := [24623]
set_option maxRecDepth 8192 in
theorem root27Candidates6_checked : candidates edge 4 113 647 root27Tail7 winner.bound = root27Candidates6 := by
  unfold candidates
  have row : edge 647 = (SortedNatTree.contains FastRows.row647) := by
    change Data.row647.contains = (SortedNatTree.contains FastRows.row647)
    exact FastRows.row647_checked.symm
  rw [row]
  decide
theorem root27Step6 : scan edge (search edge 3) 4 [113] 113 root27Tail6 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail7 winner := by
  rw [root27Tail6, scan, if_neg (by decide), root27Candidates6_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail7
    (if 3 ≤ root27Candidates6.length then search edge 3 [113, 647] root27Candidates6 760 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates7 : List Nat := [19697]
set_option maxRecDepth 8192 in
theorem root27Candidates7_checked : candidates edge 4 113 761 root27Tail8 winner.bound = root27Candidates7 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
theorem root27Step7 : scan edge (search edge 3) 4 [113] 113 root27Tail7 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail8 winner := by
  rw [root27Tail7, scan, if_neg (by decide), root27Candidates7_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail8
    (if 3 ≤ root27Candidates7.length then search edge 3 [113, 761] root27Candidates7 874 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates8 : List Nat := [1259, 6389, 6719]
set_option maxRecDepth 8192 in
theorem root27Candidates8_checked : candidates edge 4 113 983 root27Tail9 winner.bound = root27Candidates8 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root27Child8 : search edge 3 [113, 983] root27Candidates8 1096 winner = winner := by decide
theorem root27Step8 : scan edge (search edge 3) 4 [113] 113 root27Tail8 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail9 winner := by
  rw [root27Tail8, scan, if_neg (by decide), root27Candidates8_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail9
    (if 3 ≤ root27Candidates8.length then search edge 3 [113, 983] root27Candidates8 1096 winner else winner) = _
  rw [if_pos (by decide), root27Child8]
def root27Candidates9 : List Nat := [2309, 24419]
set_option maxRecDepth 8192 in
theorem root27Candidates9_checked : candidates edge 4 113 1217 root27Tail10 winner.bound = root27Candidates9 := by
  unfold candidates
  have row : edge 1217 = (SortedNatTree.contains FastRows.row1217) := by
    change Data.row1217.contains = (SortedNatTree.contains FastRows.row1217)
    exact FastRows.row1217_checked.symm
  rw [row]
  decide
theorem root27Step9 : scan edge (search edge 3) 4 [113] 113 root27Tail9 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail10 winner := by
  rw [root27Tail9, scan, if_neg (by decide), root27Candidates9_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail10
    (if 3 ≤ root27Candidates9.length then search edge 3 [113, 1217] root27Candidates9 1330 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root27Candidates10_checked : candidates edge 4 113 1259 root27Tail11 winner.bound = root27Candidates10 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
theorem root27Step10 : scan edge (search edge 3) 4 [113] 113 root27Tail10 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail11 winner := by
  rw [root27Tail10, scan, if_neg (by decide), root27Candidates10_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail11
    (if 3 ≤ root27Candidates10.length then search edge 3 [113, 1259] root27Candidates10 1372 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates11 : List Nat := [5711, 10433]
set_option maxRecDepth 8192 in
theorem root27Candidates11_checked : candidates edge 4 113 1973 root27Tail12 winner.bound = root27Candidates11 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
theorem root27Step11 : scan edge (search edge 3) 4 [113] 113 root27Tail11 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail12 winner := by
  rw [root27Tail11, scan, if_neg (by decide), root27Candidates11_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail12
    (if 3 ≤ root27Candidates11.length then search edge 3 [113, 1973] root27Candidates11 2086 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates12 : List Nat := [8237, 15809, 19697]
set_option maxRecDepth 8192 in
theorem root27Candidates12_checked : candidates edge 4 113 2309 root27Tail13 winner.bound = root27Candidates12 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root27Child12 : search edge 3 [113, 2309] root27Candidates12 2422 winner = winner := by decide
theorem root27Step12 : scan edge (search edge 3) 4 [113] 113 root27Tail12 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail13 winner := by
  rw [root27Tail12, scan, if_neg (by decide), root27Candidates12_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail13
    (if 3 ≤ root27Candidates12.length then search edge 3 [113, 2309] root27Candidates12 2422 winner else winner) = _
  rw [if_pos (by decide), root27Child12]
def root27Candidates13 : List Nat := [6719]
set_option maxRecDepth 8192 in
theorem root27Candidates13_checked : candidates edge 4 113 2861 root27Tail14 winner.bound = root27Candidates13 := by
  unfold candidates
  have row : edge 2861 = (SortedNatTree.contains FastRows.row2861) := by
    change Data.row2861.contains = (SortedNatTree.contains FastRows.row2861)
    exact FastRows.row2861_checked.symm
  rw [row]
  decide
theorem root27Step13 : scan edge (search edge 3) 4 [113] 113 root27Tail13 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail14 winner := by
  rw [root27Tail13, scan, if_neg (by decide), root27Candidates13_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail14
    (if 3 ≤ root27Candidates13.length then search edge 3 [113, 2861] root27Candidates13 2974 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root27Candidates14_checked : candidates edge 4 113 3191 root27Tail15 winner.bound = root27Candidates14 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
theorem root27Step14 : scan edge (search edge 3) 4 [113] 113 root27Tail14 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail15 winner := by
  rw [root27Tail14, scan, if_neg (by decide), root27Candidates14_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail15
    (if 3 ≤ root27Candidates14.length then search edge 3 [113, 3191] root27Candidates14 3304 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root27Candidates15_checked : candidates edge 4 113 3359 root27Tail16 winner.bound = root27Candidates15 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root27Step15 : scan edge (search edge 3) 4 [113] 113 root27Tail15 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail16 winner := by
  rw [root27Tail15, scan, if_neg (by decide), root27Candidates15_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail16
    (if 3 ≤ root27Candidates15.length then search edge 3 [113, 3359] root27Candidates15 3472 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates16 : List Nat := [10709, 22349]
set_option maxRecDepth 8192 in
theorem root27Candidates16_checked : candidates edge 4 113 3533 root27Tail17 winner.bound = root27Candidates16 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root27Step16 : scan edge (search edge 3) 4 [113] 113 root27Tail16 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail17 winner := by
  rw [root27Tail16, scan, if_neg (by decide), root27Candidates16_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail17
    (if 3 ≤ root27Candidates16.length then search edge 3 [113, 3533] root27Candidates16 3646 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates17 : List Nat := [9851, 13697]
set_option maxRecDepth 8192 in
theorem root27Candidates17_checked : candidates edge 4 113 3659 root27Tail18 winner.bound = root27Candidates17 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
theorem root27Step17 : scan edge (search edge 3) 4 [113] 113 root27Tail17 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail18 winner := by
  rw [root27Tail17, scan, if_neg (by decide), root27Candidates17_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail18
    (if 3 ≤ root27Candidates17.length then search edge 3 [113, 3659] root27Candidates17 3772 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates18 : List Nat := [14081]
set_option maxRecDepth 8192 in
theorem root27Candidates18_checked : candidates edge 4 113 3947 root27Tail19 winner.bound = root27Candidates18 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root27Step18 : scan edge (search edge 3) 4 [113] 113 root27Tail18 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail19 winner := by
  rw [root27Tail18, scan, if_neg (by decide), root27Candidates18_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail19
    (if 3 ≤ root27Candidates18.length then search edge 3 [113, 3947] root27Candidates18 4060 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates19 : List Nat := [6869, 10883]
set_option maxRecDepth 8192 in
theorem root27Candidates19_checked : candidates edge 4 113 4421 root27Tail20 winner.bound = root27Candidates19 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root27Step19 : scan edge (search edge 3) 4 [113] 113 root27Tail19 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail20 winner := by
  rw [root27Tail19, scan, if_neg (by decide), root27Candidates19_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail20
    (if 3 ≤ root27Candidates19.length then search edge 3 [113, 4421] root27Candidates19 4534 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates20 : List Nat := [6869]
set_option maxRecDepth 8192 in
theorem root27Candidates20_checked : candidates edge 4 113 4517 root27Tail21 winner.bound = root27Candidates20 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root27Step20 : scan edge (search edge 3) 4 [113] 113 root27Tail20 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail21 winner := by
  rw [root27Tail20, scan, if_neg (by decide), root27Candidates20_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail21
    (if 3 ≤ root27Candidates20.length then search edge 3 [113, 4517] root27Candidates20 4630 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates21 : List Nat := [6719]
set_option maxRecDepth 8192 in
theorem root27Candidates21_checked : candidates edge 4 113 5333 root27Tail22 winner.bound = root27Candidates21 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root27Step21 : scan edge (search edge 3) 4 [113] 113 root27Tail21 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail22 winner := by
  rw [root27Tail21, scan, if_neg (by decide), root27Candidates21_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail22
    (if 3 ≤ root27Candidates21.length then search edge 3 [113, 5333] root27Candidates21 5446 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root27Candidates22_checked : candidates edge 4 113 5711 root27Tail23 winner.bound = root27Candidates22 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root27Step22 : scan edge (search edge 3) 4 [113] 113 root27Tail22 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail23 winner := by
  rw [root27Tail22, scan, if_neg (by decide), root27Candidates22_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail23
    (if 3 ≤ root27Candidates22.length then search edge 3 [113, 5711] root27Candidates22 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root27Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root27Candidates23_checked : candidates edge 4 113 6389 root27Tail24 winner.bound = root27Candidates23 := by
  unfold candidates
  have row : edge 6389 = (SortedNatTree.contains FastRows.row6389) := by
    change Data.row6389.contains = (SortedNatTree.contains FastRows.row6389)
    exact FastRows.row6389_checked.symm
  rw [row]
  decide
theorem root27Step23 : scan edge (search edge 3) 4 [113] 113 root27Tail23 winner =
    scan edge (search edge 3) 4 [113] 113 root27Tail24 winner := by
  rw [root27Tail23, scan, if_neg (by decide), root27Candidates23_checked]
  change scan edge (search edge 3) 4 [113] 113 root27Tail24
    (if 3 ≤ root27Candidates23.length then search edge 3 [113, 6389] root27Candidates23 6502 winner else winner) = _
  rw [if_neg (by decide)]
theorem root27Scan24 : scan edge (search edge 3) 4 [113] 113 root27Tail24 winner = winner := by rw [root27Tail24, scan, if_pos (by decide)]
theorem root27Scan23 : scan edge (search edge 3) 4 [113] 113 root27Tail23 winner = winner :=
  (root27Step23).trans root27Scan24
theorem root27Scan22 : scan edge (search edge 3) 4 [113] 113 root27Tail22 winner = winner :=
  (root27Step22).trans root27Scan23
theorem root27Scan21 : scan edge (search edge 3) 4 [113] 113 root27Tail21 winner = winner :=
  (root27Step21).trans root27Scan22
theorem root27Scan20 : scan edge (search edge 3) 4 [113] 113 root27Tail20 winner = winner :=
  (root27Step20).trans root27Scan21
theorem root27Scan19 : scan edge (search edge 3) 4 [113] 113 root27Tail19 winner = winner :=
  (root27Step19).trans root27Scan20
theorem root27Scan18 : scan edge (search edge 3) 4 [113] 113 root27Tail18 winner = winner :=
  (root27Step18).trans root27Scan19
theorem root27Scan17 : scan edge (search edge 3) 4 [113] 113 root27Tail17 winner = winner :=
  (root27Step17).trans root27Scan18
theorem root27Scan16 : scan edge (search edge 3) 4 [113] 113 root27Tail16 winner = winner :=
  (root27Step16).trans root27Scan17
theorem root27Scan15 : scan edge (search edge 3) 4 [113] 113 root27Tail15 winner = winner :=
  (root27Step15).trans root27Scan16
theorem root27Scan14 : scan edge (search edge 3) 4 [113] 113 root27Tail14 winner = winner :=
  (root27Step14).trans root27Scan15
theorem root27Scan13 : scan edge (search edge 3) 4 [113] 113 root27Tail13 winner = winner :=
  (root27Step13).trans root27Scan14
theorem root27Scan12 : scan edge (search edge 3) 4 [113] 113 root27Tail12 winner = winner :=
  (root27Step12).trans root27Scan13
theorem root27Scan11 : scan edge (search edge 3) 4 [113] 113 root27Tail11 winner = winner :=
  (root27Step11).trans root27Scan12
theorem root27Scan10 : scan edge (search edge 3) 4 [113] 113 root27Tail10 winner = winner :=
  (root27Step10).trans root27Scan11
theorem root27Scan9 : scan edge (search edge 3) 4 [113] 113 root27Tail9 winner = winner :=
  (root27Step9).trans root27Scan10
theorem root27Scan8 : scan edge (search edge 3) 4 [113] 113 root27Tail8 winner = winner :=
  (root27Step8).trans root27Scan9
theorem root27Scan7 : scan edge (search edge 3) 4 [113] 113 root27Tail7 winner = winner :=
  (root27Step7).trans root27Scan8
theorem root27Scan6 : scan edge (search edge 3) 4 [113] 113 root27Tail6 winner = winner :=
  (root27Step6).trans root27Scan7
theorem root27Scan5 : scan edge (search edge 3) 4 [113] 113 root27Tail5 winner = winner :=
  (root27Step5).trans root27Scan6
theorem root27Scan4 : scan edge (search edge 3) 4 [113] 113 root27Tail4 winner = winner :=
  (root27Step4).trans root27Scan5
theorem root27Scan3 : scan edge (search edge 3) 4 [113] 113 root27Tail3 winner = winner :=
  (root27Step3).trans root27Scan4
theorem root27Scan2 : scan edge (search edge 3) 4 [113] 113 root27Tail2 winner = winner :=
  (root27Step2).trans root27Scan3
theorem root27Scan1 : scan edge (search edge 3) 4 [113] 113 root27Tail1 winner = winner :=
  (root27Step1).trans root27Scan2
theorem root27Scan0 : scan edge (search edge 3) 4 [113] 113 root27Tail0 winner = winner :=
  (root27Step0).trans root27Scan1
set_option maxRecDepth 4096 in
theorem root27Branch : search edge 4 [113] root27Candidates 113 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [113] 113 root27Tail0 winner = winner
  exact root27Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
