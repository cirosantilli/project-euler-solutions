import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root123Candidates : List Nat := Data.row701
set_option maxRecDepth 8192 in
theorem root123Candidates_checked : candidates edge 5 0 701 tail124 winner.bound = root123Candidates := by
  apply candidates_eq_row edge 5 0 701 tail124 root123Candidates winner.bound tail124_ordered
  · decide
  · decide
  · funext q
    rfl
def root123Tail73 : List Nat := []
def root123Tail72 : List Nat := 25163 :: root123Tail73
def root123Tail71 : List Nat := 24251 :: root123Tail72
def root123Tail70 : List Nat := 21929 :: root123Tail71
def root123Tail69 : List Nat := 21881 :: root123Tail70
def root123Tail68 : List Nat := 21599 :: root123Tail69
def root123Tail67 : List Nat := 21377 :: root123Tail68
def root123Tail66 : List Nat := 21017 :: root123Tail67
def root123Tail65 : List Nat := 20639 :: root123Tail66
def root123Tail64 : List Nat := 20549 :: root123Tail65
def root123Tail63 : List Nat := 18749 :: root123Tail64
def root123Tail62 : List Nat := 18701 :: root123Tail63
def root123Tail61 : List Nat := 18233 :: root123Tail62
def root123Tail60 : List Nat := 17807 :: root123Tail61
def root123Tail59 : List Nat := 16931 :: root123Tail60
def root123Tail58 : List Nat := 16901 :: root123Tail59
def root123Tail57 : List Nat := 16217 :: root123Tail58
def root123Tail56 : List Nat := 16001 :: root123Tail57
def root123Tail55 : List Nat := 15887 :: root123Tail56
def root123Tail54 : List Nat := 15809 :: root123Tail55
def root123Tail53 : List Nat := 15473 :: root123Tail54
def root123Tail52 : List Nat := 15077 :: root123Tail53
def root123Tail51 : List Nat := 14867 :: root123Tail52
def root123Tail50 : List Nat := 14657 :: root123Tail51
def root123Tail49 : List Nat := 14411 :: root123Tail50
def root123Tail48 : List Nat := 13313 :: root123Tail49
def root123Tail47 : List Nat := 12953 :: root123Tail48
def root123Tail46 : List Nat := 12491 :: root123Tail47
def root123Tail45 : List Nat := 12239 :: root123Tail46
def root123Tail44 : List Nat := 12197 :: root123Tail45
def root123Tail43 : List Nat := 11969 :: root123Tail44
def root123Tail42 : List Nat := 11807 :: root123Tail43
def root123Tail41 : List Nat := 11351 :: root123Tail42
def root123Tail40 : List Nat := 10163 :: root123Tail41
def root123Tail39 : List Nat := 9923 :: root123Tail40
def root123Tail38 : List Nat := 9851 :: root123Tail39
def root123Tail37 : List Nat := 9587 :: root123Tail38
def root123Tail36 : List Nat := 9473 :: root123Tail37
def root123Tail35 : List Nat := 9239 :: root123Tail36
def root123Tail34 : List Nat := 9137 :: root123Tail35
def root123Tail33 : List Nat := 8867 :: root123Tail34
def root123Tail32 : List Nat := 8747 :: root123Tail33
def root123Tail31 : List Nat := 8447 :: root123Tail32
def root123Tail30 : List Nat := 8171 :: root123Tail31
def root123Tail29 : List Nat := 7937 :: root123Tail30
def root123Tail28 : List Nat := 7919 :: root123Tail29
def root123Tail27 : List Nat := 7607 :: root123Tail28
def root123Tail26 : List Nat := 7541 :: root123Tail27
def root123Tail25 : List Nat := 6803 :: root123Tail26
def root123Tail24 : List Nat := 6389 :: root123Tail25
def root123Tail23 : List Nat := 6101 :: root123Tail24
def root123Tail22 : List Nat := 5501 :: root123Tail23
def root123Tail21 : List Nat := 5471 :: root123Tail22
def root123Tail20 : List Nat := 5417 :: root123Tail21
def root123Tail19 : List Nat := 5087 :: root123Tail20
def root123Tail18 : List Nat := 5003 :: root123Tail19
def root123Tail17 : List Nat := 4877 :: root123Tail18
def root123Tail16 : List Nat := 4751 :: root123Tail17
def root123Tail15 : List Nat := 4583 :: root123Tail16
def root123Tail14 : List Nat := 4373 :: root123Tail15
def root123Tail13 : List Nat := 4283 :: root123Tail14
def root123Tail12 : List Nat := 3623 :: root123Tail13
def root123Tail11 : List Nat := 3581 :: root123Tail12
def root123Tail10 : List Nat := 2477 :: root123Tail11
def root123Tail9 : List Nat := 2267 :: root123Tail10
def root123Tail8 : List Nat := 1913 :: root123Tail9
def root123Tail7 : List Nat := 1733 :: root123Tail8
def root123Tail6 : List Nat := 1607 :: root123Tail7
def root123Tail5 : List Nat := 1553 :: root123Tail6
def root123Tail4 : List Nat := 1439 :: root123Tail5
def root123Tail3 : List Nat := 1307 :: root123Tail4
def root123Tail2 : List Nat := 1283 :: root123Tail3
def root123Tail1 : List Nat := 1187 :: root123Tail2
def root123Tail0 : List Nat := 1181 :: root123Tail1
def root123Candidates0 : List Nat := [5501, 21881, 21929]
set_option maxRecDepth 8192 in
theorem root123Candidates0_checked : candidates edge 4 701 1181 root123Tail1 winner.bound = root123Candidates0 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child0 : search edge 3 [701, 1181] root123Candidates0 1882 winner = winner := by decide
theorem root123Step0 : scan edge (search edge 3) 4 [701] 701 root123Tail0 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail1 winner := by
  rw [root123Tail0, scan, if_neg (by decide), root123Candidates0_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail1
    (if 3 ≤ root123Candidates0.length then search edge 3 [701, 1181] root123Candidates0 1882 winner else winner) = _
  rw [if_pos (by decide), root123Child0]
def root123Candidates1 : List Nat := [12239, 15809, 16931]
set_option maxRecDepth 8192 in
theorem root123Candidates1_checked : candidates edge 4 701 1187 root123Tail2 winner.bound = root123Candidates1 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child1 : search edge 3 [701, 1187] root123Candidates1 1888 winner = winner := by decide
theorem root123Step1 : scan edge (search edge 3) 4 [701] 701 root123Tail1 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail2 winner := by
  rw [root123Tail1, scan, if_neg (by decide), root123Candidates1_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail2
    (if 3 ≤ root123Candidates1.length then search edge 3 [701, 1187] root123Candidates1 1888 winner else winner) = _
  rw [if_pos (by decide), root123Child1]
def root123Candidates2 : List Nat := [1913, 8747, 9137, 14411, 14867]
set_option maxRecDepth 8192 in
theorem root123Candidates2_checked : candidates edge 4 701 1283 root123Tail3 winner.bound = root123Candidates2 := by
  unfold candidates
  have row : edge 1283 = (SortedNatTree.contains FastRows.row1283) := by
    change Data.row1283.contains = (SortedNatTree.contains FastRows.row1283)
    exact FastRows.row1283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child2 : search edge 3 [701, 1283] root123Candidates2 1984 winner = winner := by decide
theorem root123Step2 : scan edge (search edge 3) 4 [701] 701 root123Tail2 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail3 winner := by
  rw [root123Tail2, scan, if_neg (by decide), root123Candidates2_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail3
    (if 3 ≤ root123Candidates2.length then search edge 3 [701, 1283] root123Candidates2 1984 winner else winner) = _
  rw [if_pos (by decide), root123Child2]
def root123Candidates3 : List Nat := [4583, 7541, 8171]
set_option maxRecDepth 8192 in
theorem root123Candidates3_checked : candidates edge 4 701 1307 root123Tail4 winner.bound = root123Candidates3 := by
  unfold candidates
  have row : edge 1307 = (SortedNatTree.contains FastRows.row1307) := by
    change Data.row1307.contains = (SortedNatTree.contains FastRows.row1307)
    exact FastRows.row1307_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child3 : search edge 3 [701, 1307] root123Candidates3 2008 winner = winner := by decide
theorem root123Step3 : scan edge (search edge 3) 4 [701] 701 root123Tail3 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail4 winner := by
  rw [root123Tail3, scan, if_neg (by decide), root123Candidates3_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail4
    (if 3 ≤ root123Candidates3.length then search edge 3 [701, 1307] root123Candidates3 2008 winner else winner) = _
  rw [if_pos (by decide), root123Child3]
def root123Candidates4 : List Nat := [9137]
set_option maxRecDepth 8192 in
theorem root123Candidates4_checked : candidates edge 4 701 1439 root123Tail5 winner.bound = root123Candidates4 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
theorem root123Step4 : scan edge (search edge 3) 4 [701] 701 root123Tail4 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail5 winner := by
  rw [root123Tail4, scan, if_neg (by decide), root123Candidates4_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail5
    (if 3 ≤ root123Candidates4.length then search edge 3 [701, 1439] root123Candidates4 2140 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates5 : List Nat := [15473]
set_option maxRecDepth 8192 in
theorem root123Candidates5_checked : candidates edge 4 701 1553 root123Tail6 winner.bound = root123Candidates5 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
theorem root123Step5 : scan edge (search edge 3) 4 [701] 701 root123Tail5 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail6 winner := by
  rw [root123Tail5, scan, if_neg (by decide), root123Candidates5_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail6
    (if 3 ≤ root123Candidates5.length then search edge 3 [701, 1553] root123Candidates5 2254 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates6 : List Nat := [5471, 7541, 8171, 16901]
set_option maxRecDepth 8192 in
theorem root123Candidates6_checked : candidates edge 4 701 1607 root123Tail7 winner.bound = root123Candidates6 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child6 : search edge 3 [701, 1607] root123Candidates6 2308 winner = winner := by decide
theorem root123Step6 : scan edge (search edge 3) 4 [701] 701 root123Tail6 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail7 winner := by
  rw [root123Tail6, scan, if_neg (by decide), root123Candidates6_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail7
    (if 3 ≤ root123Candidates6.length then search edge 3 [701, 1607] root123Candidates6 2308 winner else winner) = _
  rw [if_pos (by decide), root123Child6]
def root123Candidates7 : List Nat := [6101]
set_option maxRecDepth 8192 in
theorem root123Candidates7_checked : candidates edge 4 701 1733 root123Tail8 winner.bound = root123Candidates7 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
theorem root123Step7 : scan edge (search edge 3) 4 [701] 701 root123Tail7 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail8 winner := by
  rw [root123Tail7, scan, if_neg (by decide), root123Candidates7_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail8
    (if 3 ≤ root123Candidates7.length then search edge 3 [701, 1733] root123Candidates7 2434 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates8 : List Nat := [5087, 6389, 7919, 9851]
set_option maxRecDepth 8192 in
theorem root123Candidates8_checked : candidates edge 4 701 1913 root123Tail9 winner.bound = root123Candidates8 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child8 : search edge 3 [701, 1913] root123Candidates8 2614 winner = winner := by decide
theorem root123Step8 : scan edge (search edge 3) 4 [701] 701 root123Tail8 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail9 winner := by
  rw [root123Tail8, scan, if_neg (by decide), root123Candidates8_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail9
    (if 3 ≤ root123Candidates8.length then search edge 3 [701, 1913] root123Candidates8 2614 winner else winner) = _
  rw [if_pos (by decide), root123Child8]
def root123Candidates9 : List Nat := [6389]
set_option maxRecDepth 8192 in
theorem root123Candidates9_checked : candidates edge 4 701 2267 root123Tail10 winner.bound = root123Candidates9 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root123Step9 : scan edge (search edge 3) 4 [701] 701 root123Tail9 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail10 winner := by
  rw [root123Tail9, scan, if_neg (by decide), root123Candidates9_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail10
    (if 3 ≤ root123Candidates9.length then search edge 3 [701, 2267] root123Candidates9 2968 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root123Candidates10_checked : candidates edge 4 701 2477 root123Tail11 winner.bound = root123Candidates10 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
theorem root123Step10 : scan edge (search edge 3) 4 [701] 701 root123Tail10 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail11 winner := by
  rw [root123Tail10, scan, if_neg (by decide), root123Candidates10_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail11
    (if 3 ≤ root123Candidates10.length then search edge 3 [701, 2477] root123Candidates10 3178 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates11 : List Nat := [6101, 12239, 14411, 16217]
set_option maxRecDepth 8192 in
theorem root123Candidates11_checked : candidates edge 4 701 3581 root123Tail12 winner.bound = root123Candidates11 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child11 : search edge 3 [701, 3581] root123Candidates11 4282 winner = winner := by decide
theorem root123Step11 : scan edge (search edge 3) 4 [701] 701 root123Tail11 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail12 winner := by
  rw [root123Tail11, scan, if_neg (by decide), root123Candidates11_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail12
    (if 3 ≤ root123Candidates11.length then search edge 3 [701, 3581] root123Candidates11 4282 winner else winner) = _
  rw [if_pos (by decide), root123Child11]
def root123Candidates12 : List Nat := [16901]
set_option maxRecDepth 8192 in
theorem root123Candidates12_checked : candidates edge 4 701 3623 root123Tail13 winner.bound = root123Candidates12 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root123Step12 : scan edge (search edge 3) 4 [701] 701 root123Tail12 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail13 winner := by
  rw [root123Tail12, scan, if_neg (by decide), root123Candidates12_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail13
    (if 3 ≤ root123Candidates12.length then search edge 3 [701, 3623] root123Candidates12 4324 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates13 : List Nat := [8447, 14657, 16901]
set_option maxRecDepth 8192 in
theorem root123Candidates13_checked : candidates edge 4 701 4283 root123Tail14 winner.bound = root123Candidates13 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child13 : search edge 3 [701, 4283] root123Candidates13 4984 winner = winner := by decide
theorem root123Step13 : scan edge (search edge 3) 4 [701] 701 root123Tail13 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail14 winner := by
  rw [root123Tail13, scan, if_neg (by decide), root123Candidates13_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail14
    (if 3 ≤ root123Candidates13.length then search edge 3 [701, 4283] root123Candidates13 4984 winner else winner) = _
  rw [if_pos (by decide), root123Child13]
def root123Candidates14 : List Nat := [7541, 7919, 10163, 11807]
set_option maxRecDepth 8192 in
theorem root123Candidates14_checked : candidates edge 4 701 4373 root123Tail15 winner.bound = root123Candidates14 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child14 : search edge 3 [701, 4373] root123Candidates14 5074 winner = winner := by decide
theorem root123Step14 : scan edge (search edge 3) 4 [701] 701 root123Tail14 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail15 winner := by
  rw [root123Tail14, scan, if_neg (by decide), root123Candidates14_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail15
    (if 3 ≤ root123Candidates14.length then search edge 3 [701, 4373] root123Candidates14 5074 winner else winner) = _
  rw [if_pos (by decide), root123Child14]
def root123Candidates15 : List Nat := [5003, 9137]
set_option maxRecDepth 8192 in
theorem root123Candidates15_checked : candidates edge 4 701 4583 root123Tail16 winner.bound = root123Candidates15 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root123Step15 : scan edge (search edge 3) 4 [701] 701 root123Tail15 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail16 winner := by
  rw [root123Tail15, scan, if_neg (by decide), root123Candidates15_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail16
    (if 3 ≤ root123Candidates15.length then search edge 3 [701, 4583] root123Candidates15 5284 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates16 : List Nat := [7607, 18233]
set_option maxRecDepth 8192 in
theorem root123Candidates16_checked : candidates edge 4 701 4751 root123Tail17 winner.bound = root123Candidates16 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root123Step16 : scan edge (search edge 3) 4 [701] 701 root123Tail16 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail17 winner := by
  rw [root123Tail16, scan, if_neg (by decide), root123Candidates16_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail17
    (if 3 ≤ root123Candidates16.length then search edge 3 [701, 4751] root123Candidates16 5452 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates17 : List Nat := [5501]
set_option maxRecDepth 8192 in
theorem root123Candidates17_checked : candidates edge 4 701 4877 root123Tail18 winner.bound = root123Candidates17 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root123Step17 : scan edge (search edge 3) 4 [701] 701 root123Tail17 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail18 winner := by
  rw [root123Tail17, scan, if_neg (by decide), root123Candidates17_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail18
    (if 3 ≤ root123Candidates17.length then search edge 3 [701, 4877] root123Candidates17 5578 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates18 : List Nat := [5087, 7541, 18233]
set_option maxRecDepth 8192 in
theorem root123Candidates18_checked : candidates edge 4 701 5003 root123Tail19 winner.bound = root123Candidates18 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root123Child18 : search edge 3 [701, 5003] root123Candidates18 5704 winner = winner := by decide
theorem root123Step18 : scan edge (search edge 3) 4 [701] 701 root123Tail18 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail19 winner := by
  rw [root123Tail18, scan, if_neg (by decide), root123Candidates18_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail19
    (if 3 ≤ root123Candidates18.length then search edge 3 [701, 5003] root123Candidates18 5704 winner else winner) = _
  rw [if_pos (by decide), root123Child18]
def root123Candidates19 : List Nat := [6803, 14411]
set_option maxRecDepth 8192 in
theorem root123Candidates19_checked : candidates edge 4 701 5087 root123Tail20 winner.bound = root123Candidates19 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root123Step19 : scan edge (search edge 3) 4 [701] 701 root123Tail19 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail20 winner := by
  rw [root123Tail19, scan, if_neg (by decide), root123Candidates19_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail20
    (if 3 ≤ root123Candidates19.length then search edge 3 [701, 5087] root123Candidates19 5788 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates20 : List Nat := [9239]
set_option maxRecDepth 8192 in
theorem root123Candidates20_checked : candidates edge 4 701 5417 root123Tail21 winner.bound = root123Candidates20 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root123Step20 : scan edge (search edge 3) 4 [701] 701 root123Tail20 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail21 winner := by
  rw [root123Tail20, scan, if_neg (by decide), root123Candidates20_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail21
    (if 3 ≤ root123Candidates20.length then search edge 3 [701, 5417] root123Candidates20 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates21 : List Nat := [8171, 18233]
set_option maxRecDepth 8192 in
theorem root123Candidates21_checked : candidates edge 4 701 5471 root123Tail22 winner.bound = root123Candidates21 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root123Step21 : scan edge (search edge 3) 4 [701] 701 root123Tail21 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail22 winner := by
  rw [root123Tail21, scan, if_neg (by decide), root123Candidates21_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail22
    (if 3 ≤ root123Candidates21.length then search edge 3 [701, 5471] root123Candidates21 6172 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root123Candidates22_checked : candidates edge 4 701 5501 root123Tail23 winner.bound = root123Candidates22 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root123Step22 : scan edge (search edge 3) 4 [701] 701 root123Tail22 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail23 winner := by
  rw [root123Tail22, scan, if_neg (by decide), root123Candidates22_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail23
    (if 3 ≤ root123Candidates22.length then search edge 3 [701, 5501] root123Candidates22 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root123Candidates23 : List Nat := [9587]
set_option maxRecDepth 8192 in
theorem root123Candidates23_checked : candidates edge 4 701 6101 root123Tail24 winner.bound = root123Candidates23 := by
  unfold candidates
  have row : edge 6101 = (SortedNatTree.contains FastRows.row6101) := by
    change Data.row6101.contains = (SortedNatTree.contains FastRows.row6101)
    exact FastRows.row6101_checked.symm
  rw [row]
  decide
theorem root123Step23 : scan edge (search edge 3) 4 [701] 701 root123Tail23 winner =
    scan edge (search edge 3) 4 [701] 701 root123Tail24 winner := by
  rw [root123Tail23, scan, if_neg (by decide), root123Candidates23_checked]
  change scan edge (search edge 3) 4 [701] 701 root123Tail24
    (if 3 ≤ root123Candidates23.length then search edge 3 [701, 6101] root123Candidates23 6802 winner else winner) = _
  rw [if_neg (by decide)]
theorem root123Scan24 : scan edge (search edge 3) 4 [701] 701 root123Tail24 winner = winner := by rw [root123Tail24, scan, if_pos (by decide)]
theorem root123Scan23 : scan edge (search edge 3) 4 [701] 701 root123Tail23 winner = winner :=
  (root123Step23).trans root123Scan24
theorem root123Scan22 : scan edge (search edge 3) 4 [701] 701 root123Tail22 winner = winner :=
  (root123Step22).trans root123Scan23
theorem root123Scan21 : scan edge (search edge 3) 4 [701] 701 root123Tail21 winner = winner :=
  (root123Step21).trans root123Scan22
theorem root123Scan20 : scan edge (search edge 3) 4 [701] 701 root123Tail20 winner = winner :=
  (root123Step20).trans root123Scan21
theorem root123Scan19 : scan edge (search edge 3) 4 [701] 701 root123Tail19 winner = winner :=
  (root123Step19).trans root123Scan20
theorem root123Scan18 : scan edge (search edge 3) 4 [701] 701 root123Tail18 winner = winner :=
  (root123Step18).trans root123Scan19
theorem root123Scan17 : scan edge (search edge 3) 4 [701] 701 root123Tail17 winner = winner :=
  (root123Step17).trans root123Scan18
theorem root123Scan16 : scan edge (search edge 3) 4 [701] 701 root123Tail16 winner = winner :=
  (root123Step16).trans root123Scan17
theorem root123Scan15 : scan edge (search edge 3) 4 [701] 701 root123Tail15 winner = winner :=
  (root123Step15).trans root123Scan16
theorem root123Scan14 : scan edge (search edge 3) 4 [701] 701 root123Tail14 winner = winner :=
  (root123Step14).trans root123Scan15
theorem root123Scan13 : scan edge (search edge 3) 4 [701] 701 root123Tail13 winner = winner :=
  (root123Step13).trans root123Scan14
theorem root123Scan12 : scan edge (search edge 3) 4 [701] 701 root123Tail12 winner = winner :=
  (root123Step12).trans root123Scan13
theorem root123Scan11 : scan edge (search edge 3) 4 [701] 701 root123Tail11 winner = winner :=
  (root123Step11).trans root123Scan12
theorem root123Scan10 : scan edge (search edge 3) 4 [701] 701 root123Tail10 winner = winner :=
  (root123Step10).trans root123Scan11
theorem root123Scan9 : scan edge (search edge 3) 4 [701] 701 root123Tail9 winner = winner :=
  (root123Step9).trans root123Scan10
theorem root123Scan8 : scan edge (search edge 3) 4 [701] 701 root123Tail8 winner = winner :=
  (root123Step8).trans root123Scan9
theorem root123Scan7 : scan edge (search edge 3) 4 [701] 701 root123Tail7 winner = winner :=
  (root123Step7).trans root123Scan8
theorem root123Scan6 : scan edge (search edge 3) 4 [701] 701 root123Tail6 winner = winner :=
  (root123Step6).trans root123Scan7
theorem root123Scan5 : scan edge (search edge 3) 4 [701] 701 root123Tail5 winner = winner :=
  (root123Step5).trans root123Scan6
theorem root123Scan4 : scan edge (search edge 3) 4 [701] 701 root123Tail4 winner = winner :=
  (root123Step4).trans root123Scan5
theorem root123Scan3 : scan edge (search edge 3) 4 [701] 701 root123Tail3 winner = winner :=
  (root123Step3).trans root123Scan4
theorem root123Scan2 : scan edge (search edge 3) 4 [701] 701 root123Tail2 winner = winner :=
  (root123Step2).trans root123Scan3
theorem root123Scan1 : scan edge (search edge 3) 4 [701] 701 root123Tail1 winner = winner :=
  (root123Step1).trans root123Scan2
theorem root123Scan0 : scan edge (search edge 3) 4 [701] 701 root123Tail0 winner = winner :=
  (root123Step0).trans root123Scan1
set_option maxRecDepth 4096 in
theorem root123Branch : search edge 4 [701] root123Candidates 701 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [701] 701 root123Tail0 winner = winner
  exact root123Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
