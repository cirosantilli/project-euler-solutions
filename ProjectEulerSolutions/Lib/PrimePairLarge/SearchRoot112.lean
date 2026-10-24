import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root112Candidates : List Nat := Data.row631
set_option maxRecDepth 8192 in
theorem root112Candidates_checked : candidates edge 5 0 631 tail113 winner.bound = root112Candidates := by
  apply candidates_eq_row edge 5 0 631 tail113 root112Candidates winner.bound tail113_ordered
  · decide
  · decide
  · funext q
    rfl
def root112Tail68 : List Nat := []
def root112Tail67 : List Nat := 25171 :: root112Tail68
def root112Tail66 : List Nat := 24223 :: root112Tail67
def root112Tail65 : List Nat := 23131 :: root112Tail66
def root112Tail64 : List Nat := 22963 :: root112Tail65
def root112Tail63 : List Nat := 22573 :: root112Tail64
def root112Tail62 : List Nat := 22567 :: root112Tail63
def root112Tail61 : List Nat := 22453 :: root112Tail62
def root112Tail60 : List Nat := 22063 :: root112Tail61
def root112Tail59 : List Nat := 21601 :: root112Tail60
def root112Tail58 : List Nat := 21523 :: root112Tail59
def root112Tail57 : List Nat := 21013 :: root112Tail58
def root112Tail56 : List Nat := 20719 :: root112Tail57
def root112Tail55 : List Nat := 20479 :: root112Tail56
def root112Tail54 : List Nat := 19753 :: root112Tail55
def root112Tail53 : List Nat := 19687 :: root112Tail54
def root112Tail52 : List Nat := 19219 :: root112Tail53
def root112Tail51 : List Nat := 18199 :: root112Tail52
def root112Tail50 : List Nat := 18043 :: root112Tail51
def root112Tail49 : List Nat := 17863 :: root112Tail50
def root112Tail48 : List Nat := 17827 :: root112Tail49
def root112Tail47 : List Nat := 17419 :: root112Tail48
def root112Tail46 : List Nat := 17317 :: root112Tail47
def root112Tail45 : List Nat := 17191 :: root112Tail46
def root112Tail44 : List Nat := 16987 :: root112Tail45
def root112Tail43 : List Nat := 16567 :: root112Tail44
def root112Tail42 : List Nat := 16447 :: root112Tail43
def root112Tail41 : List Nat := 16369 :: root112Tail42
def root112Tail40 : List Nat := 16063 :: root112Tail41
def root112Tail39 : List Nat := 15991 :: root112Tail40
def root112Tail38 : List Nat := 14797 :: root112Tail39
def root112Tail37 : List Nat := 14629 :: root112Tail38
def root112Tail36 : List Nat := 13879 :: root112Tail37
def root112Tail35 : List Nat := 13669 :: root112Tail36
def root112Tail34 : List Nat := 13597 :: root112Tail35
def root112Tail33 : List Nat := 12037 :: root112Tail34
def root112Tail32 : List Nat := 11821 :: root112Tail33
def root112Tail31 : List Nat := 11329 :: root112Tail32
def root112Tail30 : List Nat := 10867 :: root112Tail31
def root112Tail29 : List Nat := 10639 :: root112Tail30
def root112Tail28 : List Nat := 9769 :: root112Tail29
def root112Tail27 : List Nat := 9613 :: root112Tail28
def root112Tail26 : List Nat := 9463 :: root112Tail27
def root112Tail25 : List Nat := 9127 :: root112Tail26
def root112Tail24 : List Nat := 9091 :: root112Tail25
def root112Tail23 : List Nat := 9007 :: root112Tail24
def root112Tail22 : List Nat := 8467 :: root112Tail23
def root112Tail21 : List Nat := 8161 :: root112Tail22
def root112Tail20 : List Nat := 7873 :: root112Tail21
def root112Tail19 : List Nat := 7681 :: root112Tail20
def root112Tail18 : List Nat := 7243 :: root112Tail19
def root112Tail17 : List Nat := 6199 :: root112Tail18
def root112Tail16 : List Nat := 5737 :: root112Tail17
def root112Tail15 : List Nat := 5557 :: root112Tail16
def root112Tail14 : List Nat := 5527 :: root112Tail15
def root112Tail13 : List Nat := 5479 :: root112Tail14
def root112Tail12 : List Nat := 4933 :: root112Tail13
def root112Tail11 : List Nat := 4801 :: root112Tail12
def root112Tail10 : List Nat := 4057 :: root112Tail11
def root112Tail9 : List Nat := 4051 :: root112Tail10
def root112Tail8 : List Nat := 3889 :: root112Tail9
def root112Tail7 : List Nat := 3709 :: root112Tail8
def root112Tail6 : List Nat := 3457 :: root112Tail7
def root112Tail5 : List Nat := 3253 :: root112Tail6
def root112Tail4 : List Nat := 2221 :: root112Tail5
def root112Tail3 : List Nat := 1483 :: root112Tail4
def root112Tail2 : List Nat := 1297 :: root112Tail3
def root112Tail1 : List Nat := 751 :: root112Tail2
def root112Tail0 : List Nat := 739 :: root112Tail1
def root112Candidates0 : List Nat := [751, 9463, 16987, 20719]
set_option maxRecDepth 8192 in
theorem root112Candidates0_checked : candidates edge 4 631 739 root112Tail1 winner.bound = root112Candidates0 := by
  unfold candidates
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child0 : search edge 3 [631, 739] root112Candidates0 1370 winner = winner := by decide
theorem root112Step0 : scan edge (search edge 3) 4 [631] 631 root112Tail0 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail1 winner := by
  rw [root112Tail0, scan, if_neg (by decide), root112Candidates0_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail1
    (if 3 ≤ root112Candidates0.length then search edge 3 [631, 739] root112Candidates0 1370 winner else winner) = _
  rw [if_pos (by decide), root112Child0]
def root112Candidates1 : List Nat := [24223]
set_option maxRecDepth 8192 in
theorem root112Candidates1_checked : candidates edge 4 631 751 root112Tail2 winner.bound = root112Candidates1 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
theorem root112Step1 : scan edge (search edge 3) 4 [631] 631 root112Tail1 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail2 winner := by
  rw [root112Tail1, scan, if_neg (by decide), root112Candidates1_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail2
    (if 3 ≤ root112Candidates1.length then search edge 3 [631, 751] root112Candidates1 1382 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root112Candidates2_checked : candidates edge 4 631 1297 root112Tail3 winner.bound = root112Candidates2 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
theorem root112Step2 : scan edge (search edge 3) 4 [631] 631 root112Tail2 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail3 winner := by
  rw [root112Tail2, scan, if_neg (by decide), root112Candidates2_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail3
    (if 3 ≤ root112Candidates2.length then search edge 3 [631, 1297] root112Candidates2 1928 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates3 : List Nat := [3457, 7873, 9127]
set_option maxRecDepth 8192 in
theorem root112Candidates3_checked : candidates edge 4 631 1483 root112Tail4 winner.bound = root112Candidates3 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child3 : search edge 3 [631, 1483] root112Candidates3 2114 winner = winner := by decide
theorem root112Step3 : scan edge (search edge 3) 4 [631] 631 root112Tail3 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail4 winner := by
  rw [root112Tail3, scan, if_neg (by decide), root112Candidates3_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail4
    (if 3 ≤ root112Candidates3.length then search edge 3 [631, 1483] root112Candidates3 2114 winner else winner) = _
  rw [if_pos (by decide), root112Child3]
def root112Candidates4 : List Nat := [5527, 5737, 7681, 7873]
set_option maxRecDepth 8192 in
theorem root112Candidates4_checked : candidates edge 4 631 2221 root112Tail5 winner.bound = root112Candidates4 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child4 : search edge 3 [631, 2221] root112Candidates4 2852 winner = winner := by decide
theorem root112Step4 : scan edge (search edge 3) 4 [631] 631 root112Tail4 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail5 winner := by
  rw [root112Tail4, scan, if_neg (by decide), root112Candidates4_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail5
    (if 3 ≤ root112Candidates4.length then search edge 3 [631, 2221] root112Candidates4 2852 winner else winner) = _
  rw [if_pos (by decide), root112Child4]
def root112Candidates5 : List Nat := [5479, 5557, 7873, 16567, 20719]
set_option maxRecDepth 8192 in
theorem root112Candidates5_checked : candidates edge 4 631 3253 root112Tail6 winner.bound = root112Candidates5 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child5 : search edge 3 [631, 3253] root112Candidates5 3884 winner = winner := by decide
theorem root112Step5 : scan edge (search edge 3) 4 [631] 631 root112Tail5 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail6 winner := by
  rw [root112Tail5, scan, if_neg (by decide), root112Candidates5_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail6
    (if 3 ≤ root112Candidates5.length then search edge 3 [631, 3253] root112Candidates5 3884 winner else winner) = _
  rw [if_pos (by decide), root112Child5]
def root112Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root112Candidates6_checked : candidates edge 4 631 3457 root112Tail7 winner.bound = root112Candidates6 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root112Step6 : scan edge (search edge 3) 4 [631] 631 root112Tail6 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail7 winner := by
  rw [root112Tail6, scan, if_neg (by decide), root112Candidates6_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail7
    (if 3 ≤ root112Candidates6.length then search edge 3 [631, 3457] root112Candidates6 4088 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates7 : List Nat := [19687]
set_option maxRecDepth 8192 in
theorem root112Candidates7_checked : candidates edge 4 631 3709 root112Tail8 winner.bound = root112Candidates7 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root112Step7 : scan edge (search edge 3) 4 [631] 631 root112Tail7 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail8 winner := by
  rw [root112Tail7, scan, if_neg (by decide), root112Candidates7_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail8
    (if 3 ≤ root112Candidates7.length then search edge 3 [631, 3709] root112Candidates7 4340 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates8 : List Nat := [13597, 17827]
set_option maxRecDepth 8192 in
theorem root112Candidates8_checked : candidates edge 4 631 3889 root112Tail9 winner.bound = root112Candidates8 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root112Step8 : scan edge (search edge 3) 4 [631] 631 root112Tail8 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail9 winner := by
  rw [root112Tail8, scan, if_neg (by decide), root112Candidates8_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail9
    (if 3 ≤ root112Candidates8.length then search edge 3 [631, 3889] root112Candidates8 4520 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root112Candidates9_checked : candidates edge 4 631 4051 root112Tail10 winner.bound = root112Candidates9 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root112Step9 : scan edge (search edge 3) 4 [631] 631 root112Tail9 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail10 winner := by
  rw [root112Tail9, scan, if_neg (by decide), root112Candidates9_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail10
    (if 3 ≤ root112Candidates9.length then search edge 3 [631, 4051] root112Candidates9 4682 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates10 : List Nat := [6199, 13597]
set_option maxRecDepth 8192 in
theorem root112Candidates10_checked : candidates edge 4 631 4057 root112Tail11 winner.bound = root112Candidates10 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root112Step10 : scan edge (search edge 3) 4 [631] 631 root112Tail10 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail11 winner := by
  rw [root112Tail10, scan, if_neg (by decide), root112Candidates10_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail11
    (if 3 ≤ root112Candidates10.length then search edge 3 [631, 4057] root112Candidates10 4688 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root112Candidates11_checked : candidates edge 4 631 4801 root112Tail12 winner.bound = root112Candidates11 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root112Step11 : scan edge (search edge 3) 4 [631] 631 root112Tail11 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail12 winner := by
  rw [root112Tail11, scan, if_neg (by decide), root112Candidates11_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail12
    (if 3 ≤ root112Candidates11.length then search edge 3 [631, 4801] root112Candidates11 5432 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates12 : List Nat := [13669]
set_option maxRecDepth 8192 in
theorem root112Candidates12_checked : candidates edge 4 631 4933 root112Tail13 winner.bound = root112Candidates12 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root112Step12 : scan edge (search edge 3) 4 [631] 631 root112Tail12 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail13 winner := by
  rw [root112Tail12, scan, if_neg (by decide), root112Candidates12_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail13
    (if 3 ≤ root112Candidates12.length then search edge 3 [631, 4933] root112Candidates12 5564 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates13 : List Nat := [9127, 13669, 14797, 16063]
set_option maxRecDepth 8192 in
theorem root112Candidates13_checked : candidates edge 4 631 5479 root112Tail14 winner.bound = root112Candidates13 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child13 : search edge 3 [631, 5479] root112Candidates13 6110 winner = winner := by decide
theorem root112Step13 : scan edge (search edge 3) 4 [631] 631 root112Tail13 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail14 winner := by
  rw [root112Tail13, scan, if_neg (by decide), root112Candidates13_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail14
    (if 3 ≤ root112Candidates13.length then search edge 3 [631, 5479] root112Candidates13 6110 winner else winner) = _
  rw [if_pos (by decide), root112Child13]
def root112Candidates14 : List Nat := [10867, 11329, 13669, 16987]
set_option maxRecDepth 8192 in
theorem root112Candidates14_checked : candidates edge 4 631 5527 root112Tail15 winner.bound = root112Candidates14 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child14 : search edge 3 [631, 5527] root112Candidates14 6158 winner = winner := by decide
theorem root112Step14 : scan edge (search edge 3) 4 [631] 631 root112Tail14 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail15 winner := by
  rw [root112Tail14, scan, if_neg (by decide), root112Candidates14_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail15
    (if 3 ≤ root112Candidates14.length then search edge 3 [631, 5527] root112Candidates14 6158 winner else winner) = _
  rw [if_pos (by decide), root112Child14]
def root112Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root112Candidates15_checked : candidates edge 4 631 5557 root112Tail16 winner.bound = root112Candidates15 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root112Step15 : scan edge (search edge 3) 4 [631] 631 root112Tail15 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail16 winner := by
  rw [root112Tail15, scan, if_neg (by decide), root112Candidates15_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail16
    (if 3 ≤ root112Candidates15.length then search edge 3 [631, 5557] root112Candidates15 6188 winner else winner) = _
  rw [if_neg (by decide)]
def root112Candidates16 : List Nat := [9613, 9769, 13597]
set_option maxRecDepth 8192 in
theorem root112Candidates16_checked : candidates edge 4 631 5737 root112Tail17 winner.bound = root112Candidates16 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root112Child16 : search edge 3 [631, 5737] root112Candidates16 6368 winner = winner := by decide
theorem root112Step16 : scan edge (search edge 3) 4 [631] 631 root112Tail16 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail17 winner := by
  rw [root112Tail16, scan, if_neg (by decide), root112Candidates16_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail17
    (if 3 ≤ root112Candidates16.length then search edge 3 [631, 5737] root112Candidates16 6368 winner else winner) = _
  rw [if_pos (by decide), root112Child16]
def root112Candidates17 : List Nat := [16987]
set_option maxRecDepth 8192 in
theorem root112Candidates17_checked : candidates edge 4 631 6199 root112Tail18 winner.bound = root112Candidates17 := by
  unfold candidates
  have row : edge 6199 = (SortedNatTree.contains FastRows.row6199) := by
    change Data.row6199.contains = (SortedNatTree.contains FastRows.row6199)
    exact FastRows.row6199_checked.symm
  rw [row]
  decide
theorem root112Step17 : scan edge (search edge 3) 4 [631] 631 root112Tail17 winner =
    scan edge (search edge 3) 4 [631] 631 root112Tail18 winner := by
  rw [root112Tail17, scan, if_neg (by decide), root112Candidates17_checked]
  change scan edge (search edge 3) 4 [631] 631 root112Tail18
    (if 3 ≤ root112Candidates17.length then search edge 3 [631, 6199] root112Candidates17 6830 winner else winner) = _
  rw [if_neg (by decide)]
theorem root112Scan18 : scan edge (search edge 3) 4 [631] 631 root112Tail18 winner = winner := by rw [root112Tail18, scan, if_pos (by decide)]
theorem root112Scan17 : scan edge (search edge 3) 4 [631] 631 root112Tail17 winner = winner :=
  (root112Step17).trans root112Scan18
theorem root112Scan16 : scan edge (search edge 3) 4 [631] 631 root112Tail16 winner = winner :=
  (root112Step16).trans root112Scan17
theorem root112Scan15 : scan edge (search edge 3) 4 [631] 631 root112Tail15 winner = winner :=
  (root112Step15).trans root112Scan16
theorem root112Scan14 : scan edge (search edge 3) 4 [631] 631 root112Tail14 winner = winner :=
  (root112Step14).trans root112Scan15
theorem root112Scan13 : scan edge (search edge 3) 4 [631] 631 root112Tail13 winner = winner :=
  (root112Step13).trans root112Scan14
theorem root112Scan12 : scan edge (search edge 3) 4 [631] 631 root112Tail12 winner = winner :=
  (root112Step12).trans root112Scan13
theorem root112Scan11 : scan edge (search edge 3) 4 [631] 631 root112Tail11 winner = winner :=
  (root112Step11).trans root112Scan12
theorem root112Scan10 : scan edge (search edge 3) 4 [631] 631 root112Tail10 winner = winner :=
  (root112Step10).trans root112Scan11
theorem root112Scan9 : scan edge (search edge 3) 4 [631] 631 root112Tail9 winner = winner :=
  (root112Step9).trans root112Scan10
theorem root112Scan8 : scan edge (search edge 3) 4 [631] 631 root112Tail8 winner = winner :=
  (root112Step8).trans root112Scan9
theorem root112Scan7 : scan edge (search edge 3) 4 [631] 631 root112Tail7 winner = winner :=
  (root112Step7).trans root112Scan8
theorem root112Scan6 : scan edge (search edge 3) 4 [631] 631 root112Tail6 winner = winner :=
  (root112Step6).trans root112Scan7
theorem root112Scan5 : scan edge (search edge 3) 4 [631] 631 root112Tail5 winner = winner :=
  (root112Step5).trans root112Scan6
theorem root112Scan4 : scan edge (search edge 3) 4 [631] 631 root112Tail4 winner = winner :=
  (root112Step4).trans root112Scan5
theorem root112Scan3 : scan edge (search edge 3) 4 [631] 631 root112Tail3 winner = winner :=
  (root112Step3).trans root112Scan4
theorem root112Scan2 : scan edge (search edge 3) 4 [631] 631 root112Tail2 winner = winner :=
  (root112Step2).trans root112Scan3
theorem root112Scan1 : scan edge (search edge 3) 4 [631] 631 root112Tail1 winner = winner :=
  (root112Step1).trans root112Scan2
theorem root112Scan0 : scan edge (search edge 3) 4 [631] 631 root112Tail0 winner = winner :=
  (root112Step0).trans root112Scan1
set_option maxRecDepth 4096 in
theorem root112Branch : search edge 4 [631] root112Candidates 631 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [631] 631 root112Tail0 winner = winner
  exact root112Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
