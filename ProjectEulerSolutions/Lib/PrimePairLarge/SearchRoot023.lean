import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root23Candidates : List Nat := Data.row101
set_option maxRecDepth 8192 in
theorem root23Candidates_checked : candidates edge 5 0 101 tail24 winner.bound = root23Candidates := by
  apply candidates_eq_row edge 5 0 101 tail24 root23Candidates winner.bound tail24_ordered
  · decide
  · decide
  · funext q
    rfl
def root23Tail80 : List Nat := []
def root23Tail79 : List Nat := 25793 :: root23Tail80
def root23Tail78 : List Nat := 25673 :: root23Tail79
def root23Tail77 : List Nat := 25541 :: root23Tail78
def root23Tail76 : List Nat := 24977 :: root23Tail77
def root23Tail75 : List Nat := 24953 :: root23Tail76
def root23Tail74 : List Nat := 24767 :: root23Tail75
def root23Tail73 : List Nat := 24677 :: root23Tail74
def root23Tail72 : List Nat := 23339 :: root23Tail73
def root23Tail71 : List Nat := 23021 :: root23Tail72
def root23Tail70 : List Nat := 22817 :: root23Tail71
def root23Tail69 : List Nat := 22481 :: root23Tail70
def root23Tail68 : List Nat := 21491 :: root23Tail69
def root23Tail67 : List Nat := 21107 :: root23Tail68
def root23Tail66 : List Nat := 20963 :: root23Tail67
def root23Tail65 : List Nat := 20753 :: root23Tail66
def root23Tail64 : List Nat := 19553 :: root23Tail65
def root23Tail63 : List Nat := 19163 :: root23Tail64
def root23Tail62 : List Nat := 19013 :: root23Tail63
def root23Tail61 : List Nat := 18839 :: root23Tail62
def root23Tail60 : List Nat := 18233 :: root23Tail61
def root23Tail59 : List Nat := 18191 :: root23Tail60
def root23Tail58 : List Nat := 17579 :: root23Tail59
def root23Tail57 : List Nat := 17231 :: root23Tail58
def root23Tail56 : List Nat := 17207 :: root23Tail57
def root23Tail55 : List Nat := 16889 :: root23Tail56
def root23Tail54 : List Nat := 16487 :: root23Tail55
def root23Tail53 : List Nat := 16361 :: root23Tail54
def root23Tail52 : List Nat := 16097 :: root23Tail53
def root23Tail51 : List Nat := 15803 :: root23Tail52
def root23Tail50 : List Nat := 15737 :: root23Tail51
def root23Tail49 : List Nat := 14957 :: root23Tail50
def root23Tail48 : List Nat := 14771 :: root23Tail49
def root23Tail47 : List Nat := 14627 :: root23Tail48
def root23Tail46 : List Nat := 14543 :: root23Tail47
def root23Tail45 : List Nat := 14207 :: root23Tail46
def root23Tail44 : List Nat := 14081 :: root23Tail45
def root23Tail43 : List Nat := 14051 :: root23Tail44
def root23Tail42 : List Nat := 13553 :: root23Tail43
def root23Tail41 : List Nat := 13241 :: root23Tail42
def root23Tail40 : List Nat := 13109 :: root23Tail41
def root23Tail39 : List Nat := 13001 :: root23Tail40
def root23Tail38 : List Nat := 12413 :: root23Tail39
def root23Tail37 : List Nat := 11939 :: root23Tail38
def root23Tail36 : List Nat := 11117 :: root23Tail37
def root23Tail35 : List Nat := 10847 :: root23Tail36
def root23Tail34 : List Nat := 9413 :: root23Tail35
def root23Tail33 : List Nat := 9377 :: root23Tail34
def root23Tail32 : List Nat := 9059 :: root23Tail33
def root23Tail31 : List Nat := 8999 :: root23Tail32
def root23Tail30 : List Nat := 7193 :: root23Tail31
def root23Tail29 : List Nat := 6971 :: root23Tail30
def root23Tail28 : List Nat := 6689 :: root23Tail29
def root23Tail27 : List Nat := 6221 :: root23Tail28
def root23Tail26 : List Nat := 6053 :: root23Tail27
def root23Tail25 : List Nat := 5471 :: root23Tail26
def root23Tail24 : List Nat := 5081 :: root23Tail25
def root23Tail23 : List Nat := 5039 :: root23Tail24
def root23Tail22 : List Nat := 4817 :: root23Tail23
def root23Tail21 : List Nat := 4337 :: root23Tail22
def root23Tail20 : List Nat := 4229 :: root23Tail21
def root23Tail19 : List Nat := 3851 :: root23Tail20
def root23Tail18 : List Nat := 3833 :: root23Tail19
def root23Tail17 : List Nat := 3527 :: root23Tail18
def root23Tail16 : List Nat := 3329 :: root23Tail17
def root23Tail15 : List Nat := 2903 :: root23Tail16
def root23Tail14 : List Nat := 2699 :: root23Tail15
def root23Tail13 : List Nat := 2657 :: root23Tail14
def root23Tail12 : List Nat := 2633 :: root23Tail13
def root23Tail11 : List Nat := 2549 :: root23Tail12
def root23Tail10 : List Nat := 1979 :: root23Tail11
def root23Tail9 : List Nat := 1889 :: root23Tail10
def root23Tail8 : List Nat := 1733 :: root23Tail9
def root23Tail7 : List Nat := 719 :: root23Tail8
def root23Tail6 : List Nat := 641 :: root23Tail7
def root23Tail5 : List Nat := 467 :: root23Tail6
def root23Tail4 : List Nat := 383 :: root23Tail5
def root23Tail3 : List Nat := 359 :: root23Tail4
def root23Tail2 : List Nat := 197 :: root23Tail3
def root23Tail1 : List Nat := 149 :: root23Tail2
def root23Tail0 : List Nat := 107 :: root23Tail1
def root23Candidates0 : List Nat := [2633, 15737]
set_option maxRecDepth 8192 in
theorem root23Candidates0_checked : candidates edge 4 101 107 root23Tail1 winner.bound = root23Candidates0 := by
  unfold candidates
  have row : edge 107 = (SortedNatTree.contains FastRows.row107) := by
    change Data.row107.contains = (SortedNatTree.contains FastRows.row107)
    exact FastRows.row107_checked.symm
  rw [row]
  decide
theorem root23Step0 : scan edge (search edge 3) 4 [101] 101 root23Tail0 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail1 winner := by
  rw [root23Tail0, scan, if_neg (by decide), root23Candidates0_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail1
    (if 3 ≤ root23Candidates0.length then search edge 3 [101, 107] root23Candidates0 208 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates1 : List Nat := [1979, 7193]
set_option maxRecDepth 8192 in
theorem root23Candidates1_checked : candidates edge 4 101 149 root23Tail2 winner.bound = root23Candidates1 := by
  unfold candidates
  have row : edge 149 = (SortedNatTree.contains FastRows.row149) := by
    change Data.row149.contains = (SortedNatTree.contains FastRows.row149)
    exact FastRows.row149_checked.symm
  rw [row]
  decide
theorem root23Step1 : scan edge (search edge 3) 4 [101] 101 root23Tail1 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail2 winner := by
  rw [root23Tail1, scan, if_neg (by decide), root23Candidates1_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail2
    (if 3 ≤ root23Candidates1.length then search edge 3 [101, 149] root23Candidates1 250 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates2 : List Nat := [641, 2657, 16097]
set_option maxRecDepth 8192 in
theorem root23Candidates2_checked : candidates edge 4 101 197 root23Tail3 winner.bound = root23Candidates2 := by
  unfold candidates
  have row : edge 197 = (SortedNatTree.contains FastRows.row197) := by
    change Data.row197.contains = (SortedNatTree.contains FastRows.row197)
    exact FastRows.row197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child2 : search edge 3 [101, 197] root23Candidates2 298 winner = winner := by decide
theorem root23Step2 : scan edge (search edge 3) 4 [101] 101 root23Tail2 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail3 winner := by
  rw [root23Tail2, scan, if_neg (by decide), root23Candidates2_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail3
    (if 3 ≤ root23Candidates2.length then search edge 3 [101, 197] root23Candidates2 298 winner else winner) = _
  rw [if_pos (by decide), root23Child2]
def root23Candidates3 : List Nat := [3833, 6689, 16889]
set_option maxRecDepth 8192 in
theorem root23Candidates3_checked : candidates edge 4 101 359 root23Tail4 winner.bound = root23Candidates3 := by
  unfold candidates
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child3 : search edge 3 [101, 359] root23Candidates3 460 winner = winner := by decide
theorem root23Step3 : scan edge (search edge 3) 4 [101] 101 root23Tail3 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail4 winner := by
  rw [root23Tail3, scan, if_neg (by decide), root23Candidates3_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail4
    (if 3 ≤ root23Candidates3.length then search edge 3 [101, 359] root23Candidates3 460 winner else winner) = _
  rw [if_pos (by decide), root23Child3]
def root23Candidates4 : List Nat := [1733, 1889, 3527, 24977]
set_option maxRecDepth 8192 in
theorem root23Candidates4_checked : candidates edge 4 101 383 root23Tail5 winner.bound = root23Candidates4 := by
  unfold candidates
  have row : edge 383 = (SortedNatTree.contains FastRows.row383) := by
    change Data.row383.contains = (SortedNatTree.contains FastRows.row383)
    exact FastRows.row383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child4 : search edge 3 [101, 383] root23Candidates4 484 winner = winner := by decide
theorem root23Step4 : scan edge (search edge 3) 4 [101] 101 root23Tail4 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail5 winner := by
  rw [root23Tail4, scan, if_neg (by decide), root23Candidates4_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail5
    (if 3 ≤ root23Candidates4.length then search edge 3 [101, 383] root23Candidates4 484 winner else winner) = _
  rw [if_pos (by decide), root23Child4]
def root23Candidates5 : List Nat := [641, 11117, 13553, 14543, 14627]
set_option maxRecDepth 8192 in
theorem root23Candidates5_checked : candidates edge 4 101 467 root23Tail6 winner.bound = root23Candidates5 := by
  unfold candidates
  have row : edge 467 = (SortedNatTree.contains FastRows.row467) := by
    change Data.row467.contains = (SortedNatTree.contains FastRows.row467)
    exact FastRows.row467_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child5 : search edge 3 [101, 467] root23Candidates5 568 winner = winner := by decide
theorem root23Step5 : scan edge (search edge 3) 4 [101] 101 root23Tail5 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail6 winner := by
  rw [root23Tail5, scan, if_neg (by decide), root23Candidates5_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail6
    (if 3 ≤ root23Candidates5.length then search edge 3 [101, 467] root23Candidates5 568 winner else winner) = _
  rw [if_pos (by decide), root23Child5]
def root23Candidates6 : List Nat := [14051, 16487]
set_option maxRecDepth 8192 in
theorem root23Candidates6_checked : candidates edge 4 101 641 root23Tail7 winner.bound = root23Candidates6 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
theorem root23Step6 : scan edge (search edge 3) 4 [101] 101 root23Tail6 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail7 winner := by
  rw [root23Tail6, scan, if_neg (by decide), root23Candidates6_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail7
    (if 3 ≤ root23Candidates6.length then search edge 3 [101, 641] root23Candidates6 742 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates7 : List Nat := [3833, 5039, 12413, 14207, 19553]
set_option maxRecDepth 8192 in
theorem root23Candidates7_checked : candidates edge 4 101 719 root23Tail8 winner.bound = root23Candidates7 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child7 : search edge 3 [101, 719] root23Candidates7 820 winner = winner := by decide
theorem root23Step7 : scan edge (search edge 3) 4 [101] 101 root23Tail7 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail8 winner := by
  rw [root23Tail7, scan, if_neg (by decide), root23Candidates7_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail8
    (if 3 ≤ root23Candidates7.length then search edge 3 [101, 719] root23Candidates7 820 winner else winner) = _
  rw [if_pos (by decide), root23Child7]
def root23Candidates8 : List Nat := [2657, 4817]
set_option maxRecDepth 8192 in
theorem root23Candidates8_checked : candidates edge 4 101 1733 root23Tail9 winner.bound = root23Candidates8 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
theorem root23Step8 : scan edge (search edge 3) 4 [101] 101 root23Tail8 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail9 winner := by
  rw [root23Tail8, scan, if_neg (by decide), root23Candidates8_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail9
    (if 3 ≤ root23Candidates8.length then search edge 3 [101, 1733] root23Candidates8 1834 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates9 : List Nat := [2633]
set_option maxRecDepth 8192 in
theorem root23Candidates9_checked : candidates edge 4 101 1889 root23Tail10 winner.bound = root23Candidates9 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root23Step9 : scan edge (search edge 3) 4 [101] 101 root23Tail9 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail10 winner := by
  rw [root23Tail9, scan, if_neg (by decide), root23Candidates9_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail10
    (if 3 ≤ root23Candidates9.length then search edge 3 [101, 1889] root23Candidates9 1990 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates10 : List Nat := [9413, 11117, 14957, 23021]
set_option maxRecDepth 8192 in
theorem root23Candidates10_checked : candidates edge 4 101 1979 root23Tail11 winner.bound = root23Candidates10 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child10 : search edge 3 [101, 1979] root23Candidates10 2080 winner = winner := by decide
theorem root23Step10 : scan edge (search edge 3) 4 [101] 101 root23Tail10 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail11 winner := by
  rw [root23Tail10, scan, if_neg (by decide), root23Candidates10_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail11
    (if 3 ≤ root23Candidates10.length then search edge 3 [101, 1979] root23Candidates10 2080 winner else winner) = _
  rw [if_pos (by decide), root23Child10]
def root23Candidates11 : List Nat := [3329, 5471, 13553, 14081, 19013]
set_option maxRecDepth 8192 in
theorem root23Candidates11_checked : candidates edge 4 101 2549 root23Tail12 winner.bound = root23Candidates11 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child11 : search edge 3 [101, 2549] root23Candidates11 2650 winner = winner := by decide
theorem root23Step11 : scan edge (search edge 3) 4 [101] 101 root23Tail11 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail12 winner := by
  rw [root23Tail11, scan, if_neg (by decide), root23Candidates11_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail12
    (if 3 ≤ root23Candidates11.length then search edge 3 [101, 2549] root23Candidates11 2650 winner else winner) = _
  rw [if_pos (by decide), root23Child11]
def root23Candidates12 : List Nat := [2699, 4337, 7193, 8999, 10847, 14771]
set_option maxRecDepth 8192 in
theorem root23Candidates12_checked : candidates edge 4 101 2633 root23Tail13 winner.bound = root23Candidates12 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root23Child12 : search edge 3 [101, 2633] root23Candidates12 2734 winner = winner := by decide
theorem root23Step12 : scan edge (search edge 3) 4 [101] 101 root23Tail12 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail13 winner := by
  rw [root23Tail12, scan, if_neg (by decide), root23Candidates12_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail13
    (if 3 ≤ root23Candidates12.length then search edge 3 [101, 2633] root23Candidates12 2734 winner else winner) = _
  rw [if_pos (by decide), root23Child12]
def root23Candidates13 : List Nat := [18191]
set_option maxRecDepth 8192 in
theorem root23Candidates13_checked : candidates edge 4 101 2657 root23Tail14 winner.bound = root23Candidates13 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root23Step13 : scan edge (search edge 3) 4 [101] 101 root23Tail13 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail14 winner := by
  rw [root23Tail13, scan, if_neg (by decide), root23Candidates13_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail14
    (if 3 ≤ root23Candidates13.length then search edge 3 [101, 2657] root23Candidates13 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates14 : List Nat := [4337]
set_option maxRecDepth 8192 in
theorem root23Candidates14_checked : candidates edge 4 101 2699 root23Tail15 winner.bound = root23Candidates14 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
theorem root23Step14 : scan edge (search edge 3) 4 [101] 101 root23Tail14 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail15 winner := by
  rw [root23Tail14, scan, if_neg (by decide), root23Candidates14_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail15
    (if 3 ≤ root23Candidates14.length then search edge 3 [101, 2699] root23Candidates14 2800 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates15 : List Nat := [14051]
set_option maxRecDepth 8192 in
theorem root23Candidates15_checked : candidates edge 4 101 2903 root23Tail16 winner.bound = root23Candidates15 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root23Step15 : scan edge (search edge 3) 4 [101] 101 root23Tail15 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail16 winner := by
  rw [root23Tail15, scan, if_neg (by decide), root23Candidates15_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail16
    (if 3 ≤ root23Candidates15.length then search edge 3 [101, 2903] root23Candidates15 3004 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates16 : List Nat := [16487]
set_option maxRecDepth 8192 in
theorem root23Candidates16_checked : candidates edge 4 101 3329 root23Tail17 winner.bound = root23Candidates16 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root23Step16 : scan edge (search edge 3) 4 [101] 101 root23Tail16 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail17 winner := by
  rw [root23Tail16, scan, if_neg (by decide), root23Candidates16_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail17
    (if 3 ≤ root23Candidates16.length then search edge 3 [101, 3329] root23Candidates16 3430 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates17 : List Nat := [19163]
set_option maxRecDepth 8192 in
theorem root23Candidates17_checked : candidates edge 4 101 3527 root23Tail18 winner.bound = root23Candidates17 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root23Step17 : scan edge (search edge 3) 4 [101] 101 root23Tail17 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail18 winner := by
  rw [root23Tail17, scan, if_neg (by decide), root23Candidates17_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail18
    (if 3 ≤ root23Candidates17.length then search edge 3 [101, 3527] root23Candidates17 3628 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates18 : List Nat := [13001]
set_option maxRecDepth 8192 in
theorem root23Candidates18_checked : candidates edge 4 101 3833 root23Tail19 winner.bound = root23Candidates18 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root23Step18 : scan edge (search edge 3) 4 [101] 101 root23Tail18 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail19 winner := by
  rw [root23Tail18, scan, if_neg (by decide), root23Candidates18_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail19
    (if 3 ≤ root23Candidates18.length then search edge 3 [101, 3833] root23Candidates18 3934 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root23Candidates19_checked : candidates edge 4 101 3851 root23Tail20 winner.bound = root23Candidates19 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root23Step19 : scan edge (search edge 3) 4 [101] 101 root23Tail19 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail20 winner := by
  rw [root23Tail19, scan, if_neg (by decide), root23Candidates19_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail20
    (if 3 ≤ root23Candidates19.length then search edge 3 [101, 3851] root23Candidates19 3952 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates20 : List Nat := [5039, 6689]
set_option maxRecDepth 8192 in
theorem root23Candidates20_checked : candidates edge 4 101 4229 root23Tail21 winner.bound = root23Candidates20 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root23Step20 : scan edge (search edge 3) 4 [101] 101 root23Tail20 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail21 winner := by
  rw [root23Tail20, scan, if_neg (by decide), root23Candidates20_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail21
    (if 3 ≤ root23Candidates20.length then search edge 3 [101, 4229] root23Candidates20 4330 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates21 : List Nat := [5471, 21107]
set_option maxRecDepth 8192 in
theorem root23Candidates21_checked : candidates edge 4 101 4337 root23Tail22 winner.bound = root23Candidates21 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root23Step21 : scan edge (search edge 3) 4 [101] 101 root23Tail21 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail22 winner := by
  rw [root23Tail21, scan, if_neg (by decide), root23Candidates21_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail22
    (if 3 ≤ root23Candidates21.length then search edge 3 [101, 4337] root23Candidates21 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates22 : List Nat := [17207]
set_option maxRecDepth 8192 in
theorem root23Candidates22_checked : candidates edge 4 101 4817 root23Tail23 winner.bound = root23Candidates22 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root23Step22 : scan edge (search edge 3) 4 [101] 101 root23Tail22 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail23 winner := by
  rw [root23Tail22, scan, if_neg (by decide), root23Candidates22_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail23
    (if 3 ≤ root23Candidates22.length then search edge 3 [101, 4817] root23Candidates22 4918 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root23Candidates23_checked : candidates edge 4 101 5039 root23Tail24 winner.bound = root23Candidates23 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root23Step23 : scan edge (search edge 3) 4 [101] 101 root23Tail23 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail24 winner := by
  rw [root23Tail23, scan, if_neg (by decide), root23Candidates23_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail24
    (if 3 ≤ root23Candidates23.length then search edge 3 [101, 5039] root23Candidates23 5140 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates24 : List Nat := [6971]
set_option maxRecDepth 8192 in
theorem root23Candidates24_checked : candidates edge 4 101 5081 root23Tail25 winner.bound = root23Candidates24 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root23Step24 : scan edge (search edge 3) 4 [101] 101 root23Tail24 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail25 winner := by
  rw [root23Tail24, scan, if_neg (by decide), root23Candidates24_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail25
    (if 3 ≤ root23Candidates24.length then search edge 3 [101, 5081] root23Candidates24 5182 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates25 : List Nat := [13001, 18233]
set_option maxRecDepth 8192 in
theorem root23Candidates25_checked : candidates edge 4 101 5471 root23Tail26 winner.bound = root23Candidates25 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root23Step25 : scan edge (search edge 3) 4 [101] 101 root23Tail25 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail26 winner := by
  rw [root23Tail25, scan, if_neg (by decide), root23Candidates25_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail26
    (if 3 ≤ root23Candidates25.length then search edge 3 [101, 5471] root23Candidates25 5572 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root23Candidates26_checked : candidates edge 4 101 6053 root23Tail27 winner.bound = root23Candidates26 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root23Step26 : scan edge (search edge 3) 4 [101] 101 root23Tail26 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail27 winner := by
  rw [root23Tail26, scan, if_neg (by decide), root23Candidates26_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail27
    (if 3 ≤ root23Candidates26.length then search edge 3 [101, 6053] root23Candidates26 6154 winner else winner) = _
  rw [if_neg (by decide)]
def root23Candidates27 : List Nat := [6971]
set_option maxRecDepth 8192 in
theorem root23Candidates27_checked : candidates edge 4 101 6221 root23Tail28 winner.bound = root23Candidates27 := by
  unfold candidates
  have row : edge 6221 = (SortedNatTree.contains FastRows.row6221) := by
    change Data.row6221.contains = (SortedNatTree.contains FastRows.row6221)
    exact FastRows.row6221_checked.symm
  rw [row]
  decide
theorem root23Step27 : scan edge (search edge 3) 4 [101] 101 root23Tail27 winner =
    scan edge (search edge 3) 4 [101] 101 root23Tail28 winner := by
  rw [root23Tail27, scan, if_neg (by decide), root23Candidates27_checked]
  change scan edge (search edge 3) 4 [101] 101 root23Tail28
    (if 3 ≤ root23Candidates27.length then search edge 3 [101, 6221] root23Candidates27 6322 winner else winner) = _
  rw [if_neg (by decide)]
theorem root23Scan28 : scan edge (search edge 3) 4 [101] 101 root23Tail28 winner = winner := by rw [root23Tail28, scan, if_pos (by decide)]
theorem root23Scan27 : scan edge (search edge 3) 4 [101] 101 root23Tail27 winner = winner :=
  (root23Step27).trans root23Scan28
theorem root23Scan26 : scan edge (search edge 3) 4 [101] 101 root23Tail26 winner = winner :=
  (root23Step26).trans root23Scan27
theorem root23Scan25 : scan edge (search edge 3) 4 [101] 101 root23Tail25 winner = winner :=
  (root23Step25).trans root23Scan26
theorem root23Scan24 : scan edge (search edge 3) 4 [101] 101 root23Tail24 winner = winner :=
  (root23Step24).trans root23Scan25
theorem root23Scan23 : scan edge (search edge 3) 4 [101] 101 root23Tail23 winner = winner :=
  (root23Step23).trans root23Scan24
theorem root23Scan22 : scan edge (search edge 3) 4 [101] 101 root23Tail22 winner = winner :=
  (root23Step22).trans root23Scan23
theorem root23Scan21 : scan edge (search edge 3) 4 [101] 101 root23Tail21 winner = winner :=
  (root23Step21).trans root23Scan22
theorem root23Scan20 : scan edge (search edge 3) 4 [101] 101 root23Tail20 winner = winner :=
  (root23Step20).trans root23Scan21
theorem root23Scan19 : scan edge (search edge 3) 4 [101] 101 root23Tail19 winner = winner :=
  (root23Step19).trans root23Scan20
theorem root23Scan18 : scan edge (search edge 3) 4 [101] 101 root23Tail18 winner = winner :=
  (root23Step18).trans root23Scan19
theorem root23Scan17 : scan edge (search edge 3) 4 [101] 101 root23Tail17 winner = winner :=
  (root23Step17).trans root23Scan18
theorem root23Scan16 : scan edge (search edge 3) 4 [101] 101 root23Tail16 winner = winner :=
  (root23Step16).trans root23Scan17
theorem root23Scan15 : scan edge (search edge 3) 4 [101] 101 root23Tail15 winner = winner :=
  (root23Step15).trans root23Scan16
theorem root23Scan14 : scan edge (search edge 3) 4 [101] 101 root23Tail14 winner = winner :=
  (root23Step14).trans root23Scan15
theorem root23Scan13 : scan edge (search edge 3) 4 [101] 101 root23Tail13 winner = winner :=
  (root23Step13).trans root23Scan14
theorem root23Scan12 : scan edge (search edge 3) 4 [101] 101 root23Tail12 winner = winner :=
  (root23Step12).trans root23Scan13
theorem root23Scan11 : scan edge (search edge 3) 4 [101] 101 root23Tail11 winner = winner :=
  (root23Step11).trans root23Scan12
theorem root23Scan10 : scan edge (search edge 3) 4 [101] 101 root23Tail10 winner = winner :=
  (root23Step10).trans root23Scan11
theorem root23Scan9 : scan edge (search edge 3) 4 [101] 101 root23Tail9 winner = winner :=
  (root23Step9).trans root23Scan10
theorem root23Scan8 : scan edge (search edge 3) 4 [101] 101 root23Tail8 winner = winner :=
  (root23Step8).trans root23Scan9
theorem root23Scan7 : scan edge (search edge 3) 4 [101] 101 root23Tail7 winner = winner :=
  (root23Step7).trans root23Scan8
theorem root23Scan6 : scan edge (search edge 3) 4 [101] 101 root23Tail6 winner = winner :=
  (root23Step6).trans root23Scan7
theorem root23Scan5 : scan edge (search edge 3) 4 [101] 101 root23Tail5 winner = winner :=
  (root23Step5).trans root23Scan6
theorem root23Scan4 : scan edge (search edge 3) 4 [101] 101 root23Tail4 winner = winner :=
  (root23Step4).trans root23Scan5
theorem root23Scan3 : scan edge (search edge 3) 4 [101] 101 root23Tail3 winner = winner :=
  (root23Step3).trans root23Scan4
theorem root23Scan2 : scan edge (search edge 3) 4 [101] 101 root23Tail2 winner = winner :=
  (root23Step2).trans root23Scan3
theorem root23Scan1 : scan edge (search edge 3) 4 [101] 101 root23Tail1 winner = winner :=
  (root23Step1).trans root23Scan2
theorem root23Scan0 : scan edge (search edge 3) 4 [101] 101 root23Tail0 winner = winner :=
  (root23Step0).trans root23Scan1
set_option maxRecDepth 4096 in
theorem root23Branch : search edge 4 [101] root23Candidates 101 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [101] 101 root23Tail0 winner = winner
  exact root23Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
