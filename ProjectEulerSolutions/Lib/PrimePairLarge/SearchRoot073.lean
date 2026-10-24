import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root73Candidates : List Nat := Data.row383
set_option maxRecDepth 8192 in
theorem root73Candidates_checked : candidates edge 5 0 383 tail74 winner.bound = root73Candidates := by
  apply candidates_eq_row edge 5 0 383 tail74 root73Candidates winner.bound tail74_ordered
  · decide
  · decide
  · funext q
    rfl
def root73Tail75 : List Nat := []
def root73Tail74 : List Nat := 25643 :: root73Tail75
def root73Tail73 : List Nat := 25589 :: root73Tail74
def root73Tail72 : List Nat := 25409 :: root73Tail73
def root73Tail71 : List Nat := 24977 :: root73Tail72
def root73Tail70 : List Nat := 23291 :: root73Tail71
def root73Tail69 : List Nat := 22877 :: root73Tail70
def root73Tail68 : List Nat := 22721 :: root73Tail69
def root73Tail67 : List Nat := 22637 :: root73Tail68
def root73Tail66 : List Nat := 21851 :: root73Tail67
def root73Tail65 : List Nat := 21821 :: root73Tail66
def root73Tail64 : List Nat := 21737 :: root73Tail65
def root73Tail63 : List Nat := 21221 :: root73Tail64
def root73Tail62 : List Nat := 20297 :: root73Tail63
def root73Tail61 : List Nat := 20147 :: root73Tail62
def root73Tail60 : List Nat := 18911 :: root73Tail61
def root73Tail59 : List Nat := 18803 :: root73Tail60
def root73Tail58 : List Nat := 18743 :: root73Tail59
def root73Tail57 : List Nat := 18143 :: root73Tail58
def root73Tail56 : List Nat := 17483 :: root73Tail57
def root73Tail55 : List Nat := 16979 :: root73Tail56
def root73Tail54 : List Nat := 16871 :: root73Tail55
def root73Tail53 : List Nat := 16001 :: root73Tail54
def root73Tail52 : List Nat := 15791 :: root73Tail53
def root73Tail51 : List Nat := 15731 :: root73Tail52
def root73Tail50 : List Nat := 15413 :: root73Tail51
def root73Tail49 : List Nat := 14669 :: root73Tail50
def root73Tail48 : List Nat := 14519 :: root73Tail49
def root73Tail47 : List Nat := 14153 :: root73Tail48
def root73Tail46 : List Nat := 13997 :: root73Tail47
def root73Tail45 : List Nat := 13829 :: root73Tail46
def root73Tail44 : List Nat := 13763 :: root73Tail45
def root73Tail43 : List Nat := 13397 :: root73Tail44
def root73Tail42 : List Nat := 13103 :: root73Tail43
def root73Tail41 : List Nat := 12347 :: root73Tail42
def root73Tail40 : List Nat := 11831 :: root73Tail41
def root73Tail39 : List Nat := 11789 :: root73Tail40
def root73Tail38 : List Nat := 11621 :: root73Tail39
def root73Tail37 : List Nat := 10949 :: root73Tail38
def root73Tail36 : List Nat := 10739 :: root73Tail37
def root73Tail35 : List Nat := 10691 :: root73Tail36
def root73Tail34 : List Nat := 10487 :: root73Tail35
def root73Tail33 : List Nat := 10037 :: root73Tail34
def root73Tail32 : List Nat := 9929 :: root73Tail33
def root73Tail31 : List Nat := 9923 :: root73Tail32
def root73Tail30 : List Nat := 9677 :: root73Tail31
def root73Tail29 : List Nat := 9239 :: root73Tail30
def root73Tail28 : List Nat := 8123 :: root73Tail29
def root73Tail27 : List Nat := 8117 :: root73Tail28
def root73Tail26 : List Nat := 7877 :: root73Tail27
def root73Tail25 : List Nat := 7853 :: root73Tail26
def root73Tail24 : List Nat := 7703 :: root73Tail25
def root73Tail23 : List Nat := 7331 :: root73Tail24
def root73Tail22 : List Nat := 6359 :: root73Tail23
def root73Tail21 : List Nat := 6317 :: root73Tail22
def root73Tail20 : List Nat := 5303 :: root73Tail21
def root73Tail19 : List Nat := 4547 :: root73Tail20
def root73Tail18 : List Nat := 4139 :: root73Tail19
def root73Tail17 : List Nat := 4091 :: root73Tail18
def root73Tail16 : List Nat := 3989 :: root73Tail17
def root73Tail15 : List Nat := 3527 :: root73Tail16
def root73Tail14 : List Nat := 3359 :: root73Tail15
def root73Tail13 : List Nat := 3257 :: root73Tail14
def root73Tail12 : List Nat := 2729 :: root73Tail13
def root73Tail11 : List Nat := 2687 :: root73Tail12
def root73Tail10 : List Nat := 2267 :: root73Tail11
def root73Tail9 : List Nat := 1889 :: root73Tail10
def root73Tail8 : List Nat := 1871 :: root73Tail9
def root73Tail7 : List Nat := 1733 :: root73Tail8
def root73Tail6 : List Nat := 1721 :: root73Tail7
def root73Tail5 : List Nat := 1493 :: root73Tail6
def root73Tail4 : List Nat := 1427 :: root73Tail5
def root73Tail3 : List Nat := 941 :: root73Tail4
def root73Tail2 : List Nat := 821 :: root73Tail3
def root73Tail1 : List Nat := 797 :: root73Tail2
def root73Tail0 : List Nat := 419 :: root73Tail1
def root73Candidates0 : List Nat := [11789, 14519]
set_option maxRecDepth 8192 in
theorem root73Candidates0_checked : candidates edge 4 383 419 root73Tail1 winner.bound = root73Candidates0 := by
  unfold candidates
  have row : edge 419 = (SortedNatTree.contains FastRows.row419) := by
    change Data.row419.contains = (SortedNatTree.contains FastRows.row419)
    exact FastRows.row419_checked.symm
  rw [row]
  decide
theorem root73Step0 : scan edge (search edge 3) 4 [383] 383 root73Tail0 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail1 winner := by
  rw [root73Tail0, scan, if_neg (by decide), root73Candidates0_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail1
    (if 3 ≤ root73Candidates0.length then search edge 3 [383, 419] root73Candidates0 802 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates1 : List Nat := [18911]
set_option maxRecDepth 8192 in
theorem root73Candidates1_checked : candidates edge 4 383 797 root73Tail2 winner.bound = root73Candidates1 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
theorem root73Step1 : scan edge (search edge 3) 4 [383] 383 root73Tail1 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail2 winner := by
  rw [root73Tail1, scan, if_neg (by decide), root73Candidates1_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail2
    (if 3 ≤ root73Candidates1.length then search edge 3 [383, 797] root73Candidates1 1180 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates2 : List Nat := [6317, 12347]
set_option maxRecDepth 8192 in
theorem root73Candidates2_checked : candidates edge 4 383 821 root73Tail3 winner.bound = root73Candidates2 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
theorem root73Step2 : scan edge (search edge 3) 4 [383] 383 root73Tail2 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail3 winner := by
  rw [root73Tail2, scan, if_neg (by decide), root73Candidates2_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail3
    (if 3 ≤ root73Candidates2.length then search edge 3 [383, 821] root73Candidates2 1204 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates3 : List Nat := [3527, 9239, 9923]
set_option maxRecDepth 8192 in
theorem root73Candidates3_checked : candidates edge 4 383 941 root73Tail4 winner.bound = root73Candidates3 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root73Child3 : search edge 3 [383, 941] root73Candidates3 1324 winner = winner := by decide
theorem root73Step3 : scan edge (search edge 3) 4 [383] 383 root73Tail3 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail4 winner := by
  rw [root73Tail3, scan, if_neg (by decide), root73Candidates3_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail4
    (if 3 ≤ root73Candidates3.length then search edge 3 [383, 941] root73Candidates3 1324 winner else winner) = _
  rw [if_pos (by decide), root73Child3]
def root73Candidates4 : List Nat := [3359, 3989, 13829, 14519]
set_option maxRecDepth 8192 in
theorem root73Candidates4_checked : candidates edge 4 383 1427 root73Tail5 winner.bound = root73Candidates4 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root73Child4 : search edge 3 [383, 1427] root73Candidates4 1810 winner = winner := by decide
theorem root73Step4 : scan edge (search edge 3) 4 [383] 383 root73Tail4 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail5 winner := by
  rw [root73Tail4, scan, if_neg (by decide), root73Candidates4_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail5
    (if 3 ≤ root73Candidates4.length then search edge 3 [383, 1427] root73Candidates4 1810 winner else winner) = _
  rw [if_pos (by decide), root73Child4]
def root73Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root73Candidates5_checked : candidates edge 4 383 1493 root73Tail6 winner.bound = root73Candidates5 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root73Step5 : scan edge (search edge 3) 4 [383] 383 root73Tail5 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail6 winner := by
  rw [root73Tail5, scan, if_neg (by decide), root73Candidates5_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail6
    (if 3 ≤ root73Candidates5.length then search edge 3 [383, 1493] root73Candidates5 1876 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates6 : List Nat := [3359, 9677, 16871]
set_option maxRecDepth 8192 in
theorem root73Candidates6_checked : candidates edge 4 383 1721 root73Tail7 winner.bound = root73Candidates6 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root73Child6 : search edge 3 [383, 1721] root73Candidates6 2104 winner = winner := by decide
theorem root73Step6 : scan edge (search edge 3) 4 [383] 383 root73Tail6 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail7 winner := by
  rw [root73Tail6, scan, if_neg (by decide), root73Candidates6_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail7
    (if 3 ≤ root73Candidates6.length then search edge 3 [383, 1721] root73Candidates6 2104 winner else winner) = _
  rw [if_pos (by decide), root73Child6]
def root73Candidates7 : List Nat := [9929, 18911]
set_option maxRecDepth 8192 in
theorem root73Candidates7_checked : candidates edge 4 383 1733 root73Tail8 winner.bound = root73Candidates7 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
theorem root73Step7 : scan edge (search edge 3) 4 [383] 383 root73Tail7 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail8 winner := by
  rw [root73Tail7, scan, if_neg (by decide), root73Candidates7_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail8
    (if 3 ≤ root73Candidates7.length then search edge 3 [383, 1733] root73Candidates7 2116 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates8 : List Nat := [4139, 11789]
set_option maxRecDepth 8192 in
theorem root73Candidates8_checked : candidates edge 4 383 1871 root73Tail9 winner.bound = root73Candidates8 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
theorem root73Step8 : scan edge (search edge 3) 4 [383] 383 root73Tail8 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail9 winner := by
  rw [root73Tail8, scan, if_neg (by decide), root73Candidates8_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail9
    (if 3 ≤ root73Candidates8.length then search edge 3 [383, 1871] root73Candidates8 2254 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates9 : List Nat := [8123, 9929, 15791]
set_option maxRecDepth 8192 in
theorem root73Candidates9_checked : candidates edge 4 383 1889 root73Tail10 winner.bound = root73Candidates9 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root73Child9 : search edge 3 [383, 1889] root73Candidates9 2272 winner = winner := by decide
theorem root73Step9 : scan edge (search edge 3) 4 [383] 383 root73Tail9 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail10 winner := by
  rw [root73Tail9, scan, if_neg (by decide), root73Candidates9_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail10
    (if 3 ≤ root73Candidates9.length then search edge 3 [383, 1889] root73Candidates9 2272 winner else winner) = _
  rw [if_pos (by decide), root73Child9]
def root73Candidates10 : List Nat := [2729, 3527]
set_option maxRecDepth 8192 in
theorem root73Candidates10_checked : candidates edge 4 383 2267 root73Tail11 winner.bound = root73Candidates10 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root73Step10 : scan edge (search edge 3) 4 [383] 383 root73Tail10 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail11 winner := by
  rw [root73Tail10, scan, if_neg (by decide), root73Candidates10_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail11
    (if 3 ≤ root73Candidates10.length then search edge 3 [383, 2267] root73Candidates10 2650 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates11 : List Nat := [7703]
set_option maxRecDepth 8192 in
theorem root73Candidates11_checked : candidates edge 4 383 2687 root73Tail12 winner.bound = root73Candidates11 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root73Step11 : scan edge (search edge 3) 4 [383] 383 root73Tail11 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail12 winner := by
  rw [root73Tail11, scan, if_neg (by decide), root73Candidates11_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail12
    (if 3 ≤ root73Candidates11.length then search edge 3 [383, 2687] root73Candidates11 3070 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates12 : List Nat := [11789, 13829]
set_option maxRecDepth 8192 in
theorem root73Candidates12_checked : candidates edge 4 383 2729 root73Tail13 winner.bound = root73Candidates12 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root73Step12 : scan edge (search edge 3) 4 [383] 383 root73Tail12 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail13 winner := by
  rw [root73Tail12, scan, if_neg (by decide), root73Candidates12_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail13
    (if 3 ≤ root73Candidates12.length then search edge 3 [383, 2729] root73Candidates12 3112 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates13 : List Nat := [4547, 12347]
set_option maxRecDepth 8192 in
theorem root73Candidates13_checked : candidates edge 4 383 3257 root73Tail14 winner.bound = root73Candidates13 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root73Step13 : scan edge (search edge 3) 4 [383] 383 root73Tail13 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail14 winner := by
  rw [root73Tail13, scan, if_neg (by decide), root73Candidates13_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail14
    (if 3 ≤ root73Candidates13.length then search edge 3 [383, 3257] root73Candidates13 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates14 : List Nat := [7853, 9677, 15791, 16871, 20297]
set_option maxRecDepth 8192 in
theorem root73Candidates14_checked : candidates edge 4 383 3359 root73Tail15 winner.bound = root73Candidates14 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root73Child14 : search edge 3 [383, 3359] root73Candidates14 3742 winner = winner := by decide
theorem root73Step14 : scan edge (search edge 3) 4 [383] 383 root73Tail14 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail15 winner := by
  rw [root73Tail14, scan, if_neg (by decide), root73Candidates14_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail15
    (if 3 ≤ root73Candidates14.length then search edge 3 [383, 3359] root73Candidates14 3742 winner else winner) = _
  rw [if_pos (by decide), root73Child14]
def root73Candidates15 : List Nat := [9239]
set_option maxRecDepth 8192 in
theorem root73Candidates15_checked : candidates edge 4 383 3527 root73Tail16 winner.bound = root73Candidates15 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root73Step15 : scan edge (search edge 3) 4 [383] 383 root73Tail15 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail16 winner := by
  rw [root73Tail15, scan, if_neg (by decide), root73Candidates15_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail16
    (if 3 ≤ root73Candidates15.length then search edge 3 [383, 3527] root73Candidates15 3910 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates16 : List Nat := [7703, 9239, 10037, 15791]
set_option maxRecDepth 8192 in
theorem root73Candidates16_checked : candidates edge 4 383 3989 root73Tail17 winner.bound = root73Candidates16 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root73Child16 : search edge 3 [383, 3989] root73Candidates16 4372 winner = winner := by decide
theorem root73Step16 : scan edge (search edge 3) 4 [383] 383 root73Tail16 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail17 winner := by
  rw [root73Tail16, scan, if_neg (by decide), root73Candidates16_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail17
    (if 3 ≤ root73Candidates16.length then search edge 3 [383, 3989] root73Candidates16 4372 winner else winner) = _
  rw [if_pos (by decide), root73Child16]
def root73Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root73Candidates17_checked : candidates edge 4 383 4091 root73Tail18 winner.bound = root73Candidates17 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root73Step17 : scan edge (search edge 3) 4 [383] 383 root73Tail17 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail18 winner := by
  rw [root73Tail17, scan, if_neg (by decide), root73Candidates17_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail18
    (if 3 ≤ root73Candidates17.length then search edge 3 [383, 4091] root73Candidates17 4474 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates18 : List Nat := [13103, 16001]
set_option maxRecDepth 8192 in
theorem root73Candidates18_checked : candidates edge 4 383 4139 root73Tail19 winner.bound = root73Candidates18 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root73Step18 : scan edge (search edge 3) 4 [383] 383 root73Tail18 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail19 winner := by
  rw [root73Tail18, scan, if_neg (by decide), root73Candidates18_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail19
    (if 3 ≤ root73Candidates18.length then search edge 3 [383, 4139] root73Candidates18 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root73Candidates19_checked : candidates edge 4 383 4547 root73Tail20 winner.bound = root73Candidates19 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root73Step19 : scan edge (search edge 3) 4 [383] 383 root73Tail19 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail20 winner := by
  rw [root73Tail19, scan, if_neg (by decide), root73Candidates19_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail20
    (if 3 ≤ root73Candidates19.length then search edge 3 [383, 4547] root73Candidates19 4930 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates20 : List Nat := [8123]
set_option maxRecDepth 8192 in
theorem root73Candidates20_checked : candidates edge 4 383 5303 root73Tail21 winner.bound = root73Candidates20 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root73Step20 : scan edge (search edge 3) 4 [383] 383 root73Tail20 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail21 winner := by
  rw [root73Tail20, scan, if_neg (by decide), root73Candidates20_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail21
    (if 3 ≤ root73Candidates20.length then search edge 3 [383, 5303] root73Candidates20 5686 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates21 : List Nat := [7853]
set_option maxRecDepth 8192 in
theorem root73Candidates21_checked : candidates edge 4 383 6317 root73Tail22 winner.bound = root73Candidates21 := by
  unfold candidates
  have row : edge 6317 = (SortedNatTree.contains FastRows.row6317) := by
    change Data.row6317.contains = (SortedNatTree.contains FastRows.row6317)
    exact FastRows.row6317_checked.symm
  rw [row]
  decide
theorem root73Step21 : scan edge (search edge 3) 4 [383] 383 root73Tail21 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail22 winner := by
  rw [root73Tail21, scan, if_neg (by decide), root73Candidates21_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail22
    (if 3 ≤ root73Candidates21.length then search edge 3 [383, 6317] root73Candidates21 6700 winner else winner) = _
  rw [if_neg (by decide)]
def root73Candidates22 : List Nat := [8123]
set_option maxRecDepth 8192 in
theorem root73Candidates22_checked : candidates edge 4 383 6359 root73Tail23 winner.bound = root73Candidates22 := by
  unfold candidates
  have row : edge 6359 = (SortedNatTree.contains FastRows.row6359) := by
    change Data.row6359.contains = (SortedNatTree.contains FastRows.row6359)
    exact FastRows.row6359_checked.symm
  rw [row]
  decide
theorem root73Step22 : scan edge (search edge 3) 4 [383] 383 root73Tail22 winner =
    scan edge (search edge 3) 4 [383] 383 root73Tail23 winner := by
  rw [root73Tail22, scan, if_neg (by decide), root73Candidates22_checked]
  change scan edge (search edge 3) 4 [383] 383 root73Tail23
    (if 3 ≤ root73Candidates22.length then search edge 3 [383, 6359] root73Candidates22 6742 winner else winner) = _
  rw [if_neg (by decide)]
theorem root73Scan23 : scan edge (search edge 3) 4 [383] 383 root73Tail23 winner = winner := by rw [root73Tail23, scan, if_pos (by decide)]
theorem root73Scan22 : scan edge (search edge 3) 4 [383] 383 root73Tail22 winner = winner :=
  (root73Step22).trans root73Scan23
theorem root73Scan21 : scan edge (search edge 3) 4 [383] 383 root73Tail21 winner = winner :=
  (root73Step21).trans root73Scan22
theorem root73Scan20 : scan edge (search edge 3) 4 [383] 383 root73Tail20 winner = winner :=
  (root73Step20).trans root73Scan21
theorem root73Scan19 : scan edge (search edge 3) 4 [383] 383 root73Tail19 winner = winner :=
  (root73Step19).trans root73Scan20
theorem root73Scan18 : scan edge (search edge 3) 4 [383] 383 root73Tail18 winner = winner :=
  (root73Step18).trans root73Scan19
theorem root73Scan17 : scan edge (search edge 3) 4 [383] 383 root73Tail17 winner = winner :=
  (root73Step17).trans root73Scan18
theorem root73Scan16 : scan edge (search edge 3) 4 [383] 383 root73Tail16 winner = winner :=
  (root73Step16).trans root73Scan17
theorem root73Scan15 : scan edge (search edge 3) 4 [383] 383 root73Tail15 winner = winner :=
  (root73Step15).trans root73Scan16
theorem root73Scan14 : scan edge (search edge 3) 4 [383] 383 root73Tail14 winner = winner :=
  (root73Step14).trans root73Scan15
theorem root73Scan13 : scan edge (search edge 3) 4 [383] 383 root73Tail13 winner = winner :=
  (root73Step13).trans root73Scan14
theorem root73Scan12 : scan edge (search edge 3) 4 [383] 383 root73Tail12 winner = winner :=
  (root73Step12).trans root73Scan13
theorem root73Scan11 : scan edge (search edge 3) 4 [383] 383 root73Tail11 winner = winner :=
  (root73Step11).trans root73Scan12
theorem root73Scan10 : scan edge (search edge 3) 4 [383] 383 root73Tail10 winner = winner :=
  (root73Step10).trans root73Scan11
theorem root73Scan9 : scan edge (search edge 3) 4 [383] 383 root73Tail9 winner = winner :=
  (root73Step9).trans root73Scan10
theorem root73Scan8 : scan edge (search edge 3) 4 [383] 383 root73Tail8 winner = winner :=
  (root73Step8).trans root73Scan9
theorem root73Scan7 : scan edge (search edge 3) 4 [383] 383 root73Tail7 winner = winner :=
  (root73Step7).trans root73Scan8
theorem root73Scan6 : scan edge (search edge 3) 4 [383] 383 root73Tail6 winner = winner :=
  (root73Step6).trans root73Scan7
theorem root73Scan5 : scan edge (search edge 3) 4 [383] 383 root73Tail5 winner = winner :=
  (root73Step5).trans root73Scan6
theorem root73Scan4 : scan edge (search edge 3) 4 [383] 383 root73Tail4 winner = winner :=
  (root73Step4).trans root73Scan5
theorem root73Scan3 : scan edge (search edge 3) 4 [383] 383 root73Tail3 winner = winner :=
  (root73Step3).trans root73Scan4
theorem root73Scan2 : scan edge (search edge 3) 4 [383] 383 root73Tail2 winner = winner :=
  (root73Step2).trans root73Scan3
theorem root73Scan1 : scan edge (search edge 3) 4 [383] 383 root73Tail1 winner = winner :=
  (root73Step1).trans root73Scan2
theorem root73Scan0 : scan edge (search edge 3) 4 [383] 383 root73Tail0 winner = winner :=
  (root73Step0).trans root73Scan1
set_option maxRecDepth 4096 in
theorem root73Branch : search edge 4 [383] root73Candidates 383 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [383] 383 root73Tail0 winner = winner
  exact root73Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
