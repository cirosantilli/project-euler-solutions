import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root30Candidates : List Nat := Data.row137
set_option maxRecDepth 8192 in
theorem root30Candidates_checked : candidates edge 5 0 137 tail31 winner.bound = root30Candidates := by
  apply candidates_eq_row edge 5 0 137 tail31 root30Candidates winner.bound tail31_ordered
  · decide
  · decide
  · funext q
    rfl
def root30Tail83 : List Nat := []
def root30Tail82 : List Nat := 25601 :: root30Tail83
def root30Tail81 : List Nat := 25247 :: root30Tail82
def root30Tail80 : List Nat := 24533 :: root30Tail81
def root30Tail79 : List Nat := 24071 :: root30Tail80
def root30Tail78 : List Nat := 23081 :: root30Tail79
def root30Tail77 : List Nat := 23057 :: root30Tail78
def root30Tail76 : List Nat := 22937 :: root30Tail77
def root30Tail75 : List Nat := 22721 :: root30Tail76
def root30Tail74 : List Nat := 22409 :: root30Tail75
def root30Tail73 : List Nat := 22277 :: root30Tail74
def root30Tail72 : List Nat := 22091 :: root30Tail73
def root30Tail71 : List Nat := 22067 :: root30Tail72
def root30Tail70 : List Nat := 21881 :: root30Tail71
def root30Tail69 : List Nat := 21569 :: root30Tail70
def root30Tail68 : List Nat := 21563 :: root30Tail69
def root30Tail67 : List Nat := 21269 :: root30Tail68
def root30Tail66 : List Nat := 21143 :: root30Tail67
def root30Tail65 : List Nat := 20543 :: root30Tail66
def root30Tail64 : List Nat := 20477 :: root30Tail65
def root30Tail63 : List Nat := 19427 :: root30Tail64
def root30Tail62 : List Nat := 18773 :: root30Tail63
def root30Tail61 : List Nat := 17909 :: root30Tail62
def root30Tail60 : List Nat := 17807 :: root30Tail61
def root30Tail59 : List Nat := 17789 :: root30Tail60
def root30Tail58 : List Nat := 17489 :: root30Tail59
def root30Tail57 : List Nat := 16931 :: root30Tail58
def root30Tail56 : List Nat := 16607 :: root30Tail57
def root30Tail55 : List Nat := 14621 :: root30Tail56
def root30Tail54 : List Nat := 13187 :: root30Tail55
def root30Tail53 : List Nat := 12791 :: root30Tail54
def root30Tail52 : List Nat := 12473 :: root30Tail53
def root30Tail51 : List Nat := 12269 :: root30Tail52
def root30Tail50 : List Nat := 12143 :: root30Tail51
def root30Tail49 : List Nat := 11927 :: root30Tail50
def root30Tail48 : List Nat := 11717 :: root30Tail49
def root30Tail47 : List Nat := 11279 :: root30Tail48
def root30Tail46 : List Nat := 10883 :: root30Tail47
def root30Tail45 : List Nat := 10391 :: root30Tail46
def root30Tail44 : List Nat := 10181 :: root30Tail45
def root30Tail43 : List Nat := 9857 :: root30Tail44
def root30Tail42 : List Nat := 9803 :: root30Tail43
def root30Tail41 : List Nat := 9491 :: root30Tail42
def root30Tail40 : List Nat := 9239 :: root30Tail41
def root30Tail39 : List Nat := 9173 :: root30Tail40
def root30Tail38 : List Nat := 8231 :: root30Tail39
def root30Tail37 : List Nat := 8219 :: root30Tail38
def root30Tail36 : List Nat := 8147 :: root30Tail37
def root30Tail35 : List Nat := 7757 :: root30Tail36
def root30Tail34 : List Nat := 7643 :: root30Tail35
def root30Tail33 : List Nat := 7577 :: root30Tail34
def root30Tail32 : List Nat := 7517 :: root30Tail33
def root30Tail31 : List Nat := 7499 :: root30Tail32
def root30Tail30 : List Nat := 7457 :: root30Tail31
def root30Tail29 : List Nat := 7127 :: root30Tail30
def root30Tail28 : List Nat := 7121 :: root30Tail29
def root30Tail27 : List Nat := 7043 :: root30Tail28
def root30Tail26 : List Nat := 6971 :: root30Tail27
def root30Tail25 : List Nat := 6491 :: root30Tail26
def root30Tail24 : List Nat := 6359 :: root30Tail25
def root30Tail23 : List Nat := 6197 :: root30Tail24
def root30Tail22 : List Nat := 6173 :: root30Tail23
def root30Tail21 : List Nat := 5981 :: root30Tail22
def root30Tail20 : List Nat := 5237 :: root30Tail21
def root30Tail19 : List Nat := 4721 :: root30Tail20
def root30Tail18 : List Nat := 4481 :: root30Tail19
def root30Tail17 : List Nat := 3989 :: root30Tail18
def root30Tail16 : List Nat := 3677 :: root30Tail17
def root30Tail15 : List Nat := 3191 :: root30Tail16
def root30Tail14 : List Nat := 3167 :: root30Tail15
def root30Tail13 : List Nat := 3137 :: root30Tail14
def root30Tail12 : List Nat := 2963 :: root30Tail13
def root30Tail11 : List Nat := 2621 :: root30Tail12
def root30Tail10 : List Nat := 1031 :: root30Tail11
def root30Tail9 : List Nat := 947 :: root30Tail10
def root30Tail8 : List Nat := 743 :: root30Tail9
def root30Tail7 : List Nat := 659 :: root30Tail8
def root30Tail6 : List Nat := 587 :: root30Tail7
def root30Tail5 : List Nat := 491 :: root30Tail6
def root30Tail4 : List Nat := 359 :: root30Tail5
def root30Tail3 : List Nat := 353 :: root30Tail4
def root30Tail2 : List Nat := 239 :: root30Tail3
def root30Tail1 : List Nat := 197 :: root30Tail2
def root30Tail0 : List Nat := 191 :: root30Tail1
def root30Candidates0 : List Nat := [5981, 13187, 21881]
set_option maxRecDepth 8192 in
theorem root30Candidates0_checked : candidates edge 4 137 191 root30Tail1 winner.bound = root30Candidates0 := by
  unfold candidates
  have row : edge 191 = (SortedNatTree.contains FastRows.row191) := by
    change Data.row191.contains = (SortedNatTree.contains FastRows.row191)
    exact FastRows.row191_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child0 : search edge 3 [137, 191] root30Candidates0 328 winner = winner := by decide
theorem root30Step0 : scan edge (search edge 3) 4 [137] 137 root30Tail0 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail1 winner := by
  rw [root30Tail0, scan, if_neg (by decide), root30Candidates0_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail1
    (if 3 ≤ root30Candidates0.length then search edge 3 [137, 191] root30Candidates0 328 winner else winner) = _
  rw [if_pos (by decide), root30Child0]
def root30Candidates1 : List Nat := [947, 6173, 9491, 20543, 21269]
set_option maxRecDepth 8192 in
theorem root30Candidates1_checked : candidates edge 4 137 197 root30Tail2 winner.bound = root30Candidates1 := by
  unfold candidates
  have row : edge 197 = (SortedNatTree.contains FastRows.row197) := by
    change Data.row197.contains = (SortedNatTree.contains FastRows.row197)
    exact FastRows.row197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child1 : search edge 3 [137, 197] root30Candidates1 334 winner = winner := by decide
theorem root30Step1 : scan edge (search edge 3) 4 [137] 137 root30Tail1 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail2 winner := by
  rw [root30Tail1, scan, if_neg (by decide), root30Candidates1_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail2
    (if 3 ≤ root30Candidates1.length then search edge 3 [137, 197] root30Candidates1 334 winner else winner) = _
  rw [if_pos (by decide), root30Child1]
def root30Candidates2 : List Nat := [947, 6197, 11717, 17909, 18773]
set_option maxRecDepth 8192 in
theorem root30Candidates2_checked : candidates edge 4 137 239 root30Tail3 winner.bound = root30Candidates2 := by
  unfold candidates
  have row : edge 239 = (SortedNatTree.contains FastRows.row239) := by
    change Data.row239.contains = (SortedNatTree.contains FastRows.row239)
    exact FastRows.row239_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child2 : search edge 3 [137, 239] root30Candidates2 376 winner = winner := by decide
theorem root30Step2 : scan edge (search edge 3) 4 [137] 137 root30Tail2 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail3 winner := by
  rw [root30Tail2, scan, if_neg (by decide), root30Candidates2_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail3
    (if 3 ≤ root30Candidates2.length then search edge 3 [137, 239] root30Candidates2 376 winner else winner) = _
  rw [if_pos (by decide), root30Child2]
def root30Candidates3 : List Nat := [359, 7517, 8219, 12143, 12269]
set_option maxRecDepth 8192 in
theorem root30Candidates3_checked : candidates edge 4 137 353 root30Tail4 winner.bound = root30Candidates3 := by
  unfold candidates
  have row : edge 353 = (SortedNatTree.contains FastRows.row353) := by
    change Data.row353.contains = (SortedNatTree.contains FastRows.row353)
    exact FastRows.row353_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child3 : search edge 3 [137, 353] root30Candidates3 490 winner = winner := by decide
theorem root30Step3 : scan edge (search edge 3) 4 [137] 137 root30Tail3 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail4 winner := by
  rw [root30Tail3, scan, if_neg (by decide), root30Candidates3_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail4
    (if 3 ≤ root30Candidates3.length then search edge 3 [137, 353] root30Candidates3 490 winner else winner) = _
  rw [if_pos (by decide), root30Child3]
def root30Candidates4 : List Nat := [11927, 14621, 16607, 21269, 21881]
set_option maxRecDepth 8192 in
theorem root30Candidates4_checked : candidates edge 4 137 359 root30Tail5 winner.bound = root30Candidates4 := by
  unfold candidates
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child4 : search edge 3 [137, 359] root30Candidates4 496 winner = winner := by decide
theorem root30Step4 : scan edge (search edge 3) 4 [137] 137 root30Tail4 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail5 winner := by
  rw [root30Tail4, scan, if_neg (by decide), root30Candidates4_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail5
    (if 3 ≤ root30Candidates4.length then search edge 3 [137, 359] root30Candidates4 496 winner else winner) = _
  rw [if_pos (by decide), root30Child4]
def root30Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root30Candidates5_checked : candidates edge 4 137 491 root30Tail6 winner.bound = root30Candidates5 := by
  unfold candidates
  have row : edge 491 = (SortedNatTree.contains FastRows.row491) := by
    change Data.row491.contains = (SortedNatTree.contains FastRows.row491)
    exact FastRows.row491_checked.symm
  rw [row]
  decide
theorem root30Step5 : scan edge (search edge 3) 4 [137] 137 root30Tail5 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail6 winner := by
  rw [root30Tail5, scan, if_neg (by decide), root30Candidates5_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail6
    (if 3 ≤ root30Candidates5.length then search edge 3 [137, 491] root30Candidates5 628 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates6 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root30Candidates6_checked : candidates edge 4 137 587 root30Tail7 winner.bound = root30Candidates6 := by
  unfold candidates
  have row : edge 587 = (SortedNatTree.contains FastRows.row587) := by
    change Data.row587.contains = (SortedNatTree.contains FastRows.row587)
    exact FastRows.row587_checked.symm
  rw [row]
  decide
theorem root30Step6 : scan edge (search edge 3) 4 [137] 137 root30Tail6 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail7 winner := by
  rw [root30Tail6, scan, if_neg (by decide), root30Candidates6_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail7
    (if 3 ≤ root30Candidates6.length then search edge 3 [137, 587] root30Candidates6 724 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates7 : List Nat := [947, 7517, 17807, 21269]
set_option maxRecDepth 8192 in
theorem root30Candidates7_checked : candidates edge 4 137 659 root30Tail8 winner.bound = root30Candidates7 := by
  unfold candidates
  have row : edge 659 = (SortedNatTree.contains FastRows.row659) := by
    change Data.row659.contains = (SortedNatTree.contains FastRows.row659)
    exact FastRows.row659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child7 : search edge 3 [137, 659] root30Candidates7 796 winner = winner := by decide
theorem root30Step7 : scan edge (search edge 3) 4 [137] 137 root30Tail7 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail8 winner := by
  rw [root30Tail7, scan, if_neg (by decide), root30Candidates7_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail8
    (if 3 ≤ root30Candidates7.length then search edge 3 [137, 659] root30Candidates7 796 winner else winner) = _
  rw [if_pos (by decide), root30Child7]
def root30Candidates8 : List Nat := [947, 11927, 22277, 23057]
set_option maxRecDepth 8192 in
theorem root30Candidates8_checked : candidates edge 4 137 743 root30Tail9 winner.bound = root30Candidates8 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child8 : search edge 3 [137, 743] root30Candidates8 880 winner = winner := by decide
theorem root30Step8 : scan edge (search edge 3) 4 [137] 137 root30Tail8 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail9 winner := by
  rw [root30Tail8, scan, if_neg (by decide), root30Candidates8_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail9
    (if 3 ≤ root30Candidates8.length then search edge 3 [137, 743] root30Candidates8 880 winner else winner) = _
  rw [if_pos (by decide), root30Child8]
def root30Candidates9 : List Nat := [5981, 16931, 20543, 21269, 21563, 22721]
set_option maxRecDepth 8192 in
theorem root30Candidates9_checked : candidates edge 4 137 947 root30Tail10 winner.bound = root30Candidates9 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child9 : search edge 3 [137, 947] root30Candidates9 1084 winner = winner := by decide
theorem root30Step9 : scan edge (search edge 3) 4 [137] 137 root30Tail9 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail10 winner := by
  rw [root30Tail9, scan, if_neg (by decide), root30Candidates9_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail10
    (if 3 ≤ root30Candidates9.length then search edge 3 [137, 947] root30Candidates9 1084 winner else winner) = _
  rw [if_pos (by decide), root30Child9]
def root30Candidates10 : List Nat := [5237]
set_option maxRecDepth 8192 in
theorem root30Candidates10_checked : candidates edge 4 137 1031 root30Tail11 winner.bound = root30Candidates10 := by
  unfold candidates
  have row : edge 1031 = (SortedNatTree.contains FastRows.row1031) := by
    change Data.row1031.contains = (SortedNatTree.contains FastRows.row1031)
    exact FastRows.row1031_checked.symm
  rw [row]
  decide
theorem root30Step10 : scan edge (search edge 3) 4 [137] 137 root30Tail10 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail11 winner := by
  rw [root30Tail10, scan, if_neg (by decide), root30Candidates10_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail11
    (if 3 ≤ root30Candidates10.length then search edge 3 [137, 1031] root30Candidates10 1168 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root30Candidates11_checked : candidates edge 4 137 2621 root30Tail12 winner.bound = root30Candidates11 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root30Step11 : scan edge (search edge 3) 4 [137] 137 root30Tail11 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail12 winner := by
  rw [root30Tail11, scan, if_neg (by decide), root30Candidates11_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail12
    (if 3 ≤ root30Candidates11.length then search edge 3 [137, 2621] root30Candidates11 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates12 : List Nat := [13187, 19427]
set_option maxRecDepth 8192 in
theorem root30Candidates12_checked : candidates edge 4 137 2963 root30Tail13 winner.bound = root30Candidates12 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root30Step12 : scan edge (search edge 3) 4 [137] 137 root30Tail12 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail13 winner := by
  rw [root30Tail12, scan, if_neg (by decide), root30Candidates12_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail13
    (if 3 ≤ root30Candidates12.length then search edge 3 [137, 2963] root30Candidates12 3100 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates13 : List Nat := [22067]
set_option maxRecDepth 8192 in
theorem root30Candidates13_checked : candidates edge 4 137 3137 root30Tail14 winner.bound = root30Candidates13 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root30Step13 : scan edge (search edge 3) 4 [137] 137 root30Tail13 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail14 winner := by
  rw [root30Tail13, scan, if_neg (by decide), root30Candidates13_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail14
    (if 3 ≤ root30Candidates13.length then search edge 3 [137, 3137] root30Candidates13 3274 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates14 : List Nat := [9239, 12269]
set_option maxRecDepth 8192 in
theorem root30Candidates14_checked : candidates edge 4 137 3167 root30Tail15 winner.bound = root30Candidates14 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root30Step14 : scan edge (search edge 3) 4 [137] 137 root30Tail14 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail15 winner := by
  rw [root30Tail14, scan, if_neg (by decide), root30Candidates14_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail15
    (if 3 ≤ root30Candidates14.length then search edge 3 [137, 3167] root30Candidates14 3304 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates15 : List Nat := [7043, 11927]
set_option maxRecDepth 8192 in
theorem root30Candidates15_checked : candidates edge 4 137 3191 root30Tail16 winner.bound = root30Candidates15 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
theorem root30Step15 : scan edge (search edge 3) 4 [137] 137 root30Tail15 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail16 winner := by
  rw [root30Tail15, scan, if_neg (by decide), root30Candidates15_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail16
    (if 3 ≤ root30Candidates15.length then search edge 3 [137, 3191] root30Candidates15 3328 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates16 : List Nat := [12269]
set_option maxRecDepth 8192 in
theorem root30Candidates16_checked : candidates edge 4 137 3677 root30Tail17 winner.bound = root30Candidates16 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root30Step16 : scan edge (search edge 3) 4 [137] 137 root30Tail16 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail17 winner := by
  rw [root30Tail16, scan, if_neg (by decide), root30Candidates16_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail17
    (if 3 ≤ root30Candidates16.length then search edge 3 [137, 3677] root30Candidates16 3814 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates17 : List Nat := [7043, 9239, 9491]
set_option maxRecDepth 8192 in
theorem root30Candidates17_checked : candidates edge 4 137 3989 root30Tail18 winner.bound = root30Candidates17 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child17 : search edge 3 [137, 3989] root30Candidates17 4126 winner = winner := by decide
theorem root30Step17 : scan edge (search edge 3) 4 [137] 137 root30Tail17 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail18 winner := by
  rw [root30Tail17, scan, if_neg (by decide), root30Candidates17_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail18
    (if 3 ≤ root30Candidates17.length then search edge 3 [137, 3989] root30Candidates17 4126 winner else winner) = _
  rw [if_pos (by decide), root30Child17]
def root30Candidates18 : List Nat := [6491, 21269]
set_option maxRecDepth 8192 in
theorem root30Candidates18_checked : candidates edge 4 137 4481 root30Tail19 winner.bound = root30Candidates18 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root30Step18 : scan edge (search edge 3) 4 [137] 137 root30Tail18 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail19 winner := by
  rw [root30Tail18, scan, if_neg (by decide), root30Candidates18_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail19
    (if 3 ≤ root30Candidates18.length then search edge 3 [137, 4481] root30Candidates18 4618 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates19 : List Nat := [6971, 9803, 10883, 12269]
set_option maxRecDepth 8192 in
theorem root30Candidates19_checked : candidates edge 4 137 4721 root30Tail20 winner.bound = root30Candidates19 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root30Child19 : search edge 3 [137, 4721] root30Candidates19 4858 winner = winner := by decide
theorem root30Step19 : scan edge (search edge 3) 4 [137] 137 root30Tail19 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail20 winner := by
  rw [root30Tail19, scan, if_neg (by decide), root30Candidates19_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail20
    (if 3 ≤ root30Candidates19.length then search edge 3 [137, 4721] root30Candidates19 4858 winner else winner) = _
  rw [if_pos (by decide), root30Child19]
def root30Candidates20 : List Nat := [7043]
set_option maxRecDepth 8192 in
theorem root30Candidates20_checked : candidates edge 4 137 5237 root30Tail21 winner.bound = root30Candidates20 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root30Step20 : scan edge (search edge 3) 4 [137] 137 root30Tail20 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail21 winner := by
  rw [root30Tail20, scan, if_neg (by decide), root30Candidates20_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail21
    (if 3 ≤ root30Candidates20.length then search edge 3 [137, 5237] root30Candidates20 5374 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root30Candidates21_checked : candidates edge 4 137 5981 root30Tail22 winner.bound = root30Candidates21 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root30Step21 : scan edge (search edge 3) 4 [137] 137 root30Tail21 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail22 winner := by
  rw [root30Tail21, scan, if_neg (by decide), root30Candidates21_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail22
    (if 3 ≤ root30Candidates21.length then search edge 3 [137, 5981] root30Candidates21 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates22 : List Nat := [8231, 17489]
set_option maxRecDepth 8192 in
theorem root30Candidates22_checked : candidates edge 4 137 6173 root30Tail23 winner.bound = root30Candidates22 := by
  unfold candidates
  have row : edge 6173 = (SortedNatTree.contains FastRows.row6173) := by
    change Data.row6173.contains = (SortedNatTree.contains FastRows.row6173)
    exact FastRows.row6173_checked.symm
  rw [row]
  decide
theorem root30Step22 : scan edge (search edge 3) 4 [137] 137 root30Tail22 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail23 winner := by
  rw [root30Tail22, scan, if_neg (by decide), root30Candidates22_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail23
    (if 3 ≤ root30Candidates22.length then search edge 3 [137, 6173] root30Candidates22 6310 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root30Candidates23_checked : candidates edge 4 137 6197 root30Tail24 winner.bound = root30Candidates23 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root30Step23 : scan edge (search edge 3) 4 [137] 137 root30Tail23 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail24 winner := by
  rw [root30Tail23, scan, if_neg (by decide), root30Candidates23_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail24
    (if 3 ≤ root30Candidates23.length then search edge 3 [137, 6197] root30Candidates23 6334 winner else winner) = _
  rw [if_neg (by decide)]
def root30Candidates24 : List Nat := [7577]
set_option maxRecDepth 8192 in
theorem root30Candidates24_checked : candidates edge 4 137 6359 root30Tail25 winner.bound = root30Candidates24 := by
  unfold candidates
  have row : edge 6359 = (SortedNatTree.contains FastRows.row6359) := by
    change Data.row6359.contains = (SortedNatTree.contains FastRows.row6359)
    exact FastRows.row6359_checked.symm
  rw [row]
  decide
theorem root30Step24 : scan edge (search edge 3) 4 [137] 137 root30Tail24 winner =
    scan edge (search edge 3) 4 [137] 137 root30Tail25 winner := by
  rw [root30Tail24, scan, if_neg (by decide), root30Candidates24_checked]
  change scan edge (search edge 3) 4 [137] 137 root30Tail25
    (if 3 ≤ root30Candidates24.length then search edge 3 [137, 6359] root30Candidates24 6496 winner else winner) = _
  rw [if_neg (by decide)]
theorem root30Scan25 : scan edge (search edge 3) 4 [137] 137 root30Tail25 winner = winner := by rw [root30Tail25, scan, if_pos (by decide)]
theorem root30Scan24 : scan edge (search edge 3) 4 [137] 137 root30Tail24 winner = winner :=
  (root30Step24).trans root30Scan25
theorem root30Scan23 : scan edge (search edge 3) 4 [137] 137 root30Tail23 winner = winner :=
  (root30Step23).trans root30Scan24
theorem root30Scan22 : scan edge (search edge 3) 4 [137] 137 root30Tail22 winner = winner :=
  (root30Step22).trans root30Scan23
theorem root30Scan21 : scan edge (search edge 3) 4 [137] 137 root30Tail21 winner = winner :=
  (root30Step21).trans root30Scan22
theorem root30Scan20 : scan edge (search edge 3) 4 [137] 137 root30Tail20 winner = winner :=
  (root30Step20).trans root30Scan21
theorem root30Scan19 : scan edge (search edge 3) 4 [137] 137 root30Tail19 winner = winner :=
  (root30Step19).trans root30Scan20
theorem root30Scan18 : scan edge (search edge 3) 4 [137] 137 root30Tail18 winner = winner :=
  (root30Step18).trans root30Scan19
theorem root30Scan17 : scan edge (search edge 3) 4 [137] 137 root30Tail17 winner = winner :=
  (root30Step17).trans root30Scan18
theorem root30Scan16 : scan edge (search edge 3) 4 [137] 137 root30Tail16 winner = winner :=
  (root30Step16).trans root30Scan17
theorem root30Scan15 : scan edge (search edge 3) 4 [137] 137 root30Tail15 winner = winner :=
  (root30Step15).trans root30Scan16
theorem root30Scan14 : scan edge (search edge 3) 4 [137] 137 root30Tail14 winner = winner :=
  (root30Step14).trans root30Scan15
theorem root30Scan13 : scan edge (search edge 3) 4 [137] 137 root30Tail13 winner = winner :=
  (root30Step13).trans root30Scan14
theorem root30Scan12 : scan edge (search edge 3) 4 [137] 137 root30Tail12 winner = winner :=
  (root30Step12).trans root30Scan13
theorem root30Scan11 : scan edge (search edge 3) 4 [137] 137 root30Tail11 winner = winner :=
  (root30Step11).trans root30Scan12
theorem root30Scan10 : scan edge (search edge 3) 4 [137] 137 root30Tail10 winner = winner :=
  (root30Step10).trans root30Scan11
theorem root30Scan9 : scan edge (search edge 3) 4 [137] 137 root30Tail9 winner = winner :=
  (root30Step9).trans root30Scan10
theorem root30Scan8 : scan edge (search edge 3) 4 [137] 137 root30Tail8 winner = winner :=
  (root30Step8).trans root30Scan9
theorem root30Scan7 : scan edge (search edge 3) 4 [137] 137 root30Tail7 winner = winner :=
  (root30Step7).trans root30Scan8
theorem root30Scan6 : scan edge (search edge 3) 4 [137] 137 root30Tail6 winner = winner :=
  (root30Step6).trans root30Scan7
theorem root30Scan5 : scan edge (search edge 3) 4 [137] 137 root30Tail5 winner = winner :=
  (root30Step5).trans root30Scan6
theorem root30Scan4 : scan edge (search edge 3) 4 [137] 137 root30Tail4 winner = winner :=
  (root30Step4).trans root30Scan5
theorem root30Scan3 : scan edge (search edge 3) 4 [137] 137 root30Tail3 winner = winner :=
  (root30Step3).trans root30Scan4
theorem root30Scan2 : scan edge (search edge 3) 4 [137] 137 root30Tail2 winner = winner :=
  (root30Step2).trans root30Scan3
theorem root30Scan1 : scan edge (search edge 3) 4 [137] 137 root30Tail1 winner = winner :=
  (root30Step1).trans root30Scan2
theorem root30Scan0 : scan edge (search edge 3) 4 [137] 137 root30Tail0 winner = winner :=
  (root30Step0).trans root30Scan1
set_option maxRecDepth 4096 in
theorem root30Branch : search edge 4 [137] root30Candidates 137 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [137] 137 root30Tail0 winner = winner
  exact root30Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
