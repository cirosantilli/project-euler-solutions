import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root25Candidates : List Nat := Data.row107
set_option maxRecDepth 8192 in
theorem root25Candidates_checked : candidates edge 5 0 107 tail26 winner.bound = root25Candidates := by
  apply candidates_eq_row edge 5 0 107 tail26 root25Candidates winner.bound tail26_ordered
  · decide
  · decide
  · funext q
    rfl
def root25Tail75 : List Nat := []
def root25Tail74 : List Nat := 25919 :: root25Tail75
def root25Tail73 : List Nat := 25577 :: root25Tail74
def root25Tail72 : List Nat := 25373 :: root25Tail73
def root25Tail71 : List Nat := 24359 :: root25Tail72
def root25Tail70 : List Nat := 23633 :: root25Tail71
def root25Tail69 : List Nat := 23561 :: root25Tail70
def root25Tail68 : List Nat := 23357 :: root25Tail69
def root25Tail67 : List Nat := 22769 :: root25Tail68
def root25Tail66 : List Nat := 22739 :: root25Tail67
def root25Tail65 : List Nat := 22697 :: root25Tail66
def root25Tail64 : List Nat := 22067 :: root25Tail65
def root25Tail63 : List Nat := 22037 :: root25Tail64
def root25Tail62 : List Nat := 21911 :: root25Tail63
def root25Tail61 : List Nat := 21383 :: root25Tail62
def root25Tail60 : List Nat := 20873 :: root25Tail61
def root25Tail59 : List Nat := 20663 :: root25Tail60
def root25Tail58 : List Nat := 20627 :: root25Tail59
def root25Tail57 : List Nat := 20483 :: root25Tail58
def root25Tail56 : List Nat := 20357 :: root25Tail57
def root25Tail55 : List Nat := 20261 :: root25Tail56
def root25Tail54 : List Nat := 19697 :: root25Tail55
def root25Tail53 : List Nat := 19463 :: root25Tail54
def root25Tail52 : List Nat := 18731 :: root25Tail53
def root25Tail51 : List Nat := 18311 :: root25Tail52
def root25Tail50 : List Nat := 17807 :: root25Tail51
def root25Tail49 : List Nat := 16661 :: root25Tail50
def root25Tail48 : List Nat := 16493 :: root25Tail49
def root25Tail47 : List Nat := 16481 :: root25Tail48
def root25Tail46 : List Nat := 16007 :: root25Tail47
def root25Tail45 : List Nat := 15767 :: root25Tail46
def root25Tail44 : List Nat := 15737 :: root25Tail45
def root25Tail43 : List Nat := 15329 :: root25Tail44
def root25Tail42 : List Nat := 15053 :: root25Tail43
def root25Tail41 : List Nat := 14897 :: root25Tail42
def root25Tail40 : List Nat := 14831 :: root25Tail41
def root25Tail39 : List Nat := 14741 :: root25Tail40
def root25Tail38 : List Nat := 13691 :: root25Tail39
def root25Tail37 : List Nat := 12983 :: root25Tail38
def root25Tail36 : List Nat := 11633 :: root25Tail37
def root25Tail35 : List Nat := 11549 :: root25Tail36
def root25Tail34 : List Nat := 10937 :: root25Tail35
def root25Tail33 : List Nat := 10853 :: root25Tail34
def root25Tail32 : List Nat := 10631 :: root25Tail33
def root25Tail31 : List Nat := 10589 :: root25Tail32
def root25Tail30 : List Nat := 10079 :: root25Tail31
def root25Tail29 : List Nat := 9887 :: root25Tail30
def root25Tail28 : List Nat := 9629 :: root25Tail29
def root25Tail27 : List Nat := 7607 :: root25Tail28
def root25Tail26 : List Nat := 6917 :: root25Tail27
def root25Tail25 : List Nat := 6563 :: root25Tail26
def root25Tail24 : List Nat := 6359 :: root25Tail25
def root25Tail23 : List Nat := 5897 :: root25Tail24
def root25Tail22 : List Nat := 5531 :: root25Tail23
def root25Tail21 : List Nat := 5507 :: root25Tail22
def root25Tail20 : List Nat := 5279 :: root25Tail21
def root25Tail19 : List Nat := 4973 :: root25Tail20
def root25Tail18 : List Nat := 4919 :: root25Tail19
def root25Tail17 : List Nat := 4691 :: root25Tail18
def root25Tail16 : List Nat := 4523 :: root25Tail17
def root25Tail15 : List Nat := 4289 :: root25Tail16
def root25Tail14 : List Nat := 3221 :: root25Tail15
def root25Tail13 : List Nat := 2999 :: root25Tail14
def root25Tail12 : List Nat := 2957 :: root25Tail13
def root25Tail11 : List Nat := 2843 :: root25Tail12
def root25Tail10 : List Nat := 2633 :: root25Tail11
def root25Tail9 : List Nat := 2381 :: root25Tail10
def root25Tail8 : List Nat := 2339 :: root25Tail9
def root25Tail7 : List Nat := 2129 :: root25Tail8
def root25Tail6 : List Nat := 1907 :: root25Tail7
def root25Tail5 : List Nat := 1601 :: root25Tail6
def root25Tail4 : List Nat := 1439 :: root25Tail5
def root25Tail3 : List Nat := 1193 :: root25Tail4
def root25Tail2 : List Nat := 1061 :: root25Tail3
def root25Tail1 : List Nat := 857 :: root25Tail2
def root25Tail0 : List Nat := 449 :: root25Tail1
def root25Candidates0 : List Nat := [10079, 22739]
set_option maxRecDepth 8192 in
theorem root25Candidates0_checked : candidates edge 4 107 449 root25Tail1 winner.bound = root25Candidates0 := by
  unfold candidates
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row]
  decide
theorem root25Step0 : scan edge (search edge 3) 4 [107] 107 root25Tail0 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail1 winner := by
  rw [root25Tail0, scan, if_neg (by decide), root25Candidates0_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail1
    (if 3 ≤ root25Candidates0.length then search edge 3 [107, 449] root25Candidates0 556 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates1 : List Nat := [1601, 2633, 10589]
set_option maxRecDepth 8192 in
theorem root25Candidates1_checked : candidates edge 4 107 857 root25Tail2 winner.bound = root25Candidates1 := by
  unfold candidates
  have row : edge 857 = (SortedNatTree.contains FastRows.row857) := by
    change Data.row857.contains = (SortedNatTree.contains FastRows.row857)
    exact FastRows.row857_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child1 : search edge 3 [107, 857] root25Candidates1 964 winner = winner := by decide
theorem root25Step1 : scan edge (search edge 3) 4 [107] 107 root25Tail1 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail2 winner := by
  rw [root25Tail1, scan, if_neg (by decide), root25Candidates1_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail2
    (if 3 ≤ root25Candidates1.length then search edge 3 [107, 857] root25Candidates1 964 winner else winner) = _
  rw [if_pos (by decide), root25Child1]
def root25Candidates2 : List Nat := [4973]
set_option maxRecDepth 8192 in
theorem root25Candidates2_checked : candidates edge 4 107 1061 root25Tail3 winner.bound = root25Candidates2 := by
  unfold candidates
  have row : edge 1061 = (SortedNatTree.contains FastRows.row1061) := by
    change Data.row1061.contains = (SortedNatTree.contains FastRows.row1061)
    exact FastRows.row1061_checked.symm
  rw [row]
  decide
theorem root25Step2 : scan edge (search edge 3) 4 [107] 107 root25Tail2 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail3 winner := by
  rw [root25Tail2, scan, if_neg (by decide), root25Candidates2_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail3
    (if 3 ≤ root25Candidates2.length then search edge 3 [107, 1061] root25Candidates2 1168 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates3 : List Nat := [5507, 16007, 22739]
set_option maxRecDepth 8192 in
theorem root25Candidates3_checked : candidates edge 4 107 1193 root25Tail4 winner.bound = root25Candidates3 := by
  unfold candidates
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child3 : search edge 3 [107, 1193] root25Candidates3 1300 winner = winner := by decide
theorem root25Step3 : scan edge (search edge 3) 4 [107] 107 root25Tail3 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail4 winner := by
  rw [root25Tail3, scan, if_neg (by decide), root25Candidates3_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail4
    (if 3 ≤ root25Candidates3.length then search edge 3 [107, 1193] root25Candidates3 1300 winner else winner) = _
  rw [if_pos (by decide), root25Child3]
def root25Candidates4 : List Nat := [5279, 10589, 19463, 21383]
set_option maxRecDepth 8192 in
theorem root25Candidates4_checked : candidates edge 4 107 1439 root25Tail5 winner.bound = root25Candidates4 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child4 : search edge 3 [107, 1439] root25Candidates4 1546 winner = winner := by decide
theorem root25Step4 : scan edge (search edge 3) 4 [107] 107 root25Tail4 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail5 winner := by
  rw [root25Tail4, scan, if_neg (by decide), root25Candidates4_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail5
    (if 3 ≤ root25Candidates4.length then search edge 3 [107, 1439] root25Candidates4 1546 winner else winner) = _
  rw [if_pos (by decide), root25Child4]
def root25Candidates5 : List Nat := [6359, 14897, 22769]
set_option maxRecDepth 8192 in
theorem root25Candidates5_checked : candidates edge 4 107 1601 root25Tail6 winner.bound = root25Candidates5 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child5 : search edge 3 [107, 1601] root25Candidates5 1708 winner = winner := by decide
theorem root25Step5 : scan edge (search edge 3) 4 [107] 107 root25Tail5 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail6 winner := by
  rw [root25Tail5, scan, if_neg (by decide), root25Candidates5_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail6
    (if 3 ≤ root25Candidates5.length then search edge 3 [107, 1601] root25Candidates5 1708 winner else winner) = _
  rw [if_pos (by decide), root25Child5]
def root25Candidates6 : List Nat := [5897, 16007]
set_option maxRecDepth 8192 in
theorem root25Candidates6_checked : candidates edge 4 107 1907 root25Tail7 winner.bound = root25Candidates6 := by
  unfold candidates
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row]
  decide
theorem root25Step6 : scan edge (search edge 3) 4 [107] 107 root25Tail6 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail7 winner := by
  rw [root25Tail6, scan, if_neg (by decide), root25Candidates6_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail7
    (if 3 ≤ root25Candidates6.length then search edge 3 [107, 1907] root25Candidates6 2014 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates7 : List Nat := [10937, 16481, 19697]
set_option maxRecDepth 8192 in
theorem root25Candidates7_checked : candidates edge 4 107 2129 root25Tail8 winner.bound = root25Candidates7 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child7 : search edge 3 [107, 2129] root25Candidates7 2236 winner = winner := by decide
theorem root25Step7 : scan edge (search edge 3) 4 [107] 107 root25Tail7 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail8 winner := by
  rw [root25Tail7, scan, if_neg (by decide), root25Candidates7_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail8
    (if 3 ≤ root25Candidates7.length then search edge 3 [107, 2129] root25Candidates7 2236 winner else winner) = _
  rw [if_pos (by decide), root25Child7]
def root25Candidates8 : List Nat := [14741]
set_option maxRecDepth 8192 in
theorem root25Candidates8_checked : candidates edge 4 107 2339 root25Tail9 winner.bound = root25Candidates8 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root25Step8 : scan edge (search edge 3) 4 [107] 107 root25Tail8 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail9 winner := by
  rw [root25Tail8, scan, if_neg (by decide), root25Candidates8_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail9
    (if 3 ≤ root25Candidates8.length then search edge 3 [107, 2339] root25Candidates8 2446 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates9 : List Nat := [3221, 6563, 22739]
set_option maxRecDepth 8192 in
theorem root25Candidates9_checked : candidates edge 4 107 2381 root25Tail10 winner.bound = root25Candidates9 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child9 : search edge 3 [107, 2381] root25Candidates9 2488 winner = winner := by decide
theorem root25Step9 : scan edge (search edge 3) 4 [107] 107 root25Tail9 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail10 winner := by
  rw [root25Tail9, scan, if_neg (by decide), root25Candidates9_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail10
    (if 3 ≤ root25Candidates9.length then search edge 3 [107, 2381] root25Candidates9 2488 winner else winner) = _
  rw [if_pos (by decide), root25Child9]
def root25Candidates10 : List Nat := [11549]
set_option maxRecDepth 8192 in
theorem root25Candidates10_checked : candidates edge 4 107 2633 root25Tail11 winner.bound = root25Candidates10 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root25Step10 : scan edge (search edge 3) 4 [107] 107 root25Tail10 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail11 winner := by
  rw [root25Tail10, scan, if_neg (by decide), root25Candidates10_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail11
    (if 3 ≤ root25Candidates10.length then search edge 3 [107, 2633] root25Candidates10 2740 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates11 : List Nat := [19463]
set_option maxRecDepth 8192 in
theorem root25Candidates11_checked : candidates edge 4 107 2843 root25Tail12 winner.bound = root25Candidates11 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root25Step11 : scan edge (search edge 3) 4 [107] 107 root25Tail11 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail12 winner := by
  rw [root25Tail11, scan, if_neg (by decide), root25Candidates11_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail12
    (if 3 ≤ root25Candidates11.length then search edge 3 [107, 2843] root25Candidates11 2950 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates12 : List Nat := [4973]
set_option maxRecDepth 8192 in
theorem root25Candidates12_checked : candidates edge 4 107 2957 root25Tail13 winner.bound = root25Candidates12 := by
  unfold candidates
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row]
  decide
theorem root25Step12 : scan edge (search edge 3) 4 [107] 107 root25Tail12 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail13 winner := by
  rw [root25Tail12, scan, if_neg (by decide), root25Candidates12_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail13
    (if 3 ≤ root25Candidates12.length then search edge 3 [107, 2957] root25Candidates12 3064 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates13 : List Nat := [22067]
set_option maxRecDepth 8192 in
theorem root25Candidates13_checked : candidates edge 4 107 2999 root25Tail14 winner.bound = root25Candidates13 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root25Step13 : scan edge (search edge 3) 4 [107] 107 root25Tail13 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail14 winner := by
  rw [root25Tail13, scan, if_neg (by decide), root25Candidates13_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail14
    (if 3 ≤ root25Candidates13.length then search edge 3 [107, 2999] root25Candidates13 3106 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates14 : List Nat := [19697]
set_option maxRecDepth 8192 in
theorem root25Candidates14_checked : candidates edge 4 107 3221 root25Tail15 winner.bound = root25Candidates14 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root25Step14 : scan edge (search edge 3) 4 [107] 107 root25Tail14 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail15 winner := by
  rw [root25Tail14, scan, if_neg (by decide), root25Candidates14_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail15
    (if 3 ≤ root25Candidates14.length then search edge 3 [107, 3221] root25Candidates14 3328 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates15 : List Nat := [5507, 15737]
set_option maxRecDepth 8192 in
theorem root25Candidates15_checked : candidates edge 4 107 4289 root25Tail16 winner.bound = root25Candidates15 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root25Step15 : scan edge (search edge 3) 4 [107] 107 root25Tail15 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail16 winner := by
  rw [root25Tail15, scan, if_neg (by decide), root25Candidates15_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail16
    (if 3 ≤ root25Candidates15.length then search edge 3 [107, 4289] root25Candidates15 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root25Candidates16_checked : candidates edge 4 107 4523 root25Tail17 winner.bound = root25Candidates16 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root25Step16 : scan edge (search edge 3) 4 [107] 107 root25Tail16 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail17 winner := by
  rw [root25Tail16, scan, if_neg (by decide), root25Candidates16_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail17
    (if 3 ≤ root25Candidates16.length then search edge 3 [107, 4523] root25Candidates16 4630 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates17 : List Nat := [16481, 18311]
set_option maxRecDepth 8192 in
theorem root25Candidates17_checked : candidates edge 4 107 4691 root25Tail18 winner.bound = root25Candidates17 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root25Step17 : scan edge (search edge 3) 4 [107] 107 root25Tail17 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail18 winner := by
  rw [root25Tail17, scan, if_neg (by decide), root25Candidates17_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail18
    (if 3 ≤ root25Candidates17.length then search edge 3 [107, 4691] root25Candidates17 4798 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates18 : List Nat := [5897, 7607, 10631, 13691]
set_option maxRecDepth 8192 in
theorem root25Candidates18_checked : candidates edge 4 107 4919 root25Tail19 winner.bound = root25Candidates18 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root25Child18 : search edge 3 [107, 4919] root25Candidates18 5026 winner = winner := by decide
theorem root25Step18 : scan edge (search edge 3) 4 [107] 107 root25Tail18 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail19 winner := by
  rw [root25Tail18, scan, if_neg (by decide), root25Candidates18_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail19
    (if 3 ≤ root25Candidates18.length then search edge 3 [107, 4919] root25Candidates18 5026 winner else winner) = _
  rw [if_pos (by decide), root25Child18]
def root25Candidates19 : List Nat := [9629, 10631]
set_option maxRecDepth 8192 in
theorem root25Candidates19_checked : candidates edge 4 107 4973 root25Tail20 winner.bound = root25Candidates19 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root25Step19 : scan edge (search edge 3) 4 [107] 107 root25Tail19 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail20 winner := by
  rw [root25Tail19, scan, if_neg (by decide), root25Candidates19_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail20
    (if 3 ≤ root25Candidates19.length then search edge 3 [107, 4973] root25Candidates19 5080 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root25Candidates20_checked : candidates edge 4 107 5279 root25Tail21 winner.bound = root25Candidates20 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root25Step20 : scan edge (search edge 3) 4 [107] 107 root25Tail20 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail21 winner := by
  rw [root25Tail20, scan, if_neg (by decide), root25Candidates20_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail21
    (if 3 ≤ root25Candidates20.length then search edge 3 [107, 5279] root25Candidates20 5386 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates21 : List Nat := [14831, 18731]
set_option maxRecDepth 8192 in
theorem root25Candidates21_checked : candidates edge 4 107 5507 root25Tail22 winner.bound = root25Candidates21 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root25Step21 : scan edge (search edge 3) 4 [107] 107 root25Tail21 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail22 winner := by
  rw [root25Tail21, scan, if_neg (by decide), root25Candidates21_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail22
    (if 3 ≤ root25Candidates21.length then search edge 3 [107, 5507] root25Candidates21 5614 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates22 : List Nat := [11549]
set_option maxRecDepth 8192 in
theorem root25Candidates22_checked : candidates edge 4 107 5531 root25Tail23 winner.bound = root25Candidates22 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root25Step22 : scan edge (search edge 3) 4 [107] 107 root25Tail22 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail23 winner := by
  rw [root25Tail22, scan, if_neg (by decide), root25Candidates22_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail23
    (if 3 ≤ root25Candidates22.length then search edge 3 [107, 5531] root25Candidates22 5638 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates23 : List Nat := [10937]
set_option maxRecDepth 8192 in
theorem root25Candidates23_checked : candidates edge 4 107 5897 root25Tail24 winner.bound = root25Candidates23 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root25Step23 : scan edge (search edge 3) 4 [107] 107 root25Tail23 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail24 winner := by
  rw [root25Tail23, scan, if_neg (by decide), root25Candidates23_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail24
    (if 3 ≤ root25Candidates23.length then search edge 3 [107, 5897] root25Candidates23 6004 winner else winner) = _
  rw [if_neg (by decide)]
def root25Candidates24 : List Nat := [15767]
set_option maxRecDepth 8192 in
theorem root25Candidates24_checked : candidates edge 4 107 6359 root25Tail25 winner.bound = root25Candidates24 := by
  unfold candidates
  have row : edge 6359 = (SortedNatTree.contains FastRows.row6359) := by
    change Data.row6359.contains = (SortedNatTree.contains FastRows.row6359)
    exact FastRows.row6359_checked.symm
  rw [row]
  decide
theorem root25Step24 : scan edge (search edge 3) 4 [107] 107 root25Tail24 winner =
    scan edge (search edge 3) 4 [107] 107 root25Tail25 winner := by
  rw [root25Tail24, scan, if_neg (by decide), root25Candidates24_checked]
  change scan edge (search edge 3) 4 [107] 107 root25Tail25
    (if 3 ≤ root25Candidates24.length then search edge 3 [107, 6359] root25Candidates24 6466 winner else winner) = _
  rw [if_neg (by decide)]
theorem root25Scan25 : scan edge (search edge 3) 4 [107] 107 root25Tail25 winner = winner := by rw [root25Tail25, scan, if_pos (by decide)]
theorem root25Scan24 : scan edge (search edge 3) 4 [107] 107 root25Tail24 winner = winner :=
  (root25Step24).trans root25Scan25
theorem root25Scan23 : scan edge (search edge 3) 4 [107] 107 root25Tail23 winner = winner :=
  (root25Step23).trans root25Scan24
theorem root25Scan22 : scan edge (search edge 3) 4 [107] 107 root25Tail22 winner = winner :=
  (root25Step22).trans root25Scan23
theorem root25Scan21 : scan edge (search edge 3) 4 [107] 107 root25Tail21 winner = winner :=
  (root25Step21).trans root25Scan22
theorem root25Scan20 : scan edge (search edge 3) 4 [107] 107 root25Tail20 winner = winner :=
  (root25Step20).trans root25Scan21
theorem root25Scan19 : scan edge (search edge 3) 4 [107] 107 root25Tail19 winner = winner :=
  (root25Step19).trans root25Scan20
theorem root25Scan18 : scan edge (search edge 3) 4 [107] 107 root25Tail18 winner = winner :=
  (root25Step18).trans root25Scan19
theorem root25Scan17 : scan edge (search edge 3) 4 [107] 107 root25Tail17 winner = winner :=
  (root25Step17).trans root25Scan18
theorem root25Scan16 : scan edge (search edge 3) 4 [107] 107 root25Tail16 winner = winner :=
  (root25Step16).trans root25Scan17
theorem root25Scan15 : scan edge (search edge 3) 4 [107] 107 root25Tail15 winner = winner :=
  (root25Step15).trans root25Scan16
theorem root25Scan14 : scan edge (search edge 3) 4 [107] 107 root25Tail14 winner = winner :=
  (root25Step14).trans root25Scan15
theorem root25Scan13 : scan edge (search edge 3) 4 [107] 107 root25Tail13 winner = winner :=
  (root25Step13).trans root25Scan14
theorem root25Scan12 : scan edge (search edge 3) 4 [107] 107 root25Tail12 winner = winner :=
  (root25Step12).trans root25Scan13
theorem root25Scan11 : scan edge (search edge 3) 4 [107] 107 root25Tail11 winner = winner :=
  (root25Step11).trans root25Scan12
theorem root25Scan10 : scan edge (search edge 3) 4 [107] 107 root25Tail10 winner = winner :=
  (root25Step10).trans root25Scan11
theorem root25Scan9 : scan edge (search edge 3) 4 [107] 107 root25Tail9 winner = winner :=
  (root25Step9).trans root25Scan10
theorem root25Scan8 : scan edge (search edge 3) 4 [107] 107 root25Tail8 winner = winner :=
  (root25Step8).trans root25Scan9
theorem root25Scan7 : scan edge (search edge 3) 4 [107] 107 root25Tail7 winner = winner :=
  (root25Step7).trans root25Scan8
theorem root25Scan6 : scan edge (search edge 3) 4 [107] 107 root25Tail6 winner = winner :=
  (root25Step6).trans root25Scan7
theorem root25Scan5 : scan edge (search edge 3) 4 [107] 107 root25Tail5 winner = winner :=
  (root25Step5).trans root25Scan6
theorem root25Scan4 : scan edge (search edge 3) 4 [107] 107 root25Tail4 winner = winner :=
  (root25Step4).trans root25Scan5
theorem root25Scan3 : scan edge (search edge 3) 4 [107] 107 root25Tail3 winner = winner :=
  (root25Step3).trans root25Scan4
theorem root25Scan2 : scan edge (search edge 3) 4 [107] 107 root25Tail2 winner = winner :=
  (root25Step2).trans root25Scan3
theorem root25Scan1 : scan edge (search edge 3) 4 [107] 107 root25Tail1 winner = winner :=
  (root25Step1).trans root25Scan2
theorem root25Scan0 : scan edge (search edge 3) 4 [107] 107 root25Tail0 winner = winner :=
  (root25Step0).trans root25Scan1
set_option maxRecDepth 4096 in
theorem root25Branch : search edge 4 [107] root25Candidates 107 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [107] 107 root25Tail0 winner = winner
  exact root25Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
