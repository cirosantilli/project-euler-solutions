import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root51Candidates : List Nat := Data.row251
set_option maxRecDepth 8192 in
theorem root51Candidates_checked : candidates edge 5 0 251 tail52 winner.bound = root51Candidates := by
  apply candidates_eq_row edge 5 0 251 tail52 root51Candidates winner.bound tail52_ordered
  · decide
  · decide
  · funext q
    rfl
def root51Tail81 : List Nat := []
def root51Tail80 : List Nat := 25673 :: root51Tail81
def root51Tail79 : List Nat := 24971 :: root51Tail80
def root51Tail78 : List Nat := 24917 :: root51Tail79
def root51Tail77 : List Nat := 24683 :: root51Tail78
def root51Tail76 : List Nat := 23993 :: root51Tail77
def root51Tail75 : List Nat := 23669 :: root51Tail76
def root51Tail74 : List Nat := 23081 :: root51Tail75
def root51Tail73 : List Nat := 22859 :: root51Tail74
def root51Tail72 : List Nat := 22817 :: root51Tail73
def root51Tail71 : List Nat := 22751 :: root51Tail72
def root51Tail70 : List Nat := 22367 :: root51Tail71
def root51Tail69 : List Nat := 22037 :: root51Tail70
def root51Tail68 : List Nat := 21893 :: root51Tail69
def root51Tail67 : List Nat := 21611 :: root51Tail68
def root51Tail66 : List Nat := 21521 :: root51Tail67
def root51Tail65 : List Nat := 21269 :: root51Tail66
def root51Tail64 : List Nat := 21011 :: root51Tail65
def root51Tail63 : List Nat := 20939 :: root51Tail64
def root51Tail62 : List Nat := 20549 :: root51Tail63
def root51Tail61 : List Nat := 20399 :: root51Tail62
def root51Tail60 : List Nat := 20021 :: root51Tail61
def root51Tail59 : List Nat := 18329 :: root51Tail60
def root51Tail58 : List Nat := 16607 :: root51Tail59
def root51Tail57 : List Nat := 16493 :: root51Tail58
def root51Tail56 : List Nat := 15731 :: root51Tail57
def root51Tail55 : List Nat := 15641 :: root51Tail56
def root51Tail54 : List Nat := 15173 :: root51Tail55
def root51Tail53 : List Nat := 14081 :: root51Tail54
def root51Tail52 : List Nat := 14051 :: root51Tail53
def root51Tail51 : List Nat := 13367 :: root51Tail52
def root51Tail50 : List Nat := 13337 :: root51Tail51
def root51Tail49 : List Nat := 13331 :: root51Tail50
def root51Tail48 : List Nat := 12911 :: root51Tail49
def root51Tail47 : List Nat := 11813 :: root51Tail48
def root51Tail46 : List Nat := 11579 :: root51Tail47
def root51Tail45 : List Nat := 11447 :: root51Tail46
def root51Tail44 : List Nat := 10427 :: root51Tail45
def root51Tail43 : List Nat := 10067 :: root51Tail44
def root51Tail42 : List Nat := 9467 :: root51Tail43
def root51Tail41 : List Nat := 8951 :: root51Tail42
def root51Tail40 : List Nat := 8933 :: root51Tail41
def root51Tail39 : List Nat := 8783 :: root51Tail40
def root51Tail38 : List Nat := 8387 :: root51Tail39
def root51Tail37 : List Nat := 8093 :: root51Tail38
def root51Tail36 : List Nat := 7919 :: root51Tail37
def root51Tail35 : List Nat := 7901 :: root51Tail36
def root51Tail34 : List Nat := 7499 :: root51Tail35
def root51Tail33 : List Nat := 7127 :: root51Tail34
def root51Tail32 : List Nat := 7079 :: root51Tail33
def root51Tail31 : List Nat := 6863 :: root51Tail32
def root51Tail30 : List Nat := 6581 :: root51Tail31
def root51Tail29 : List Nat := 6473 :: root51Tail30
def root51Tail28 : List Nat := 6197 :: root51Tail29
def root51Tail27 : List Nat := 5897 :: root51Tail28
def root51Tail26 : List Nat := 5651 :: root51Tail27
def root51Tail25 : List Nat := 5273 :: root51Tail26
def root51Tail24 : List Nat := 5237 :: root51Tail25
def root51Tail23 : List Nat := 5189 :: root51Tail24
def root51Tail22 : List Nat := 5147 :: root51Tail23
def root51Tail21 : List Nat := 4877 :: root51Tail22
def root51Tail20 : List Nat := 4019 :: root51Tail21
def root51Tail19 : List Nat := 3803 :: root51Tail20
def root51Tail18 : List Nat := 3557 :: root51Tail19
def root51Tail17 : List Nat := 3347 :: root51Tail18
def root51Tail16 : List Nat := 3011 :: root51Tail17
def root51Tail15 : List Nat := 2999 :: root51Tail16
def root51Tail14 : List Nat := 2801 :: root51Tail15
def root51Tail13 : List Nat := 2789 :: root51Tail14
def root51Tail12 : List Nat := 2417 :: root51Tail13
def root51Tail11 : List Nat := 2207 :: root51Tail12
def root51Tail10 : List Nat := 2063 :: root51Tail11
def root51Tail9 : List Nat := 1979 :: root51Tail10
def root51Tail8 : List Nat := 1709 :: root51Tail9
def root51Tail7 : List Nat := 1433 :: root51Tail8
def root51Tail6 : List Nat := 1151 :: root51Tail7
def root51Tail5 : List Nat := 971 :: root51Tail6
def root51Tail4 : List Nat := 941 :: root51Tail5
def root51Tail3 : List Nat := 761 :: root51Tail4
def root51Tail2 : List Nat := 491 :: root51Tail3
def root51Tail1 : List Nat := 431 :: root51Tail2
def root51Tail0 : List Nat := 347 :: root51Tail1
def root51Candidates0 : List Nat := [7499]
set_option maxRecDepth 8192 in
theorem root51Candidates0_checked : candidates edge 4 251 347 root51Tail1 winner.bound = root51Candidates0 := by
  unfold candidates
  have row : edge 347 = (SortedNatTree.contains FastRows.row347) := by
    change Data.row347.contains = (SortedNatTree.contains FastRows.row347)
    exact FastRows.row347_checked.symm
  rw [row]
  decide
theorem root51Step0 : scan edge (search edge 3) 4 [251] 251 root51Tail0 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail1 winner := by
  rw [root51Tail0, scan, if_neg (by decide), root51Candidates0_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail1
    (if 3 ≤ root51Candidates0.length then search edge 3 [251, 347] root51Candidates0 598 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates1 : List Nat := [2417, 6197, 7127, 13331]
set_option maxRecDepth 8192 in
theorem root51Candidates1_checked : candidates edge 4 251 431 root51Tail2 winner.bound = root51Candidates1 := by
  unfold candidates
  have row : edge 431 = (SortedNatTree.contains FastRows.row431) := by
    change Data.row431.contains = (SortedNatTree.contains FastRows.row431)
    exact FastRows.row431_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root51Child1 : search edge 3 [251, 431] root51Candidates1 682 winner = winner := by decide
theorem root51Step1 : scan edge (search edge 3) 4 [251] 251 root51Tail1 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail2 winner := by
  rw [root51Tail1, scan, if_neg (by decide), root51Candidates1_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail2
    (if 3 ≤ root51Candidates1.length then search edge 3 [251, 431] root51Candidates1 682 winner else winner) = _
  rw [if_pos (by decide), root51Child1]
def root51Candidates2 : List Nat := [7079, 11813, 20549]
set_option maxRecDepth 8192 in
theorem root51Candidates2_checked : candidates edge 4 251 491 root51Tail3 winner.bound = root51Candidates2 := by
  unfold candidates
  have row : edge 491 = (SortedNatTree.contains FastRows.row491) := by
    change Data.row491.contains = (SortedNatTree.contains FastRows.row491)
    exact FastRows.row491_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root51Child2 : search edge 3 [251, 491] root51Candidates2 742 winner = winner := by decide
theorem root51Step2 : scan edge (search edge 3) 4 [251] 251 root51Tail2 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail3 winner := by
  rw [root51Tail2, scan, if_neg (by decide), root51Candidates2_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail3
    (if 3 ≤ root51Candidates2.length then search edge 3 [251, 491] root51Candidates2 742 winner else winner) = _
  rw [if_pos (by decide), root51Child2]
def root51Candidates3 : List Nat := [20939, 23081]
set_option maxRecDepth 8192 in
theorem root51Candidates3_checked : candidates edge 4 251 761 root51Tail4 winner.bound = root51Candidates3 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
theorem root51Step3 : scan edge (search edge 3) 4 [251] 251 root51Tail3 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail4 winner := by
  rw [root51Tail3, scan, if_neg (by decide), root51Candidates3_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail4
    (if 3 ≤ root51Candidates3.length then search edge 3 [251, 761] root51Candidates3 1012 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates4 : List Nat := [1709]
set_option maxRecDepth 8192 in
theorem root51Candidates4_checked : candidates edge 4 251 941 root51Tail5 winner.bound = root51Candidates4 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
theorem root51Step4 : scan edge (search edge 3) 4 [251] 251 root51Tail4 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail5 winner := by
  rw [root51Tail4, scan, if_neg (by decide), root51Candidates4_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail5
    (if 3 ≤ root51Candidates4.length then search edge 3 [251, 941] root51Candidates4 1192 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates5 : List Nat := [7919]
set_option maxRecDepth 8192 in
theorem root51Candidates5_checked : candidates edge 4 251 971 root51Tail6 winner.bound = root51Candidates5 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
theorem root51Step5 : scan edge (search edge 3) 4 [251] 251 root51Tail5 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail6 winner := by
  rw [root51Tail5, scan, if_neg (by decide), root51Candidates5_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail6
    (if 3 ≤ root51Candidates5.length then search edge 3 [251, 971] root51Candidates5 1222 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates6 : List Nat := [1433, 2789, 5897, 6473, 6863, 10427, 20939]
set_option maxRecDepth 8192 in
theorem root51Candidates6_checked : candidates edge 4 251 1151 root51Tail7 winner.bound = root51Candidates6 := by
  unfold candidates
  have row : edge 1151 = (SortedNatTree.contains FastRows.row1151) := by
    change Data.row1151.contains = (SortedNatTree.contains FastRows.row1151)
    exact FastRows.row1151_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root51Child6 : search edge 3 [251, 1151] root51Candidates6 1402 winner = winner := by decide
theorem root51Step6 : scan edge (search edge 3) 4 [251] 251 root51Tail6 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail7 winner := by
  rw [root51Tail6, scan, if_neg (by decide), root51Candidates6_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail7
    (if 3 ≤ root51Candidates6.length then search edge 3 [251, 1151] root51Candidates6 1402 winner else winner) = _
  rw [if_pos (by decide), root51Child6]
def root51Candidates7 : List Nat := [2999]
set_option maxRecDepth 8192 in
theorem root51Candidates7_checked : candidates edge 4 251 1433 root51Tail8 winner.bound = root51Candidates7 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
theorem root51Step7 : scan edge (search edge 3) 4 [251] 251 root51Tail7 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail8 winner := by
  rw [root51Tail7, scan, if_neg (by decide), root51Candidates7_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail8
    (if 3 ≤ root51Candidates7.length then search edge 3 [251, 1433] root51Candidates7 1684 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates8 : List Nat := [22751]
set_option maxRecDepth 8192 in
theorem root51Candidates8_checked : candidates edge 4 251 1709 root51Tail9 winner.bound = root51Candidates8 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
theorem root51Step8 : scan edge (search edge 3) 4 [251] 251 root51Tail8 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail9 winner := by
  rw [root51Tail8, scan, if_neg (by decide), root51Candidates8_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail9
    (if 3 ≤ root51Candidates8.length then search edge 3 [251, 1709] root51Candidates8 1960 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates9 : List Nat := [2207, 2999, 22037, 22367]
set_option maxRecDepth 8192 in
theorem root51Candidates9_checked : candidates edge 4 251 1979 root51Tail10 winner.bound = root51Candidates9 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root51Child9 : search edge 3 [251, 1979] root51Candidates9 2230 winner = winner := by decide
theorem root51Step9 : scan edge (search edge 3) 4 [251] 251 root51Tail9 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail10 winner := by
  rw [root51Tail9, scan, if_neg (by decide), root51Candidates9_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail10
    (if 3 ≤ root51Candidates9.length then search edge 3 [251, 1979] root51Candidates9 2230 winner else winner) = _
  rw [if_pos (by decide), root51Child9]
def root51Candidates10 : List Nat := [3803, 11579]
set_option maxRecDepth 8192 in
theorem root51Candidates10_checked : candidates edge 4 251 2063 root51Tail11 winner.bound = root51Candidates10 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root51Step10 : scan edge (search edge 3) 4 [251] 251 root51Tail10 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail11 winner := by
  rw [root51Tail10, scan, if_neg (by decide), root51Candidates10_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail11
    (if 3 ≤ root51Candidates10.length then search edge 3 [251, 2063] root51Candidates10 2314 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates11 : List Nat := [3347]
set_option maxRecDepth 8192 in
theorem root51Candidates11_checked : candidates edge 4 251 2207 root51Tail12 winner.bound = root51Candidates11 := by
  unfold candidates
  have row : edge 2207 = (SortedNatTree.contains FastRows.row2207) := by
    change Data.row2207.contains = (SortedNatTree.contains FastRows.row2207)
    exact FastRows.row2207_checked.symm
  rw [row]
  decide
theorem root51Step11 : scan edge (search edge 3) 4 [251] 251 root51Tail11 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail12 winner := by
  rw [root51Tail11, scan, if_neg (by decide), root51Candidates11_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail12
    (if 3 ≤ root51Candidates11.length then search edge 3 [251, 2207] root51Candidates11 2458 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates12 : List Nat := [8951]
set_option maxRecDepth 8192 in
theorem root51Candidates12_checked : candidates edge 4 251 2417 root51Tail13 winner.bound = root51Candidates12 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root51Step12 : scan edge (search edge 3) 4 [251] 251 root51Tail12 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail13 winner := by
  rw [root51Tail12, scan, if_neg (by decide), root51Candidates12_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail13
    (if 3 ≤ root51Candidates12.length then search edge 3 [251, 2417] root51Candidates12 2668 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates13 : List Nat := [8951]
set_option maxRecDepth 8192 in
theorem root51Candidates13_checked : candidates edge 4 251 2789 root51Tail14 winner.bound = root51Candidates13 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root51Step13 : scan edge (search edge 3) 4 [251] 251 root51Tail13 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail14 winner := by
  rw [root51Tail13, scan, if_neg (by decide), root51Candidates13_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail14
    (if 3 ≤ root51Candidates13.length then search edge 3 [251, 2789] root51Candidates13 3040 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates14 : List Nat := [3803, 7919, 8387]
set_option maxRecDepth 8192 in
theorem root51Candidates14_checked : candidates edge 4 251 2801 root51Tail15 winner.bound = root51Candidates14 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root51Child14 : search edge 3 [251, 2801] root51Candidates14 3052 winner = winner := by decide
theorem root51Step14 : scan edge (search edge 3) 4 [251] 251 root51Tail14 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail15 winner := by
  rw [root51Tail14, scan, if_neg (by decide), root51Candidates14_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail15
    (if 3 ≤ root51Candidates14.length then search edge 3 [251, 2801] root51Candidates14 3052 winner else winner) = _
  rw [if_pos (by decide), root51Child14]
def root51Candidates15 : List Nat := [8951, 16607]
set_option maxRecDepth 8192 in
theorem root51Candidates15_checked : candidates edge 4 251 2999 root51Tail16 winner.bound = root51Candidates15 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root51Step15 : scan edge (search edge 3) 4 [251] 251 root51Tail15 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail16 winner := by
  rw [root51Tail15, scan, if_neg (by decide), root51Candidates15_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail16
    (if 3 ≤ root51Candidates15.length then search edge 3 [251, 2999] root51Candidates15 3250 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates16 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root51Candidates16_checked : candidates edge 4 251 3011 root51Tail17 winner.bound = root51Candidates16 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root51Step16 : scan edge (search edge 3) 4 [251] 251 root51Tail16 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail17 winner := by
  rw [root51Tail16, scan, if_neg (by decide), root51Candidates16_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail17
    (if 3 ≤ root51Candidates16.length then search edge 3 [251, 3011] root51Candidates16 3262 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates17 : List Nat := [10427, 21011]
set_option maxRecDepth 8192 in
theorem root51Candidates17_checked : candidates edge 4 251 3347 root51Tail18 winner.bound = root51Candidates17 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root51Step17 : scan edge (search edge 3) 4 [251] 251 root51Tail17 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail18 winner := by
  rw [root51Tail17, scan, if_neg (by decide), root51Candidates17_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail18
    (if 3 ≤ root51Candidates17.length then search edge 3 [251, 3347] root51Candidates17 3598 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates18 : List Nat := [3803, 21269]
set_option maxRecDepth 8192 in
theorem root51Candidates18_checked : candidates edge 4 251 3557 root51Tail19 winner.bound = root51Candidates18 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root51Step18 : scan edge (search edge 3) 4 [251] 251 root51Tail18 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail19 winner := by
  rw [root51Tail18, scan, if_neg (by decide), root51Candidates18_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail19
    (if 3 ≤ root51Candidates18.length then search edge 3 [251, 3557] root51Candidates18 3808 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root51Candidates19_checked : candidates edge 4 251 3803 root51Tail20 winner.bound = root51Candidates19 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root51Step19 : scan edge (search edge 3) 4 [251] 251 root51Tail19 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail20 winner := by
  rw [root51Tail19, scan, if_neg (by decide), root51Candidates19_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail20
    (if 3 ≤ root51Candidates19.length then search edge 3 [251, 3803] root51Candidates19 4054 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root51Candidates20_checked : candidates edge 4 251 4019 root51Tail21 winner.bound = root51Candidates20 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root51Step20 : scan edge (search edge 3) 4 [251] 251 root51Tail20 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail21 winner := by
  rw [root51Tail20, scan, if_neg (by decide), root51Candidates20_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail21
    (if 3 ≤ root51Candidates20.length then search edge 3 [251, 4019] root51Candidates20 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root51Candidates21_checked : candidates edge 4 251 4877 root51Tail22 winner.bound = root51Candidates21 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root51Step21 : scan edge (search edge 3) 4 [251] 251 root51Tail21 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail22 winner := by
  rw [root51Tail21, scan, if_neg (by decide), root51Candidates21_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail22
    (if 3 ≤ root51Candidates21.length then search edge 3 [251, 4877] root51Candidates21 5128 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates22 : List Nat := [5273]
set_option maxRecDepth 8192 in
theorem root51Candidates22_checked : candidates edge 4 251 5147 root51Tail23 winner.bound = root51Candidates22 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root51Step22 : scan edge (search edge 3) 4 [251] 251 root51Tail22 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail23 winner := by
  rw [root51Tail22, scan, if_neg (by decide), root51Candidates22_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail23
    (if 3 ≤ root51Candidates22.length then search edge 3 [251, 5147] root51Candidates22 5398 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root51Candidates23_checked : candidates edge 4 251 5189 root51Tail24 winner.bound = root51Candidates23 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root51Step23 : scan edge (search edge 3) 4 [251] 251 root51Tail23 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail24 winner := by
  rw [root51Tail23, scan, if_neg (by decide), root51Candidates23_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail24
    (if 3 ≤ root51Candidates23.length then search edge 3 [251, 5189] root51Candidates23 5440 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root51Candidates24_checked : candidates edge 4 251 5237 root51Tail25 winner.bound = root51Candidates24 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root51Step24 : scan edge (search edge 3) 4 [251] 251 root51Tail24 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail25 winner := by
  rw [root51Tail24, scan, if_neg (by decide), root51Candidates24_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail25
    (if 3 ≤ root51Candidates24.length then search edge 3 [251, 5237] root51Candidates24 5488 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates25 : List Nat := [5651, 7127, 8093]
set_option maxRecDepth 8192 in
theorem root51Candidates25_checked : candidates edge 4 251 5273 root51Tail26 winner.bound = root51Candidates25 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root51Child25 : search edge 3 [251, 5273] root51Candidates25 5524 winner = winner := by decide
theorem root51Step25 : scan edge (search edge 3) 4 [251] 251 root51Tail25 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail26 winner := by
  rw [root51Tail25, scan, if_neg (by decide), root51Candidates25_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail26
    (if 3 ≤ root51Candidates25.length then search edge 3 [251, 5273] root51Candidates25 5524 winner else winner) = _
  rw [if_pos (by decide), root51Child25]
def root51Candidates26 : List Nat := [7499, 8093]
set_option maxRecDepth 8192 in
theorem root51Candidates26_checked : candidates edge 4 251 5651 root51Tail27 winner.bound = root51Candidates26 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root51Step26 : scan edge (search edge 3) 4 [251] 251 root51Tail26 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail27 winner := by
  rw [root51Tail26, scan, if_neg (by decide), root51Candidates26_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail27
    (if 3 ≤ root51Candidates26.length then search edge 3 [251, 5651] root51Candidates26 5902 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates27 : List Nat := [9467]
set_option maxRecDepth 8192 in
theorem root51Candidates27_checked : candidates edge 4 251 5897 root51Tail28 winner.bound = root51Candidates27 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root51Step27 : scan edge (search edge 3) 4 [251] 251 root51Tail27 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail28 winner := by
  rw [root51Tail27, scan, if_neg (by decide), root51Candidates27_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail28
    (if 3 ≤ root51Candidates27.length then search edge 3 [251, 5897] root51Candidates27 6148 winner else winner) = _
  rw [if_neg (by decide)]
def root51Candidates28 : List Nat := []
set_option maxRecDepth 8192 in
theorem root51Candidates28_checked : candidates edge 4 251 6197 root51Tail29 winner.bound = root51Candidates28 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root51Step28 : scan edge (search edge 3) 4 [251] 251 root51Tail28 winner =
    scan edge (search edge 3) 4 [251] 251 root51Tail29 winner := by
  rw [root51Tail28, scan, if_neg (by decide), root51Candidates28_checked]
  change scan edge (search edge 3) 4 [251] 251 root51Tail29
    (if 3 ≤ root51Candidates28.length then search edge 3 [251, 6197] root51Candidates28 6448 winner else winner) = _
  rw [if_neg (by decide)]
theorem root51Scan29 : scan edge (search edge 3) 4 [251] 251 root51Tail29 winner = winner := by rw [root51Tail29, scan, if_pos (by decide)]
theorem root51Scan28 : scan edge (search edge 3) 4 [251] 251 root51Tail28 winner = winner :=
  (root51Step28).trans root51Scan29
theorem root51Scan27 : scan edge (search edge 3) 4 [251] 251 root51Tail27 winner = winner :=
  (root51Step27).trans root51Scan28
theorem root51Scan26 : scan edge (search edge 3) 4 [251] 251 root51Tail26 winner = winner :=
  (root51Step26).trans root51Scan27
theorem root51Scan25 : scan edge (search edge 3) 4 [251] 251 root51Tail25 winner = winner :=
  (root51Step25).trans root51Scan26
theorem root51Scan24 : scan edge (search edge 3) 4 [251] 251 root51Tail24 winner = winner :=
  (root51Step24).trans root51Scan25
theorem root51Scan23 : scan edge (search edge 3) 4 [251] 251 root51Tail23 winner = winner :=
  (root51Step23).trans root51Scan24
theorem root51Scan22 : scan edge (search edge 3) 4 [251] 251 root51Tail22 winner = winner :=
  (root51Step22).trans root51Scan23
theorem root51Scan21 : scan edge (search edge 3) 4 [251] 251 root51Tail21 winner = winner :=
  (root51Step21).trans root51Scan22
theorem root51Scan20 : scan edge (search edge 3) 4 [251] 251 root51Tail20 winner = winner :=
  (root51Step20).trans root51Scan21
theorem root51Scan19 : scan edge (search edge 3) 4 [251] 251 root51Tail19 winner = winner :=
  (root51Step19).trans root51Scan20
theorem root51Scan18 : scan edge (search edge 3) 4 [251] 251 root51Tail18 winner = winner :=
  (root51Step18).trans root51Scan19
theorem root51Scan17 : scan edge (search edge 3) 4 [251] 251 root51Tail17 winner = winner :=
  (root51Step17).trans root51Scan18
theorem root51Scan16 : scan edge (search edge 3) 4 [251] 251 root51Tail16 winner = winner :=
  (root51Step16).trans root51Scan17
theorem root51Scan15 : scan edge (search edge 3) 4 [251] 251 root51Tail15 winner = winner :=
  (root51Step15).trans root51Scan16
theorem root51Scan14 : scan edge (search edge 3) 4 [251] 251 root51Tail14 winner = winner :=
  (root51Step14).trans root51Scan15
theorem root51Scan13 : scan edge (search edge 3) 4 [251] 251 root51Tail13 winner = winner :=
  (root51Step13).trans root51Scan14
theorem root51Scan12 : scan edge (search edge 3) 4 [251] 251 root51Tail12 winner = winner :=
  (root51Step12).trans root51Scan13
theorem root51Scan11 : scan edge (search edge 3) 4 [251] 251 root51Tail11 winner = winner :=
  (root51Step11).trans root51Scan12
theorem root51Scan10 : scan edge (search edge 3) 4 [251] 251 root51Tail10 winner = winner :=
  (root51Step10).trans root51Scan11
theorem root51Scan9 : scan edge (search edge 3) 4 [251] 251 root51Tail9 winner = winner :=
  (root51Step9).trans root51Scan10
theorem root51Scan8 : scan edge (search edge 3) 4 [251] 251 root51Tail8 winner = winner :=
  (root51Step8).trans root51Scan9
theorem root51Scan7 : scan edge (search edge 3) 4 [251] 251 root51Tail7 winner = winner :=
  (root51Step7).trans root51Scan8
theorem root51Scan6 : scan edge (search edge 3) 4 [251] 251 root51Tail6 winner = winner :=
  (root51Step6).trans root51Scan7
theorem root51Scan5 : scan edge (search edge 3) 4 [251] 251 root51Tail5 winner = winner :=
  (root51Step5).trans root51Scan6
theorem root51Scan4 : scan edge (search edge 3) 4 [251] 251 root51Tail4 winner = winner :=
  (root51Step4).trans root51Scan5
theorem root51Scan3 : scan edge (search edge 3) 4 [251] 251 root51Tail3 winner = winner :=
  (root51Step3).trans root51Scan4
theorem root51Scan2 : scan edge (search edge 3) 4 [251] 251 root51Tail2 winner = winner :=
  (root51Step2).trans root51Scan3
theorem root51Scan1 : scan edge (search edge 3) 4 [251] 251 root51Tail1 winner = winner :=
  (root51Step1).trans root51Scan2
theorem root51Scan0 : scan edge (search edge 3) 4 [251] 251 root51Tail0 winner = winner :=
  (root51Step0).trans root51Scan1
set_option maxRecDepth 4096 in
theorem root51Branch : search edge 4 [251] root51Candidates 251 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [251] 251 root51Tail0 winner = winner
  exact root51Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
