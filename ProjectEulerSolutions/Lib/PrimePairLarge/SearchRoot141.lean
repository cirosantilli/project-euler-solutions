import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root141Candidates : List Nat := Data.row827
set_option maxRecDepth 8192 in
theorem root141Candidates_checked : candidates edge 5 0 827 tail142 winner.bound = root141Candidates := by
  apply candidates_eq_row edge 5 0 827 tail142 root141Candidates winner.bound tail142_ordered
  · decide
  · decide
  · funext q
    rfl
def root141Tail71 : List Nat := []
def root141Tail70 : List Nat := 25169 :: root141Tail71
def root141Tail69 : List Nat := 25037 :: root141Tail70
def root141Tail68 : List Nat := 24821 :: root141Tail69
def root141Tail67 : List Nat := 24659 :: root141Tail68
def root141Tail66 : List Nat := 24611 :: root141Tail67
def root141Tail65 : List Nat := 24197 :: root141Tail66
def root141Tail64 : List Nat := 23063 :: root141Tail65
def root141Tail63 : List Nat := 22859 :: root141Tail64
def root141Tail62 : List Nat := 22721 :: root141Tail63
def root141Tail61 : List Nat := 22643 :: root141Tail62
def root141Tail60 : List Nat := 22571 :: root141Tail61
def root141Tail59 : List Nat := 22349 :: root141Tail60
def root141Tail58 : List Nat := 21893 :: root141Tail59
def root141Tail57 : List Nat := 21767 :: root141Tail58
def root141Tail56 : List Nat := 21347 :: root141Tail57
def root141Tail55 : List Nat := 20747 :: root141Tail56
def root141Tail54 : List Nat := 20549 :: root141Tail55
def root141Tail53 : List Nat := 20543 :: root141Tail54
def root141Tail52 : List Nat := 20201 :: root141Tail53
def root141Tail51 : List Nat := 19961 :: root141Tail52
def root141Tail50 : List Nat := 18947 :: root141Tail51
def root141Tail49 : List Nat := 18773 :: root141Tail50
def root141Tail48 : List Nat := 17489 :: root141Tail49
def root141Tail47 : List Nat := 17477 :: root141Tail48
def root141Tail46 : List Nat := 16553 :: root141Tail47
def root141Tail45 : List Nat := 16421 :: root141Tail46
def root141Tail44 : List Nat := 15551 :: root141Tail45
def root141Tail43 : List Nat := 15467 :: root141Tail44
def root141Tail42 : List Nat := 14747 :: root141Tail43
def root141Tail41 : List Nat := 14717 :: root141Tail42
def root141Tail40 : List Nat := 14627 :: root141Tail41
def root141Tail39 : List Nat := 14081 :: root141Tail40
def root141Tail38 : List Nat := 13997 :: root141Tail39
def root141Tail37 : List Nat := 13877 :: root141Tail38
def root141Tail36 : List Nat := 13121 :: root141Tail37
def root141Tail35 : List Nat := 12653 :: root141Tail36
def root141Tail34 : List Nat := 12647 :: root141Tail35
def root141Tail33 : List Nat := 11897 :: root141Tail34
def root141Tail32 : List Nat := 11483 :: root141Tail33
def root141Tail31 : List Nat := 11093 :: root141Tail32
def root141Tail30 : List Nat := 10631 :: root141Tail31
def root141Tail29 : List Nat := 10391 :: root141Tail30
def root141Tail28 : List Nat := 10337 :: root141Tail29
def root141Tail27 : List Nat := 10163 :: root141Tail28
def root141Tail26 : List Nat := 9461 :: root141Tail27
def root141Tail25 : List Nat := 8951 :: root141Tail26
def root141Tail24 : List Nat := 8663 :: root141Tail25
def root141Tail23 : List Nat := 6143 :: root141Tail24
def root141Tail22 : List Nat := 5717 :: root141Tail23
def root141Tail21 : List Nat := 5471 :: root141Tail22
def root141Tail20 : List Nat := 5417 :: root141Tail21
def root141Tail19 : List Nat := 5303 :: root141Tail20
def root141Tail18 : List Nat := 4463 :: root141Tail19
def root141Tail17 : List Nat := 3719 :: root141Tail18
def root141Tail16 : List Nat := 3593 :: root141Tail17
def root141Tail15 : List Nat := 3527 :: root141Tail16
def root141Tail14 : List Nat := 3299 :: root141Tail15
def root141Tail13 : List Nat := 2909 :: root141Tail14
def root141Tail12 : List Nat := 2819 :: root141Tail13
def root141Tail11 : List Nat := 2711 :: root141Tail12
def root141Tail10 : List Nat := 2351 :: root141Tail11
def root141Tail9 : List Nat := 2273 :: root141Tail10
def root141Tail8 : List Nat := 2267 :: root141Tail9
def root141Tail7 : List Nat := 1871 :: root141Tail8
def root141Tail6 : List Nat := 1847 :: root141Tail7
def root141Tail5 : List Nat := 1811 :: root141Tail6
def root141Tail4 : List Nat := 1721 :: root141Tail5
def root141Tail3 : List Nat := 1301 :: root141Tail4
def root141Tail2 : List Nat := 1049 :: root141Tail3
def root141Tail1 : List Nat := 1019 :: root141Tail2
def root141Tail0 : List Nat := 857 :: root141Tail1
def root141Candidates0 : List Nat := [1301, 4463, 10337, 15551, 19961]
set_option maxRecDepth 8192 in
theorem root141Candidates0_checked : candidates edge 4 827 857 root141Tail1 winner.bound = root141Candidates0 := by
  unfold candidates
  have row : edge 857 = (SortedNatTree.contains FastRows.row857) := by
    change Data.row857.contains = (SortedNatTree.contains FastRows.row857)
    exact FastRows.row857_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root141Child0 : search edge 3 [827, 857] root141Candidates0 1684 winner = winner := by decide
theorem root141Step0 : scan edge (search edge 3) 4 [827] 827 root141Tail0 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail1 winner := by
  rw [root141Tail0, scan, if_neg (by decide), root141Candidates0_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail1
    (if 3 ≤ root141Candidates0.length then search edge 3 [827, 857] root141Candidates0 1684 winner else winner) = _
  rw [if_pos (by decide), root141Child0]
def root141Candidates1 : List Nat := [22349]
set_option maxRecDepth 8192 in
theorem root141Candidates1_checked : candidates edge 4 827 1019 root141Tail2 winner.bound = root141Candidates1 := by
  unfold candidates
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row]
  decide
theorem root141Step1 : scan edge (search edge 3) 4 [827] 827 root141Tail1 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail2 winner := by
  rw [root141Tail1, scan, if_neg (by decide), root141Candidates1_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail2
    (if 3 ≤ root141Candidates1.length then search edge 3 [827, 1019] root141Candidates1 1846 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates2 : List Nat := [1847, 13997, 14081, 16421, 22859]
set_option maxRecDepth 8192 in
theorem root141Candidates2_checked : candidates edge 4 827 1049 root141Tail3 winner.bound = root141Candidates2 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root141Child2 : search edge 3 [827, 1049] root141Candidates2 1876 winner = winner := by decide
theorem root141Step2 : scan edge (search edge 3) 4 [827] 827 root141Tail2 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail3 winner := by
  rw [root141Tail2, scan, if_neg (by decide), root141Candidates2_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail3
    (if 3 ≤ root141Candidates2.length then search edge 3 [827, 1049] root141Candidates2 1876 winner else winner) = _
  rw [if_pos (by decide), root141Child2]
def root141Candidates3 : List Nat := [16421]
set_option maxRecDepth 8192 in
theorem root141Candidates3_checked : candidates edge 4 827 1301 root141Tail4 winner.bound = root141Candidates3 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
theorem root141Step3 : scan edge (search edge 3) 4 [827] 827 root141Tail3 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail4 winner := by
  rw [root141Tail3, scan, if_neg (by decide), root141Candidates3_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail4
    (if 3 ≤ root141Candidates3.length then search edge 3 [827, 1301] root141Candidates3 2128 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates4 : List Nat := [14717, 20549, 21767]
set_option maxRecDepth 8192 in
theorem root141Candidates4_checked : candidates edge 4 827 1721 root141Tail5 winner.bound = root141Candidates4 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root141Child4 : search edge 3 [827, 1721] root141Candidates4 2548 winner = winner := by decide
theorem root141Step4 : scan edge (search edge 3) 4 [827] 827 root141Tail4 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail5 winner := by
  rw [root141Tail4, scan, if_neg (by decide), root141Candidates4_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail5
    (if 3 ≤ root141Candidates4.length then search edge 3 [827, 1721] root141Candidates4 2548 winner else winner) = _
  rw [if_pos (by decide), root141Child4]
def root141Candidates5 : List Nat := [17477]
set_option maxRecDepth 8192 in
theorem root141Candidates5_checked : candidates edge 4 827 1811 root141Tail6 winner.bound = root141Candidates5 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
theorem root141Step5 : scan edge (search edge 3) 4 [827] 827 root141Tail5 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail6 winner := by
  rw [root141Tail5, scan, if_neg (by decide), root141Candidates5_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail6
    (if 3 ≤ root141Candidates5.length then search edge 3 [827, 1811] root141Candidates5 2638 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates6 : List Nat := [8951]
set_option maxRecDepth 8192 in
theorem root141Candidates6_checked : candidates edge 4 827 1847 root141Tail7 winner.bound = root141Candidates6 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
theorem root141Step6 : scan edge (search edge 3) 4 [827] 827 root141Tail6 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail7 winner := by
  rw [root141Tail6, scan, if_neg (by decide), root141Candidates6_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail7
    (if 3 ≤ root141Candidates6.length then search edge 3 [827, 1847] root141Candidates6 2674 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates7 : List Nat := [14747]
set_option maxRecDepth 8192 in
theorem root141Candidates7_checked : candidates edge 4 827 1871 root141Tail8 winner.bound = root141Candidates7 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
theorem root141Step7 : scan edge (search edge 3) 4 [827] 827 root141Tail7 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail8 winner := by
  rw [root141Tail7, scan, if_neg (by decide), root141Candidates7_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail8
    (if 3 ≤ root141Candidates7.length then search edge 3 [827, 1871] root141Candidates7 2698 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates8 : List Nat := [3527, 15551]
set_option maxRecDepth 8192 in
theorem root141Candidates8_checked : candidates edge 4 827 2267 root141Tail9 winner.bound = root141Candidates8 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root141Step8 : scan edge (search edge 3) 4 [827] 827 root141Tail8 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail9 winner := by
  rw [root141Tail8, scan, if_neg (by decide), root141Candidates8_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail9
    (if 3 ≤ root141Candidates8.length then search edge 3 [827, 2267] root141Candidates8 3094 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates9 : List Nat := [16553, 20543]
set_option maxRecDepth 8192 in
theorem root141Candidates9_checked : candidates edge 4 827 2273 root141Tail10 winner.bound = root141Candidates9 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
theorem root141Step9 : scan edge (search edge 3) 4 [827] 827 root141Tail9 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail10 winner := by
  rw [root141Tail9, scan, if_neg (by decide), root141Candidates9_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail10
    (if 3 ≤ root141Candidates9.length then search edge 3 [827, 2273] root141Candidates9 3100 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates10 : List Nat := [16421, 22571]
set_option maxRecDepth 8192 in
theorem root141Candidates10_checked : candidates edge 4 827 2351 root141Tail11 winner.bound = root141Candidates10 := by
  unfold candidates
  have row : edge 2351 = (SortedNatTree.contains FastRows.row2351) := by
    change Data.row2351.contains = (SortedNatTree.contains FastRows.row2351)
    exact FastRows.row2351_checked.symm
  rw [row]
  decide
theorem root141Step10 : scan edge (search edge 3) 4 [827] 827 root141Tail10 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail11 winner := by
  rw [root141Tail10, scan, if_neg (by decide), root141Candidates10_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail11
    (if 3 ≤ root141Candidates10.length then search edge 3 [827, 2351] root141Candidates10 3178 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root141Candidates11_checked : candidates edge 4 827 2711 root141Tail12 winner.bound = root141Candidates11 := by
  unfold candidates
  have row : edge 2711 = (SortedNatTree.contains FastRows.row2711) := by
    change Data.row2711.contains = (SortedNatTree.contains FastRows.row2711)
    exact FastRows.row2711_checked.symm
  rw [row]
  decide
theorem root141Step11 : scan edge (search edge 3) 4 [827] 827 root141Tail11 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail12 winner := by
  rw [root141Tail11, scan, if_neg (by decide), root141Candidates11_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail12
    (if 3 ≤ root141Candidates11.length then search edge 3 [827, 2711] root141Candidates11 3538 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root141Candidates12_checked : candidates edge 4 827 2819 root141Tail13 winner.bound = root141Candidates12 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root141Step12 : scan edge (search edge 3) 4 [827] 827 root141Tail12 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail13 winner := by
  rw [root141Tail12, scan, if_neg (by decide), root141Candidates12_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail13
    (if 3 ≤ root141Candidates12.length then search edge 3 [827, 2819] root141Candidates12 3646 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates13 : List Nat := [21767]
set_option maxRecDepth 8192 in
theorem root141Candidates13_checked : candidates edge 4 827 2909 root141Tail14 winner.bound = root141Candidates13 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
theorem root141Step13 : scan edge (search edge 3) 4 [827] 827 root141Tail13 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail14 winner := by
  rw [root141Tail13, scan, if_neg (by decide), root141Candidates13_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail14
    (if 3 ≤ root141Candidates13.length then search edge 3 [827, 2909] root141Candidates13 3736 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root141Candidates14_checked : candidates edge 4 827 3299 root141Tail15 winner.bound = root141Candidates14 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root141Step14 : scan edge (search edge 3) 4 [827] 827 root141Tail14 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail15 winner := by
  rw [root141Tail14, scan, if_neg (by decide), root141Candidates14_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail15
    (if 3 ≤ root141Candidates14.length then search edge 3 [827, 3299] root141Candidates14 4126 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates15 : List Nat := [16421, 20747]
set_option maxRecDepth 8192 in
theorem root141Candidates15_checked : candidates edge 4 827 3527 root141Tail16 winner.bound = root141Candidates15 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root141Step15 : scan edge (search edge 3) 4 [827] 827 root141Tail15 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail16 winner := by
  rw [root141Tail15, scan, if_neg (by decide), root141Candidates15_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail16
    (if 3 ≤ root141Candidates15.length then search edge 3 [827, 3527] root141Candidates15 4354 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates16 : List Nat := [3719, 16553]
set_option maxRecDepth 8192 in
theorem root141Candidates16_checked : candidates edge 4 827 3593 root141Tail17 winner.bound = root141Candidates16 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root141Step16 : scan edge (search edge 3) 4 [827] 827 root141Tail16 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail17 winner := by
  rw [root141Tail16, scan, if_neg (by decide), root141Candidates16_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail17
    (if 3 ≤ root141Candidates16.length then search edge 3 [827, 3593] root141Candidates16 4420 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates17 : List Nat := [8663]
set_option maxRecDepth 8192 in
theorem root141Candidates17_checked : candidates edge 4 827 3719 root141Tail18 winner.bound = root141Candidates17 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root141Step17 : scan edge (search edge 3) 4 [827] 827 root141Tail17 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail18 winner := by
  rw [root141Tail17, scan, if_neg (by decide), root141Candidates17_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail18
    (if 3 ≤ root141Candidates17.length then search edge 3 [827, 3719] root141Candidates17 4546 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates18 : List Nat := [18947]
set_option maxRecDepth 8192 in
theorem root141Candidates18_checked : candidates edge 4 827 4463 root141Tail19 winner.bound = root141Candidates18 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root141Step18 : scan edge (search edge 3) 4 [827] 827 root141Tail18 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail19 winner := by
  rw [root141Tail18, scan, if_neg (by decide), root141Candidates18_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail19
    (if 3 ≤ root141Candidates18.length then search edge 3 [827, 4463] root141Candidates18 5290 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates19 : List Nat := [13877]
set_option maxRecDepth 8192 in
theorem root141Candidates19_checked : candidates edge 4 827 5303 root141Tail20 winner.bound = root141Candidates19 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root141Step19 : scan edge (search edge 3) 4 [827] 827 root141Tail19 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail20 winner := by
  rw [root141Tail19, scan, if_neg (by decide), root141Candidates19_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail20
    (if 3 ≤ root141Candidates19.length then search edge 3 [827, 5303] root141Candidates19 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates20 : List Nat := [12653, 14081, 14627]
set_option maxRecDepth 8192 in
theorem root141Candidates20_checked : candidates edge 4 827 5417 root141Tail21 winner.bound = root141Candidates20 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root141Child20 : search edge 3 [827, 5417] root141Candidates20 6244 winner = winner := by decide
theorem root141Step20 : scan edge (search edge 3) 4 [827] 827 root141Tail20 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail21 winner := by
  rw [root141Tail20, scan, if_neg (by decide), root141Candidates20_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail21
    (if 3 ≤ root141Candidates20.length then search edge 3 [827, 5417] root141Candidates20 6244 winner else winner) = _
  rw [if_pos (by decide), root141Child20]
def root141Candidates21 : List Nat := [12647]
set_option maxRecDepth 8192 in
theorem root141Candidates21_checked : candidates edge 4 827 5471 root141Tail22 winner.bound = root141Candidates21 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root141Step21 : scan edge (search edge 3) 4 [827] 827 root141Tail21 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail22 winner := by
  rw [root141Tail21, scan, if_neg (by decide), root141Candidates21_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail22
    (if 3 ≤ root141Candidates21.length then search edge 3 [827, 5471] root141Candidates21 6298 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates22 : List Nat := [10631]
set_option maxRecDepth 8192 in
theorem root141Candidates22_checked : candidates edge 4 827 5717 root141Tail23 winner.bound = root141Candidates22 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root141Step22 : scan edge (search edge 3) 4 [827] 827 root141Tail22 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail23 winner := by
  rw [root141Tail22, scan, if_neg (by decide), root141Candidates22_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail23
    (if 3 ≤ root141Candidates22.length then search edge 3 [827, 5717] root141Candidates22 6544 winner else winner) = _
  rw [if_neg (by decide)]
def root141Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root141Candidates23_checked : candidates edge 4 827 6143 root141Tail24 winner.bound = root141Candidates23 := by
  unfold candidates
  have row : edge 6143 = (SortedNatTree.contains FastRows.row6143) := by
    change Data.row6143.contains = (SortedNatTree.contains FastRows.row6143)
    exact FastRows.row6143_checked.symm
  rw [row]
  decide
theorem root141Step23 : scan edge (search edge 3) 4 [827] 827 root141Tail23 winner =
    scan edge (search edge 3) 4 [827] 827 root141Tail24 winner := by
  rw [root141Tail23, scan, if_neg (by decide), root141Candidates23_checked]
  change scan edge (search edge 3) 4 [827] 827 root141Tail24
    (if 3 ≤ root141Candidates23.length then search edge 3 [827, 6143] root141Candidates23 6970 winner else winner) = _
  rw [if_neg (by decide)]
theorem root141Scan24 : scan edge (search edge 3) 4 [827] 827 root141Tail24 winner = winner := by rw [root141Tail24, scan, if_pos (by decide)]
theorem root141Scan23 : scan edge (search edge 3) 4 [827] 827 root141Tail23 winner = winner :=
  (root141Step23).trans root141Scan24
theorem root141Scan22 : scan edge (search edge 3) 4 [827] 827 root141Tail22 winner = winner :=
  (root141Step22).trans root141Scan23
theorem root141Scan21 : scan edge (search edge 3) 4 [827] 827 root141Tail21 winner = winner :=
  (root141Step21).trans root141Scan22
theorem root141Scan20 : scan edge (search edge 3) 4 [827] 827 root141Tail20 winner = winner :=
  (root141Step20).trans root141Scan21
theorem root141Scan19 : scan edge (search edge 3) 4 [827] 827 root141Tail19 winner = winner :=
  (root141Step19).trans root141Scan20
theorem root141Scan18 : scan edge (search edge 3) 4 [827] 827 root141Tail18 winner = winner :=
  (root141Step18).trans root141Scan19
theorem root141Scan17 : scan edge (search edge 3) 4 [827] 827 root141Tail17 winner = winner :=
  (root141Step17).trans root141Scan18
theorem root141Scan16 : scan edge (search edge 3) 4 [827] 827 root141Tail16 winner = winner :=
  (root141Step16).trans root141Scan17
theorem root141Scan15 : scan edge (search edge 3) 4 [827] 827 root141Tail15 winner = winner :=
  (root141Step15).trans root141Scan16
theorem root141Scan14 : scan edge (search edge 3) 4 [827] 827 root141Tail14 winner = winner :=
  (root141Step14).trans root141Scan15
theorem root141Scan13 : scan edge (search edge 3) 4 [827] 827 root141Tail13 winner = winner :=
  (root141Step13).trans root141Scan14
theorem root141Scan12 : scan edge (search edge 3) 4 [827] 827 root141Tail12 winner = winner :=
  (root141Step12).trans root141Scan13
theorem root141Scan11 : scan edge (search edge 3) 4 [827] 827 root141Tail11 winner = winner :=
  (root141Step11).trans root141Scan12
theorem root141Scan10 : scan edge (search edge 3) 4 [827] 827 root141Tail10 winner = winner :=
  (root141Step10).trans root141Scan11
theorem root141Scan9 : scan edge (search edge 3) 4 [827] 827 root141Tail9 winner = winner :=
  (root141Step9).trans root141Scan10
theorem root141Scan8 : scan edge (search edge 3) 4 [827] 827 root141Tail8 winner = winner :=
  (root141Step8).trans root141Scan9
theorem root141Scan7 : scan edge (search edge 3) 4 [827] 827 root141Tail7 winner = winner :=
  (root141Step7).trans root141Scan8
theorem root141Scan6 : scan edge (search edge 3) 4 [827] 827 root141Tail6 winner = winner :=
  (root141Step6).trans root141Scan7
theorem root141Scan5 : scan edge (search edge 3) 4 [827] 827 root141Tail5 winner = winner :=
  (root141Step5).trans root141Scan6
theorem root141Scan4 : scan edge (search edge 3) 4 [827] 827 root141Tail4 winner = winner :=
  (root141Step4).trans root141Scan5
theorem root141Scan3 : scan edge (search edge 3) 4 [827] 827 root141Tail3 winner = winner :=
  (root141Step3).trans root141Scan4
theorem root141Scan2 : scan edge (search edge 3) 4 [827] 827 root141Tail2 winner = winner :=
  (root141Step2).trans root141Scan3
theorem root141Scan1 : scan edge (search edge 3) 4 [827] 827 root141Tail1 winner = winner :=
  (root141Step1).trans root141Scan2
theorem root141Scan0 : scan edge (search edge 3) 4 [827] 827 root141Tail0 winner = winner :=
  (root141Step0).trans root141Scan1
set_option maxRecDepth 4096 in
theorem root141Branch : search edge 4 [827] root141Candidates 827 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [827] 827 root141Tail0 winner = winner
  exact root141Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
