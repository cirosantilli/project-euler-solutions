import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root151Candidates : List Nat := Data.row887
set_option maxRecDepth 8192 in
theorem root151Candidates_checked : candidates edge 5 0 887 tail152 winner.bound = root151Candidates := by
  apply candidates_eq_row edge 5 0 887 tail152 root151Candidates winner.bound tail152_ordered
  · decide
  · decide
  · funext q
    rfl
def root151Tail65 : List Nat := []
def root151Tail64 : List Nat := 25097 :: root151Tail65
def root151Tail63 : List Nat := 24917 :: root151Tail64
def root151Tail62 : List Nat := 24527 :: root151Tail63
def root151Tail61 : List Nat := 23993 :: root151Tail62
def root151Tail60 : List Nat := 23633 :: root151Tail61
def root151Tail59 : List Nat := 22871 :: root151Tail60
def root151Tail58 : List Nat := 22691 :: root151Tail59
def root151Tail57 : List Nat := 21911 :: root151Tail58
def root151Tail56 : List Nat := 21851 :: root151Tail57
def root151Tail55 : List Nat := 20981 :: root151Tail56
def root151Tail54 : List Nat := 20903 :: root151Tail55
def root151Tail53 : List Nat := 20771 :: root151Tail54
def root151Tail52 : List Nat := 19937 :: root151Tail53
def root151Tail51 : List Nat := 18719 :: root151Tail52
def root151Tail50 : List Nat := 17903 :: root151Tail51
def root151Tail49 : List Nat := 17597 :: root151Tail50
def root151Tail48 : List Nat := 17159 :: root151Tail49
def root151Tail47 : List Nat := 16493 :: root151Tail48
def root151Tail46 : List Nat := 16139 :: root151Tail47
def root151Tail45 : List Nat := 15971 :: root151Tail46
def root151Tail44 : List Nat := 15581 :: root151Tail45
def root151Tail43 : List Nat := 13883 :: root151Tail44
def root151Tail42 : List Nat := 13799 :: root151Tail43
def root151Tail41 : List Nat := 13421 :: root151Tail42
def root151Tail40 : List Nat := 13127 :: root151Tail41
def root151Tail39 : List Nat := 12959 :: root151Tail40
def root151Tail38 : List Nat := 12671 :: root151Tail39
def root151Tail37 : List Nat := 12413 :: root151Tail38
def root151Tail36 : List Nat := 12401 :: root151Tail37
def root151Tail35 : List Nat := 12161 :: root151Tail36
def root151Tail34 : List Nat := 11813 :: root151Tail35
def root151Tail33 : List Nat := 11789 :: root151Tail34
def root151Tail32 : List Nat := 11369 :: root151Tail33
def root151Tail31 : List Nat := 11351 :: root151Tail32
def root151Tail30 : List Nat := 10331 :: root151Tail31
def root151Tail29 : List Nat := 9551 :: root151Tail30
def root151Tail28 : List Nat := 9293 :: root151Tail29
def root151Tail27 : List Nat := 8861 :: root151Tail28
def root151Tail26 : List Nat := 8423 :: root151Tail27
def root151Tail25 : List Nat := 8087 :: root151Tail26
def root151Tail24 : List Nat := 7727 :: root151Tail25
def root151Tail23 : List Nat := 7499 :: root151Tail24
def root151Tail22 : List Nat := 7109 :: root151Tail23
def root151Tail21 : List Nat := 7013 :: root151Tail22
def root151Tail20 : List Nat := 6911 :: root151Tail21
def root151Tail19 : List Nat := 6863 :: root151Tail20
def root151Tail18 : List Nat := 6653 :: root151Tail19
def root151Tail17 : List Nat := 6449 :: root151Tail18
def root151Tail16 : List Nat := 5651 :: root151Tail17
def root151Tail15 : List Nat := 5417 :: root151Tail16
def root151Tail14 : List Nat := 5303 :: root151Tail15
def root151Tail13 : List Nat := 4157 :: root151Tail14
def root151Tail12 : List Nat := 4049 :: root151Tail13
def root151Tail11 : List Nat := 3803 :: root151Tail12
def root151Tail10 : List Nat := 3251 :: root151Tail11
def root151Tail9 : List Nat := 3089 :: root151Tail10
def root151Tail8 : List Nat := 2741 :: root151Tail9
def root151Tail7 : List Nat := 2153 :: root151Tail8
def root151Tail6 : List Nat := 2099 :: root151Tail7
def root151Tail5 : List Nat := 1931 :: root151Tail6
def root151Tail4 : List Nat := 1847 :: root151Tail5
def root151Tail3 : List Nat := 1697 :: root151Tail4
def root151Tail2 : List Nat := 1553 :: root151Tail3
def root151Tail1 : List Nat := 1427 :: root151Tail2
def root151Tail0 : List Nat := 1091 :: root151Tail1
def root151Candidates0 : List Nat := [1697, 1847, 15581, 20903]
set_option maxRecDepth 8192 in
theorem root151Candidates0_checked : candidates edge 4 887 1091 root151Tail1 winner.bound = root151Candidates0 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root151Child0 : search edge 3 [887, 1091] root151Candidates0 1978 winner = winner := by decide
theorem root151Step0 : scan edge (search edge 3) 4 [887] 887 root151Tail0 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail1 winner := by
  rw [root151Tail0, scan, if_neg (by decide), root151Candidates0_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail1
    (if 3 ≤ root151Candidates0.length then search edge 3 [887, 1091] root151Candidates0 1978 winner else winner) = _
  rw [if_pos (by decide), root151Child0]
def root151Candidates1 : List Nat := [1931, 4157]
set_option maxRecDepth 8192 in
theorem root151Candidates1_checked : candidates edge 4 887 1427 root151Tail2 winner.bound = root151Candidates1 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
theorem root151Step1 : scan edge (search edge 3) 4 [887] 887 root151Tail1 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail2 winner := by
  rw [root151Tail1, scan, if_neg (by decide), root151Candidates1_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail2
    (if 3 ≤ root151Candidates1.length then search edge 3 [887, 1427] root151Candidates1 2314 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates2 : List Nat := [1697, 4049, 7013, 12401]
set_option maxRecDepth 8192 in
theorem root151Candidates2_checked : candidates edge 4 887 1553 root151Tail3 winner.bound = root151Candidates2 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root151Child2 : search edge 3 [887, 1553] root151Candidates2 2440 winner = winner := by decide
theorem root151Step2 : scan edge (search edge 3) 4 [887] 887 root151Tail2 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail3 winner := by
  rw [root151Tail2, scan, if_neg (by decide), root151Candidates2_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail3
    (if 3 ≤ root151Candidates2.length then search edge 3 [887, 1553] root151Candidates2 2440 winner else winner) = _
  rw [if_pos (by decide), root151Child2]
def root151Candidates3 : List Nat := [2741, 11813, 13421]
set_option maxRecDepth 8192 in
theorem root151Candidates3_checked : candidates edge 4 887 1697 root151Tail4 winner.bound = root151Candidates3 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root151Child3 : search edge 3 [887, 1697] root151Candidates3 2584 winner = winner := by decide
theorem root151Step3 : scan edge (search edge 3) 4 [887] 887 root151Tail3 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail4 winner := by
  rw [root151Tail3, scan, if_neg (by decide), root151Candidates3_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail4
    (if 3 ≤ root151Candidates3.length then search edge 3 [887, 1697] root151Candidates3 2584 winner else winner) = _
  rw [if_pos (by decide), root151Child3]
def root151Candidates4 : List Nat := [16139]
set_option maxRecDepth 8192 in
theorem root151Candidates4_checked : candidates edge 4 887 1847 root151Tail5 winner.bound = root151Candidates4 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
theorem root151Step4 : scan edge (search edge 3) 4 [887] 887 root151Tail4 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail5 winner := by
  rw [root151Tail4, scan, if_neg (by decide), root151Candidates4_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail5
    (if 3 ≤ root151Candidates4.length then search edge 3 [887, 1847] root151Candidates4 2734 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates5 : List Nat := [11813]
set_option maxRecDepth 8192 in
theorem root151Candidates5_checked : candidates edge 4 887 1931 root151Tail6 winner.bound = root151Candidates5 := by
  unfold candidates
  have row : edge 1931 = (SortedNatTree.contains FastRows.row1931) := by
    change Data.row1931.contains = (SortedNatTree.contains FastRows.row1931)
    exact FastRows.row1931_checked.symm
  rw [row]
  decide
theorem root151Step5 : scan edge (search edge 3) 4 [887] 887 root151Tail5 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail6 winner := by
  rw [root151Tail5, scan, if_neg (by decide), root151Candidates5_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail6
    (if 3 ≤ root151Candidates5.length then search edge 3 [887, 1931] root151Candidates5 2818 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates6 : List Nat := [6863, 17597]
set_option maxRecDepth 8192 in
theorem root151Candidates6_checked : candidates edge 4 887 2099 root151Tail7 winner.bound = root151Candidates6 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
theorem root151Step6 : scan edge (search edge 3) 4 [887] 887 root151Tail6 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail7 winner := by
  rw [root151Tail6, scan, if_neg (by decide), root151Candidates6_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail7
    (if 3 ≤ root151Candidates6.length then search edge 3 [887, 2099] root151Candidates6 2986 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates7 : List Nat := [6863, 20981, 22871]
set_option maxRecDepth 8192 in
theorem root151Candidates7_checked : candidates edge 4 887 2153 root151Tail8 winner.bound = root151Candidates7 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root151Child7 : search edge 3 [887, 2153] root151Candidates7 3040 winner = winner := by decide
theorem root151Step7 : scan edge (search edge 3) 4 [887] 887 root151Tail7 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail8 winner := by
  rw [root151Tail7, scan, if_neg (by decide), root151Candidates7_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail8
    (if 3 ≤ root151Candidates7.length then search edge 3 [887, 2153] root151Candidates7 3040 winner else winner) = _
  rw [if_pos (by decide), root151Child7]
def root151Candidates8 : List Nat := [6911, 13799]
set_option maxRecDepth 8192 in
theorem root151Candidates8_checked : candidates edge 4 887 2741 root151Tail9 winner.bound = root151Candidates8 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root151Step8 : scan edge (search edge 3) 4 [887] 887 root151Tail8 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail9 winner := by
  rw [root151Tail8, scan, if_neg (by decide), root151Candidates8_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail9
    (if 3 ≤ root151Candidates8.length then search edge 3 [887, 2741] root151Candidates8 3628 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates9 : List Nat := [5303, 11789]
set_option maxRecDepth 8192 in
theorem root151Candidates9_checked : candidates edge 4 887 3089 root151Tail10 winner.bound = root151Candidates9 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root151Step9 : scan edge (search edge 3) 4 [887] 887 root151Tail9 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail10 winner := by
  rw [root151Tail9, scan, if_neg (by decide), root151Candidates9_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail10
    (if 3 ≤ root151Candidates9.length then search edge 3 [887, 3089] root151Candidates9 3976 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root151Candidates10_checked : candidates edge 4 887 3251 root151Tail11 winner.bound = root151Candidates10 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root151Step10 : scan edge (search edge 3) 4 [887] 887 root151Tail10 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail11 winner := by
  rw [root151Tail10, scan, if_neg (by decide), root151Candidates10_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail11
    (if 3 ≤ root151Candidates10.length then search edge 3 [887, 3251] root151Candidates10 4138 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates11 : List Nat := [12671, 13127, 13421, 17159]
set_option maxRecDepth 8192 in
theorem root151Candidates11_checked : candidates edge 4 887 3803 root151Tail12 winner.bound = root151Candidates11 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root151Child11 : search edge 3 [887, 3803] root151Candidates11 4690 winner = winner := by decide
theorem root151Step11 : scan edge (search edge 3) 4 [887] 887 root151Tail11 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail12 winner := by
  rw [root151Tail11, scan, if_neg (by decide), root151Candidates11_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail12
    (if 3 ≤ root151Candidates11.length then search edge 3 [887, 3803] root151Candidates11 4690 winner else winner) = _
  rw [if_pos (by decide), root151Child11]
def root151Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root151Candidates12_checked : candidates edge 4 887 4049 root151Tail13 winner.bound = root151Candidates12 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root151Step12 : scan edge (search edge 3) 4 [887] 887 root151Tail12 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail13 winner := by
  rw [root151Tail12, scan, if_neg (by decide), root151Candidates12_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail13
    (if 3 ≤ root151Candidates12.length then search edge 3 [887, 4049] root151Candidates12 4936 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root151Candidates13_checked : candidates edge 4 887 4157 root151Tail14 winner.bound = root151Candidates13 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root151Step13 : scan edge (search edge 3) 4 [887] 887 root151Tail13 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail14 winner := by
  rw [root151Tail13, scan, if_neg (by decide), root151Candidates13_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail14
    (if 3 ≤ root151Candidates13.length then search edge 3 [887, 4157] root151Candidates13 5044 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root151Candidates14_checked : candidates edge 4 887 5303 root151Tail15 winner.bound = root151Candidates14 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root151Step14 : scan edge (search edge 3) 4 [887] 887 root151Tail14 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail15 winner := by
  rw [root151Tail14, scan, if_neg (by decide), root151Candidates14_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail15
    (if 3 ≤ root151Candidates14.length then search edge 3 [887, 5303] root151Candidates14 6190 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates15 : List Nat := [10331, 11813]
set_option maxRecDepth 8192 in
theorem root151Candidates15_checked : candidates edge 4 887 5417 root151Tail16 winner.bound = root151Candidates15 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root151Step15 : scan edge (search edge 3) 4 [887] 887 root151Tail15 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail16 winner := by
  rw [root151Tail15, scan, if_neg (by decide), root151Candidates15_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail16
    (if 3 ≤ root151Candidates15.length then search edge 3 [887, 5417] root151Candidates15 6304 winner else winner) = _
  rw [if_neg (by decide)]
def root151Candidates16 : List Nat := [7499, 12959]
set_option maxRecDepth 8192 in
theorem root151Candidates16_checked : candidates edge 4 887 5651 root151Tail17 winner.bound = root151Candidates16 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root151Step16 : scan edge (search edge 3) 4 [887] 887 root151Tail16 winner =
    scan edge (search edge 3) 4 [887] 887 root151Tail17 winner := by
  rw [root151Tail16, scan, if_neg (by decide), root151Candidates16_checked]
  change scan edge (search edge 3) 4 [887] 887 root151Tail17
    (if 3 ≤ root151Candidates16.length then search edge 3 [887, 5651] root151Candidates16 6538 winner else winner) = _
  rw [if_neg (by decide)]
theorem root151Scan17 : scan edge (search edge 3) 4 [887] 887 root151Tail17 winner = winner := by rw [root151Tail17, scan, if_pos (by decide)]
theorem root151Scan16 : scan edge (search edge 3) 4 [887] 887 root151Tail16 winner = winner :=
  (root151Step16).trans root151Scan17
theorem root151Scan15 : scan edge (search edge 3) 4 [887] 887 root151Tail15 winner = winner :=
  (root151Step15).trans root151Scan16
theorem root151Scan14 : scan edge (search edge 3) 4 [887] 887 root151Tail14 winner = winner :=
  (root151Step14).trans root151Scan15
theorem root151Scan13 : scan edge (search edge 3) 4 [887] 887 root151Tail13 winner = winner :=
  (root151Step13).trans root151Scan14
theorem root151Scan12 : scan edge (search edge 3) 4 [887] 887 root151Tail12 winner = winner :=
  (root151Step12).trans root151Scan13
theorem root151Scan11 : scan edge (search edge 3) 4 [887] 887 root151Tail11 winner = winner :=
  (root151Step11).trans root151Scan12
theorem root151Scan10 : scan edge (search edge 3) 4 [887] 887 root151Tail10 winner = winner :=
  (root151Step10).trans root151Scan11
theorem root151Scan9 : scan edge (search edge 3) 4 [887] 887 root151Tail9 winner = winner :=
  (root151Step9).trans root151Scan10
theorem root151Scan8 : scan edge (search edge 3) 4 [887] 887 root151Tail8 winner = winner :=
  (root151Step8).trans root151Scan9
theorem root151Scan7 : scan edge (search edge 3) 4 [887] 887 root151Tail7 winner = winner :=
  (root151Step7).trans root151Scan8
theorem root151Scan6 : scan edge (search edge 3) 4 [887] 887 root151Tail6 winner = winner :=
  (root151Step6).trans root151Scan7
theorem root151Scan5 : scan edge (search edge 3) 4 [887] 887 root151Tail5 winner = winner :=
  (root151Step5).trans root151Scan6
theorem root151Scan4 : scan edge (search edge 3) 4 [887] 887 root151Tail4 winner = winner :=
  (root151Step4).trans root151Scan5
theorem root151Scan3 : scan edge (search edge 3) 4 [887] 887 root151Tail3 winner = winner :=
  (root151Step3).trans root151Scan4
theorem root151Scan2 : scan edge (search edge 3) 4 [887] 887 root151Tail2 winner = winner :=
  (root151Step2).trans root151Scan3
theorem root151Scan1 : scan edge (search edge 3) 4 [887] 887 root151Tail1 winner = winner :=
  (root151Step1).trans root151Scan2
theorem root151Scan0 : scan edge (search edge 3) 4 [887] 887 root151Tail0 winner = winner :=
  (root151Step0).trans root151Scan1
set_option maxRecDepth 4096 in
theorem root151Branch : search edge 4 [887] root151Candidates 887 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [887] 887 root151Tail0 winner = winner
  exact root151Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
