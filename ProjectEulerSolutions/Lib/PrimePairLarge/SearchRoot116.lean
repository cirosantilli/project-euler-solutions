import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root116Candidates : List Nat := Data.row653
set_option maxRecDepth 8192 in
theorem root116Candidates_checked : candidates edge 5 0 653 tail117 winner.bound = root116Candidates := by
  apply candidates_eq_row edge 5 0 653 tail117 root116Candidates winner.bound tail117_ordered
  · decide
  · decide
  · funext q
    rfl
def root116Tail54 : List Nat := []
def root116Tail53 : List Nat := 24953 :: root116Tail54
def root116Tail52 : List Nat := 24923 :: root116Tail53
def root116Tail51 : List Nat := 24317 :: root116Tail52
def root116Tail50 : List Nat := 24239 :: root116Tail51
def root116Tail49 : List Nat := 23327 :: root116Tail50
def root116Tail48 : List Nat := 22511 :: root116Tail49
def root116Tail47 : List Nat := 22469 :: root116Tail48
def root116Tail46 : List Nat := 21881 :: root116Tail47
def root116Tail45 : List Nat := 21647 :: root116Tail46
def root116Tail44 : List Nat := 21617 :: root116Tail45
def root116Tail43 : List Nat := 20219 :: root116Tail44
def root116Tail42 : List Nat := 19433 :: root116Tail43
def root116Tail41 : List Nat := 18191 :: root116Tail42
def root116Tail40 : List Nat := 17669 :: root116Tail41
def root116Tail39 : List Nat := 17489 :: root116Tail40
def root116Tail38 : List Nat := 15959 :: root116Tail39
def root116Tail37 : List Nat := 15881 :: root116Tail38
def root116Tail36 : List Nat := 15149 :: root116Tail37
def root116Tail35 : List Nat := 14321 :: root116Tail36
def root116Tail34 : List Nat := 14243 :: root116Tail35
def root116Tail33 : List Nat := 14087 :: root116Tail34
def root116Tail32 : List Nat := 13049 :: root116Tail33
def root116Tail31 : List Nat := 12251 :: root116Tail32
def root116Tail30 : List Nat := 11777 :: root116Tail31
def root116Tail29 : List Nat := 11681 :: root116Tail30
def root116Tail28 : List Nat := 11273 :: root116Tail29
def root116Tail27 : List Nat := 10853 :: root116Tail28
def root116Tail26 : List Nat := 10433 :: root116Tail27
def root116Tail25 : List Nat := 10079 :: root116Tail26
def root116Tail24 : List Nat := 8963 :: root116Tail25
def root116Tail23 : List Nat := 8747 :: root116Tail24
def root116Tail22 : List Nat := 8447 :: root116Tail23
def root116Tail21 : List Nat := 7757 :: root116Tail22
def root116Tail20 : List Nat := 7211 :: root116Tail21
def root116Tail19 : List Nat := 7109 :: root116Tail20
def root116Tail18 : List Nat := 6947 :: root116Tail19
def root116Tail17 : List Nat := 6863 :: root116Tail18
def root116Tail16 : List Nat := 5897 :: root116Tail17
def root116Tail15 : List Nat := 5387 :: root116Tail16
def root116Tail14 : List Nat := 4289 :: root116Tail15
def root116Tail13 : List Nat := 3911 :: root116Tail14
def root116Tail12 : List Nat := 3347 :: root116Tail13
def root116Tail11 : List Nat := 3251 :: root116Tail12
def root116Tail10 : List Nat := 3209 :: root116Tail11
def root116Tail9 : List Nat := 3167 :: root116Tail10
def root116Tail8 : List Nat := 2693 :: root116Tail9
def root116Tail7 : List Nat := 2531 :: root116Tail8
def root116Tail6 : List Nat := 2399 :: root116Tail7
def root116Tail5 : List Nat := 2003 :: root116Tail6
def root116Tail4 : List Nat := 1523 :: root116Tail5
def root116Tail3 : List Nat := 1289 :: root116Tail4
def root116Tail2 : List Nat := 1097 :: root116Tail3
def root116Tail1 : List Nat := 941 :: root116Tail2
def root116Tail0 : List Nat := 659 :: root116Tail1
def root116Candidates0 : List Nat := [11777, 14087, 14243, 22511]
set_option maxRecDepth 8192 in
theorem root116Candidates0_checked : candidates edge 4 653 659 root116Tail1 winner.bound = root116Candidates0 := by
  unfold candidates
  have row : edge 659 = (SortedNatTree.contains FastRows.row659) := by
    change Data.row659.contains = (SortedNatTree.contains FastRows.row659)
    exact FastRows.row659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root116Child0 : search edge 3 [653, 659] root116Candidates0 1312 winner = winner := by decide
theorem root116Step0 : scan edge (search edge 3) 4 [653] 653 root116Tail0 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail1 winner := by
  rw [root116Tail0, scan, if_neg (by decide), root116Candidates0_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail1
    (if 3 ≤ root116Candidates0.length then search edge 3 [653, 659] root116Candidates0 1312 winner else winner) = _
  rw [if_pos (by decide), root116Child0]
def root116Candidates1 : List Nat := [14243]
set_option maxRecDepth 8192 in
theorem root116Candidates1_checked : candidates edge 4 653 941 root116Tail2 winner.bound = root116Candidates1 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
theorem root116Step1 : scan edge (search edge 3) 4 [653] 653 root116Tail1 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail2 winner := by
  rw [root116Tail1, scan, if_neg (by decide), root116Candidates1_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail2
    (if 3 ≤ root116Candidates1.length then search edge 3 [653, 941] root116Candidates1 1594 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates2 : List Nat := [3251]
set_option maxRecDepth 8192 in
theorem root116Candidates2_checked : candidates edge 4 653 1097 root116Tail3 winner.bound = root116Candidates2 := by
  unfold candidates
  have row : edge 1097 = (SortedNatTree.contains FastRows.row1097) := by
    change Data.row1097.contains = (SortedNatTree.contains FastRows.row1097)
    exact FastRows.row1097_checked.symm
  rw [row]
  decide
theorem root116Step2 : scan edge (search edge 3) 4 [653] 653 root116Tail2 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail3 winner := by
  rw [root116Tail2, scan, if_neg (by decide), root116Candidates2_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail3
    (if 3 ≤ root116Candidates2.length then search edge 3 [653, 1097] root116Candidates2 1750 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates3_checked : candidates edge 4 653 1289 root116Tail4 winner.bound = root116Candidates3 := by
  unfold candidates
  have row : edge 1289 = (SortedNatTree.contains FastRows.row1289) := by
    change Data.row1289.contains = (SortedNatTree.contains FastRows.row1289)
    exact FastRows.row1289_checked.symm
  rw [row]
  decide
theorem root116Step3 : scan edge (search edge 3) 4 [653] 653 root116Tail3 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail4 winner := by
  rw [root116Tail3, scan, if_neg (by decide), root116Candidates3_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail4
    (if 3 ≤ root116Candidates3.length then search edge 3 [653, 1289] root116Candidates3 1942 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates4_checked : candidates edge 4 653 1523 root116Tail5 winner.bound = root116Candidates4 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
theorem root116Step4 : scan edge (search edge 3) 4 [653] 653 root116Tail4 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail5 winner := by
  rw [root116Tail4, scan, if_neg (by decide), root116Candidates4_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail5
    (if 3 ≤ root116Candidates4.length then search edge 3 [653, 1523] root116Candidates4 2176 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates5 : List Nat := [7211, 8747, 11777, 15959]
set_option maxRecDepth 8192 in
theorem root116Candidates5_checked : candidates edge 4 653 2003 root116Tail6 winner.bound = root116Candidates5 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root116Child5 : search edge 3 [653, 2003] root116Candidates5 2656 winner = winner := by decide
theorem root116Step5 : scan edge (search edge 3) 4 [653] 653 root116Tail5 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail6 winner := by
  rw [root116Tail5, scan, if_neg (by decide), root116Candidates5_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail6
    (if 3 ≤ root116Candidates5.length then search edge 3 [653, 2003] root116Candidates5 2656 winner else winner) = _
  rw [if_pos (by decide), root116Child5]
def root116Candidates6 : List Nat := [3251, 3911, 15959]
set_option maxRecDepth 8192 in
theorem root116Candidates6_checked : candidates edge 4 653 2399 root116Tail7 winner.bound = root116Candidates6 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root116Child6 : search edge 3 [653, 2399] root116Candidates6 3052 winner = winner := by decide
theorem root116Step6 : scan edge (search edge 3) 4 [653] 653 root116Tail6 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail7 winner := by
  rw [root116Tail6, scan, if_neg (by decide), root116Candidates6_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail7
    (if 3 ≤ root116Candidates6.length then search edge 3 [653, 2399] root116Candidates6 3052 winner else winner) = _
  rw [if_pos (by decide), root116Child6]
def root116Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates7_checked : candidates edge 4 653 2531 root116Tail8 winner.bound = root116Candidates7 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
theorem root116Step7 : scan edge (search edge 3) 4 [653] 653 root116Tail7 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail8 winner := by
  rw [root116Tail7, scan, if_neg (by decide), root116Candidates7_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail8
    (if 3 ≤ root116Candidates7.length then search edge 3 [653, 2531] root116Candidates7 3184 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates8_checked : candidates edge 4 653 2693 root116Tail9 winner.bound = root116Candidates8 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root116Step8 : scan edge (search edge 3) 4 [653] 653 root116Tail8 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail9 winner := by
  rw [root116Tail8, scan, if_neg (by decide), root116Candidates8_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail9
    (if 3 ≤ root116Candidates8.length then search edge 3 [653, 2693] root116Candidates8 3346 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates9_checked : candidates edge 4 653 3167 root116Tail10 winner.bound = root116Candidates9 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root116Step9 : scan edge (search edge 3) 4 [653] 653 root116Tail9 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail10 winner := by
  rw [root116Tail9, scan, if_neg (by decide), root116Candidates9_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail10
    (if 3 ≤ root116Candidates9.length then search edge 3 [653, 3167] root116Candidates9 3820 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates10_checked : candidates edge 4 653 3209 root116Tail11 winner.bound = root116Candidates10 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root116Step10 : scan edge (search edge 3) 4 [653] 653 root116Tail10 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail11 winner := by
  rw [root116Tail10, scan, if_neg (by decide), root116Candidates10_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail11
    (if 3 ≤ root116Candidates10.length then search edge 3 [653, 3209] root116Candidates10 3862 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates11 : List Nat := [8963]
set_option maxRecDepth 8192 in
theorem root116Candidates11_checked : candidates edge 4 653 3251 root116Tail12 winner.bound = root116Candidates11 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root116Step11 : scan edge (search edge 3) 4 [653] 653 root116Tail11 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail12 winner := by
  rw [root116Tail11, scan, if_neg (by decide), root116Candidates11_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail12
    (if 3 ≤ root116Candidates11.length then search edge 3 [653, 3251] root116Candidates11 3904 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates12 : List Nat := [11777]
set_option maxRecDepth 8192 in
theorem root116Candidates12_checked : candidates edge 4 653 3347 root116Tail13 winner.bound = root116Candidates12 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root116Step12 : scan edge (search edge 3) 4 [653] 653 root116Tail12 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail13 winner := by
  rw [root116Tail12, scan, if_neg (by decide), root116Candidates12_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail13
    (if 3 ≤ root116Candidates12.length then search edge 3 [653, 3347] root116Candidates12 4000 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates13_checked : candidates edge 4 653 3911 root116Tail14 winner.bound = root116Candidates13 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root116Step13 : scan edge (search edge 3) 4 [653] 653 root116Tail13 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail14 winner := by
  rw [root116Tail13, scan, if_neg (by decide), root116Candidates13_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail14
    (if 3 ≤ root116Candidates13.length then search edge 3 [653, 3911] root116Candidates13 4564 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates14_checked : candidates edge 4 653 4289 root116Tail15 winner.bound = root116Candidates14 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root116Step14 : scan edge (search edge 3) 4 [653] 653 root116Tail14 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail15 winner := by
  rw [root116Tail14, scan, if_neg (by decide), root116Candidates14_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail15
    (if 3 ≤ root116Candidates14.length then search edge 3 [653, 4289] root116Candidates14 4942 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates15_checked : candidates edge 4 653 5387 root116Tail16 winner.bound = root116Candidates15 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root116Step15 : scan edge (search edge 3) 4 [653] 653 root116Tail15 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail16 winner := by
  rw [root116Tail15, scan, if_neg (by decide), root116Candidates15_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail16
    (if 3 ≤ root116Candidates15.length then search edge 3 [653, 5387] root116Candidates15 6040 winner else winner) = _
  rw [if_neg (by decide)]
def root116Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root116Candidates16_checked : candidates edge 4 653 5897 root116Tail17 winner.bound = root116Candidates16 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root116Step16 : scan edge (search edge 3) 4 [653] 653 root116Tail16 winner =
    scan edge (search edge 3) 4 [653] 653 root116Tail17 winner := by
  rw [root116Tail16, scan, if_neg (by decide), root116Candidates16_checked]
  change scan edge (search edge 3) 4 [653] 653 root116Tail17
    (if 3 ≤ root116Candidates16.length then search edge 3 [653, 5897] root116Candidates16 6550 winner else winner) = _
  rw [if_neg (by decide)]
theorem root116Scan17 : scan edge (search edge 3) 4 [653] 653 root116Tail17 winner = winner := by rw [root116Tail17, scan, if_pos (by decide)]
theorem root116Scan16 : scan edge (search edge 3) 4 [653] 653 root116Tail16 winner = winner :=
  (root116Step16).trans root116Scan17
theorem root116Scan15 : scan edge (search edge 3) 4 [653] 653 root116Tail15 winner = winner :=
  (root116Step15).trans root116Scan16
theorem root116Scan14 : scan edge (search edge 3) 4 [653] 653 root116Tail14 winner = winner :=
  (root116Step14).trans root116Scan15
theorem root116Scan13 : scan edge (search edge 3) 4 [653] 653 root116Tail13 winner = winner :=
  (root116Step13).trans root116Scan14
theorem root116Scan12 : scan edge (search edge 3) 4 [653] 653 root116Tail12 winner = winner :=
  (root116Step12).trans root116Scan13
theorem root116Scan11 : scan edge (search edge 3) 4 [653] 653 root116Tail11 winner = winner :=
  (root116Step11).trans root116Scan12
theorem root116Scan10 : scan edge (search edge 3) 4 [653] 653 root116Tail10 winner = winner :=
  (root116Step10).trans root116Scan11
theorem root116Scan9 : scan edge (search edge 3) 4 [653] 653 root116Tail9 winner = winner :=
  (root116Step9).trans root116Scan10
theorem root116Scan8 : scan edge (search edge 3) 4 [653] 653 root116Tail8 winner = winner :=
  (root116Step8).trans root116Scan9
theorem root116Scan7 : scan edge (search edge 3) 4 [653] 653 root116Tail7 winner = winner :=
  (root116Step7).trans root116Scan8
theorem root116Scan6 : scan edge (search edge 3) 4 [653] 653 root116Tail6 winner = winner :=
  (root116Step6).trans root116Scan7
theorem root116Scan5 : scan edge (search edge 3) 4 [653] 653 root116Tail5 winner = winner :=
  (root116Step5).trans root116Scan6
theorem root116Scan4 : scan edge (search edge 3) 4 [653] 653 root116Tail4 winner = winner :=
  (root116Step4).trans root116Scan5
theorem root116Scan3 : scan edge (search edge 3) 4 [653] 653 root116Tail3 winner = winner :=
  (root116Step3).trans root116Scan4
theorem root116Scan2 : scan edge (search edge 3) 4 [653] 653 root116Tail2 winner = winner :=
  (root116Step2).trans root116Scan3
theorem root116Scan1 : scan edge (search edge 3) 4 [653] 653 root116Tail1 winner = winner :=
  (root116Step1).trans root116Scan2
theorem root116Scan0 : scan edge (search edge 3) 4 [653] 653 root116Tail0 winner = winner :=
  (root116Step0).trans root116Scan1
set_option maxRecDepth 4096 in
theorem root116Branch : search edge 4 [653] root116Candidates 653 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [653] 653 root116Tail0 winner = winner
  exact root116Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
