import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root66Candidates : List Nat := Data.row347
set_option maxRecDepth 8192 in
theorem root66Candidates_checked : candidates edge 5 0 347 tail67 winner.bound = root66Candidates := by
  apply candidates_eq_row edge 5 0 347 tail67 root66Candidates winner.bound tail67_ordered
  · decide
  · decide
  · funext q
    rfl
def root66Tail68 : List Nat := []
def root66Tail67 : List Nat := 25367 :: root66Tail68
def root66Tail66 : List Nat := 25121 :: root66Tail67
def root66Tail65 : List Nat := 24317 :: root66Tail66
def root66Tail64 : List Nat := 23879 :: root66Tail65
def root66Tail63 : List Nat := 23603 :: root66Tail64
def root66Tail62 : List Nat := 23021 :: root66Tail63
def root66Tail61 : List Nat := 22679 :: root66Tail62
def root66Tail60 : List Nat := 22307 :: root66Tail61
def root66Tail59 : List Nat := 22283 :: root66Tail60
def root66Tail58 : List Nat := 22073 :: root66Tail59
def root66Tail57 : List Nat := 20747 :: root66Tail58
def root66Tail56 : List Nat := 20369 :: root66Tail57
def root66Tail55 : List Nat := 20183 :: root66Tail56
def root66Tail54 : List Nat := 20177 :: root66Tail55
def root66Tail53 : List Nat := 20123 :: root66Tail54
def root66Tail52 : List Nat := 19697 :: root66Tail53
def root66Tail51 : List Nat := 19571 :: root66Tail52
def root66Tail50 : List Nat := 19469 :: root66Tail51
def root66Tail49 : List Nat := 19211 :: root66Tail50
def root66Tail48 : List Nat := 18041 :: root66Tail49
def root66Tail47 : List Nat := 17159 :: root66Tail48
def root66Tail46 : List Nat := 17027 :: root66Tail47
def root66Tail45 : List Nat := 16823 :: root66Tail46
def root66Tail44 : List Nat := 16763 :: root66Tail45
def root66Tail43 : List Nat := 16421 :: root66Tail44
def root66Tail42 : List Nat := 15137 :: root66Tail43
def root66Tail41 : List Nat := 15101 :: root66Tail42
def root66Tail40 : List Nat := 14591 :: root66Tail41
def root66Tail39 : List Nat := 13901 :: root66Tail40
def root66Tail38 : List Nat := 13883 :: root66Tail39
def root66Tail37 : List Nat := 13001 :: root66Tail38
def root66Tail36 : List Nat := 12941 :: root66Tail37
def root66Tail35 : List Nat := 11903 :: root66Tail36
def root66Tail34 : List Nat := 11783 :: root66Tail35
def root66Tail33 : List Nat := 11717 :: root66Tail34
def root66Tail32 : List Nat := 11681 :: root66Tail33
def root66Tail31 : List Nat := 11489 :: root66Tail32
def root66Tail30 : List Nat := 11213 :: root66Tail31
def root66Tail29 : List Nat := 10433 :: root66Tail30
def root66Tail28 : List Nat := 10223 :: root66Tail29
def root66Tail27 : List Nat := 9857 :: root66Tail28
def root66Tail26 : List Nat := 9767 :: root66Tail27
def root66Tail25 : List Nat := 8513 :: root66Tail26
def root66Tail24 : List Nat := 8009 :: root66Tail25
def root66Tail23 : List Nat := 7841 :: root66Tail24
def root66Tail22 : List Nat := 7559 :: root66Tail23
def root66Tail21 : List Nat := 7517 :: root66Tail22
def root66Tail20 : List Nat := 7499 :: root66Tail21
def root66Tail19 : List Nat := 7121 :: root66Tail20
def root66Tail18 : List Nat := 6761 :: root66Tail19
def root66Tail17 : List Nat := 6719 :: root66Tail18
def root66Tail16 : List Nat := 6569 :: root66Tail17
def root66Tail15 : List Nat := 6257 :: root66Tail16
def root66Tail14 : List Nat := 5981 :: root66Tail15
def root66Tail13 : List Nat := 5861 :: root66Tail14
def root66Tail12 : List Nat := 5477 :: root66Tail13
def root66Tail11 : List Nat := 4517 :: root66Tail12
def root66Tail10 : List Nat := 3923 :: root66Tail11
def root66Tail9 : List Nat := 3719 :: root66Tail10
def root66Tail8 : List Nat := 3677 :: root66Tail9
def root66Tail7 : List Nat := 2213 :: root66Tail8
def root66Tail6 : List Nat := 1847 :: root66Tail7
def root66Tail5 : List Nat := 1697 :: root66Tail6
def root66Tail4 : List Nat := 1301 :: root66Tail5
def root66Tail3 : List Nat := 1031 :: root66Tail4
def root66Tail2 : List Nat := 983 :: root66Tail3
def root66Tail1 : List Nat := 443 :: root66Tail2
def root66Tail0 : List Nat := 401 :: root66Tail1
def root66Candidates0 : List Nat := [2213]
set_option maxRecDepth 8192 in
theorem root66Candidates0_checked : candidates edge 4 347 401 root66Tail1 winner.bound = root66Candidates0 := by
  unfold candidates
  have row : edge 401 = (SortedNatTree.contains FastRows.row401) := by
    change Data.row401.contains = (SortedNatTree.contains FastRows.row401)
    exact FastRows.row401_checked.symm
  rw [row]
  decide
theorem root66Step0 : scan edge (search edge 3) 4 [347] 347 root66Tail0 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail1 winner := by
  rw [root66Tail0, scan, if_neg (by decide), root66Candidates0_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail1
    (if 3 ≤ root66Candidates0.length then search edge 3 [347, 401] root66Candidates0 748 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates1 : List Nat := [983, 9857, 11717, 14591, 23603, 24317, 25121]
set_option maxRecDepth 8192 in
theorem root66Candidates1_checked : candidates edge 4 347 443 root66Tail2 winner.bound = root66Candidates1 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root66Child1 : search edge 3 [347, 443] root66Candidates1 790 winner = winner := by decide
theorem root66Step1 : scan edge (search edge 3) 4 [347] 347 root66Tail1 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail2 winner := by
  rw [root66Tail1, scan, if_neg (by decide), root66Candidates1_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail2
    (if 3 ≤ root66Candidates1.length then search edge 3 [347, 443] root66Candidates1 790 winner else winner) = _
  rw [if_pos (by decide), root66Child1]
def root66Candidates2 : List Nat := [6719, 9857, 11489]
set_option maxRecDepth 8192 in
theorem root66Candidates2_checked : candidates edge 4 347 983 root66Tail3 winner.bound = root66Candidates2 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root66Child2 : search edge 3 [347, 983] root66Candidates2 1330 winner = winner := by decide
theorem root66Step2 : scan edge (search edge 3) 4 [347] 347 root66Tail2 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail3 winner := by
  rw [root66Tail2, scan, if_neg (by decide), root66Candidates2_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail3
    (if 3 ≤ root66Candidates2.length then search edge 3 [347, 983] root66Candidates2 1330 winner else winner) = _
  rw [if_pos (by decide), root66Child2]
def root66Candidates3 : List Nat := [16421]
set_option maxRecDepth 8192 in
theorem root66Candidates3_checked : candidates edge 4 347 1031 root66Tail4 winner.bound = root66Candidates3 := by
  unfold candidates
  have row : edge 1031 = (SortedNatTree.contains FastRows.row1031) := by
    change Data.row1031.contains = (SortedNatTree.contains FastRows.row1031)
    exact FastRows.row1031_checked.symm
  rw [row]
  decide
theorem root66Step3 : scan edge (search edge 3) 4 [347] 347 root66Tail3 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail4 winner := by
  rw [root66Tail3, scan, if_neg (by decide), root66Candidates3_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail4
    (if 3 ≤ root66Candidates3.length then search edge 3 [347, 1031] root66Candidates3 1378 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates4 : List Nat := [9857, 10223, 14591, 16421]
set_option maxRecDepth 8192 in
theorem root66Candidates4_checked : candidates edge 4 347 1301 root66Tail5 winner.bound = root66Candidates4 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root66Child4 : search edge 3 [347, 1301] root66Candidates4 1648 winner = winner := by decide
theorem root66Step4 : scan edge (search edge 3) 4 [347] 347 root66Tail4 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail5 winner := by
  rw [root66Tail4, scan, if_neg (by decide), root66Candidates4_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail5
    (if 3 ≤ root66Candidates4.length then search edge 3 [347, 1301] root66Candidates4 1648 winner else winner) = _
  rw [if_pos (by decide), root66Child4]
def root66Candidates5 : List Nat := [5477, 11681]
set_option maxRecDepth 8192 in
theorem root66Candidates5_checked : candidates edge 4 347 1697 root66Tail6 winner.bound = root66Candidates5 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root66Step5 : scan edge (search edge 3) 4 [347] 347 root66Tail5 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail6 winner := by
  rw [root66Tail5, scan, if_neg (by decide), root66Candidates5_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail6
    (if 3 ≤ root66Candidates5.length then search edge 3 [347, 1697] root66Candidates5 2044 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root66Candidates6_checked : candidates edge 4 347 1847 root66Tail7 winner.bound = root66Candidates6 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
theorem root66Step6 : scan edge (search edge 3) 4 [347] 347 root66Tail6 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail7 winner := by
  rw [root66Tail6, scan, if_neg (by decide), root66Candidates6_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail7
    (if 3 ≤ root66Candidates6.length then search edge 3 [347, 1847] root66Candidates6 2194 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root66Candidates7_checked : candidates edge 4 347 2213 root66Tail8 winner.bound = root66Candidates7 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root66Step7 : scan edge (search edge 3) 4 [347] 347 root66Tail7 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail8 winner := by
  rw [root66Tail7, scan, if_neg (by decide), root66Candidates7_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail8
    (if 3 ≤ root66Candidates7.length then search edge 3 [347, 2213] root66Candidates7 2560 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates8 : List Nat := [16763]
set_option maxRecDepth 8192 in
theorem root66Candidates8_checked : candidates edge 4 347 3677 root66Tail9 winner.bound = root66Candidates8 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root66Step8 : scan edge (search edge 3) 4 [347] 347 root66Tail8 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail9 winner := by
  rw [root66Tail8, scan, if_neg (by decide), root66Candidates8_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail9
    (if 3 ≤ root66Candidates8.length then search edge 3 [347, 3677] root66Candidates8 4024 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root66Candidates9_checked : candidates edge 4 347 3719 root66Tail10 winner.bound = root66Candidates9 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root66Step9 : scan edge (search edge 3) 4 [347] 347 root66Tail9 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail10 winner := by
  rw [root66Tail9, scan, if_neg (by decide), root66Candidates9_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail10
    (if 3 ≤ root66Candidates9.length then search edge 3 [347, 3719] root66Candidates9 4066 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates10 : List Nat := [11783]
set_option maxRecDepth 8192 in
theorem root66Candidates10_checked : candidates edge 4 347 3923 root66Tail11 winner.bound = root66Candidates10 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root66Step10 : scan edge (search edge 3) 4 [347] 347 root66Tail10 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail11 winner := by
  rw [root66Tail10, scan, if_neg (by decide), root66Candidates10_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail11
    (if 3 ≤ root66Candidates10.length then search edge 3 [347, 3923] root66Candidates10 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates11 : List Nat := [12941]
set_option maxRecDepth 8192 in
theorem root66Candidates11_checked : candidates edge 4 347 4517 root66Tail12 winner.bound = root66Candidates11 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root66Step11 : scan edge (search edge 3) 4 [347] 347 root66Tail11 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail12 winner := by
  rw [root66Tail11, scan, if_neg (by decide), root66Candidates11_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail12
    (if 3 ≤ root66Candidates11.length then search edge 3 [347, 4517] root66Candidates11 4864 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root66Candidates12_checked : candidates edge 4 347 5477 root66Tail13 winner.bound = root66Candidates12 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root66Step12 : scan edge (search edge 3) 4 [347] 347 root66Tail12 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail13 winner := by
  rw [root66Tail12, scan, if_neg (by decide), root66Candidates12_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail13
    (if 3 ≤ root66Candidates12.length then search edge 3 [347, 5477] root66Candidates12 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates13 : List Nat := [11681, 12941]
set_option maxRecDepth 8192 in
theorem root66Candidates13_checked : candidates edge 4 347 5861 root66Tail14 winner.bound = root66Candidates13 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root66Step13 : scan edge (search edge 3) 4 [347] 347 root66Tail13 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail14 winner := by
  rw [root66Tail13, scan, if_neg (by decide), root66Candidates13_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail14
    (if 3 ≤ root66Candidates13.length then search edge 3 [347, 5861] root66Candidates13 6208 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root66Candidates14_checked : candidates edge 4 347 5981 root66Tail15 winner.bound = root66Candidates14 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root66Step14 : scan edge (search edge 3) 4 [347] 347 root66Tail14 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail15 winner := by
  rw [root66Tail14, scan, if_neg (by decide), root66Candidates14_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail15
    (if 3 ≤ root66Candidates14.length then search edge 3 [347, 5981] root66Candidates14 6328 winner else winner) = _
  rw [if_neg (by decide)]
def root66Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root66Candidates15_checked : candidates edge 4 347 6257 root66Tail16 winner.bound = root66Candidates15 := by
  unfold candidates
  have row : edge 6257 = (SortedNatTree.contains FastRows.row6257) := by
    change Data.row6257.contains = (SortedNatTree.contains FastRows.row6257)
    exact FastRows.row6257_checked.symm
  rw [row]
  decide
theorem root66Step15 : scan edge (search edge 3) 4 [347] 347 root66Tail15 winner =
    scan edge (search edge 3) 4 [347] 347 root66Tail16 winner := by
  rw [root66Tail15, scan, if_neg (by decide), root66Candidates15_checked]
  change scan edge (search edge 3) 4 [347] 347 root66Tail16
    (if 3 ≤ root66Candidates15.length then search edge 3 [347, 6257] root66Candidates15 6604 winner else winner) = _
  rw [if_neg (by decide)]
theorem root66Scan16 : scan edge (search edge 3) 4 [347] 347 root66Tail16 winner = winner := by rw [root66Tail16, scan, if_pos (by decide)]
theorem root66Scan15 : scan edge (search edge 3) 4 [347] 347 root66Tail15 winner = winner :=
  (root66Step15).trans root66Scan16
theorem root66Scan14 : scan edge (search edge 3) 4 [347] 347 root66Tail14 winner = winner :=
  (root66Step14).trans root66Scan15
theorem root66Scan13 : scan edge (search edge 3) 4 [347] 347 root66Tail13 winner = winner :=
  (root66Step13).trans root66Scan14
theorem root66Scan12 : scan edge (search edge 3) 4 [347] 347 root66Tail12 winner = winner :=
  (root66Step12).trans root66Scan13
theorem root66Scan11 : scan edge (search edge 3) 4 [347] 347 root66Tail11 winner = winner :=
  (root66Step11).trans root66Scan12
theorem root66Scan10 : scan edge (search edge 3) 4 [347] 347 root66Tail10 winner = winner :=
  (root66Step10).trans root66Scan11
theorem root66Scan9 : scan edge (search edge 3) 4 [347] 347 root66Tail9 winner = winner :=
  (root66Step9).trans root66Scan10
theorem root66Scan8 : scan edge (search edge 3) 4 [347] 347 root66Tail8 winner = winner :=
  (root66Step8).trans root66Scan9
theorem root66Scan7 : scan edge (search edge 3) 4 [347] 347 root66Tail7 winner = winner :=
  (root66Step7).trans root66Scan8
theorem root66Scan6 : scan edge (search edge 3) 4 [347] 347 root66Tail6 winner = winner :=
  (root66Step6).trans root66Scan7
theorem root66Scan5 : scan edge (search edge 3) 4 [347] 347 root66Tail5 winner = winner :=
  (root66Step5).trans root66Scan6
theorem root66Scan4 : scan edge (search edge 3) 4 [347] 347 root66Tail4 winner = winner :=
  (root66Step4).trans root66Scan5
theorem root66Scan3 : scan edge (search edge 3) 4 [347] 347 root66Tail3 winner = winner :=
  (root66Step3).trans root66Scan4
theorem root66Scan2 : scan edge (search edge 3) 4 [347] 347 root66Tail2 winner = winner :=
  (root66Step2).trans root66Scan3
theorem root66Scan1 : scan edge (search edge 3) 4 [347] 347 root66Tail1 winner = winner :=
  (root66Step1).trans root66Scan2
theorem root66Scan0 : scan edge (search edge 3) 4 [347] 347 root66Tail0 winner = winner :=
  (root66Step0).trans root66Scan1
set_option maxRecDepth 4096 in
theorem root66Branch : search edge 4 [347] root66Candidates 347 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [347] 347 root66Tail0 winner = winner
  exact root66Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
