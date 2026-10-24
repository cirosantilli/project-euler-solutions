import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root146Candidates : List Nat := Data.row859
set_option maxRecDepth 8192 in
theorem root146Candidates_checked : candidates edge 5 0 859 tail147 winner.bound = root146Candidates := by
  apply candidates_eq_row edge 5 0 859 tail147 root146Candidates winner.bound tail147_ordered
  · decide
  · decide
  · funext q
    rfl
def root146Tail59 : List Nat := []
def root146Tail58 : List Nat := 25153 :: root146Tail59
def root146Tail57 : List Nat := 24781 :: root146Tail58
def root146Tail56 : List Nat := 24691 :: root146Tail57
def root146Tail55 : List Nat := 24469 :: root146Tail56
def root146Tail54 : List Nat := 24229 :: root146Tail55
def root146Tail53 : List Nat := 24049 :: root146Tail54
def root146Tail52 : List Nat := 23977 :: root146Tail53
def root146Tail51 : List Nat := 22189 :: root146Tail52
def root146Tail50 : List Nat := 22093 :: root146Tail51
def root146Tail49 : List Nat := 21163 :: root146Tail50
def root146Tail48 : List Nat := 20161 :: root146Tail49
def root146Tail47 : List Nat := 20143 :: root146Tail48
def root146Tail46 : List Nat := 19333 :: root146Tail47
def root146Tail45 : List Nat := 18757 :: root146Tail46
def root146Tail44 : List Nat := 18433 :: root146Tail45
def root146Tail43 : List Nat := 18211 :: root146Tail44
def root146Tail42 : List Nat := 17851 :: root146Tail43
def root146Tail41 : List Nat := 17257 :: root146Tail42
def root146Tail40 : List Nat := 17191 :: root146Tail41
def root146Tail39 : List Nat := 16573 :: root146Tail40
def root146Tail38 : List Nat := 15733 :: root146Tail39
def root146Tail37 : List Nat := 15241 :: root146Tail38
def root146Tail36 : List Nat := 14869 :: root146Tail37
def root146Tail35 : List Nat := 14293 :: root146Tail36
def root146Tail34 : List Nat := 14221 :: root146Tail35
def root146Tail33 : List Nat := 12823 :: root146Tail34
def root146Tail32 : List Nat := 12511 :: root146Tail33
def root146Tail31 : List Nat := 11719 :: root146Tail32
def root146Tail30 : List Nat := 11689 :: root146Tail31
def root146Tail29 : List Nat := 11239 :: root146Tail30
def root146Tail28 : List Nat := 11131 :: root146Tail29
def root146Tail27 : List Nat := 10459 :: root146Tail28
def root146Tail26 : List Nat := 10303 :: root146Tail27
def root146Tail25 : List Nat := 10177 :: root146Tail26
def root146Tail24 : List Nat := 9397 :: root146Tail25
def root146Tail23 : List Nat := 8353 :: root146Tail24
def root146Tail22 : List Nat := 8221 :: root146Tail23
def root146Tail21 : List Nat := 8101 :: root146Tail22
def root146Tail20 : List Nat := 7759 :: root146Tail21
def root146Tail19 : List Nat := 7669 :: root146Tail20
def root146Tail18 : List Nat := 7639 :: root146Tail19
def root146Tail17 : List Nat := 7417 :: root146Tail18
def root146Tail16 : List Nat := 7219 :: root146Tail17
def root146Tail15 : List Nat := 7213 :: root146Tail16
def root146Tail14 : List Nat := 6199 :: root146Tail15
def root146Tail13 : List Nat := 5683 :: root146Tail14
def root146Tail12 : List Nat := 5623 :: root146Tail13
def root146Tail11 : List Nat := 5023 :: root146Tail12
def root146Tail10 : List Nat := 4549 :: root146Tail11
def root146Tail9 : List Nat := 4507 :: root146Tail10
def root146Tail8 : List Nat := 3943 :: root146Tail9
def root146Tail7 : List Nat := 3847 :: root146Tail8
def root146Tail6 : List Nat := 3391 :: root146Tail7
def root146Tail5 : List Nat := 3307 :: root146Tail6
def root146Tail4 : List Nat := 2659 :: root146Tail5
def root146Tail3 : List Nat := 1543 :: root146Tail4
def root146Tail2 : List Nat := 1381 :: root146Tail3
def root146Tail1 : List Nat := 1129 :: root146Tail2
def root146Tail0 : List Nat := 919 :: root146Tail1
def root146Candidates0 : List Nat := [10459, 14293, 17191, 17851, 22189]
set_option maxRecDepth 8192 in
theorem root146Candidates0_checked : candidates edge 4 859 919 root146Tail1 winner.bound = root146Candidates0 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root146Child0 : search edge 3 [859, 919] root146Candidates0 1778 winner = winner := by decide
theorem root146Step0 : scan edge (search edge 3) 4 [859] 859 root146Tail0 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail1 winner := by
  rw [root146Tail0, scan, if_neg (by decide), root146Candidates0_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail1
    (if 3 ≤ root146Candidates0.length then search edge 3 [859, 919] root146Candidates0 1778 winner else winner) = _
  rw [if_pos (by decide), root146Child0]
def root146Candidates1 : List Nat := [7417, 18433, 21163, 22189]
set_option maxRecDepth 8192 in
theorem root146Candidates1_checked : candidates edge 4 859 1129 root146Tail2 winner.bound = root146Candidates1 := by
  unfold candidates
  have row : edge 1129 = (SortedNatTree.contains FastRows.row1129) := by
    change Data.row1129.contains = (SortedNatTree.contains FastRows.row1129)
    exact FastRows.row1129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root146Child1 : search edge 3 [859, 1129] root146Candidates1 1988 winner = winner := by decide
theorem root146Step1 : scan edge (search edge 3) 4 [859] 859 root146Tail1 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail2 winner := by
  rw [root146Tail1, scan, if_neg (by decide), root146Candidates1_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail2
    (if 3 ≤ root146Candidates1.length then search edge 3 [859, 1129] root146Candidates1 1988 winner else winner) = _
  rw [if_pos (by decide), root146Child1]
def root146Candidates2 : List Nat := [17191]
set_option maxRecDepth 8192 in
theorem root146Candidates2_checked : candidates edge 4 859 1381 root146Tail3 winner.bound = root146Candidates2 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root146Step2 : scan edge (search edge 3) 4 [859] 859 root146Tail2 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail3 winner := by
  rw [root146Tail2, scan, if_neg (by decide), root146Candidates2_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail3
    (if 3 ≤ root146Candidates2.length then search edge 3 [859, 1381] root146Candidates2 2240 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates3 : List Nat := [16573, 20143]
set_option maxRecDepth 8192 in
theorem root146Candidates3_checked : candidates edge 4 859 1543 root146Tail4 winner.bound = root146Candidates3 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
theorem root146Step3 : scan edge (search edge 3) 4 [859] 859 root146Tail3 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail4 winner := by
  rw [root146Tail3, scan, if_neg (by decide), root146Candidates3_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail4
    (if 3 ≤ root146Candidates3.length then search edge 3 [859, 1543] root146Candidates3 2402 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root146Candidates4_checked : candidates edge 4 859 2659 root146Tail5 winner.bound = root146Candidates4 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
theorem root146Step4 : scan edge (search edge 3) 4 [859] 859 root146Tail4 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail5 winner := by
  rw [root146Tail4, scan, if_neg (by decide), root146Candidates4_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail5
    (if 3 ≤ root146Candidates4.length then search edge 3 [859, 2659] root146Candidates4 3518 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root146Candidates5_checked : candidates edge 4 859 3307 root146Tail6 winner.bound = root146Candidates5 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root146Step5 : scan edge (search edge 3) 4 [859] 859 root146Tail5 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail6 winner := by
  rw [root146Tail5, scan, if_neg (by decide), root146Candidates5_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail6
    (if 3 ≤ root146Candidates5.length then search edge 3 [859, 3307] root146Candidates5 4166 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates6 : List Nat := [6199]
set_option maxRecDepth 8192 in
theorem root146Candidates6_checked : candidates edge 4 859 3391 root146Tail7 winner.bound = root146Candidates6 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
theorem root146Step6 : scan edge (search edge 3) 4 [859] 859 root146Tail6 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail7 winner := by
  rw [root146Tail6, scan, if_neg (by decide), root146Candidates6_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail7
    (if 3 ≤ root146Candidates6.length then search edge 3 [859, 3391] root146Candidates6 4250 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates7 : List Nat := [8221]
set_option maxRecDepth 8192 in
theorem root146Candidates7_checked : candidates edge 4 859 3847 root146Tail8 winner.bound = root146Candidates7 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root146Step7 : scan edge (search edge 3) 4 [859] 859 root146Tail7 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail8 winner := by
  rw [root146Tail7, scan, if_neg (by decide), root146Candidates7_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail8
    (if 3 ≤ root146Candidates7.length then search edge 3 [859, 3847] root146Candidates7 4706 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root146Candidates8_checked : candidates edge 4 859 3943 root146Tail9 winner.bound = root146Candidates8 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root146Step8 : scan edge (search edge 3) 4 [859] 859 root146Tail8 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail9 winner := by
  rw [root146Tail8, scan, if_neg (by decide), root146Candidates8_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail9
    (if 3 ≤ root146Candidates8.length then search edge 3 [859, 3943] root146Candidates8 4802 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates9 : List Nat := [7759, 18211]
set_option maxRecDepth 8192 in
theorem root146Candidates9_checked : candidates edge 4 859 4507 root146Tail10 winner.bound = root146Candidates9 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root146Step9 : scan edge (search edge 3) 4 [859] 859 root146Tail9 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail10 winner := by
  rw [root146Tail9, scan, if_neg (by decide), root146Candidates9_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail10
    (if 3 ≤ root146Candidates9.length then search edge 3 [859, 4507] root146Candidates9 5366 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates10 : List Nat := [11239]
set_option maxRecDepth 8192 in
theorem root146Candidates10_checked : candidates edge 4 859 4549 root146Tail11 winner.bound = root146Candidates10 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root146Step10 : scan edge (search edge 3) 4 [859] 859 root146Tail10 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail11 winner := by
  rw [root146Tail10, scan, if_neg (by decide), root146Candidates10_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail11
    (if 3 ≤ root146Candidates10.length then search edge 3 [859, 4549] root146Candidates10 5408 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates11 : List Nat := [18211]
set_option maxRecDepth 8192 in
theorem root146Candidates11_checked : candidates edge 4 859 5023 root146Tail12 winner.bound = root146Candidates11 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root146Step11 : scan edge (search edge 3) 4 [859] 859 root146Tail11 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail12 winner := by
  rw [root146Tail11, scan, if_neg (by decide), root146Candidates11_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail12
    (if 3 ≤ root146Candidates11.length then search edge 3 [859, 5023] root146Candidates11 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates12 : List Nat := [7639]
set_option maxRecDepth 8192 in
theorem root146Candidates12_checked : candidates edge 4 859 5623 root146Tail13 winner.bound = root146Candidates12 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root146Step12 : scan edge (search edge 3) 4 [859] 859 root146Tail12 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail13 winner := by
  rw [root146Tail12, scan, if_neg (by decide), root146Candidates12_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail13
    (if 3 ≤ root146Candidates12.length then search edge 3 [859, 5623] root146Candidates12 6482 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates13 : List Nat := [14221]
set_option maxRecDepth 8192 in
theorem root146Candidates13_checked : candidates edge 4 859 5683 root146Tail14 winner.bound = root146Candidates13 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root146Step13 : scan edge (search edge 3) 4 [859] 859 root146Tail13 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail14 winner := by
  rw [root146Tail13, scan, if_neg (by decide), root146Candidates13_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail14
    (if 3 ≤ root146Candidates13.length then search edge 3 [859, 5683] root146Candidates13 6542 winner else winner) = _
  rw [if_neg (by decide)]
def root146Candidates14 : List Nat := [7219]
set_option maxRecDepth 8192 in
theorem root146Candidates14_checked : candidates edge 4 859 6199 root146Tail15 winner.bound = root146Candidates14 := by
  unfold candidates
  have row : edge 6199 = (SortedNatTree.contains FastRows.row6199) := by
    change Data.row6199.contains = (SortedNatTree.contains FastRows.row6199)
    exact FastRows.row6199_checked.symm
  rw [row]
  decide
theorem root146Step14 : scan edge (search edge 3) 4 [859] 859 root146Tail14 winner =
    scan edge (search edge 3) 4 [859] 859 root146Tail15 winner := by
  rw [root146Tail14, scan, if_neg (by decide), root146Candidates14_checked]
  change scan edge (search edge 3) 4 [859] 859 root146Tail15
    (if 3 ≤ root146Candidates14.length then search edge 3 [859, 6199] root146Candidates14 7058 winner else winner) = _
  rw [if_neg (by decide)]
theorem root146Scan15 : scan edge (search edge 3) 4 [859] 859 root146Tail15 winner = winner := by rw [root146Tail15, scan, if_pos (by decide)]
theorem root146Scan14 : scan edge (search edge 3) 4 [859] 859 root146Tail14 winner = winner :=
  (root146Step14).trans root146Scan15
theorem root146Scan13 : scan edge (search edge 3) 4 [859] 859 root146Tail13 winner = winner :=
  (root146Step13).trans root146Scan14
theorem root146Scan12 : scan edge (search edge 3) 4 [859] 859 root146Tail12 winner = winner :=
  (root146Step12).trans root146Scan13
theorem root146Scan11 : scan edge (search edge 3) 4 [859] 859 root146Tail11 winner = winner :=
  (root146Step11).trans root146Scan12
theorem root146Scan10 : scan edge (search edge 3) 4 [859] 859 root146Tail10 winner = winner :=
  (root146Step10).trans root146Scan11
theorem root146Scan9 : scan edge (search edge 3) 4 [859] 859 root146Tail9 winner = winner :=
  (root146Step9).trans root146Scan10
theorem root146Scan8 : scan edge (search edge 3) 4 [859] 859 root146Tail8 winner = winner :=
  (root146Step8).trans root146Scan9
theorem root146Scan7 : scan edge (search edge 3) 4 [859] 859 root146Tail7 winner = winner :=
  (root146Step7).trans root146Scan8
theorem root146Scan6 : scan edge (search edge 3) 4 [859] 859 root146Tail6 winner = winner :=
  (root146Step6).trans root146Scan7
theorem root146Scan5 : scan edge (search edge 3) 4 [859] 859 root146Tail5 winner = winner :=
  (root146Step5).trans root146Scan6
theorem root146Scan4 : scan edge (search edge 3) 4 [859] 859 root146Tail4 winner = winner :=
  (root146Step4).trans root146Scan5
theorem root146Scan3 : scan edge (search edge 3) 4 [859] 859 root146Tail3 winner = winner :=
  (root146Step3).trans root146Scan4
theorem root146Scan2 : scan edge (search edge 3) 4 [859] 859 root146Tail2 winner = winner :=
  (root146Step2).trans root146Scan3
theorem root146Scan1 : scan edge (search edge 3) 4 [859] 859 root146Tail1 winner = winner :=
  (root146Step1).trans root146Scan2
theorem root146Scan0 : scan edge (search edge 3) 4 [859] 859 root146Tail0 winner = winner :=
  (root146Step0).trans root146Scan1
set_option maxRecDepth 4096 in
theorem root146Branch : search edge 4 [859] root146Candidates 859 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [859] 859 root146Tail0 winner = winner
  exact root146Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
