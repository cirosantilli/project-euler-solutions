import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root101Candidates : List Nat := Data.row569
set_option maxRecDepth 8192 in
theorem root101Candidates_checked : candidates edge 5 0 569 tail102 winner.bound = root101Candidates := by
  apply candidates_eq_row edge 5 0 569 tail102 root101Candidates winner.bound tail102_ordered
  · decide
  · decide
  · funext q
    rfl
def root101Tail54 : List Nat := []
def root101Tail53 : List Nat := 25409 :: root101Tail54
def root101Tail52 : List Nat := 24821 :: root101Tail53
def root101Tail51 : List Nat := 24683 :: root101Tail52
def root101Tail50 : List Nat := 22937 :: root101Tail51
def root101Tail49 : List Nat := 20399 :: root101Tail50
def root101Tail48 : List Nat := 17321 :: root101Tail49
def root101Tail47 : List Nat := 17093 :: root101Tail48
def root101Tail46 : List Nat := 16553 :: root101Tail47
def root101Tail45 : List Nat := 15809 :: root101Tail46
def root101Tail44 : List Nat := 15737 :: root101Tail45
def root101Tail43 : List Nat := 15581 :: root101Tail44
def root101Tail42 : List Nat := 15401 :: root101Tail43
def root101Tail41 : List Nat := 14939 :: root101Tail42
def root101Tail40 : List Nat := 14831 :: root101Tail41
def root101Tail39 : List Nat := 14411 :: root101Tail40
def root101Tail38 : List Nat := 14369 :: root101Tail39
def root101Tail37 : List Nat := 13469 :: root101Tail38
def root101Tail36 : List Nat := 13187 :: root101Tail37
def root101Tail35 : List Nat := 12227 :: root101Tail36
def root101Tail34 : List Nat := 11831 :: root101Tail35
def root101Tail33 : List Nat := 11681 :: root101Tail34
def root101Tail32 : List Nat := 11633 :: root101Tail33
def root101Tail31 : List Nat := 11471 :: root101Tail32
def root101Tail30 : List Nat := 11423 :: root101Tail31
def root101Tail29 : List Nat := 11243 :: root101Tail30
def root101Tail28 : List Nat := 11159 :: root101Tail29
def root101Tail27 : List Nat := 11117 :: root101Tail28
def root101Tail26 : List Nat := 11093 :: root101Tail27
def root101Tail25 : List Nat := 9839 :: root101Tail26
def root101Tail24 : List Nat := 9371 :: root101Tail25
def root101Tail23 : List Nat := 9311 :: root101Tail24
def root101Tail22 : List Nat := 9203 :: root101Tail23
def root101Tail21 : List Nat := 8933 :: root101Tail22
def root101Tail20 : List Nat := 8831 :: root101Tail21
def root101Tail19 : List Nat := 8219 :: root101Tail20
def root101Tail18 : List Nat := 8039 :: root101Tail19
def root101Tail17 : List Nat := 7673 :: root101Tail18
def root101Tail16 : List Nat := 6959 :: root101Tail17
def root101Tail15 : List Nat := 5741 :: root101Tail16
def root101Tail14 : List Nat := 4919 :: root101Tail15
def root101Tail13 : List Nat := 4673 :: root101Tail14
def root101Tail12 : List Nat := 3617 :: root101Tail13
def root101Tail11 : List Nat := 3203 :: root101Tail12
def root101Tail10 : List Nat := 2861 :: root101Tail11
def root101Tail9 : List Nat := 2699 :: root101Tail10
def root101Tail8 : List Nat := 2657 :: root101Tail9
def root101Tail7 : List Nat := 1601 :: root101Tail8
def root101Tail6 : List Nat := 1487 :: root101Tail7
def root101Tail5 : List Nat := 1151 :: root101Tail6
def root101Tail4 : List Nat := 887 :: root101Tail5
def root101Tail3 : List Nat := 809 :: root101Tail4
def root101Tail2 : List Nat := 797 :: root101Tail3
def root101Tail1 : List Nat := 773 :: root101Tail2
def root101Tail0 : List Nat := 659 :: root101Tail1
def root101Candidates0 : List Nat := [9839]
set_option maxRecDepth 8192 in
theorem root101Candidates0_checked : candidates edge 4 569 659 root101Tail1 winner.bound = root101Candidates0 := by
  unfold candidates
  have row : edge 659 = (SortedNatTree.contains FastRows.row659) := by
    change Data.row659.contains = (SortedNatTree.contains FastRows.row659)
    exact FastRows.row659_checked.symm
  rw [row]
  decide
theorem root101Step0 : scan edge (search edge 3) 4 [569] 569 root101Tail0 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail1 winner := by
  rw [root101Tail0, scan, if_neg (by decide), root101Candidates0_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail1
    (if 3 ≤ root101Candidates0.length then search edge 3 [569, 659] root101Candidates0 1228 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates1 : List Nat := [9311, 11681]
set_option maxRecDepth 8192 in
theorem root101Candidates1_checked : candidates edge 4 569 773 root101Tail2 winner.bound = root101Candidates1 := by
  unfold candidates
  have row : edge 773 = (SortedNatTree.contains FastRows.row773) := by
    change Data.row773.contains = (SortedNatTree.contains FastRows.row773)
    exact FastRows.row773_checked.symm
  rw [row]
  decide
theorem root101Step1 : scan edge (search edge 3) 4 [569] 569 root101Tail1 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail2 winner := by
  rw [root101Tail1, scan, if_neg (by decide), root101Candidates1_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail2
    (if 3 ≤ root101Candidates1.length then search edge 3 [569, 773] root101Candidates1 1342 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates2 : List Nat := [11243]
set_option maxRecDepth 8192 in
theorem root101Candidates2_checked : candidates edge 4 569 797 root101Tail3 winner.bound = root101Candidates2 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
theorem root101Step2 : scan edge (search edge 3) 4 [569] 569 root101Tail2 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail3 winner := by
  rw [root101Tail2, scan, if_neg (by decide), root101Candidates2_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail3
    (if 3 ≤ root101Candidates2.length then search edge 3 [569, 797] root101Candidates2 1366 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root101Candidates3_checked : candidates edge 4 569 809 root101Tail4 winner.bound = root101Candidates3 := by
  unfold candidates
  have row : edge 809 = (SortedNatTree.contains FastRows.row809) := by
    change Data.row809.contains = (SortedNatTree.contains FastRows.row809)
    exact FastRows.row809_checked.symm
  rw [row]
  decide
theorem root101Step3 : scan edge (search edge 3) 4 [569] 569 root101Tail3 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail4 winner := by
  rw [root101Tail3, scan, if_neg (by decide), root101Candidates3_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail4
    (if 3 ≤ root101Candidates3.length then search edge 3 [569, 809] root101Candidates3 1378 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates4 : List Nat := [15581]
set_option maxRecDepth 8192 in
theorem root101Candidates4_checked : candidates edge 4 569 887 root101Tail5 winner.bound = root101Candidates4 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
theorem root101Step4 : scan edge (search edge 3) 4 [569] 569 root101Tail4 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail5 winner := by
  rw [root101Tail4, scan, if_neg (by decide), root101Candidates4_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail5
    (if 3 ≤ root101Candidates4.length then search edge 3 [569, 887] root101Candidates4 1456 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates5 : List Nat := [22937]
set_option maxRecDepth 8192 in
theorem root101Candidates5_checked : candidates edge 4 569 1151 root101Tail6 winner.bound = root101Candidates5 := by
  unfold candidates
  have row : edge 1151 = (SortedNatTree.contains FastRows.row1151) := by
    change Data.row1151.contains = (SortedNatTree.contains FastRows.row1151)
    exact FastRows.row1151_checked.symm
  rw [row]
  decide
theorem root101Step5 : scan edge (search edge 3) 4 [569] 569 root101Tail5 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail6 winner := by
  rw [root101Tail5, scan, if_neg (by decide), root101Candidates5_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail6
    (if 3 ≤ root101Candidates5.length then search edge 3 [569, 1151] root101Candidates5 1720 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates6 : List Nat := [1601]
set_option maxRecDepth 8192 in
theorem root101Candidates6_checked : candidates edge 4 569 1487 root101Tail7 winner.bound = root101Candidates6 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
theorem root101Step6 : scan edge (search edge 3) 4 [569] 569 root101Tail6 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail7 winner := by
  rw [root101Tail6, scan, if_neg (by decide), root101Candidates6_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail7
    (if 3 ≤ root101Candidates6.length then search edge 3 [569, 1487] root101Candidates6 2056 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates7 : List Nat := [9203, 14369]
set_option maxRecDepth 8192 in
theorem root101Candidates7_checked : candidates edge 4 569 1601 root101Tail8 winner.bound = root101Candidates7 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root101Step7 : scan edge (search edge 3) 4 [569] 569 root101Tail7 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail8 winner := by
  rw [root101Tail7, scan, if_neg (by decide), root101Candidates7_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail8
    (if 3 ≤ root101Candidates7.length then search edge 3 [569, 1601] root101Candidates7 2170 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root101Candidates8_checked : candidates edge 4 569 2657 root101Tail9 winner.bound = root101Candidates8 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root101Step8 : scan edge (search edge 3) 4 [569] 569 root101Tail8 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail9 winner := by
  rw [root101Tail8, scan, if_neg (by decide), root101Candidates8_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail9
    (if 3 ≤ root101Candidates8.length then search edge 3 [569, 2657] root101Candidates8 3226 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates9 : List Nat := [8039, 9839]
set_option maxRecDepth 8192 in
theorem root101Candidates9_checked : candidates edge 4 569 2699 root101Tail10 winner.bound = root101Candidates9 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
theorem root101Step9 : scan edge (search edge 3) 4 [569] 569 root101Tail9 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail10 winner := by
  rw [root101Tail9, scan, if_neg (by decide), root101Candidates9_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail10
    (if 3 ≤ root101Candidates9.length then search edge 3 [569, 2699] root101Candidates9 3268 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates10 : List Nat := [4919, 11159, 11831]
set_option maxRecDepth 8192 in
theorem root101Candidates10_checked : candidates edge 4 569 2861 root101Tail11 winner.bound = root101Candidates10 := by
  unfold candidates
  have row : edge 2861 = (SortedNatTree.contains FastRows.row2861) := by
    change Data.row2861.contains = (SortedNatTree.contains FastRows.row2861)
    exact FastRows.row2861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root101Child10 : search edge 3 [569, 2861] root101Candidates10 3430 winner = winner := by decide
theorem root101Step10 : scan edge (search edge 3) 4 [569] 569 root101Tail10 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail11 winner := by
  rw [root101Tail10, scan, if_neg (by decide), root101Candidates10_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail11
    (if 3 ≤ root101Candidates10.length then search edge 3 [569, 2861] root101Candidates10 3430 winner else winner) = _
  rw [if_pos (by decide), root101Child10]
def root101Candidates11 : List Nat := [8219, 8933, 9311]
set_option maxRecDepth 8192 in
theorem root101Candidates11_checked : candidates edge 4 569 3203 root101Tail12 winner.bound = root101Candidates11 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root101Child11 : search edge 3 [569, 3203] root101Candidates11 3772 winner = winner := by decide
theorem root101Step11 : scan edge (search edge 3) 4 [569] 569 root101Tail11 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail12 winner := by
  rw [root101Tail11, scan, if_neg (by decide), root101Candidates11_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail12
    (if 3 ≤ root101Candidates11.length then search edge 3 [569, 3203] root101Candidates11 3772 winner else winner) = _
  rw [if_pos (by decide), root101Child11]
def root101Candidates12 : List Nat := [11633]
set_option maxRecDepth 8192 in
theorem root101Candidates12_checked : candidates edge 4 569 3617 root101Tail13 winner.bound = root101Candidates12 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root101Step12 : scan edge (search edge 3) 4 [569] 569 root101Tail12 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail13 winner := by
  rw [root101Tail12, scan, if_neg (by decide), root101Candidates12_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail13
    (if 3 ≤ root101Candidates12.length then search edge 3 [569, 3617] root101Candidates12 4186 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root101Candidates13_checked : candidates edge 4 569 4673 root101Tail14 winner.bound = root101Candidates13 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root101Step13 : scan edge (search edge 3) 4 [569] 569 root101Tail13 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail14 winner := by
  rw [root101Tail13, scan, if_neg (by decide), root101Candidates13_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail14
    (if 3 ≤ root101Candidates13.length then search edge 3 [569, 4673] root101Candidates13 5242 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates14 : List Nat := [9203, 11093]
set_option maxRecDepth 8192 in
theorem root101Candidates14_checked : candidates edge 4 569 4919 root101Tail15 winner.bound = root101Candidates14 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root101Step14 : scan edge (search edge 3) 4 [569] 569 root101Tail14 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail15 winner := by
  rw [root101Tail14, scan, if_neg (by decide), root101Candidates14_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail15
    (if 3 ≤ root101Candidates14.length then search edge 3 [569, 4919] root101Candidates14 5488 winner else winner) = _
  rw [if_neg (by decide)]
def root101Candidates15 : List Nat := [6959, 8219, 11633]
set_option maxRecDepth 8192 in
theorem root101Candidates15_checked : candidates edge 4 569 5741 root101Tail16 winner.bound = root101Candidates15 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root101Child15 : search edge 3 [569, 5741] root101Candidates15 6310 winner = winner := by decide
theorem root101Step15 : scan edge (search edge 3) 4 [569] 569 root101Tail15 winner =
    scan edge (search edge 3) 4 [569] 569 root101Tail16 winner := by
  rw [root101Tail15, scan, if_neg (by decide), root101Candidates15_checked]
  change scan edge (search edge 3) 4 [569] 569 root101Tail16
    (if 3 ≤ root101Candidates15.length then search edge 3 [569, 5741] root101Candidates15 6310 winner else winner) = _
  rw [if_pos (by decide), root101Child15]
theorem root101Scan16 : scan edge (search edge 3) 4 [569] 569 root101Tail16 winner = winner := by rw [root101Tail16, scan, if_pos (by decide)]
theorem root101Scan15 : scan edge (search edge 3) 4 [569] 569 root101Tail15 winner = winner :=
  (root101Step15).trans root101Scan16
theorem root101Scan14 : scan edge (search edge 3) 4 [569] 569 root101Tail14 winner = winner :=
  (root101Step14).trans root101Scan15
theorem root101Scan13 : scan edge (search edge 3) 4 [569] 569 root101Tail13 winner = winner :=
  (root101Step13).trans root101Scan14
theorem root101Scan12 : scan edge (search edge 3) 4 [569] 569 root101Tail12 winner = winner :=
  (root101Step12).trans root101Scan13
theorem root101Scan11 : scan edge (search edge 3) 4 [569] 569 root101Tail11 winner = winner :=
  (root101Step11).trans root101Scan12
theorem root101Scan10 : scan edge (search edge 3) 4 [569] 569 root101Tail10 winner = winner :=
  (root101Step10).trans root101Scan11
theorem root101Scan9 : scan edge (search edge 3) 4 [569] 569 root101Tail9 winner = winner :=
  (root101Step9).trans root101Scan10
theorem root101Scan8 : scan edge (search edge 3) 4 [569] 569 root101Tail8 winner = winner :=
  (root101Step8).trans root101Scan9
theorem root101Scan7 : scan edge (search edge 3) 4 [569] 569 root101Tail7 winner = winner :=
  (root101Step7).trans root101Scan8
theorem root101Scan6 : scan edge (search edge 3) 4 [569] 569 root101Tail6 winner = winner :=
  (root101Step6).trans root101Scan7
theorem root101Scan5 : scan edge (search edge 3) 4 [569] 569 root101Tail5 winner = winner :=
  (root101Step5).trans root101Scan6
theorem root101Scan4 : scan edge (search edge 3) 4 [569] 569 root101Tail4 winner = winner :=
  (root101Step4).trans root101Scan5
theorem root101Scan3 : scan edge (search edge 3) 4 [569] 569 root101Tail3 winner = winner :=
  (root101Step3).trans root101Scan4
theorem root101Scan2 : scan edge (search edge 3) 4 [569] 569 root101Tail2 winner = winner :=
  (root101Step2).trans root101Scan3
theorem root101Scan1 : scan edge (search edge 3) 4 [569] 569 root101Tail1 winner = winner :=
  (root101Step1).trans root101Scan2
theorem root101Scan0 : scan edge (search edge 3) 4 [569] 569 root101Tail0 winner = winner :=
  (root101Step0).trans root101Scan1
set_option maxRecDepth 4096 in
theorem root101Branch : search edge 4 [569] root101Candidates 569 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [569] 569 root101Tail0 winner = winner
  exact root101Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
