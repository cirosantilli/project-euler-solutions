import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root221Candidates : List Nat := Data.row1423
set_option maxRecDepth 8192 in
theorem root221Candidates_checked : candidates edge 5 0 1423 tail222 winner.bound = root221Candidates := by
  apply candidates_eq_row edge 5 0 1423 tail222 root221Candidates winner.bound tail222_ordered
  · decide
  · decide
  · funext q
    rfl
def root221Tail46 : List Nat := []
def root221Tail45 : List Nat := 24121 :: root221Tail46
def root221Tail44 : List Nat := 24019 :: root221Tail45
def root221Tail43 : List Nat := 23911 :: root221Tail44
def root221Tail42 : List Nat := 23029 :: root221Tail43
def root221Tail41 : List Nat := 22651 :: root221Tail42
def root221Tail40 : List Nat := 21139 :: root221Tail41
def root221Tail39 : List Nat := 20809 :: root221Tail40
def root221Tail38 : List Nat := 20749 :: root221Tail39
def root221Tail37 : List Nat := 20407 :: root221Tail38
def root221Tail36 : List Nat := 20047 :: root221Tail37
def root221Tail35 : List Nat := 19753 :: root221Tail36
def root221Tail34 : List Nat := 19483 :: root221Tail35
def root221Tail33 : List Nat := 18661 :: root221Tail34
def root221Tail32 : List Nat := 17389 :: root221Tail33
def root221Tail31 : List Nat := 16189 :: root221Tail32
def root221Tail30 : List Nat := 15307 :: root221Tail31
def root221Tail29 : List Nat := 14929 :: root221Tail30
def root221Tail28 : List Nat := 14083 :: root221Tail29
def root221Tail27 : List Nat := 13441 :: root221Tail28
def root221Tail26 : List Nat := 13399 :: root221Tail27
def root221Tail25 : List Nat := 13099 :: root221Tail26
def root221Tail24 : List Nat := 12721 :: root221Tail25
def root221Tail23 : List Nat := 11677 :: root221Tail24
def root221Tail22 : List Nat := 10939 :: root221Tail23
def root221Tail21 : List Nat := 9769 :: root221Tail22
def root221Tail20 : List Nat := 9343 :: root221Tail21
def root221Tail19 : List Nat := 9157 :: root221Tail20
def root221Tail18 : List Nat := 8839 :: root221Tail19
def root221Tail17 : List Nat := 8017 :: root221Tail18
def root221Tail16 : List Nat := 6427 :: root221Tail17
def root221Tail15 : List Nat := 6367 :: root221Tail16
def root221Tail14 : List Nat := 6337 :: root221Tail15
def root221Tail13 : List Nat := 5821 :: root221Tail14
def root221Tail12 : List Nat := 5791 :: root221Tail13
def root221Tail11 : List Nat := 5521 :: root221Tail12
def root221Tail10 : List Nat := 4057 :: root221Tail11
def root221Tail9 : List Nat := 4027 :: root221Tail10
def root221Tail8 : List Nat := 3253 :: root221Tail9
def root221Tail7 : List Nat := 3187 :: root221Tail8
def root221Tail6 : List Nat := 2797 :: root221Tail7
def root221Tail5 : List Nat := 2389 :: root221Tail6
def root221Tail4 : List Nat := 2287 :: root221Tail5
def root221Tail3 : List Nat := 2137 :: root221Tail4
def root221Tail2 : List Nat := 2017 :: root221Tail3
def root221Tail1 : List Nat := 1999 :: root221Tail2
def root221Tail0 : List Nat := 1699 :: root221Tail1
def root221Candidates0 : List Nat := [5791, 13099, 13399]
set_option maxRecDepth 8192 in
theorem root221Candidates0_checked : candidates edge 4 1423 1699 root221Tail1 winner.bound = root221Candidates0 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root221Child0 : search edge 3 [1423, 1699] root221Candidates0 3122 winner = winner := by decide
theorem root221Step0 : scan edge (search edge 3) 4 [1423] 1423 root221Tail0 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail1 winner := by
  rw [root221Tail0, scan, if_neg (by decide), root221Candidates0_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail1
    (if 3 ≤ root221Candidates0.length then search edge 3 [1423, 1699] root221Candidates0 3122 winner else winner) = _
  rw [if_pos (by decide), root221Child0]
def root221Candidates1 : List Nat := [19483]
set_option maxRecDepth 8192 in
theorem root221Candidates1_checked : candidates edge 4 1423 1999 root221Tail2 winner.bound = root221Candidates1 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root221Step1 : scan edge (search edge 3) 4 [1423] 1423 root221Tail1 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail2 winner := by
  rw [root221Tail1, scan, if_neg (by decide), root221Candidates1_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail2
    (if 3 ≤ root221Candidates1.length then search edge 3 [1423, 1999] root221Candidates1 3422 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates2 : List Nat := [19753, 20047]
set_option maxRecDepth 8192 in
theorem root221Candidates2_checked : candidates edge 4 1423 2017 root221Tail3 winner.bound = root221Candidates2 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root221Step2 : scan edge (search edge 3) 4 [1423] 1423 root221Tail2 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail3 winner := by
  rw [root221Tail2, scan, if_neg (by decide), root221Candidates2_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail3
    (if 3 ≤ root221Candidates2.length then search edge 3 [1423, 2017] root221Candidates2 3440 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates3 : List Nat := [9157, 20407]
set_option maxRecDepth 8192 in
theorem root221Candidates3_checked : candidates edge 4 1423 2137 root221Tail4 winner.bound = root221Candidates3 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
theorem root221Step3 : scan edge (search edge 3) 4 [1423] 1423 root221Tail3 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail4 winner := by
  rw [root221Tail3, scan, if_neg (by decide), root221Candidates3_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail4
    (if 3 ≤ root221Candidates3.length then search edge 3 [1423, 2137] root221Candidates3 3560 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates4 : List Nat := [3187]
set_option maxRecDepth 8192 in
theorem root221Candidates4_checked : candidates edge 4 1423 2287 root221Tail5 winner.bound = root221Candidates4 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
theorem root221Step4 : scan edge (search edge 3) 4 [1423] 1423 root221Tail4 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail5 winner := by
  rw [root221Tail4, scan, if_neg (by decide), root221Candidates4_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail5
    (if 3 ≤ root221Candidates4.length then search edge 3 [1423, 2287] root221Candidates4 3710 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates5 : List Nat := [6427]
set_option maxRecDepth 8192 in
theorem root221Candidates5_checked : candidates edge 4 1423 2389 root221Tail6 winner.bound = root221Candidates5 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
theorem root221Step5 : scan edge (search edge 3) 4 [1423] 1423 root221Tail5 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail6 winner := by
  rw [root221Tail5, scan, if_neg (by decide), root221Candidates5_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail6
    (if 3 ≤ root221Candidates5.length then search edge 3 [1423, 2389] root221Candidates5 3812 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates6 : List Nat := [6367, 8017]
set_option maxRecDepth 8192 in
theorem root221Candidates6_checked : candidates edge 4 1423 2797 root221Tail7 winner.bound = root221Candidates6 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root221Step6 : scan edge (search edge 3) 4 [1423] 1423 root221Tail6 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail7 winner := by
  rw [root221Tail6, scan, if_neg (by decide), root221Candidates6_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail7
    (if 3 ≤ root221Candidates6.length then search edge 3 [1423, 2797] root221Candidates6 4220 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates7 : List Nat := [8017]
set_option maxRecDepth 8192 in
theorem root221Candidates7_checked : candidates edge 4 1423 3187 root221Tail8 winner.bound = root221Candidates7 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root221Step7 : scan edge (search edge 3) 4 [1423] 1423 root221Tail7 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail8 winner := by
  rw [root221Tail7, scan, if_neg (by decide), root221Candidates7_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail8
    (if 3 ≤ root221Candidates7.length then search edge 3 [1423, 3187] root221Candidates7 4610 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root221Candidates8_checked : candidates edge 4 1423 3253 root221Tail9 winner.bound = root221Candidates8 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root221Step8 : scan edge (search edge 3) 4 [1423] 1423 root221Tail8 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail9 winner := by
  rw [root221Tail8, scan, if_neg (by decide), root221Candidates8_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail9
    (if 3 ≤ root221Candidates8.length then search edge 3 [1423, 3253] root221Candidates8 4676 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates9 : List Nat := [9157, 14083, 14929]
set_option maxRecDepth 8192 in
theorem root221Candidates9_checked : candidates edge 4 1423 4027 root221Tail10 winner.bound = root221Candidates9 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root221Child9 : search edge 3 [1423, 4027] root221Candidates9 5450 winner = winner := by decide
theorem root221Step9 : scan edge (search edge 3) 4 [1423] 1423 root221Tail9 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail10 winner := by
  rw [root221Tail9, scan, if_neg (by decide), root221Candidates9_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail10
    (if 3 ≤ root221Candidates9.length then search edge 3 [1423, 4027] root221Candidates9 5450 winner else winner) = _
  rw [if_pos (by decide), root221Child9]
def root221Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root221Candidates10_checked : candidates edge 4 1423 4057 root221Tail11 winner.bound = root221Candidates10 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root221Step10 : scan edge (search edge 3) 4 [1423] 1423 root221Tail10 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail11 winner := by
  rw [root221Tail10, scan, if_neg (by decide), root221Candidates10_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail11
    (if 3 ≤ root221Candidates10.length then search edge 3 [1423, 4057] root221Candidates10 5480 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates11 : List Nat := [12721]
set_option maxRecDepth 8192 in
theorem root221Candidates11_checked : candidates edge 4 1423 5521 root221Tail12 winner.bound = root221Candidates11 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root221Step11 : scan edge (search edge 3) 4 [1423] 1423 root221Tail11 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail12 winner := by
  rw [root221Tail11, scan, if_neg (by decide), root221Candidates11_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail12
    (if 3 ≤ root221Candidates11.length then search edge 3 [1423, 5521] root221Candidates11 6944 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates12 : List Nat := [16189]
set_option maxRecDepth 8192 in
theorem root221Candidates12_checked : candidates edge 4 1423 5791 root221Tail13 winner.bound = root221Candidates12 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root221Step12 : scan edge (search edge 3) 4 [1423] 1423 root221Tail12 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail13 winner := by
  rw [root221Tail12, scan, if_neg (by decide), root221Candidates12_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail13
    (if 3 ≤ root221Candidates12.length then search edge 3 [1423, 5791] root221Candidates12 7214 winner else winner) = _
  rw [if_neg (by decide)]
def root221Candidates13 : List Nat := [12721]
set_option maxRecDepth 8192 in
theorem root221Candidates13_checked : candidates edge 4 1423 5821 root221Tail14 winner.bound = root221Candidates13 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root221Step13 : scan edge (search edge 3) 4 [1423] 1423 root221Tail13 winner =
    scan edge (search edge 3) 4 [1423] 1423 root221Tail14 winner := by
  rw [root221Tail13, scan, if_neg (by decide), root221Candidates13_checked]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail14
    (if 3 ≤ root221Candidates13.length then search edge 3 [1423, 5821] root221Candidates13 7244 winner else winner) = _
  rw [if_neg (by decide)]
theorem root221Scan14 : scan edge (search edge 3) 4 [1423] 1423 root221Tail14 winner = winner := by rw [root221Tail14, scan, if_pos (by decide)]
theorem root221Scan13 : scan edge (search edge 3) 4 [1423] 1423 root221Tail13 winner = winner :=
  (root221Step13).trans root221Scan14
theorem root221Scan12 : scan edge (search edge 3) 4 [1423] 1423 root221Tail12 winner = winner :=
  (root221Step12).trans root221Scan13
theorem root221Scan11 : scan edge (search edge 3) 4 [1423] 1423 root221Tail11 winner = winner :=
  (root221Step11).trans root221Scan12
theorem root221Scan10 : scan edge (search edge 3) 4 [1423] 1423 root221Tail10 winner = winner :=
  (root221Step10).trans root221Scan11
theorem root221Scan9 : scan edge (search edge 3) 4 [1423] 1423 root221Tail9 winner = winner :=
  (root221Step9).trans root221Scan10
theorem root221Scan8 : scan edge (search edge 3) 4 [1423] 1423 root221Tail8 winner = winner :=
  (root221Step8).trans root221Scan9
theorem root221Scan7 : scan edge (search edge 3) 4 [1423] 1423 root221Tail7 winner = winner :=
  (root221Step7).trans root221Scan8
theorem root221Scan6 : scan edge (search edge 3) 4 [1423] 1423 root221Tail6 winner = winner :=
  (root221Step6).trans root221Scan7
theorem root221Scan5 : scan edge (search edge 3) 4 [1423] 1423 root221Tail5 winner = winner :=
  (root221Step5).trans root221Scan6
theorem root221Scan4 : scan edge (search edge 3) 4 [1423] 1423 root221Tail4 winner = winner :=
  (root221Step4).trans root221Scan5
theorem root221Scan3 : scan edge (search edge 3) 4 [1423] 1423 root221Tail3 winner = winner :=
  (root221Step3).trans root221Scan4
theorem root221Scan2 : scan edge (search edge 3) 4 [1423] 1423 root221Tail2 winner = winner :=
  (root221Step2).trans root221Scan3
theorem root221Scan1 : scan edge (search edge 3) 4 [1423] 1423 root221Tail1 winner = winner :=
  (root221Step1).trans root221Scan2
theorem root221Scan0 : scan edge (search edge 3) 4 [1423] 1423 root221Tail0 winner = winner :=
  (root221Step0).trans root221Scan1
set_option maxRecDepth 4096 in
theorem root221Branch : search edge 4 [1423] root221Candidates 1423 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1423] 1423 root221Tail0 winner = winner
  exact root221Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
