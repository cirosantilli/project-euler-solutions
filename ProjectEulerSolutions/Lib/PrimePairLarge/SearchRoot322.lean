import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root322Candidates : List Nat := Data.row2153
set_option maxRecDepth 8192 in
theorem root322Candidates_checked : candidates edge 5 0 2153 tail323 winner.bound = root322Candidates := by
  apply candidates_eq_row edge 5 0 2153 tail323 root322Candidates winner.bound tail323_ordered
  · decide
  · decide
  · funext q
    rfl
def root322Tail40 : List Nat := []
def root322Tail39 : List Nat := 23189 :: root322Tail40
def root322Tail38 : List Nat := 22871 :: root322Tail39
def root322Tail37 : List Nat := 21569 :: root322Tail38
def root322Tail36 : List Nat := 20981 :: root322Tail37
def root322Tail35 : List Nat := 20477 :: root322Tail36
def root322Tail34 : List Nat := 20219 :: root322Tail35
def root322Tail33 : List Nat := 20123 :: root322Tail34
def root322Tail32 : List Nat := 19541 :: root322Tail33
def root322Tail31 : List Nat := 19463 :: root322Tail32
def root322Tail30 : List Nat := 19301 :: root322Tail31
def root322Tail29 : List Nat := 19157 :: root322Tail30
def root322Tail28 : List Nat := 18911 :: root322Tail29
def root322Tail27 : List Nat := 18041 :: root322Tail28
def root322Tail26 : List Nat := 17579 :: root322Tail27
def root322Tail25 : List Nat := 17321 :: root322Tail26
def root322Tail24 : List Nat := 16103 :: root322Tail25
def root322Tail23 : List Nat := 14621 :: root322Tail24
def root322Tail22 : List Nat := 14207 :: root322Tail23
def root322Tail21 : List Nat := 13997 :: root322Tail22
def root322Tail20 : List Nat := 13781 :: root322Tail21
def root322Tail19 : List Nat := 13001 :: root322Tail20
def root322Tail18 : List Nat := 11897 :: root322Tail19
def root322Tail17 : List Nat := 10853 :: root322Tail18
def root322Tail16 : List Nat := 10301 :: root322Tail17
def root322Tail15 : List Nat := 9839 :: root322Tail16
def root322Tail14 : List Nat := 9209 :: root322Tail15
def root322Tail13 : List Nat := 9137 :: root322Tail14
def root322Tail12 : List Nat := 7151 :: root322Tail13
def root322Tail11 : List Nat := 7121 :: root322Tail12
def root322Tail10 : List Nat := 6869 :: root322Tail11
def root322Tail9 : List Nat := 6863 :: root322Tail10
def root322Tail8 : List Nat := 6143 :: root322Tail9
def root322Tail7 : List Nat := 5351 :: root322Tail8
def root322Tail6 : List Nat := 5273 :: root322Tail7
def root322Tail5 : List Nat := 5237 :: root322Tail6
def root322Tail4 : List Nat := 4967 :: root322Tail5
def root322Tail3 : List Nat := 4397 :: root322Tail4
def root322Tail2 : List Nat := 2969 :: root322Tail3
def root322Tail1 : List Nat := 2663 :: root322Tail2
def root322Tail0 : List Nat := 2417 :: root322Tail1
def root322Candidates0 : List Nat := [17579]
set_option maxRecDepth 8192 in
theorem root322Candidates0_checked : candidates edge 4 2153 2417 root322Tail1 winner.bound = root322Candidates0 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root322Step0 : scan edge (search edge 3) 4 [2153] 2153 root322Tail0 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail1 winner := by
  rw [root322Tail0, scan, if_neg (by decide), root322Candidates0_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail1
    (if 3 ≤ root322Candidates0.length then search edge 3 [2153, 2417] root322Candidates0 4570 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates1 : List Nat := [13997]
set_option maxRecDepth 8192 in
theorem root322Candidates1_checked : candidates edge 4 2153 2663 root322Tail2 winner.bound = root322Candidates1 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root322Step1 : scan edge (search edge 3) 4 [2153] 2153 root322Tail1 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail2 winner := by
  rw [root322Tail1, scan, if_neg (by decide), root322Candidates1_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail2
    (if 3 ≤ root322Candidates1.length then search edge 3 [2153, 2663] root322Candidates1 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root322Candidates2_checked : candidates edge 4 2153 2969 root322Tail3 winner.bound = root322Candidates2 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root322Step2 : scan edge (search edge 3) 4 [2153] 2153 root322Tail2 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail3 winner := by
  rw [root322Tail2, scan, if_neg (by decide), root322Candidates2_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail3
    (if 3 ≤ root322Candidates2.length then search edge 3 [2153, 2969] root322Candidates2 5122 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root322Candidates3_checked : candidates edge 4 2153 4397 root322Tail4 winner.bound = root322Candidates3 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root322Step3 : scan edge (search edge 3) 4 [2153] 2153 root322Tail3 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail4 winner := by
  rw [root322Tail3, scan, if_neg (by decide), root322Candidates3_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail4
    (if 3 ≤ root322Candidates3.length then search edge 3 [2153, 4397] root322Candidates3 6550 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root322Candidates4_checked : candidates edge 4 2153 4967 root322Tail5 winner.bound = root322Candidates4 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root322Step4 : scan edge (search edge 3) 4 [2153] 2153 root322Tail4 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail5 winner := by
  rw [root322Tail4, scan, if_neg (by decide), root322Candidates4_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail5
    (if 3 ≤ root322Candidates4.length then search edge 3 [2153, 4967] root322Candidates4 7120 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates5 : List Nat := [7151]
set_option maxRecDepth 8192 in
theorem root322Candidates5_checked : candidates edge 4 2153 5237 root322Tail6 winner.bound = root322Candidates5 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root322Step5 : scan edge (search edge 3) 4 [2153] 2153 root322Tail5 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail6 winner := by
  rw [root322Tail5, scan, if_neg (by decide), root322Candidates5_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail6
    (if 3 ≤ root322Candidates5.length then search edge 3 [2153, 5237] root322Candidates5 7390 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root322Candidates6_checked : candidates edge 4 2153 5273 root322Tail7 winner.bound = root322Candidates6 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root322Step6 : scan edge (search edge 3) 4 [2153] 2153 root322Tail6 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail7 winner := by
  rw [root322Tail6, scan, if_neg (by decide), root322Candidates6_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail7
    (if 3 ≤ root322Candidates6.length then search edge 3 [2153, 5273] root322Candidates6 7426 winner else winner) = _
  rw [if_neg (by decide)]
def root322Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root322Candidates7_checked : candidates edge 4 2153 5351 root322Tail8 winner.bound = root322Candidates7 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root322Step7 : scan edge (search edge 3) 4 [2153] 2153 root322Tail7 winner =
    scan edge (search edge 3) 4 [2153] 2153 root322Tail8 winner := by
  rw [root322Tail7, scan, if_neg (by decide), root322Candidates7_checked]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail8
    (if 3 ≤ root322Candidates7.length then search edge 3 [2153, 5351] root322Candidates7 7504 winner else winner) = _
  rw [if_neg (by decide)]
theorem root322Scan8 : scan edge (search edge 3) 4 [2153] 2153 root322Tail8 winner = winner := by rw [root322Tail8, scan, if_pos (by decide)]
theorem root322Scan7 : scan edge (search edge 3) 4 [2153] 2153 root322Tail7 winner = winner :=
  (root322Step7).trans root322Scan8
theorem root322Scan6 : scan edge (search edge 3) 4 [2153] 2153 root322Tail6 winner = winner :=
  (root322Step6).trans root322Scan7
theorem root322Scan5 : scan edge (search edge 3) 4 [2153] 2153 root322Tail5 winner = winner :=
  (root322Step5).trans root322Scan6
theorem root322Scan4 : scan edge (search edge 3) 4 [2153] 2153 root322Tail4 winner = winner :=
  (root322Step4).trans root322Scan5
theorem root322Scan3 : scan edge (search edge 3) 4 [2153] 2153 root322Tail3 winner = winner :=
  (root322Step3).trans root322Scan4
theorem root322Scan2 : scan edge (search edge 3) 4 [2153] 2153 root322Tail2 winner = winner :=
  (root322Step2).trans root322Scan3
theorem root322Scan1 : scan edge (search edge 3) 4 [2153] 2153 root322Tail1 winner = winner :=
  (root322Step1).trans root322Scan2
theorem root322Scan0 : scan edge (search edge 3) 4 [2153] 2153 root322Tail0 winner = winner :=
  (root322Step0).trans root322Scan1
set_option maxRecDepth 4096 in
theorem root322Branch : search edge 4 [2153] root322Candidates 2153 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2153] 2153 root322Tail0 winner = winner
  exact root322Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
