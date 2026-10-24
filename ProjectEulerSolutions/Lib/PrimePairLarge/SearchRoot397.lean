import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root397Candidates : List Nat := Data.row2741
set_option maxRecDepth 8192 in
theorem root397Candidates_checked : candidates edge 5 0 2741 tail398 winner.bound = root397Candidates := by
  apply candidates_eq_row edge 5 0 2741 tail398 root397Candidates winner.bound tail398_ordered
  · decide
  · decide
  · funext q
    rfl
def root397Tail46 : List Nat := []
def root397Tail45 : List Nat := 22271 :: root397Tail46
def root397Tail44 : List Nat := 22193 :: root397Tail45
def root397Tail43 : List Nat := 21563 :: root397Tail44
def root397Tail42 : List Nat := 21017 :: root397Tail43
def root397Tail41 : List Nat := 20921 :: root397Tail42
def root397Tail40 : List Nat := 20897 :: root397Tail41
def root397Tail39 : List Nat := 20681 :: root397Tail40
def root397Tail38 : List Nat := 20123 :: root397Tail39
def root397Tail37 : List Nat := 18371 :: root397Tail38
def root397Tail36 : List Nat := 18287 :: root397Tail37
def root397Tail35 : List Nat := 17183 :: root397Tail36
def root397Tail34 : List Nat := 16433 :: root397Tail35
def root397Tail33 : List Nat := 15131 :: root397Tail34
def root397Tail32 : List Nat := 14549 :: root397Tail33
def root397Tail31 : List Nat := 14411 :: root397Tail32
def root397Tail30 : List Nat := 13799 :: root397Tail31
def root397Tail29 : List Nat := 13709 :: root397Tail30
def root397Tail28 : List Nat := 12899 :: root397Tail29
def root397Tail27 : List Nat := 12491 :: root397Tail28
def root397Tail26 : List Nat := 12071 :: root397Tail27
def root397Tail25 : List Nat := 11597 :: root397Tail26
def root397Tail24 : List Nat := 10799 :: root397Tail25
def root397Tail23 : List Nat := 10589 :: root397Tail24
def root397Tail22 : List Nat := 9851 :: root397Tail23
def root397Tail21 : List Nat := 9467 :: root397Tail22
def root397Tail20 : List Nat := 8747 :: root397Tail21
def root397Tail19 : List Nat := 8537 :: root397Tail20
def root397Tail18 : List Nat := 7757 :: root397Tail19
def root397Tail17 : List Nat := 7457 :: root397Tail18
def root397Tail16 : List Nat := 7331 :: root397Tail17
def root397Tail15 : List Nat := 6911 :: root397Tail16
def root397Tail14 : List Nat := 6899 :: root397Tail15
def root397Tail13 : List Nat := 6737 :: root397Tail14
def root397Tail12 : List Nat := 6689 :: root397Tail13
def root397Tail11 : List Nat := 6353 :: root397Tail12
def root397Tail10 : List Nat := 6113 :: root397Tail11
def root397Tail9 : List Nat := 4799 :: root397Tail10
def root397Tail8 : List Nat := 4649 :: root397Tail9
def root397Tail7 : List Nat := 4253 :: root397Tail8
def root397Tail6 : List Nat := 4133 :: root397Tail7
def root397Tail5 : List Nat := 3917 :: root397Tail6
def root397Tail4 : List Nat := 3389 :: root397Tail5
def root397Tail3 : List Nat := 3329 :: root397Tail4
def root397Tail2 : List Nat := 3119 :: root397Tail3
def root397Tail1 : List Nat := 2963 :: root397Tail2
def root397Tail0 : List Nat := 2837 :: root397Tail1
def root397Candidates0 : List Nat := [7757]
set_option maxRecDepth 8192 in
theorem root397Candidates0_checked : candidates edge 4 2741 2837 root397Tail1 winner.bound = root397Candidates0 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
theorem root397Step0 : scan edge (search edge 3) 4 [2741] 2741 root397Tail0 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail1 winner := by
  rw [root397Tail0, scan, if_neg (by decide), root397Candidates0_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail1
    (if 3 ≤ root397Candidates0.length then search edge 3 [2741, 2837] root397Candidates0 5578 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates1 : List Nat := [6689, 9851]
set_option maxRecDepth 8192 in
theorem root397Candidates1_checked : candidates edge 4 2741 2963 root397Tail2 winner.bound = root397Candidates1 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root397Step1 : scan edge (search edge 3) 4 [2741] 2741 root397Tail1 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail2 winner := by
  rw [root397Tail1, scan, if_neg (by decide), root397Candidates1_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail2
    (if 3 ≤ root397Candidates1.length then search edge 3 [2741, 2963] root397Candidates1 5704 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates2 : List Nat := [16433]
set_option maxRecDepth 8192 in
theorem root397Candidates2_checked : candidates edge 4 2741 3119 root397Tail3 winner.bound = root397Candidates2 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
theorem root397Step2 : scan edge (search edge 3) 4 [2741] 2741 root397Tail2 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail3 winner := by
  rw [root397Tail2, scan, if_neg (by decide), root397Candidates2_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail3
    (if 3 ≤ root397Candidates2.length then search edge 3 [2741, 3119] root397Candidates2 5860 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates3 : List Nat := [4649, 12491]
set_option maxRecDepth 8192 in
theorem root397Candidates3_checked : candidates edge 4 2741 3329 root397Tail4 winner.bound = root397Candidates3 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root397Step3 : scan edge (search edge 3) 4 [2741] 2741 root397Tail3 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail4 winner := by
  rw [root397Tail3, scan, if_neg (by decide), root397Candidates3_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail4
    (if 3 ≤ root397Candidates3.length then search edge 3 [2741, 3329] root397Candidates3 6070 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates4 : List Nat := [4799, 14549]
set_option maxRecDepth 8192 in
theorem root397Candidates4_checked : candidates edge 4 2741 3389 root397Tail5 winner.bound = root397Candidates4 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
theorem root397Step4 : scan edge (search edge 3) 4 [2741] 2741 root397Tail4 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail5 winner := by
  rw [root397Tail4, scan, if_neg (by decide), root397Candidates4_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail5
    (if 3 ≤ root397Candidates4.length then search edge 3 [2741, 3389] root397Candidates4 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates5 : List Nat := [4649, 6899, 7331]
set_option maxRecDepth 8192 in
theorem root397Candidates5_checked : candidates edge 4 2741 3917 root397Tail6 winner.bound = root397Candidates5 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root397Child5 : search edge 3 [2741, 3917] root397Candidates5 6658 winner = winner := by decide
theorem root397Step5 : scan edge (search edge 3) 4 [2741] 2741 root397Tail5 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail6 winner := by
  rw [root397Tail5, scan, if_neg (by decide), root397Candidates5_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail6
    (if 3 ≤ root397Candidates5.length then search edge 3 [2741, 3917] root397Candidates5 6658 winner else winner) = _
  rw [if_pos (by decide), root397Child5]
def root397Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root397Candidates6_checked : candidates edge 4 2741 4133 root397Tail7 winner.bound = root397Candidates6 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root397Step6 : scan edge (search edge 3) 4 [2741] 2741 root397Tail6 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail7 winner := by
  rw [root397Tail6, scan, if_neg (by decide), root397Candidates6_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail7
    (if 3 ≤ root397Candidates6.length then search edge 3 [2741, 4133] root397Candidates6 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root397Candidates7_checked : candidates edge 4 2741 4253 root397Tail8 winner.bound = root397Candidates7 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root397Step7 : scan edge (search edge 3) 4 [2741] 2741 root397Tail7 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail8 winner := by
  rw [root397Tail7, scan, if_neg (by decide), root397Candidates7_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail8
    (if 3 ≤ root397Candidates7.length then search edge 3 [2741, 4253] root397Candidates7 6994 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates8 : List Nat := [8537]
set_option maxRecDepth 8192 in
theorem root397Candidates8_checked : candidates edge 4 2741 4649 root397Tail9 winner.bound = root397Candidates8 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root397Step8 : scan edge (search edge 3) 4 [2741] 2741 root397Tail8 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail9 winner := by
  rw [root397Tail8, scan, if_neg (by decide), root397Candidates8_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail9
    (if 3 ≤ root397Candidates8.length then search edge 3 [2741, 4649] root397Candidates8 7390 winner else winner) = _
  rw [if_neg (by decide)]
def root397Candidates9 : List Nat := [18287]
set_option maxRecDepth 8192 in
theorem root397Candidates9_checked : candidates edge 4 2741 4799 root397Tail10 winner.bound = root397Candidates9 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root397Step9 : scan edge (search edge 3) 4 [2741] 2741 root397Tail9 winner =
    scan edge (search edge 3) 4 [2741] 2741 root397Tail10 winner := by
  rw [root397Tail9, scan, if_neg (by decide), root397Candidates9_checked]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail10
    (if 3 ≤ root397Candidates9.length then search edge 3 [2741, 4799] root397Candidates9 7540 winner else winner) = _
  rw [if_neg (by decide)]
theorem root397Scan10 : scan edge (search edge 3) 4 [2741] 2741 root397Tail10 winner = winner := by rw [root397Tail10, scan, if_pos (by decide)]
theorem root397Scan9 : scan edge (search edge 3) 4 [2741] 2741 root397Tail9 winner = winner :=
  (root397Step9).trans root397Scan10
theorem root397Scan8 : scan edge (search edge 3) 4 [2741] 2741 root397Tail8 winner = winner :=
  (root397Step8).trans root397Scan9
theorem root397Scan7 : scan edge (search edge 3) 4 [2741] 2741 root397Tail7 winner = winner :=
  (root397Step7).trans root397Scan8
theorem root397Scan6 : scan edge (search edge 3) 4 [2741] 2741 root397Tail6 winner = winner :=
  (root397Step6).trans root397Scan7
theorem root397Scan5 : scan edge (search edge 3) 4 [2741] 2741 root397Tail5 winner = winner :=
  (root397Step5).trans root397Scan6
theorem root397Scan4 : scan edge (search edge 3) 4 [2741] 2741 root397Tail4 winner = winner :=
  (root397Step4).trans root397Scan5
theorem root397Scan3 : scan edge (search edge 3) 4 [2741] 2741 root397Tail3 winner = winner :=
  (root397Step3).trans root397Scan4
theorem root397Scan2 : scan edge (search edge 3) 4 [2741] 2741 root397Tail2 winner = winner :=
  (root397Step2).trans root397Scan3
theorem root397Scan1 : scan edge (search edge 3) 4 [2741] 2741 root397Tail1 winner = winner :=
  (root397Step1).trans root397Scan2
theorem root397Scan0 : scan edge (search edge 3) 4 [2741] 2741 root397Tail0 winner = winner :=
  (root397Step0).trans root397Scan1
set_option maxRecDepth 4096 in
theorem root397Branch : search edge 4 [2741] root397Candidates 2741 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2741] 2741 root397Tail0 winner = winner
  exact root397Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
