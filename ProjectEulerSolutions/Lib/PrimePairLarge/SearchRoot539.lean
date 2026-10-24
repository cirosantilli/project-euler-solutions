import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root539Candidates : List Nat := Data.row3917
set_option maxRecDepth 8192 in
theorem root539Candidates_checked : candidates edge 5 0 3917 tail540 winner.bound = root539Candidates := by
  apply candidates_eq_row edge 5 0 3917 tail540 root539Candidates winner.bound tail540_ordered
  · decide
  · decide
  · funext q
    rfl
def root539Tail30 : List Nat := []
def root539Tail29 : List Nat := 20183 :: root539Tail30
def root539Tail28 : List Nat := 18869 :: root539Tail29
def root539Tail27 : List Nat := 18443 :: root539Tail28
def root539Tail26 : List Nat := 17609 :: root539Tail27
def root539Tail25 : List Nat := 17231 :: root539Tail26
def root539Tail24 : List Nat := 15671 :: root539Tail25
def root539Tail23 : List Nat := 14249 :: root539Tail24
def root539Tail22 : List Nat := 13457 :: root539Tail23
def root539Tail21 : List Nat := 13163 :: root539Tail22
def root539Tail20 : List Nat := 12791 :: root539Tail21
def root539Tail19 : List Nat := 12329 :: root539Tail20
def root539Tail18 : List Nat := 12107 :: root539Tail19
def root539Tail17 : List Nat := 10181 :: root539Tail18
def root539Tail16 : List Nat := 8753 :: root539Tail17
def root539Tail15 : List Nat := 8543 :: root539Tail16
def root539Tail14 : List Nat := 7499 :: root539Tail15
def root539Tail13 : List Nat := 7331 :: root539Tail14
def root539Tail12 : List Nat := 6899 :: root539Tail13
def root539Tail11 : List Nat := 6779 :: root539Tail12
def root539Tail10 : List Nat := 6323 :: root539Tail11
def root539Tail9 : List Nat := 5987 :: root539Tail10
def root539Tail8 : List Nat := 5483 :: root539Tail9
def root539Tail7 : List Nat := 5399 :: root539Tail8
def root539Tail6 : List Nat := 5189 :: root539Tail7
def root539Tail5 : List Nat := 4649 :: root539Tail6
def root539Tail4 : List Nat := 4349 :: root539Tail5
def root539Tail3 : List Nat := 4337 :: root539Tail4
def root539Tail2 : List Nat := 4271 :: root539Tail3
def root539Tail1 : List Nat := 4211 :: root539Tail2
def root539Tail0 : List Nat := 4019 :: root539Tail1
def root539Candidates0 : List Nat := [4211, 10181]
set_option maxRecDepth 8192 in
theorem root539Candidates0_checked : candidates edge 4 3917 4019 root539Tail1 winner.bound = root539Candidates0 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root539Step0 : scan edge (search edge 3) 4 [3917] 3917 root539Tail0 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail1 winner := by
  rw [root539Tail0, scan, if_neg (by decide), root539Candidates0_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail1
    (if 3 ≤ root539Candidates0.length then search edge 3 [3917, 4019] root539Candidates0 7936 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates1 : List Nat := [5483, 10181]
set_option maxRecDepth 8192 in
theorem root539Candidates1_checked : candidates edge 4 3917 4211 root539Tail2 winner.bound = root539Candidates1 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root539Step1 : scan edge (search edge 3) 4 [3917] 3917 root539Tail1 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail2 winner := by
  rw [root539Tail1, scan, if_neg (by decide), root539Candidates1_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail2
    (if 3 ≤ root539Candidates1.length then search edge 3 [3917, 4211] root539Candidates1 8128 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root539Candidates2_checked : candidates edge 4 3917 4271 root539Tail3 winner.bound = root539Candidates2 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root539Step2 : scan edge (search edge 3) 4 [3917] 3917 root539Tail2 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail3 winner := by
  rw [root539Tail2, scan, if_neg (by decide), root539Candidates2_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail3
    (if 3 ≤ root539Candidates2.length then search edge 3 [3917, 4271] root539Candidates2 8188 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root539Candidates3_checked : candidates edge 4 3917 4337 root539Tail4 winner.bound = root539Candidates3 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root539Step3 : scan edge (search edge 3) 4 [3917] 3917 root539Tail3 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail4 winner := by
  rw [root539Tail3, scan, if_neg (by decide), root539Candidates3_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail4
    (if 3 ≤ root539Candidates3.length then search edge 3 [3917, 4337] root539Candidates3 8254 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates4 : List Nat := [5483]
set_option maxRecDepth 8192 in
theorem root539Candidates4_checked : candidates edge 4 3917 4349 root539Tail5 winner.bound = root539Candidates4 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root539Step4 : scan edge (search edge 3) 4 [3917] 3917 root539Tail4 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail5 winner := by
  rw [root539Tail4, scan, if_neg (by decide), root539Candidates4_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail5
    (if 3 ≤ root539Candidates4.length then search edge 3 [3917, 4349] root539Candidates4 8266 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates5 : List Nat := [5399, 5987, 6323]
set_option maxRecDepth 8192 in
theorem root539Candidates5_checked : candidates edge 4 3917 4649 root539Tail6 winner.bound = root539Candidates5 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root539Child5 : search edge 3 [3917, 4649] root539Candidates5 8566 winner = winner := by decide
theorem root539Step5 : scan edge (search edge 3) 4 [3917] 3917 root539Tail5 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail6 winner := by
  rw [root539Tail5, scan, if_neg (by decide), root539Candidates5_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail6
    (if 3 ≤ root539Candidates5.length then search edge 3 [3917, 4649] root539Candidates5 8566 winner else winner) = _
  rw [if_pos (by decide), root539Child5]
def root539Candidates6 : List Nat := [15671]
set_option maxRecDepth 8192 in
theorem root539Candidates6_checked : candidates edge 4 3917 5189 root539Tail7 winner.bound = root539Candidates6 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root539Step6 : scan edge (search edge 3) 4 [3917] 3917 root539Tail6 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail7 winner := by
  rw [root539Tail6, scan, if_neg (by decide), root539Candidates6_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail7
    (if 3 ≤ root539Candidates6.length then search edge 3 [3917, 5189] root539Candidates6 9106 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root539Candidates7_checked : candidates edge 4 3917 5399 root539Tail8 winner.bound = root539Candidates7 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root539Step7 : scan edge (search edge 3) 4 [3917] 3917 root539Tail7 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail8 winner := by
  rw [root539Tail7, scan, if_neg (by decide), root539Candidates7_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail8
    (if 3 ≤ root539Candidates7.length then search edge 3 [3917, 5399] root539Candidates7 9316 winner else winner) = _
  rw [if_neg (by decide)]
def root539Candidates8 : List Nat := [5987]
set_option maxRecDepth 8192 in
theorem root539Candidates8_checked : candidates edge 4 3917 5483 root539Tail9 winner.bound = root539Candidates8 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root539Step8 : scan edge (search edge 3) 4 [3917] 3917 root539Tail8 winner =
    scan edge (search edge 3) 4 [3917] 3917 root539Tail9 winner := by
  rw [root539Tail8, scan, if_neg (by decide), root539Candidates8_checked]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail9
    (if 3 ≤ root539Candidates8.length then search edge 3 [3917, 5483] root539Candidates8 9400 winner else winner) = _
  rw [if_neg (by decide)]
theorem root539Scan9 : scan edge (search edge 3) 4 [3917] 3917 root539Tail9 winner = winner := by rw [root539Tail9, scan, if_pos (by decide)]
theorem root539Scan8 : scan edge (search edge 3) 4 [3917] 3917 root539Tail8 winner = winner :=
  (root539Step8).trans root539Scan9
theorem root539Scan7 : scan edge (search edge 3) 4 [3917] 3917 root539Tail7 winner = winner :=
  (root539Step7).trans root539Scan8
theorem root539Scan6 : scan edge (search edge 3) 4 [3917] 3917 root539Tail6 winner = winner :=
  (root539Step6).trans root539Scan7
theorem root539Scan5 : scan edge (search edge 3) 4 [3917] 3917 root539Tail5 winner = winner :=
  (root539Step5).trans root539Scan6
theorem root539Scan4 : scan edge (search edge 3) 4 [3917] 3917 root539Tail4 winner = winner :=
  (root539Step4).trans root539Scan5
theorem root539Scan3 : scan edge (search edge 3) 4 [3917] 3917 root539Tail3 winner = winner :=
  (root539Step3).trans root539Scan4
theorem root539Scan2 : scan edge (search edge 3) 4 [3917] 3917 root539Tail2 winner = winner :=
  (root539Step2).trans root539Scan3
theorem root539Scan1 : scan edge (search edge 3) 4 [3917] 3917 root539Tail1 winner = winner :=
  (root539Step1).trans root539Scan2
theorem root539Scan0 : scan edge (search edge 3) 4 [3917] 3917 root539Tail0 winner = winner :=
  (root539Step0).trans root539Scan1
set_option maxRecDepth 4096 in
theorem root539Branch : search edge 4 [3917] root539Candidates 3917 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3917] 3917 root539Tail0 winner = winner
  exact root539Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
