import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root274Candidates : List Nat := Data.row1787
set_option maxRecDepth 8192 in
theorem root274Candidates_checked : candidates edge 5 0 1787 tail275 winner.bound = root274Candidates := by
  apply candidates_eq_row edge 5 0 1787 tail275 root274Candidates winner.bound tail275_ordered
  · decide
  · decide
  · funext q
    rfl
def root274Tail49 : List Nat := []
def root274Tail48 : List Nat := 23813 :: root274Tail49
def root274Tail47 : List Nat := 22937 :: root274Tail48
def root274Tail46 : List Nat := 22871 :: root274Tail47
def root274Tail45 : List Nat := 22541 :: root274Tail46
def root274Tail44 : List Nat := 20753 :: root274Tail45
def root274Tail43 : List Nat := 19661 :: root274Tail44
def root274Tail42 : List Nat := 18797 :: root274Tail43
def root274Tail41 : List Nat := 18719 :: root274Tail42
def root274Tail40 : List Nat := 18443 :: root274Tail41
def root274Tail39 : List Nat := 18401 :: root274Tail40
def root274Tail38 : List Nat := 17921 :: root274Tail39
def root274Tail37 : List Nat := 17471 :: root274Tail38
def root274Tail36 : List Nat := 15959 :: root274Tail37
def root274Tail35 : List Nat := 15671 :: root274Tail36
def root274Tail34 : List Nat := 14621 :: root274Tail35
def root274Tail33 : List Nat := 14243 :: root274Tail34
def root274Tail32 : List Nat := 13553 :: root274Tail33
def root274Tail31 : List Nat := 13001 :: root274Tail32
def root274Tail30 : List Nat := 12269 :: root274Tail31
def root274Tail29 : List Nat := 11933 :: root274Tail30
def root274Tail28 : List Nat := 11801 :: root274Tail29
def root274Tail27 : List Nat := 11003 :: root274Tail28
def root274Tail26 : List Nat := 10733 :: root274Tail27
def root274Tail25 : List Nat := 10457 :: root274Tail26
def root274Tail24 : List Nat := 9791 :: root274Tail25
def root274Tail23 : List Nat := 9479 :: root274Tail24
def root274Tail22 : List Nat := 9221 :: root274Tail23
def root274Tail21 : List Nat := 8681 :: root274Tail22
def root274Tail20 : List Nat := 8447 :: root274Tail21
def root274Tail19 : List Nat := 8387 :: root274Tail20
def root274Tail18 : List Nat := 7901 :: root274Tail19
def root274Tail17 : List Nat := 7823 :: root274Tail18
def root274Tail16 : List Nat := 7757 :: root274Tail17
def root274Tail15 : List Nat := 7547 :: root274Tail16
def root274Tail14 : List Nat := 6983 :: root274Tail15
def root274Tail13 : List Nat := 6323 :: root274Tail14
def root274Tail12 : List Nat := 6269 :: root274Tail13
def root274Tail11 : List Nat := 6011 :: root274Tail12
def root274Tail10 : List Nat := 5927 :: root274Tail11
def root274Tail9 : List Nat := 5591 :: root274Tail10
def root274Tail8 : List Nat := 5387 :: root274Tail9
def root274Tail7 : List Nat := 4937 :: root274Tail8
def root274Tail6 : List Nat := 4919 :: root274Tail7
def root274Tail5 : List Nat := 4391 :: root274Tail6
def root274Tail4 : List Nat := 4349 :: root274Tail5
def root274Tail3 : List Nat := 4271 :: root274Tail4
def root274Tail2 : List Nat := 4211 :: root274Tail3
def root274Tail1 : List Nat := 3761 :: root274Tail2
def root274Tail0 : List Nat := 2447 :: root274Tail1
def root274Candidates0 : List Nat := [4937, 6323]
set_option maxRecDepth 8192 in
theorem root274Candidates0_checked : candidates edge 4 1787 2447 root274Tail1 winner.bound = root274Candidates0 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root274Step0 : scan edge (search edge 3) 4 [1787] 1787 root274Tail0 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail1 winner := by
  rw [root274Tail0, scan, if_neg (by decide), root274Candidates0_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail1
    (if 3 ≤ root274Candidates0.length then search edge 3 [1787, 2447] root274Candidates0 4234 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates1 : List Nat := [4391, 17921, 19661]
set_option maxRecDepth 8192 in
theorem root274Candidates1_checked : candidates edge 4 1787 3761 root274Tail2 winner.bound = root274Candidates1 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root274Child1 : search edge 3 [1787, 3761] root274Candidates1 5548 winner = winner := by decide
theorem root274Step1 : scan edge (search edge 3) 4 [1787] 1787 root274Tail1 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail2 winner := by
  rw [root274Tail1, scan, if_neg (by decide), root274Candidates1_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail2
    (if 3 ≤ root274Candidates1.length then search edge 3 [1787, 3761] root274Candidates1 5548 winner else winner) = _
  rw [if_pos (by decide), root274Child1]
def root274Candidates2 : List Nat := [9221]
set_option maxRecDepth 8192 in
theorem root274Candidates2_checked : candidates edge 4 1787 4211 root274Tail3 winner.bound = root274Candidates2 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root274Step2 : scan edge (search edge 3) 4 [1787] 1787 root274Tail2 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail3 winner := by
  rw [root274Tail2, scan, if_neg (by decide), root274Candidates2_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail3
    (if 3 ≤ root274Candidates2.length then search edge 3 [1787, 4211] root274Candidates2 5998 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates3 : List Nat := [7823]
set_option maxRecDepth 8192 in
theorem root274Candidates3_checked : candidates edge 4 1787 4271 root274Tail4 winner.bound = root274Candidates3 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root274Step3 : scan edge (search edge 3) 4 [1787] 1787 root274Tail3 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail4 winner := by
  rw [root274Tail3, scan, if_neg (by decide), root274Candidates3_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail4
    (if 3 ≤ root274Candidates3.length then search edge 3 [1787, 4271] root274Candidates3 6058 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates4 : List Nat := [13001, 17471]
set_option maxRecDepth 8192 in
theorem root274Candidates4_checked : candidates edge 4 1787 4349 root274Tail5 winner.bound = root274Candidates4 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root274Step4 : scan edge (search edge 3) 4 [1787] 1787 root274Tail4 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail5 winner := by
  rw [root274Tail4, scan, if_neg (by decide), root274Candidates4_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail5
    (if 3 ≤ root274Candidates4.length then search edge 3 [1787, 4349] root274Candidates4 6136 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates5 : List Nat := [7901]
set_option maxRecDepth 8192 in
theorem root274Candidates5_checked : candidates edge 4 1787 4391 root274Tail6 winner.bound = root274Candidates5 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root274Step5 : scan edge (search edge 3) 4 [1787] 1787 root274Tail5 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail6 winner := by
  rw [root274Tail5, scan, if_neg (by decide), root274Candidates5_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail6
    (if 3 ≤ root274Candidates5.length then search edge 3 [1787, 4391] root274Candidates5 6178 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root274Candidates6_checked : candidates edge 4 1787 4919 root274Tail7 winner.bound = root274Candidates6 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root274Step6 : scan edge (search edge 3) 4 [1787] 1787 root274Tail6 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail7 winner := by
  rw [root274Tail6, scan, if_neg (by decide), root274Candidates6_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail7
    (if 3 ≤ root274Candidates6.length then search edge 3 [1787, 4919] root274Candidates6 6706 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates7 : List Nat := [9221]
set_option maxRecDepth 8192 in
theorem root274Candidates7_checked : candidates edge 4 1787 4937 root274Tail8 winner.bound = root274Candidates7 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root274Step7 : scan edge (search edge 3) 4 [1787] 1787 root274Tail7 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail8 winner := by
  rw [root274Tail7, scan, if_neg (by decide), root274Candidates7_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail8
    (if 3 ≤ root274Candidates7.length then search edge 3 [1787, 4937] root274Candidates7 6724 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates8 : List Nat := [6269, 9479]
set_option maxRecDepth 8192 in
theorem root274Candidates8_checked : candidates edge 4 1787 5387 root274Tail9 winner.bound = root274Candidates8 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root274Step8 : scan edge (search edge 3) 4 [1787] 1787 root274Tail8 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail9 winner := by
  rw [root274Tail8, scan, if_neg (by decide), root274Candidates8_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail9
    (if 3 ≤ root274Candidates8.length then search edge 3 [1787, 5387] root274Candidates8 7174 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates9 : List Nat := [13553, 18401]
set_option maxRecDepth 8192 in
theorem root274Candidates9_checked : candidates edge 4 1787 5591 root274Tail10 winner.bound = root274Candidates9 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root274Step9 : scan edge (search edge 3) 4 [1787] 1787 root274Tail9 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail10 winner := by
  rw [root274Tail9, scan, if_neg (by decide), root274Candidates9_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail10
    (if 3 ≤ root274Candidates9.length then search edge 3 [1787, 5591] root274Candidates9 7378 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates10 : List Nat := [13001]
set_option maxRecDepth 8192 in
theorem root274Candidates10_checked : candidates edge 4 1787 5927 root274Tail11 winner.bound = root274Candidates10 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root274Step10 : scan edge (search edge 3) 4 [1787] 1787 root274Tail10 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail11 winner := by
  rw [root274Tail10, scan, if_neg (by decide), root274Candidates10_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail11
    (if 3 ≤ root274Candidates10.length then search edge 3 [1787, 5927] root274Candidates10 7714 winner else winner) = _
  rw [if_neg (by decide)]
def root274Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root274Candidates11_checked : candidates edge 4 1787 6011 root274Tail12 winner.bound = root274Candidates11 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root274Step11 : scan edge (search edge 3) 4 [1787] 1787 root274Tail11 winner =
    scan edge (search edge 3) 4 [1787] 1787 root274Tail12 winner := by
  rw [root274Tail11, scan, if_neg (by decide), root274Candidates11_checked]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail12
    (if 3 ≤ root274Candidates11.length then search edge 3 [1787, 6011] root274Candidates11 7798 winner else winner) = _
  rw [if_neg (by decide)]
theorem root274Scan12 : scan edge (search edge 3) 4 [1787] 1787 root274Tail12 winner = winner := by rw [root274Tail12, scan, if_pos (by decide)]
theorem root274Scan11 : scan edge (search edge 3) 4 [1787] 1787 root274Tail11 winner = winner :=
  (root274Step11).trans root274Scan12
theorem root274Scan10 : scan edge (search edge 3) 4 [1787] 1787 root274Tail10 winner = winner :=
  (root274Step10).trans root274Scan11
theorem root274Scan9 : scan edge (search edge 3) 4 [1787] 1787 root274Tail9 winner = winner :=
  (root274Step9).trans root274Scan10
theorem root274Scan8 : scan edge (search edge 3) 4 [1787] 1787 root274Tail8 winner = winner :=
  (root274Step8).trans root274Scan9
theorem root274Scan7 : scan edge (search edge 3) 4 [1787] 1787 root274Tail7 winner = winner :=
  (root274Step7).trans root274Scan8
theorem root274Scan6 : scan edge (search edge 3) 4 [1787] 1787 root274Tail6 winner = winner :=
  (root274Step6).trans root274Scan7
theorem root274Scan5 : scan edge (search edge 3) 4 [1787] 1787 root274Tail5 winner = winner :=
  (root274Step5).trans root274Scan6
theorem root274Scan4 : scan edge (search edge 3) 4 [1787] 1787 root274Tail4 winner = winner :=
  (root274Step4).trans root274Scan5
theorem root274Scan3 : scan edge (search edge 3) 4 [1787] 1787 root274Tail3 winner = winner :=
  (root274Step3).trans root274Scan4
theorem root274Scan2 : scan edge (search edge 3) 4 [1787] 1787 root274Tail2 winner = winner :=
  (root274Step2).trans root274Scan3
theorem root274Scan1 : scan edge (search edge 3) 4 [1787] 1787 root274Tail1 winner = winner :=
  (root274Step1).trans root274Scan2
theorem root274Scan0 : scan edge (search edge 3) 4 [1787] 1787 root274Tail0 winner = winner :=
  (root274Step0).trans root274Scan1
set_option maxRecDepth 4096 in
theorem root274Branch : search edge 4 [1787] root274Candidates 1787 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1787] 1787 root274Tail0 winner = winner
  exact root274Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
