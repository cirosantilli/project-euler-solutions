import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root367Candidates : List Nat := Data.row2531
set_option maxRecDepth 8192 in
theorem root367Candidates_checked : candidates edge 5 0 2531 tail368 winner.bound = root367Candidates := by
  apply candidates_eq_row edge 5 0 2531 tail368 root367Candidates winner.bound tail368_ordered
  · decide
  · decide
  · funext q
    rfl
def root367Tail47 : List Nat := []
def root367Tail46 : List Nat := 23291 :: root367Tail47
def root367Tail45 : List Nat := 22481 :: root367Tail46
def root367Tail44 : List Nat := 21893 :: root367Tail45
def root367Tail43 : List Nat := 20963 :: root367Tail44
def root367Tail42 : List Nat := 20759 :: root367Tail43
def root367Tail41 : List Nat := 18401 :: root367Tail42
def root367Tail40 : List Nat := 17957 :: root367Tail41
def root367Tail39 : List Nat := 17099 :: root367Tail40
def root367Tail38 : List Nat := 16943 :: root367Tail39
def root367Tail37 : List Nat := 16823 :: root367Tail38
def root367Tail36 : List Nat := 16649 :: root367Tail37
def root367Tail35 : List Nat := 16427 :: root367Tail36
def root367Tail34 : List Nat := 16187 :: root367Tail35
def root367Tail33 : List Nat := 15809 :: root367Tail34
def root367Tail32 : List Nat := 14753 :: root367Tail33
def root367Tail31 : List Nat := 14633 :: root367Tail32
def root367Tail30 : List Nat := 14249 :: root367Tail31
def root367Tail29 : List Nat := 13577 :: root367Tail30
def root367Tail28 : List Nat := 12611 :: root367Tail29
def root367Tail27 : List Nat := 12413 :: root367Tail28
def root367Tail26 : List Nat := 12329 :: root367Tail27
def root367Tail25 : List Nat := 12149 :: root367Tail26
def root367Tail24 : List Nat := 11981 :: root367Tail25
def root367Tail23 : List Nat := 11867 :: root367Tail24
def root367Tail22 : List Nat := 11471 :: root367Tail23
def root367Tail21 : List Nat := 9689 :: root367Tail22
def root367Tail20 : List Nat := 9293 :: root367Tail21
def root367Tail19 : List Nat := 9221 :: root367Tail20
def root367Tail18 : List Nat := 9011 :: root367Tail19
def root367Tail17 : List Nat := 8753 :: root367Tail18
def root367Tail16 : List Nat := 8669 :: root367Tail17
def root367Tail15 : List Nat := 8171 :: root367Tail16
def root367Tail14 : List Nat := 8123 :: root367Tail15
def root367Tail13 : List Nat := 7883 :: root367Tail14
def root367Tail12 : List Nat := 6701 :: root367Tail13
def root367Tail11 : List Nat := 6659 :: root367Tail12
def root367Tail10 : List Nat := 6329 :: root367Tail11
def root367Tail9 : List Nat := 5651 :: root367Tail10
def root367Tail8 : List Nat := 5309 :: root367Tail9
def root367Tail7 : List Nat := 5273 :: root367Tail8
def root367Tail6 : List Nat := 4967 :: root367Tail7
def root367Tail5 : List Nat := 4889 :: root367Tail6
def root367Tail4 : List Nat := 4643 :: root367Tail5
def root367Tail3 : List Nat := 3797 :: root367Tail4
def root367Tail2 : List Nat := 3539 :: root367Tail3
def root367Tail1 : List Nat := 2741 :: root367Tail2
def root367Tail0 : List Nat := 2549 :: root367Tail1
def root367Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root367Candidates0_checked : candidates edge 4 2531 2549 root367Tail1 winner.bound = root367Candidates0 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root367Step0 : scan edge (search edge 3) 4 [2531] 2531 root367Tail0 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail1 winner := by
  rw [root367Tail0, scan, if_neg (by decide), root367Candidates0_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail1
    (if 3 ≤ root367Candidates0.length then search edge 3 [2531, 2549] root367Candidates0 5080 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root367Candidates1_checked : candidates edge 4 2531 2741 root367Tail2 winner.bound = root367Candidates1 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root367Step1 : scan edge (search edge 3) 4 [2531] 2531 root367Tail1 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail2 winner := by
  rw [root367Tail1, scan, if_neg (by decide), root367Candidates1_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail2
    (if 3 ≤ root367Candidates1.length then search edge 3 [2531, 2741] root367Candidates1 5272 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates2 : List Nat := [11981]
set_option maxRecDepth 8192 in
theorem root367Candidates2_checked : candidates edge 4 2531 3539 root367Tail3 winner.bound = root367Candidates2 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root367Step2 : scan edge (search edge 3) 4 [2531] 2531 root367Tail2 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail3 winner := by
  rw [root367Tail2, scan, if_neg (by decide), root367Candidates2_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail3
    (if 3 ≤ root367Candidates2.length then search edge 3 [2531, 3539] root367Candidates2 6070 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates3 : List Nat := [12413]
set_option maxRecDepth 8192 in
theorem root367Candidates3_checked : candidates edge 4 2531 3797 root367Tail4 winner.bound = root367Candidates3 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root367Step3 : scan edge (search edge 3) 4 [2531] 2531 root367Tail3 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail4 winner := by
  rw [root367Tail3, scan, if_neg (by decide), root367Candidates3_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail4
    (if 3 ≤ root367Candidates3.length then search edge 3 [2531, 3797] root367Candidates3 6328 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates4 : List Nat := [16823, 18401]
set_option maxRecDepth 8192 in
theorem root367Candidates4_checked : candidates edge 4 2531 4643 root367Tail5 winner.bound = root367Candidates4 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root367Step4 : scan edge (search edge 3) 4 [2531] 2531 root367Tail4 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail5 winner := by
  rw [root367Tail4, scan, if_neg (by decide), root367Candidates4_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail5
    (if 3 ≤ root367Candidates4.length then search edge 3 [2531, 4643] root367Candidates4 7174 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates5 : List Nat := [8123]
set_option maxRecDepth 8192 in
theorem root367Candidates5_checked : candidates edge 4 2531 4889 root367Tail6 winner.bound = root367Candidates5 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root367Step5 : scan edge (search edge 3) 4 [2531] 2531 root367Tail5 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail6 winner := by
  rw [root367Tail5, scan, if_neg (by decide), root367Candidates5_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail6
    (if 3 ≤ root367Candidates5.length then search edge 3 [2531, 4889] root367Candidates5 7420 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates6 : List Nat := [13577]
set_option maxRecDepth 8192 in
theorem root367Candidates6_checked : candidates edge 4 2531 4967 root367Tail7 winner.bound = root367Candidates6 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root367Step6 : scan edge (search edge 3) 4 [2531] 2531 root367Tail6 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail7 winner := by
  rw [root367Tail6, scan, if_neg (by decide), root367Candidates6_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail7
    (if 3 ≤ root367Candidates6.length then search edge 3 [2531, 4967] root367Candidates6 7498 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates7 : List Nat := [5651]
set_option maxRecDepth 8192 in
theorem root367Candidates7_checked : candidates edge 4 2531 5273 root367Tail8 winner.bound = root367Candidates7 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root367Step7 : scan edge (search edge 3) 4 [2531] 2531 root367Tail7 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail8 winner := by
  rw [root367Tail7, scan, if_neg (by decide), root367Candidates7_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail8
    (if 3 ≤ root367Candidates7.length then search edge 3 [2531, 5273] root367Candidates7 7804 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root367Candidates8_checked : candidates edge 4 2531 5309 root367Tail9 winner.bound = root367Candidates8 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root367Step8 : scan edge (search edge 3) 4 [2531] 2531 root367Tail8 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail9 winner := by
  rw [root367Tail8, scan, if_neg (by decide), root367Candidates8_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail9
    (if 3 ≤ root367Candidates8.length then search edge 3 [2531, 5309] root367Candidates8 7840 winner else winner) = _
  rw [if_neg (by decide)]
def root367Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root367Candidates9_checked : candidates edge 4 2531 5651 root367Tail10 winner.bound = root367Candidates9 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root367Step9 : scan edge (search edge 3) 4 [2531] 2531 root367Tail9 winner =
    scan edge (search edge 3) 4 [2531] 2531 root367Tail10 winner := by
  rw [root367Tail9, scan, if_neg (by decide), root367Candidates9_checked]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail10
    (if 3 ≤ root367Candidates9.length then search edge 3 [2531, 5651] root367Candidates9 8182 winner else winner) = _
  rw [if_neg (by decide)]
theorem root367Scan10 : scan edge (search edge 3) 4 [2531] 2531 root367Tail10 winner = winner := by rw [root367Tail10, scan, if_pos (by decide)]
theorem root367Scan9 : scan edge (search edge 3) 4 [2531] 2531 root367Tail9 winner = winner :=
  (root367Step9).trans root367Scan10
theorem root367Scan8 : scan edge (search edge 3) 4 [2531] 2531 root367Tail8 winner = winner :=
  (root367Step8).trans root367Scan9
theorem root367Scan7 : scan edge (search edge 3) 4 [2531] 2531 root367Tail7 winner = winner :=
  (root367Step7).trans root367Scan8
theorem root367Scan6 : scan edge (search edge 3) 4 [2531] 2531 root367Tail6 winner = winner :=
  (root367Step6).trans root367Scan7
theorem root367Scan5 : scan edge (search edge 3) 4 [2531] 2531 root367Tail5 winner = winner :=
  (root367Step5).trans root367Scan6
theorem root367Scan4 : scan edge (search edge 3) 4 [2531] 2531 root367Tail4 winner = winner :=
  (root367Step4).trans root367Scan5
theorem root367Scan3 : scan edge (search edge 3) 4 [2531] 2531 root367Tail3 winner = winner :=
  (root367Step3).trans root367Scan4
theorem root367Scan2 : scan edge (search edge 3) 4 [2531] 2531 root367Tail2 winner = winner :=
  (root367Step2).trans root367Scan3
theorem root367Scan1 : scan edge (search edge 3) 4 [2531] 2531 root367Tail1 winner = winner :=
  (root367Step1).trans root367Scan2
theorem root367Scan0 : scan edge (search edge 3) 4 [2531] 2531 root367Tail0 winner = winner :=
  (root367Step0).trans root367Scan1
set_option maxRecDepth 4096 in
theorem root367Branch : search edge 4 [2531] root367Candidates 2531 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2531] 2531 root367Tail0 winner = winner
  exact root367Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
