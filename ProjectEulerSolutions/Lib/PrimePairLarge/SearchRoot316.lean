import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root316Candidates : List Nat := Data.row2113
set_option maxRecDepth 8192 in
theorem root316Candidates_checked : candidates edge 5 0 2113 tail317 winner.bound = root316Candidates := by
  apply candidates_eq_row edge 5 0 2113 tail317 root316Candidates winner.bound tail317_ordered
  · decide
  · decide
  · funext q
    rfl
def root316Tail45 : List Nat := []
def root316Tail44 : List Nat := 23911 :: root316Tail45
def root316Tail43 : List Nat := 22621 :: root316Tail44
def root316Tail42 : List Nat := 22369 :: root316Tail43
def root316Tail41 : List Nat := 22303 :: root316Tail42
def root316Tail40 : List Nat := 21379 :: root316Tail41
def root316Tail39 : List Nat := 21157 :: root316Tail40
def root316Tail38 : List Nat := 21139 :: root316Tail39
def root316Tail37 : List Nat := 19813 :: root316Tail38
def root316Tail36 : List Nat := 18973 :: root316Tail37
def root316Tail35 : List Nat := 18451 :: root316Tail36
def root316Tail34 : List Nat := 16453 :: root316Tail35
def root316Tail33 : List Nat := 16381 :: root316Tail34
def root316Tail32 : List Nat := 15733 :: root316Tail33
def root316Tail31 : List Nat := 15583 :: root316Tail32
def root316Tail30 : List Nat := 14461 :: root316Tail31
def root316Tail29 : List Nat := 12919 :: root316Tail30
def root316Tail28 : List Nat := 12097 :: root316Tail29
def root316Tail27 : List Nat := 11497 :: root316Tail28
def root316Tail26 : List Nat := 11467 :: root316Tail27
def root316Tail25 : List Nat := 11239 :: root316Tail26
def root316Tail24 : List Nat := 10711 :: root316Tail25
def root316Tail23 : List Nat := 10513 :: root316Tail24
def root316Tail22 : List Nat := 9649 :: root316Tail23
def root316Tail21 : List Nat := 9439 :: root316Tail22
def root316Tail20 : List Nat := 9337 :: root316Tail21
def root316Tail19 : List Nat := 9127 :: root316Tail20
def root316Tail18 : List Nat := 8713 :: root316Tail19
def root316Tail17 : List Nat := 7927 :: root316Tail18
def root316Tail16 : List Nat := 7393 :: root316Tail17
def root316Tail15 : List Nat := 7309 :: root316Tail16
def root316Tail14 : List Nat := 6553 :: root316Tail15
def root316Tail13 : List Nat := 6481 :: root316Tail14
def root316Tail12 : List Nat := 6277 :: root316Tail13
def root316Tail11 : List Nat := 6067 :: root316Tail12
def root316Tail10 : List Nat := 5431 :: root316Tail11
def root316Tail9 : List Nat := 5419 :: root316Tail10
def root316Tail8 : List Nat := 5227 :: root316Tail9
def root316Tail7 : List Nat := 4507 :: root316Tail8
def root316Tail6 : List Nat := 4093 :: root316Tail7
def root316Tail5 : List Nat := 3727 :: root316Tail6
def root316Tail4 : List Nat := 3583 :: root316Tail5
def root316Tail3 : List Nat := 3433 :: root316Tail4
def root316Tail2 : List Nat := 3121 :: root316Tail3
def root316Tail1 : List Nat := 2659 :: root316Tail2
def root316Tail0 : List Nat := 2131 :: root316Tail1
def root316Candidates0 : List Nat := [12919]
set_option maxRecDepth 8192 in
theorem root316Candidates0_checked : candidates edge 4 2113 2131 root316Tail1 winner.bound = root316Candidates0 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root316Step0 : scan edge (search edge 3) 4 [2113] 2113 root316Tail0 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail1 winner := by
  rw [root316Tail0, scan, if_neg (by decide), root316Candidates0_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail1
    (if 3 ≤ root316Candidates0.length then search edge 3 [2113, 2131] root316Candidates0 4244 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates1 : List Nat := [7927, 9127, 18973]
set_option maxRecDepth 8192 in
theorem root316Candidates1_checked : candidates edge 4 2113 2659 root316Tail2 winner.bound = root316Candidates1 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root316Child1 : search edge 3 [2113, 2659] root316Candidates1 4772 winner = winner := by decide
theorem root316Step1 : scan edge (search edge 3) 4 [2113] 2113 root316Tail1 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail2 winner := by
  rw [root316Tail1, scan, if_neg (by decide), root316Candidates1_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail2
    (if 3 ≤ root316Candidates1.length then search edge 3 [2113, 2659] root316Candidates1 4772 winner else winner) = _
  rw [if_pos (by decide), root316Child1]
def root316Candidates2 : List Nat := [3433, 4507, 7393, 11497, 18973]
set_option maxRecDepth 8192 in
theorem root316Candidates2_checked : candidates edge 4 2113 3121 root316Tail3 winner.bound = root316Candidates2 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root316Child2 : search edge 3 [2113, 3121] root316Candidates2 5234 winner = winner := by decide
theorem root316Step2 : scan edge (search edge 3) 4 [2113] 2113 root316Tail2 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail3 winner := by
  rw [root316Tail2, scan, if_neg (by decide), root316Candidates2_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail3
    (if 3 ≤ root316Candidates2.length then search edge 3 [2113, 3121] root316Candidates2 5234 winner else winner) = _
  rw [if_pos (by decide), root316Child2]
def root316Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root316Candidates3_checked : candidates edge 4 2113 3433 root316Tail4 winner.bound = root316Candidates3 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root316Step3 : scan edge (search edge 3) 4 [2113] 2113 root316Tail3 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail4 winner := by
  rw [root316Tail3, scan, if_neg (by decide), root316Candidates3_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail4
    (if 3 ≤ root316Candidates3.length then search edge 3 [2113, 3433] root316Candidates3 5546 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates4 : List Nat := [6067]
set_option maxRecDepth 8192 in
theorem root316Candidates4_checked : candidates edge 4 2113 3583 root316Tail5 winner.bound = root316Candidates4 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root316Step4 : scan edge (search edge 3) 4 [2113] 2113 root316Tail4 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail5 winner := by
  rw [root316Tail4, scan, if_neg (by decide), root316Candidates4_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail5
    (if 3 ≤ root316Candidates4.length then search edge 3 [2113, 3583] root316Candidates4 5696 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates5 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root316Candidates5_checked : candidates edge 4 2113 3727 root316Tail6 winner.bound = root316Candidates5 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root316Step5 : scan edge (search edge 3) 4 [2113] 2113 root316Tail5 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail6 winner := by
  rw [root316Tail5, scan, if_neg (by decide), root316Candidates5_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail6
    (if 3 ≤ root316Candidates5.length then search edge 3 [2113, 3727] root316Candidates5 5840 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates6 : List Nat := [11467]
set_option maxRecDepth 8192 in
theorem root316Candidates6_checked : candidates edge 4 2113 4093 root316Tail7 winner.bound = root316Candidates6 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root316Step6 : scan edge (search edge 3) 4 [2113] 2113 root316Tail6 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail7 winner := by
  rw [root316Tail6, scan, if_neg (by decide), root316Candidates6_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail7
    (if 3 ≤ root316Candidates6.length then search edge 3 [2113, 4093] root316Candidates6 6206 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates7 : List Nat := [5419, 18451]
set_option maxRecDepth 8192 in
theorem root316Candidates7_checked : candidates edge 4 2113 4507 root316Tail8 winner.bound = root316Candidates7 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root316Step7 : scan edge (search edge 3) 4 [2113] 2113 root316Tail7 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail8 winner := by
  rw [root316Tail7, scan, if_neg (by decide), root316Candidates7_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail8
    (if 3 ≤ root316Candidates7.length then search edge 3 [2113, 4507] root316Candidates7 6620 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates8 : List Nat := [6481]
set_option maxRecDepth 8192 in
theorem root316Candidates8_checked : candidates edge 4 2113 5227 root316Tail9 winner.bound = root316Candidates8 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root316Step8 : scan edge (search edge 3) 4 [2113] 2113 root316Tail8 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail9 winner := by
  rw [root316Tail8, scan, if_neg (by decide), root316Candidates8_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail9
    (if 3 ≤ root316Candidates8.length then search edge 3 [2113, 5227] root316Candidates8 7340 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates9 : List Nat := [11239, 11467]
set_option maxRecDepth 8192 in
theorem root316Candidates9_checked : candidates edge 4 2113 5419 root316Tail10 winner.bound = root316Candidates9 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root316Step9 : scan edge (search edge 3) 4 [2113] 2113 root316Tail9 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail10 winner := by
  rw [root316Tail9, scan, if_neg (by decide), root316Candidates9_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail10
    (if 3 ≤ root316Candidates9.length then search edge 3 [2113, 5419] root316Candidates9 7532 winner else winner) = _
  rw [if_neg (by decide)]
def root316Candidates10 : List Nat := [9649]
set_option maxRecDepth 8192 in
theorem root316Candidates10_checked : candidates edge 4 2113 5431 root316Tail11 winner.bound = root316Candidates10 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root316Step10 : scan edge (search edge 3) 4 [2113] 2113 root316Tail10 winner =
    scan edge (search edge 3) 4 [2113] 2113 root316Tail11 winner := by
  rw [root316Tail10, scan, if_neg (by decide), root316Candidates10_checked]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail11
    (if 3 ≤ root316Candidates10.length then search edge 3 [2113, 5431] root316Candidates10 7544 winner else winner) = _
  rw [if_neg (by decide)]
theorem root316Scan11 : scan edge (search edge 3) 4 [2113] 2113 root316Tail11 winner = winner := by rw [root316Tail11, scan, if_pos (by decide)]
theorem root316Scan10 : scan edge (search edge 3) 4 [2113] 2113 root316Tail10 winner = winner :=
  (root316Step10).trans root316Scan11
theorem root316Scan9 : scan edge (search edge 3) 4 [2113] 2113 root316Tail9 winner = winner :=
  (root316Step9).trans root316Scan10
theorem root316Scan8 : scan edge (search edge 3) 4 [2113] 2113 root316Tail8 winner = winner :=
  (root316Step8).trans root316Scan9
theorem root316Scan7 : scan edge (search edge 3) 4 [2113] 2113 root316Tail7 winner = winner :=
  (root316Step7).trans root316Scan8
theorem root316Scan6 : scan edge (search edge 3) 4 [2113] 2113 root316Tail6 winner = winner :=
  (root316Step6).trans root316Scan7
theorem root316Scan5 : scan edge (search edge 3) 4 [2113] 2113 root316Tail5 winner = winner :=
  (root316Step5).trans root316Scan6
theorem root316Scan4 : scan edge (search edge 3) 4 [2113] 2113 root316Tail4 winner = winner :=
  (root316Step4).trans root316Scan5
theorem root316Scan3 : scan edge (search edge 3) 4 [2113] 2113 root316Tail3 winner = winner :=
  (root316Step3).trans root316Scan4
theorem root316Scan2 : scan edge (search edge 3) 4 [2113] 2113 root316Tail2 winner = winner :=
  (root316Step2).trans root316Scan3
theorem root316Scan1 : scan edge (search edge 3) 4 [2113] 2113 root316Tail1 winner = winner :=
  (root316Step1).trans root316Scan2
theorem root316Scan0 : scan edge (search edge 3) 4 [2113] 2113 root316Tail0 winner = winner :=
  (root316Step0).trans root316Scan1
set_option maxRecDepth 4096 in
theorem root316Branch : search edge 4 [2113] root316Candidates 2113 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2113] 2113 root316Tail0 winner = winner
  exact root316Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
