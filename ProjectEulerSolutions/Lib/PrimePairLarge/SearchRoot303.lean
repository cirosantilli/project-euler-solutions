import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root303Candidates : List Nat := Data.row2017
set_option maxRecDepth 8192 in
theorem root303Candidates_checked : candidates edge 5 0 2017 tail304 winner.bound = root303Candidates := by
  apply candidates_eq_row edge 5 0 2017 tail304 root303Candidates winner.bound tail304_ordered
  · decide
  · decide
  · funext q
    rfl
def root303Tail43 : List Nat := []
def root303Tail42 : List Nat := 23689 :: root303Tail43
def root303Tail41 : List Nat := 23251 :: root303Tail42
def root303Tail40 : List Nat := 23143 :: root303Tail41
def root303Tail39 : List Nat := 21319 :: root303Tail40
def root303Tail38 : List Nat := 21283 :: root303Tail39
def root303Tail37 : List Nat := 20593 :: root303Tail38
def root303Tail36 : List Nat := 20047 :: root303Tail37
def root303Tail35 : List Nat := 19753 :: root303Tail36
def root303Tail34 : List Nat := 19681 :: root303Tail35
def root303Tail33 : List Nat := 19489 :: root303Tail34
def root303Tail32 : List Nat := 18973 :: root303Tail33
def root303Tail31 : List Nat := 18367 :: root303Tail32
def root303Tail30 : List Nat := 18061 :: root303Tail31
def root303Tail29 : List Nat := 17977 :: root303Tail30
def root303Tail28 : List Nat := 16087 :: root303Tail29
def root303Tail27 : List Nat := 12919 :: root303Tail28
def root303Tail26 : List Nat := 12697 :: root303Tail27
def root303Tail25 : List Nat := 11299 :: root303Tail26
def root303Tail24 : List Nat := 11197 :: root303Tail25
def root303Tail23 : List Nat := 10987 :: root303Tail24
def root303Tail22 : List Nat := 10711 :: root303Tail23
def root303Tail21 : List Nat := 10651 :: root303Tail22
def root303Tail20 : List Nat := 9787 :: root303Tail21
def root303Tail19 : List Nat := 9739 :: root303Tail20
def root303Tail18 : List Nat := 9001 :: root303Tail19
def root303Tail17 : List Nat := 8971 :: root303Tail18
def root303Tail16 : List Nat := 8707 :: root303Tail17
def root303Tail15 : List Nat := 8467 :: root303Tail16
def root303Tail14 : List Nat := 8317 :: root303Tail15
def root303Tail13 : List Nat := 8311 :: root303Tail14
def root303Tail12 : List Nat := 7603 :: root303Tail13
def root303Tail11 : List Nat := 7237 :: root303Tail12
def root303Tail10 : List Nat := 6967 :: root303Tail11
def root303Tail9 : List Nat := 6661 :: root303Tail10
def root303Tail8 : List Nat := 5653 :: root303Tail9
def root303Tail7 : List Nat := 4621 :: root303Tail8
def root303Tail6 : List Nat := 4423 :: root303Tail7
def root303Tail5 : List Nat := 4339 :: root303Tail6
def root303Tail4 : List Nat := 3919 :: root303Tail5
def root303Tail3 : List Nat := 3313 :: root303Tail4
def root303Tail2 : List Nat := 2689 :: root303Tail3
def root303Tail1 : List Nat := 2617 :: root303Tail2
def root303Tail0 : List Nat := 2221 :: root303Tail1
def root303Candidates0 : List Nat := [20047]
set_option maxRecDepth 8192 in
theorem root303Candidates0_checked : candidates edge 4 2017 2221 root303Tail1 winner.bound = root303Candidates0 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
theorem root303Step0 : scan edge (search edge 3) 4 [2017] 2017 root303Tail0 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail1 winner := by
  rw [root303Tail0, scan, if_neg (by decide), root303Candidates0_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail1
    (if 3 ≤ root303Candidates0.length then search edge 3 [2017, 2221] root303Candidates0 4238 winner else winner) = _
  rw [if_neg (by decide)]
def root303Candidates1 : List Nat := [10651]
set_option maxRecDepth 8192 in
theorem root303Candidates1_checked : candidates edge 4 2017 2617 root303Tail2 winner.bound = root303Candidates1 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
theorem root303Step1 : scan edge (search edge 3) 4 [2017] 2017 root303Tail1 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail2 winner := by
  rw [root303Tail1, scan, if_neg (by decide), root303Candidates1_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail2
    (if 3 ≤ root303Candidates1.length then search edge 3 [2017, 2617] root303Candidates1 4634 winner else winner) = _
  rw [if_neg (by decide)]
def root303Candidates2 : List Nat := [4423, 8317, 12697, 20593, 21319]
set_option maxRecDepth 8192 in
theorem root303Candidates2_checked : candidates edge 4 2017 2689 root303Tail3 winner.bound = root303Candidates2 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root303Child2 : search edge 3 [2017, 2689] root303Candidates2 4706 winner = winner := by decide
theorem root303Step2 : scan edge (search edge 3) 4 [2017] 2017 root303Tail2 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail3 winner := by
  rw [root303Tail2, scan, if_neg (by decide), root303Candidates2_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail3
    (if 3 ≤ root303Candidates2.length then search edge 3 [2017, 2689] root303Candidates2 4706 winner else winner) = _
  rw [if_pos (by decide), root303Child2]
def root303Candidates3 : List Nat := [9739]
set_option maxRecDepth 8192 in
theorem root303Candidates3_checked : candidates edge 4 2017 3313 root303Tail4 winner.bound = root303Candidates3 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root303Step3 : scan edge (search edge 3) 4 [2017] 2017 root303Tail3 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail4 winner := by
  rw [root303Tail3, scan, if_neg (by decide), root303Candidates3_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail4
    (if 3 ≤ root303Candidates3.length then search edge 3 [2017, 3313] root303Candidates3 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root303Candidates4 : List Nat := [8317]
set_option maxRecDepth 8192 in
theorem root303Candidates4_checked : candidates edge 4 2017 3919 root303Tail5 winner.bound = root303Candidates4 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root303Step4 : scan edge (search edge 3) 4 [2017] 2017 root303Tail4 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail5 winner := by
  rw [root303Tail4, scan, if_neg (by decide), root303Candidates4_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail5
    (if 3 ≤ root303Candidates4.length then search edge 3 [2017, 3919] root303Candidates4 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root303Candidates5 : List Nat := [7603, 8317, 10651]
set_option maxRecDepth 8192 in
theorem root303Candidates5_checked : candidates edge 4 2017 4339 root303Tail6 winner.bound = root303Candidates5 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root303Child5 : search edge 3 [2017, 4339] root303Candidates5 6356 winner = winner := by decide
theorem root303Step5 : scan edge (search edge 3) 4 [2017] 2017 root303Tail5 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail6 winner := by
  rw [root303Tail5, scan, if_neg (by decide), root303Candidates5_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail6
    (if 3 ≤ root303Candidates5.length then search edge 3 [2017, 4339] root303Candidates5 6356 winner else winner) = _
  rw [if_pos (by decide), root303Child5]
def root303Candidates6 : List Nat := [8317, 16087]
set_option maxRecDepth 8192 in
theorem root303Candidates6_checked : candidates edge 4 2017 4423 root303Tail7 winner.bound = root303Candidates6 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root303Step6 : scan edge (search edge 3) 4 [2017] 2017 root303Tail6 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail7 winner := by
  rw [root303Tail6, scan, if_neg (by decide), root303Candidates6_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail7
    (if 3 ≤ root303Candidates6.length then search edge 3 [2017, 4423] root303Candidates6 6440 winner else winner) = _
  rw [if_neg (by decide)]
def root303Candidates7 : List Nat := [10651]
set_option maxRecDepth 8192 in
theorem root303Candidates7_checked : candidates edge 4 2017 4621 root303Tail8 winner.bound = root303Candidates7 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root303Step7 : scan edge (search edge 3) 4 [2017] 2017 root303Tail7 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail8 winner := by
  rw [root303Tail7, scan, if_neg (by decide), root303Candidates7_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail8
    (if 3 ≤ root303Candidates7.length then search edge 3 [2017, 4621] root303Candidates7 6638 winner else winner) = _
  rw [if_neg (by decide)]
def root303Candidates8 : List Nat := [10987, 12919]
set_option maxRecDepth 8192 in
theorem root303Candidates8_checked : candidates edge 4 2017 5653 root303Tail9 winner.bound = root303Candidates8 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root303Step8 : scan edge (search edge 3) 4 [2017] 2017 root303Tail8 winner =
    scan edge (search edge 3) 4 [2017] 2017 root303Tail9 winner := by
  rw [root303Tail8, scan, if_neg (by decide), root303Candidates8_checked]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail9
    (if 3 ≤ root303Candidates8.length then search edge 3 [2017, 5653] root303Candidates8 7670 winner else winner) = _
  rw [if_neg (by decide)]
theorem root303Scan9 : scan edge (search edge 3) 4 [2017] 2017 root303Tail9 winner = winner := by rw [root303Tail9, scan, if_pos (by decide)]
theorem root303Scan8 : scan edge (search edge 3) 4 [2017] 2017 root303Tail8 winner = winner :=
  (root303Step8).trans root303Scan9
theorem root303Scan7 : scan edge (search edge 3) 4 [2017] 2017 root303Tail7 winner = winner :=
  (root303Step7).trans root303Scan8
theorem root303Scan6 : scan edge (search edge 3) 4 [2017] 2017 root303Tail6 winner = winner :=
  (root303Step6).trans root303Scan7
theorem root303Scan5 : scan edge (search edge 3) 4 [2017] 2017 root303Tail5 winner = winner :=
  (root303Step5).trans root303Scan6
theorem root303Scan4 : scan edge (search edge 3) 4 [2017] 2017 root303Tail4 winner = winner :=
  (root303Step4).trans root303Scan5
theorem root303Scan3 : scan edge (search edge 3) 4 [2017] 2017 root303Tail3 winner = winner :=
  (root303Step3).trans root303Scan4
theorem root303Scan2 : scan edge (search edge 3) 4 [2017] 2017 root303Tail2 winner = winner :=
  (root303Step2).trans root303Scan3
theorem root303Scan1 : scan edge (search edge 3) 4 [2017] 2017 root303Tail1 winner = winner :=
  (root303Step1).trans root303Scan2
theorem root303Scan0 : scan edge (search edge 3) 4 [2017] 2017 root303Tail0 winner = winner :=
  (root303Step0).trans root303Scan1
set_option maxRecDepth 4096 in
theorem root303Branch : search edge 4 [2017] root303Candidates 2017 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2017] 2017 root303Tail0 winner = winner
  exact root303Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
