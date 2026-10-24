import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root228Candidates : List Nat := Data.row1453
set_option maxRecDepth 8192 in
theorem root228Candidates_checked : candidates edge 5 0 1453 tail229 winner.bound = root228Candidates := by
  apply candidates_eq_row edge 5 0 1453 tail229 root228Candidates winner.bound tail229_ordered
  · decide
  · decide
  · funext q
    rfl
def root228Tail43 : List Nat := []
def root228Tail42 : List Nat := 24547 :: root228Tail43
def root228Tail41 : List Nat := 21817 :: root228Tail42
def root228Tail40 : List Nat := 21577 :: root228Tail41
def root228Tail39 : List Nat := 20809 :: root228Tail40
def root228Tail38 : List Nat := 20407 :: root228Tail39
def root228Tail37 : List Nat := 20101 :: root228Tail38
def root228Tail36 : List Nat := 20023 :: root228Tail37
def root228Tail35 : List Nat := 19687 :: root228Tail36
def root228Tail34 : List Nat := 18427 :: root228Tail35
def root228Tail33 : List Nat := 18181 :: root228Tail34
def root228Tail32 : List Nat := 17971 :: root228Tail33
def root228Tail31 : List Nat := 17827 :: root228Tail32
def root228Tail30 : List Nat := 16111 :: root228Tail31
def root228Tail29 : List Nat := 14551 :: root228Tail30
def root228Tail28 : List Nat := 14401 :: root228Tail29
def root228Tail27 : List Nat := 14107 :: root228Tail28
def root228Tail26 : List Nat := 13759 :: root228Tail27
def root228Tail25 : List Nat := 11443 :: root228Tail26
def root228Tail24 : List Nat := 9787 :: root228Tail25
def root228Tail23 : List Nat := 9781 :: root228Tail24
def root228Tail22 : List Nat := 9103 :: root228Tail23
def root228Tail21 : List Nat := 8929 :: root228Tail22
def root228Tail20 : List Nat := 8089 :: root228Tail21
def root228Tail19 : List Nat := 7933 :: root228Tail20
def root228Tail18 : List Nat := 7927 :: root228Tail19
def root228Tail17 : List Nat := 7417 :: root228Tail18
def root228Tail16 : List Nat := 7129 :: root228Tail17
def root228Tail15 : List Nat := 6763 :: root228Tail16
def root228Tail14 : List Nat := 5923 :: root228Tail15
def root228Tail13 : List Nat := 5743 :: root228Tail14
def root228Tail12 : List Nat := 5581 :: root228Tail13
def root228Tail11 : List Nat := 5281 :: root228Tail12
def root228Tail10 : List Nat := 5113 :: root228Tail11
def root228Tail9 : List Nat := 4783 :: root228Tail10
def root228Tail8 : List Nat := 4651 :: root228Tail9
def root228Tail7 : List Nat := 4483 :: root228Tail8
def root228Tail6 : List Nat := 3361 :: root228Tail7
def root228Tail5 : List Nat := 3049 :: root228Tail6
def root228Tail4 : List Nat := 2797 :: root228Tail5
def root228Tail3 : List Nat := 2593 :: root228Tail4
def root228Tail2 : List Nat := 2437 :: root228Tail3
def root228Tail1 : List Nat := 1879 :: root228Tail2
def root228Tail0 : List Nat := 1669 :: root228Tail1
def root228Candidates0 : List Nat := [2593, 4651, 5743]
set_option maxRecDepth 8192 in
theorem root228Candidates0_checked : candidates edge 4 1453 1669 root228Tail1 winner.bound = root228Candidates0 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root228Child0 : search edge 3 [1453, 1669] root228Candidates0 3122 winner = winner := by decide
theorem root228Step0 : scan edge (search edge 3) 4 [1453] 1453 root228Tail0 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail1 winner := by
  rw [root228Tail0, scan, if_neg (by decide), root228Candidates0_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail1
    (if 3 ≤ root228Candidates0.length then search edge 3 [1453, 1669] root228Candidates0 3122 winner else winner) = _
  rw [if_pos (by decide), root228Child0]
def root228Candidates1 : List Nat := [9103, 21817]
set_option maxRecDepth 8192 in
theorem root228Candidates1_checked : candidates edge 4 1453 1879 root228Tail2 winner.bound = root228Candidates1 := by
  unfold candidates
  have row : edge 1879 = (SortedNatTree.contains FastRows.row1879) := by
    change Data.row1879.contains = (SortedNatTree.contains FastRows.row1879)
    exact FastRows.row1879_checked.symm
  rw [row]
  decide
theorem root228Step1 : scan edge (search edge 3) 4 [1453] 1453 root228Tail1 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail2 winner := by
  rw [root228Tail1, scan, if_neg (by decide), root228Candidates1_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail2
    (if 3 ≤ root228Candidates1.length then search edge 3 [1453, 1879] root228Candidates1 3332 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates2 : List Nat := [3361]
set_option maxRecDepth 8192 in
theorem root228Candidates2_checked : candidates edge 4 1453 2437 root228Tail3 winner.bound = root228Candidates2 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root228Step2 : scan edge (search edge 3) 4 [1453] 1453 root228Tail2 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail3 winner := by
  rw [root228Tail2, scan, if_neg (by decide), root228Candidates2_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail3
    (if 3 ≤ root228Candidates2.length then search edge 3 [1453, 2437] root228Candidates2 3890 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates3 : List Nat := [5281, 7927, 18427]
set_option maxRecDepth 8192 in
theorem root228Candidates3_checked : candidates edge 4 1453 2593 root228Tail4 winner.bound = root228Candidates3 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root228Child3 : search edge 3 [1453, 2593] root228Candidates3 4046 winner = winner := by decide
theorem root228Step3 : scan edge (search edge 3) 4 [1453] 1453 root228Tail3 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail4 winner := by
  rw [root228Tail3, scan, if_neg (by decide), root228Candidates3_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail4
    (if 3 ≤ root228Candidates3.length then search edge 3 [1453, 2593] root228Candidates3 4046 winner else winner) = _
  rw [if_pos (by decide), root228Child3]
def root228Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root228Candidates4_checked : candidates edge 4 1453 2797 root228Tail5 winner.bound = root228Candidates4 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root228Step4 : scan edge (search edge 3) 4 [1453] 1453 root228Tail4 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail5 winner := by
  rw [root228Tail4, scan, if_neg (by decide), root228Candidates4_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail5
    (if 3 ≤ root228Candidates4.length then search edge 3 [1453, 2797] root228Candidates4 4250 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates5 : List Nat := [5581, 8089, 9787, 16111]
set_option maxRecDepth 8192 in
theorem root228Candidates5_checked : candidates edge 4 1453 3049 root228Tail6 winner.bound = root228Candidates5 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root228Child5 : search edge 3 [1453, 3049] root228Candidates5 4502 winner = winner := by decide
theorem root228Step5 : scan edge (search edge 3) 4 [1453] 1453 root228Tail5 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail6 winner := by
  rw [root228Tail5, scan, if_neg (by decide), root228Candidates5_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail6
    (if 3 ≤ root228Candidates5.length then search edge 3 [1453, 3049] root228Candidates5 4502 winner else winner) = _
  rw [if_pos (by decide), root228Child5]
def root228Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root228Candidates6_checked : candidates edge 4 1453 3361 root228Tail7 winner.bound = root228Candidates6 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root228Step6 : scan edge (search edge 3) 4 [1453] 1453 root228Tail6 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail7 winner := by
  rw [root228Tail6, scan, if_neg (by decide), root228Candidates6_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail7
    (if 3 ≤ root228Candidates6.length then search edge 3 [1453, 3361] root228Candidates6 4814 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates7 : List Nat := [5113]
set_option maxRecDepth 8192 in
theorem root228Candidates7_checked : candidates edge 4 1453 4483 root228Tail8 winner.bound = root228Candidates7 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root228Step7 : scan edge (search edge 3) 4 [1453] 1453 root228Tail7 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail8 winner := by
  rw [root228Tail7, scan, if_neg (by decide), root228Candidates7_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail8
    (if 3 ≤ root228Candidates7.length then search edge 3 [1453, 4483] root228Candidates7 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root228Candidates8_checked : candidates edge 4 1453 4651 root228Tail9 winner.bound = root228Candidates8 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root228Step8 : scan edge (search edge 3) 4 [1453] 1453 root228Tail8 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail9 winner := by
  rw [root228Tail8, scan, if_neg (by decide), root228Candidates8_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail9
    (if 3 ≤ root228Candidates8.length then search edge 3 [1453, 4651] root228Candidates8 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates9 : List Nat := [9787, 17971]
set_option maxRecDepth 8192 in
theorem root228Candidates9_checked : candidates edge 4 1453 4783 root228Tail10 winner.bound = root228Candidates9 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root228Step9 : scan edge (search edge 3) 4 [1453] 1453 root228Tail9 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail10 winner := by
  rw [root228Tail9, scan, if_neg (by decide), root228Candidates9_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail10
    (if 3 ≤ root228Candidates9.length then search edge 3 [1453, 4783] root228Candidates9 6236 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates10 : List Nat := [8929]
set_option maxRecDepth 8192 in
theorem root228Candidates10_checked : candidates edge 4 1453 5113 root228Tail11 winner.bound = root228Candidates10 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root228Step10 : scan edge (search edge 3) 4 [1453] 1453 root228Tail10 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail11 winner := by
  rw [root228Tail10, scan, if_neg (by decide), root228Candidates10_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail11
    (if 3 ≤ root228Candidates10.length then search edge 3 [1453, 5113] root228Candidates10 6566 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates11 : List Nat := [14107]
set_option maxRecDepth 8192 in
theorem root228Candidates11_checked : candidates edge 4 1453 5281 root228Tail12 winner.bound = root228Candidates11 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root228Step11 : scan edge (search edge 3) 4 [1453] 1453 root228Tail11 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail12 winner := by
  rw [root228Tail11, scan, if_neg (by decide), root228Candidates11_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail12
    (if 3 ≤ root228Candidates11.length then search edge 3 [1453, 5281] root228Candidates11 6734 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates12 : List Nat := [14107]
set_option maxRecDepth 8192 in
theorem root228Candidates12_checked : candidates edge 4 1453 5581 root228Tail13 winner.bound = root228Candidates12 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root228Step12 : scan edge (search edge 3) 4 [1453] 1453 root228Tail12 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail13 winner := by
  rw [root228Tail12, scan, if_neg (by decide), root228Candidates12_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail13
    (if 3 ≤ root228Candidates12.length then search edge 3 [1453, 5581] root228Candidates12 7034 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates13 : List Nat := [14107, 14551]
set_option maxRecDepth 8192 in
theorem root228Candidates13_checked : candidates edge 4 1453 5743 root228Tail14 winner.bound = root228Candidates13 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root228Step13 : scan edge (search edge 3) 4 [1453] 1453 root228Tail13 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail14 winner := by
  rw [root228Tail13, scan, if_neg (by decide), root228Candidates13_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail14
    (if 3 ≤ root228Candidates13.length then search edge 3 [1453, 5743] root228Candidates13 7196 winner else winner) = _
  rw [if_neg (by decide)]
def root228Candidates14 : List Nat := [7417, 8929, 16111]
set_option maxRecDepth 8192 in
theorem root228Candidates14_checked : candidates edge 4 1453 5923 root228Tail15 winner.bound = root228Candidates14 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root228Child14 : search edge 3 [1453, 5923] root228Candidates14 7376 winner = winner := by decide
theorem root228Step14 : scan edge (search edge 3) 4 [1453] 1453 root228Tail14 winner =
    scan edge (search edge 3) 4 [1453] 1453 root228Tail15 winner := by
  rw [root228Tail14, scan, if_neg (by decide), root228Candidates14_checked]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail15
    (if 3 ≤ root228Candidates14.length then search edge 3 [1453, 5923] root228Candidates14 7376 winner else winner) = _
  rw [if_pos (by decide), root228Child14]
theorem root228Scan15 : scan edge (search edge 3) 4 [1453] 1453 root228Tail15 winner = winner := by rw [root228Tail15, scan, if_pos (by decide)]
theorem root228Scan14 : scan edge (search edge 3) 4 [1453] 1453 root228Tail14 winner = winner :=
  (root228Step14).trans root228Scan15
theorem root228Scan13 : scan edge (search edge 3) 4 [1453] 1453 root228Tail13 winner = winner :=
  (root228Step13).trans root228Scan14
theorem root228Scan12 : scan edge (search edge 3) 4 [1453] 1453 root228Tail12 winner = winner :=
  (root228Step12).trans root228Scan13
theorem root228Scan11 : scan edge (search edge 3) 4 [1453] 1453 root228Tail11 winner = winner :=
  (root228Step11).trans root228Scan12
theorem root228Scan10 : scan edge (search edge 3) 4 [1453] 1453 root228Tail10 winner = winner :=
  (root228Step10).trans root228Scan11
theorem root228Scan9 : scan edge (search edge 3) 4 [1453] 1453 root228Tail9 winner = winner :=
  (root228Step9).trans root228Scan10
theorem root228Scan8 : scan edge (search edge 3) 4 [1453] 1453 root228Tail8 winner = winner :=
  (root228Step8).trans root228Scan9
theorem root228Scan7 : scan edge (search edge 3) 4 [1453] 1453 root228Tail7 winner = winner :=
  (root228Step7).trans root228Scan8
theorem root228Scan6 : scan edge (search edge 3) 4 [1453] 1453 root228Tail6 winner = winner :=
  (root228Step6).trans root228Scan7
theorem root228Scan5 : scan edge (search edge 3) 4 [1453] 1453 root228Tail5 winner = winner :=
  (root228Step5).trans root228Scan6
theorem root228Scan4 : scan edge (search edge 3) 4 [1453] 1453 root228Tail4 winner = winner :=
  (root228Step4).trans root228Scan5
theorem root228Scan3 : scan edge (search edge 3) 4 [1453] 1453 root228Tail3 winner = winner :=
  (root228Step3).trans root228Scan4
theorem root228Scan2 : scan edge (search edge 3) 4 [1453] 1453 root228Tail2 winner = winner :=
  (root228Step2).trans root228Scan3
theorem root228Scan1 : scan edge (search edge 3) 4 [1453] 1453 root228Tail1 winner = winner :=
  (root228Step1).trans root228Scan2
theorem root228Scan0 : scan edge (search edge 3) 4 [1453] 1453 root228Tail0 winner = winner :=
  (root228Step0).trans root228Scan1
set_option maxRecDepth 4096 in
theorem root228Branch : search edge 4 [1453] root228Candidates 1453 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1453] 1453 root228Tail0 winner = winner
  exact root228Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
