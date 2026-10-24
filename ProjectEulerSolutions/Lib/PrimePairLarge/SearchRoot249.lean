import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root249Candidates : List Nat := Data.row1601
set_option maxRecDepth 8192 in
theorem root249Candidates_checked : candidates edge 5 0 1601 tail250 winner.bound = root249Candidates := by
  apply candidates_eq_row edge 5 0 1601 tail250 root249Candidates winner.bound tail250_ordered
  · decide
  · decide
  · funext q
    rfl
def root249Tail49 : List Nat := []
def root249Tail48 : List Nat := 24281 :: root249Tail49
def root249Tail47 : List Nat := 24023 :: root249Tail48
def root249Tail46 : List Nat := 23981 :: root249Tail47
def root249Tail45 : List Nat := 23333 :: root249Tail46
def root249Tail44 : List Nat := 22769 :: root249Tail45
def root249Tail43 : List Nat := 22727 :: root249Tail44
def root249Tail42 : List Nat := 22307 :: root249Tail43
def root249Tail41 : List Nat := 21881 :: root249Tail42
def root249Tail40 : List Nat := 21821 :: root249Tail41
def root249Tail39 : List Nat := 19763 :: root249Tail40
def root249Tail38 : List Nat := 18269 :: root249Tail39
def root249Tail37 : List Nat := 16673 :: root249Tail38
def root249Tail36 : List Nat := 16301 :: root249Tail37
def root249Tail35 : List Nat := 16067 :: root249Tail36
def root249Tail34 : List Nat := 15233 :: root249Tail35
def root249Tail33 : List Nat := 15017 :: root249Tail34
def root249Tail32 : List Nat := 14951 :: root249Tail33
def root249Tail31 : List Nat := 14897 :: root249Tail32
def root249Tail30 : List Nat := 14621 :: root249Tail31
def root249Tail29 : List Nat := 14489 :: root249Tail30
def root249Tail28 : List Nat := 14369 :: root249Tail29
def root249Tail27 : List Nat := 13103 :: root249Tail28
def root249Tail26 : List Nat := 11969 :: root249Tail27
def root249Tail25 : List Nat := 11447 :: root249Tail26
def root249Tail24 : List Nat := 11261 :: root249Tail25
def root249Tail23 : List Nat := 10463 :: root249Tail24
def root249Tail22 : List Nat := 10193 :: root249Tail23
def root249Tail21 : List Nat := 9749 :: root249Tail22
def root249Tail20 : List Nat := 9473 :: root249Tail21
def root249Tail19 : List Nat := 9257 :: root249Tail20
def root249Tail18 : List Nat := 9203 :: root249Tail19
def root249Tail17 : List Nat := 9011 :: root249Tail18
def root249Tail16 : List Nat := 8837 :: root249Tail17
def root249Tail15 : List Nat := 8819 :: root249Tail16
def root249Tail14 : List Nat := 8753 :: root249Tail15
def root249Tail13 : List Nat := 7307 :: root249Tail14
def root249Tail12 : List Nat := 7229 :: root249Tail13
def root249Tail11 : List Nat := 6701 :: root249Tail12
def root249Tail10 : List Nat := 6359 :: root249Tail11
def root249Tail9 : List Nat := 6113 :: root249Tail10
def root249Tail8 : List Nat := 6029 :: root249Tail9
def root249Tail7 : List Nat := 4637 :: root249Tail8
def root249Tail6 : List Nat := 3533 :: root249Tail7
def root249Tail5 : List Nat := 2879 :: root249Tail6
def root249Tail4 : List Nat := 2207 :: root249Tail5
def root249Tail3 : List Nat := 2063 :: root249Tail4
def root249Tail2 : List Nat := 1847 :: root249Tail3
def root249Tail1 : List Nat := 1823 :: root249Tail2
def root249Tail0 : List Nat := 1811 :: root249Tail1
def root249Candidates0 : List Nat := [21821]
set_option maxRecDepth 8192 in
theorem root249Candidates0_checked : candidates edge 4 1601 1811 root249Tail1 winner.bound = root249Candidates0 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
theorem root249Step0 : scan edge (search edge 3) 4 [1601] 1601 root249Tail0 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail1 winner := by
  rw [root249Tail0, scan, if_neg (by decide), root249Candidates0_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail1
    (if 3 ≤ root249Candidates0.length then search edge 3 [1601, 1811] root249Candidates0 3412 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates1 : List Nat := [2063]
set_option maxRecDepth 8192 in
theorem root249Candidates1_checked : candidates edge 4 1601 1823 root249Tail2 winner.bound = root249Candidates1 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
theorem root249Step1 : scan edge (search edge 3) 4 [1601] 1601 root249Tail1 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail2 winner := by
  rw [root249Tail1, scan, if_neg (by decide), root249Candidates1_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail2
    (if 3 ≤ root249Candidates1.length then search edge 3 [1601, 1823] root249Candidates1 3424 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates2 : List Nat := [7307]
set_option maxRecDepth 8192 in
theorem root249Candidates2_checked : candidates edge 4 1601 1847 root249Tail3 winner.bound = root249Candidates2 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
theorem root249Step2 : scan edge (search edge 3) 4 [1601] 1601 root249Tail2 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail3 winner := by
  rw [root249Tail2, scan, if_neg (by decide), root249Candidates2_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail3
    (if 3 ≤ root249Candidates2.length then search edge 3 [1601, 1847] root249Candidates2 3448 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates3 : List Nat := [14621, 16673]
set_option maxRecDepth 8192 in
theorem root249Candidates3_checked : candidates edge 4 1601 2063 root249Tail4 winner.bound = root249Candidates3 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root249Step3 : scan edge (search edge 3) 4 [1601] 1601 root249Tail3 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail4 winner := by
  rw [root249Tail3, scan, if_neg (by decide), root249Candidates3_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail4
    (if 3 ≤ root249Candidates3.length then search edge 3 [1601, 2063] root249Candidates3 3664 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates4 : List Nat := [15017]
set_option maxRecDepth 8192 in
theorem root249Candidates4_checked : candidates edge 4 1601 2207 root249Tail5 winner.bound = root249Candidates4 := by
  unfold candidates
  have row : edge 2207 = (SortedNatTree.contains FastRows.row2207) := by
    change Data.row2207.contains = (SortedNatTree.contains FastRows.row2207)
    exact FastRows.row2207_checked.symm
  rw [row]
  decide
theorem root249Step4 : scan edge (search edge 3) 4 [1601] 1601 root249Tail4 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail5 winner := by
  rw [root249Tail4, scan, if_neg (by decide), root249Candidates4_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail5
    (if 3 ≤ root249Candidates4.length then search edge 3 [1601, 2207] root249Candidates4 3808 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates5 : List Nat := [6701, 10193]
set_option maxRecDepth 8192 in
theorem root249Candidates5_checked : candidates edge 4 1601 2879 root249Tail6 winner.bound = root249Candidates5 := by
  unfold candidates
  have row : edge 2879 = (SortedNatTree.contains FastRows.row2879) := by
    change Data.row2879.contains = (SortedNatTree.contains FastRows.row2879)
    exact FastRows.row2879_checked.symm
  rw [row]
  decide
theorem root249Step5 : scan edge (search edge 3) 4 [1601] 1601 root249Tail5 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail6 winner := by
  rw [root249Tail5, scan, if_neg (by decide), root249Candidates5_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail6
    (if 3 ≤ root249Candidates5.length then search edge 3 [1601, 2879] root249Candidates5 4480 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates6 : List Nat := [6113]
set_option maxRecDepth 8192 in
theorem root249Candidates6_checked : candidates edge 4 1601 3533 root249Tail7 winner.bound = root249Candidates6 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root249Step6 : scan edge (search edge 3) 4 [1601] 1601 root249Tail6 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail7 winner := by
  rw [root249Tail6, scan, if_neg (by decide), root249Candidates6_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail7
    (if 3 ≤ root249Candidates6.length then search edge 3 [1601, 3533] root249Candidates6 5134 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root249Candidates7_checked : candidates edge 4 1601 4637 root249Tail8 winner.bound = root249Candidates7 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root249Step7 : scan edge (search edge 3) 4 [1601] 1601 root249Tail7 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail8 winner := by
  rw [root249Tail7, scan, if_neg (by decide), root249Candidates7_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail8
    (if 3 ≤ root249Candidates7.length then search edge 3 [1601, 4637] root249Candidates7 6238 winner else winner) = _
  rw [if_neg (by decide)]
def root249Candidates8 : List Nat := [9203, 11447, 14369, 15233, 16673, 18269]
set_option maxRecDepth 8192 in
theorem root249Candidates8_checked : candidates edge 4 1601 6029 root249Tail9 winner.bound = root249Candidates8 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root249Child8 : search edge 3 [1601, 6029] root249Candidates8 7630 winner = winner := by decide
theorem root249Step8 : scan edge (search edge 3) 4 [1601] 1601 root249Tail8 winner =
    scan edge (search edge 3) 4 [1601] 1601 root249Tail9 winner := by
  rw [root249Tail8, scan, if_neg (by decide), root249Candidates8_checked]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail9
    (if 3 ≤ root249Candidates8.length then search edge 3 [1601, 6029] root249Candidates8 7630 winner else winner) = _
  rw [if_pos (by decide), root249Child8]
theorem root249Scan9 : scan edge (search edge 3) 4 [1601] 1601 root249Tail9 winner = winner := by rw [root249Tail9, scan, if_pos (by decide)]
theorem root249Scan8 : scan edge (search edge 3) 4 [1601] 1601 root249Tail8 winner = winner :=
  (root249Step8).trans root249Scan9
theorem root249Scan7 : scan edge (search edge 3) 4 [1601] 1601 root249Tail7 winner = winner :=
  (root249Step7).trans root249Scan8
theorem root249Scan6 : scan edge (search edge 3) 4 [1601] 1601 root249Tail6 winner = winner :=
  (root249Step6).trans root249Scan7
theorem root249Scan5 : scan edge (search edge 3) 4 [1601] 1601 root249Tail5 winner = winner :=
  (root249Step5).trans root249Scan6
theorem root249Scan4 : scan edge (search edge 3) 4 [1601] 1601 root249Tail4 winner = winner :=
  (root249Step4).trans root249Scan5
theorem root249Scan3 : scan edge (search edge 3) 4 [1601] 1601 root249Tail3 winner = winner :=
  (root249Step3).trans root249Scan4
theorem root249Scan2 : scan edge (search edge 3) 4 [1601] 1601 root249Tail2 winner = winner :=
  (root249Step2).trans root249Scan3
theorem root249Scan1 : scan edge (search edge 3) 4 [1601] 1601 root249Tail1 winner = winner :=
  (root249Step1).trans root249Scan2
theorem root249Scan0 : scan edge (search edge 3) 4 [1601] 1601 root249Tail0 winner = winner :=
  (root249Step0).trans root249Scan1
set_option maxRecDepth 4096 in
theorem root249Branch : search edge 4 [1601] root249Candidates 1601 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1601] 1601 root249Tail0 winner = winner
  exact root249Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
