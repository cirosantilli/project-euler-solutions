import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root352Candidates : List Nat := Data.row2389
set_option maxRecDepth 8192 in
theorem root352Candidates_checked : candidates edge 5 0 2389 tail353 winner.bound = root352Candidates := by
  apply candidates_eq_row edge 5 0 2389 tail353 root352Candidates winner.bound tail353_ordered
  · decide
  · decide
  · funext q
    rfl
def root352Tail40 : List Nat := []
def root352Tail39 : List Nat := 21937 :: root352Tail40
def root352Tail38 : List Nat := 20707 :: root352Tail39
def root352Tail37 : List Nat := 20173 :: root352Tail38
def root352Tail36 : List Nat := 19381 :: root352Tail37
def root352Tail35 : List Nat := 19231 :: root352Tail36
def root352Tail34 : List Nat := 18523 :: root352Tail35
def root352Tail33 : List Nat := 17929 :: root352Tail34
def root352Tail32 : List Nat := 17659 :: root352Tail33
def root352Tail31 : List Nat := 16921 :: root352Tail32
def root352Tail30 : List Nat := 16063 :: root352Tail31
def root352Tail29 : List Nat := 15877 :: root352Tail30
def root352Tail28 : List Nat := 15661 :: root352Tail29
def root352Tail27 : List Nat := 13309 :: root352Tail28
def root352Tail26 : List Nat := 12823 :: root352Tail27
def root352Tail25 : List Nat := 12433 :: root352Tail26
def root352Tail24 : List Nat := 12301 :: root352Tail25
def root352Tail23 : List Nat := 12007 :: root352Tail24
def root352Tail22 : List Nat := 11731 :: root352Tail23
def root352Tail21 : List Nat := 10513 :: root352Tail22
def root352Tail20 : List Nat := 9697 :: root352Tail21
def root352Tail19 : List Nat := 9619 :: root352Tail20
def root352Tail18 : List Nat := 8803 :: root352Tail19
def root352Tail17 : List Nat := 8761 :: root352Tail18
def root352Tail16 : List Nat := 7561 :: root352Tail17
def root352Tail15 : List Nat := 7477 :: root352Tail16
def root352Tail14 : List Nat := 7351 :: root352Tail15
def root352Tail13 : List Nat := 6907 :: root352Tail14
def root352Tail12 : List Nat := 6871 :: root352Tail13
def root352Tail11 : List Nat := 6703 :: root352Tail12
def root352Tail10 : List Nat := 6529 :: root352Tail11
def root352Tail9 : List Nat := 6427 :: root352Tail10
def root352Tail8 : List Nat := 5479 :: root352Tail9
def root352Tail7 : List Nat := 5167 :: root352Tail8
def root352Tail6 : List Nat := 5023 :: root352Tail7
def root352Tail5 : List Nat := 4957 :: root352Tail6
def root352Tail4 : List Nat := 4243 :: root352Tail5
def root352Tail3 : List Nat := 3733 :: root352Tail4
def root352Tail2 : List Nat := 2767 :: root352Tail3
def root352Tail1 : List Nat := 2521 :: root352Tail2
def root352Tail0 : List Nat := 2503 :: root352Tail1
def root352Candidates0 : List Nat := [19231]
set_option maxRecDepth 8192 in
theorem root352Candidates0_checked : candidates edge 4 2389 2503 root352Tail1 winner.bound = root352Candidates0 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
theorem root352Step0 : scan edge (search edge 3) 4 [2389] 2389 root352Tail0 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail1 winner := by
  rw [root352Tail0, scan, if_neg (by decide), root352Candidates0_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail1
    (if 3 ≤ root352Candidates0.length then search edge 3 [2389, 2503] root352Candidates0 4892 winner else winner) = _
  rw [if_neg (by decide)]
def root352Candidates1 : List Nat := [7561]
set_option maxRecDepth 8192 in
theorem root352Candidates1_checked : candidates edge 4 2389 2521 root352Tail2 winner.bound = root352Candidates1 := by
  unfold candidates
  have row : edge 2521 = (SortedNatTree.contains FastRows.row2521) := by
    change Data.row2521.contains = (SortedNatTree.contains FastRows.row2521)
    exact FastRows.row2521_checked.symm
  rw [row]
  decide
theorem root352Step1 : scan edge (search edge 3) 4 [2389] 2389 root352Tail1 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail2 winner := by
  rw [root352Tail1, scan, if_neg (by decide), root352Candidates1_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail2
    (if 3 ≤ root352Candidates1.length then search edge 3 [2389, 2521] root352Candidates1 4910 winner else winner) = _
  rw [if_neg (by decide)]
def root352Candidates2 : List Nat := [3733, 7477, 7561]
set_option maxRecDepth 8192 in
theorem root352Candidates2_checked : candidates edge 4 2389 2767 root352Tail3 winner.bound = root352Candidates2 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root352Child2 : search edge 3 [2389, 2767] root352Candidates2 5156 winner = winner := by decide
theorem root352Step2 : scan edge (search edge 3) 4 [2389] 2389 root352Tail2 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail3 winner := by
  rw [root352Tail2, scan, if_neg (by decide), root352Candidates2_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail3
    (if 3 ≤ root352Candidates2.length then search edge 3 [2389, 2767] root352Candidates2 5156 winner else winner) = _
  rw [if_pos (by decide), root352Child2]
def root352Candidates3 : List Nat := [6529, 10513, 13309]
set_option maxRecDepth 8192 in
theorem root352Candidates3_checked : candidates edge 4 2389 3733 root352Tail4 winner.bound = root352Candidates3 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root352Child3 : search edge 3 [2389, 3733] root352Candidates3 6122 winner = winner := by decide
theorem root352Step3 : scan edge (search edge 3) 4 [2389] 2389 root352Tail3 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail4 winner := by
  rw [root352Tail3, scan, if_neg (by decide), root352Candidates3_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail4
    (if 3 ≤ root352Candidates3.length then search edge 3 [2389, 3733] root352Candidates3 6122 winner else winner) = _
  rw [if_pos (by decide), root352Child3]
def root352Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root352Candidates4_checked : candidates edge 4 2389 4243 root352Tail5 winner.bound = root352Candidates4 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root352Step4 : scan edge (search edge 3) 4 [2389] 2389 root352Tail4 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail5 winner := by
  rw [root352Tail4, scan, if_neg (by decide), root352Candidates4_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail5
    (if 3 ≤ root352Candidates4.length then search edge 3 [2389, 4243] root352Candidates4 6632 winner else winner) = _
  rw [if_neg (by decide)]
def root352Candidates5 : List Nat := [5023, 18523]
set_option maxRecDepth 8192 in
theorem root352Candidates5_checked : candidates edge 4 2389 4957 root352Tail6 winner.bound = root352Candidates5 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root352Step5 : scan edge (search edge 3) 4 [2389] 2389 root352Tail5 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail6 winner := by
  rw [root352Tail5, scan, if_neg (by decide), root352Candidates5_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail6
    (if 3 ≤ root352Candidates5.length then search edge 3 [2389, 4957] root352Candidates5 7346 winner else winner) = _
  rw [if_neg (by decide)]
def root352Candidates6 : List Nat := [7477]
set_option maxRecDepth 8192 in
theorem root352Candidates6_checked : candidates edge 4 2389 5023 root352Tail7 winner.bound = root352Candidates6 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root352Step6 : scan edge (search edge 3) 4 [2389] 2389 root352Tail6 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail7 winner := by
  rw [root352Tail6, scan, if_neg (by decide), root352Candidates6_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail7
    (if 3 ≤ root352Candidates6.length then search edge 3 [2389, 5023] root352Candidates6 7412 winner else winner) = _
  rw [if_neg (by decide)]
def root352Candidates7 : List Nat := [8761, 9619]
set_option maxRecDepth 8192 in
theorem root352Candidates7_checked : candidates edge 4 2389 5167 root352Tail8 winner.bound = root352Candidates7 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root352Step7 : scan edge (search edge 3) 4 [2389] 2389 root352Tail7 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail8 winner := by
  rw [root352Tail7, scan, if_neg (by decide), root352Candidates7_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail8
    (if 3 ≤ root352Candidates7.length then search edge 3 [2389, 5167] root352Candidates7 7556 winner else winner) = _
  rw [if_neg (by decide)]
def root352Candidates8 : List Nat := [16063, 17659]
set_option maxRecDepth 8192 in
theorem root352Candidates8_checked : candidates edge 4 2389 5479 root352Tail9 winner.bound = root352Candidates8 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root352Step8 : scan edge (search edge 3) 4 [2389] 2389 root352Tail8 winner =
    scan edge (search edge 3) 4 [2389] 2389 root352Tail9 winner := by
  rw [root352Tail8, scan, if_neg (by decide), root352Candidates8_checked]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail9
    (if 3 ≤ root352Candidates8.length then search edge 3 [2389, 5479] root352Candidates8 7868 winner else winner) = _
  rw [if_neg (by decide)]
theorem root352Scan9 : scan edge (search edge 3) 4 [2389] 2389 root352Tail9 winner = winner := by rw [root352Tail9, scan, if_pos (by decide)]
theorem root352Scan8 : scan edge (search edge 3) 4 [2389] 2389 root352Tail8 winner = winner :=
  (root352Step8).trans root352Scan9
theorem root352Scan7 : scan edge (search edge 3) 4 [2389] 2389 root352Tail7 winner = winner :=
  (root352Step7).trans root352Scan8
theorem root352Scan6 : scan edge (search edge 3) 4 [2389] 2389 root352Tail6 winner = winner :=
  (root352Step6).trans root352Scan7
theorem root352Scan5 : scan edge (search edge 3) 4 [2389] 2389 root352Tail5 winner = winner :=
  (root352Step5).trans root352Scan6
theorem root352Scan4 : scan edge (search edge 3) 4 [2389] 2389 root352Tail4 winner = winner :=
  (root352Step4).trans root352Scan5
theorem root352Scan3 : scan edge (search edge 3) 4 [2389] 2389 root352Tail3 winner = winner :=
  (root352Step3).trans root352Scan4
theorem root352Scan2 : scan edge (search edge 3) 4 [2389] 2389 root352Tail2 winner = winner :=
  (root352Step2).trans root352Scan3
theorem root352Scan1 : scan edge (search edge 3) 4 [2389] 2389 root352Tail1 winner = winner :=
  (root352Step1).trans root352Scan2
theorem root352Scan0 : scan edge (search edge 3) 4 [2389] 2389 root352Tail0 winner = winner :=
  (root352Step0).trans root352Scan1
set_option maxRecDepth 4096 in
theorem root352Branch : search edge 4 [2389] root352Candidates 2389 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2389] 2389 root352Tail0 winner = winner
  exact root352Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
