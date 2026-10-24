import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root289Candidates : List Nat := Data.row1907
set_option maxRecDepth 8192 in
theorem root289Candidates_checked : candidates edge 5 0 1907 tail290 winner.bound = root289Candidates := by
  apply candidates_eq_row edge 5 0 1907 tail290 root289Candidates winner.bound tail290_ordered
  · decide
  · decide
  · funext q
    rfl
def root289Tail46 : List Nat := []
def root289Tail45 : List Nat := 24077 :: root289Tail46
def root289Tail44 : List Nat := 23831 :: root289Tail45
def root289Tail43 : List Nat := 23399 :: root289Tail44
def root289Tail42 : List Nat := 22907 :: root289Tail43
def root289Tail41 : List Nat := 22619 :: root289Tail42
def root289Tail40 : List Nat := 22613 :: root289Tail41
def root289Tail39 : List Nat := 22277 :: root289Tail40
def root289Tail38 : List Nat := 22229 :: root289Tail39
def root289Tail37 : List Nat := 21977 :: root289Tail38
def root289Tail36 : List Nat := 21767 :: root289Tail37
def root289Tail35 : List Nat := 21191 :: root289Tail36
def root289Tail34 : List Nat := 20183 :: root289Tail35
def root289Tail33 : List Nat := 18251 :: root289Tail34
def root289Tail32 : List Nat := 17489 :: root289Tail33
def root289Tail31 : List Nat := 17327 :: root289Tail32
def root289Tail30 : List Nat := 16763 :: root289Tail31
def root289Tail29 : List Nat := 16301 :: root289Tail30
def root289Tail28 : List Nat := 16007 :: root289Tail29
def root289Tail27 : List Nat := 15731 :: root289Tail28
def root289Tail26 : List Nat := 14753 :: root289Tail27
def root289Tail25 : List Nat := 14543 :: root289Tail26
def root289Tail24 : List Nat := 14369 :: root289Tail25
def root289Tail23 : List Nat := 12227 :: root289Tail24
def root289Tail22 : List Nat := 11867 :: root289Tail23
def root289Tail21 : List Nat := 11273 :: root289Tail22
def root289Tail20 : List Nat := 10337 :: root289Tail21
def root289Tail19 : List Nat := 10103 :: root289Tail20
def root289Tail18 : List Nat := 9677 :: root289Tail19
def root289Tail17 : List Nat := 9341 :: root289Tail18
def root289Tail16 : List Nat := 9041 :: root289Tail17
def root289Tail15 : List Nat := 8867 :: root289Tail16
def root289Tail14 : List Nat := 8819 :: root289Tail15
def root289Tail13 : List Nat := 8231 :: root289Tail14
def root289Tail12 : List Nat := 7481 :: root289Tail13
def root289Tail11 : List Nat := 5897 :: root289Tail12
def root289Tail10 : List Nat := 5669 :: root289Tail11
def root289Tail9 : List Nat := 5099 :: root289Tail10
def root289Tail8 : List Nat := 4637 :: root289Tail9
def root289Tail7 : List Nat := 4127 :: root289Tail8
def root289Tail6 : List Nat := 3797 :: root289Tail7
def root289Tail5 : List Nat := 3581 :: root289Tail6
def root289Tail4 : List Nat := 3083 :: root289Tail5
def root289Tail3 : List Nat := 2777 :: root289Tail4
def root289Tail2 : List Nat := 2621 :: root289Tail3
def root289Tail1 : List Nat := 2243 :: root289Tail2
def root289Tail0 : List Nat := 2063 :: root289Tail1
def root289Candidates0 : List Nat := [2243]
set_option maxRecDepth 8192 in
theorem root289Candidates0_checked : candidates edge 4 1907 2063 root289Tail1 winner.bound = root289Candidates0 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root289Step0 : scan edge (search edge 3) 4 [1907] 1907 root289Tail0 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail1 winner := by
  rw [root289Tail0, scan, if_neg (by decide), root289Candidates0_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail1
    (if 3 ≤ root289Candidates0.length then search edge 3 [1907, 2063] root289Candidates0 3970 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates1 : List Nat := [18251, 21191]
set_option maxRecDepth 8192 in
theorem root289Candidates1_checked : candidates edge 4 1907 2243 root289Tail2 winner.bound = root289Candidates1 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
theorem root289Step1 : scan edge (search edge 3) 4 [1907] 1907 root289Tail1 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail2 winner := by
  rw [root289Tail1, scan, if_neg (by decide), root289Candidates1_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail2
    (if 3 ≤ root289Candidates1.length then search edge 3 [1907, 2243] root289Candidates1 4150 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates2 : List Nat := [16301]
set_option maxRecDepth 8192 in
theorem root289Candidates2_checked : candidates edge 4 1907 2621 root289Tail3 winner.bound = root289Candidates2 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root289Step2 : scan edge (search edge 3) 4 [1907] 1907 root289Tail2 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail3 winner := by
  rw [root289Tail2, scan, if_neg (by decide), root289Candidates2_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail3
    (if 3 ≤ root289Candidates2.length then search edge 3 [1907, 2621] root289Candidates2 4528 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates3 : List Nat := [8819]
set_option maxRecDepth 8192 in
theorem root289Candidates3_checked : candidates edge 4 1907 2777 root289Tail4 winner.bound = root289Candidates3 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root289Step3 : scan edge (search edge 3) 4 [1907] 1907 root289Tail3 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail4 winner := by
  rw [root289Tail3, scan, if_neg (by decide), root289Candidates3_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail4
    (if 3 ≤ root289Candidates3.length then search edge 3 [1907, 2777] root289Candidates3 4684 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates4 : List Nat := [3797, 18251]
set_option maxRecDepth 8192 in
theorem root289Candidates4_checked : candidates edge 4 1907 3083 root289Tail5 winner.bound = root289Candidates4 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root289Step4 : scan edge (search edge 3) 4 [1907] 1907 root289Tail4 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail5 winner := by
  rw [root289Tail4, scan, if_neg (by decide), root289Candidates4_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail5
    (if 3 ≤ root289Candidates4.length then search edge 3 [1907, 3083] root289Candidates4 4990 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates5 : List Nat := [16763]
set_option maxRecDepth 8192 in
theorem root289Candidates5_checked : candidates edge 4 1907 3581 root289Tail6 winner.bound = root289Candidates5 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root289Step5 : scan edge (search edge 3) 4 [1907] 1907 root289Tail5 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail6 winner := by
  rw [root289Tail5, scan, if_neg (by decide), root289Candidates5_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail6
    (if 3 ≤ root289Candidates5.length then search edge 3 [1907, 3581] root289Candidates5 5488 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root289Candidates6_checked : candidates edge 4 1907 3797 root289Tail7 winner.bound = root289Candidates6 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root289Step6 : scan edge (search edge 3) 4 [1907] 1907 root289Tail6 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail7 winner := by
  rw [root289Tail6, scan, if_neg (by decide), root289Candidates6_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail7
    (if 3 ≤ root289Candidates6.length then search edge 3 [1907, 3797] root289Candidates6 5704 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates7 : List Nat := [17489]
set_option maxRecDepth 8192 in
theorem root289Candidates7_checked : candidates edge 4 1907 4127 root289Tail8 winner.bound = root289Candidates7 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root289Step7 : scan edge (search edge 3) 4 [1907] 1907 root289Tail7 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail8 winner := by
  rw [root289Tail7, scan, if_neg (by decide), root289Candidates7_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail8
    (if 3 ≤ root289Candidates7.length then search edge 3 [1907, 4127] root289Candidates7 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates8 : List Nat := [9041, 17489]
set_option maxRecDepth 8192 in
theorem root289Candidates8_checked : candidates edge 4 1907 4637 root289Tail9 winner.bound = root289Candidates8 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root289Step8 : scan edge (search edge 3) 4 [1907] 1907 root289Tail8 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail9 winner := by
  rw [root289Tail8, scan, if_neg (by decide), root289Candidates8_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail9
    (if 3 ≤ root289Candidates8.length then search edge 3 [1907, 4637] root289Candidates8 6544 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates9 : List Nat := [8231]
set_option maxRecDepth 8192 in
theorem root289Candidates9_checked : candidates edge 4 1907 5099 root289Tail10 winner.bound = root289Candidates9 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root289Step9 : scan edge (search edge 3) 4 [1907] 1907 root289Tail9 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail10 winner := by
  rw [root289Tail9, scan, if_neg (by decide), root289Candidates9_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail10
    (if 3 ≤ root289Candidates9.length then search edge 3 [1907, 5099] root289Candidates9 7006 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates10 : List Nat := [8819]
set_option maxRecDepth 8192 in
theorem root289Candidates10_checked : candidates edge 4 1907 5669 root289Tail11 winner.bound = root289Candidates10 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root289Step10 : scan edge (search edge 3) 4 [1907] 1907 root289Tail10 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail11 winner := by
  rw [root289Tail10, scan, if_neg (by decide), root289Candidates10_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail11
    (if 3 ≤ root289Candidates10.length then search edge 3 [1907, 5669] root289Candidates10 7576 winner else winner) = _
  rw [if_neg (by decide)]
def root289Candidates11 : List Nat := [8867, 17327]
set_option maxRecDepth 8192 in
theorem root289Candidates11_checked : candidates edge 4 1907 5897 root289Tail12 winner.bound = root289Candidates11 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root289Step11 : scan edge (search edge 3) 4 [1907] 1907 root289Tail11 winner =
    scan edge (search edge 3) 4 [1907] 1907 root289Tail12 winner := by
  rw [root289Tail11, scan, if_neg (by decide), root289Candidates11_checked]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail12
    (if 3 ≤ root289Candidates11.length then search edge 3 [1907, 5897] root289Candidates11 7804 winner else winner) = _
  rw [if_neg (by decide)]
theorem root289Scan12 : scan edge (search edge 3) 4 [1907] 1907 root289Tail12 winner = winner := by rw [root289Tail12, scan, if_pos (by decide)]
theorem root289Scan11 : scan edge (search edge 3) 4 [1907] 1907 root289Tail11 winner = winner :=
  (root289Step11).trans root289Scan12
theorem root289Scan10 : scan edge (search edge 3) 4 [1907] 1907 root289Tail10 winner = winner :=
  (root289Step10).trans root289Scan11
theorem root289Scan9 : scan edge (search edge 3) 4 [1907] 1907 root289Tail9 winner = winner :=
  (root289Step9).trans root289Scan10
theorem root289Scan8 : scan edge (search edge 3) 4 [1907] 1907 root289Tail8 winner = winner :=
  (root289Step8).trans root289Scan9
theorem root289Scan7 : scan edge (search edge 3) 4 [1907] 1907 root289Tail7 winner = winner :=
  (root289Step7).trans root289Scan8
theorem root289Scan6 : scan edge (search edge 3) 4 [1907] 1907 root289Tail6 winner = winner :=
  (root289Step6).trans root289Scan7
theorem root289Scan5 : scan edge (search edge 3) 4 [1907] 1907 root289Tail5 winner = winner :=
  (root289Step5).trans root289Scan6
theorem root289Scan4 : scan edge (search edge 3) 4 [1907] 1907 root289Tail4 winner = winner :=
  (root289Step4).trans root289Scan5
theorem root289Scan3 : scan edge (search edge 3) 4 [1907] 1907 root289Tail3 winner = winner :=
  (root289Step3).trans root289Scan4
theorem root289Scan2 : scan edge (search edge 3) 4 [1907] 1907 root289Tail2 winner = winner :=
  (root289Step2).trans root289Scan3
theorem root289Scan1 : scan edge (search edge 3) 4 [1907] 1907 root289Tail1 winner = winner :=
  (root289Step1).trans root289Scan2
theorem root289Scan0 : scan edge (search edge 3) 4 [1907] 1907 root289Tail0 winner = winner :=
  (root289Step0).trans root289Scan1
set_option maxRecDepth 4096 in
theorem root289Branch : search edge 4 [1907] root289Candidates 1907 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1907] 1907 root289Tail0 winner = winner
  exact root289Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
