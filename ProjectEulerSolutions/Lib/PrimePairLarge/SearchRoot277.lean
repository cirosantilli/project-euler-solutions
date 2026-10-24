import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root277Candidates : List Nat := Data.row1811
set_option maxRecDepth 8192 in
theorem root277Candidates_checked : candidates edge 5 0 1811 tail278 winner.bound = root277Candidates := by
  apply candidates_eq_row edge 5 0 1811 tail278 root277Candidates winner.bound tail278_ordered
  · decide
  · decide
  · funext q
    rfl
def root277Tail50 : List Nat := []
def root277Tail49 : List Nat := 24113 :: root277Tail50
def root277Tail48 : List Nat := 24107 :: root277Tail49
def root277Tail47 : List Nat := 24071 :: root277Tail48
def root277Tail46 : List Nat := 23417 :: root277Tail47
def root277Tail45 : List Nat := 23081 :: root277Tail46
def root277Tail44 : List Nat := 23057 :: root277Tail45
def root277Tail43 : List Nat := 22937 :: root277Tail44
def root277Tail42 : List Nat := 21911 :: root277Tail43
def root277Tail41 : List Nat := 21821 :: root277Tail42
def root277Tail40 : List Nat := 21491 :: root277Tail41
def root277Tail39 : List Nat := 19889 :: root277Tail40
def root277Tail38 : List Nat := 18713 :: root277Tail39
def root277Tail37 : List Nat := 17477 :: root277Tail38
def root277Tail36 : List Nat := 17207 :: root277Tail37
def root277Tail35 : List Nat := 16631 :: root277Tail36
def root277Tail34 : List Nat := 16619 :: root277Tail35
def root277Tail33 : List Nat := 16547 :: root277Tail34
def root277Tail32 : List Nat := 16451 :: root277Tail33
def root277Tail31 : List Nat := 15881 :: root277Tail32
def root277Tail30 : List Nat := 15287 :: root277Tail31
def root277Tail29 : List Nat := 14699 :: root277Tail30
def root277Tail28 : List Nat := 14159 :: root277Tail29
def root277Tail27 : List Nat := 14033 :: root277Tail28
def root277Tail26 : List Nat := 14009 :: root277Tail27
def root277Tail25 : List Nat := 13901 :: root277Tail26
def root277Tail24 : List Nat := 13859 :: root277Tail25
def root277Tail23 : List Nat := 12263 :: root277Tail24
def root277Tail22 : List Nat := 12011 :: root277Tail23
def root277Tail21 : List Nat := 11423 :: root277Tail22
def root277Tail20 : List Nat := 10799 :: root277Tail21
def root277Tail19 : List Nat := 10733 :: root277Tail20
def root277Tail18 : List Nat := 10691 :: root277Tail19
def root277Tail17 : List Nat := 9281 :: root277Tail18
def root277Tail16 : List Nat := 8699 :: root277Tail17
def root277Tail15 : List Nat := 8669 :: root277Tail16
def root277Tail14 : List Nat := 7817 :: root277Tail15
def root277Tail13 : List Nat := 7559 :: root277Tail14
def root277Tail12 : List Nat := 6863 :: root277Tail13
def root277Tail11 : List Nat := 6653 :: root277Tail12
def root277Tail10 : List Nat := 6473 :: root277Tail11
def root277Tail9 : List Nat := 5381 :: root277Tail10
def root277Tail8 : List Nat := 5297 :: root277Tail9
def root277Tail7 : List Nat := 5039 :: root277Tail8
def root277Tail6 : List Nat := 3257 :: root277Tail7
def root277Tail5 : List Nat := 2609 :: root277Tail6
def root277Tail4 : List Nat := 2441 :: root277Tail5
def root277Tail3 : List Nat := 2333 :: root277Tail4
def root277Tail2 : List Nat := 2141 :: root277Tail3
def root277Tail1 : List Nat := 1949 :: root277Tail2
def root277Tail0 : List Nat := 1931 :: root277Tail1
def root277Candidates0 : List Nat := [5297, 15287]
set_option maxRecDepth 8192 in
theorem root277Candidates0_checked : candidates edge 4 1811 1931 root277Tail1 winner.bound = root277Candidates0 := by
  unfold candidates
  have row : edge 1931 = (SortedNatTree.contains FastRows.row1931) := by
    change Data.row1931.contains = (SortedNatTree.contains FastRows.row1931)
    exact FastRows.row1931_checked.symm
  rw [row]
  decide
theorem root277Step0 : scan edge (search edge 3) 4 [1811] 1811 root277Tail0 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail1 winner := by
  rw [root277Tail0, scan, if_neg (by decide), root277Candidates0_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail1
    (if 3 ≤ root277Candidates0.length then search edge 3 [1811, 1931] root277Candidates0 3742 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates1 : List Nat := [6863]
set_option maxRecDepth 8192 in
theorem root277Candidates1_checked : candidates edge 4 1811 1949 root277Tail2 winner.bound = root277Candidates1 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root277Step1 : scan edge (search edge 3) 4 [1811] 1811 root277Tail1 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail2 winner := by
  rw [root277Tail1, scan, if_neg (by decide), root277Candidates1_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail2
    (if 3 ≤ root277Candidates1.length then search edge 3 [1811, 1949] root277Candidates1 3760 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates2 : List Nat := [13901, 17207]
set_option maxRecDepth 8192 in
theorem root277Candidates2_checked : candidates edge 4 1811 2141 root277Tail3 winner.bound = root277Candidates2 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root277Step2 : scan edge (search edge 3) 4 [1811] 1811 root277Tail2 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail3 winner := by
  rw [root277Tail2, scan, if_neg (by decide), root277Candidates2_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail3
    (if 3 ≤ root277Candidates2.length then search edge 3 [1811, 2141] root277Candidates2 3952 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates3 : List Nat := [5297, 15881, 16451]
set_option maxRecDepth 8192 in
theorem root277Candidates3_checked : candidates edge 4 1811 2333 root277Tail4 winner.bound = root277Candidates3 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root277Child3 : search edge 3 [1811, 2333] root277Candidates3 4144 winner = winner := by decide
theorem root277Step3 : scan edge (search edge 3) 4 [1811] 1811 root277Tail3 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail4 winner := by
  rw [root277Tail3, scan, if_neg (by decide), root277Candidates3_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail4
    (if 3 ≤ root277Candidates3.length then search edge 3 [1811, 2333] root277Candidates3 4144 winner else winner) = _
  rw [if_pos (by decide), root277Child3]
def root277Candidates4 : List Nat := [5297, 5381, 11423, 17207]
set_option maxRecDepth 8192 in
theorem root277Candidates4_checked : candidates edge 4 1811 2441 root277Tail5 winner.bound = root277Candidates4 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root277Child4 : search edge 3 [1811, 2441] root277Candidates4 4252 winner = winner := by decide
theorem root277Step4 : scan edge (search edge 3) 4 [1811] 1811 root277Tail4 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail5 winner := by
  rw [root277Tail4, scan, if_neg (by decide), root277Candidates4_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail5
    (if 3 ≤ root277Candidates4.length then search edge 3 [1811, 2441] root277Candidates4 4252 winner else winner) = _
  rw [if_pos (by decide), root277Child4]
def root277Candidates5 : List Nat := [3257, 14033]
set_option maxRecDepth 8192 in
theorem root277Candidates5_checked : candidates edge 4 1811 2609 root277Tail6 winner.bound = root277Candidates5 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
theorem root277Step5 : scan edge (search edge 3) 4 [1811] 1811 root277Tail5 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail6 winner := by
  rw [root277Tail5, scan, if_neg (by decide), root277Candidates5_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail6
    (if 3 ≤ root277Candidates5.length then search edge 3 [1811, 2609] root277Candidates5 4420 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates6 : List Nat := [18713]
set_option maxRecDepth 8192 in
theorem root277Candidates6_checked : candidates edge 4 1811 3257 root277Tail7 winner.bound = root277Candidates6 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root277Step6 : scan edge (search edge 3) 4 [1811] 1811 root277Tail6 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail7 winner := by
  rw [root277Tail6, scan, if_neg (by decide), root277Candidates6_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail7
    (if 3 ≤ root277Candidates6.length then search edge 3 [1811, 3257] root277Candidates6 5068 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root277Candidates7_checked : candidates edge 4 1811 5039 root277Tail8 winner.bound = root277Candidates7 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root277Step7 : scan edge (search edge 3) 4 [1811] 1811 root277Tail7 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail8 winner := by
  rw [root277Tail7, scan, if_neg (by decide), root277Candidates7_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail8
    (if 3 ≤ root277Candidates7.length then search edge 3 [1811, 5039] root277Candidates7 6850 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates8 : List Nat := [18713]
set_option maxRecDepth 8192 in
theorem root277Candidates8_checked : candidates edge 4 1811 5297 root277Tail9 winner.bound = root277Candidates8 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root277Step8 : scan edge (search edge 3) 4 [1811] 1811 root277Tail8 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail9 winner := by
  rw [root277Tail8, scan, if_neg (by decide), root277Candidates8_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail9
    (if 3 ≤ root277Candidates8.length then search edge 3 [1811, 5297] root277Candidates8 7108 winner else winner) = _
  rw [if_neg (by decide)]
def root277Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root277Candidates9_checked : candidates edge 4 1811 5381 root277Tail10 winner.bound = root277Candidates9 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root277Step9 : scan edge (search edge 3) 4 [1811] 1811 root277Tail9 winner =
    scan edge (search edge 3) 4 [1811] 1811 root277Tail10 winner := by
  rw [root277Tail9, scan, if_neg (by decide), root277Candidates9_checked]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail10
    (if 3 ≤ root277Candidates9.length then search edge 3 [1811, 5381] root277Candidates9 7192 winner else winner) = _
  rw [if_neg (by decide)]
theorem root277Scan10 : scan edge (search edge 3) 4 [1811] 1811 root277Tail10 winner = winner := by rw [root277Tail10, scan, if_pos (by decide)]
theorem root277Scan9 : scan edge (search edge 3) 4 [1811] 1811 root277Tail9 winner = winner :=
  (root277Step9).trans root277Scan10
theorem root277Scan8 : scan edge (search edge 3) 4 [1811] 1811 root277Tail8 winner = winner :=
  (root277Step8).trans root277Scan9
theorem root277Scan7 : scan edge (search edge 3) 4 [1811] 1811 root277Tail7 winner = winner :=
  (root277Step7).trans root277Scan8
theorem root277Scan6 : scan edge (search edge 3) 4 [1811] 1811 root277Tail6 winner = winner :=
  (root277Step6).trans root277Scan7
theorem root277Scan5 : scan edge (search edge 3) 4 [1811] 1811 root277Tail5 winner = winner :=
  (root277Step5).trans root277Scan6
theorem root277Scan4 : scan edge (search edge 3) 4 [1811] 1811 root277Tail4 winner = winner :=
  (root277Step4).trans root277Scan5
theorem root277Scan3 : scan edge (search edge 3) 4 [1811] 1811 root277Tail3 winner = winner :=
  (root277Step3).trans root277Scan4
theorem root277Scan2 : scan edge (search edge 3) 4 [1811] 1811 root277Tail2 winner = winner :=
  (root277Step2).trans root277Scan3
theorem root277Scan1 : scan edge (search edge 3) 4 [1811] 1811 root277Tail1 winner = winner :=
  (root277Step1).trans root277Scan2
theorem root277Scan0 : scan edge (search edge 3) 4 [1811] 1811 root277Tail0 winner = winner :=
  (root277Step0).trans root277Scan1
set_option maxRecDepth 4096 in
theorem root277Branch : search edge 4 [1811] root277Candidates 1811 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1811] 1811 root277Tail0 winner = winner
  exact root277Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
