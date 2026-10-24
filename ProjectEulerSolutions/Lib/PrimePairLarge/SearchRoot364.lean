import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root364Candidates : List Nat := Data.row2477
set_option maxRecDepth 8192 in
theorem root364Candidates_checked : candidates edge 5 0 2477 tail365 winner.bound = root364Candidates := by
  apply candidates_eq_row edge 5 0 2477 tail365 root364Candidates winner.bound tail365_ordered
  · decide
  · decide
  · funext q
    rfl
def root364Tail38 : List Nat := []
def root364Tail37 : List Nat := 19013 :: root364Tail38
def root364Tail36 : List Nat := 18617 :: root364Tail37
def root364Tail35 : List Nat := 17891 :: root364Tail36
def root364Tail34 : List Nat := 17027 :: root364Tail35
def root364Tail33 : List Nat := 16871 :: root364Tail34
def root364Tail32 : List Nat := 16607 :: root364Tail33
def root364Tail31 : List Nat := 15737 :: root364Tail32
def root364Tail30 : List Nat := 15527 :: root364Tail31
def root364Tail29 : List Nat := 14591 :: root364Tail30
def root364Tail28 : List Nat := 13859 :: root364Tail29
def root364Tail27 : List Nat := 13751 :: root364Tail28
def root364Tail26 : List Nat := 13577 :: root364Tail27
def root364Tail25 : List Nat := 13229 :: root364Tail26
def root364Tail24 : List Nat := 13217 :: root364Tail25
def root364Tail23 : List Nat := 12821 :: root364Tail24
def root364Tail22 : List Nat := 12251 :: root364Tail23
def root364Tail21 : List Nat := 12071 :: root364Tail22
def root364Tail20 : List Nat := 11549 :: root364Tail21
def root364Tail19 : List Nat := 10601 :: root364Tail20
def root364Tail18 : List Nat := 9929 :: root364Tail19
def root364Tail17 : List Nat := 9719 :: root364Tail18
def root364Tail16 : List Nat := 9437 :: root364Tail17
def root364Tail15 : List Nat := 9341 :: root364Tail16
def root364Tail14 : List Nat := 8807 :: root364Tail15
def root364Tail13 : List Nat := 8783 :: root364Tail14
def root364Tail12 : List Nat := 8681 :: root364Tail13
def root364Tail11 : List Nat := 8669 :: root364Tail12
def root364Tail10 : List Nat := 8513 :: root364Tail11
def root364Tail9 : List Nat := 7853 :: root364Tail10
def root364Tail8 : List Nat := 7703 :: root364Tail9
def root364Tail7 : List Nat := 5381 :: root364Tail8
def root364Tail6 : List Nat := 5039 :: root364Tail7
def root364Tail5 : List Nat := 4133 :: root364Tail6
def root364Tail4 : List Nat := 4073 :: root364Tail5
def root364Tail3 : List Nat := 3797 :: root364Tail4
def root364Tail2 : List Nat := 2939 :: root364Tail3
def root364Tail1 : List Nat := 2801 :: root364Tail2
def root364Tail0 : List Nat := 2663 :: root364Tail1
def root364Candidates0 : List Nat := [8783]
set_option maxRecDepth 8192 in
theorem root364Candidates0_checked : candidates edge 4 2477 2663 root364Tail1 winner.bound = root364Candidates0 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root364Step0 : scan edge (search edge 3) 4 [2477] 2477 root364Tail0 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail1 winner := by
  rw [root364Tail0, scan, if_neg (by decide), root364Candidates0_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail1
    (if 3 ≤ root364Candidates0.length then search edge 3 [2477, 2663] root364Candidates0 5140 winner else winner) = _
  rw [if_neg (by decide)]
def root364Candidates1 : List Nat := [4073]
set_option maxRecDepth 8192 in
theorem root364Candidates1_checked : candidates edge 4 2477 2801 root364Tail2 winner.bound = root364Candidates1 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
theorem root364Step1 : scan edge (search edge 3) 4 [2477] 2477 root364Tail1 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail2 winner := by
  rw [root364Tail1, scan, if_neg (by decide), root364Candidates1_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail2
    (if 3 ≤ root364Candidates1.length then search edge 3 [2477, 2801] root364Candidates1 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root364Candidates2 : List Nat := [4133, 8783, 12821]
set_option maxRecDepth 8192 in
theorem root364Candidates2_checked : candidates edge 4 2477 2939 root364Tail3 winner.bound = root364Candidates2 := by
  unfold candidates
  have row : edge 2939 = (SortedNatTree.contains FastRows.row2939) := by
    change Data.row2939.contains = (SortedNatTree.contains FastRows.row2939)
    exact FastRows.row2939_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root364Child2 : search edge 3 [2477, 2939] root364Candidates2 5416 winner = winner := by decide
theorem root364Step2 : scan edge (search edge 3) 4 [2477] 2477 root364Tail2 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail3 winner := by
  rw [root364Tail2, scan, if_neg (by decide), root364Candidates2_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail3
    (if 3 ≤ root364Candidates2.length then search edge 3 [2477, 2939] root364Candidates2 5416 winner else winner) = _
  rw [if_pos (by decide), root364Child2]
def root364Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root364Candidates3_checked : candidates edge 4 2477 3797 root364Tail4 winner.bound = root364Candidates3 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root364Step3 : scan edge (search edge 3) 4 [2477] 2477 root364Tail3 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail4 winner := by
  rw [root364Tail3, scan, if_neg (by decide), root364Candidates3_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail4
    (if 3 ≤ root364Candidates3.length then search edge 3 [2477, 3797] root364Candidates3 6274 winner else winner) = _
  rw [if_neg (by decide)]
def root364Candidates4 : List Nat := [9341]
set_option maxRecDepth 8192 in
theorem root364Candidates4_checked : candidates edge 4 2477 4073 root364Tail5 winner.bound = root364Candidates4 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root364Step4 : scan edge (search edge 3) 4 [2477] 2477 root364Tail4 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail5 winner := by
  rw [root364Tail4, scan, if_neg (by decide), root364Candidates4_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail5
    (if 3 ≤ root364Candidates4.length then search edge 3 [2477, 4073] root364Candidates4 6550 winner else winner) = _
  rw [if_neg (by decide)]
def root364Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root364Candidates5_checked : candidates edge 4 2477 4133 root364Tail6 winner.bound = root364Candidates5 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root364Step5 : scan edge (search edge 3) 4 [2477] 2477 root364Tail5 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail6 winner := by
  rw [root364Tail5, scan, if_neg (by decide), root364Candidates5_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail6
    (if 3 ≤ root364Candidates5.length then search edge 3 [2477, 4133] root364Candidates5 6610 winner else winner) = _
  rw [if_neg (by decide)]
def root364Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root364Candidates6_checked : candidates edge 4 2477 5039 root364Tail7 winner.bound = root364Candidates6 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root364Step6 : scan edge (search edge 3) 4 [2477] 2477 root364Tail6 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail7 winner := by
  rw [root364Tail6, scan, if_neg (by decide), root364Candidates6_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail7
    (if 3 ≤ root364Candidates6.length then search edge 3 [2477, 5039] root364Candidates6 7516 winner else winner) = _
  rw [if_neg (by decide)]
def root364Candidates7 : List Nat := [9929, 11549, 13229]
set_option maxRecDepth 8192 in
theorem root364Candidates7_checked : candidates edge 4 2477 5381 root364Tail8 winner.bound = root364Candidates7 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root364Child7 : search edge 3 [2477, 5381] root364Candidates7 7858 winner = winner := by decide
theorem root364Step7 : scan edge (search edge 3) 4 [2477] 2477 root364Tail7 winner =
    scan edge (search edge 3) 4 [2477] 2477 root364Tail8 winner := by
  rw [root364Tail7, scan, if_neg (by decide), root364Candidates7_checked]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail8
    (if 3 ≤ root364Candidates7.length then search edge 3 [2477, 5381] root364Candidates7 7858 winner else winner) = _
  rw [if_pos (by decide), root364Child7]
theorem root364Scan8 : scan edge (search edge 3) 4 [2477] 2477 root364Tail8 winner = winner := by rw [root364Tail8, scan, if_pos (by decide)]
theorem root364Scan7 : scan edge (search edge 3) 4 [2477] 2477 root364Tail7 winner = winner :=
  (root364Step7).trans root364Scan8
theorem root364Scan6 : scan edge (search edge 3) 4 [2477] 2477 root364Tail6 winner = winner :=
  (root364Step6).trans root364Scan7
theorem root364Scan5 : scan edge (search edge 3) 4 [2477] 2477 root364Tail5 winner = winner :=
  (root364Step5).trans root364Scan6
theorem root364Scan4 : scan edge (search edge 3) 4 [2477] 2477 root364Tail4 winner = winner :=
  (root364Step4).trans root364Scan5
theorem root364Scan3 : scan edge (search edge 3) 4 [2477] 2477 root364Tail3 winner = winner :=
  (root364Step3).trans root364Scan4
theorem root364Scan2 : scan edge (search edge 3) 4 [2477] 2477 root364Tail2 winner = winner :=
  (root364Step2).trans root364Scan3
theorem root364Scan1 : scan edge (search edge 3) 4 [2477] 2477 root364Tail1 winner = winner :=
  (root364Step1).trans root364Scan2
theorem root364Scan0 : scan edge (search edge 3) 4 [2477] 2477 root364Tail0 winner = winner :=
  (root364Step0).trans root364Scan1
set_option maxRecDepth 4096 in
theorem root364Branch : search edge 4 [2477] root364Candidates 2477 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2477] 2477 root364Tail0 winner = winner
  exact root364Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
