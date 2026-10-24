import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root421Candidates : List Nat := Data.row2939
set_option maxRecDepth 8192 in
theorem root421Candidates_checked : candidates edge 5 0 2939 tail422 winner.bound = root421Candidates := by
  apply candidates_eq_row edge 5 0 2939 tail422 root421Candidates winner.bound tail422_ordered
  · decide
  · decide
  · funext q
    rfl
def root421Tail39 : List Nat := []
def root421Tail38 : List Nat := 22133 :: root421Tail39
def root421Tail37 : List Nat := 21587 :: root421Tail38
def root421Tail36 : List Nat := 21269 :: root421Tail37
def root421Tail35 : List Nat := 20441 :: root421Tail36
def root421Tail34 : List Nat := 19793 :: root421Tail35
def root421Tail33 : List Nat := 18701 :: root421Tail34
def root421Tail32 : List Nat := 16661 :: root421Tail33
def root421Tail31 : List Nat := 16253 :: root421Tail32
def root421Tail30 : List Nat := 15473 :: root421Tail31
def root421Tail29 : List Nat := 14321 :: root421Tail30
def root421Tail28 : List Nat := 13877 :: root421Tail29
def root421Tail27 : List Nat := 13709 :: root421Tail28
def root421Tail26 : List Nat := 12821 :: root421Tail27
def root421Tail25 : List Nat := 12041 :: root421Tail26
def root421Tail24 : List Nat := 11789 :: root421Tail25
def root421Tail23 : List Nat := 11621 :: root421Tail24
def root421Tail22 : List Nat := 11003 :: root421Tail23
def root421Tail21 : List Nat := 10847 :: root421Tail22
def root421Tail20 : List Nat := 9791 :: root421Tail21
def root421Tail19 : List Nat := 9677 :: root421Tail20
def root421Tail18 : List Nat := 9551 :: root421Tail19
def root421Tail17 : List Nat := 9203 :: root421Tail18
def root421Tail16 : List Nat := 8783 :: root421Tail17
def root421Tail15 : List Nat := 8573 :: root421Tail16
def root421Tail14 : List Nat := 7211 :: root421Tail15
def root421Tail13 : List Nat := 6863 :: root421Tail14
def root421Tail12 : List Nat := 5981 :: root421Tail13
def root421Tail11 : List Nat := 5573 :: root421Tail12
def root421Tail10 : List Nat := 5417 :: root421Tail11
def root421Tail9 : List Nat := 5297 :: root421Tail10
def root421Tail8 : List Nat := 4679 :: root421Tail9
def root421Tail7 : List Nat := 4217 :: root421Tail8
def root421Tail6 : List Nat := 4133 :: root421Tail7
def root421Tail5 : List Nat := 4019 :: root421Tail6
def root421Tail4 : List Nat := 3671 :: root421Tail5
def root421Tail3 : List Nat := 3557 :: root421Tail4
def root421Tail2 : List Nat := 3491 :: root421Tail3
def root421Tail1 : List Nat := 3167 :: root421Tail2
def root421Tail0 : List Nat := 2999 :: root421Tail1
def root421Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root421Candidates0_checked : candidates edge 4 2939 2999 root421Tail1 winner.bound = root421Candidates0 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root421Step0 : scan edge (search edge 3) 4 [2939] 2939 root421Tail0 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail1 winner := by
  rw [root421Tail0, scan, if_neg (by decide), root421Candidates0_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail1
    (if 3 ≤ root421Candidates0.length then search edge 3 [2939, 2999] root421Candidates0 5938 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates1 : List Nat := [4217]
set_option maxRecDepth 8192 in
theorem root421Candidates1_checked : candidates edge 4 2939 3167 root421Tail2 winner.bound = root421Candidates1 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root421Step1 : scan edge (search edge 3) 4 [2939] 2939 root421Tail1 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail2 winner := by
  rw [root421Tail1, scan, if_neg (by decide), root421Candidates1_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail2
    (if 3 ≤ root421Candidates1.length then search edge 3 [2939, 3167] root421Candidates1 6106 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root421Candidates2_checked : candidates edge 4 2939 3491 root421Tail3 winner.bound = root421Candidates2 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root421Step2 : scan edge (search edge 3) 4 [2939] 2939 root421Tail2 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail3 winner := by
  rw [root421Tail2, scan, if_neg (by decide), root421Candidates2_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail3
    (if 3 ≤ root421Candidates2.length then search edge 3 [2939, 3491] root421Candidates2 6430 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates3 : List Nat := [5417, 11789, 14321]
set_option maxRecDepth 8192 in
theorem root421Candidates3_checked : candidates edge 4 2939 3557 root421Tail4 winner.bound = root421Candidates3 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root421Child3 : search edge 3 [2939, 3557] root421Candidates3 6496 winner = winner := by decide
theorem root421Step3 : scan edge (search edge 3) 4 [2939] 2939 root421Tail3 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail4 winner := by
  rw [root421Tail3, scan, if_neg (by decide), root421Candidates3_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail4
    (if 3 ≤ root421Candidates3.length then search edge 3 [2939, 3557] root421Candidates3 6496 winner else winner) = _
  rw [if_pos (by decide), root421Child3]
def root421Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root421Candidates4_checked : candidates edge 4 2939 3671 root421Tail5 winner.bound = root421Candidates4 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root421Step4 : scan edge (search edge 3) 4 [2939] 2939 root421Tail4 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail5 winner := by
  rw [root421Tail4, scan, if_neg (by decide), root421Candidates4_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail5
    (if 3 ≤ root421Candidates4.length then search edge 3 [2939, 3671] root421Candidates4 6610 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates5 : List Nat := [7211]
set_option maxRecDepth 8192 in
theorem root421Candidates5_checked : candidates edge 4 2939 4019 root421Tail6 winner.bound = root421Candidates5 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root421Step5 : scan edge (search edge 3) 4 [2939] 2939 root421Tail5 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail6 winner := by
  rw [root421Tail5, scan, if_neg (by decide), root421Candidates5_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail6
    (if 3 ≤ root421Candidates5.length then search edge 3 [2939, 4019] root421Candidates5 6958 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root421Candidates6_checked : candidates edge 4 2939 4133 root421Tail7 winner.bound = root421Candidates6 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root421Step6 : scan edge (search edge 3) 4 [2939] 2939 root421Tail6 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail7 winner := by
  rw [root421Tail6, scan, if_neg (by decide), root421Candidates6_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail7
    (if 3 ≤ root421Candidates6.length then search edge 3 [2939, 4133] root421Candidates6 7072 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates7 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root421Candidates7_checked : candidates edge 4 2939 4217 root421Tail8 winner.bound = root421Candidates7 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root421Step7 : scan edge (search edge 3) 4 [2939] 2939 root421Tail7 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail8 winner := by
  rw [root421Tail7, scan, if_neg (by decide), root421Candidates7_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail8
    (if 3 ≤ root421Candidates7.length then search edge 3 [2939, 4217] root421Candidates7 7156 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates8 : List Nat := [8783]
set_option maxRecDepth 8192 in
theorem root421Candidates8_checked : candidates edge 4 2939 4679 root421Tail9 winner.bound = root421Candidates8 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root421Step8 : scan edge (search edge 3) 4 [2939] 2939 root421Tail8 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail9 winner := by
  rw [root421Tail8, scan, if_neg (by decide), root421Candidates8_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail9
    (if 3 ≤ root421Candidates8.length then search edge 3 [2939, 4679] root421Candidates8 7618 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates9 : List Nat := [9791]
set_option maxRecDepth 8192 in
theorem root421Candidates9_checked : candidates edge 4 2939 5297 root421Tail10 winner.bound = root421Candidates9 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root421Step9 : scan edge (search edge 3) 4 [2939] 2939 root421Tail9 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail10 winner := by
  rw [root421Tail9, scan, if_neg (by decide), root421Candidates9_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail10
    (if 3 ≤ root421Candidates9.length then search edge 3 [2939, 5297] root421Candidates9 8236 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates10 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root421Candidates10_checked : candidates edge 4 2939 5417 root421Tail11 winner.bound = root421Candidates10 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root421Step10 : scan edge (search edge 3) 4 [2939] 2939 root421Tail10 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail11 winner := by
  rw [root421Tail10, scan, if_neg (by decide), root421Candidates10_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail11
    (if 3 ≤ root421Candidates10.length then search edge 3 [2939, 5417] root421Candidates10 8356 winner else winner) = _
  rw [if_neg (by decide)]
def root421Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root421Candidates11_checked : candidates edge 4 2939 5573 root421Tail12 winner.bound = root421Candidates11 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root421Step11 : scan edge (search edge 3) 4 [2939] 2939 root421Tail11 winner =
    scan edge (search edge 3) 4 [2939] 2939 root421Tail12 winner := by
  rw [root421Tail11, scan, if_neg (by decide), root421Candidates11_checked]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail12
    (if 3 ≤ root421Candidates11.length then search edge 3 [2939, 5573] root421Candidates11 8512 winner else winner) = _
  rw [if_neg (by decide)]
theorem root421Scan12 : scan edge (search edge 3) 4 [2939] 2939 root421Tail12 winner = winner := by rw [root421Tail12, scan, if_pos (by decide)]
theorem root421Scan11 : scan edge (search edge 3) 4 [2939] 2939 root421Tail11 winner = winner :=
  (root421Step11).trans root421Scan12
theorem root421Scan10 : scan edge (search edge 3) 4 [2939] 2939 root421Tail10 winner = winner :=
  (root421Step10).trans root421Scan11
theorem root421Scan9 : scan edge (search edge 3) 4 [2939] 2939 root421Tail9 winner = winner :=
  (root421Step9).trans root421Scan10
theorem root421Scan8 : scan edge (search edge 3) 4 [2939] 2939 root421Tail8 winner = winner :=
  (root421Step8).trans root421Scan9
theorem root421Scan7 : scan edge (search edge 3) 4 [2939] 2939 root421Tail7 winner = winner :=
  (root421Step7).trans root421Scan8
theorem root421Scan6 : scan edge (search edge 3) 4 [2939] 2939 root421Tail6 winner = winner :=
  (root421Step6).trans root421Scan7
theorem root421Scan5 : scan edge (search edge 3) 4 [2939] 2939 root421Tail5 winner = winner :=
  (root421Step5).trans root421Scan6
theorem root421Scan4 : scan edge (search edge 3) 4 [2939] 2939 root421Tail4 winner = winner :=
  (root421Step4).trans root421Scan5
theorem root421Scan3 : scan edge (search edge 3) 4 [2939] 2939 root421Tail3 winner = winner :=
  (root421Step3).trans root421Scan4
theorem root421Scan2 : scan edge (search edge 3) 4 [2939] 2939 root421Tail2 winner = winner :=
  (root421Step2).trans root421Scan3
theorem root421Scan1 : scan edge (search edge 3) 4 [2939] 2939 root421Tail1 winner = winner :=
  (root421Step1).trans root421Scan2
theorem root421Scan0 : scan edge (search edge 3) 4 [2939] 2939 root421Tail0 winner = winner :=
  (root421Step0).trans root421Scan1
set_option maxRecDepth 4096 in
theorem root421Branch : search edge 4 [2939] root421Candidates 2939 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2939] 2939 root421Tail0 winner = winner
  exact root421Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
