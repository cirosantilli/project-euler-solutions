import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root256Candidates : List Nat := Data.row1637
set_option maxRecDepth 8192 in
theorem root256Candidates_checked : candidates edge 5 0 1637 tail257 winner.bound = root256Candidates := by
  apply candidates_eq_row edge 5 0 1637 tail257 root256Candidates winner.bound tail257_ordered
  · decide
  · decide
  · funext q
    rfl
def root256Tail46 : List Nat := []
def root256Tail45 : List Nat := 23561 :: root256Tail46
def root256Tail44 : List Nat := 23297 :: root256Tail45
def root256Tail43 : List Nat := 22877 :: root256Tail44
def root256Tail42 : List Nat := 22859 :: root256Tail43
def root256Tail41 : List Nat := 22571 :: root256Tail42
def root256Tail40 : List Nat := 22271 :: root256Tail41
def root256Tail39 : List Nat := 21851 :: root256Tail40
def root256Tail38 : List Nat := 20483 :: root256Tail39
def root256Tail37 : List Nat := 20327 :: root256Tail38
def root256Tail36 : List Nat := 20261 :: root256Tail37
def root256Tail35 : List Nat := 19289 :: root256Tail36
def root256Tail34 : List Nat := 18539 :: root256Tail35
def root256Tail33 : List Nat := 16937 :: root256Tail34
def root256Tail32 : List Nat := 16481 :: root256Tail33
def root256Tail31 : List Nat := 15467 :: root256Tail32
def root256Tail30 : List Nat := 15329 :: root256Tail31
def root256Tail29 : List Nat := 15107 :: root256Tail30
def root256Tail28 : List Nat := 14741 :: root256Tail29
def root256Tail27 : List Nat := 14669 :: root256Tail28
def root256Tail26 : List Nat := 14633 :: root256Tail27
def root256Tail25 : List Nat := 13397 :: root256Tail26
def root256Tail24 : List Nat := 12347 :: root256Tail25
def root256Tail23 : List Nat := 11777 :: root256Tail24
def root256Tail22 : List Nat := 10667 :: root256Tail23
def root256Tail21 : List Nat := 9833 :: root256Tail22
def root256Tail20 : List Nat := 9479 :: root256Tail21
def root256Tail19 : List Nat := 8933 :: root256Tail20
def root256Tail18 : List Nat := 8363 :: root256Tail19
def root256Tail17 : List Nat := 8237 :: root256Tail18
def root256Tail16 : List Nat := 8039 :: root256Tail17
def root256Tail15 : List Nat := 8009 :: root256Tail16
def root256Tail14 : List Nat := 7841 :: root256Tail15
def root256Tail13 : List Nat := 7607 :: root256Tail14
def root256Tail12 : List Nat := 7013 :: root256Tail13
def root256Tail11 : List Nat := 6053 :: root256Tail12
def root256Tail10 : List Nat := 5897 :: root256Tail11
def root256Tail9 : List Nat := 5669 :: root256Tail10
def root256Tail8 : List Nat := 4973 :: root256Tail9
def root256Tail7 : List Nat := 4931 :: root256Tail8
def root256Tail6 : List Nat := 4649 :: root256Tail7
def root256Tail5 : List Nat := 4073 :: root256Tail6
def root256Tail4 : List Nat := 3911 :: root256Tail5
def root256Tail3 : List Nat := 3359 :: root256Tail4
def root256Tail2 : List Nat := 2903 :: root256Tail3
def root256Tail1 : List Nat := 1889 :: root256Tail2
def root256Tail0 : List Nat := 1709 :: root256Tail1
def root256Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root256Candidates0_checked : candidates edge 4 1637 1709 root256Tail1 winner.bound = root256Candidates0 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
theorem root256Step0 : scan edge (search edge 3) 4 [1637] 1637 root256Tail0 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail1 winner := by
  rw [root256Tail0, scan, if_neg (by decide), root256Candidates0_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail1
    (if 3 ≤ root256Candidates0.length then search edge 3 [1637, 1709] root256Candidates0 3346 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root256Candidates1_checked : candidates edge 4 1637 1889 root256Tail2 winner.bound = root256Candidates1 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root256Step1 : scan edge (search edge 3) 4 [1637] 1637 root256Tail1 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail2 winner := by
  rw [root256Tail1, scan, if_neg (by decide), root256Candidates1_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail2
    (if 3 ≤ root256Candidates1.length then search edge 3 [1637, 1889] root256Candidates1 3526 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates2 : List Nat := [10667]
set_option maxRecDepth 8192 in
theorem root256Candidates2_checked : candidates edge 4 1637 2903 root256Tail3 winner.bound = root256Candidates2 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
theorem root256Step2 : scan edge (search edge 3) 4 [1637] 1637 root256Tail2 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail3 winner := by
  rw [root256Tail2, scan, if_neg (by decide), root256Candidates2_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail3
    (if 3 ≤ root256Candidates2.length then search edge 3 [1637, 2903] root256Candidates2 4540 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root256Candidates3_checked : candidates edge 4 1637 3359 root256Tail4 winner.bound = root256Candidates3 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root256Step3 : scan edge (search edge 3) 4 [1637] 1637 root256Tail3 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail4 winner := by
  rw [root256Tail3, scan, if_neg (by decide), root256Candidates3_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail4
    (if 3 ≤ root256Candidates3.length then search edge 3 [1637, 3359] root256Candidates3 4996 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates4 : List Nat := [14741]
set_option maxRecDepth 8192 in
theorem root256Candidates4_checked : candidates edge 4 1637 3911 root256Tail5 winner.bound = root256Candidates4 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root256Step4 : scan edge (search edge 3) 4 [1637] 1637 root256Tail4 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail5 winner := by
  rw [root256Tail4, scan, if_neg (by decide), root256Candidates4_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail5
    (if 3 ≤ root256Candidates4.length then search edge 3 [1637, 3911] root256Candidates4 5548 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates5 : List Nat := [20261]
set_option maxRecDepth 8192 in
theorem root256Candidates5_checked : candidates edge 4 1637 4073 root256Tail6 winner.bound = root256Candidates5 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root256Step5 : scan edge (search edge 3) 4 [1637] 1637 root256Tail5 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail6 winner := by
  rw [root256Tail5, scan, if_neg (by decide), root256Candidates5_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail6
    (if 3 ≤ root256Candidates5.length then search edge 3 [1637, 4073] root256Candidates5 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates6 : List Nat := [14633, 15329]
set_option maxRecDepth 8192 in
theorem root256Candidates6_checked : candidates edge 4 1637 4649 root256Tail7 winner.bound = root256Candidates6 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root256Step6 : scan edge (search edge 3) 4 [1637] 1637 root256Tail6 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail7 winner := by
  rw [root256Tail6, scan, if_neg (by decide), root256Candidates6_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail7
    (if 3 ≤ root256Candidates6.length then search edge 3 [1637, 4649] root256Candidates6 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates7 : List Nat := [9833]
set_option maxRecDepth 8192 in
theorem root256Candidates7_checked : candidates edge 4 1637 4931 root256Tail8 winner.bound = root256Candidates7 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root256Step7 : scan edge (search edge 3) 4 [1637] 1637 root256Tail7 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail8 winner := by
  rw [root256Tail7, scan, if_neg (by decide), root256Candidates7_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail8
    (if 3 ≤ root256Candidates7.length then search edge 3 [1637, 4931] root256Candidates7 6568 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root256Candidates8_checked : candidates edge 4 1637 4973 root256Tail9 winner.bound = root256Candidates8 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root256Step8 : scan edge (search edge 3) 4 [1637] 1637 root256Tail8 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail9 winner := by
  rw [root256Tail8, scan, if_neg (by decide), root256Candidates8_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail9
    (if 3 ≤ root256Candidates8.length then search edge 3 [1637, 4973] root256Candidates8 6610 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates9 : List Nat := [9479, 16937]
set_option maxRecDepth 8192 in
theorem root256Candidates9_checked : candidates edge 4 1637 5669 root256Tail10 winner.bound = root256Candidates9 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root256Step9 : scan edge (search edge 3) 4 [1637] 1637 root256Tail9 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail10 winner := by
  rw [root256Tail9, scan, if_neg (by decide), root256Candidates9_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail10
    (if 3 ≤ root256Candidates9.length then search edge 3 [1637, 5669] root256Candidates9 7306 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root256Candidates10_checked : candidates edge 4 1637 5897 root256Tail11 winner.bound = root256Candidates10 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root256Step10 : scan edge (search edge 3) 4 [1637] 1637 root256Tail10 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail11 winner := by
  rw [root256Tail10, scan, if_neg (by decide), root256Candidates10_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail11
    (if 3 ≤ root256Candidates10.length then search edge 3 [1637, 5897] root256Candidates10 7534 winner else winner) = _
  rw [if_neg (by decide)]
def root256Candidates11 : List Nat := [15329]
set_option maxRecDepth 8192 in
theorem root256Candidates11_checked : candidates edge 4 1637 6053 root256Tail12 winner.bound = root256Candidates11 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root256Step11 : scan edge (search edge 3) 4 [1637] 1637 root256Tail11 winner =
    scan edge (search edge 3) 4 [1637] 1637 root256Tail12 winner := by
  rw [root256Tail11, scan, if_neg (by decide), root256Candidates11_checked]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail12
    (if 3 ≤ root256Candidates11.length then search edge 3 [1637, 6053] root256Candidates11 7690 winner else winner) = _
  rw [if_neg (by decide)]
theorem root256Scan12 : scan edge (search edge 3) 4 [1637] 1637 root256Tail12 winner = winner := by rw [root256Tail12, scan, if_pos (by decide)]
theorem root256Scan11 : scan edge (search edge 3) 4 [1637] 1637 root256Tail11 winner = winner :=
  (root256Step11).trans root256Scan12
theorem root256Scan10 : scan edge (search edge 3) 4 [1637] 1637 root256Tail10 winner = winner :=
  (root256Step10).trans root256Scan11
theorem root256Scan9 : scan edge (search edge 3) 4 [1637] 1637 root256Tail9 winner = winner :=
  (root256Step9).trans root256Scan10
theorem root256Scan8 : scan edge (search edge 3) 4 [1637] 1637 root256Tail8 winner = winner :=
  (root256Step8).trans root256Scan9
theorem root256Scan7 : scan edge (search edge 3) 4 [1637] 1637 root256Tail7 winner = winner :=
  (root256Step7).trans root256Scan8
theorem root256Scan6 : scan edge (search edge 3) 4 [1637] 1637 root256Tail6 winner = winner :=
  (root256Step6).trans root256Scan7
theorem root256Scan5 : scan edge (search edge 3) 4 [1637] 1637 root256Tail5 winner = winner :=
  (root256Step5).trans root256Scan6
theorem root256Scan4 : scan edge (search edge 3) 4 [1637] 1637 root256Tail4 winner = winner :=
  (root256Step4).trans root256Scan5
theorem root256Scan3 : scan edge (search edge 3) 4 [1637] 1637 root256Tail3 winner = winner :=
  (root256Step3).trans root256Scan4
theorem root256Scan2 : scan edge (search edge 3) 4 [1637] 1637 root256Tail2 winner = winner :=
  (root256Step2).trans root256Scan3
theorem root256Scan1 : scan edge (search edge 3) 4 [1637] 1637 root256Tail1 winner = winner :=
  (root256Step1).trans root256Scan2
theorem root256Scan0 : scan edge (search edge 3) 4 [1637] 1637 root256Tail0 winner = winner :=
  (root256Step0).trans root256Scan1
set_option maxRecDepth 4096 in
theorem root256Branch : search edge 4 [1637] root256Candidates 1637 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1637] 1637 root256Tail0 winner = winner
  exact root256Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
