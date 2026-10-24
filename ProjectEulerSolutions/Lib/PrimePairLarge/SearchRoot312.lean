import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root312Candidates : List Nat := Data.row2087
set_option maxRecDepth 8192 in
theorem root312Candidates_checked : candidates edge 5 0 2087 tail313 winner.bound = root312Candidates := by
  apply candidates_eq_row edge 5 0 2087 tail313 root312Candidates winner.bound tail313_ordered
  · decide
  · decide
  · funext q
    rfl
def root312Tail37 : List Nat := []
def root312Tail36 : List Nat := 23633 :: root312Tail37
def root312Tail35 : List Nat := 23561 :: root312Tail36
def root312Tail34 : List Nat := 23027 :: root312Tail35
def root312Tail33 : List Nat := 23003 :: root312Tail34
def root312Tail32 : List Nat := 22511 :: root312Tail33
def root312Tail31 : List Nat := 21059 :: root312Tail32
def root312Tail30 : List Nat := 20411 :: root312Tail31
def root312Tail29 : List Nat := 19391 :: root312Tail30
def root312Tail28 : List Nat := 19079 :: root312Tail29
def root312Tail27 : List Nat := 17837 :: root312Tail28
def root312Tail26 : List Nat := 17627 :: root312Tail27
def root312Tail25 : List Nat := 17291 :: root312Tail26
def root312Tail24 : List Nat := 17231 :: root312Tail25
def root312Tail23 : List Nat := 16229 :: root312Tail24
def root312Tail22 : List Nat := 15971 :: root312Tail23
def root312Tail21 : List Nat := 15473 :: root312Tail22
def root312Tail20 : List Nat := 15401 :: root312Tail21
def root312Tail19 : List Nat := 14657 :: root312Tail20
def root312Tail18 : List Nat := 14543 :: root312Tail19
def root312Tail17 : List Nat := 13217 :: root312Tail18
def root312Tail16 : List Nat := 12011 :: root312Tail17
def root312Tail15 : List Nat := 11831 :: root312Tail16
def root312Tail14 : List Nat := 9437 :: root312Tail15
def root312Tail13 : List Nat := 9029 :: root312Tail14
def root312Tail12 : List Nat := 8969 :: root312Tail13
def root312Tail11 : List Nat := 8537 :: root312Tail12
def root312Tail10 : List Nat := 7907 :: root312Tail11
def root312Tail9 : List Nat := 7559 :: root312Tail10
def root312Tail8 : List Nat := 7079 :: root312Tail9
def root312Tail7 : List Nat := 6689 :: root312Tail8
def root312Tail6 : List Nat := 5333 :: root312Tail7
def root312Tail5 : List Nat := 5009 :: root312Tail6
def root312Tail4 : List Nat := 4937 :: root312Tail5
def root312Tail3 : List Nat := 4463 :: root312Tail4
def root312Tail2 : List Nat := 3821 :: root312Tail3
def root312Tail1 : List Nat := 2417 :: root312Tail2
def root312Tail0 : List Nat := 2153 :: root312Tail1
def root312Candidates0 : List Nat := [2417]
set_option maxRecDepth 8192 in
theorem root312Candidates0_checked : candidates edge 4 2087 2153 root312Tail1 winner.bound = root312Candidates0 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
theorem root312Step0 : scan edge (search edge 3) 4 [2087] 2087 root312Tail0 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail1 winner := by
  rw [root312Tail0, scan, if_neg (by decide), root312Candidates0_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail1
    (if 3 ≤ root312Candidates0.length then search edge 3 [2087, 2153] root312Candidates0 4240 winner else winner) = _
  rw [if_neg (by decide)]
def root312Candidates1 : List Nat := [9437]
set_option maxRecDepth 8192 in
theorem root312Candidates1_checked : candidates edge 4 2087 2417 root312Tail2 winner.bound = root312Candidates1 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root312Step1 : scan edge (search edge 3) 4 [2087] 2087 root312Tail1 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail2 winner := by
  rw [root312Tail1, scan, if_neg (by decide), root312Candidates1_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail2
    (if 3 ≤ root312Candidates1.length then search edge 3 [2087, 2417] root312Candidates1 4504 winner else winner) = _
  rw [if_neg (by decide)]
def root312Candidates2 : List Nat := [14657, 17627]
set_option maxRecDepth 8192 in
theorem root312Candidates2_checked : candidates edge 4 2087 3821 root312Tail3 winner.bound = root312Candidates2 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root312Step2 : scan edge (search edge 3) 4 [2087] 2087 root312Tail2 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail3 winner := by
  rw [root312Tail2, scan, if_neg (by decide), root312Candidates2_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail3
    (if 3 ≤ root312Candidates2.length then search edge 3 [2087, 3821] root312Candidates2 5908 winner else winner) = _
  rw [if_neg (by decide)]
def root312Candidates3 : List Nat := [7079]
set_option maxRecDepth 8192 in
theorem root312Candidates3_checked : candidates edge 4 2087 4463 root312Tail4 winner.bound = root312Candidates3 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root312Step3 : scan edge (search edge 3) 4 [2087] 2087 root312Tail3 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail4 winner := by
  rw [root312Tail3, scan, if_neg (by decide), root312Candidates3_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail4
    (if 3 ≤ root312Candidates3.length then search edge 3 [2087, 4463] root312Candidates3 6550 winner else winner) = _
  rw [if_neg (by decide)]
def root312Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root312Candidates4_checked : candidates edge 4 2087 4937 root312Tail5 winner.bound = root312Candidates4 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root312Step4 : scan edge (search edge 3) 4 [2087] 2087 root312Tail4 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail5 winner := by
  rw [root312Tail4, scan, if_neg (by decide), root312Candidates4_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail5
    (if 3 ≤ root312Candidates4.length then search edge 3 [2087, 4937] root312Candidates4 7024 winner else winner) = _
  rw [if_neg (by decide)]
def root312Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root312Candidates5_checked : candidates edge 4 2087 5009 root312Tail6 winner.bound = root312Candidates5 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root312Step5 : scan edge (search edge 3) 4 [2087] 2087 root312Tail5 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail6 winner := by
  rw [root312Tail5, scan, if_neg (by decide), root312Candidates5_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail6
    (if 3 ≤ root312Candidates5.length then search edge 3 [2087, 5009] root312Candidates5 7096 winner else winner) = _
  rw [if_neg (by decide)]
def root312Candidates6 : List Nat := [9029]
set_option maxRecDepth 8192 in
theorem root312Candidates6_checked : candidates edge 4 2087 5333 root312Tail7 winner.bound = root312Candidates6 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root312Step6 : scan edge (search edge 3) 4 [2087] 2087 root312Tail6 winner =
    scan edge (search edge 3) 4 [2087] 2087 root312Tail7 winner := by
  rw [root312Tail6, scan, if_neg (by decide), root312Candidates6_checked]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail7
    (if 3 ≤ root312Candidates6.length then search edge 3 [2087, 5333] root312Candidates6 7420 winner else winner) = _
  rw [if_neg (by decide)]
theorem root312Scan7 : scan edge (search edge 3) 4 [2087] 2087 root312Tail7 winner = winner := by rw [root312Tail7, scan, if_pos (by decide)]
theorem root312Scan6 : scan edge (search edge 3) 4 [2087] 2087 root312Tail6 winner = winner :=
  (root312Step6).trans root312Scan7
theorem root312Scan5 : scan edge (search edge 3) 4 [2087] 2087 root312Tail5 winner = winner :=
  (root312Step5).trans root312Scan6
theorem root312Scan4 : scan edge (search edge 3) 4 [2087] 2087 root312Tail4 winner = winner :=
  (root312Step4).trans root312Scan5
theorem root312Scan3 : scan edge (search edge 3) 4 [2087] 2087 root312Tail3 winner = winner :=
  (root312Step3).trans root312Scan4
theorem root312Scan2 : scan edge (search edge 3) 4 [2087] 2087 root312Tail2 winner = winner :=
  (root312Step2).trans root312Scan3
theorem root312Scan1 : scan edge (search edge 3) 4 [2087] 2087 root312Tail1 winner = winner :=
  (root312Step1).trans root312Scan2
theorem root312Scan0 : scan edge (search edge 3) 4 [2087] 2087 root312Tail0 winner = winner :=
  (root312Step0).trans root312Scan1
set_option maxRecDepth 4096 in
theorem root312Branch : search edge 4 [2087] root312Candidates 2087 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2087] 2087 root312Tail0 winner = winner
  exact root312Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
