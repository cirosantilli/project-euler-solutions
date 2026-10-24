import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root276Candidates : List Nat := Data.row1801
set_option maxRecDepth 8192 in
theorem root276Candidates_checked : candidates edge 5 0 1801 tail277 winner.bound = root276Candidates := by
  apply candidates_eq_row edge 5 0 1801 tail277 root276Candidates winner.bound tail277_ordered
  · decide
  · decide
  · funext q
    rfl
def root276Tail32 : List Nat := []
def root276Tail31 : List Nat := 23539 :: root276Tail32
def root276Tail30 : List Nat := 23227 :: root276Tail31
def root276Tail29 : List Nat := 22303 :: root276Tail30
def root276Tail28 : List Nat := 21589 :: root276Tail29
def root276Tail27 : List Nat := 21397 :: root276Tail28
def root276Tail26 : List Nat := 20533 :: root276Tail27
def root276Tail25 : List Nat := 18919 :: root276Tail26
def root276Tail24 : List Nat := 18691 :: root276Tail25
def root276Tail23 : List Nat := 18181 :: root276Tail24
def root276Tail22 : List Nat := 14821 :: root276Tail23
def root276Tail21 : List Nat := 14593 :: root276Tail22
def root276Tail20 : List Nat := 13537 :: root276Tail21
def root276Tail19 : List Nat := 12277 :: root276Tail20
def root276Tail18 : List Nat := 11353 :: root276Tail19
def root276Tail17 : List Nat := 11149 :: root276Tail18
def root276Tail16 : List Nat := 8233 :: root276Tail17
def root276Tail15 : List Nat := 7669 :: root276Tail16
def root276Tail14 : List Nat := 7621 :: root276Tail15
def root276Tail13 : List Nat := 7333 :: root276Tail14
def root276Tail12 : List Nat := 6907 :: root276Tail13
def root276Tail11 : List Nat := 6829 :: root276Tail12
def root276Tail10 : List Nat := 6637 :: root276Tail11
def root276Tail9 : List Nat := 5689 :: root276Tail10
def root276Tail8 : List Nat := 5167 :: root276Tail9
def root276Tail7 : List Nat := 4933 :: root276Tail8
def root276Tail6 : List Nat := 4177 :: root276Tail7
def root276Tail5 : List Nat := 3823 :: root276Tail6
def root276Tail4 : List Nat := 3613 :: root276Tail5
def root276Tail3 : List Nat := 3547 :: root276Tail4
def root276Tail2 : List Nat := 3253 :: root276Tail3
def root276Tail1 : List Nat := 2647 :: root276Tail2
def root276Tail0 : List Nat := 2311 :: root276Tail1
def root276Candidates0 : List Nat := [2647]
set_option maxRecDepth 8192 in
theorem root276Candidates0_checked : candidates edge 4 1801 2311 root276Tail1 winner.bound = root276Candidates0 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root276Step0 : scan edge (search edge 3) 4 [1801] 1801 root276Tail0 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail1 winner := by
  rw [root276Tail0, scan, if_neg (by decide), root276Candidates0_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail1
    (if 3 ≤ root276Candidates0.length then search edge 3 [1801, 2311] root276Candidates0 4112 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root276Candidates1_checked : candidates edge 4 1801 2647 root276Tail2 winner.bound = root276Candidates1 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root276Step1 : scan edge (search edge 3) 4 [1801] 1801 root276Tail1 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail2 winner := by
  rw [root276Tail1, scan, if_neg (by decide), root276Candidates1_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail2
    (if 3 ≤ root276Candidates1.length then search edge 3 [1801, 2647] root276Candidates1 4448 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates2 : List Nat := [3547, 18691]
set_option maxRecDepth 8192 in
theorem root276Candidates2_checked : candidates edge 4 1801 3253 root276Tail3 winner.bound = root276Candidates2 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root276Step2 : scan edge (search edge 3) 4 [1801] 1801 root276Tail2 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail3 winner := by
  rw [root276Tail2, scan, if_neg (by decide), root276Candidates2_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail3
    (if 3 ≤ root276Candidates2.length then search edge 3 [1801, 3253] root276Candidates2 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates3 : List Nat := [13537]
set_option maxRecDepth 8192 in
theorem root276Candidates3_checked : candidates edge 4 1801 3547 root276Tail4 winner.bound = root276Candidates3 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root276Step3 : scan edge (search edge 3) 4 [1801] 1801 root276Tail3 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail4 winner := by
  rw [root276Tail3, scan, if_neg (by decide), root276Candidates3_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail4
    (if 3 ≤ root276Candidates3.length then search edge 3 [1801, 3547] root276Candidates3 5348 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root276Candidates4_checked : candidates edge 4 1801 3613 root276Tail5 winner.bound = root276Candidates4 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root276Step4 : scan edge (search edge 3) 4 [1801] 1801 root276Tail4 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail5 winner := by
  rw [root276Tail4, scan, if_neg (by decide), root276Candidates4_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail5
    (if 3 ≤ root276Candidates4.length then search edge 3 [1801, 3613] root276Candidates4 5414 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates5 : List Nat := [5167]
set_option maxRecDepth 8192 in
theorem root276Candidates5_checked : candidates edge 4 1801 3823 root276Tail6 winner.bound = root276Candidates5 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root276Step5 : scan edge (search edge 3) 4 [1801] 1801 root276Tail5 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail6 winner := by
  rw [root276Tail5, scan, if_neg (by decide), root276Candidates5_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail6
    (if 3 ≤ root276Candidates5.length then search edge 3 [1801, 3823] root276Candidates5 5624 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root276Candidates6_checked : candidates edge 4 1801 4177 root276Tail7 winner.bound = root276Candidates6 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root276Step6 : scan edge (search edge 3) 4 [1801] 1801 root276Tail6 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail7 winner := by
  rw [root276Tail6, scan, if_neg (by decide), root276Candidates6_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail7
    (if 3 ≤ root276Candidates6.length then search edge 3 [1801, 4177] root276Candidates6 5978 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root276Candidates7_checked : candidates edge 4 1801 4933 root276Tail8 winner.bound = root276Candidates7 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root276Step7 : scan edge (search edge 3) 4 [1801] 1801 root276Tail7 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail8 winner := by
  rw [root276Tail7, scan, if_neg (by decide), root276Candidates7_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail8
    (if 3 ≤ root276Candidates7.length then search edge 3 [1801, 4933] root276Candidates7 6734 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root276Candidates8_checked : candidates edge 4 1801 5167 root276Tail9 winner.bound = root276Candidates8 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root276Step8 : scan edge (search edge 3) 4 [1801] 1801 root276Tail8 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail9 winner := by
  rw [root276Tail8, scan, if_neg (by decide), root276Candidates8_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail9
    (if 3 ≤ root276Candidates8.length then search edge 3 [1801, 5167] root276Candidates8 6968 winner else winner) = _
  rw [if_neg (by decide)]
def root276Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root276Candidates9_checked : candidates edge 4 1801 5689 root276Tail10 winner.bound = root276Candidates9 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root276Step9 : scan edge (search edge 3) 4 [1801] 1801 root276Tail9 winner =
    scan edge (search edge 3) 4 [1801] 1801 root276Tail10 winner := by
  rw [root276Tail9, scan, if_neg (by decide), root276Candidates9_checked]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail10
    (if 3 ≤ root276Candidates9.length then search edge 3 [1801, 5689] root276Candidates9 7490 winner else winner) = _
  rw [if_neg (by decide)]
theorem root276Scan10 : scan edge (search edge 3) 4 [1801] 1801 root276Tail10 winner = winner := by rw [root276Tail10, scan, if_pos (by decide)]
theorem root276Scan9 : scan edge (search edge 3) 4 [1801] 1801 root276Tail9 winner = winner :=
  (root276Step9).trans root276Scan10
theorem root276Scan8 : scan edge (search edge 3) 4 [1801] 1801 root276Tail8 winner = winner :=
  (root276Step8).trans root276Scan9
theorem root276Scan7 : scan edge (search edge 3) 4 [1801] 1801 root276Tail7 winner = winner :=
  (root276Step7).trans root276Scan8
theorem root276Scan6 : scan edge (search edge 3) 4 [1801] 1801 root276Tail6 winner = winner :=
  (root276Step6).trans root276Scan7
theorem root276Scan5 : scan edge (search edge 3) 4 [1801] 1801 root276Tail5 winner = winner :=
  (root276Step5).trans root276Scan6
theorem root276Scan4 : scan edge (search edge 3) 4 [1801] 1801 root276Tail4 winner = winner :=
  (root276Step4).trans root276Scan5
theorem root276Scan3 : scan edge (search edge 3) 4 [1801] 1801 root276Tail3 winner = winner :=
  (root276Step3).trans root276Scan4
theorem root276Scan2 : scan edge (search edge 3) 4 [1801] 1801 root276Tail2 winner = winner :=
  (root276Step2).trans root276Scan3
theorem root276Scan1 : scan edge (search edge 3) 4 [1801] 1801 root276Tail1 winner = winner :=
  (root276Step1).trans root276Scan2
theorem root276Scan0 : scan edge (search edge 3) 4 [1801] 1801 root276Tail0 winner = winner :=
  (root276Step0).trans root276Scan1
set_option maxRecDepth 4096 in
theorem root276Branch : search edge 4 [1801] root276Candidates 1801 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1801] 1801 root276Tail0 winner = winner
  exact root276Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
