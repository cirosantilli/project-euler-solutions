import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root377Candidates : List Nat := Data.row2617
set_option maxRecDepth 8192 in
theorem root377Candidates_checked : candidates edge 5 0 2617 tail378 winner.bound = root377Candidates := by
  apply candidates_eq_row edge 5 0 2617 tail378 root377Candidates winner.bound tail378_ordered
  · decide
  · decide
  · funext q
    rfl
def root377Tail43 : List Nat := []
def root377Tail42 : List Nat := 21601 :: root377Tail43
def root377Tail41 : List Nat := 21487 :: root377Tail42
def root377Tail40 : List Nat := 20551 :: root377Tail41
def root377Tail39 : List Nat := 20521 :: root377Tail40
def root377Tail38 : List Nat := 20509 :: root377Tail39
def root377Tail37 : List Nat := 20359 :: root377Tail38
def root377Tail36 : List Nat := 19333 :: root377Tail37
def root377Tail35 : List Nat := 19309 :: root377Tail36
def root377Tail34 : List Nat := 18661 :: root377Tail35
def root377Tail33 : List Nat := 17791 :: root377Tail34
def root377Tail32 : List Nat := 17707 :: root377Tail33
def root377Tail31 : List Nat := 17041 :: root377Tail32
def root377Tail30 : List Nat := 16993 :: root377Tail31
def root377Tail29 : List Nat := 16759 :: root377Tail30
def root377Tail28 : List Nat := 16519 :: root377Tail29
def root377Tail27 : List Nat := 16381 :: root377Tail28
def root377Tail26 : List Nat := 16339 :: root377Tail27
def root377Tail25 : List Nat := 16033 :: root377Tail26
def root377Tail24 : List Nat := 15583 :: root377Tail25
def root377Tail23 : List Nat := 14923 :: root377Tail24
def root377Tail22 : List Nat := 12163 :: root377Tail23
def root377Tail21 : List Nat := 11941 :: root377Tail22
def root377Tail20 : List Nat := 10651 :: root377Tail21
def root377Tail19 : List Nat := 10357 :: root377Tail20
def root377Tail18 : List Nat := 9817 :: root377Tail19
def root377Tail17 : List Nat := 9661 :: root377Tail18
def root377Tail16 : List Nat := 9157 :: root377Tail17
def root377Tail15 : List Nat := 8641 :: root377Tail16
def root377Tail14 : List Nat := 8629 :: root377Tail15
def root377Tail13 : List Nat := 8161 :: root377Tail14
def root377Tail12 : List Nat := 7741 :: root377Tail13
def root377Tail11 : List Nat := 7507 :: root377Tail12
def root377Tail10 : List Nat := 7213 :: root377Tail11
def root377Tail9 : List Nat := 5839 :: root377Tail10
def root377Tail8 : List Nat := 5689 :: root377Tail9
def root377Tail7 : List Nat := 4801 :: root377Tail8
def root377Tail6 : List Nat := 4567 :: root377Tail7
def root377Tail5 : List Nat := 4273 :: root377Tail6
def root377Tail4 : List Nat := 3571 :: root377Tail5
def root377Tail3 : List Nat := 3181 :: root377Tail4
def root377Tail2 : List Nat := 2857 :: root377Tail3
def root377Tail1 : List Nat := 2833 :: root377Tail2
def root377Tail0 : List Nat := 2803 :: root377Tail1
def root377Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root377Candidates0_checked : candidates edge 4 2617 2803 root377Tail1 winner.bound = root377Candidates0 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root377Step0 : scan edge (search edge 3) 4 [2617] 2617 root377Tail0 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail1 winner := by
  rw [root377Tail0, scan, if_neg (by decide), root377Candidates0_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail1
    (if 3 ≤ root377Candidates0.length then search edge 3 [2617, 2803] root377Candidates0 5420 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root377Candidates1_checked : candidates edge 4 2617 2833 root377Tail2 winner.bound = root377Candidates1 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root377Step1 : scan edge (search edge 3) 4 [2617] 2617 root377Tail1 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail2 winner := by
  rw [root377Tail1, scan, if_neg (by decide), root377Candidates1_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail2
    (if 3 ≤ root377Candidates1.length then search edge 3 [2617, 2833] root377Candidates1 5450 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root377Candidates2_checked : candidates edge 4 2617 2857 root377Tail3 winner.bound = root377Candidates2 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
theorem root377Step2 : scan edge (search edge 3) 4 [2617] 2617 root377Tail2 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail3 winner := by
  rw [root377Tail2, scan, if_neg (by decide), root377Candidates2_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail3
    (if 3 ≤ root377Candidates2.length then search edge 3 [2617, 2857] root377Candidates2 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates3 : List Nat := [4273, 7507]
set_option maxRecDepth 8192 in
theorem root377Candidates3_checked : candidates edge 4 2617 3181 root377Tail4 winner.bound = root377Candidates3 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
theorem root377Step3 : scan edge (search edge 3) 4 [2617] 2617 root377Tail3 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail4 winner := by
  rw [root377Tail3, scan, if_neg (by decide), root377Candidates3_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail4
    (if 3 ≤ root377Candidates3.length then search edge 3 [2617, 3181] root377Candidates3 5798 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates4 : List Nat := [4801]
set_option maxRecDepth 8192 in
theorem root377Candidates4_checked : candidates edge 4 2617 3571 root377Tail5 winner.bound = root377Candidates4 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root377Step4 : scan edge (search edge 3) 4 [2617] 2617 root377Tail4 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail5 winner := by
  rw [root377Tail4, scan, if_neg (by decide), root377Candidates4_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail5
    (if 3 ≤ root377Candidates4.length then search edge 3 [2617, 3571] root377Candidates4 6188 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates5 : List Nat := [4567]
set_option maxRecDepth 8192 in
theorem root377Candidates5_checked : candidates edge 4 2617 4273 root377Tail6 winner.bound = root377Candidates5 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root377Step5 : scan edge (search edge 3) 4 [2617] 2617 root377Tail5 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail6 winner := by
  rw [root377Tail5, scan, if_neg (by decide), root377Candidates5_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail6
    (if 3 ≤ root377Candidates5.length then search edge 3 [2617, 4273] root377Candidates5 6890 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root377Candidates6_checked : candidates edge 4 2617 4567 root377Tail7 winner.bound = root377Candidates6 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root377Step6 : scan edge (search edge 3) 4 [2617] 2617 root377Tail6 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail7 winner := by
  rw [root377Tail6, scan, if_neg (by decide), root377Candidates6_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail7
    (if 3 ≤ root377Candidates6.length then search edge 3 [2617, 4567] root377Candidates6 7184 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates7 : List Nat := [5839, 12163, 16381]
set_option maxRecDepth 8192 in
theorem root377Candidates7_checked : candidates edge 4 2617 4801 root377Tail8 winner.bound = root377Candidates7 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root377Child7 : search edge 3 [2617, 4801] root377Candidates7 7418 winner = winner := by decide
theorem root377Step7 : scan edge (search edge 3) 4 [2617] 2617 root377Tail7 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail8 winner := by
  rw [root377Tail7, scan, if_neg (by decide), root377Candidates7_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail8
    (if 3 ≤ root377Candidates7.length then search edge 3 [2617, 4801] root377Candidates7 7418 winner else winner) = _
  rw [if_pos (by decide), root377Child7]
def root377Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root377Candidates8_checked : candidates edge 4 2617 5689 root377Tail9 winner.bound = root377Candidates8 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root377Step8 : scan edge (search edge 3) 4 [2617] 2617 root377Tail8 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail9 winner := by
  rw [root377Tail8, scan, if_neg (by decide), root377Candidates8_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail9
    (if 3 ≤ root377Candidates8.length then search edge 3 [2617, 5689] root377Candidates8 8306 winner else winner) = _
  rw [if_neg (by decide)]
def root377Candidates9 : List Nat := [15583]
set_option maxRecDepth 8192 in
theorem root377Candidates9_checked : candidates edge 4 2617 5839 root377Tail10 winner.bound = root377Candidates9 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root377Step9 : scan edge (search edge 3) 4 [2617] 2617 root377Tail9 winner =
    scan edge (search edge 3) 4 [2617] 2617 root377Tail10 winner := by
  rw [root377Tail9, scan, if_neg (by decide), root377Candidates9_checked]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail10
    (if 3 ≤ root377Candidates9.length then search edge 3 [2617, 5839] root377Candidates9 8456 winner else winner) = _
  rw [if_neg (by decide)]
theorem root377Scan10 : scan edge (search edge 3) 4 [2617] 2617 root377Tail10 winner = winner := by rw [root377Tail10, scan, if_pos (by decide)]
theorem root377Scan9 : scan edge (search edge 3) 4 [2617] 2617 root377Tail9 winner = winner :=
  (root377Step9).trans root377Scan10
theorem root377Scan8 : scan edge (search edge 3) 4 [2617] 2617 root377Tail8 winner = winner :=
  (root377Step8).trans root377Scan9
theorem root377Scan7 : scan edge (search edge 3) 4 [2617] 2617 root377Tail7 winner = winner :=
  (root377Step7).trans root377Scan8
theorem root377Scan6 : scan edge (search edge 3) 4 [2617] 2617 root377Tail6 winner = winner :=
  (root377Step6).trans root377Scan7
theorem root377Scan5 : scan edge (search edge 3) 4 [2617] 2617 root377Tail5 winner = winner :=
  (root377Step5).trans root377Scan6
theorem root377Scan4 : scan edge (search edge 3) 4 [2617] 2617 root377Tail4 winner = winner :=
  (root377Step4).trans root377Scan5
theorem root377Scan3 : scan edge (search edge 3) 4 [2617] 2617 root377Tail3 winner = winner :=
  (root377Step3).trans root377Scan4
theorem root377Scan2 : scan edge (search edge 3) 4 [2617] 2617 root377Tail2 winner = winner :=
  (root377Step2).trans root377Scan3
theorem root377Scan1 : scan edge (search edge 3) 4 [2617] 2617 root377Tail1 winner = winner :=
  (root377Step1).trans root377Scan2
theorem root377Scan0 : scan edge (search edge 3) 4 [2617] 2617 root377Tail0 winner = winner :=
  (root377Step0).trans root377Scan1
set_option maxRecDepth 4096 in
theorem root377Branch : search edge 4 [2617] root377Candidates 2617 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2617] 2617 root377Tail0 winner = winner
  exact root377Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
