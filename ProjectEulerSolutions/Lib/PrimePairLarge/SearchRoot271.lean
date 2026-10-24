import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root271Candidates : List Nat := Data.row1759
set_option maxRecDepth 8192 in
theorem root271Candidates_checked : candidates edge 5 0 1759 tail272 winner.bound = root271Candidates := by
  apply candidates_eq_row edge 5 0 1759 tail272 root271Candidates winner.bound tail272_ordered
  · decide
  · decide
  · funext q
    rfl
def root271Tail41 : List Nat := []
def root271Tail40 : List Nat := 23581 :: root271Tail41
def root271Tail39 : List Nat := 22063 :: root271Tail40
def root271Tail38 : List Nat := 19441 :: root271Tail39
def root271Tail37 : List Nat := 18661 :: root271Tail38
def root271Tail36 : List Nat := 18493 :: root271Tail37
def root271Tail35 : List Nat := 18451 :: root271Tail36
def root271Tail34 : List Nat := 17239 :: root271Tail35
def root271Tail33 : List Nat := 16729 :: root271Tail34
def root271Tail32 : List Nat := 16693 :: root271Tail33
def root271Tail31 : List Nat := 16141 :: root271Tail32
def root271Tail30 : List Nat := 15331 :: root271Tail31
def root271Tail29 : List Nat := 15307 :: root271Tail30
def root271Tail28 : List Nat := 15121 :: root271Tail29
def root271Tail27 : List Nat := 13879 :: root271Tail28
def root271Tail26 : List Nat := 13567 :: root271Tail27
def root271Tail25 : List Nat := 12853 :: root271Tail26
def root271Tail24 : List Nat := 12049 :: root271Tail25
def root271Tail23 : List Nat := 10687 :: root271Tail24
def root271Tail22 : List Nat := 9973 :: root271Tail23
def root271Tail21 : List Nat := 9613 :: root271Tail22
def root271Tail20 : List Nat := 9277 :: root271Tail21
def root271Tail19 : List Nat := 8941 :: root271Tail20
def root271Tail18 : List Nat := 8839 :: root271Tail19
def root271Tail17 : List Nat := 7417 :: root271Tail18
def root271Tail16 : List Nat := 7351 :: root271Tail17
def root271Tail15 : List Nat := 6703 :: root271Tail16
def root271Tail14 : List Nat := 6637 :: root271Tail15
def root271Tail13 : List Nat := 6367 :: root271Tail14
def root271Tail12 : List Nat := 6361 :: root271Tail13
def root271Tail11 : List Nat := 6199 :: root271Tail12
def root271Tail10 : List Nat := 5689 :: root271Tail11
def root271Tail9 : List Nat := 5233 :: root271Tail10
def root271Tail8 : List Nat := 4177 :: root271Tail9
def root271Tail7 : List Nat := 3889 :: root271Tail8
def root271Tail6 : List Nat := 3847 :: root271Tail7
def root271Tail5 : List Nat := 3733 :: root271Tail6
def root271Tail4 : List Nat := 3571 :: root271Tail5
def root271Tail3 : List Nat := 3319 :: root271Tail4
def root271Tail2 : List Nat := 3001 :: root271Tail3
def root271Tail1 : List Nat := 2749 :: root271Tail2
def root271Tail0 : List Nat := 1831 :: root271Tail1
def root271Candidates0 : List Nat := [8941]
set_option maxRecDepth 8192 in
theorem root271Candidates0_checked : candidates edge 4 1759 1831 root271Tail1 winner.bound = root271Candidates0 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
theorem root271Step0 : scan edge (search edge 3) 4 [1759] 1759 root271Tail0 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail1 winner := by
  rw [root271Tail0, scan, if_neg (by decide), root271Candidates0_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail1
    (if 3 ≤ root271Candidates0.length then search edge 3 [1759, 1831] root271Candidates0 3590 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root271Candidates1_checked : candidates edge 4 1759 2749 root271Tail2 winner.bound = root271Candidates1 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root271Step1 : scan edge (search edge 3) 4 [1759] 1759 root271Tail1 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail2 winner := by
  rw [root271Tail1, scan, if_neg (by decide), root271Candidates1_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail2
    (if 3 ≤ root271Candidates1.length then search edge 3 [1759, 2749] root271Candidates1 4508 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates2 : List Nat := [10687]
set_option maxRecDepth 8192 in
theorem root271Candidates2_checked : candidates edge 4 1759 3001 root271Tail3 winner.bound = root271Candidates2 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
theorem root271Step2 : scan edge (search edge 3) 4 [1759] 1759 root271Tail2 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail3 winner := by
  rw [root271Tail2, scan, if_neg (by decide), root271Candidates2_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail3
    (if 3 ≤ root271Candidates2.length then search edge 3 [1759, 3001] root271Candidates2 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates3 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root271Candidates3_checked : candidates edge 4 1759 3319 root271Tail4 winner.bound = root271Candidates3 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
theorem root271Step3 : scan edge (search edge 3) 4 [1759] 1759 root271Tail3 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail4 winner := by
  rw [root271Tail3, scan, if_neg (by decide), root271Candidates3_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail4
    (if 3 ≤ root271Candidates3.length then search edge 3 [1759, 3319] root271Candidates3 5078 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root271Candidates4_checked : candidates edge 4 1759 3571 root271Tail5 winner.bound = root271Candidates4 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root271Step4 : scan edge (search edge 3) 4 [1759] 1759 root271Tail4 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail5 winner := by
  rw [root271Tail4, scan, if_neg (by decide), root271Candidates4_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail5
    (if 3 ≤ root271Candidates4.length then search edge 3 [1759, 3571] root271Candidates4 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root271Candidates5_checked : candidates edge 4 1759 3733 root271Tail6 winner.bound = root271Candidates5 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root271Step5 : scan edge (search edge 3) 4 [1759] 1759 root271Tail5 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail6 winner := by
  rw [root271Tail5, scan, if_neg (by decide), root271Candidates5_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail6
    (if 3 ≤ root271Candidates5.length then search edge 3 [1759, 3733] root271Candidates5 5492 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates6 : List Nat := [6367, 12049, 17239]
set_option maxRecDepth 8192 in
theorem root271Candidates6_checked : candidates edge 4 1759 3847 root271Tail7 winner.bound = root271Candidates6 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root271Child6 : search edge 3 [1759, 3847] root271Candidates6 5606 winner = winner := by decide
theorem root271Step6 : scan edge (search edge 3) 4 [1759] 1759 root271Tail6 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail7 winner := by
  rw [root271Tail6, scan, if_neg (by decide), root271Candidates6_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail7
    (if 3 ≤ root271Candidates6.length then search edge 3 [1759, 3847] root271Candidates6 5606 winner else winner) = _
  rw [if_pos (by decide), root271Child6]
def root271Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root271Candidates7_checked : candidates edge 4 1759 3889 root271Tail8 winner.bound = root271Candidates7 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root271Step7 : scan edge (search edge 3) 4 [1759] 1759 root271Tail7 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail8 winner := by
  rw [root271Tail7, scan, if_neg (by decide), root271Candidates7_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail8
    (if 3 ≤ root271Candidates7.length then search edge 3 [1759, 3889] root271Candidates7 5648 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates8 : List Nat := [9973, 19441]
set_option maxRecDepth 8192 in
theorem root271Candidates8_checked : candidates edge 4 1759 4177 root271Tail9 winner.bound = root271Candidates8 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root271Step8 : scan edge (search edge 3) 4 [1759] 1759 root271Tail8 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail9 winner := by
  rw [root271Tail8, scan, if_neg (by decide), root271Candidates8_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail9
    (if 3 ≤ root271Candidates8.length then search edge 3 [1759, 4177] root271Candidates8 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates9 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root271Candidates9_checked : candidates edge 4 1759 5233 root271Tail10 winner.bound = root271Candidates9 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root271Step9 : scan edge (search edge 3) 4 [1759] 1759 root271Tail9 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail10 winner := by
  rw [root271Tail9, scan, if_neg (by decide), root271Candidates9_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail10
    (if 3 ≤ root271Candidates9.length then search edge 3 [1759, 5233] root271Candidates9 6992 winner else winner) = _
  rw [if_neg (by decide)]
def root271Candidates10 : List Nat := [15121]
set_option maxRecDepth 8192 in
theorem root271Candidates10_checked : candidates edge 4 1759 5689 root271Tail11 winner.bound = root271Candidates10 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root271Step10 : scan edge (search edge 3) 4 [1759] 1759 root271Tail10 winner =
    scan edge (search edge 3) 4 [1759] 1759 root271Tail11 winner := by
  rw [root271Tail10, scan, if_neg (by decide), root271Candidates10_checked]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail11
    (if 3 ≤ root271Candidates10.length then search edge 3 [1759, 5689] root271Candidates10 7448 winner else winner) = _
  rw [if_neg (by decide)]
theorem root271Scan11 : scan edge (search edge 3) 4 [1759] 1759 root271Tail11 winner = winner := by rw [root271Tail11, scan, if_pos (by decide)]
theorem root271Scan10 : scan edge (search edge 3) 4 [1759] 1759 root271Tail10 winner = winner :=
  (root271Step10).trans root271Scan11
theorem root271Scan9 : scan edge (search edge 3) 4 [1759] 1759 root271Tail9 winner = winner :=
  (root271Step9).trans root271Scan10
theorem root271Scan8 : scan edge (search edge 3) 4 [1759] 1759 root271Tail8 winner = winner :=
  (root271Step8).trans root271Scan9
theorem root271Scan7 : scan edge (search edge 3) 4 [1759] 1759 root271Tail7 winner = winner :=
  (root271Step7).trans root271Scan8
theorem root271Scan6 : scan edge (search edge 3) 4 [1759] 1759 root271Tail6 winner = winner :=
  (root271Step6).trans root271Scan7
theorem root271Scan5 : scan edge (search edge 3) 4 [1759] 1759 root271Tail5 winner = winner :=
  (root271Step5).trans root271Scan6
theorem root271Scan4 : scan edge (search edge 3) 4 [1759] 1759 root271Tail4 winner = winner :=
  (root271Step4).trans root271Scan5
theorem root271Scan3 : scan edge (search edge 3) 4 [1759] 1759 root271Tail3 winner = winner :=
  (root271Step3).trans root271Scan4
theorem root271Scan2 : scan edge (search edge 3) 4 [1759] 1759 root271Tail2 winner = winner :=
  (root271Step2).trans root271Scan3
theorem root271Scan1 : scan edge (search edge 3) 4 [1759] 1759 root271Tail1 winner = winner :=
  (root271Step1).trans root271Scan2
theorem root271Scan0 : scan edge (search edge 3) 4 [1759] 1759 root271Tail0 winner = winner :=
  (root271Step0).trans root271Scan1
set_option maxRecDepth 4096 in
theorem root271Branch : search edge 4 [1759] root271Candidates 1759 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1759] 1759 root271Tail0 winner = winner
  exact root271Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
