import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root331Candidates : List Nat := Data.row2243
set_option maxRecDepth 8192 in
theorem root331Candidates_checked : candidates edge 5 0 2243 tail332 winner.bound = root331Candidates := by
  apply candidates_eq_row edge 5 0 2243 tail332 root331Candidates winner.bound tail332_ordered
  · decide
  · decide
  · funext q
    rfl
def root331Tail40 : List Nat := []
def root331Tail39 : List Nat := 23459 :: root331Tail40
def root331Tail38 : List Nat := 22643 :: root331Tail39
def root331Tail37 : List Nat := 21893 :: root331Tail38
def root331Tail36 : List Nat := 21683 :: root331Tail37
def root331Tail35 : List Nat := 21191 :: root331Tail36
def root331Tail34 : List Nat := 21107 :: root331Tail35
def root331Tail33 : List Nat := 19793 :: root331Tail34
def root331Tail32 : List Nat := 19457 :: root331Tail33
def root331Tail31 : List Nat := 18797 :: root331Tail32
def root331Tail30 : List Nat := 18251 :: root331Tail31
def root331Tail29 : List Nat := 18191 :: root331Tail30
def root331Tail28 : List Nat := 17189 :: root331Tail29
def root331Tail27 : List Nat := 17033 :: root331Tail28
def root331Tail26 : List Nat := 16931 :: root331Tail27
def root331Tail25 : List Nat := 16487 :: root331Tail26
def root331Tail24 : List Nat := 16427 :: root331Tail25
def root331Tail23 : List Nat := 16349 :: root331Tail24
def root331Tail22 : List Nat := 13127 :: root331Tail23
def root331Tail21 : List Nat := 12743 :: root331Tail22
def root331Tail20 : List Nat := 12527 :: root331Tail21
def root331Tail19 : List Nat := 10433 :: root331Tail20
def root331Tail18 : List Nat := 10133 :: root331Tail19
def root331Tail17 : List Nat := 10061 :: root331Tail18
def root331Tail16 : List Nat := 9497 :: root331Tail17
def root331Tail15 : List Nat := 9377 :: root331Tail16
def root331Tail14 : List Nat := 8573 :: root331Tail15
def root331Tail13 : List Nat := 8081 :: root331Tail14
def root331Tail12 : List Nat := 7487 :: root331Tail13
def root331Tail11 : List Nat := 7193 :: root331Tail12
def root331Tail10 : List Nat := 6977 :: root331Tail11
def root331Tail9 : List Nat := 6899 :: root331Tail10
def root331Tail8 : List Nat := 6599 :: root331Tail9
def root331Tail7 : List Nat := 5927 :: root331Tail8
def root331Tail6 : List Nat := 5477 :: root331Tail7
def root331Tail5 : List Nat := 4457 :: root331Tail6
def root331Tail4 : List Nat := 4133 :: root331Tail5
def root331Tail3 : List Nat := 3539 :: root331Tail4
def root331Tail2 : List Nat := 2789 :: root331Tail3
def root331Tail1 : List Nat := 2411 :: root331Tail2
def root331Tail0 : List Nat := 2273 :: root331Tail1
def root331Candidates0 : List Nat := [2789, 18797]
set_option maxRecDepth 8192 in
theorem root331Candidates0_checked : candidates edge 4 2243 2273 root331Tail1 winner.bound = root331Candidates0 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
theorem root331Step0 : scan edge (search edge 3) 4 [2243] 2243 root331Tail0 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail1 winner := by
  rw [root331Tail0, scan, if_neg (by decide), root331Candidates0_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail1
    (if 3 ≤ root331Candidates0.length then search edge 3 [2243, 2273] root331Candidates0 4516 winner else winner) = _
  rw [if_neg (by decide)]
def root331Candidates1 : List Nat := [9497]
set_option maxRecDepth 8192 in
theorem root331Candidates1_checked : candidates edge 4 2243 2411 root331Tail2 winner.bound = root331Candidates1 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root331Step1 : scan edge (search edge 3) 4 [2243] 2243 root331Tail1 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail2 winner := by
  rw [root331Tail1, scan, if_neg (by decide), root331Candidates1_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail2
    (if 3 ≤ root331Candidates1.length then search edge 3 [2243, 2411] root331Candidates1 4654 winner else winner) = _
  rw [if_neg (by decide)]
def root331Candidates2 : List Nat := [4133, 9497]
set_option maxRecDepth 8192 in
theorem root331Candidates2_checked : candidates edge 4 2243 2789 root331Tail3 winner.bound = root331Candidates2 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root331Step2 : scan edge (search edge 3) 4 [2243] 2243 root331Tail2 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail3 winner := by
  rw [root331Tail2, scan, if_neg (by decide), root331Candidates2_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail3
    (if 3 ≤ root331Candidates2.length then search edge 3 [2243, 2789] root331Candidates2 5032 winner else winner) = _
  rw [if_neg (by decide)]
def root331Candidates3 : List Nat := [19457]
set_option maxRecDepth 8192 in
theorem root331Candidates3_checked : candidates edge 4 2243 3539 root331Tail4 winner.bound = root331Candidates3 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root331Step3 : scan edge (search edge 3) 4 [2243] 2243 root331Tail3 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail4 winner := by
  rw [root331Tail3, scan, if_neg (by decide), root331Candidates3_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail4
    (if 3 ≤ root331Candidates3.length then search edge 3 [2243, 3539] root331Candidates3 5782 winner else winner) = _
  rw [if_neg (by decide)]
def root331Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root331Candidates4_checked : candidates edge 4 2243 4133 root331Tail5 winner.bound = root331Candidates4 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root331Step4 : scan edge (search edge 3) 4 [2243] 2243 root331Tail4 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail5 winner := by
  rw [root331Tail4, scan, if_neg (by decide), root331Candidates4_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail5
    (if 3 ≤ root331Candidates4.length then search edge 3 [2243, 4133] root331Candidates4 6376 winner else winner) = _
  rw [if_neg (by decide)]
def root331Candidates5 : List Nat := [10061]
set_option maxRecDepth 8192 in
theorem root331Candidates5_checked : candidates edge 4 2243 4457 root331Tail6 winner.bound = root331Candidates5 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root331Step5 : scan edge (search edge 3) 4 [2243] 2243 root331Tail5 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail6 winner := by
  rw [root331Tail5, scan, if_neg (by decide), root331Candidates5_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail6
    (if 3 ≤ root331Candidates5.length then search edge 3 [2243, 4457] root331Candidates5 6700 winner else winner) = _
  rw [if_neg (by decide)]
def root331Candidates6 : List Nat := [5927, 6599, 8081]
set_option maxRecDepth 8192 in
theorem root331Candidates6_checked : candidates edge 4 2243 5477 root331Tail7 winner.bound = root331Candidates6 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root331Child6 : search edge 3 [2243, 5477] root331Candidates6 7720 winner = winner := by decide
theorem root331Step6 : scan edge (search edge 3) 4 [2243] 2243 root331Tail6 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail7 winner := by
  rw [root331Tail6, scan, if_neg (by decide), root331Candidates6_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail7
    (if 3 ≤ root331Candidates6.length then search edge 3 [2243, 5477] root331Candidates6 7720 winner else winner) = _
  rw [if_pos (by decide), root331Child6]
def root331Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root331Candidates7_checked : candidates edge 4 2243 5927 root331Tail8 winner.bound = root331Candidates7 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root331Step7 : scan edge (search edge 3) 4 [2243] 2243 root331Tail7 winner =
    scan edge (search edge 3) 4 [2243] 2243 root331Tail8 winner := by
  rw [root331Tail7, scan, if_neg (by decide), root331Candidates7_checked]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail8
    (if 3 ≤ root331Candidates7.length then search edge 3 [2243, 5927] root331Candidates7 8170 winner else winner) = _
  rw [if_neg (by decide)]
theorem root331Scan8 : scan edge (search edge 3) 4 [2243] 2243 root331Tail8 winner = winner := by rw [root331Tail8, scan, if_pos (by decide)]
theorem root331Scan7 : scan edge (search edge 3) 4 [2243] 2243 root331Tail7 winner = winner :=
  (root331Step7).trans root331Scan8
theorem root331Scan6 : scan edge (search edge 3) 4 [2243] 2243 root331Tail6 winner = winner :=
  (root331Step6).trans root331Scan7
theorem root331Scan5 : scan edge (search edge 3) 4 [2243] 2243 root331Tail5 winner = winner :=
  (root331Step5).trans root331Scan6
theorem root331Scan4 : scan edge (search edge 3) 4 [2243] 2243 root331Tail4 winner = winner :=
  (root331Step4).trans root331Scan5
theorem root331Scan3 : scan edge (search edge 3) 4 [2243] 2243 root331Tail3 winner = winner :=
  (root331Step3).trans root331Scan4
theorem root331Scan2 : scan edge (search edge 3) 4 [2243] 2243 root331Tail2 winner = winner :=
  (root331Step2).trans root331Scan3
theorem root331Scan1 : scan edge (search edge 3) 4 [2243] 2243 root331Tail1 winner = winner :=
  (root331Step1).trans root331Scan2
theorem root331Scan0 : scan edge (search edge 3) 4 [2243] 2243 root331Tail0 winner = winner :=
  (root331Step0).trans root331Scan1
set_option maxRecDepth 4096 in
theorem root331Branch : search edge 4 [2243] root331Candidates 2243 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2243] 2243 root331Tail0 winner = winner
  exact root331Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
