import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root333Candidates : List Nat := Data.row2267
set_option maxRecDepth 8192 in
theorem root333Candidates_checked : candidates edge 5 0 2267 tail334 winner.bound = root333Candidates := by
  apply candidates_eq_row edge 5 0 2267 tail334 root333Candidates winner.bound tail334_ordered
  · decide
  · decide
  · funext q
    rfl
def root333Tail46 : List Nat := []
def root333Tail45 : List Nat := 23687 :: root333Tail46
def root333Tail44 : List Nat := 22751 :: root333Tail45
def root333Tail43 : List Nat := 21839 :: root333Tail44
def root333Tail42 : List Nat := 21683 :: root333Tail43
def root333Tail41 : List Nat := 19937 :: root333Tail42
def root333Tail40 : List Nat := 19841 :: root333Tail41
def root333Tail39 : List Nat := 18731 :: root333Tail40
def root333Tail38 : List Nat := 18269 :: root333Tail39
def root333Tail37 : List Nat := 18257 :: root333Tail38
def root333Tail36 : List Nat := 16883 :: root333Tail37
def root333Tail35 : List Nat := 16481 :: root333Tail36
def root333Tail34 : List Nat := 15551 :: root333Tail35
def root333Tail33 : List Nat := 15497 :: root333Tail34
def root333Tail32 : List Nat := 13259 :: root333Tail33
def root333Tail31 : List Nat := 12659 :: root333Tail32
def root333Tail30 : List Nat := 12611 :: root333Tail31
def root333Tail29 : List Nat := 12041 :: root333Tail30
def root333Tail28 : List Nat := 10973 :: root333Tail29
def root333Tail27 : List Nat := 10331 :: root333Tail28
def root333Tail26 : List Nat := 9551 :: root333Tail27
def root333Tail25 : List Nat := 9413 :: root333Tail26
def root333Tail24 : List Nat := 9257 :: root333Tail25
def root333Tail23 : List Nat := 9227 :: root333Tail24
def root333Tail22 : List Nat := 9203 :: root333Tail23
def root333Tail21 : List Nat := 8861 :: root333Tail22
def root333Tail20 : List Nat := 8819 :: root333Tail21
def root333Tail19 : List Nat := 8741 :: root333Tail20
def root333Tail18 : List Nat := 8219 :: root333Tail19
def root333Tail17 : List Nat := 8147 :: root333Tail18
def root333Tail16 : List Nat := 7673 :: root333Tail17
def root333Tail15 : List Nat := 6917 :: root333Tail16
def root333Tail14 : List Nat := 6833 :: root333Tail15
def root333Tail13 : List Nat := 6791 :: root333Tail14
def root333Tail12 : List Nat := 6389 :: root333Tail13
def root333Tail11 : List Nat := 6053 :: root333Tail12
def root333Tail10 : List Nat := 5897 :: root333Tail11
def root333Tail9 : List Nat := 5879 :: root333Tail10
def root333Tail8 : List Nat := 5867 :: root333Tail9
def root333Tail7 : List Nat := 5843 :: root333Tail8
def root333Tail6 : List Nat := 5669 :: root333Tail7
def root333Tail5 : List Nat := 4721 :: root333Tail6
def root333Tail4 : List Nat := 4637 :: root333Tail5
def root333Tail3 : List Nat := 3527 :: root333Tail4
def root333Tail2 : List Nat := 3323 :: root333Tail3
def root333Tail1 : List Nat := 2939 :: root333Tail2
def root333Tail0 : List Nat := 2729 :: root333Tail1
def root333Candidates0 : List Nat := [10973, 12659]
set_option maxRecDepth 8192 in
theorem root333Candidates0_checked : candidates edge 4 2267 2729 root333Tail1 winner.bound = root333Candidates0 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root333Step0 : scan edge (search edge 3) 4 [2267] 2267 root333Tail0 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail1 winner := by
  rw [root333Tail0, scan, if_neg (by decide), root333Candidates0_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail1
    (if 3 ≤ root333Candidates0.length then search edge 3 [2267, 2729] root333Candidates0 4996 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates1 : List Nat := [9203, 9551, 12041]
set_option maxRecDepth 8192 in
theorem root333Candidates1_checked : candidates edge 4 2267 2939 root333Tail2 winner.bound = root333Candidates1 := by
  unfold candidates
  have row : edge 2939 = (SortedNatTree.contains FastRows.row2939) := by
    change Data.row2939.contains = (SortedNatTree.contains FastRows.row2939)
    exact FastRows.row2939_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root333Child1 : search edge 3 [2267, 2939] root333Candidates1 5206 winner = winner := by decide
theorem root333Step1 : scan edge (search edge 3) 4 [2267] 2267 root333Tail1 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail2 winner := by
  rw [root333Tail1, scan, if_neg (by decide), root333Candidates1_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail2
    (if 3 ≤ root333Candidates1.length then search edge 3 [2267, 2939] root333Candidates1 5206 winner else winner) = _
  rw [if_pos (by decide), root333Child1]
def root333Candidates2 : List Nat := [8861, 18731]
set_option maxRecDepth 8192 in
theorem root333Candidates2_checked : candidates edge 4 2267 3323 root333Tail3 winner.bound = root333Candidates2 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
theorem root333Step2 : scan edge (search edge 3) 4 [2267] 2267 root333Tail2 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail3 winner := by
  rw [root333Tail2, scan, if_neg (by decide), root333Candidates2_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail3
    (if 3 ≤ root333Candidates2.length then search edge 3 [2267, 3323] root333Candidates2 5590 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates3 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root333Candidates3_checked : candidates edge 4 2267 3527 root333Tail4 winner.bound = root333Candidates3 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root333Step3 : scan edge (search edge 3) 4 [2267] 2267 root333Tail3 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail4 winner := by
  rw [root333Tail3, scan, if_neg (by decide), root333Candidates3_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail4
    (if 3 ≤ root333Candidates3.length then search edge 3 [2267, 3527] root333Candidates3 5794 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root333Candidates4_checked : candidates edge 4 2267 4637 root333Tail5 winner.bound = root333Candidates4 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root333Step4 : scan edge (search edge 3) 4 [2267] 2267 root333Tail4 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail5 winner := by
  rw [root333Tail4, scan, if_neg (by decide), root333Candidates4_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail5
    (if 3 ≤ root333Candidates4.length then search edge 3 [2267, 4637] root333Candidates4 6904 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates5 : List Nat := [9257]
set_option maxRecDepth 8192 in
theorem root333Candidates5_checked : candidates edge 4 2267 4721 root333Tail6 winner.bound = root333Candidates5 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root333Step5 : scan edge (search edge 3) 4 [2267] 2267 root333Tail5 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail6 winner := by
  rw [root333Tail5, scan, if_neg (by decide), root333Candidates5_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail6
    (if 3 ≤ root333Candidates5.length then search edge 3 [2267, 4721] root333Candidates5 6988 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates6 : List Nat := [5879, 8819, 15497]
set_option maxRecDepth 8192 in
theorem root333Candidates6_checked : candidates edge 4 2267 5669 root333Tail7 winner.bound = root333Candidates6 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root333Child6 : search edge 3 [2267, 5669] root333Candidates6 7936 winner = winner := by decide
theorem root333Step6 : scan edge (search edge 3) 4 [2267] 2267 root333Tail6 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail7 winner := by
  rw [root333Tail6, scan, if_neg (by decide), root333Candidates6_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail7
    (if 3 ≤ root333Candidates6.length then search edge 3 [2267, 5669] root333Candidates6 7936 winner else winner) = _
  rw [if_pos (by decide), root333Child6]
def root333Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root333Candidates7_checked : candidates edge 4 2267 5843 root333Tail8 winner.bound = root333Candidates7 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root333Step7 : scan edge (search edge 3) 4 [2267] 2267 root333Tail7 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail8 winner := by
  rw [root333Tail7, scan, if_neg (by decide), root333Candidates7_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail8
    (if 3 ≤ root333Candidates7.length then search edge 3 [2267, 5843] root333Candidates7 8110 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root333Candidates8_checked : candidates edge 4 2267 5867 root333Tail9 winner.bound = root333Candidates8 := by
  unfold candidates
  have row : edge 5867 = (SortedNatTree.contains FastRows.row5867) := by
    change Data.row5867.contains = (SortedNatTree.contains FastRows.row5867)
    exact FastRows.row5867_checked.symm
  rw [row]
  decide
theorem root333Step8 : scan edge (search edge 3) 4 [2267] 2267 root333Tail8 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail9 winner := by
  rw [root333Tail8, scan, if_neg (by decide), root333Candidates8_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail9
    (if 3 ≤ root333Candidates8.length then search edge 3 [2267, 5867] root333Candidates8 8134 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates9 : List Nat := [16883]
set_option maxRecDepth 8192 in
theorem root333Candidates9_checked : candidates edge 4 2267 5879 root333Tail10 winner.bound = root333Candidates9 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root333Step9 : scan edge (search edge 3) 4 [2267] 2267 root333Tail9 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail10 winner := by
  rw [root333Tail9, scan, if_neg (by decide), root333Candidates9_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail10
    (if 3 ≤ root333Candidates9.length then search edge 3 [2267, 5879] root333Candidates9 8146 winner else winner) = _
  rw [if_neg (by decide)]
def root333Candidates10 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root333Candidates10_checked : candidates edge 4 2267 5897 root333Tail11 winner.bound = root333Candidates10 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root333Step10 : scan edge (search edge 3) 4 [2267] 2267 root333Tail10 winner =
    scan edge (search edge 3) 4 [2267] 2267 root333Tail11 winner := by
  rw [root333Tail10, scan, if_neg (by decide), root333Candidates10_checked]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail11
    (if 3 ≤ root333Candidates10.length then search edge 3 [2267, 5897] root333Candidates10 8164 winner else winner) = _
  rw [if_neg (by decide)]
theorem root333Scan11 : scan edge (search edge 3) 4 [2267] 2267 root333Tail11 winner = winner := by rw [root333Tail11, scan, if_pos (by decide)]
theorem root333Scan10 : scan edge (search edge 3) 4 [2267] 2267 root333Tail10 winner = winner :=
  (root333Step10).trans root333Scan11
theorem root333Scan9 : scan edge (search edge 3) 4 [2267] 2267 root333Tail9 winner = winner :=
  (root333Step9).trans root333Scan10
theorem root333Scan8 : scan edge (search edge 3) 4 [2267] 2267 root333Tail8 winner = winner :=
  (root333Step8).trans root333Scan9
theorem root333Scan7 : scan edge (search edge 3) 4 [2267] 2267 root333Tail7 winner = winner :=
  (root333Step7).trans root333Scan8
theorem root333Scan6 : scan edge (search edge 3) 4 [2267] 2267 root333Tail6 winner = winner :=
  (root333Step6).trans root333Scan7
theorem root333Scan5 : scan edge (search edge 3) 4 [2267] 2267 root333Tail5 winner = winner :=
  (root333Step5).trans root333Scan6
theorem root333Scan4 : scan edge (search edge 3) 4 [2267] 2267 root333Tail4 winner = winner :=
  (root333Step4).trans root333Scan5
theorem root333Scan3 : scan edge (search edge 3) 4 [2267] 2267 root333Tail3 winner = winner :=
  (root333Step3).trans root333Scan4
theorem root333Scan2 : scan edge (search edge 3) 4 [2267] 2267 root333Tail2 winner = winner :=
  (root333Step2).trans root333Scan3
theorem root333Scan1 : scan edge (search edge 3) 4 [2267] 2267 root333Tail1 winner = winner :=
  (root333Step1).trans root333Scan2
theorem root333Scan0 : scan edge (search edge 3) 4 [2267] 2267 root333Tail0 winner = winner :=
  (root333Step0).trans root333Scan1
set_option maxRecDepth 4096 in
theorem root333Branch : search edge 4 [2267] root333Candidates 2267 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2267] 2267 root333Tail0 winner = winner
  exact root333Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
