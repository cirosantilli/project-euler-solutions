import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root334Candidates : List Nat := Data.row2269
set_option maxRecDepth 8192 in
theorem root334Candidates_checked : candidates edge 5 0 2269 tail335 winner.bound = root334Candidates := by
  apply candidates_eq_row edge 5 0 2269 tail335 root334Candidates winner.bound tail335_ordered
  · decide
  · decide
  · funext q
    rfl
def root334Tail51 : List Nat := []
def root334Tail50 : List Nat := 23761 :: root334Tail51
def root334Tail49 : List Nat := 23269 :: root334Tail50
def root334Tail48 : List Nat := 23227 :: root334Tail49
def root334Tail47 : List Nat := 22483 :: root334Tail48
def root334Tail46 : List Nat := 21727 :: root334Tail47
def root334Tail45 : List Nat := 21391 :: root334Tail46
def root334Tail44 : List Nat := 21379 :: root334Tail45
def root334Tail43 : List Nat := 21001 :: root334Tail44
def root334Tail42 : List Nat := 20521 :: root334Tail43
def root334Tail41 : List Nat := 18379 :: root334Tail42
def root334Tail40 : List Nat := 17881 :: root334Tail41
def root334Tail39 : List Nat := 17191 :: root334Tail40
def root334Tail38 : List Nat := 17077 :: root334Tail39
def root334Tail37 : List Nat := 16111 :: root334Tail38
def root334Tail36 : List Nat := 15427 :: root334Tail37
def root334Tail35 : List Nat := 15373 :: root334Tail36
def root334Tail34 : List Nat := 13633 :: root334Tail35
def root334Tail33 : List Nat := 13339 :: root334Tail34
def root334Tail32 : List Nat := 11941 :: root334Tail33
def root334Tail31 : List Nat := 11131 :: root334Tail32
def root334Tail30 : List Nat := 11071 :: root334Tail31
def root334Tail29 : List Nat := 10429 :: root334Tail30
def root334Tail28 : List Nat := 10093 :: root334Tail29
def root334Tail27 : List Nat := 9949 :: root334Tail28
def root334Tail26 : List Nat := 9697 :: root334Tail27
def root334Tail25 : List Nat := 9631 :: root334Tail26
def root334Tail24 : List Nat := 9337 :: root334Tail25
def root334Tail23 : List Nat := 9181 :: root334Tail24
def root334Tail22 : List Nat := 9151 :: root334Tail23
def root334Tail21 : List Nat := 9109 :: root334Tail22
def root334Tail20 : List Nat := 8779 :: root334Tail21
def root334Tail19 : List Nat := 8563 :: root334Tail20
def root334Tail18 : List Nat := 8443 :: root334Tail19
def root334Tail17 : List Nat := 7879 :: root334Tail18
def root334Tail16 : List Nat := 7561 :: root334Tail17
def root334Tail15 : List Nat := 7489 :: root334Tail16
def root334Tail14 : List Nat := 7237 :: root334Tail15
def root334Tail13 : List Nat := 7027 :: root334Tail14
def root334Tail12 : List Nat := 5923 :: root334Tail13
def root334Tail11 : List Nat := 5821 :: root334Tail12
def root334Tail10 : List Nat := 5347 :: root334Tail11
def root334Tail9 : List Nat := 5107 :: root334Tail10
def root334Tail8 : List Nat := 4909 :: root334Tail9
def root334Tail7 : List Nat := 4663 :: root334Tail8
def root334Tail6 : List Nat := 4621 :: root334Tail7
def root334Tail5 : List Nat := 4057 :: root334Tail6
def root334Tail4 : List Nat := 3739 :: root334Tail5
def root334Tail3 : List Nat := 3613 :: root334Tail4
def root334Tail2 : List Nat := 3373 :: root334Tail3
def root334Tail1 : List Nat := 3067 :: root334Tail2
def root334Tail0 : List Nat := 2281 :: root334Tail1
def root334Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root334Candidates0_checked : candidates edge 4 2269 2281 root334Tail1 winner.bound = root334Candidates0 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root334Step0 : scan edge (search edge 3) 4 [2269] 2269 root334Tail0 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail1 winner := by
  rw [root334Tail0, scan, if_neg (by decide), root334Candidates0_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail1
    (if 3 ≤ root334Candidates0.length then search edge 3 [2269, 2281] root334Candidates0 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates1 : List Nat := [3373, 4909, 7489, 11941]
set_option maxRecDepth 8192 in
theorem root334Candidates1_checked : candidates edge 4 2269 3067 root334Tail2 winner.bound = root334Candidates1 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root334Child1 : search edge 3 [2269, 3067] root334Candidates1 5336 winner = winner := by decide
theorem root334Step1 : scan edge (search edge 3) 4 [2269] 2269 root334Tail1 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail2 winner := by
  rw [root334Tail1, scan, if_neg (by decide), root334Candidates1_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail2
    (if 3 ≤ root334Candidates1.length then search edge 3 [2269, 3067] root334Candidates1 5336 winner else winner) = _
  rw [if_pos (by decide), root334Child1]
def root334Candidates2 : List Nat := [4621]
set_option maxRecDepth 8192 in
theorem root334Candidates2_checked : candidates edge 4 2269 3373 root334Tail3 winner.bound = root334Candidates2 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root334Step2 : scan edge (search edge 3) 4 [2269] 2269 root334Tail2 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail3 winner := by
  rw [root334Tail2, scan, if_neg (by decide), root334Candidates2_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail3
    (if 3 ≤ root334Candidates2.length then search edge 3 [2269, 3373] root334Candidates2 5642 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates3 : List Nat := [5821, 9151]
set_option maxRecDepth 8192 in
theorem root334Candidates3_checked : candidates edge 4 2269 3613 root334Tail4 winner.bound = root334Candidates3 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root334Step3 : scan edge (search edge 3) 4 [2269] 2269 root334Tail3 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail4 winner := by
  rw [root334Tail3, scan, if_neg (by decide), root334Candidates3_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail4
    (if 3 ≤ root334Candidates3.length then search edge 3 [2269, 3613] root334Candidates3 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates4 : List Nat := [5107, 9151]
set_option maxRecDepth 8192 in
theorem root334Candidates4_checked : candidates edge 4 2269 3739 root334Tail5 winner.bound = root334Candidates4 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root334Step4 : scan edge (search edge 3) 4 [2269] 2269 root334Tail4 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail5 winner := by
  rw [root334Tail4, scan, if_neg (by decide), root334Candidates4_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail5
    (if 3 ≤ root334Candidates4.length then search edge 3 [2269, 3739] root334Candidates4 6008 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates5 : List Nat := [4663, 7879, 9109, 17881]
set_option maxRecDepth 8192 in
theorem root334Candidates5_checked : candidates edge 4 2269 4057 root334Tail6 winner.bound = root334Candidates5 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root334Child5 : search edge 3 [2269, 4057] root334Candidates5 6326 winner = winner := by decide
theorem root334Step5 : scan edge (search edge 3) 4 [2269] 2269 root334Tail5 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail6 winner := by
  rw [root334Tail5, scan, if_neg (by decide), root334Candidates5_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail6
    (if 3 ≤ root334Candidates5.length then search edge 3 [2269, 4057] root334Candidates5 6326 winner else winner) = _
  rw [if_pos (by decide), root334Child5]
def root334Candidates6 : List Nat := [5347, 7489, 9337]
set_option maxRecDepth 8192 in
theorem root334Candidates6_checked : candidates edge 4 2269 4621 root334Tail7 winner.bound = root334Candidates6 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root334Child6 : search edge 3 [2269, 4621] root334Candidates6 6890 winner = winner := by decide
theorem root334Step6 : scan edge (search edge 3) 4 [2269] 2269 root334Tail6 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail7 winner := by
  rw [root334Tail6, scan, if_neg (by decide), root334Candidates6_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail7
    (if 3 ≤ root334Candidates6.length then search edge 3 [2269, 4621] root334Candidates6 6890 winner else winner) = _
  rw [if_pos (by decide), root334Child6]
def root334Candidates7 : List Nat := [7879, 9949, 17077]
set_option maxRecDepth 8192 in
theorem root334Candidates7_checked : candidates edge 4 2269 4663 root334Tail8 winner.bound = root334Candidates7 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root334Child7 : search edge 3 [2269, 4663] root334Candidates7 6932 winner = winner := by decide
theorem root334Step7 : scan edge (search edge 3) 4 [2269] 2269 root334Tail7 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail8 winner := by
  rw [root334Tail7, scan, if_neg (by decide), root334Candidates7_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail8
    (if 3 ≤ root334Candidates7.length then search edge 3 [2269, 4663] root334Candidates7 6932 winner else winner) = _
  rw [if_pos (by decide), root334Child7]
def root334Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root334Candidates8_checked : candidates edge 4 2269 4909 root334Tail9 winner.bound = root334Candidates8 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root334Step8 : scan edge (search edge 3) 4 [2269] 2269 root334Tail8 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail9 winner := by
  rw [root334Tail8, scan, if_neg (by decide), root334Candidates8_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail9
    (if 3 ≤ root334Candidates8.length then search edge 3 [2269, 4909] root334Candidates8 7178 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates9 : List Nat := [7879, 18379]
set_option maxRecDepth 8192 in
theorem root334Candidates9_checked : candidates edge 4 2269 5107 root334Tail10 winner.bound = root334Candidates9 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root334Step9 : scan edge (search edge 3) 4 [2269] 2269 root334Tail9 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail10 winner := by
  rw [root334Tail9, scan, if_neg (by decide), root334Candidates9_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail10
    (if 3 ≤ root334Candidates9.length then search edge 3 [2269, 5107] root334Candidates9 7376 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates10 : List Nat := [7027, 16111]
set_option maxRecDepth 8192 in
theorem root334Candidates10_checked : candidates edge 4 2269 5347 root334Tail11 winner.bound = root334Candidates10 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root334Step10 : scan edge (search edge 3) 4 [2269] 2269 root334Tail10 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail11 winner := by
  rw [root334Tail10, scan, if_neg (by decide), root334Candidates10_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail11
    (if 3 ≤ root334Candidates10.length then search edge 3 [2269, 5347] root334Candidates10 7616 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates11 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root334Candidates11_checked : candidates edge 4 2269 5821 root334Tail12 winner.bound = root334Candidates11 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root334Step11 : scan edge (search edge 3) 4 [2269] 2269 root334Tail11 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail12 winner := by
  rw [root334Tail11, scan, if_neg (by decide), root334Candidates11_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail12
    (if 3 ≤ root334Candidates11.length then search edge 3 [2269, 5821] root334Candidates11 8090 winner else winner) = _
  rw [if_neg (by decide)]
def root334Candidates12 : List Nat := [16111]
set_option maxRecDepth 8192 in
theorem root334Candidates12_checked : candidates edge 4 2269 5923 root334Tail13 winner.bound = root334Candidates12 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
theorem root334Step12 : scan edge (search edge 3) 4 [2269] 2269 root334Tail12 winner =
    scan edge (search edge 3) 4 [2269] 2269 root334Tail13 winner := by
  rw [root334Tail12, scan, if_neg (by decide), root334Candidates12_checked]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail13
    (if 3 ≤ root334Candidates12.length then search edge 3 [2269, 5923] root334Candidates12 8192 winner else winner) = _
  rw [if_neg (by decide)]
theorem root334Scan13 : scan edge (search edge 3) 4 [2269] 2269 root334Tail13 winner = winner := by rw [root334Tail13, scan, if_pos (by decide)]
theorem root334Scan12 : scan edge (search edge 3) 4 [2269] 2269 root334Tail12 winner = winner :=
  (root334Step12).trans root334Scan13
theorem root334Scan11 : scan edge (search edge 3) 4 [2269] 2269 root334Tail11 winner = winner :=
  (root334Step11).trans root334Scan12
theorem root334Scan10 : scan edge (search edge 3) 4 [2269] 2269 root334Tail10 winner = winner :=
  (root334Step10).trans root334Scan11
theorem root334Scan9 : scan edge (search edge 3) 4 [2269] 2269 root334Tail9 winner = winner :=
  (root334Step9).trans root334Scan10
theorem root334Scan8 : scan edge (search edge 3) 4 [2269] 2269 root334Tail8 winner = winner :=
  (root334Step8).trans root334Scan9
theorem root334Scan7 : scan edge (search edge 3) 4 [2269] 2269 root334Tail7 winner = winner :=
  (root334Step7).trans root334Scan8
theorem root334Scan6 : scan edge (search edge 3) 4 [2269] 2269 root334Tail6 winner = winner :=
  (root334Step6).trans root334Scan7
theorem root334Scan5 : scan edge (search edge 3) 4 [2269] 2269 root334Tail5 winner = winner :=
  (root334Step5).trans root334Scan6
theorem root334Scan4 : scan edge (search edge 3) 4 [2269] 2269 root334Tail4 winner = winner :=
  (root334Step4).trans root334Scan5
theorem root334Scan3 : scan edge (search edge 3) 4 [2269] 2269 root334Tail3 winner = winner :=
  (root334Step3).trans root334Scan4
theorem root334Scan2 : scan edge (search edge 3) 4 [2269] 2269 root334Tail2 winner = winner :=
  (root334Step2).trans root334Scan3
theorem root334Scan1 : scan edge (search edge 3) 4 [2269] 2269 root334Tail1 winner = winner :=
  (root334Step1).trans root334Scan2
theorem root334Scan0 : scan edge (search edge 3) 4 [2269] 2269 root334Tail0 winner = winner :=
  (root334Step0).trans root334Scan1
set_option maxRecDepth 4096 in
theorem root334Branch : search edge 4 [2269] root334Candidates 2269 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2269] 2269 root334Tail0 winner = winner
  exact root334Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
