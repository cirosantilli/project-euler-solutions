import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root258Candidates : List Nat := Data.row1663
set_option maxRecDepth 8192 in
theorem root258Candidates_checked : candidates edge 5 0 1663 tail259 winner.bound = root258Candidates := by
  apply candidates_eq_row edge 5 0 1663 tail259 root258Candidates winner.bound tail259_ordered
  · decide
  · decide
  · funext q
    rfl
def root258Tail53 : List Nat := []
def root258Tail52 : List Nat := 23371 :: root258Tail53
def root258Tail51 : List Nat := 23293 :: root258Tail52
def root258Tail50 : List Nat := 22381 :: root258Tail51
def root258Tail49 : List Nat := 21577 :: root258Tail50
def root258Tail48 : List Nat := 20899 :: root258Tail49
def root258Tail47 : List Nat := 20443 :: root258Tail48
def root258Tail46 : List Nat := 20347 :: root258Tail47
def root258Tail45 : List Nat := 20071 :: root258Tail46
def root258Tail44 : List Nat := 18583 :: root258Tail45
def root258Tail43 : List Nat := 16963 :: root258Tail44
def root258Tail42 : List Nat := 16417 :: root258Tail43
def root258Tail41 : List Nat := 16273 :: root258Tail42
def root258Tail40 : List Nat := 16231 :: root258Tail41
def root258Tail39 : List Nat := 15901 :: root258Tail40
def root258Tail38 : List Nat := 15859 :: root258Tail39
def root258Tail37 : List Nat := 14143 :: root258Tail38
def root258Tail36 : List Nat := 14107 :: root258Tail37
def root258Tail35 : List Nat := 13687 :: root258Tail36
def root258Tail34 : List Nat := 13681 :: root258Tail35
def root258Tail33 : List Nat := 13597 :: root258Tail34
def root258Tail32 : List Nat := 13339 :: root258Tail33
def root258Tail31 : List Nat := 12973 :: root258Tail32
def root258Tail30 : List Nat := 12889 :: root258Tail31
def root258Tail29 : List Nat := 12583 :: root258Tail30
def root258Tail28 : List Nat := 12409 :: root258Tail29
def root258Tail27 : List Nat := 11923 :: root258Tail28
def root258Tail26 : List Nat := 11083 :: root258Tail27
def root258Tail25 : List Nat := 10909 :: root258Tail26
def root258Tail24 : List Nat := 10831 :: root258Tail25
def root258Tail23 : List Nat := 10267 :: root258Tail24
def root258Tail22 : List Nat := 9871 :: root258Tail23
def root258Tail21 : List Nat := 9769 :: root258Tail22
def root258Tail20 : List Nat := 9619 :: root258Tail21
def root258Tail19 : List Nat := 9391 :: root258Tail20
def root258Tail18 : List Nat := 9349 :: root258Tail19
def root258Tail17 : List Nat := 9067 :: root258Tail18
def root258Tail16 : List Nat := 8443 :: root258Tail17
def root258Tail15 : List Nat := 8221 :: root258Tail16
def root258Tail14 : List Nat := 7963 :: root258Tail15
def root258Tail13 : List Nat := 7879 :: root258Tail14
def root258Tail12 : List Nat := 7219 :: root258Tail13
def root258Tail11 : List Nat := 6577 :: root258Tail12
def root258Tail10 : List Nat := 6007 :: root258Tail11
def root258Tail9 : List Nat := 5653 :: root258Tail10
def root258Tail8 : List Nat := 5449 :: root258Tail9
def root258Tail7 : List Nat := 5413 :: root258Tail8
def root258Tail6 : List Nat := 4987 :: root258Tail7
def root258Tail5 : List Nat := 4831 :: root258Tail6
def root258Tail4 : List Nat := 4231 :: root258Tail5
def root258Tail3 : List Nat := 4027 :: root258Tail4
def root258Tail2 : List Nat := 3613 :: root258Tail3
def root258Tail1 : List Nat := 3229 :: root258Tail2
def root258Tail0 : List Nat := 2857 :: root258Tail1
def root258Candidates0 : List Nat := [3229]
set_option maxRecDepth 8192 in
theorem root258Candidates0_checked : candidates edge 4 1663 2857 root258Tail1 winner.bound = root258Candidates0 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
theorem root258Step0 : scan edge (search edge 3) 4 [1663] 1663 root258Tail0 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail1 winner := by
  rw [root258Tail0, scan, if_neg (by decide), root258Candidates0_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail1
    (if 3 ≤ root258Candidates0.length then search edge 3 [1663, 2857] root258Candidates0 4520 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates1 : List Nat := [6007]
set_option maxRecDepth 8192 in
theorem root258Candidates1_checked : candidates edge 4 1663 3229 root258Tail2 winner.bound = root258Candidates1 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root258Step1 : scan edge (search edge 3) 4 [1663] 1663 root258Tail1 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail2 winner := by
  rw [root258Tail1, scan, if_neg (by decide), root258Candidates1_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail2
    (if 3 ≤ root258Candidates1.length then search edge 3 [1663, 3229] root258Candidates1 4892 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates2 : List Nat := [4831]
set_option maxRecDepth 8192 in
theorem root258Candidates2_checked : candidates edge 4 1663 3613 root258Tail3 winner.bound = root258Candidates2 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root258Step2 : scan edge (search edge 3) 4 [1663] 1663 root258Tail2 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail3 winner := by
  rw [root258Tail2, scan, if_neg (by decide), root258Candidates2_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail3
    (if 3 ≤ root258Candidates2.length then search edge 3 [1663, 3613] root258Candidates2 5276 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates3 : List Nat := [5413]
set_option maxRecDepth 8192 in
theorem root258Candidates3_checked : candidates edge 4 1663 4027 root258Tail4 winner.bound = root258Candidates3 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
theorem root258Step3 : scan edge (search edge 3) 4 [1663] 1663 root258Tail3 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail4 winner := by
  rw [root258Tail3, scan, if_neg (by decide), root258Candidates3_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail4
    (if 3 ≤ root258Candidates3.length then search edge 3 [1663, 4027] root258Candidates3 5690 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates4 : List Nat := [6577, 9769, 14143]
set_option maxRecDepth 8192 in
theorem root258Candidates4_checked : candidates edge 4 1663 4231 root258Tail5 winner.bound = root258Candidates4 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root258Child4 : search edge 3 [1663, 4231] root258Candidates4 5894 winner = winner := by decide
theorem root258Step4 : scan edge (search edge 3) 4 [1663] 1663 root258Tail4 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail5 winner := by
  rw [root258Tail4, scan, if_neg (by decide), root258Candidates4_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail5
    (if 3 ≤ root258Candidates4.length then search edge 3 [1663, 4231] root258Candidates4 5894 winner else winner) = _
  rw [if_pos (by decide), root258Child4]
def root258Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root258Candidates5_checked : candidates edge 4 1663 4831 root258Tail6 winner.bound = root258Candidates5 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root258Step5 : scan edge (search edge 3) 4 [1663] 1663 root258Tail5 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail6 winner := by
  rw [root258Tail5, scan, if_neg (by decide), root258Candidates5_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail6
    (if 3 ≤ root258Candidates5.length then search edge 3 [1663, 4831] root258Candidates5 6494 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates6 : List Nat := [12583]
set_option maxRecDepth 8192 in
theorem root258Candidates6_checked : candidates edge 4 1663 4987 root258Tail7 winner.bound = root258Candidates6 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root258Step6 : scan edge (search edge 3) 4 [1663] 1663 root258Tail6 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail7 winner := by
  rw [root258Tail6, scan, if_neg (by decide), root258Candidates6_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail7
    (if 3 ≤ root258Candidates6.length then search edge 3 [1663, 4987] root258Candidates6 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates7 : List Nat := [9619]
set_option maxRecDepth 8192 in
theorem root258Candidates7_checked : candidates edge 4 1663 5413 root258Tail8 winner.bound = root258Candidates7 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root258Step7 : scan edge (search edge 3) 4 [1663] 1663 root258Tail7 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail8 winner := by
  rw [root258Tail7, scan, if_neg (by decide), root258Candidates7_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail8
    (if 3 ≤ root258Candidates7.length then search edge 3 [1663, 5413] root258Candidates7 7076 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates8 : List Nat := [15901]
set_option maxRecDepth 8192 in
theorem root258Candidates8_checked : candidates edge 4 1663 5449 root258Tail9 winner.bound = root258Candidates8 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root258Step8 : scan edge (search edge 3) 4 [1663] 1663 root258Tail8 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail9 winner := by
  rw [root258Tail8, scan, if_neg (by decide), root258Candidates8_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail9
    (if 3 ≤ root258Candidates8.length then search edge 3 [1663, 5449] root258Candidates8 7112 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates9 : List Nat := [11083]
set_option maxRecDepth 8192 in
theorem root258Candidates9_checked : candidates edge 4 1663 5653 root258Tail10 winner.bound = root258Candidates9 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root258Step9 : scan edge (search edge 3) 4 [1663] 1663 root258Tail9 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail10 winner := by
  rw [root258Tail9, scan, if_neg (by decide), root258Candidates9_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail10
    (if 3 ≤ root258Candidates9.length then search edge 3 [1663, 5653] root258Candidates9 7316 winner else winner) = _
  rw [if_neg (by decide)]
def root258Candidates10 : List Nat := [8443]
set_option maxRecDepth 8192 in
theorem root258Candidates10_checked : candidates edge 4 1663 6007 root258Tail11 winner.bound = root258Candidates10 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root258Step10 : scan edge (search edge 3) 4 [1663] 1663 root258Tail10 winner =
    scan edge (search edge 3) 4 [1663] 1663 root258Tail11 winner := by
  rw [root258Tail10, scan, if_neg (by decide), root258Candidates10_checked]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail11
    (if 3 ≤ root258Candidates10.length then search edge 3 [1663, 6007] root258Candidates10 7670 winner else winner) = _
  rw [if_neg (by decide)]
theorem root258Scan11 : scan edge (search edge 3) 4 [1663] 1663 root258Tail11 winner = winner := by rw [root258Tail11, scan, if_pos (by decide)]
theorem root258Scan10 : scan edge (search edge 3) 4 [1663] 1663 root258Tail10 winner = winner :=
  (root258Step10).trans root258Scan11
theorem root258Scan9 : scan edge (search edge 3) 4 [1663] 1663 root258Tail9 winner = winner :=
  (root258Step9).trans root258Scan10
theorem root258Scan8 : scan edge (search edge 3) 4 [1663] 1663 root258Tail8 winner = winner :=
  (root258Step8).trans root258Scan9
theorem root258Scan7 : scan edge (search edge 3) 4 [1663] 1663 root258Tail7 winner = winner :=
  (root258Step7).trans root258Scan8
theorem root258Scan6 : scan edge (search edge 3) 4 [1663] 1663 root258Tail6 winner = winner :=
  (root258Step6).trans root258Scan7
theorem root258Scan5 : scan edge (search edge 3) 4 [1663] 1663 root258Tail5 winner = winner :=
  (root258Step5).trans root258Scan6
theorem root258Scan4 : scan edge (search edge 3) 4 [1663] 1663 root258Tail4 winner = winner :=
  (root258Step4).trans root258Scan5
theorem root258Scan3 : scan edge (search edge 3) 4 [1663] 1663 root258Tail3 winner = winner :=
  (root258Step3).trans root258Scan4
theorem root258Scan2 : scan edge (search edge 3) 4 [1663] 1663 root258Tail2 winner = winner :=
  (root258Step2).trans root258Scan3
theorem root258Scan1 : scan edge (search edge 3) 4 [1663] 1663 root258Tail1 winner = winner :=
  (root258Step1).trans root258Scan2
theorem root258Scan0 : scan edge (search edge 3) 4 [1663] 1663 root258Tail0 winner = winner :=
  (root258Step0).trans root258Scan1
set_option maxRecDepth 4096 in
theorem root258Branch : search edge 4 [1663] root258Candidates 1663 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1663] 1663 root258Tail0 winner = winner
  exact root258Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
