import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root171Candidates : List Nat := Data.row1033
set_option maxRecDepth 8192 in
theorem root171Candidates_checked : candidates edge 5 0 1033 tail172 winner.bound = root171Candidates := by
  apply candidates_eq_row edge 5 0 1033 tail172 root171Candidates winner.bound tail172_ordered
  · decide
  · decide
  · funext q
    rfl
def root171Tail54 : List Nat := []
def root171Tail53 : List Nat := 23173 :: root171Tail54
def root171Tail52 : List Nat := 22381 :: root171Tail53
def root171Tail51 : List Nat := 22291 :: root171Tail52
def root171Tail50 : List Nat := 22129 :: root171Tail51
def root171Tail49 : List Nat := 20521 :: root171Tail50
def root171Tail48 : List Nat := 20431 :: root171Tail49
def root171Tail47 : List Nat := 19861 :: root171Tail48
def root171Tail46 : List Nat := 19507 :: root171Tail47
def root171Tail45 : List Nat := 19441 :: root171Tail46
def root171Tail44 : List Nat := 19267 :: root171Tail45
def root171Tail43 : List Nat := 19213 :: root171Tail44
def root171Tail42 : List Nat := 18517 :: root171Tail43
def root171Tail41 : List Nat := 18307 :: root171Tail42
def root171Tail40 : List Nat := 16231 :: root171Tail41
def root171Tail39 : List Nat := 16063 :: root171Tail40
def root171Tail38 : List Nat := 15937 :: root171Tail39
def root171Tail37 : List Nat := 15439 :: root171Tail38
def root171Tail36 : List Nat := 14149 :: root171Tail37
def root171Tail35 : List Nat := 13681 :: root171Tail36
def root171Tail34 : List Nat := 12457 :: root171Tail35
def root171Tail33 : List Nat := 12073 :: root171Tail34
def root171Tail32 : List Nat := 11959 :: root171Tail33
def root171Tail31 : List Nat := 11743 :: root171Tail32
def root171Tail30 : List Nat := 11317 :: root171Tail31
def root171Tail29 : List Nat := 11161 :: root171Tail30
def root171Tail28 : List Nat := 11149 :: root171Tail29
def root171Tail27 : List Nat := 11083 :: root171Tail28
def root171Tail26 : List Nat := 10903 :: root171Tail27
def root171Tail25 : List Nat := 10861 :: root171Tail26
def root171Tail24 : List Nat := 10159 :: root171Tail25
def root171Tail23 : List Nat := 9907 :: root171Tail24
def root171Tail22 : List Nat := 9733 :: root171Tail23
def root171Tail21 : List Nat := 9601 :: root171Tail22
def root171Tail20 : List Nat := 8821 :: root171Tail21
def root171Tail19 : List Nat := 8719 :: root171Tail20
def root171Tail18 : List Nat := 8353 :: root171Tail19
def root171Tail17 : List Nat := 8269 :: root171Tail18
def root171Tail16 : List Nat := 8179 :: root171Tail17
def root171Tail15 : List Nat := 8089 :: root171Tail16
def root171Tail14 : List Nat := 7417 :: root171Tail15
def root171Tail13 : List Nat := 6451 :: root171Tail14
def root171Tail12 : List Nat := 6421 :: root171Tail13
def root171Tail11 : List Nat := 6301 :: root171Tail12
def root171Tail10 : List Nat := 6091 :: root171Tail11
def root171Tail9 : List Nat := 5869 :: root171Tail10
def root171Tail8 : List Nat := 5233 :: root171Tail9
def root171Tail7 : List Nat := 4231 :: root171Tail8
def root171Tail6 : List Nat := 3529 :: root171Tail7
def root171Tail5 : List Nat := 3067 :: root171Tail6
def root171Tail4 : List Nat := 2131 :: root171Tail5
def root171Tail3 : List Nat := 1759 :: root171Tail4
def root171Tail2 : List Nat := 1663 :: root171Tail3
def root171Tail1 : List Nat := 1543 :: root171Tail2
def root171Tail0 : List Nat := 1249 :: root171Tail1
def root171Candidates0 : List Nat := [2131, 11959]
set_option maxRecDepth 8192 in
theorem root171Candidates0_checked : candidates edge 4 1033 1249 root171Tail1 winner.bound = root171Candidates0 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
theorem root171Step0 : scan edge (search edge 3) 4 [1033] 1033 root171Tail0 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail1 winner := by
  rw [root171Tail0, scan, if_neg (by decide), root171Candidates0_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail1
    (if 3 ≤ root171Candidates0.length then search edge 3 [1033, 1249] root171Candidates0 2282 winner else winner) = _
  rw [if_neg (by decide)]
def root171Candidates1 : List Nat := [6451, 11161, 11743, 12073, 18517, 19507, 22381]
set_option maxRecDepth 8192 in
theorem root171Candidates1_checked : candidates edge 4 1033 1543 root171Tail2 winner.bound = root171Candidates1 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root171Child1 : search edge 3 [1033, 1543] root171Candidates1 2576 winner = winner := by decide
theorem root171Step1 : scan edge (search edge 3) 4 [1033] 1033 root171Tail1 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail2 winner := by
  rw [root171Tail1, scan, if_neg (by decide), root171Candidates1_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail2
    (if 3 ≤ root171Candidates1.length then search edge 3 [1033, 1543] root171Candidates1 2576 winner else winner) = _
  rw [if_pos (by decide), root171Child1]
def root171Candidates2 : List Nat := [4231, 11083, 13681, 16231, 22381]
set_option maxRecDepth 8192 in
theorem root171Candidates2_checked : candidates edge 4 1033 1663 root171Tail3 winner.bound = root171Candidates2 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root171Child2 : search edge 3 [1033, 1663] root171Candidates2 2696 winner = winner := by decide
theorem root171Step2 : scan edge (search edge 3) 4 [1033] 1033 root171Tail2 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail3 winner := by
  rw [root171Tail2, scan, if_neg (by decide), root171Candidates2_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail3
    (if 3 ≤ root171Candidates2.length then search edge 3 [1033, 1663] root171Candidates2 2696 winner else winner) = _
  rw [if_pos (by decide), root171Child2]
def root171Candidates3 : List Nat := [5233, 7417, 19441]
set_option maxRecDepth 8192 in
theorem root171Candidates3_checked : candidates edge 4 1033 1759 root171Tail4 winner.bound = root171Candidates3 := by
  unfold candidates
  have row : edge 1759 = (SortedNatTree.contains FastRows.row1759) := by
    change Data.row1759.contains = (SortedNatTree.contains FastRows.row1759)
    exact FastRows.row1759_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root171Child3 : search edge 3 [1033, 1759] root171Candidates3 2792 winner = winner := by decide
theorem root171Step3 : scan edge (search edge 3) 4 [1033] 1033 root171Tail3 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail4 winner := by
  rw [root171Tail3, scan, if_neg (by decide), root171Candidates3_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail4
    (if 3 ≤ root171Candidates3.length then search edge 3 [1033, 1759] root171Candidates3 2792 winner else winner) = _
  rw [if_pos (by decide), root171Child3]
def root171Candidates4 : List Nat := [10159, 19267, 19441]
set_option maxRecDepth 8192 in
theorem root171Candidates4_checked : candidates edge 4 1033 2131 root171Tail5 winner.bound = root171Candidates4 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root171Child4 : search edge 3 [1033, 2131] root171Candidates4 3164 winner = winner := by decide
theorem root171Step4 : scan edge (search edge 3) 4 [1033] 1033 root171Tail4 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail5 winner := by
  rw [root171Tail4, scan, if_neg (by decide), root171Candidates4_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail5
    (if 3 ≤ root171Candidates4.length then search edge 3 [1033, 2131] root171Candidates4 3164 winner else winner) = _
  rw [if_pos (by decide), root171Child4]
def root171Candidates5 : List Nat := [8269, 15439]
set_option maxRecDepth 8192 in
theorem root171Candidates5_checked : candidates edge 4 1033 3067 root171Tail6 winner.bound = root171Candidates5 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root171Step5 : scan edge (search edge 3) 4 [1033] 1033 root171Tail5 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail6 winner := by
  rw [root171Tail5, scan, if_neg (by decide), root171Candidates5_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail6
    (if 3 ≤ root171Candidates5.length then search edge 3 [1033, 3067] root171Candidates5 4100 winner else winner) = _
  rw [if_neg (by decide)]
def root171Candidates6 : List Nat := [20521]
set_option maxRecDepth 8192 in
theorem root171Candidates6_checked : candidates edge 4 1033 3529 root171Tail7 winner.bound = root171Candidates6 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root171Step6 : scan edge (search edge 3) 4 [1033] 1033 root171Tail6 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail7 winner := by
  rw [root171Tail6, scan, if_neg (by decide), root171Candidates6_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail7
    (if 3 ≤ root171Candidates6.length then search edge 3 [1033, 3529] root171Candidates6 4562 winner else winner) = _
  rw [if_neg (by decide)]
def root171Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root171Candidates7_checked : candidates edge 4 1033 4231 root171Tail8 winner.bound = root171Candidates7 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root171Step7 : scan edge (search edge 3) 4 [1033] 1033 root171Tail7 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail8 winner := by
  rw [root171Tail7, scan, if_neg (by decide), root171Candidates7_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail8
    (if 3 ≤ root171Candidates7.length then search edge 3 [1033, 4231] root171Candidates7 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root171Candidates8 : List Nat := [6451, 8089, 9601, 11317, 19441]
set_option maxRecDepth 8192 in
theorem root171Candidates8_checked : candidates edge 4 1033 5233 root171Tail9 winner.bound = root171Candidates8 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root171Child8 : search edge 3 [1033, 5233] root171Candidates8 6266 winner = winner := by decide
theorem root171Step8 : scan edge (search edge 3) 4 [1033] 1033 root171Tail8 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail9 winner := by
  rw [root171Tail8, scan, if_neg (by decide), root171Candidates8_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail9
    (if 3 ≤ root171Candidates8.length then search edge 3 [1033, 5233] root171Candidates8 6266 winner else winner) = _
  rw [if_pos (by decide), root171Child8]
def root171Candidates9 : List Nat := [8089, 15937]
set_option maxRecDepth 8192 in
theorem root171Candidates9_checked : candidates edge 4 1033 5869 root171Tail10 winner.bound = root171Candidates9 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root171Step9 : scan edge (search edge 3) 4 [1033] 1033 root171Tail9 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail10 winner := by
  rw [root171Tail9, scan, if_neg (by decide), root171Candidates9_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail10
    (if 3 ≤ root171Candidates9.length then search edge 3 [1033, 5869] root171Candidates9 6902 winner else winner) = _
  rw [if_neg (by decide)]
def root171Candidates10 : List Nat := [10861]
set_option maxRecDepth 8192 in
theorem root171Candidates10_checked : candidates edge 4 1033 6091 root171Tail11 winner.bound = root171Candidates10 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root171Step10 : scan edge (search edge 3) 4 [1033] 1033 root171Tail10 winner =
    scan edge (search edge 3) 4 [1033] 1033 root171Tail11 winner := by
  rw [root171Tail10, scan, if_neg (by decide), root171Candidates10_checked]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail11
    (if 3 ≤ root171Candidates10.length then search edge 3 [1033, 6091] root171Candidates10 7124 winner else winner) = _
  rw [if_neg (by decide)]
theorem root171Scan11 : scan edge (search edge 3) 4 [1033] 1033 root171Tail11 winner = winner := by rw [root171Tail11, scan, if_pos (by decide)]
theorem root171Scan10 : scan edge (search edge 3) 4 [1033] 1033 root171Tail10 winner = winner :=
  (root171Step10).trans root171Scan11
theorem root171Scan9 : scan edge (search edge 3) 4 [1033] 1033 root171Tail9 winner = winner :=
  (root171Step9).trans root171Scan10
theorem root171Scan8 : scan edge (search edge 3) 4 [1033] 1033 root171Tail8 winner = winner :=
  (root171Step8).trans root171Scan9
theorem root171Scan7 : scan edge (search edge 3) 4 [1033] 1033 root171Tail7 winner = winner :=
  (root171Step7).trans root171Scan8
theorem root171Scan6 : scan edge (search edge 3) 4 [1033] 1033 root171Tail6 winner = winner :=
  (root171Step6).trans root171Scan7
theorem root171Scan5 : scan edge (search edge 3) 4 [1033] 1033 root171Tail5 winner = winner :=
  (root171Step5).trans root171Scan6
theorem root171Scan4 : scan edge (search edge 3) 4 [1033] 1033 root171Tail4 winner = winner :=
  (root171Step4).trans root171Scan5
theorem root171Scan3 : scan edge (search edge 3) 4 [1033] 1033 root171Tail3 winner = winner :=
  (root171Step3).trans root171Scan4
theorem root171Scan2 : scan edge (search edge 3) 4 [1033] 1033 root171Tail2 winner = winner :=
  (root171Step2).trans root171Scan3
theorem root171Scan1 : scan edge (search edge 3) 4 [1033] 1033 root171Tail1 winner = winner :=
  (root171Step1).trans root171Scan2
theorem root171Scan0 : scan edge (search edge 3) 4 [1033] 1033 root171Tail0 winner = winner :=
  (root171Step0).trans root171Scan1
set_option maxRecDepth 4096 in
theorem root171Branch : search edge 4 [1033] root171Candidates 1033 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1033] 1033 root171Tail0 winner = winner
  exact root171Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
