import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root193Candidates : List Nat := Data.row1193
set_option maxRecDepth 8192 in
theorem root193Candidates_checked : candidates edge 5 0 1193 tail194 winner.bound = root193Candidates := by
  apply candidates_eq_row edge 5 0 1193 tail194 root193Candidates winner.bound tail194_ordered
  · decide
  · decide
  · funext q
    rfl
def root193Tail53 : List Nat := []
def root193Tail52 : List Nat := 23873 :: root193Tail53
def root193Tail51 : List Nat := 23189 :: root193Tail52
def root193Tail50 : List Nat := 23159 :: root193Tail51
def root193Tail49 : List Nat := 22739 :: root193Tail50
def root193Tail48 : List Nat := 20753 :: root193Tail49
def root193Tail47 : List Nat := 20549 :: root193Tail48
def root193Tail46 : List Nat := 17477 :: root193Tail47
def root193Tail45 : List Nat := 16763 :: root193Tail46
def root193Tail44 : List Nat := 16007 :: root193Tail45
def root193Tail43 : List Nat := 15377 :: root193Tail44
def root193Tail42 : List Nat := 15173 :: root193Tail43
def root193Tail41 : List Nat := 15101 :: root193Tail42
def root193Tail40 : List Nat := 14627 :: root193Tail41
def root193Tail39 : List Nat := 14549 :: root193Tail40
def root193Tail38 : List Nat := 14543 :: root193Tail39
def root193Tail37 : List Nat := 14423 :: root193Tail38
def root193Tail36 : List Nat := 13757 :: root193Tail37
def root193Tail35 : List Nat := 13577 :: root193Tail36
def root193Tail34 : List Nat := 13487 :: root193Tail35
def root193Tail33 : List Nat := 13043 :: root193Tail34
def root193Tail32 : List Nat := 11813 :: root193Tail33
def root193Tail31 : List Nat := 11471 :: root193Tail32
def root193Tail30 : List Nat := 10859 :: root193Tail31
def root193Tail29 : List Nat := 10691 :: root193Tail30
def root193Tail28 : List Nat := 10211 :: root193Tail29
def root193Tail27 : List Nat := 9749 :: root193Tail28
def root193Tail26 : List Nat := 9377 :: root193Tail27
def root193Tail25 : List Nat := 9341 :: root193Tail26
def root193Tail24 : List Nat := 9029 :: root193Tail25
def root193Tail23 : List Nat := 8543 :: root193Tail24
def root193Tail22 : List Nat := 8501 :: root193Tail23
def root193Tail21 : List Nat := 8039 :: root193Tail22
def root193Tail20 : List Nat := 7829 :: root193Tail21
def root193Tail19 : List Nat := 7559 :: root193Tail20
def root193Tail18 : List Nat := 7193 :: root193Tail19
def root193Tail17 : List Nat := 7103 :: root193Tail18
def root193Tail16 : List Nat := 6779 :: root193Tail17
def root193Tail15 : List Nat := 6689 :: root193Tail16
def root193Tail14 : List Nat := 6131 :: root193Tail15
def root193Tail13 : List Nat := 5801 :: root193Tail14
def root193Tail12 : List Nat := 5519 :: root193Tail13
def root193Tail11 : List Nat := 5507 :: root193Tail12
def root193Tail10 : List Nat := 5009 :: root193Tail11
def root193Tail9 : List Nat := 4931 :: root193Tail10
def root193Tail8 : List Nat := 4133 :: root193Tail9
def root193Tail7 : List Nat := 3923 :: root193Tail8
def root193Tail6 : List Nat := 3671 :: root193Tail7
def root193Tail5 : List Nat := 2777 :: root193Tail6
def root193Tail4 : List Nat := 2441 :: root193Tail5
def root193Tail3 : List Nat := 2153 :: root193Tail4
def root193Tail2 : List Nat := 1901 :: root193Tail3
def root193Tail1 : List Nat := 1889 :: root193Tail2
def root193Tail0 : List Nat := 1553 :: root193Tail1
def root193Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root193Candidates0_checked : candidates edge 4 1193 1553 root193Tail1 winner.bound = root193Candidates0 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
theorem root193Step0 : scan edge (search edge 3) 4 [1193] 1193 root193Tail0 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail1 winner := by
  rw [root193Tail0, scan, if_neg (by decide), root193Candidates0_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail1
    (if 3 ≤ root193Candidates0.length then search edge 3 [1193, 1553] root193Candidates0 2746 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates1 : List Nat := [2777]
set_option maxRecDepth 8192 in
theorem root193Candidates1_checked : candidates edge 4 1193 1889 root193Tail2 winner.bound = root193Candidates1 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
theorem root193Step1 : scan edge (search edge 3) 4 [1193] 1193 root193Tail1 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail2 winner := by
  rw [root193Tail1, scan, if_neg (by decide), root193Candidates1_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail2
    (if 3 ≤ root193Candidates1.length then search edge 3 [1193, 1889] root193Candidates1 3082 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates2 : List Nat := [2153, 20753]
set_option maxRecDepth 8192 in
theorem root193Candidates2_checked : candidates edge 4 1193 1901 root193Tail3 winner.bound = root193Candidates2 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root193Step2 : scan edge (search edge 3) 4 [1193] 1193 root193Tail2 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail3 winner := by
  rw [root193Tail2, scan, if_neg (by decide), root193Candidates2_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail3
    (if 3 ≤ root193Candidates2.length then search edge 3 [1193, 1901] root193Candidates2 3094 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root193Candidates3_checked : candidates edge 4 1193 2153 root193Tail4 winner.bound = root193Candidates3 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
theorem root193Step3 : scan edge (search edge 3) 4 [1193] 1193 root193Tail3 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail4 winner := by
  rw [root193Tail3, scan, if_neg (by decide), root193Candidates3_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail4
    (if 3 ≤ root193Candidates3.length then search edge 3 [1193, 2153] root193Candidates3 3346 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates4 : List Nat := [3671, 9341]
set_option maxRecDepth 8192 in
theorem root193Candidates4_checked : candidates edge 4 1193 2441 root193Tail5 winner.bound = root193Candidates4 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
theorem root193Step4 : scan edge (search edge 3) 4 [1193] 1193 root193Tail4 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail5 winner := by
  rw [root193Tail4, scan, if_neg (by decide), root193Candidates4_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail5
    (if 3 ≤ root193Candidates4.length then search edge 3 [1193, 2441] root193Candidates4 3634 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates5 : List Nat := [8039]
set_option maxRecDepth 8192 in
theorem root193Candidates5_checked : candidates edge 4 1193 2777 root193Tail6 winner.bound = root193Candidates5 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root193Step5 : scan edge (search edge 3) 4 [1193] 1193 root193Tail5 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail6 winner := by
  rw [root193Tail5, scan, if_neg (by decide), root193Candidates5_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail6
    (if 3 ≤ root193Candidates5.length then search edge 3 [1193, 2777] root193Candidates5 3970 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates6 : List Nat := [6131, 8501, 15173]
set_option maxRecDepth 8192 in
theorem root193Candidates6_checked : candidates edge 4 1193 3671 root193Tail7 winner.bound = root193Candidates6 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root193Child6 : search edge 3 [1193, 3671] root193Candidates6 4864 winner = winner := by decide
theorem root193Step6 : scan edge (search edge 3) 4 [1193] 1193 root193Tail6 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail7 winner := by
  rw [root193Tail6, scan, if_neg (by decide), root193Candidates6_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail7
    (if 3 ≤ root193Candidates6.length then search edge 3 [1193, 3671] root193Candidates6 4864 winner else winner) = _
  rw [if_pos (by decide), root193Child6]
def root193Candidates7 : List Nat := [4133, 5519, 9377]
set_option maxRecDepth 8192 in
theorem root193Candidates7_checked : candidates edge 4 1193 3923 root193Tail8 winner.bound = root193Candidates7 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root193Child7 : search edge 3 [1193, 3923] root193Candidates7 5116 winner = winner := by decide
theorem root193Step7 : scan edge (search edge 3) 4 [1193] 1193 root193Tail7 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail8 winner := by
  rw [root193Tail7, scan, if_neg (by decide), root193Candidates7_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail8
    (if 3 ≤ root193Candidates7.length then search edge 3 [1193, 3923] root193Candidates7 5116 winner else winner) = _
  rw [if_pos (by decide), root193Child7]
def root193Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root193Candidates8_checked : candidates edge 4 1193 4133 root193Tail9 winner.bound = root193Candidates8 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root193Step8 : scan edge (search edge 3) 4 [1193] 1193 root193Tail8 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail9 winner := by
  rw [root193Tail8, scan, if_neg (by decide), root193Candidates8_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail9
    (if 3 ≤ root193Candidates8.length then search edge 3 [1193, 4133] root193Candidates8 5326 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates9 : List Nat := [5009, 8501, 11471, 13487]
set_option maxRecDepth 8192 in
theorem root193Candidates9_checked : candidates edge 4 1193 4931 root193Tail10 winner.bound = root193Candidates9 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root193Child9 : search edge 3 [1193, 4931] root193Candidates9 6124 winner = winner := by decide
theorem root193Step9 : scan edge (search edge 3) 4 [1193] 1193 root193Tail9 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail10 winner := by
  rw [root193Tail9, scan, if_neg (by decide), root193Candidates9_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail10
    (if 3 ≤ root193Candidates9.length then search edge 3 [1193, 4931] root193Candidates9 6124 winner else winner) = _
  rw [if_pos (by decide), root193Child9]
def root193Candidates10 : List Nat := [8501, 13043]
set_option maxRecDepth 8192 in
theorem root193Candidates10_checked : candidates edge 4 1193 5009 root193Tail11 winner.bound = root193Candidates10 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root193Step10 : scan edge (search edge 3) 4 [1193] 1193 root193Tail10 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail11 winner := by
  rw [root193Tail10, scan, if_neg (by decide), root193Candidates10_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail11
    (if 3 ≤ root193Candidates10.length then search edge 3 [1193, 5009] root193Candidates10 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates11 : List Nat := [7829, 15377]
set_option maxRecDepth 8192 in
theorem root193Candidates11_checked : candidates edge 4 1193 5507 root193Tail12 winner.bound = root193Candidates11 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root193Step11 : scan edge (search edge 3) 4 [1193] 1193 root193Tail11 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail12 winner := by
  rw [root193Tail11, scan, if_neg (by decide), root193Candidates11_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail12
    (if 3 ≤ root193Candidates11.length then search edge 3 [1193, 5507] root193Candidates11 6700 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root193Candidates12_checked : candidates edge 4 1193 5519 root193Tail13 winner.bound = root193Candidates12 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root193Step12 : scan edge (search edge 3) 4 [1193] 1193 root193Tail12 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail13 winner := by
  rw [root193Tail12, scan, if_neg (by decide), root193Candidates12_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail13
    (if 3 ≤ root193Candidates12.length then search edge 3 [1193, 5519] root193Candidates12 6712 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates13 : List Nat := [6131, 11471]
set_option maxRecDepth 8192 in
theorem root193Candidates13_checked : candidates edge 4 1193 5801 root193Tail14 winner.bound = root193Candidates13 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root193Step13 : scan edge (search edge 3) 4 [1193] 1193 root193Tail13 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail14 winner := by
  rw [root193Tail13, scan, if_neg (by decide), root193Candidates13_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail14
    (if 3 ≤ root193Candidates13.length then search edge 3 [1193, 5801] root193Candidates13 6994 winner else winner) = _
  rw [if_neg (by decide)]
def root193Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root193Candidates14_checked : candidates edge 4 1193 6131 root193Tail15 winner.bound = root193Candidates14 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root193Step14 : scan edge (search edge 3) 4 [1193] 1193 root193Tail14 winner =
    scan edge (search edge 3) 4 [1193] 1193 root193Tail15 winner := by
  rw [root193Tail14, scan, if_neg (by decide), root193Candidates14_checked]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail15
    (if 3 ≤ root193Candidates14.length then search edge 3 [1193, 6131] root193Candidates14 7324 winner else winner) = _
  rw [if_neg (by decide)]
theorem root193Scan15 : scan edge (search edge 3) 4 [1193] 1193 root193Tail15 winner = winner := by rw [root193Tail15, scan, if_pos (by decide)]
theorem root193Scan14 : scan edge (search edge 3) 4 [1193] 1193 root193Tail14 winner = winner :=
  (root193Step14).trans root193Scan15
theorem root193Scan13 : scan edge (search edge 3) 4 [1193] 1193 root193Tail13 winner = winner :=
  (root193Step13).trans root193Scan14
theorem root193Scan12 : scan edge (search edge 3) 4 [1193] 1193 root193Tail12 winner = winner :=
  (root193Step12).trans root193Scan13
theorem root193Scan11 : scan edge (search edge 3) 4 [1193] 1193 root193Tail11 winner = winner :=
  (root193Step11).trans root193Scan12
theorem root193Scan10 : scan edge (search edge 3) 4 [1193] 1193 root193Tail10 winner = winner :=
  (root193Step10).trans root193Scan11
theorem root193Scan9 : scan edge (search edge 3) 4 [1193] 1193 root193Tail9 winner = winner :=
  (root193Step9).trans root193Scan10
theorem root193Scan8 : scan edge (search edge 3) 4 [1193] 1193 root193Tail8 winner = winner :=
  (root193Step8).trans root193Scan9
theorem root193Scan7 : scan edge (search edge 3) 4 [1193] 1193 root193Tail7 winner = winner :=
  (root193Step7).trans root193Scan8
theorem root193Scan6 : scan edge (search edge 3) 4 [1193] 1193 root193Tail6 winner = winner :=
  (root193Step6).trans root193Scan7
theorem root193Scan5 : scan edge (search edge 3) 4 [1193] 1193 root193Tail5 winner = winner :=
  (root193Step5).trans root193Scan6
theorem root193Scan4 : scan edge (search edge 3) 4 [1193] 1193 root193Tail4 winner = winner :=
  (root193Step4).trans root193Scan5
theorem root193Scan3 : scan edge (search edge 3) 4 [1193] 1193 root193Tail3 winner = winner :=
  (root193Step3).trans root193Scan4
theorem root193Scan2 : scan edge (search edge 3) 4 [1193] 1193 root193Tail2 winner = winner :=
  (root193Step2).trans root193Scan3
theorem root193Scan1 : scan edge (search edge 3) 4 [1193] 1193 root193Tail1 winner = winner :=
  (root193Step1).trans root193Scan2
theorem root193Scan0 : scan edge (search edge 3) 4 [1193] 1193 root193Tail0 winner = winner :=
  (root193Step0).trans root193Scan1
set_option maxRecDepth 4096 in
theorem root193Branch : search edge 4 [1193] root193Candidates 1193 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1193] 1193 root193Tail0 winner = winner
  exact root193Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
