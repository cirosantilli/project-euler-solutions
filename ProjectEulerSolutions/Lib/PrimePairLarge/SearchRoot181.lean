import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root181Candidates : List Nat := Data.row1097
set_option maxRecDepth 8192 in
theorem root181Candidates_checked : candidates edge 5 0 1097 tail182 winner.bound = root181Candidates := by
  apply candidates_eq_row edge 5 0 1097 tail182 root181Candidates winner.bound tail182_ordered
  · decide
  · decide
  · funext q
    rfl
def root181Tail51 : List Nat := []
def root181Tail50 : List Nat := 23813 :: root181Tail51
def root181Tail49 : List Nat := 23297 :: root181Tail50
def root181Tail48 : List Nat := 23189 :: root181Tail49
def root181Tail47 : List Nat := 21611 :: root181Tail48
def root181Tail46 : List Nat := 21149 :: root181Tail47
def root181Tail45 : List Nat := 19391 :: root181Tail46
def root181Tail44 : List Nat := 19301 :: root181Tail45
def root181Tail43 : List Nat := 18041 :: root181Tail44
def root181Tail42 : List Nat := 17519 :: root181Tail43
def root181Tail41 : List Nat := 17327 :: root181Tail42
def root181Tail40 : List Nat := 17117 :: root181Tail41
def root181Tail39 : List Nat := 16421 :: root181Tail40
def root181Tail38 : List Nat := 15797 :: root181Tail39
def root181Tail37 : List Nat := 15401 :: root181Tail38
def root181Tail36 : List Nat := 13859 :: root181Tail37
def root181Tail35 : List Nat := 13697 :: root181Tail36
def root181Tail34 : List Nat := 13523 :: root181Tail35
def root181Tail33 : List Nat := 13469 :: root181Tail34
def root181Tail32 : List Nat := 13337 :: root181Tail33
def root181Tail31 : List Nat := 13259 :: root181Tail32
def root181Tail30 : List Nat := 13229 :: root181Tail31
def root181Tail29 : List Nat := 12377 :: root181Tail30
def root181Tail28 : List Nat := 12263 :: root181Tail29
def root181Tail27 : List Nat := 11717 :: root181Tail28
def root181Tail26 : List Nat := 10859 :: root181Tail27
def root181Tail25 : List Nat := 10709 :: root181Tail26
def root181Tail24 : List Nat := 10487 :: root181Tail25
def root181Tail23 : List Nat := 10193 :: root181Tail24
def root181Tail22 : List Nat := 10067 :: root181Tail23
def root181Tail21 : List Nat := 9281 :: root181Tail22
def root181Tail20 : List Nat := 8609 :: root181Tail21
def root181Tail19 : List Nat := 8087 :: root181Tail20
def root181Tail18 : List Nat := 7727 :: root181Tail19
def root181Tail17 : List Nat := 7559 :: root181Tail18
def root181Tail16 : List Nat := 7517 :: root181Tail17
def root181Tail15 : List Nat := 6473 :: root181Tail16
def root181Tail14 : List Nat := 6323 :: root181Tail15
def root181Tail13 : List Nat := 6113 :: root181Tail14
def root181Tail12 : List Nat := 5651 :: root181Tail13
def root181Tail11 : List Nat := 5087 :: root181Tail12
def root181Tail10 : List Nat := 4751 :: root181Tail11
def root181Tail9 : List Nat := 4637 :: root181Tail10
def root181Tail8 : List Nat := 4493 :: root181Tail9
def root181Tail7 : List Nat := 4409 :: root181Tail8
def root181Tail6 : List Nat := 4283 :: root181Tail7
def root181Tail5 : List Nat := 3593 :: root181Tail6
def root181Tail4 : List Nat := 3533 :: root181Tail5
def root181Tail3 : List Nat := 3251 :: root181Tail4
def root181Tail2 : List Nat := 2999 :: root181Tail3
def root181Tail1 : List Nat := 2393 :: root181Tail2
def root181Tail0 : List Nat := 1637 :: root181Tail1
def root181Candidates0 : List Nat := [23297]
set_option maxRecDepth 8192 in
theorem root181Candidates0_checked : candidates edge 4 1097 1637 root181Tail1 winner.bound = root181Candidates0 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
theorem root181Step0 : scan edge (search edge 3) 4 [1097] 1097 root181Tail0 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail1 winner := by
  rw [root181Tail0, scan, if_neg (by decide), root181Candidates0_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail1
    (if 3 ≤ root181Candidates0.length then search edge 3 [1097, 1637] root181Candidates0 2734 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root181Candidates1_checked : candidates edge 4 1097 2393 root181Tail2 winner.bound = root181Candidates1 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root181Step1 : scan edge (search edge 3) 4 [1097] 1097 root181Tail1 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail2 winner := by
  rw [root181Tail1, scan, if_neg (by decide), root181Candidates1_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail2
    (if 3 ≤ root181Candidates1.length then search edge 3 [1097, 2393] root181Candidates1 3490 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates2 : List Nat := [17117]
set_option maxRecDepth 8192 in
theorem root181Candidates2_checked : candidates edge 4 1097 2999 root181Tail3 winner.bound = root181Candidates2 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root181Step2 : scan edge (search edge 3) 4 [1097] 1097 root181Tail2 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail3 winner := by
  rw [root181Tail2, scan, if_neg (by decide), root181Candidates2_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail3
    (if 3 ≤ root181Candidates2.length then search edge 3 [1097, 2999] root181Candidates2 4096 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates3 : List Nat := [4409]
set_option maxRecDepth 8192 in
theorem root181Candidates3_checked : candidates edge 4 1097 3251 root181Tail4 winner.bound = root181Candidates3 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root181Step3 : scan edge (search edge 3) 4 [1097] 1097 root181Tail3 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail4 winner := by
  rw [root181Tail3, scan, if_neg (by decide), root181Candidates3_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail4
    (if 3 ≤ root181Candidates3.length then search edge 3 [1097, 3251] root181Candidates3 4348 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates4 : List Nat := [3593, 6113, 10709]
set_option maxRecDepth 8192 in
theorem root181Candidates4_checked : candidates edge 4 1097 3533 root181Tail5 winner.bound = root181Candidates4 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root181Child4 : search edge 3 [1097, 3533] root181Candidates4 4630 winner = winner := by decide
theorem root181Step4 : scan edge (search edge 3) 4 [1097] 1097 root181Tail4 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail5 winner := by
  rw [root181Tail4, scan, if_neg (by decide), root181Candidates4_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail5
    (if 3 ≤ root181Candidates4.length then search edge 3 [1097, 3533] root181Candidates4 4630 winner else winner) = _
  rw [if_pos (by decide), root181Child4]
def root181Candidates5 : List Nat := [5087, 10067, 13469, 19391]
set_option maxRecDepth 8192 in
theorem root181Candidates5_checked : candidates edge 4 1097 3593 root181Tail6 winner.bound = root181Candidates5 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root181Child5 : search edge 3 [1097, 3593] root181Candidates5 4690 winner = winner := by decide
theorem root181Step5 : scan edge (search edge 3) 4 [1097] 1097 root181Tail5 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail6 winner := by
  rw [root181Tail5, scan, if_neg (by decide), root181Candidates5_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail6
    (if 3 ≤ root181Candidates5.length then search edge 3 [1097, 3593] root181Candidates5 4690 winner else winner) = _
  rw [if_pos (by decide), root181Child5]
def root181Candidates6 : List Nat := [19391]
set_option maxRecDepth 8192 in
theorem root181Candidates6_checked : candidates edge 4 1097 4283 root181Tail7 winner.bound = root181Candidates6 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root181Step6 : scan edge (search edge 3) 4 [1097] 1097 root181Tail6 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail7 winner := by
  rw [root181Tail6, scan, if_neg (by decide), root181Candidates6_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail7
    (if 3 ≤ root181Candidates6.length then search edge 3 [1097, 4283] root181Candidates6 5380 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates7 : List Nat := [5087, 13469, 13523]
set_option maxRecDepth 8192 in
theorem root181Candidates7_checked : candidates edge 4 1097 4409 root181Tail8 winner.bound = root181Candidates7 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root181Child7 : search edge 3 [1097, 4409] root181Candidates7 5506 winner = winner := by decide
theorem root181Step7 : scan edge (search edge 3) 4 [1097] 1097 root181Tail7 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail8 winner := by
  rw [root181Tail7, scan, if_neg (by decide), root181Candidates7_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail8
    (if 3 ≤ root181Candidates7.length then search edge 3 [1097, 4409] root181Candidates7 5506 winner else winner) = _
  rw [if_pos (by decide), root181Child7]
def root181Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root181Candidates8_checked : candidates edge 4 1097 4493 root181Tail9 winner.bound = root181Candidates8 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root181Step8 : scan edge (search edge 3) 4 [1097] 1097 root181Tail8 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail9 winner := by
  rw [root181Tail8, scan, if_neg (by decide), root181Candidates8_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail9
    (if 3 ≤ root181Candidates8.length then search edge 3 [1097, 4493] root181Candidates8 5590 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates9 : List Nat := [7517]
set_option maxRecDepth 8192 in
theorem root181Candidates9_checked : candidates edge 4 1097 4637 root181Tail10 winner.bound = root181Candidates9 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root181Step9 : scan edge (search edge 3) 4 [1097] 1097 root181Tail9 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail10 winner := by
  rw [root181Tail9, scan, if_neg (by decide), root181Candidates9_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail10
    (if 3 ≤ root181Candidates9.length then search edge 3 [1097, 4637] root181Candidates9 5734 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates10 : List Nat := [6113, 8609, 17519]
set_option maxRecDepth 8192 in
theorem root181Candidates10_checked : candidates edge 4 1097 4751 root181Tail11 winner.bound = root181Candidates10 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root181Child10 : search edge 3 [1097, 4751] root181Candidates10 5848 winner = winner := by decide
theorem root181Step10 : scan edge (search edge 3) 4 [1097] 1097 root181Tail10 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail11 winner := by
  rw [root181Tail10, scan, if_neg (by decide), root181Candidates10_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail11
    (if 3 ≤ root181Candidates10.length then search edge 3 [1097, 4751] root181Candidates10 5848 winner else winner) = _
  rw [if_pos (by decide), root181Child10]
def root181Candidates11 : List Nat := [19301, 19391]
set_option maxRecDepth 8192 in
theorem root181Candidates11_checked : candidates edge 4 1097 5087 root181Tail12 winner.bound = root181Candidates11 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root181Step11 : scan edge (search edge 3) 4 [1097] 1097 root181Tail11 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail12 winner := by
  rw [root181Tail11, scan, if_neg (by decide), root181Candidates11_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail12
    (if 3 ≤ root181Candidates11.length then search edge 3 [1097, 5087] root181Candidates11 6184 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates12 : List Nat := [13259]
set_option maxRecDepth 8192 in
theorem root181Candidates12_checked : candidates edge 4 1097 5651 root181Tail13 winner.bound = root181Candidates12 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root181Step12 : scan edge (search edge 3) 4 [1097] 1097 root181Tail12 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail13 winner := by
  rw [root181Tail12, scan, if_neg (by decide), root181Candidates12_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail13
    (if 3 ≤ root181Candidates12.length then search edge 3 [1097, 5651] root181Candidates12 6748 winner else winner) = _
  rw [if_neg (by decide)]
def root181Candidates13 : List Nat := [17519]
set_option maxRecDepth 8192 in
theorem root181Candidates13_checked : candidates edge 4 1097 6113 root181Tail14 winner.bound = root181Candidates13 := by
  unfold candidates
  have row : edge 6113 = (SortedNatTree.contains FastRows.row6113) := by
    change Data.row6113.contains = (SortedNatTree.contains FastRows.row6113)
    exact FastRows.row6113_checked.symm
  rw [row]
  decide
theorem root181Step13 : scan edge (search edge 3) 4 [1097] 1097 root181Tail13 winner =
    scan edge (search edge 3) 4 [1097] 1097 root181Tail14 winner := by
  rw [root181Tail13, scan, if_neg (by decide), root181Candidates13_checked]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail14
    (if 3 ≤ root181Candidates13.length then search edge 3 [1097, 6113] root181Candidates13 7210 winner else winner) = _
  rw [if_neg (by decide)]
theorem root181Scan14 : scan edge (search edge 3) 4 [1097] 1097 root181Tail14 winner = winner := by rw [root181Tail14, scan, if_pos (by decide)]
theorem root181Scan13 : scan edge (search edge 3) 4 [1097] 1097 root181Tail13 winner = winner :=
  (root181Step13).trans root181Scan14
theorem root181Scan12 : scan edge (search edge 3) 4 [1097] 1097 root181Tail12 winner = winner :=
  (root181Step12).trans root181Scan13
theorem root181Scan11 : scan edge (search edge 3) 4 [1097] 1097 root181Tail11 winner = winner :=
  (root181Step11).trans root181Scan12
theorem root181Scan10 : scan edge (search edge 3) 4 [1097] 1097 root181Tail10 winner = winner :=
  (root181Step10).trans root181Scan11
theorem root181Scan9 : scan edge (search edge 3) 4 [1097] 1097 root181Tail9 winner = winner :=
  (root181Step9).trans root181Scan10
theorem root181Scan8 : scan edge (search edge 3) 4 [1097] 1097 root181Tail8 winner = winner :=
  (root181Step8).trans root181Scan9
theorem root181Scan7 : scan edge (search edge 3) 4 [1097] 1097 root181Tail7 winner = winner :=
  (root181Step7).trans root181Scan8
theorem root181Scan6 : scan edge (search edge 3) 4 [1097] 1097 root181Tail6 winner = winner :=
  (root181Step6).trans root181Scan7
theorem root181Scan5 : scan edge (search edge 3) 4 [1097] 1097 root181Tail5 winner = winner :=
  (root181Step5).trans root181Scan6
theorem root181Scan4 : scan edge (search edge 3) 4 [1097] 1097 root181Tail4 winner = winner :=
  (root181Step4).trans root181Scan5
theorem root181Scan3 : scan edge (search edge 3) 4 [1097] 1097 root181Tail3 winner = winner :=
  (root181Step3).trans root181Scan4
theorem root181Scan2 : scan edge (search edge 3) 4 [1097] 1097 root181Tail2 winner = winner :=
  (root181Step2).trans root181Scan3
theorem root181Scan1 : scan edge (search edge 3) 4 [1097] 1097 root181Tail1 winner = winner :=
  (root181Step1).trans root181Scan2
theorem root181Scan0 : scan edge (search edge 3) 4 [1097] 1097 root181Tail0 winner = winner :=
  (root181Step0).trans root181Scan1
set_option maxRecDepth 4096 in
theorem root181Branch : search edge 4 [1097] root181Candidates 1097 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1097] 1097 root181Tail0 winner = winner
  exact root181Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
