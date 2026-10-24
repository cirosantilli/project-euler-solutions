import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root456Candidates : List Nat := Data.row3253
set_option maxRecDepth 8192 in
theorem root456Candidates_checked : candidates edge 5 0 3253 tail457 winner.bound = root456Candidates := by
  apply candidates_eq_row edge 5 0 3253 tail457 root456Candidates winner.bound tail457_ordered
  · decide
  · decide
  · funext q
    rfl
def root456Tail43 : List Nat := []
def root456Tail42 : List Nat := 22027 :: root456Tail43
def root456Tail41 : List Nat := 20719 :: root456Tail42
def root456Tail40 : List Nat := 20389 :: root456Tail41
def root456Tail39 : List Nat := 20161 :: root456Tail40
def root456Tail38 : List Nat := 19717 :: root456Tail39
def root456Tail37 : List Nat := 18691 :: root456Tail38
def root456Tail36 : List Nat := 18493 :: root456Tail37
def root456Tail35 : List Nat := 18289 :: root456Tail36
def root456Tail34 : List Nat := 17257 :: root456Tail35
def root456Tail33 : List Nat := 16603 :: root456Tail34
def root456Tail32 : List Nat := 16567 :: root456Tail33
def root456Tail31 : List Nat := 15121 :: root456Tail32
def root456Tail30 : List Nat := 14503 :: root456Tail31
def root456Tail29 : List Nat := 14293 :: root456Tail30
def root456Tail28 : List Nat := 13249 :: root456Tail29
def root456Tail27 : List Nat := 13033 :: root456Tail28
def root456Tail26 : List Nat := 12577 :: root456Tail27
def root456Tail25 : List Nat := 12073 :: root456Tail26
def root456Tail24 : List Nat := 12049 :: root456Tail25
def root456Tail23 : List Nat := 10501 :: root456Tail24
def root456Tail22 : List Nat := 9973 :: root456Tail23
def root456Tail21 : List Nat := 9967 :: root456Tail22
def root456Tail20 : List Nat := 9817 :: root456Tail21
def root456Tail19 : List Nat := 8893 :: root456Tail20
def root456Tail18 : List Nat := 8641 :: root456Tail19
def root456Tail17 : List Nat := 8263 :: root456Tail18
def root456Tail16 : List Nat := 7927 :: root456Tail17
def root456Tail15 : List Nat := 7873 :: root456Tail16
def root456Tail14 : List Nat := 7717 :: root456Tail15
def root456Tail13 : List Nat := 6571 :: root456Tail14
def root456Tail12 : List Nat := 6067 :: root456Tail13
def root456Tail11 : List Nat := 6043 :: root456Tail12
def root456Tail10 : List Nat := 5851 :: root456Tail11
def root456Tail9 : List Nat := 5557 :: root456Tail10
def root456Tail8 : List Nat := 5479 :: root456Tail9
def root456Tail7 : List Nat := 5197 :: root456Tail8
def root456Tail6 : List Nat := 4357 :: root456Tail7
def root456Tail5 : List Nat := 4339 :: root456Tail6
def root456Tail4 : List Nat := 4231 :: root456Tail5
def root456Tail3 : List Nat := 4219 :: root456Tail4
def root456Tail2 : List Nat := 3607 :: root456Tail3
def root456Tail1 : List Nat := 3547 :: root456Tail2
def root456Tail0 : List Nat := 3517 :: root456Tail1
def root456Candidates0 : List Nat := [7717, 8263]
set_option maxRecDepth 8192 in
theorem root456Candidates0_checked : candidates edge 4 3253 3517 root456Tail1 winner.bound = root456Candidates0 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root456Step0 : scan edge (search edge 3) 4 [3253] 3253 root456Tail0 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail1 winner := by
  rw [root456Tail0, scan, if_neg (by decide), root456Candidates0_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail1
    (if 3 ≤ root456Candidates0.length then search edge 3 [3253, 3517] root456Candidates0 6770 winner else winner) = _
  rw [if_neg (by decide)]
def root456Candidates1 : List Nat := [4357, 9817]
set_option maxRecDepth 8192 in
theorem root456Candidates1_checked : candidates edge 4 3253 3547 root456Tail2 winner.bound = root456Candidates1 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root456Step1 : scan edge (search edge 3) 4 [3253] 3253 root456Tail1 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail2 winner := by
  rw [root456Tail1, scan, if_neg (by decide), root456Candidates1_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail2
    (if 3 ≤ root456Candidates1.length then search edge 3 [3253, 3547] root456Candidates1 6800 winner else winner) = _
  rw [if_neg (by decide)]
def root456Candidates2 : List Nat := [6571, 13033, 18493]
set_option maxRecDepth 8192 in
theorem root456Candidates2_checked : candidates edge 4 3253 3607 root456Tail3 winner.bound = root456Candidates2 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root456Child2 : search edge 3 [3253, 3607] root456Candidates2 6860 winner = winner := by decide
theorem root456Step2 : scan edge (search edge 3) 4 [3253] 3253 root456Tail2 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail3 winner := by
  rw [root456Tail2, scan, if_neg (by decide), root456Candidates2_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail3
    (if 3 ≤ root456Candidates2.length then search edge 3 [3253, 3607] root456Candidates2 6860 winner else winner) = _
  rw [if_pos (by decide), root456Child2]
def root456Candidates3 : List Nat := [18493]
set_option maxRecDepth 8192 in
theorem root456Candidates3_checked : candidates edge 4 3253 4219 root456Tail4 winner.bound = root456Candidates3 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
theorem root456Step3 : scan edge (search edge 3) 4 [3253] 3253 root456Tail3 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail4 winner := by
  rw [root456Tail3, scan, if_neg (by decide), root456Candidates3_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail4
    (if 3 ≤ root456Candidates3.length then search edge 3 [3253, 4219] root456Candidates3 7472 winner else winner) = _
  rw [if_neg (by decide)]
def root456Candidates4 : List Nat := [4339, 5197]
set_option maxRecDepth 8192 in
theorem root456Candidates4_checked : candidates edge 4 3253 4231 root456Tail5 winner.bound = root456Candidates4 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root456Step4 : scan edge (search edge 3) 4 [3253] 3253 root456Tail4 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail5 winner := by
  rw [root456Tail4, scan, if_neg (by decide), root456Candidates4_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail5
    (if 3 ≤ root456Candidates4.length then search edge 3 [3253, 4231] root456Candidates4 7484 winner else winner) = _
  rw [if_neg (by decide)]
def root456Candidates5 : List Nat := [7927]
set_option maxRecDepth 8192 in
theorem root456Candidates5_checked : candidates edge 4 3253 4339 root456Tail6 winner.bound = root456Candidates5 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root456Step5 : scan edge (search edge 3) 4 [3253] 3253 root456Tail5 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail6 winner := by
  rw [root456Tail5, scan, if_neg (by decide), root456Candidates5_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail6
    (if 3 ≤ root456Candidates5.length then search edge 3 [3253, 4339] root456Candidates5 7592 winner else winner) = _
  rw [if_neg (by decide)]
def root456Candidates6 : List Nat := [5479, 8893, 18289]
set_option maxRecDepth 8192 in
theorem root456Candidates6_checked : candidates edge 4 3253 4357 root456Tail7 winner.bound = root456Candidates6 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root456Child6 : search edge 3 [3253, 4357] root456Candidates6 7610 winner = winner := by decide
theorem root456Step6 : scan edge (search edge 3) 4 [3253] 3253 root456Tail6 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail7 winner := by
  rw [root456Tail6, scan, if_neg (by decide), root456Candidates6_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail7
    (if 3 ≤ root456Candidates6.length then search edge 3 [3253, 4357] root456Candidates6 7610 winner else winner) = _
  rw [if_pos (by decide), root456Child6]
def root456Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root456Candidates7_checked : candidates edge 4 3253 5197 root456Tail8 winner.bound = root456Candidates7 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root456Step7 : scan edge (search edge 3) 4 [3253] 3253 root456Tail7 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail8 winner := by
  rw [root456Tail7, scan, if_neg (by decide), root456Candidates7_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail8
    (if 3 ≤ root456Candidates7.length then search edge 3 [3253, 5197] root456Candidates7 8450 winner else winner) = _
  rw [if_neg (by decide)]
def root456Candidates8 : List Nat := [7927, 9817, 9973, 10501]
set_option maxRecDepth 8192 in
theorem root456Candidates8_checked : candidates edge 4 3253 5479 root456Tail9 winner.bound = root456Candidates8 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root456Child8 : search edge 3 [3253, 5479] root456Candidates8 8732 winner = winner := by decide
theorem root456Step8 : scan edge (search edge 3) 4 [3253] 3253 root456Tail8 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail9 winner := by
  rw [root456Tail8, scan, if_neg (by decide), root456Candidates8_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail9
    (if 3 ≤ root456Candidates8.length then search edge 3 [3253, 5479] root456Candidates8 8732 winner else winner) = _
  rw [if_pos (by decide), root456Child8]
def root456Candidates9 : List Nat := [6067]
set_option maxRecDepth 8192 in
theorem root456Candidates9_checked : candidates edge 4 3253 5557 root456Tail10 winner.bound = root456Candidates9 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root456Step9 : scan edge (search edge 3) 4 [3253] 3253 root456Tail9 winner =
    scan edge (search edge 3) 4 [3253] 3253 root456Tail10 winner := by
  rw [root456Tail9, scan, if_neg (by decide), root456Candidates9_checked]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail10
    (if 3 ≤ root456Candidates9.length then search edge 3 [3253, 5557] root456Candidates9 8810 winner else winner) = _
  rw [if_neg (by decide)]
theorem root456Scan10 : scan edge (search edge 3) 4 [3253] 3253 root456Tail10 winner = winner := by rw [root456Tail10, scan, if_pos (by decide)]
theorem root456Scan9 : scan edge (search edge 3) 4 [3253] 3253 root456Tail9 winner = winner :=
  (root456Step9).trans root456Scan10
theorem root456Scan8 : scan edge (search edge 3) 4 [3253] 3253 root456Tail8 winner = winner :=
  (root456Step8).trans root456Scan9
theorem root456Scan7 : scan edge (search edge 3) 4 [3253] 3253 root456Tail7 winner = winner :=
  (root456Step7).trans root456Scan8
theorem root456Scan6 : scan edge (search edge 3) 4 [3253] 3253 root456Tail6 winner = winner :=
  (root456Step6).trans root456Scan7
theorem root456Scan5 : scan edge (search edge 3) 4 [3253] 3253 root456Tail5 winner = winner :=
  (root456Step5).trans root456Scan6
theorem root456Scan4 : scan edge (search edge 3) 4 [3253] 3253 root456Tail4 winner = winner :=
  (root456Step4).trans root456Scan5
theorem root456Scan3 : scan edge (search edge 3) 4 [3253] 3253 root456Tail3 winner = winner :=
  (root456Step3).trans root456Scan4
theorem root456Scan2 : scan edge (search edge 3) 4 [3253] 3253 root456Tail2 winner = winner :=
  (root456Step2).trans root456Scan3
theorem root456Scan1 : scan edge (search edge 3) 4 [3253] 3253 root456Tail1 winner = winner :=
  (root456Step1).trans root456Scan2
theorem root456Scan0 : scan edge (search edge 3) 4 [3253] 3253 root456Tail0 winner = winner :=
  (root456Step0).trans root456Scan1
set_option maxRecDepth 4096 in
theorem root456Branch : search edge 4 [3253] root456Candidates 3253 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3253] 3253 root456Tail0 winner = winner
  exact root456Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
