import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root270Candidates : List Nat := Data.row1753
set_option maxRecDepth 8192 in
theorem root270Candidates_checked : candidates edge 5 0 1753 tail271 winner.bound = root270Candidates := by
  apply candidates_eq_row edge 5 0 1753 tail271 root270Candidates winner.bound tail271_ordered
  · decide
  · decide
  · funext q
    rfl
def root270Tail56 : List Nat := []
def root270Tail55 : List Nat := 23761 :: root270Tail56
def root270Tail54 : List Nat := 23209 :: root270Tail55
def root270Tail53 : List Nat := 21859 :: root270Tail54
def root270Tail52 : List Nat := 21499 :: root270Tail53
def root270Tail51 : List Nat := 21313 :: root270Tail52
def root270Tail50 : List Nat := 21121 :: root270Tail51
def root270Tail49 : List Nat := 20947 :: root270Tail50
def root270Tail48 : List Nat := 19717 :: root270Tail49
def root270Tail47 : List Nat := 18133 :: root270Tail48
def root270Tail46 : List Nat := 17881 :: root270Tail47
def root270Tail45 : List Nat := 17491 :: root270Tail46
def root270Tail44 : List Nat := 17341 :: root270Tail45
def root270Tail43 : List Nat := 15277 :: root270Tail44
def root270Tail42 : List Nat := 14389 :: root270Tail43
def root270Tail41 : List Nat := 14107 :: root270Tail42
def root270Tail40 : List Nat := 13309 :: root270Tail41
def root270Tail39 : List Nat := 13267 :: root270Tail40
def root270Tail38 : List Nat := 12457 :: root270Tail39
def root270Tail37 : List Nat := 11839 :: root270Tail38
def root270Tail36 : List Nat := 11527 :: root270Tail37
def root270Tail35 : List Nat := 11113 :: root270Tail36
def root270Tail34 : List Nat := 10789 :: root270Tail35
def root270Tail33 : List Nat := 10657 :: root270Tail34
def root270Tail32 : List Nat := 10159 :: root270Tail33
def root270Tail31 : List Nat := 9433 :: root270Tail32
def root270Tail30 : List Nat := 9091 :: root270Tail31
def root270Tail29 : List Nat := 9007 :: root270Tail30
def root270Tail28 : List Nat := 8929 :: root270Tail29
def root270Tail27 : List Nat := 8893 :: root270Tail28
def root270Tail26 : List Nat := 8887 :: root270Tail27
def root270Tail25 : List Nat := 8161 :: root270Tail26
def root270Tail24 : List Nat := 8089 :: root270Tail25
def root270Tail23 : List Nat := 7789 :: root270Tail24
def root270Tail22 : List Nat := 7669 :: root270Tail23
def root270Tail21 : List Nat := 7549 :: root270Tail22
def root270Tail20 : List Nat := 7411 :: root270Tail21
def root270Tail19 : List Nat := 7321 :: root270Tail20
def root270Tail18 : List Nat := 7237 :: root270Tail19
def root270Tail17 : List Nat := 6907 :: root270Tail18
def root270Tail16 : List Nat := 6823 :: root270Tail17
def root270Tail15 : List Nat := 6691 :: root270Tail16
def root270Tail14 : List Nat := 6199 :: root270Tail15
def root270Tail13 : List Nat := 5851 :: root270Tail14
def root270Tail12 : List Nat := 5827 :: root270Tail13
def root270Tail11 : List Nat := 5281 :: root270Tail12
def root270Tail10 : List Nat := 5197 :: root270Tail11
def root270Tail9 : List Nat := 4987 :: root270Tail10
def root270Tail8 : List Nat := 4903 :: root270Tail9
def root270Tail7 : List Nat := 4567 :: root270Tail8
def root270Tail6 : List Nat := 3847 :: root270Tail7
def root270Tail5 : List Nat := 3709 :: root270Tail6
def root270Tail4 : List Nat := 3673 :: root270Tail5
def root270Tail3 : List Nat := 3643 :: root270Tail4
def root270Tail2 : List Nat := 3547 :: root270Tail3
def root270Tail1 : List Nat := 1951 :: root270Tail2
def root270Tail0 : List Nat := 1873 :: root270Tail1
def root270Candidates0 : List Nat := [9433, 10657, 14107]
set_option maxRecDepth 8192 in
theorem root270Candidates0_checked : candidates edge 4 1753 1873 root270Tail1 winner.bound = root270Candidates0 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child0 : search edge 3 [1753, 1873] root270Candidates0 3626 winner = winner := by decide
theorem root270Step0 : scan edge (search edge 3) 4 [1753] 1753 root270Tail0 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail1 winner := by
  rw [root270Tail0, scan, if_neg (by decide), root270Candidates0_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail1
    (if 3 ≤ root270Candidates0.length then search edge 3 [1753, 1873] root270Candidates0 3626 winner else winner) = _
  rw [if_pos (by decide), root270Child0]
def root270Candidates1 : List Nat := [3547, 3643, 6823]
set_option maxRecDepth 8192 in
theorem root270Candidates1_checked : candidates edge 4 1753 1951 root270Tail2 winner.bound = root270Candidates1 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child1 : search edge 3 [1753, 1951] root270Candidates1 3704 winner = winner := by decide
theorem root270Step1 : scan edge (search edge 3) 4 [1753] 1753 root270Tail1 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail2 winner := by
  rw [root270Tail1, scan, if_neg (by decide), root270Candidates1_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail2
    (if 3 ≤ root270Candidates1.length then search edge 3 [1753, 1951] root270Candidates1 3704 winner else winner) = _
  rw [if_pos (by decide), root270Child1]
def root270Candidates2 : List Nat := [3643, 5827, 11839]
set_option maxRecDepth 8192 in
theorem root270Candidates2_checked : candidates edge 4 1753 3547 root270Tail3 winner.bound = root270Candidates2 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child2 : search edge 3 [1753, 3547] root270Candidates2 5300 winner = winner := by decide
theorem root270Step2 : scan edge (search edge 3) 4 [1753] 1753 root270Tail2 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail3 winner := by
  rw [root270Tail2, scan, if_neg (by decide), root270Candidates2_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail3
    (if 3 ≤ root270Candidates2.length then search edge 3 [1753, 3547] root270Candidates2 5300 winner else winner) = _
  rw [if_pos (by decide), root270Child2]
def root270Candidates3 : List Nat := [4987, 6199, 8929, 17881]
set_option maxRecDepth 8192 in
theorem root270Candidates3_checked : candidates edge 4 1753 3643 root270Tail4 winner.bound = root270Candidates3 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child3 : search edge 3 [1753, 3643] root270Candidates3 5396 winner = winner := by decide
theorem root270Step3 : scan edge (search edge 3) 4 [1753] 1753 root270Tail3 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail4 winner := by
  rw [root270Tail3, scan, if_neg (by decide), root270Candidates3_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail4
    (if 3 ≤ root270Candidates3.length then search edge 3 [1753, 3643] root270Candidates3 5396 winner else winner) = _
  rw [if_pos (by decide), root270Child3]
def root270Candidates4 : List Nat := [6691]
set_option maxRecDepth 8192 in
theorem root270Candidates4_checked : candidates edge 4 1753 3673 root270Tail5 winner.bound = root270Candidates4 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root270Step4 : scan edge (search edge 3) 4 [1753] 1753 root270Tail4 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail5 winner := by
  rw [root270Tail4, scan, if_neg (by decide), root270Candidates4_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail5
    (if 3 ≤ root270Candidates4.length then search edge 3 [1753, 3673] root270Candidates4 5426 winner else winner) = _
  rw [if_neg (by decide)]
def root270Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root270Candidates5_checked : candidates edge 4 1753 3709 root270Tail6 winner.bound = root270Candidates5 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root270Step5 : scan edge (search edge 3) 4 [1753] 1753 root270Tail5 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail6 winner := by
  rw [root270Tail5, scan, if_neg (by decide), root270Candidates5_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail6
    (if 3 ≤ root270Candidates5.length then search edge 3 [1753, 3709] root270Candidates5 5462 winner else winner) = _
  rw [if_neg (by decide)]
def root270Candidates6 : List Nat := [5197, 8887, 8929, 10789, 13267, 15277]
set_option maxRecDepth 8192 in
theorem root270Candidates6_checked : candidates edge 4 1753 3847 root270Tail7 winner.bound = root270Candidates6 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child6 : search edge 3 [1753, 3847] root270Candidates6 5600 winner = winner := by decide
theorem root270Step6 : scan edge (search edge 3) 4 [1753] 1753 root270Tail6 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail7 winner := by
  rw [root270Tail6, scan, if_neg (by decide), root270Candidates6_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail7
    (if 3 ≤ root270Candidates6.length then search edge 3 [1753, 3847] root270Candidates6 5600 winner else winner) = _
  rw [if_pos (by decide), root270Child6]
def root270Candidates7 : List Nat := [7789, 9091]
set_option maxRecDepth 8192 in
theorem root270Candidates7_checked : candidates edge 4 1753 4567 root270Tail8 winner.bound = root270Candidates7 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root270Step7 : scan edge (search edge 3) 4 [1753] 1753 root270Tail7 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail8 winner := by
  rw [root270Tail7, scan, if_neg (by decide), root270Candidates7_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail8
    (if 3 ≤ root270Candidates7.length then search edge 3 [1753, 4567] root270Candidates7 6320 winner else winner) = _
  rw [if_neg (by decide)]
def root270Candidates8 : List Nat := [7789, 8893, 9007, 17341]
set_option maxRecDepth 8192 in
theorem root270Candidates8_checked : candidates edge 4 1753 4903 root270Tail9 winner.bound = root270Candidates8 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child8 : search edge 3 [1753, 4903] root270Candidates8 6656 winner = winner := by decide
theorem root270Step8 : scan edge (search edge 3) 4 [1753] 1753 root270Tail8 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail9 winner := by
  rw [root270Tail8, scan, if_neg (by decide), root270Candidates8_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail9
    (if 3 ≤ root270Candidates8.length then search edge 3 [1753, 4903] root270Candidates8 6656 winner else winner) = _
  rw [if_pos (by decide), root270Child8]
def root270Candidates9 : List Nat := [5197]
set_option maxRecDepth 8192 in
theorem root270Candidates9_checked : candidates edge 4 1753 4987 root270Tail10 winner.bound = root270Candidates9 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root270Step9 : scan edge (search edge 3) 4 [1753] 1753 root270Tail9 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail10 winner := by
  rw [root270Tail9, scan, if_neg (by decide), root270Candidates9_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail10
    (if 3 ≤ root270Candidates9.length then search edge 3 [1753, 4987] root270Candidates9 6740 winner else winner) = _
  rw [if_neg (by decide)]
def root270Candidates10 : List Nat := [8929, 10657, 15277]
set_option maxRecDepth 8192 in
theorem root270Candidates10_checked : candidates edge 4 1753 5197 root270Tail11 winner.bound = root270Candidates10 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root270Child10 : search edge 3 [1753, 5197] root270Candidates10 6950 winner = winner := by decide
theorem root270Step10 : scan edge (search edge 3) 4 [1753] 1753 root270Tail10 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail11 winner := by
  rw [root270Tail10, scan, if_neg (by decide), root270Candidates10_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail11
    (if 3 ≤ root270Candidates10.length then search edge 3 [1753, 5197] root270Candidates10 6950 winner else winner) = _
  rw [if_pos (by decide), root270Child10]
def root270Candidates11 : List Nat := [9433, 14107]
set_option maxRecDepth 8192 in
theorem root270Candidates11_checked : candidates edge 4 1753 5281 root270Tail12 winner.bound = root270Candidates11 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root270Step11 : scan edge (search edge 3) 4 [1753] 1753 root270Tail11 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail12 winner := by
  rw [root270Tail11, scan, if_neg (by decide), root270Candidates11_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail12
    (if 3 ≤ root270Candidates11.length then search edge 3 [1753, 5281] root270Candidates11 7034 winner else winner) = _
  rw [if_neg (by decide)]
def root270Candidates12 : List Nat := [8089]
set_option maxRecDepth 8192 in
theorem root270Candidates12_checked : candidates edge 4 1753 5827 root270Tail13 winner.bound = root270Candidates12 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root270Step12 : scan edge (search edge 3) 4 [1753] 1753 root270Tail12 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail13 winner := by
  rw [root270Tail12, scan, if_neg (by decide), root270Candidates12_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail13
    (if 3 ≤ root270Candidates12.length then search edge 3 [1753, 5827] root270Candidates12 7580 winner else winner) = _
  rw [if_neg (by decide)]
def root270Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root270Candidates13_checked : candidates edge 4 1753 5851 root270Tail14 winner.bound = root270Candidates13 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root270Step13 : scan edge (search edge 3) 4 [1753] 1753 root270Tail13 winner =
    scan edge (search edge 3) 4 [1753] 1753 root270Tail14 winner := by
  rw [root270Tail13, scan, if_neg (by decide), root270Candidates13_checked]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail14
    (if 3 ≤ root270Candidates13.length then search edge 3 [1753, 5851] root270Candidates13 7604 winner else winner) = _
  rw [if_neg (by decide)]
theorem root270Scan14 : scan edge (search edge 3) 4 [1753] 1753 root270Tail14 winner = winner := by rw [root270Tail14, scan, if_pos (by decide)]
theorem root270Scan13 : scan edge (search edge 3) 4 [1753] 1753 root270Tail13 winner = winner :=
  (root270Step13).trans root270Scan14
theorem root270Scan12 : scan edge (search edge 3) 4 [1753] 1753 root270Tail12 winner = winner :=
  (root270Step12).trans root270Scan13
theorem root270Scan11 : scan edge (search edge 3) 4 [1753] 1753 root270Tail11 winner = winner :=
  (root270Step11).trans root270Scan12
theorem root270Scan10 : scan edge (search edge 3) 4 [1753] 1753 root270Tail10 winner = winner :=
  (root270Step10).trans root270Scan11
theorem root270Scan9 : scan edge (search edge 3) 4 [1753] 1753 root270Tail9 winner = winner :=
  (root270Step9).trans root270Scan10
theorem root270Scan8 : scan edge (search edge 3) 4 [1753] 1753 root270Tail8 winner = winner :=
  (root270Step8).trans root270Scan9
theorem root270Scan7 : scan edge (search edge 3) 4 [1753] 1753 root270Tail7 winner = winner :=
  (root270Step7).trans root270Scan8
theorem root270Scan6 : scan edge (search edge 3) 4 [1753] 1753 root270Tail6 winner = winner :=
  (root270Step6).trans root270Scan7
theorem root270Scan5 : scan edge (search edge 3) 4 [1753] 1753 root270Tail5 winner = winner :=
  (root270Step5).trans root270Scan6
theorem root270Scan4 : scan edge (search edge 3) 4 [1753] 1753 root270Tail4 winner = winner :=
  (root270Step4).trans root270Scan5
theorem root270Scan3 : scan edge (search edge 3) 4 [1753] 1753 root270Tail3 winner = winner :=
  (root270Step3).trans root270Scan4
theorem root270Scan2 : scan edge (search edge 3) 4 [1753] 1753 root270Tail2 winner = winner :=
  (root270Step2).trans root270Scan3
theorem root270Scan1 : scan edge (search edge 3) 4 [1753] 1753 root270Tail1 winner = winner :=
  (root270Step1).trans root270Scan2
theorem root270Scan0 : scan edge (search edge 3) 4 [1753] 1753 root270Tail0 winner = winner :=
  (root270Step0).trans root270Scan1
set_option maxRecDepth 4096 in
theorem root270Branch : search edge 4 [1753] root270Candidates 1753 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1753] 1753 root270Tail0 winner = winner
  exact root270Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
