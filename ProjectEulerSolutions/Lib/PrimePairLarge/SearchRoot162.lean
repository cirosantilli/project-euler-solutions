import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root162Candidates : List Nat := Data.row977
set_option maxRecDepth 8192 in
theorem root162Candidates_checked : candidates edge 5 0 977 tail163 winner.bound = root162Candidates := by
  apply candidates_eq_row edge 5 0 977 tail163 root162Candidates winner.bound tail163_ordered
  · decide
  · decide
  · funext q
    rfl
def root162Tail58 : List Nat := []
def root162Tail57 : List Nat := 24281 :: root162Tail58
def root162Tail56 : List Nat := 23957 :: root162Tail57
def root162Tail55 : List Nat := 23609 :: root162Tail56
def root162Tail54 : List Nat := 23063 :: root162Tail55
def root162Tail53 : List Nat := 22787 :: root162Tail54
def root162Tail52 : List Nat := 22433 :: root162Tail53
def root162Tail51 : List Nat := 22133 :: root162Tail52
def root162Tail50 : List Nat := 21713 :: root162Tail51
def root162Tail49 : List Nat := 21467 :: root162Tail50
def root162Tail48 : List Nat := 21191 :: root162Tail49
def root162Tail47 : List Nat := 21059 :: root162Tail48
def root162Tail46 : List Nat := 20333 :: root162Tail47
def root162Tail45 : List Nat := 20177 :: root162Tail46
def root162Tail44 : List Nat := 19949 :: root162Tail45
def root162Tail43 : List Nat := 19913 :: root162Tail44
def root162Tail42 : List Nat := 19301 :: root162Tail43
def root162Tail41 : List Nat := 18899 :: root162Tail42
def root162Tail40 : List Nat := 17909 :: root162Tail41
def root162Tail39 : List Nat := 17573 :: root162Tail40
def root162Tail38 : List Nat := 16763 :: root162Tail39
def root162Tail37 : List Nat := 15287 :: root162Tail38
def root162Tail36 : List Nat := 14831 :: root162Tail37
def root162Tail35 : List Nat := 14423 :: root162Tail36
def root162Tail34 : List Nat := 14411 :: root162Tail35
def root162Tail33 : List Nat := 14087 :: root162Tail34
def root162Tail32 : List Nat := 13313 :: root162Tail33
def root162Tail31 : List Nat := 13037 :: root162Tail32
def root162Tail30 : List Nat := 12983 :: root162Tail31
def root162Tail29 : List Nat := 12689 :: root162Tail30
def root162Tail28 : List Nat := 12347 :: root162Tail29
def root162Tail27 : List Nat := 12161 :: root162Tail28
def root162Tail26 : List Nat := 12011 :: root162Tail27
def root162Tail25 : List Nat := 11717 :: root162Tail26
def root162Tail24 : List Nat := 11399 :: root162Tail25
def root162Tail23 : List Nat := 11027 :: root162Tail24
def root162Tail22 : List Nat := 10607 :: root162Tail23
def root162Tail21 : List Nat := 10079 :: root162Tail22
def root162Tail20 : List Nat := 9941 :: root162Tail21
def root162Tail19 : List Nat := 9311 :: root162Tail20
def root162Tail18 : List Nat := 9239 :: root162Tail19
def root162Tail17 : List Nat := 8819 :: root162Tail18
def root162Tail16 : List Nat := 8753 :: root162Tail17
def root162Tail15 : List Nat := 8297 :: root162Tail16
def root162Tail14 : List Nat := 7853 :: root162Tail15
def root162Tail13 : List Nat := 5039 :: root162Tail14
def root162Tail12 : List Nat := 5021 :: root162Tail13
def root162Tail11 : List Nat := 4049 :: root162Tail12
def root162Tail10 : List Nat := 3851 :: root162Tail11
def root162Tail9 : List Nat := 3767 :: root162Tail10
def root162Tail8 : List Nat := 3593 :: root162Tail9
def root162Tail7 : List Nat := 3557 :: root162Tail8
def root162Tail6 : List Nat := 2753 :: root162Tail7
def root162Tail5 : List Nat := 2141 :: root162Tail6
def root162Tail4 : List Nat := 2129 :: root162Tail5
def root162Tail3 : List Nat := 2039 :: root162Tail4
def root162Tail2 : List Nat := 1409 :: root162Tail3
def root162Tail1 : List Nat := 1259 :: root162Tail2
def root162Tail0 : List Nat := 1217 :: root162Tail1
def root162Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root162Candidates0_checked : candidates edge 4 977 1217 root162Tail1 winner.bound = root162Candidates0 := by
  unfold candidates
  have row : edge 1217 = (SortedNatTree.contains FastRows.row1217) := by
    change Data.row1217.contains = (SortedNatTree.contains FastRows.row1217)
    exact FastRows.row1217_checked.symm
  rw [row]
  decide
theorem root162Step0 : scan edge (search edge 3) 4 [977] 977 root162Tail0 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail1 winner := by
  rw [root162Tail0, scan, if_neg (by decide), root162Candidates0_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail1
    (if 3 ≤ root162Candidates0.length then search edge 3 [977, 1217] root162Candidates0 2194 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates1 : List Nat := [5039, 12011, 20177]
set_option maxRecDepth 8192 in
theorem root162Candidates1_checked : candidates edge 4 977 1259 root162Tail2 winner.bound = root162Candidates1 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root162Child1 : search edge 3 [977, 1259] root162Candidates1 2236 winner = winner := by decide
theorem root162Step1 : scan edge (search edge 3) 4 [977] 977 root162Tail1 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail2 winner := by
  rw [root162Tail1, scan, if_neg (by decide), root162Candidates1_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail2
    (if 3 ≤ root162Candidates1.length then search edge 3 [977, 1259] root162Candidates1 2236 winner else winner) = _
  rw [if_pos (by decide), root162Child1]
def root162Candidates2 : List Nat := [23609]
set_option maxRecDepth 8192 in
theorem root162Candidates2_checked : candidates edge 4 977 1409 root162Tail3 winner.bound = root162Candidates2 := by
  unfold candidates
  have row : edge 1409 = (SortedNatTree.contains FastRows.row1409) := by
    change Data.row1409.contains = (SortedNatTree.contains FastRows.row1409)
    exact FastRows.row1409_checked.symm
  rw [row]
  decide
theorem root162Step2 : scan edge (search edge 3) 4 [977] 977 root162Tail2 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail3 winner := by
  rw [root162Tail2, scan, if_neg (by decide), root162Candidates2_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail3
    (if 3 ≤ root162Candidates2.length then search edge 3 [977, 1409] root162Candidates2 2386 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates3 : List Nat := [3767, 10079]
set_option maxRecDepth 8192 in
theorem root162Candidates3_checked : candidates edge 4 977 2039 root162Tail4 winner.bound = root162Candidates3 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
theorem root162Step3 : scan edge (search edge 3) 4 [977] 977 root162Tail3 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail4 winner := by
  rw [root162Tail3, scan, if_neg (by decide), root162Candidates3_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail4
    (if 3 ≤ root162Candidates3.length then search edge 3 [977, 2039] root162Candidates3 3016 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates4 : List Nat := [14087]
set_option maxRecDepth 8192 in
theorem root162Candidates4_checked : candidates edge 4 977 2129 root162Tail5 winner.bound = root162Candidates4 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root162Step4 : scan edge (search edge 3) 4 [977] 977 root162Tail4 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail5 winner := by
  rw [root162Tail4, scan, if_neg (by decide), root162Candidates4_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail5
    (if 3 ≤ root162Candidates4.length then search edge 3 [977, 2129] root162Candidates4 3106 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates5 : List Nat := [4049]
set_option maxRecDepth 8192 in
theorem root162Candidates5_checked : candidates edge 4 977 2141 root162Tail6 winner.bound = root162Candidates5 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root162Step5 : scan edge (search edge 3) 4 [977] 977 root162Tail5 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail6 winner := by
  rw [root162Tail5, scan, if_neg (by decide), root162Candidates5_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail6
    (if 3 ≤ root162Candidates5.length then search edge 3 [977, 2141] root162Candidates5 3118 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates6 : List Nat := [12347]
set_option maxRecDepth 8192 in
theorem root162Candidates6_checked : candidates edge 4 977 2753 root162Tail7 winner.bound = root162Candidates6 := by
  unfold candidates
  have row : edge 2753 = (SortedNatTree.contains FastRows.row2753) := by
    change Data.row2753.contains = (SortedNatTree.contains FastRows.row2753)
    exact FastRows.row2753_checked.symm
  rw [row]
  decide
theorem root162Step6 : scan edge (search edge 3) 4 [977] 977 root162Tail6 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail7 winner := by
  rw [root162Tail6, scan, if_neg (by decide), root162Candidates6_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail7
    (if 3 ≤ root162Candidates6.length then search edge 3 [977, 2753] root162Candidates6 3730 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates7 : List Nat := [5021, 11717, 13037]
set_option maxRecDepth 8192 in
theorem root162Candidates7_checked : candidates edge 4 977 3557 root162Tail8 winner.bound = root162Candidates7 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root162Child7 : search edge 3 [977, 3557] root162Candidates7 4534 winner = winner := by decide
theorem root162Step7 : scan edge (search edge 3) 4 [977] 977 root162Tail7 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail8 winner := by
  rw [root162Tail7, scan, if_neg (by decide), root162Candidates7_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail8
    (if 3 ≤ root162Candidates7.length then search edge 3 [977, 3557] root162Candidates7 4534 winner else winner) = _
  rw [if_pos (by decide), root162Child7]
def root162Candidates8 : List Nat := [5021, 8297, 10607, 11027]
set_option maxRecDepth 8192 in
theorem root162Candidates8_checked : candidates edge 4 977 3593 root162Tail9 winner.bound = root162Candidates8 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root162Child8 : search edge 3 [977, 3593] root162Candidates8 4570 winner = winner := by decide
theorem root162Step8 : scan edge (search edge 3) 4 [977] 977 root162Tail8 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail9 winner := by
  rw [root162Tail8, scan, if_neg (by decide), root162Candidates8_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail9
    (if 3 ≤ root162Candidates8.length then search edge 3 [977, 3593] root162Candidates8 4570 winner else winner) = _
  rw [if_pos (by decide), root162Child8]
def root162Candidates9 : List Nat := [11717, 14087]
set_option maxRecDepth 8192 in
theorem root162Candidates9_checked : candidates edge 4 977 3767 root162Tail10 winner.bound = root162Candidates9 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root162Step9 : scan edge (search edge 3) 4 [977] 977 root162Tail9 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail10 winner := by
  rw [root162Tail9, scan, if_neg (by decide), root162Candidates9_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail10
    (if 3 ≤ root162Candidates9.length then search edge 3 [977, 3767] root162Candidates9 4744 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates10 : List Nat := [11717, 13037]
set_option maxRecDepth 8192 in
theorem root162Candidates10_checked : candidates edge 4 977 3851 root162Tail11 winner.bound = root162Candidates10 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root162Step10 : scan edge (search edge 3) 4 [977] 977 root162Tail10 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail11 winner := by
  rw [root162Tail10, scan, if_neg (by decide), root162Candidates10_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail11
    (if 3 ≤ root162Candidates10.length then search edge 3 [977, 3851] root162Candidates10 4828 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates11 : List Nat := [12689]
set_option maxRecDepth 8192 in
theorem root162Candidates11_checked : candidates edge 4 977 4049 root162Tail12 winner.bound = root162Candidates11 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root162Step11 : scan edge (search edge 3) 4 [977] 977 root162Tail11 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail12 winner := by
  rw [root162Tail11, scan, if_neg (by decide), root162Candidates11_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail12
    (if 3 ≤ root162Candidates11.length then search edge 3 [977, 4049] root162Candidates11 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root162Candidates12_checked : candidates edge 4 977 5021 root162Tail13 winner.bound = root162Candidates12 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root162Step12 : scan edge (search edge 3) 4 [977] 977 root162Tail12 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail13 winner := by
  rw [root162Tail12, scan, if_neg (by decide), root162Candidates12_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail13
    (if 3 ≤ root162Candidates12.length then search edge 3 [977, 5021] root162Candidates12 5998 winner else winner) = _
  rw [if_neg (by decide)]
def root162Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root162Candidates13_checked : candidates edge 4 977 5039 root162Tail14 winner.bound = root162Candidates13 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root162Step13 : scan edge (search edge 3) 4 [977] 977 root162Tail13 winner =
    scan edge (search edge 3) 4 [977] 977 root162Tail14 winner := by
  rw [root162Tail13, scan, if_neg (by decide), root162Candidates13_checked]
  change scan edge (search edge 3) 4 [977] 977 root162Tail14
    (if 3 ≤ root162Candidates13.length then search edge 3 [977, 5039] root162Candidates13 6016 winner else winner) = _
  rw [if_neg (by decide)]
theorem root162Scan14 : scan edge (search edge 3) 4 [977] 977 root162Tail14 winner = winner := by rw [root162Tail14, scan, if_pos (by decide)]
theorem root162Scan13 : scan edge (search edge 3) 4 [977] 977 root162Tail13 winner = winner :=
  (root162Step13).trans root162Scan14
theorem root162Scan12 : scan edge (search edge 3) 4 [977] 977 root162Tail12 winner = winner :=
  (root162Step12).trans root162Scan13
theorem root162Scan11 : scan edge (search edge 3) 4 [977] 977 root162Tail11 winner = winner :=
  (root162Step11).trans root162Scan12
theorem root162Scan10 : scan edge (search edge 3) 4 [977] 977 root162Tail10 winner = winner :=
  (root162Step10).trans root162Scan11
theorem root162Scan9 : scan edge (search edge 3) 4 [977] 977 root162Tail9 winner = winner :=
  (root162Step9).trans root162Scan10
theorem root162Scan8 : scan edge (search edge 3) 4 [977] 977 root162Tail8 winner = winner :=
  (root162Step8).trans root162Scan9
theorem root162Scan7 : scan edge (search edge 3) 4 [977] 977 root162Tail7 winner = winner :=
  (root162Step7).trans root162Scan8
theorem root162Scan6 : scan edge (search edge 3) 4 [977] 977 root162Tail6 winner = winner :=
  (root162Step6).trans root162Scan7
theorem root162Scan5 : scan edge (search edge 3) 4 [977] 977 root162Tail5 winner = winner :=
  (root162Step5).trans root162Scan6
theorem root162Scan4 : scan edge (search edge 3) 4 [977] 977 root162Tail4 winner = winner :=
  (root162Step4).trans root162Scan5
theorem root162Scan3 : scan edge (search edge 3) 4 [977] 977 root162Tail3 winner = winner :=
  (root162Step3).trans root162Scan4
theorem root162Scan2 : scan edge (search edge 3) 4 [977] 977 root162Tail2 winner = winner :=
  (root162Step2).trans root162Scan3
theorem root162Scan1 : scan edge (search edge 3) 4 [977] 977 root162Tail1 winner = winner :=
  (root162Step1).trans root162Scan2
theorem root162Scan0 : scan edge (search edge 3) 4 [977] 977 root162Tail0 winner = winner :=
  (root162Step0).trans root162Scan1
set_option maxRecDepth 4096 in
theorem root162Branch : search edge 4 [977] root162Candidates 977 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [977] 977 root162Tail0 winner = winner
  exact root162Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
