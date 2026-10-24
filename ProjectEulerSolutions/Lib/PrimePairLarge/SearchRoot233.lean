import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root233Candidates : List Nat := Data.row1487
set_option maxRecDepth 8192 in
theorem root233Candidates_checked : candidates edge 5 0 1487 tail234 winner.bound = root233Candidates := by
  apply candidates_eq_row edge 5 0 1487 tail234 root233Candidates winner.bound tail234_ordered
  · decide
  · decide
  · funext q
    rfl
def root233Tail34 : List Nat := []
def root233Tail33 : List Nat := 23873 :: root233Tail34
def root233Tail32 : List Nat := 23357 :: root233Tail33
def root233Tail31 : List Nat := 23321 :: root233Tail32
def root233Tail30 : List Nat := 22901 :: root233Tail31
def root233Tail29 : List Nat := 22271 :: root233Tail30
def root233Tail28 : List Nat := 22193 :: root233Tail29
def root233Tail27 : List Nat := 20753 :: root233Tail28
def root233Tail26 : List Nat := 20717 :: root233Tail27
def root233Tail25 : List Nat := 19379 :: root233Tail26
def root233Tail24 : List Nat := 19079 :: root233Tail25
def root233Tail23 : List Nat := 19073 :: root233Tail24
def root233Tail22 : List Nat := 17489 :: root233Tail23
def root233Tail21 : List Nat := 16943 :: root233Tail22
def root233Tail20 : List Nat := 16649 :: root233Tail21
def root233Tail19 : List Nat := 16193 :: root233Tail20
def root233Tail18 : List Nat := 13463 :: root233Tail19
def root233Tail17 : List Nat := 10799 :: root233Tail18
def root233Tail16 : List Nat := 8867 :: root233Tail17
def root233Tail15 : List Nat := 8573 :: root233Tail16
def root233Tail14 : List Nat := 7727 :: root233Tail15
def root233Tail13 : List Nat := 7013 :: root233Tail14
def root233Tail12 : List Nat := 6761 :: root233Tail13
def root233Tail11 : List Nat := 6473 :: root233Tail12
def root233Tail10 : List Nat := 5843 :: root233Tail11
def root233Tail9 : List Nat := 5801 :: root233Tail10
def root233Tail8 : List Nat := 5717 :: root233Tail9
def root233Tail7 : List Nat := 3671 :: root233Tail8
def root233Tail6 : List Nat := 3659 :: root233Tail7
def root233Tail5 : List Nat := 2729 :: root233Tail6
def root233Tail4 : List Nat := 2267 :: root233Tail5
def root233Tail3 : List Nat := 1979 :: root233Tail4
def root233Tail2 : List Nat := 1907 :: root233Tail3
def root233Tail1 : List Nat := 1601 :: root233Tail2
def root233Tail0 : List Nat := 1553 :: root233Tail1
def root233Candidates0 : List Nat := [7013, 19379]
set_option maxRecDepth 8192 in
theorem root233Candidates0_checked : candidates edge 4 1487 1553 root233Tail1 winner.bound = root233Candidates0 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
theorem root233Step0 : scan edge (search edge 3) 4 [1487] 1487 root233Tail0 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail1 winner := by
  rw [root233Tail0, scan, if_neg (by decide), root233Candidates0_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail1
    (if 3 ≤ root233Candidates0.length then search edge 3 [1487, 1553] root233Candidates0 3040 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root233Candidates1_checked : candidates edge 4 1487 1601 root233Tail2 winner.bound = root233Candidates1 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root233Step1 : scan edge (search edge 3) 4 [1487] 1487 root233Tail1 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail2 winner := by
  rw [root233Tail1, scan, if_neg (by decide), root233Candidates1_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail2
    (if 3 ≤ root233Candidates1.length then search edge 3 [1487, 1601] root233Candidates1 3088 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates2 : List Nat := [8867, 17489]
set_option maxRecDepth 8192 in
theorem root233Candidates2_checked : candidates edge 4 1487 1907 root233Tail3 winner.bound = root233Candidates2 := by
  unfold candidates
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row]
  decide
theorem root233Step2 : scan edge (search edge 3) 4 [1487] 1487 root233Tail2 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail3 winner := by
  rw [root233Tail2, scan, if_neg (by decide), root233Candidates2_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail3
    (if 3 ≤ root233Candidates2.length then search edge 3 [1487, 1907] root233Candidates2 3394 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root233Candidates3_checked : candidates edge 4 1487 1979 root233Tail4 winner.bound = root233Candidates3 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root233Step3 : scan edge (search edge 3) 4 [1487] 1487 root233Tail3 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail4 winner := by
  rw [root233Tail3, scan, if_neg (by decide), root233Candidates3_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail4
    (if 3 ≤ root233Candidates3.length then search edge 3 [1487, 1979] root233Candidates3 3466 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates4 : List Nat := [2729, 5843]
set_option maxRecDepth 8192 in
theorem root233Candidates4_checked : candidates edge 4 1487 2267 root233Tail5 winner.bound = root233Candidates4 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root233Step4 : scan edge (search edge 3) 4 [1487] 1487 root233Tail4 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail5 winner := by
  rw [root233Tail4, scan, if_neg (by decide), root233Candidates4_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail5
    (if 3 ≤ root233Candidates4.length then search edge 3 [1487, 2267] root233Candidates4 3754 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates5 : List Nat := [6473]
set_option maxRecDepth 8192 in
theorem root233Candidates5_checked : candidates edge 4 1487 2729 root233Tail6 winner.bound = root233Candidates5 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root233Step5 : scan edge (search edge 3) 4 [1487] 1487 root233Tail5 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail6 winner := by
  rw [root233Tail5, scan, if_neg (by decide), root233Candidates5_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail6
    (if 3 ≤ root233Candidates5.length then search edge 3 [1487, 2729] root233Candidates5 4216 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root233Candidates6_checked : candidates edge 4 1487 3659 root233Tail7 winner.bound = root233Candidates6 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
theorem root233Step6 : scan edge (search edge 3) 4 [1487] 1487 root233Tail6 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail7 winner := by
  rw [root233Tail6, scan, if_neg (by decide), root233Candidates6_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail7
    (if 3 ≤ root233Candidates6.length then search edge 3 [1487, 3659] root233Candidates6 5146 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates7 : List Nat := [19073]
set_option maxRecDepth 8192 in
theorem root233Candidates7_checked : candidates edge 4 1487 3671 root233Tail8 winner.bound = root233Candidates7 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root233Step7 : scan edge (search edge 3) 4 [1487] 1487 root233Tail7 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail8 winner := by
  rw [root233Tail7, scan, if_neg (by decide), root233Candidates7_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail8
    (if 3 ≤ root233Candidates7.length then search edge 3 [1487, 3671] root233Candidates7 5158 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root233Candidates8_checked : candidates edge 4 1487 5717 root233Tail9 winner.bound = root233Candidates8 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root233Step8 : scan edge (search edge 3) 4 [1487] 1487 root233Tail8 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail9 winner := by
  rw [root233Tail8, scan, if_neg (by decide), root233Candidates8_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail9
    (if 3 ≤ root233Candidates8.length then search edge 3 [1487, 5717] root233Candidates8 7204 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root233Candidates9_checked : candidates edge 4 1487 5801 root233Tail10 winner.bound = root233Candidates9 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root233Step9 : scan edge (search edge 3) 4 [1487] 1487 root233Tail9 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail10 winner := by
  rw [root233Tail9, scan, if_neg (by decide), root233Candidates9_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail10
    (if 3 ≤ root233Candidates9.length then search edge 3 [1487, 5801] root233Candidates9 7288 winner else winner) = _
  rw [if_neg (by decide)]
def root233Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root233Candidates10_checked : candidates edge 4 1487 5843 root233Tail11 winner.bound = root233Candidates10 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root233Step10 : scan edge (search edge 3) 4 [1487] 1487 root233Tail10 winner =
    scan edge (search edge 3) 4 [1487] 1487 root233Tail11 winner := by
  rw [root233Tail10, scan, if_neg (by decide), root233Candidates10_checked]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail11
    (if 3 ≤ root233Candidates10.length then search edge 3 [1487, 5843] root233Candidates10 7330 winner else winner) = _
  rw [if_neg (by decide)]
theorem root233Scan11 : scan edge (search edge 3) 4 [1487] 1487 root233Tail11 winner = winner := by rw [root233Tail11, scan, if_pos (by decide)]
theorem root233Scan10 : scan edge (search edge 3) 4 [1487] 1487 root233Tail10 winner = winner :=
  (root233Step10).trans root233Scan11
theorem root233Scan9 : scan edge (search edge 3) 4 [1487] 1487 root233Tail9 winner = winner :=
  (root233Step9).trans root233Scan10
theorem root233Scan8 : scan edge (search edge 3) 4 [1487] 1487 root233Tail8 winner = winner :=
  (root233Step8).trans root233Scan9
theorem root233Scan7 : scan edge (search edge 3) 4 [1487] 1487 root233Tail7 winner = winner :=
  (root233Step7).trans root233Scan8
theorem root233Scan6 : scan edge (search edge 3) 4 [1487] 1487 root233Tail6 winner = winner :=
  (root233Step6).trans root233Scan7
theorem root233Scan5 : scan edge (search edge 3) 4 [1487] 1487 root233Tail5 winner = winner :=
  (root233Step5).trans root233Scan6
theorem root233Scan4 : scan edge (search edge 3) 4 [1487] 1487 root233Tail4 winner = winner :=
  (root233Step4).trans root233Scan5
theorem root233Scan3 : scan edge (search edge 3) 4 [1487] 1487 root233Tail3 winner = winner :=
  (root233Step3).trans root233Scan4
theorem root233Scan2 : scan edge (search edge 3) 4 [1487] 1487 root233Tail2 winner = winner :=
  (root233Step2).trans root233Scan3
theorem root233Scan1 : scan edge (search edge 3) 4 [1487] 1487 root233Tail1 winner = winner :=
  (root233Step1).trans root233Scan2
theorem root233Scan0 : scan edge (search edge 3) 4 [1487] 1487 root233Tail0 winner = winner :=
  (root233Step0).trans root233Scan1
set_option maxRecDepth 4096 in
theorem root233Branch : search edge 4 [1487] root233Candidates 1487 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1487] 1487 root233Tail0 winner = winner
  exact root233Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
