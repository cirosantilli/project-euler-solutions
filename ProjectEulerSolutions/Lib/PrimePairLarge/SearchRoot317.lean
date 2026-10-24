import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root317Candidates : List Nat := Data.row2129
set_option maxRecDepth 8192 in
theorem root317Candidates_checked : candidates edge 5 0 2129 tail318 winner.bound = root317Candidates := by
  apply candidates_eq_row edge 5 0 2129 tail318 root317Candidates winner.bound tail318_ordered
  · decide
  · decide
  · funext q
    rfl
def root317Tail51 : List Nat := []
def root317Tail50 : List Nat := 23813 :: root317Tail51
def root317Tail49 : List Nat := 23549 :: root317Tail50
def root317Tail48 : List Nat := 23327 :: root317Tail49
def root317Tail47 : List Nat := 21563 :: root317Tail48
def root317Tail46 : List Nat := 21491 :: root317Tail47
def root317Tail45 : List Nat := 21407 :: root317Tail46
def root317Tail44 : List Nat := 21221 :: root317Tail45
def root317Tail43 : List Nat := 21179 :: root317Tail44
def root317Tail42 : List Nat := 20507 :: root317Tail43
def root317Tail41 : List Nat := 20051 :: root317Tail42
def root317Tail40 : List Nat := 19979 :: root317Tail41
def root317Tail39 : List Nat := 19697 :: root317Tail40
def root317Tail38 : List Nat := 19037 :: root317Tail39
def root317Tail37 : List Nat := 19001 :: root317Tail38
def root317Tail36 : List Nat := 18341 :: root317Tail37
def root317Tail35 : List Nat := 18059 :: root317Tail36
def root317Tail34 : List Nat := 16547 :: root317Tail35
def root317Tail33 : List Nat := 16481 :: root317Tail34
def root317Tail32 : List Nat := 15641 :: root317Tail33
def root317Tail31 : List Nat := 14759 :: root317Tail32
def root317Tail30 : List Nat := 14087 :: root317Tail31
def root317Tail29 : List Nat := 13751 :: root317Tail30
def root317Tail28 : List Nat := 13229 :: root317Tail29
def root317Tail27 : List Nat := 13163 :: root317Tail28
def root317Tail26 : List Nat := 12263 :: root317Tail27
def root317Tail25 : List Nat := 11969 :: root317Tail26
def root317Tail24 : List Nat := 10937 :: root317Tail25
def root317Tail23 : List Nat := 10457 :: root317Tail24
def root317Tail22 : List Nat := 10259 :: root317Tail23
def root317Tail21 : List Nat := 9029 :: root317Tail22
def root317Tail20 : List Nat := 8999 :: root317Tail21
def root317Tail19 : List Nat := 7949 :: root317Tail20
def root317Tail18 : List Nat := 7937 :: root317Tail19
def root317Tail17 : List Nat := 7559 :: root317Tail18
def root317Tail16 : List Nat := 7247 :: root317Tail17
def root317Tail15 : List Nat := 7193 :: root317Tail16
def root317Tail14 : List Nat := 7121 :: root317Tail15
def root317Tail13 : List Nat := 6791 :: root317Tail14
def root317Tail12 : List Nat := 6491 :: root317Tail13
def root317Tail11 : List Nat := 6269 :: root317Tail12
def root317Tail10 : List Nat := 6203 :: root317Tail11
def root317Tail9 : List Nat := 5783 :: root317Tail10
def root317Tail8 : List Nat := 5051 :: root317Tail9
def root317Tail7 : List Nat := 4937 :: root317Tail8
def root317Tail6 : List Nat := 4517 :: root317Tail7
def root317Tail5 : List Nat := 4271 :: root317Tail6
def root317Tail4 : List Nat := 4001 :: root317Tail5
def root317Tail3 : List Nat := 3947 :: root317Tail4
def root317Tail2 : List Nat := 3089 :: root317Tail3
def root317Tail1 : List Nat := 2729 :: root317Tail2
def root317Tail0 : List Nat := 2423 :: root317Tail1
def root317Candidates0 : List Nat := [18341, 21221, 21407]
set_option maxRecDepth 8192 in
theorem root317Candidates0_checked : candidates edge 4 2129 2423 root317Tail1 winner.bound = root317Candidates0 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root317Child0 : search edge 3 [2129, 2423] root317Candidates0 4552 winner = winner := by decide
theorem root317Step0 : scan edge (search edge 3) 4 [2129] 2129 root317Tail0 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail1 winner := by
  rw [root317Tail0, scan, if_neg (by decide), root317Candidates0_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail1
    (if 3 ≤ root317Candidates0.length then search edge 3 [2129, 2423] root317Candidates0 4552 winner else winner) = _
  rw [if_pos (by decide), root317Child0]
def root317Candidates1 : List Nat := [18341, 19979]
set_option maxRecDepth 8192 in
theorem root317Candidates1_checked : candidates edge 4 2129 2729 root317Tail2 winner.bound = root317Candidates1 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root317Step1 : scan edge (search edge 3) 4 [2129] 2129 root317Tail1 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail2 winner := by
  rw [root317Tail1, scan, if_neg (by decide), root317Candidates1_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail2
    (if 3 ≤ root317Candidates1.length then search edge 3 [2129, 2729] root317Candidates1 4858 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates2 : List Nat := [18341]
set_option maxRecDepth 8192 in
theorem root317Candidates2_checked : candidates edge 4 2129 3089 root317Tail3 winner.bound = root317Candidates2 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root317Step2 : scan edge (search edge 3) 4 [2129] 2129 root317Tail2 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail3 winner := by
  rw [root317Tail2, scan, if_neg (by decide), root317Candidates2_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail3
    (if 3 ≤ root317Candidates2.length then search edge 3 [2129, 3089] root317Candidates2 5218 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root317Candidates3_checked : candidates edge 4 2129 3947 root317Tail4 winner.bound = root317Candidates3 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root317Step3 : scan edge (search edge 3) 4 [2129] 2129 root317Tail3 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail4 winner := by
  rw [root317Tail3, scan, if_neg (by decide), root317Candidates3_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail4
    (if 3 ≤ root317Candidates3.length then search edge 3 [2129, 3947] root317Candidates3 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates4 : List Nat := [5783, 7121]
set_option maxRecDepth 8192 in
theorem root317Candidates4_checked : candidates edge 4 2129 4001 root317Tail5 winner.bound = root317Candidates4 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root317Step4 : scan edge (search edge 3) 4 [2129] 2129 root317Tail4 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail5 winner := by
  rw [root317Tail4, scan, if_neg (by decide), root317Candidates4_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail5
    (if 3 ≤ root317Candidates4.length then search edge 3 [2129, 4001] root317Candidates4 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates5 : List Nat := [4517, 9029]
set_option maxRecDepth 8192 in
theorem root317Candidates5_checked : candidates edge 4 2129 4271 root317Tail6 winner.bound = root317Candidates5 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root317Step5 : scan edge (search edge 3) 4 [2129] 2129 root317Tail5 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail6 winner := by
  rw [root317Tail5, scan, if_neg (by decide), root317Candidates5_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail6
    (if 3 ≤ root317Candidates5.length then search edge 3 [2129, 4271] root317Candidates5 6400 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root317Candidates6_checked : candidates edge 4 2129 4517 root317Tail7 winner.bound = root317Candidates6 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root317Step6 : scan edge (search edge 3) 4 [2129] 2129 root317Tail6 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail7 winner := by
  rw [root317Tail6, scan, if_neg (by decide), root317Candidates6_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail7
    (if 3 ≤ root317Candidates6.length then search edge 3 [2129, 4517] root317Candidates6 6646 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates7 : List Nat := [6491, 7247]
set_option maxRecDepth 8192 in
theorem root317Candidates7_checked : candidates edge 4 2129 4937 root317Tail8 winner.bound = root317Candidates7 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root317Step7 : scan edge (search edge 3) 4 [2129] 2129 root317Tail7 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail8 winner := by
  rw [root317Tail7, scan, if_neg (by decide), root317Candidates7_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail8
    (if 3 ≤ root317Candidates7.length then search edge 3 [2129, 4937] root317Candidates7 7066 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates8 : List Nat := [7949]
set_option maxRecDepth 8192 in
theorem root317Candidates8_checked : candidates edge 4 2129 5051 root317Tail9 winner.bound = root317Candidates8 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root317Step8 : scan edge (search edge 3) 4 [2129] 2129 root317Tail8 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail9 winner := by
  rw [root317Tail8, scan, if_neg (by decide), root317Candidates8_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail9
    (if 3 ≤ root317Candidates8.length then search edge 3 [2129, 5051] root317Candidates8 7180 winner else winner) = _
  rw [if_neg (by decide)]
def root317Candidates9 : List Nat := [6491]
set_option maxRecDepth 8192 in
theorem root317Candidates9_checked : candidates edge 4 2129 5783 root317Tail10 winner.bound = root317Candidates9 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root317Step9 : scan edge (search edge 3) 4 [2129] 2129 root317Tail9 winner =
    scan edge (search edge 3) 4 [2129] 2129 root317Tail10 winner := by
  rw [root317Tail9, scan, if_neg (by decide), root317Candidates9_checked]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail10
    (if 3 ≤ root317Candidates9.length then search edge 3 [2129, 5783] root317Candidates9 7912 winner else winner) = _
  rw [if_neg (by decide)]
theorem root317Scan10 : scan edge (search edge 3) 4 [2129] 2129 root317Tail10 winner = winner := by rw [root317Tail10, scan, if_pos (by decide)]
theorem root317Scan9 : scan edge (search edge 3) 4 [2129] 2129 root317Tail9 winner = winner :=
  (root317Step9).trans root317Scan10
theorem root317Scan8 : scan edge (search edge 3) 4 [2129] 2129 root317Tail8 winner = winner :=
  (root317Step8).trans root317Scan9
theorem root317Scan7 : scan edge (search edge 3) 4 [2129] 2129 root317Tail7 winner = winner :=
  (root317Step7).trans root317Scan8
theorem root317Scan6 : scan edge (search edge 3) 4 [2129] 2129 root317Tail6 winner = winner :=
  (root317Step6).trans root317Scan7
theorem root317Scan5 : scan edge (search edge 3) 4 [2129] 2129 root317Tail5 winner = winner :=
  (root317Step5).trans root317Scan6
theorem root317Scan4 : scan edge (search edge 3) 4 [2129] 2129 root317Tail4 winner = winner :=
  (root317Step4).trans root317Scan5
theorem root317Scan3 : scan edge (search edge 3) 4 [2129] 2129 root317Tail3 winner = winner :=
  (root317Step3).trans root317Scan4
theorem root317Scan2 : scan edge (search edge 3) 4 [2129] 2129 root317Tail2 winner = winner :=
  (root317Step2).trans root317Scan3
theorem root317Scan1 : scan edge (search edge 3) 4 [2129] 2129 root317Tail1 winner = winner :=
  (root317Step1).trans root317Scan2
theorem root317Scan0 : scan edge (search edge 3) 4 [2129] 2129 root317Tail0 winner = winner :=
  (root317Step0).trans root317Scan1
set_option maxRecDepth 4096 in
theorem root317Branch : search edge 4 [2129] root317Candidates 2129 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2129] 2129 root317Tail0 winner = winner
  exact root317Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
