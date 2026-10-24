import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root280Candidates : List Nat := Data.row1847
set_option maxRecDepth 8192 in
theorem root280Candidates_checked : candidates edge 5 0 1847 tail281 winner.bound = root280Candidates := by
  apply candidates_eq_row edge 5 0 1847 tail281 root280Candidates winner.bound tail281_ordered
  · decide
  · decide
  · funext q
    rfl
def root280Tail48 : List Nat := []
def root280Tail47 : List Nat := 24107 :: root280Tail48
def root280Tail46 : List Nat := 23663 :: root280Tail47
def root280Tail45 : List Nat := 23447 :: root280Tail46
def root280Tail44 : List Nat := 21599 :: root280Tail45
def root280Tail43 : List Nat := 21521 :: root280Tail44
def root280Tail42 : List Nat := 20051 :: root280Tail43
def root280Tail41 : List Nat := 19991 :: root280Tail42
def root280Tail40 : List Nat := 18257 :: root280Tail41
def root280Tail39 : List Nat := 17837 :: root280Tail40
def root280Tail38 : List Nat := 17783 :: root280Tail39
def root280Tail37 : List Nat := 17681 :: root280Tail38
def root280Tail36 : List Nat := 17657 :: root280Tail37
def root280Tail35 : List Nat := 16139 :: root280Tail36
def root280Tail34 : List Nat := 15767 :: root280Tail35
def root280Tail33 : List Nat := 15299 :: root280Tail34
def root280Tail32 : List Nat := 13913 :: root280Tail33
def root280Tail31 : List Nat := 13229 :: root280Tail32
def root280Tail30 : List Nat := 12071 :: root280Tail31
def root280Tail29 : List Nat := 10613 :: root280Tail30
def root280Tail28 : List Nat := 10259 :: root280Tail29
def root280Tail27 : List Nat := 9941 :: root280Tail28
def root280Tail26 : List Nat := 9743 :: root280Tail27
def root280Tail25 : List Nat := 9311 :: root280Tail26
def root280Tail24 : List Nat := 8951 :: root280Tail25
def root280Tail23 : List Nat := 8783 :: root280Tail24
def root280Tail22 : List Nat := 8693 :: root280Tail23
def root280Tail21 : List Nat := 8123 :: root280Tail22
def root280Tail20 : List Nat := 8069 :: root280Tail21
def root280Tail19 : List Nat := 7703 :: root280Tail20
def root280Tail18 : List Nat := 7523 :: root280Tail19
def root280Tail17 : List Nat := 7307 :: root280Tail18
def root280Tail16 : List Nat := 6173 :: root280Tail17
def root280Tail15 : List Nat := 5927 :: root280Tail16
def root280Tail14 : List Nat := 5801 :: root280Tail15
def root280Tail13 : List Nat := 5519 :: root280Tail14
def root280Tail12 : List Nat := 4919 :: root280Tail13
def root280Tail11 : List Nat := 4691 :: root280Tail12
def root280Tail10 : List Nat := 4457 :: root280Tail11
def root280Tail9 : List Nat := 4373 :: root280Tail10
def root280Tail8 : List Nat := 4283 :: root280Tail9
def root280Tail7 : List Nat := 4007 :: root280Tail8
def root280Tail6 : List Nat := 3863 :: root280Tail7
def root280Tail5 : List Nat := 3779 :: root280Tail6
def root280Tail4 : List Nat := 3671 :: root280Tail5
def root280Tail3 : List Nat := 3011 :: root280Tail4
def root280Tail2 : List Nat := 2969 :: root280Tail3
def root280Tail1 : List Nat := 2687 :: root280Tail2
def root280Tail0 : List Nat := 2423 :: root280Tail1
def root280Candidates0 : List Nat := [3863, 13913, 15767]
set_option maxRecDepth 8192 in
theorem root280Candidates0_checked : candidates edge 4 1847 2423 root280Tail1 winner.bound = root280Candidates0 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root280Child0 : search edge 3 [1847, 2423] root280Candidates0 4270 winner = winner := by decide
theorem root280Step0 : scan edge (search edge 3) 4 [1847] 1847 root280Tail0 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail1 winner := by
  rw [root280Tail0, scan, if_neg (by decide), root280Candidates0_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail1
    (if 3 ≤ root280Candidates0.length then search edge 3 [1847, 2423] root280Candidates0 4270 winner else winner) = _
  rw [if_pos (by decide), root280Child0]
def root280Candidates1 : List Nat := [4007, 7703, 8069, 8693, 17783]
set_option maxRecDepth 8192 in
theorem root280Candidates1_checked : candidates edge 4 1847 2687 root280Tail2 winner.bound = root280Candidates1 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root280Child1 : search edge 3 [1847, 2687] root280Candidates1 4534 winner = winner := by decide
theorem root280Step1 : scan edge (search edge 3) 4 [1847] 1847 root280Tail1 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail2 winner := by
  rw [root280Tail1, scan, if_neg (by decide), root280Candidates1_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail2
    (if 3 ≤ root280Candidates1.length then search edge 3 [1847, 2687] root280Candidates1 4534 winner else winner) = _
  rw [if_pos (by decide), root280Child1]
def root280Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root280Candidates2_checked : candidates edge 4 1847 2969 root280Tail3 winner.bound = root280Candidates2 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root280Step2 : scan edge (search edge 3) 4 [1847] 1847 root280Tail2 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail3 winner := by
  rw [root280Tail2, scan, if_neg (by decide), root280Candidates2_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail3
    (if 3 ≤ root280Candidates2.length then search edge 3 [1847, 2969] root280Candidates2 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root280Candidates3_checked : candidates edge 4 1847 3011 root280Tail4 winner.bound = root280Candidates3 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root280Step3 : scan edge (search edge 3) 4 [1847] 1847 root280Tail3 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail4 winner := by
  rw [root280Tail3, scan, if_neg (by decide), root280Candidates3_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail4
    (if 3 ≤ root280Candidates3.length then search edge 3 [1847, 3011] root280Candidates3 4858 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates4 : List Nat := [7523]
set_option maxRecDepth 8192 in
theorem root280Candidates4_checked : candidates edge 4 1847 3671 root280Tail5 winner.bound = root280Candidates4 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root280Step4 : scan edge (search edge 3) 4 [1847] 1847 root280Tail4 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail5 winner := by
  rw [root280Tail4, scan, if_neg (by decide), root280Candidates4_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail5
    (if 3 ≤ root280Candidates4.length then search edge 3 [1847, 3671] root280Candidates4 5518 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates5 : List Nat := [3863, 7523, 9941, 17783]
set_option maxRecDepth 8192 in
theorem root280Candidates5_checked : candidates edge 4 1847 3779 root280Tail6 winner.bound = root280Candidates5 := by
  unfold candidates
  have row : edge 3779 = (SortedNatTree.contains FastRows.row3779) := by
    change Data.row3779.contains = (SortedNatTree.contains FastRows.row3779)
    exact FastRows.row3779_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root280Child5 : search edge 3 [1847, 3779] root280Candidates5 5626 winner = winner := by decide
theorem root280Step5 : scan edge (search edge 3) 4 [1847] 1847 root280Tail5 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail6 winner := by
  rw [root280Tail5, scan, if_neg (by decide), root280Candidates5_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail6
    (if 3 ≤ root280Candidates5.length then search edge 3 [1847, 3779] root280Candidates5 5626 winner else winner) = _
  rw [if_pos (by decide), root280Child5]
def root280Candidates6 : List Nat := [4007]
set_option maxRecDepth 8192 in
theorem root280Candidates6_checked : candidates edge 4 1847 3863 root280Tail7 winner.bound = root280Candidates6 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root280Step6 : scan edge (search edge 3) 4 [1847] 1847 root280Tail6 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail7 winner := by
  rw [root280Tail6, scan, if_neg (by decide), root280Candidates6_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail7
    (if 3 ≤ root280Candidates6.length then search edge 3 [1847, 3863] root280Candidates6 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates7 : List Nat := [8693]
set_option maxRecDepth 8192 in
theorem root280Candidates7_checked : candidates edge 4 1847 4007 root280Tail8 winner.bound = root280Candidates7 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
theorem root280Step7 : scan edge (search edge 3) 4 [1847] 1847 root280Tail7 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail8 winner := by
  rw [root280Tail7, scan, if_neg (by decide), root280Candidates7_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail8
    (if 3 ≤ root280Candidates7.length then search edge 3 [1847, 4007] root280Candidates7 5854 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates8 : List Nat := [8069, 10613]
set_option maxRecDepth 8192 in
theorem root280Candidates8_checked : candidates edge 4 1847 4283 root280Tail9 winner.bound = root280Candidates8 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root280Step8 : scan edge (search edge 3) 4 [1847] 1847 root280Tail8 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail9 winner := by
  rw [root280Tail8, scan, if_neg (by decide), root280Candidates8_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail9
    (if 3 ≤ root280Candidates8.length then search edge 3 [1847, 4283] root280Candidates8 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates9 : List Nat := [5801, 5927]
set_option maxRecDepth 8192 in
theorem root280Candidates9_checked : candidates edge 4 1847 4373 root280Tail10 winner.bound = root280Candidates9 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root280Step9 : scan edge (search edge 3) 4 [1847] 1847 root280Tail9 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail10 winner := by
  rw [root280Tail9, scan, if_neg (by decide), root280Candidates9_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail10
    (if 3 ≤ root280Candidates9.length then search edge 3 [1847, 4373] root280Candidates9 6220 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root280Candidates10_checked : candidates edge 4 1847 4457 root280Tail11 winner.bound = root280Candidates10 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root280Step10 : scan edge (search edge 3) 4 [1847] 1847 root280Tail10 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail11 winner := by
  rw [root280Tail10, scan, if_neg (by decide), root280Candidates10_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail11
    (if 3 ≤ root280Candidates10.length then search edge 3 [1847, 4457] root280Candidates10 6304 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root280Candidates11_checked : candidates edge 4 1847 4691 root280Tail12 winner.bound = root280Candidates11 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root280Step11 : scan edge (search edge 3) 4 [1847] 1847 root280Tail11 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail12 winner := by
  rw [root280Tail11, scan, if_neg (by decide), root280Candidates11_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail12
    (if 3 ≤ root280Candidates11.length then search edge 3 [1847, 4691] root280Candidates11 6538 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates12 : List Nat := [8783]
set_option maxRecDepth 8192 in
theorem root280Candidates12_checked : candidates edge 4 1847 4919 root280Tail13 winner.bound = root280Candidates12 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root280Step12 : scan edge (search edge 3) 4 [1847] 1847 root280Tail12 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail13 winner := by
  rw [root280Tail12, scan, if_neg (by decide), root280Candidates12_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail13
    (if 3 ≤ root280Candidates12.length then search edge 3 [1847, 4919] root280Candidates12 6766 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root280Candidates13_checked : candidates edge 4 1847 5519 root280Tail14 winner.bound = root280Candidates13 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root280Step13 : scan edge (search edge 3) 4 [1847] 1847 root280Tail13 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail14 winner := by
  rw [root280Tail13, scan, if_neg (by decide), root280Candidates13_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail14
    (if 3 ≤ root280Candidates13.length then search edge 3 [1847, 5519] root280Candidates13 7366 winner else winner) = _
  rw [if_neg (by decide)]
def root280Candidates14 : List Nat := [6173, 8693, 8783]
set_option maxRecDepth 8192 in
theorem root280Candidates14_checked : candidates edge 4 1847 5801 root280Tail15 winner.bound = root280Candidates14 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root280Child14 : search edge 3 [1847, 5801] root280Candidates14 7648 winner = winner := by decide
theorem root280Step14 : scan edge (search edge 3) 4 [1847] 1847 root280Tail14 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail15 winner := by
  rw [root280Tail14, scan, if_neg (by decide), root280Candidates14_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail15
    (if 3 ≤ root280Candidates14.length then search edge 3 [1847, 5801] root280Candidates14 7648 winner else winner) = _
  rw [if_pos (by decide), root280Child14]
def root280Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root280Candidates15_checked : candidates edge 4 1847 5927 root280Tail16 winner.bound = root280Candidates15 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root280Step15 : scan edge (search edge 3) 4 [1847] 1847 root280Tail15 winner =
    scan edge (search edge 3) 4 [1847] 1847 root280Tail16 winner := by
  rw [root280Tail15, scan, if_neg (by decide), root280Candidates15_checked]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail16
    (if 3 ≤ root280Candidates15.length then search edge 3 [1847, 5927] root280Candidates15 7774 winner else winner) = _
  rw [if_neg (by decide)]
theorem root280Scan16 : scan edge (search edge 3) 4 [1847] 1847 root280Tail16 winner = winner := by rw [root280Tail16, scan, if_pos (by decide)]
theorem root280Scan15 : scan edge (search edge 3) 4 [1847] 1847 root280Tail15 winner = winner :=
  (root280Step15).trans root280Scan16
theorem root280Scan14 : scan edge (search edge 3) 4 [1847] 1847 root280Tail14 winner = winner :=
  (root280Step14).trans root280Scan15
theorem root280Scan13 : scan edge (search edge 3) 4 [1847] 1847 root280Tail13 winner = winner :=
  (root280Step13).trans root280Scan14
theorem root280Scan12 : scan edge (search edge 3) 4 [1847] 1847 root280Tail12 winner = winner :=
  (root280Step12).trans root280Scan13
theorem root280Scan11 : scan edge (search edge 3) 4 [1847] 1847 root280Tail11 winner = winner :=
  (root280Step11).trans root280Scan12
theorem root280Scan10 : scan edge (search edge 3) 4 [1847] 1847 root280Tail10 winner = winner :=
  (root280Step10).trans root280Scan11
theorem root280Scan9 : scan edge (search edge 3) 4 [1847] 1847 root280Tail9 winner = winner :=
  (root280Step9).trans root280Scan10
theorem root280Scan8 : scan edge (search edge 3) 4 [1847] 1847 root280Tail8 winner = winner :=
  (root280Step8).trans root280Scan9
theorem root280Scan7 : scan edge (search edge 3) 4 [1847] 1847 root280Tail7 winner = winner :=
  (root280Step7).trans root280Scan8
theorem root280Scan6 : scan edge (search edge 3) 4 [1847] 1847 root280Tail6 winner = winner :=
  (root280Step6).trans root280Scan7
theorem root280Scan5 : scan edge (search edge 3) 4 [1847] 1847 root280Tail5 winner = winner :=
  (root280Step5).trans root280Scan6
theorem root280Scan4 : scan edge (search edge 3) 4 [1847] 1847 root280Tail4 winner = winner :=
  (root280Step4).trans root280Scan5
theorem root280Scan3 : scan edge (search edge 3) 4 [1847] 1847 root280Tail3 winner = winner :=
  (root280Step3).trans root280Scan4
theorem root280Scan2 : scan edge (search edge 3) 4 [1847] 1847 root280Tail2 winner = winner :=
  (root280Step2).trans root280Scan3
theorem root280Scan1 : scan edge (search edge 3) 4 [1847] 1847 root280Tail1 winner = winner :=
  (root280Step1).trans root280Scan2
theorem root280Scan0 : scan edge (search edge 3) 4 [1847] 1847 root280Tail0 winner = winner :=
  (root280Step0).trans root280Scan1
set_option maxRecDepth 4096 in
theorem root280Branch : search edge 4 [1847] root280Candidates 1847 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1847] 1847 root280Tail0 winner = winner
  exact root280Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
