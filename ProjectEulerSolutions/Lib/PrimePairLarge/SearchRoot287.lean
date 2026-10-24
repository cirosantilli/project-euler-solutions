import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root287Candidates : List Nat := Data.row1889
set_option maxRecDepth 8192 in
theorem root287Candidates_checked : candidates edge 5 0 1889 tail288 winner.bound = root287Candidates := by
  apply candidates_eq_row edge 5 0 1889 tail288 root287Candidates winner.bound tail288_ordered
  · decide
  · decide
  · funext q
    rfl
def root287Tail49 : List Nat := []
def root287Tail48 : List Nat := 23993 :: root287Tail49
def root287Tail47 : List Nat := 23561 :: root287Tail48
def root287Tail46 : List Nat := 23003 :: root287Tail47
def root287Tail45 : List Nat := 22901 :: root287Tail46
def root287Tail44 : List Nat := 22067 :: root287Tail45
def root287Tail43 : List Nat := 19697 :: root287Tail44
def root287Tail42 : List Nat := 18617 :: root287Tail43
def root287Tail41 : List Nat := 18077 :: root287Tail42
def root287Tail40 : List Nat := 17609 :: root287Tail41
def root287Tail39 : List Nat := 17291 :: root287Tail40
def root287Tail38 : List Nat := 15791 :: root287Tail39
def root287Tail37 : List Nat := 15443 :: root287Tail38
def root287Tail36 : List Nat := 14753 :: root287Tail37
def root287Tail35 : List Nat := 13619 :: root287Tail36
def root287Tail34 : List Nat := 13499 :: root287Tail35
def root287Tail33 : List Nat := 13187 :: root287Tail34
def root287Tail32 : List Nat := 12527 :: root287Tail33
def root287Tail31 : List Nat := 12107 :: root287Tail32
def root287Tail30 : List Nat := 12011 :: root287Tail31
def root287Tail29 : List Nat := 11717 :: root287Tail30
def root287Tail28 : List Nat := 10889 :: root287Tail29
def root287Tail27 : List Nat := 10781 :: root287Tail28
def root287Tail26 : List Nat := 10529 :: root287Tail27
def root287Tail25 : List Nat := 10457 :: root287Tail26
def root287Tail24 : List Nat := 10289 :: root287Tail25
def root287Tail23 : List Nat := 10193 :: root287Tail24
def root287Tail22 : List Nat := 10181 :: root287Tail23
def root287Tail21 : List Nat := 9941 :: root287Tail22
def root287Tail20 : List Nat := 9929 :: root287Tail21
def root287Tail19 : List Nat := 9539 :: root287Tail20
def root287Tail18 : List Nat := 9533 :: root287Tail19
def root287Tail17 : List Nat := 8861 :: root287Tail18
def root287Tail16 : List Nat := 8291 :: root287Tail17
def root287Tail15 : List Nat := 8123 :: root287Tail16
def root287Tail14 : List Nat := 7019 :: root287Tail15
def root287Tail13 : List Nat := 6833 :: root287Tail14
def root287Tail12 : List Nat := 5417 :: root287Tail13
def root287Tail11 : List Nat := 5297 :: root287Tail12
def root287Tail10 : List Nat := 4919 :: root287Tail11
def root287Tail9 : List Nat := 4019 :: root287Tail10
def root287Tail8 : List Nat := 3491 :: root287Tail9
def root287Tail7 : List Nat := 2999 :: root287Tail8
def root287Tail6 : List Nat := 2927 :: root287Tail7
def root287Tail5 : List Nat := 2843 :: root287Tail6
def root287Tail4 : List Nat := 2777 :: root287Tail5
def root287Tail3 : List Nat := 2633 :: root287Tail4
def root287Tail2 : List Nat := 2243 :: root287Tail3
def root287Tail1 : List Nat := 2129 :: root287Tail2
def root287Tail0 : List Nat := 2087 :: root287Tail1
def root287Candidates0 : List Nat := [12011, 17291]
set_option maxRecDepth 8192 in
theorem root287Candidates0_checked : candidates edge 4 1889 2087 root287Tail1 winner.bound = root287Candidates0 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root287Step0 : scan edge (search edge 3) 4 [1889] 1889 root287Tail0 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail1 winner := by
  rw [root287Tail0, scan, if_neg (by decide), root287Candidates0_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail1
    (if 3 ≤ root287Candidates0.length then search edge 3 [1889, 2087] root287Candidates0 3976 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates1 : List Nat := [10457, 19697]
set_option maxRecDepth 8192 in
theorem root287Candidates1_checked : candidates edge 4 1889 2129 root287Tail2 winner.bound = root287Candidates1 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root287Step1 : scan edge (search edge 3) 4 [1889] 1889 root287Tail1 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail2 winner := by
  rw [root287Tail1, scan, if_neg (by decide), root287Candidates1_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail2
    (if 3 ≤ root287Candidates1.length then search edge 3 [1889, 2129] root287Candidates1 4018 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates2 : List Nat := [12527]
set_option maxRecDepth 8192 in
theorem root287Candidates2_checked : candidates edge 4 1889 2243 root287Tail3 winner.bound = root287Candidates2 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
theorem root287Step2 : scan edge (search edge 3) 4 [1889] 1889 root287Tail2 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail3 winner := by
  rw [root287Tail2, scan, if_neg (by decide), root287Candidates2_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail3
    (if 3 ≤ root287Candidates2.length then search edge 3 [1889, 2243] root287Candidates2 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root287Candidates3_checked : candidates edge 4 1889 2633 root287Tail4 winner.bound = root287Candidates3 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root287Step3 : scan edge (search edge 3) 4 [1889] 1889 root287Tail3 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail4 winner := by
  rw [root287Tail3, scan, if_neg (by decide), root287Candidates3_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail4
    (if 3 ≤ root287Candidates3.length then search edge 3 [1889, 2633] root287Candidates3 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates4 : List Nat := [8291, 9533, 10781]
set_option maxRecDepth 8192 in
theorem root287Candidates4_checked : candidates edge 4 1889 2777 root287Tail5 winner.bound = root287Candidates4 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root287Child4 : search edge 3 [1889, 2777] root287Candidates4 4666 winner = winner := by decide
theorem root287Step4 : scan edge (search edge 3) 4 [1889] 1889 root287Tail4 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail5 winner := by
  rw [root287Tail4, scan, if_neg (by decide), root287Candidates4_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail5
    (if 3 ≤ root287Candidates4.length then search edge 3 [1889, 2777] root287Candidates4 4666 winner else winner) = _
  rw [if_pos (by decide), root287Child4]
def root287Candidates5 : List Nat := [10457]
set_option maxRecDepth 8192 in
theorem root287Candidates5_checked : candidates edge 4 1889 2843 root287Tail6 winner.bound = root287Candidates5 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root287Step5 : scan edge (search edge 3) 4 [1889] 1889 root287Tail5 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail6 winner := by
  rw [root287Tail5, scan, if_neg (by decide), root287Candidates5_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail6
    (if 3 ≤ root287Candidates5.length then search edge 3 [1889, 2843] root287Candidates5 4732 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root287Candidates6_checked : candidates edge 4 1889 2927 root287Tail7 winner.bound = root287Candidates6 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root287Step6 : scan edge (search edge 3) 4 [1889] 1889 root287Tail6 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail7 winner := by
  rw [root287Tail6, scan, if_neg (by decide), root287Candidates6_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail7
    (if 3 ≤ root287Candidates6.length then search edge 3 [1889, 2927] root287Candidates6 4816 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates7 : List Nat := [17609]
set_option maxRecDepth 8192 in
theorem root287Candidates7_checked : candidates edge 4 1889 2999 root287Tail8 winner.bound = root287Candidates7 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root287Step7 : scan edge (search edge 3) 4 [1889] 1889 root287Tail7 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail8 winner := by
  rw [root287Tail7, scan, if_neg (by decide), root287Candidates7_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail8
    (if 3 ≤ root287Candidates7.length then search edge 3 [1889, 2999] root287Candidates7 4888 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates8 : List Nat := [8123]
set_option maxRecDepth 8192 in
theorem root287Candidates8_checked : candidates edge 4 1889 3491 root287Tail9 winner.bound = root287Candidates8 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root287Step8 : scan edge (search edge 3) 4 [1889] 1889 root287Tail8 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail9 winner := by
  rw [root287Tail8, scan, if_neg (by decide), root287Candidates8_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail9
    (if 3 ≤ root287Candidates8.length then search edge 3 [1889, 3491] root287Candidates8 5380 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates9 : List Nat := [9941, 10181]
set_option maxRecDepth 8192 in
theorem root287Candidates9_checked : candidates edge 4 1889 4019 root287Tail10 winner.bound = root287Candidates9 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root287Step9 : scan edge (search edge 3) 4 [1889] 1889 root287Tail9 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail10 winner := by
  rw [root287Tail9, scan, if_neg (by decide), root287Candidates9_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail10
    (if 3 ≤ root287Candidates9.length then search edge 3 [1889, 4019] root287Candidates9 5908 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates10 : List Nat := [10781]
set_option maxRecDepth 8192 in
theorem root287Candidates10_checked : candidates edge 4 1889 4919 root287Tail11 winner.bound = root287Candidates10 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root287Step10 : scan edge (search edge 3) 4 [1889] 1889 root287Tail10 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail11 winner := by
  rw [root287Tail10, scan, if_neg (by decide), root287Candidates10_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail11
    (if 3 ≤ root287Candidates10.length then search edge 3 [1889, 4919] root287Candidates10 6808 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root287Candidates11_checked : candidates edge 4 1889 5297 root287Tail12 winner.bound = root287Candidates11 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root287Step11 : scan edge (search edge 3) 4 [1889] 1889 root287Tail11 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail12 winner := by
  rw [root287Tail11, scan, if_neg (by decide), root287Candidates11_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail12
    (if 3 ≤ root287Candidates11.length then search edge 3 [1889, 5297] root287Candidates11 7186 winner else winner) = _
  rw [if_neg (by decide)]
def root287Candidates12 : List Nat := [9533, 13187]
set_option maxRecDepth 8192 in
theorem root287Candidates12_checked : candidates edge 4 1889 5417 root287Tail13 winner.bound = root287Candidates12 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root287Step12 : scan edge (search edge 3) 4 [1889] 1889 root287Tail12 winner =
    scan edge (search edge 3) 4 [1889] 1889 root287Tail13 winner := by
  rw [root287Tail12, scan, if_neg (by decide), root287Candidates12_checked]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail13
    (if 3 ≤ root287Candidates12.length then search edge 3 [1889, 5417] root287Candidates12 7306 winner else winner) = _
  rw [if_neg (by decide)]
theorem root287Scan13 : scan edge (search edge 3) 4 [1889] 1889 root287Tail13 winner = winner := by rw [root287Tail13, scan, if_pos (by decide)]
theorem root287Scan12 : scan edge (search edge 3) 4 [1889] 1889 root287Tail12 winner = winner :=
  (root287Step12).trans root287Scan13
theorem root287Scan11 : scan edge (search edge 3) 4 [1889] 1889 root287Tail11 winner = winner :=
  (root287Step11).trans root287Scan12
theorem root287Scan10 : scan edge (search edge 3) 4 [1889] 1889 root287Tail10 winner = winner :=
  (root287Step10).trans root287Scan11
theorem root287Scan9 : scan edge (search edge 3) 4 [1889] 1889 root287Tail9 winner = winner :=
  (root287Step9).trans root287Scan10
theorem root287Scan8 : scan edge (search edge 3) 4 [1889] 1889 root287Tail8 winner = winner :=
  (root287Step8).trans root287Scan9
theorem root287Scan7 : scan edge (search edge 3) 4 [1889] 1889 root287Tail7 winner = winner :=
  (root287Step7).trans root287Scan8
theorem root287Scan6 : scan edge (search edge 3) 4 [1889] 1889 root287Tail6 winner = winner :=
  (root287Step6).trans root287Scan7
theorem root287Scan5 : scan edge (search edge 3) 4 [1889] 1889 root287Tail5 winner = winner :=
  (root287Step5).trans root287Scan6
theorem root287Scan4 : scan edge (search edge 3) 4 [1889] 1889 root287Tail4 winner = winner :=
  (root287Step4).trans root287Scan5
theorem root287Scan3 : scan edge (search edge 3) 4 [1889] 1889 root287Tail3 winner = winner :=
  (root287Step3).trans root287Scan4
theorem root287Scan2 : scan edge (search edge 3) 4 [1889] 1889 root287Tail2 winner = winner :=
  (root287Step2).trans root287Scan3
theorem root287Scan1 : scan edge (search edge 3) 4 [1889] 1889 root287Tail1 winner = winner :=
  (root287Step1).trans root287Scan2
theorem root287Scan0 : scan edge (search edge 3) 4 [1889] 1889 root287Tail0 winner = winner :=
  (root287Step0).trans root287Scan1
set_option maxRecDepth 4096 in
theorem root287Branch : search edge 4 [1889] root287Candidates 1889 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1889] 1889 root287Tail0 winner = winner
  exact root287Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
