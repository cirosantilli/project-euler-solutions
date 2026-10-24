import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root179Candidates : List Nat := Data.row1091
set_option maxRecDepth 8192 in
theorem root179Candidates_checked : candidates edge 5 0 1091 tail180 winner.bound = root179Candidates := by
  apply candidates_eq_row edge 5 0 1091 tail180 root179Candidates winner.bound tail180_ordered
  · decide
  · decide
  · funext q
    rfl
def root179Tail44 : List Nat := []
def root179Tail43 : List Nat := 22901 :: root179Tail44
def root179Tail42 : List Nat := 22721 :: root179Tail43
def root179Tail41 : List Nat := 22193 :: root179Tail42
def root179Tail40 : List Nat := 21839 :: root179Tail41
def root179Tail39 : List Nat := 21773 :: root179Tail40
def root179Tail38 : List Nat := 21557 :: root179Tail39
def root179Tail37 : List Nat := 20903 :: root179Tail38
def root179Tail36 : List Nat := 20261 :: root179Tail37
def root179Tail35 : List Nat := 19319 :: root179Tail36
def root179Tail34 : List Nat := 19121 :: root179Tail35
def root179Tail33 : List Nat := 19037 :: root179Tail34
def root179Tail32 : List Nat := 18749 :: root179Tail33
def root179Tail31 : List Nat := 18413 :: root179Tail32
def root179Tail30 : List Nat := 17909 :: root179Tail31
def root179Tail29 : List Nat := 17387 :: root179Tail30
def root179Tail28 : List Nat := 16871 :: root179Tail29
def root179Tail27 : List Nat := 15581 :: root179Tail28
def root179Tail26 : List Nat := 11903 :: root179Tail27
def root179Tail25 : List Nat := 11549 :: root179Tail26
def root179Tail24 : List Nat := 10181 :: root179Tail25
def root179Tail23 : List Nat := 9833 :: root179Tail24
def root179Tail22 : List Nat := 9803 :: root179Tail23
def root179Tail21 : List Nat := 9791 :: root179Tail22
def root179Tail20 : List Nat := 9479 :: root179Tail21
def root179Tail19 : List Nat := 9173 :: root179Tail20
def root179Tail18 : List Nat := 8933 :: root179Tail19
def root179Tail17 : List Nat := 8681 :: root179Tail18
def root179Tail16 : List Nat := 8543 :: root179Tail17
def root179Tail15 : List Nat := 8447 :: root179Tail16
def root179Tail14 : List Nat := 8111 :: root179Tail15
def root179Tail13 : List Nat := 8081 :: root179Tail14
def root179Tail12 : List Nat := 7307 :: root179Tail13
def root179Tail11 : List Nat := 4931 :: root179Tail12
def root179Tail10 : List Nat := 4919 :: root179Tail11
def root179Tail9 : List Nat := 4373 :: root179Tail10
def root179Tail8 : List Nat := 3527 :: root179Tail9
def root179Tail7 : List Nat := 3041 :: root179Tail8
def root179Tail6 : List Nat := 2633 :: root179Tail7
def root179Tail5 : List Nat := 2267 :: root179Tail6
def root179Tail4 : List Nat := 1949 :: root179Tail5
def root179Tail3 : List Nat := 1889 :: root179Tail4
def root179Tail2 : List Nat := 1847 :: root179Tail3
def root179Tail1 : List Nat := 1697 :: root179Tail2
def root179Tail0 : List Nat := 1319 :: root179Tail1
def root179Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root179Candidates0_checked : candidates edge 4 1091 1319 root179Tail1 winner.bound = root179Candidates0 := by
  unfold candidates
  have row : edge 1319 = (SortedNatTree.contains FastRows.row1319) := by
    change Data.row1319.contains = (SortedNatTree.contains FastRows.row1319)
    exact FastRows.row1319_checked.symm
  rw [row]
  decide
theorem root179Step0 : scan edge (search edge 3) 4 [1091] 1091 root179Tail0 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail1 winner := by
  rw [root179Tail0, scan, if_neg (by decide), root179Candidates0_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail1
    (if 3 ≤ root179Candidates0.length then search edge 3 [1091, 1319] root179Candidates0 2410 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates1 : List Nat := [3041, 3527, 4931, 22901]
set_option maxRecDepth 8192 in
theorem root179Candidates1_checked : candidates edge 4 1091 1697 root179Tail2 winner.bound = root179Candidates1 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root179Child1 : search edge 3 [1091, 1697] root179Candidates1 2788 winner = winner := by decide
theorem root179Step1 : scan edge (search edge 3) 4 [1091] 1091 root179Tail1 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail2 winner := by
  rw [root179Tail1, scan, if_neg (by decide), root179Candidates1_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail2
    (if 3 ≤ root179Candidates1.length then search edge 3 [1091, 1697] root179Candidates1 2788 winner else winner) = _
  rw [if_pos (by decide), root179Child1]
def root179Candidates2 : List Nat := [4373, 4919, 7307]
set_option maxRecDepth 8192 in
theorem root179Candidates2_checked : candidates edge 4 1091 1847 root179Tail3 winner.bound = root179Candidates2 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root179Child2 : search edge 3 [1091, 1847] root179Candidates2 2938 winner = winner := by decide
theorem root179Step2 : scan edge (search edge 3) 4 [1091] 1091 root179Tail2 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail3 winner := by
  rw [root179Tail2, scan, if_neg (by decide), root179Candidates2_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail3
    (if 3 ≤ root179Candidates2.length then search edge 3 [1091, 1847] root179Candidates2 2938 winner else winner) = _
  rw [if_pos (by decide), root179Child2]
def root179Candidates3 : List Nat := [2633, 4919, 10181, 22901]
set_option maxRecDepth 8192 in
theorem root179Candidates3_checked : candidates edge 4 1091 1889 root179Tail4 winner.bound = root179Candidates3 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root179Child3 : search edge 3 [1091, 1889] root179Candidates3 2980 winner = winner := by decide
theorem root179Step3 : scan edge (search edge 3) 4 [1091] 1091 root179Tail3 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail4 winner := by
  rw [root179Tail3, scan, if_neg (by decide), root179Candidates3_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail4
    (if 3 ≤ root179Candidates3.length then search edge 3 [1091, 1889] root179Candidates3 2980 winner else winner) = _
  rw [if_pos (by decide), root179Child3]
def root179Candidates4 : List Nat := [20261, 21773]
set_option maxRecDepth 8192 in
theorem root179Candidates4_checked : candidates edge 4 1091 1949 root179Tail5 winner.bound = root179Candidates4 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root179Step4 : scan edge (search edge 3) 4 [1091] 1091 root179Tail4 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail5 winner := by
  rw [root179Tail4, scan, if_neg (by decide), root179Candidates4_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail5
    (if 3 ≤ root179Candidates4.length then search edge 3 [1091, 1949] root179Candidates4 3040 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates5 : List Nat := [3527, 21839]
set_option maxRecDepth 8192 in
theorem root179Candidates5_checked : candidates edge 4 1091 2267 root179Tail6 winner.bound = root179Candidates5 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root179Step5 : scan edge (search edge 3) 4 [1091] 1091 root179Tail5 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail6 winner := by
  rw [root179Tail5, scan, if_neg (by decide), root179Candidates5_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail6
    (if 3 ≤ root179Candidates5.length then search edge 3 [1091, 2267] root179Candidates5 3358 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates6 : List Nat := [11549]
set_option maxRecDepth 8192 in
theorem root179Candidates6_checked : candidates edge 4 1091 2633 root179Tail7 winner.bound = root179Candidates6 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root179Step6 : scan edge (search edge 3) 4 [1091] 1091 root179Tail6 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail7 winner := by
  rw [root179Tail6, scan, if_neg (by decide), root179Candidates6_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail7
    (if 3 ≤ root179Candidates6.length then search edge 3 [1091, 2633] root179Candidates6 3724 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates7 : List Nat := [11903, 17909]
set_option maxRecDepth 8192 in
theorem root179Candidates7_checked : candidates edge 4 1091 3041 root179Tail8 winner.bound = root179Candidates7 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root179Step7 : scan edge (search edge 3) 4 [1091] 1091 root179Tail7 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail8 winner := by
  rw [root179Tail7, scan, if_neg (by decide), root179Candidates7_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail8
    (if 3 ≤ root179Candidates7.length then search edge 3 [1091, 3041] root179Candidates7 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root179Candidates8_checked : candidates edge 4 1091 3527 root179Tail9 winner.bound = root179Candidates8 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root179Step8 : scan edge (search edge 3) 4 [1091] 1091 root179Tail8 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail9 winner := by
  rw [root179Tail8, scan, if_neg (by decide), root179Candidates8_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail9
    (if 3 ≤ root179Candidates8.length then search edge 3 [1091, 3527] root179Candidates8 4618 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root179Candidates9_checked : candidates edge 4 1091 4373 root179Tail10 winner.bound = root179Candidates9 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root179Step9 : scan edge (search edge 3) 4 [1091] 1091 root179Tail9 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail10 winner := by
  rw [root179Tail9, scan, if_neg (by decide), root179Candidates9_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail10
    (if 3 ≤ root179Candidates9.length then search edge 3 [1091, 4373] root179Candidates9 5464 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root179Candidates10_checked : candidates edge 4 1091 4919 root179Tail11 winner.bound = root179Candidates10 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root179Step10 : scan edge (search edge 3) 4 [1091] 1091 root179Tail10 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail11 winner := by
  rw [root179Tail10, scan, if_neg (by decide), root179Candidates10_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail11
    (if 3 ≤ root179Candidates10.length then search edge 3 [1091, 4919] root179Candidates10 6010 winner else winner) = _
  rw [if_neg (by decide)]
def root179Candidates11 : List Nat := [9833]
set_option maxRecDepth 8192 in
theorem root179Candidates11_checked : candidates edge 4 1091 4931 root179Tail12 winner.bound = root179Candidates11 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root179Step11 : scan edge (search edge 3) 4 [1091] 1091 root179Tail11 winner =
    scan edge (search edge 3) 4 [1091] 1091 root179Tail12 winner := by
  rw [root179Tail11, scan, if_neg (by decide), root179Candidates11_checked]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail12
    (if 3 ≤ root179Candidates11.length then search edge 3 [1091, 4931] root179Candidates11 6022 winner else winner) = _
  rw [if_neg (by decide)]
theorem root179Scan12 : scan edge (search edge 3) 4 [1091] 1091 root179Tail12 winner = winner := by rw [root179Tail12, scan, if_pos (by decide)]
theorem root179Scan11 : scan edge (search edge 3) 4 [1091] 1091 root179Tail11 winner = winner :=
  (root179Step11).trans root179Scan12
theorem root179Scan10 : scan edge (search edge 3) 4 [1091] 1091 root179Tail10 winner = winner :=
  (root179Step10).trans root179Scan11
theorem root179Scan9 : scan edge (search edge 3) 4 [1091] 1091 root179Tail9 winner = winner :=
  (root179Step9).trans root179Scan10
theorem root179Scan8 : scan edge (search edge 3) 4 [1091] 1091 root179Tail8 winner = winner :=
  (root179Step8).trans root179Scan9
theorem root179Scan7 : scan edge (search edge 3) 4 [1091] 1091 root179Tail7 winner = winner :=
  (root179Step7).trans root179Scan8
theorem root179Scan6 : scan edge (search edge 3) 4 [1091] 1091 root179Tail6 winner = winner :=
  (root179Step6).trans root179Scan7
theorem root179Scan5 : scan edge (search edge 3) 4 [1091] 1091 root179Tail5 winner = winner :=
  (root179Step5).trans root179Scan6
theorem root179Scan4 : scan edge (search edge 3) 4 [1091] 1091 root179Tail4 winner = winner :=
  (root179Step4).trans root179Scan5
theorem root179Scan3 : scan edge (search edge 3) 4 [1091] 1091 root179Tail3 winner = winner :=
  (root179Step3).trans root179Scan4
theorem root179Scan2 : scan edge (search edge 3) 4 [1091] 1091 root179Tail2 winner = winner :=
  (root179Step2).trans root179Scan3
theorem root179Scan1 : scan edge (search edge 3) 4 [1091] 1091 root179Tail1 winner = winner :=
  (root179Step1).trans root179Scan2
theorem root179Scan0 : scan edge (search edge 3) 4 [1091] 1091 root179Tail0 winner = winner :=
  (root179Step0).trans root179Scan1
set_option maxRecDepth 4096 in
theorem root179Branch : search edge 4 [1091] root179Candidates 1091 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1091] 1091 root179Tail0 winner = winner
  exact root179Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
