import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root236Candidates : List Nat := Data.row1499
set_option maxRecDepth 8192 in
theorem root236Candidates_checked : candidates edge 5 0 1499 tail237 winner.bound = root236Candidates := by
  apply candidates_eq_row edge 5 0 1499 tail237 root236Candidates winner.bound tail237_ordered
  · decide
  · decide
  · funext q
    rfl
def root236Tail35 : List Nat := []
def root236Tail34 : List Nat := 23831 :: root236Tail35
def root236Tail33 : List Nat := 23369 :: root236Tail34
def root236Tail32 : List Nat := 22709 :: root236Tail33
def root236Tail31 : List Nat := 22697 :: root236Tail32
def root236Tail30 : List Nat := 21839 :: root236Tail31
def root236Tail29 : List Nat := 20873 :: root236Tail30
def root236Tail28 : List Nat := 20063 :: root236Tail29
def root236Tail27 : List Nat := 19793 :: root236Tail28
def root236Tail26 : List Nat := 19403 :: root236Tail27
def root236Tail25 : List Nat := 17903 :: root236Tail26
def root236Tail24 : List Nat := 14657 :: root236Tail25
def root236Tail23 : List Nat := 11369 :: root236Tail24
def root236Tail22 : List Nat := 10487 :: root236Tail23
def root236Tail21 : List Nat := 10223 :: root236Tail22
def root236Tail20 : List Nat := 9221 :: root236Tail21
def root236Tail19 : List Nat := 8693 :: root236Tail20
def root236Tail18 : List Nat := 8273 :: root236Tail19
def root236Tail17 : List Nat := 8171 :: root236Tail18
def root236Tail16 : List Nat := 7589 :: root236Tail17
def root236Tail15 : List Nat := 7331 :: root236Tail16
def root236Tail14 : List Nat := 7121 :: root236Tail15
def root236Tail13 : List Nat := 6563 :: root236Tail14
def root236Tail12 : List Nat := 6521 :: root236Tail13
def root236Tail11 : List Nat := 6323 :: root236Tail12
def root236Tail10 : List Nat := 4253 :: root236Tail11
def root236Tail9 : List Nat := 4013 :: root236Tail10
def root236Tail8 : List Nat := 3947 :: root236Tail9
def root236Tail7 : List Nat := 3359 :: root236Tail8
def root236Tail6 : List Nat := 2897 :: root236Tail7
def root236Tail5 : List Nat := 2801 :: root236Tail6
def root236Tail4 : List Nat := 2633 :: root236Tail5
def root236Tail3 : List Nat := 2543 :: root236Tail4
def root236Tail2 : List Nat := 2309 :: root236Tail3
def root236Tail1 : List Nat := 2129 :: root236Tail2
def root236Tail0 : List Nat := 1619 :: root236Tail1
def root236Candidates0 : List Nat := [7331]
set_option maxRecDepth 8192 in
theorem root236Candidates0_checked : candidates edge 4 1499 1619 root236Tail1 winner.bound = root236Candidates0 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
theorem root236Step0 : scan edge (search edge 3) 4 [1499] 1499 root236Tail0 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail1 winner := by
  rw [root236Tail0, scan, if_neg (by decide), root236Candidates0_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail1
    (if 3 ≤ root236Candidates0.length then search edge 3 [1499, 1619] root236Candidates0 3118 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates1 : List Nat := [3947, 7121]
set_option maxRecDepth 8192 in
theorem root236Candidates1_checked : candidates edge 4 1499 2129 root236Tail2 winner.bound = root236Candidates1 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
theorem root236Step1 : scan edge (search edge 3) 4 [1499] 1499 root236Tail1 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail2 winner := by
  rw [root236Tail1, scan, if_neg (by decide), root236Candidates1_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail2
    (if 3 ≤ root236Candidates1.length then search edge 3 [1499, 2129] root236Candidates1 3628 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root236Candidates2_checked : candidates edge 4 1499 2309 root236Tail3 winner.bound = root236Candidates2 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
theorem root236Step2 : scan edge (search edge 3) 4 [1499] 1499 root236Tail2 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail3 winner := by
  rw [root236Tail2, scan, if_neg (by decide), root236Candidates2_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail3
    (if 3 ≤ root236Candidates2.length then search edge 3 [1499, 2309] root236Candidates2 3808 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates3 : List Nat := [7331, 10487]
set_option maxRecDepth 8192 in
theorem root236Candidates3_checked : candidates edge 4 1499 2543 root236Tail4 winner.bound = root236Candidates3 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
theorem root236Step3 : scan edge (search edge 3) 4 [1499] 1499 root236Tail3 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail4 winner := by
  rw [root236Tail3, scan, if_neg (by decide), root236Candidates3_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail4
    (if 3 ≤ root236Candidates3.length then search edge 3 [1499, 2543] root236Candidates3 4042 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root236Candidates4_checked : candidates edge 4 1499 2633 root236Tail5 winner.bound = root236Candidates4 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root236Step4 : scan edge (search edge 3) 4 [1499] 1499 root236Tail4 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail5 winner := by
  rw [root236Tail4, scan, if_neg (by decide), root236Candidates4_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail5
    (if 3 ≤ root236Candidates4.length then search edge 3 [1499, 2633] root236Candidates4 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates5 : List Nat := [7331, 8171, 11369, 19403]
set_option maxRecDepth 8192 in
theorem root236Candidates5_checked : candidates edge 4 1499 2801 root236Tail6 winner.bound = root236Candidates5 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root236Child5 : search edge 3 [1499, 2801] root236Candidates5 4300 winner = winner := by decide
theorem root236Step5 : scan edge (search edge 3) 4 [1499] 1499 root236Tail5 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail6 winner := by
  rw [root236Tail5, scan, if_neg (by decide), root236Candidates5_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail6
    (if 3 ≤ root236Candidates5.length then search edge 3 [1499, 2801] root236Candidates5 4300 winner else winner) = _
  rw [if_pos (by decide), root236Child5]
def root236Candidates6 : List Nat := [3947, 19403]
set_option maxRecDepth 8192 in
theorem root236Candidates6_checked : candidates edge 4 1499 2897 root236Tail7 winner.bound = root236Candidates6 := by
  unfold candidates
  have row : edge 2897 = (SortedNatTree.contains FastRows.row2897) := by
    change Data.row2897.contains = (SortedNatTree.contains FastRows.row2897)
    exact FastRows.row2897_checked.symm
  rw [row]
  decide
theorem root236Step6 : scan edge (search edge 3) 4 [1499] 1499 root236Tail6 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail7 winner := by
  rw [root236Tail6, scan, if_neg (by decide), root236Candidates6_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail7
    (if 3 ≤ root236Candidates6.length then search edge 3 [1499, 2897] root236Candidates6 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root236Candidates7_checked : candidates edge 4 1499 3359 root236Tail8 winner.bound = root236Candidates7 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root236Step7 : scan edge (search edge 3) 4 [1499] 1499 root236Tail7 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail8 winner := by
  rw [root236Tail7, scan, if_neg (by decide), root236Candidates7_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail8
    (if 3 ≤ root236Candidates7.length then search edge 3 [1499, 3359] root236Candidates7 4858 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root236Candidates8_checked : candidates edge 4 1499 3947 root236Tail9 winner.bound = root236Candidates8 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root236Step8 : scan edge (search edge 3) 4 [1499] 1499 root236Tail8 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail9 winner := by
  rw [root236Tail8, scan, if_neg (by decide), root236Candidates8_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail9
    (if 3 ≤ root236Candidates8.length then search edge 3 [1499, 3947] root236Candidates8 5446 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates9 : List Nat := [19403]
set_option maxRecDepth 8192 in
theorem root236Candidates9_checked : candidates edge 4 1499 4013 root236Tail10 winner.bound = root236Candidates9 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root236Step9 : scan edge (search edge 3) 4 [1499] 1499 root236Tail9 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail10 winner := by
  rw [root236Tail9, scan, if_neg (by decide), root236Candidates9_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail10
    (if 3 ≤ root236Candidates9.length then search edge 3 [1499, 4013] root236Candidates9 5512 winner else winner) = _
  rw [if_neg (by decide)]
def root236Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root236Candidates10_checked : candidates edge 4 1499 4253 root236Tail11 winner.bound = root236Candidates10 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root236Step10 : scan edge (search edge 3) 4 [1499] 1499 root236Tail10 winner =
    scan edge (search edge 3) 4 [1499] 1499 root236Tail11 winner := by
  rw [root236Tail10, scan, if_neg (by decide), root236Candidates10_checked]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail11
    (if 3 ≤ root236Candidates10.length then search edge 3 [1499, 4253] root236Candidates10 5752 winner else winner) = _
  rw [if_neg (by decide)]
theorem root236Scan11 : scan edge (search edge 3) 4 [1499] 1499 root236Tail11 winner = winner := by rw [root236Tail11, scan, if_pos (by decide)]
theorem root236Scan10 : scan edge (search edge 3) 4 [1499] 1499 root236Tail10 winner = winner :=
  (root236Step10).trans root236Scan11
theorem root236Scan9 : scan edge (search edge 3) 4 [1499] 1499 root236Tail9 winner = winner :=
  (root236Step9).trans root236Scan10
theorem root236Scan8 : scan edge (search edge 3) 4 [1499] 1499 root236Tail8 winner = winner :=
  (root236Step8).trans root236Scan9
theorem root236Scan7 : scan edge (search edge 3) 4 [1499] 1499 root236Tail7 winner = winner :=
  (root236Step7).trans root236Scan8
theorem root236Scan6 : scan edge (search edge 3) 4 [1499] 1499 root236Tail6 winner = winner :=
  (root236Step6).trans root236Scan7
theorem root236Scan5 : scan edge (search edge 3) 4 [1499] 1499 root236Tail5 winner = winner :=
  (root236Step5).trans root236Scan6
theorem root236Scan4 : scan edge (search edge 3) 4 [1499] 1499 root236Tail4 winner = winner :=
  (root236Step4).trans root236Scan5
theorem root236Scan3 : scan edge (search edge 3) 4 [1499] 1499 root236Tail3 winner = winner :=
  (root236Step3).trans root236Scan4
theorem root236Scan2 : scan edge (search edge 3) 4 [1499] 1499 root236Tail2 winner = winner :=
  (root236Step2).trans root236Scan3
theorem root236Scan1 : scan edge (search edge 3) 4 [1499] 1499 root236Tail1 winner = winner :=
  (root236Step1).trans root236Scan2
theorem root236Scan0 : scan edge (search edge 3) 4 [1499] 1499 root236Tail0 winner = winner :=
  (root236Step0).trans root236Scan1
set_option maxRecDepth 4096 in
theorem root236Branch : search edge 4 [1499] root236Candidates 1499 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1499] 1499 root236Tail0 winner = winner
  exact root236Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
