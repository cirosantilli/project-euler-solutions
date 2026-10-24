import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root194Candidates : List Nat := Data.row1201
set_option maxRecDepth 8192 in
theorem root194Candidates_checked : candidates edge 5 0 1201 tail195 winner.bound = root194Candidates := by
  apply candidates_eq_row edge 5 0 1201 tail195 root194Candidates winner.bound tail195_ordered
  · decide
  · decide
  · funext q
    rfl
def root194Tail50 : List Nat := []
def root194Tail49 : List Nat := 24481 :: root194Tail50
def root194Tail48 : List Nat := 24007 :: root194Tail49
def root194Tail47 : List Nat := 23473 :: root194Tail48
def root194Tail46 : List Nat := 21493 :: root194Tail47
def root194Tail45 : List Nat := 21247 :: root194Tail46
def root194Tail44 : List Nat := 21163 :: root194Tail45
def root194Tail43 : List Nat := 21061 :: root194Tail44
def root194Tail42 : List Nat := 20899 :: root194Tail43
def root194Tail41 : List Nat := 20431 :: root194Tail42
def root194Tail40 : List Nat := 19861 :: root194Tail41
def root194Tail39 : List Nat := 19687 :: root194Tail40
def root194Tail38 : List Nat := 19471 :: root194Tail39
def root194Tail37 : List Nat := 18919 :: root194Tail38
def root194Tail36 : List Nat := 17509 :: root194Tail37
def root194Tail35 : List Nat := 17497 :: root194Tail36
def root194Tail34 : List Nat := 17077 :: root194Tail35
def root194Tail33 : List Nat := 16417 :: root194Tail34
def root194Tail32 : List Nat := 15817 :: root194Tail33
def root194Tail31 : List Nat := 15349 :: root194Tail32
def root194Tail30 : List Nat := 14431 :: root194Tail31
def root194Tail29 : List Nat := 13183 :: root194Tail30
def root194Tail28 : List Nat := 12763 :: root194Tail29
def root194Tail27 : List Nat := 12757 :: root194Tail28
def root194Tail26 : List Nat := 11821 :: root194Tail27
def root194Tail25 : List Nat := 11071 :: root194Tail26
def root194Tail24 : List Nat := 9781 :: root194Tail25
def root194Tail23 : List Nat := 9619 :: root194Tail24
def root194Tail22 : List Nat := 9349 :: root194Tail23
def root194Tail21 : List Nat := 8641 :: root194Tail22
def root194Tail20 : List Nat := 7459 :: root194Tail21
def root194Tail19 : List Nat := 7207 :: root194Tail20
def root194Tail18 : List Nat := 6577 :: root194Tail19
def root194Tail17 : List Nat := 6337 :: root194Tail18
def root194Tail16 : List Nat := 5953 :: root194Tail17
def root194Tail15 : List Nat := 5281 :: root194Tail16
def root194Tail14 : List Nat := 5167 :: root194Tail15
def root194Tail13 : List Nat := 4861 :: root194Tail14
def root194Tail12 : List Nat := 4663 :: root194Tail13
def root194Tail11 : List Nat := 4201 :: root194Tail12
def root194Tail10 : List Nat := 3931 :: root194Tail11
def root194Tail9 : List Nat := 3769 :: root194Tail10
def root194Tail8 : List Nat := 3637 :: root194Tail9
def root194Tail7 : List Nat := 3229 :: root194Tail8
def root194Tail6 : List Nat := 2551 :: root194Tail7
def root194Tail5 : List Nat := 2137 :: root194Tail6
def root194Tail4 : List Nat := 2017 :: root194Tail5
def root194Tail3 : List Nat := 1873 :: root194Tail4
def root194Tail2 : List Nat := 1621 :: root194Tail3
def root194Tail1 : List Nat := 1459 :: root194Tail2
def root194Tail0 : List Nat := 1303 :: root194Tail1
def root194Candidates0 : List Nat := [7207, 12757, 20899]
set_option maxRecDepth 8192 in
theorem root194Candidates0_checked : candidates edge 4 1201 1303 root194Tail1 winner.bound = root194Candidates0 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root194Child0 : search edge 3 [1201, 1303] root194Candidates0 2504 winner = winner := by decide
theorem root194Step0 : scan edge (search edge 3) 4 [1201] 1201 root194Tail0 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail1 winner := by
  rw [root194Tail0, scan, if_neg (by decide), root194Candidates0_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail1
    (if 3 ≤ root194Candidates0.length then search edge 3 [1201, 1303] root194Candidates0 2504 winner else winner) = _
  rw [if_pos (by decide), root194Child0]
def root194Candidates1 : List Nat := [1873, 15817, 17509, 18919, 20899]
set_option maxRecDepth 8192 in
theorem root194Candidates1_checked : candidates edge 4 1201 1459 root194Tail2 winner.bound = root194Candidates1 := by
  unfold candidates
  have row : edge 1459 = (SortedNatTree.contains FastRows.row1459) := by
    change Data.row1459.contains = (SortedNatTree.contains FastRows.row1459)
    exact FastRows.row1459_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root194Child1 : search edge 3 [1201, 1459] root194Candidates1 2660 winner = winner := by decide
theorem root194Step1 : scan edge (search edge 3) 4 [1201] 1201 root194Tail1 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail2 winner := by
  rw [root194Tail1, scan, if_neg (by decide), root194Candidates1_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail2
    (if 3 ≤ root194Candidates1.length then search edge 3 [1201, 1459] root194Candidates1 2660 winner else winner) = _
  rw [if_pos (by decide), root194Child1]
def root194Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root194Candidates2_checked : candidates edge 4 1201 1621 root194Tail3 winner.bound = root194Candidates2 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
theorem root194Step2 : scan edge (search edge 3) 4 [1201] 1201 root194Tail2 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail3 winner := by
  rw [root194Tail2, scan, if_neg (by decide), root194Candidates2_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail3
    (if 3 ≤ root194Candidates2.length then search edge 3 [1201, 1621] root194Candidates2 2822 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates3 : List Nat := [2017]
set_option maxRecDepth 8192 in
theorem root194Candidates3_checked : candidates edge 4 1201 1873 root194Tail4 winner.bound = root194Candidates3 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root194Step3 : scan edge (search edge 3) 4 [1201] 1201 root194Tail3 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail4 winner := by
  rw [root194Tail3, scan, if_neg (by decide), root194Candidates3_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail4
    (if 3 ≤ root194Candidates3.length then search edge 3 [1201, 1873] root194Candidates3 3074 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root194Candidates4_checked : candidates edge 4 1201 2017 root194Tail5 winner.bound = root194Candidates4 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root194Step4 : scan edge (search edge 3) 4 [1201] 1201 root194Tail4 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail5 winner := by
  rw [root194Tail4, scan, if_neg (by decide), root194Candidates4_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail5
    (if 3 ≤ root194Candidates4.length then search edge 3 [1201, 2017] root194Candidates4 3218 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates5 : List Nat := [7459]
set_option maxRecDepth 8192 in
theorem root194Candidates5_checked : candidates edge 4 1201 2137 root194Tail6 winner.bound = root194Candidates5 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
theorem root194Step5 : scan edge (search edge 3) 4 [1201] 1201 root194Tail5 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail6 winner := by
  rw [root194Tail5, scan, if_neg (by decide), root194Candidates5_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail6
    (if 3 ≤ root194Candidates5.length then search edge 3 [1201, 2137] root194Candidates5 3338 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates6 : List Nat := [12763]
set_option maxRecDepth 8192 in
theorem root194Candidates6_checked : candidates edge 4 1201 2551 root194Tail7 winner.bound = root194Candidates6 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root194Step6 : scan edge (search edge 3) 4 [1201] 1201 root194Tail6 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail7 winner := by
  rw [root194Tail6, scan, if_neg (by decide), root194Candidates6_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail7
    (if 3 ≤ root194Candidates6.length then search edge 3 [1201, 2551] root194Candidates6 3752 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates7 : List Nat := [17509]
set_option maxRecDepth 8192 in
theorem root194Candidates7_checked : candidates edge 4 1201 3229 root194Tail8 winner.bound = root194Candidates7 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root194Step7 : scan edge (search edge 3) 4 [1201] 1201 root194Tail7 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail8 winner := by
  rw [root194Tail7, scan, if_neg (by decide), root194Candidates7_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail8
    (if 3 ≤ root194Candidates7.length then search edge 3 [1201, 3229] root194Candidates7 4430 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root194Candidates8_checked : candidates edge 4 1201 3637 root194Tail9 winner.bound = root194Candidates8 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root194Step8 : scan edge (search edge 3) 4 [1201] 1201 root194Tail8 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail9 winner := by
  rw [root194Tail8, scan, if_neg (by decide), root194Candidates8_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail9
    (if 3 ≤ root194Candidates8.length then search edge 3 [1201, 3637] root194Candidates8 4838 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates9 : List Nat := [19687]
set_option maxRecDepth 8192 in
theorem root194Candidates9_checked : candidates edge 4 1201 3769 root194Tail10 winner.bound = root194Candidates9 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root194Step9 : scan edge (search edge 3) 4 [1201] 1201 root194Tail9 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail10 winner := by
  rw [root194Tail9, scan, if_neg (by decide), root194Candidates9_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail10
    (if 3 ≤ root194Candidates9.length then search edge 3 [1201, 3769] root194Candidates9 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates10 : List Nat := [7459, 9619, 19471]
set_option maxRecDepth 8192 in
theorem root194Candidates10_checked : candidates edge 4 1201 3931 root194Tail11 winner.bound = root194Candidates10 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root194Child10 : search edge 3 [1201, 3931] root194Candidates10 5132 winner = winner := by decide
theorem root194Step10 : scan edge (search edge 3) 4 [1201] 1201 root194Tail10 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail11 winner := by
  rw [root194Tail10, scan, if_neg (by decide), root194Candidates10_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail11
    (if 3 ≤ root194Candidates10.length then search edge 3 [1201, 3931] root194Candidates10 5132 winner else winner) = _
  rw [if_pos (by decide), root194Child10]
def root194Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root194Candidates11_checked : candidates edge 4 1201 4201 root194Tail12 winner.bound = root194Candidates11 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root194Step11 : scan edge (search edge 3) 4 [1201] 1201 root194Tail11 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail12 winner := by
  rw [root194Tail11, scan, if_neg (by decide), root194Candidates11_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail12
    (if 3 ≤ root194Candidates11.length then search edge 3 [1201, 4201] root194Candidates11 5402 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates12 : List Nat := [17077]
set_option maxRecDepth 8192 in
theorem root194Candidates12_checked : candidates edge 4 1201 4663 root194Tail13 winner.bound = root194Candidates12 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root194Step12 : scan edge (search edge 3) 4 [1201] 1201 root194Tail12 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail13 winner := by
  rw [root194Tail12, scan, if_neg (by decide), root194Candidates12_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail13
    (if 3 ≤ root194Candidates12.length then search edge 3 [1201, 4663] root194Candidates12 5864 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates13 : List Nat := [5953]
set_option maxRecDepth 8192 in
theorem root194Candidates13_checked : candidates edge 4 1201 4861 root194Tail14 winner.bound = root194Candidates13 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root194Step13 : scan edge (search edge 3) 4 [1201] 1201 root194Tail13 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail14 winner := by
  rw [root194Tail13, scan, if_neg (by decide), root194Candidates13_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail14
    (if 3 ≤ root194Candidates13.length then search edge 3 [1201, 4861] root194Candidates13 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates14 : List Nat := [9619]
set_option maxRecDepth 8192 in
theorem root194Candidates14_checked : candidates edge 4 1201 5167 root194Tail15 winner.bound = root194Candidates14 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root194Step14 : scan edge (search edge 3) 4 [1201] 1201 root194Tail14 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail15 winner := by
  rw [root194Tail14, scan, if_neg (by decide), root194Candidates14_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail15
    (if 3 ≤ root194Candidates14.length then search edge 3 [1201, 5167] root194Candidates14 6368 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root194Candidates15_checked : candidates edge 4 1201 5281 root194Tail16 winner.bound = root194Candidates15 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root194Step15 : scan edge (search edge 3) 4 [1201] 1201 root194Tail15 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail16 winner := by
  rw [root194Tail15, scan, if_neg (by decide), root194Candidates15_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail16
    (if 3 ≤ root194Candidates15.length then search edge 3 [1201, 5281] root194Candidates15 6482 winner else winner) = _
  rw [if_neg (by decide)]
def root194Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root194Candidates16_checked : candidates edge 4 1201 5953 root194Tail17 winner.bound = root194Candidates16 := by
  unfold candidates
  have row : edge 5953 = (SortedNatTree.contains FastRows.row5953) := by
    change Data.row5953.contains = (SortedNatTree.contains FastRows.row5953)
    exact FastRows.row5953_checked.symm
  rw [row]
  decide
theorem root194Step16 : scan edge (search edge 3) 4 [1201] 1201 root194Tail16 winner =
    scan edge (search edge 3) 4 [1201] 1201 root194Tail17 winner := by
  rw [root194Tail16, scan, if_neg (by decide), root194Candidates16_checked]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail17
    (if 3 ≤ root194Candidates16.length then search edge 3 [1201, 5953] root194Candidates16 7154 winner else winner) = _
  rw [if_neg (by decide)]
theorem root194Scan17 : scan edge (search edge 3) 4 [1201] 1201 root194Tail17 winner = winner := by rw [root194Tail17, scan, if_pos (by decide)]
theorem root194Scan16 : scan edge (search edge 3) 4 [1201] 1201 root194Tail16 winner = winner :=
  (root194Step16).trans root194Scan17
theorem root194Scan15 : scan edge (search edge 3) 4 [1201] 1201 root194Tail15 winner = winner :=
  (root194Step15).trans root194Scan16
theorem root194Scan14 : scan edge (search edge 3) 4 [1201] 1201 root194Tail14 winner = winner :=
  (root194Step14).trans root194Scan15
theorem root194Scan13 : scan edge (search edge 3) 4 [1201] 1201 root194Tail13 winner = winner :=
  (root194Step13).trans root194Scan14
theorem root194Scan12 : scan edge (search edge 3) 4 [1201] 1201 root194Tail12 winner = winner :=
  (root194Step12).trans root194Scan13
theorem root194Scan11 : scan edge (search edge 3) 4 [1201] 1201 root194Tail11 winner = winner :=
  (root194Step11).trans root194Scan12
theorem root194Scan10 : scan edge (search edge 3) 4 [1201] 1201 root194Tail10 winner = winner :=
  (root194Step10).trans root194Scan11
theorem root194Scan9 : scan edge (search edge 3) 4 [1201] 1201 root194Tail9 winner = winner :=
  (root194Step9).trans root194Scan10
theorem root194Scan8 : scan edge (search edge 3) 4 [1201] 1201 root194Tail8 winner = winner :=
  (root194Step8).trans root194Scan9
theorem root194Scan7 : scan edge (search edge 3) 4 [1201] 1201 root194Tail7 winner = winner :=
  (root194Step7).trans root194Scan8
theorem root194Scan6 : scan edge (search edge 3) 4 [1201] 1201 root194Tail6 winner = winner :=
  (root194Step6).trans root194Scan7
theorem root194Scan5 : scan edge (search edge 3) 4 [1201] 1201 root194Tail5 winner = winner :=
  (root194Step5).trans root194Scan6
theorem root194Scan4 : scan edge (search edge 3) 4 [1201] 1201 root194Tail4 winner = winner :=
  (root194Step4).trans root194Scan5
theorem root194Scan3 : scan edge (search edge 3) 4 [1201] 1201 root194Tail3 winner = winner :=
  (root194Step3).trans root194Scan4
theorem root194Scan2 : scan edge (search edge 3) 4 [1201] 1201 root194Tail2 winner = winner :=
  (root194Step2).trans root194Scan3
theorem root194Scan1 : scan edge (search edge 3) 4 [1201] 1201 root194Tail1 winner = winner :=
  (root194Step1).trans root194Scan2
theorem root194Scan0 : scan edge (search edge 3) 4 [1201] 1201 root194Tail0 winner = winner :=
  (root194Step0).trans root194Scan1
set_option maxRecDepth 4096 in
theorem root194Branch : search edge 4 [1201] root194Candidates 1201 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1201] 1201 root194Tail0 winner = winner
  exact root194Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
