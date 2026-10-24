import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root103Candidates : List Nat := Data.row577
set_option maxRecDepth 8192 in
theorem root103Candidates_checked : candidates edge 5 0 577 tail104 winner.bound = root103Candidates := by
  apply candidates_eq_row edge 5 0 577 tail104 root103Candidates winner.bound tail104_ordered
  · decide
  · decide
  · funext q
    rfl
def root103Tail64 : List Nat := []
def root103Tail63 : List Nat := 25453 :: root103Tail64
def root103Tail62 : List Nat := 25117 :: root103Tail63
def root103Tail61 : List Nat := 24781 :: root103Tail62
def root103Tail60 : List Nat := 24379 :: root103Tail61
def root103Tail59 : List Nat := 24019 :: root103Tail60
def root103Tail58 : List Nat := 23563 :: root103Tail59
def root103Tail57 : List Nat := 23011 :: root103Tail58
def root103Tail56 : List Nat := 22129 :: root103Tail57
def root103Tail55 : List Nat := 21673 :: root103Tail56
def root103Tail54 : List Nat := 21211 :: root103Tail55
def root103Tail53 : List Nat := 20431 :: root103Tail54
def root103Tail52 : List Nat := 20101 :: root103Tail53
def root103Tail51 : List Nat := 19993 :: root103Tail52
def root103Tail50 : List Nat := 19819 :: root103Tail51
def root103Tail49 : List Nat := 19507 :: root103Tail50
def root103Tail48 : List Nat := 19231 :: root103Tail49
def root103Tail47 : List Nat := 18097 :: root103Tail48
def root103Tail46 : List Nat := 17203 :: root103Tail47
def root103Tail45 : List Nat := 16993 :: root103Tail46
def root103Tail44 : List Nat := 16657 :: root103Tail45
def root103Tail43 : List Nat := 16381 :: root103Tail44
def root103Tail42 : List Nat := 16273 :: root103Tail43
def root103Tail41 : List Nat := 15661 :: root103Tail42
def root103Tail40 : List Nat := 15643 :: root103Tail41
def root103Tail39 : List Nat := 15271 :: root103Tail40
def root103Tail38 : List Nat := 15121 :: root103Tail39
def root103Tail37 : List Nat := 14947 :: root103Tail38
def root103Tail36 : List Nat := 14887 :: root103Tail37
def root103Tail35 : List Nat := 14323 :: root103Tail36
def root103Tail34 : List Nat := 13477 :: root103Tail35
def root103Tail33 : List Nat := 12583 :: root103Tail34
def root103Tail32 : List Nat := 12289 :: root103Tail33
def root103Tail31 : List Nat := 12157 :: root103Tail32
def root103Tail30 : List Nat := 12007 :: root103Tail31
def root103Tail29 : List Nat := 11779 :: root103Tail30
def root103Tail28 : List Nat := 11047 :: root103Tail29
def root103Tail27 : List Nat := 10663 :: root103Tail28
def root103Tail26 : List Nat := 10567 :: root103Tail27
def root103Tail25 : List Nat := 10267 :: root103Tail26
def root103Tail24 : List Nat := 10099 :: root103Tail25
def root103Tail23 : List Nat := 10039 :: root103Tail24
def root103Tail22 : List Nat := 9343 :: root103Tail23
def root103Tail21 : List Nat := 8737 :: root103Tail22
def root103Tail20 : List Nat := 8329 :: root103Tail21
def root103Tail19 : List Nat := 7963 :: root103Tail20
def root103Tail18 : List Nat := 7951 :: root103Tail19
def root103Tail17 : List Nat := 7459 :: root103Tail18
def root103Tail16 : List Nat := 6949 :: root103Tail17
def root103Tail15 : List Nat := 5839 :: root103Tail16
def root103Tail14 : List Nat := 5821 :: root103Tail15
def root103Tail13 : List Nat := 5683 :: root103Tail14
def root103Tail12 : List Nat := 5431 :: root103Tail13
def root103Tail11 : List Nat := 4759 :: root103Tail12
def root103Tail10 : List Nat := 4597 :: root103Tail11
def root103Tail9 : List Nat := 3919 :: root103Tail10
def root103Tail8 : List Nat := 3709 :: root103Tail9
def root103Tail7 : List Nat := 3331 :: root103Tail8
def root103Tail6 : List Nat := 2617 :: root103Tail7
def root103Tail5 : List Nat := 2539 :: root103Tail6
def root103Tail4 : List Nat := 1723 :: root103Tail5
def root103Tail3 : List Nat := 1327 :: root103Tail4
def root103Tail2 : List Nat := 937 :: root103Tail3
def root103Tail1 : List Nat := 757 :: root103Tail2
def root103Tail0 : List Nat := 613 :: root103Tail1
def root103Candidates0 : List Nat := [15643, 16273, 23011, 24379]
set_option maxRecDepth 8192 in
theorem root103Candidates0_checked : candidates edge 4 577 613 root103Tail1 winner.bound = root103Candidates0 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child0 : search edge 3 [577, 613] root103Candidates0 1190 winner = winner := by decide
theorem root103Step0 : scan edge (search edge 3) 4 [577] 577 root103Tail0 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail1 winner := by
  rw [root103Tail0, scan, if_neg (by decide), root103Candidates0_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail1
    (if 3 ≤ root103Candidates0.length then search edge 3 [577, 613] root103Candidates0 1190 winner else winner) = _
  rw [if_pos (by decide), root103Child0]
def root103Candidates1 : List Nat := [10567]
set_option maxRecDepth 8192 in
theorem root103Candidates1_checked : candidates edge 4 577 757 root103Tail2 winner.bound = root103Candidates1 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
theorem root103Step1 : scan edge (search edge 3) 4 [577] 577 root103Tail1 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail2 winner := by
  rw [root103Tail1, scan, if_neg (by decide), root103Candidates1_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail2
    (if 3 ≤ root103Candidates1.length then search edge 3 [577, 757] root103Candidates1 1334 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root103Candidates2_checked : candidates edge 4 577 937 root103Tail3 winner.bound = root103Candidates2 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
theorem root103Step2 : scan edge (search edge 3) 4 [577] 577 root103Tail2 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail3 winner := by
  rw [root103Tail2, scan, if_neg (by decide), root103Candidates2_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail3
    (if 3 ≤ root103Candidates2.length then search edge 3 [577, 937] root103Candidates2 1514 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates3 : List Nat := [1723, 7963, 8737]
set_option maxRecDepth 8192 in
theorem root103Candidates3_checked : candidates edge 4 577 1327 root103Tail4 winner.bound = root103Candidates3 := by
  unfold candidates
  have row : edge 1327 = (SortedNatTree.contains FastRows.row1327) := by
    change Data.row1327.contains = (SortedNatTree.contains FastRows.row1327)
    exact FastRows.row1327_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child3 : search edge 3 [577, 1327] root103Candidates3 1904 winner = winner := by decide
theorem root103Step3 : scan edge (search edge 3) 4 [577] 577 root103Tail3 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail4 winner := by
  rw [root103Tail3, scan, if_neg (by decide), root103Candidates3_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail4
    (if 3 ≤ root103Candidates3.length then search edge 3 [577, 1327] root103Candidates3 1904 winner else winner) = _
  rw [if_pos (by decide), root103Child3]
def root103Candidates4 : List Nat := [5431, 7951, 8329, 15271]
set_option maxRecDepth 8192 in
theorem root103Candidates4_checked : candidates edge 4 577 1723 root103Tail5 winner.bound = root103Candidates4 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child4 : search edge 3 [577, 1723] root103Candidates4 2300 winner = winner := by decide
theorem root103Step4 : scan edge (search edge 3) 4 [577] 577 root103Tail4 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail5 winner := by
  rw [root103Tail4, scan, if_neg (by decide), root103Candidates4_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail5
    (if 3 ≤ root103Candidates4.length then search edge 3 [577, 1723] root103Candidates4 2300 winner else winner) = _
  rw [if_pos (by decide), root103Child4]
def root103Candidates5 : List Nat := [7951, 15643, 15661]
set_option maxRecDepth 8192 in
theorem root103Candidates5_checked : candidates edge 4 577 2539 root103Tail6 winner.bound = root103Candidates5 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child5 : search edge 3 [577, 2539] root103Candidates5 3116 winner = winner := by decide
theorem root103Step5 : scan edge (search edge 3) 4 [577] 577 root103Tail5 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail6 winner := by
  rw [root103Tail5, scan, if_neg (by decide), root103Candidates5_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail6
    (if 3 ≤ root103Candidates5.length then search edge 3 [577, 2539] root103Candidates5 3116 winner else winner) = _
  rw [if_pos (by decide), root103Child5]
def root103Candidates6 : List Nat := [5839, 16381, 16993]
set_option maxRecDepth 8192 in
theorem root103Candidates6_checked : candidates edge 4 577 2617 root103Tail7 winner.bound = root103Candidates6 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child6 : search edge 3 [577, 2617] root103Candidates6 3194 winner = winner := by decide
theorem root103Step6 : scan edge (search edge 3) 4 [577] 577 root103Tail6 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail7 winner := by
  rw [root103Tail6, scan, if_neg (by decide), root103Candidates6_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail7
    (if 3 ≤ root103Candidates6.length then search edge 3 [577, 2617] root103Candidates6 3194 winner else winner) = _
  rw [if_pos (by decide), root103Child6]
def root103Candidates7 : List Nat := [8737]
set_option maxRecDepth 8192 in
theorem root103Candidates7_checked : candidates edge 4 577 3331 root103Tail8 winner.bound = root103Candidates7 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root103Step7 : scan edge (search edge 3) 4 [577] 577 root103Tail7 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail8 winner := by
  rw [root103Tail7, scan, if_neg (by decide), root103Candidates7_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail8
    (if 3 ≤ root103Candidates7.length then search edge 3 [577, 3331] root103Candidates7 3908 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root103Candidates8_checked : candidates edge 4 577 3709 root103Tail9 winner.bound = root103Candidates8 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root103Step8 : scan edge (search edge 3) 4 [577] 577 root103Tail8 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail9 winner := by
  rw [root103Tail8, scan, if_neg (by decide), root103Candidates8_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail9
    (if 3 ≤ root103Candidates8.length then search edge 3 [577, 3709] root103Candidates8 4286 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates9 : List Nat := [19993, 20101]
set_option maxRecDepth 8192 in
theorem root103Candidates9_checked : candidates edge 4 577 3919 root103Tail10 winner.bound = root103Candidates9 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root103Step9 : scan edge (search edge 3) 4 [577] 577 root103Tail9 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail10 winner := by
  rw [root103Tail9, scan, if_neg (by decide), root103Candidates9_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail10
    (if 3 ≤ root103Candidates9.length then search edge 3 [577, 3919] root103Candidates9 4496 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates10 : List Nat := [11779, 15661, 20431]
set_option maxRecDepth 8192 in
theorem root103Candidates10_checked : candidates edge 4 577 4597 root103Tail11 winner.bound = root103Candidates10 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child10 : search edge 3 [577, 4597] root103Candidates10 5174 winner = winner := by decide
theorem root103Step10 : scan edge (search edge 3) 4 [577] 577 root103Tail10 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail11 winner := by
  rw [root103Tail10, scan, if_neg (by decide), root103Candidates10_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail11
    (if 3 ≤ root103Candidates10.length then search edge 3 [577, 4597] root103Candidates10 5174 winner else winner) = _
  rw [if_pos (by decide), root103Child10]
def root103Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root103Candidates11_checked : candidates edge 4 577 4759 root103Tail12 winner.bound = root103Candidates11 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root103Step11 : scan edge (search edge 3) 4 [577] 577 root103Tail11 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail12 winner := by
  rw [root103Tail11, scan, if_neg (by decide), root103Candidates11_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail12
    (if 3 ≤ root103Candidates11.length then search edge 3 [577, 4759] root103Candidates11 5336 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root103Candidates12_checked : candidates edge 4 577 5431 root103Tail13 winner.bound = root103Candidates12 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root103Step12 : scan edge (search edge 3) 4 [577] 577 root103Tail12 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail13 winner := by
  rw [root103Tail12, scan, if_neg (by decide), root103Candidates12_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail13
    (if 3 ≤ root103Candidates12.length then search edge 3 [577, 5431] root103Candidates12 6008 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root103Candidates13_checked : candidates edge 4 577 5683 root103Tail14 winner.bound = root103Candidates13 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root103Step13 : scan edge (search edge 3) 4 [577] 577 root103Tail13 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail14 winner := by
  rw [root103Tail13, scan, if_neg (by decide), root103Candidates13_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail14
    (if 3 ≤ root103Candidates13.length then search edge 3 [577, 5683] root103Candidates13 6260 winner else winner) = _
  rw [if_neg (by decide)]
def root103Candidates14 : List Nat := [10567, 10663, 19231]
set_option maxRecDepth 8192 in
theorem root103Candidates14_checked : candidates edge 4 577 5821 root103Tail15 winner.bound = root103Candidates14 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root103Child14 : search edge 3 [577, 5821] root103Candidates14 6398 winner = winner := by decide
theorem root103Step14 : scan edge (search edge 3) 4 [577] 577 root103Tail14 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail15 winner := by
  rw [root103Tail14, scan, if_neg (by decide), root103Candidates14_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail15
    (if 3 ≤ root103Candidates14.length then search edge 3 [577, 5821] root103Candidates14 6398 winner else winner) = _
  rw [if_pos (by decide), root103Child14]
def root103Candidates15 : List Nat := [11047, 15121]
set_option maxRecDepth 8192 in
theorem root103Candidates15_checked : candidates edge 4 577 5839 root103Tail16 winner.bound = root103Candidates15 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root103Step15 : scan edge (search edge 3) 4 [577] 577 root103Tail15 winner =
    scan edge (search edge 3) 4 [577] 577 root103Tail16 winner := by
  rw [root103Tail15, scan, if_neg (by decide), root103Candidates15_checked]
  change scan edge (search edge 3) 4 [577] 577 root103Tail16
    (if 3 ≤ root103Candidates15.length then search edge 3 [577, 5839] root103Candidates15 6416 winner else winner) = _
  rw [if_neg (by decide)]
theorem root103Scan16 : scan edge (search edge 3) 4 [577] 577 root103Tail16 winner = winner := by rw [root103Tail16, scan, if_pos (by decide)]
theorem root103Scan15 : scan edge (search edge 3) 4 [577] 577 root103Tail15 winner = winner :=
  (root103Step15).trans root103Scan16
theorem root103Scan14 : scan edge (search edge 3) 4 [577] 577 root103Tail14 winner = winner :=
  (root103Step14).trans root103Scan15
theorem root103Scan13 : scan edge (search edge 3) 4 [577] 577 root103Tail13 winner = winner :=
  (root103Step13).trans root103Scan14
theorem root103Scan12 : scan edge (search edge 3) 4 [577] 577 root103Tail12 winner = winner :=
  (root103Step12).trans root103Scan13
theorem root103Scan11 : scan edge (search edge 3) 4 [577] 577 root103Tail11 winner = winner :=
  (root103Step11).trans root103Scan12
theorem root103Scan10 : scan edge (search edge 3) 4 [577] 577 root103Tail10 winner = winner :=
  (root103Step10).trans root103Scan11
theorem root103Scan9 : scan edge (search edge 3) 4 [577] 577 root103Tail9 winner = winner :=
  (root103Step9).trans root103Scan10
theorem root103Scan8 : scan edge (search edge 3) 4 [577] 577 root103Tail8 winner = winner :=
  (root103Step8).trans root103Scan9
theorem root103Scan7 : scan edge (search edge 3) 4 [577] 577 root103Tail7 winner = winner :=
  (root103Step7).trans root103Scan8
theorem root103Scan6 : scan edge (search edge 3) 4 [577] 577 root103Tail6 winner = winner :=
  (root103Step6).trans root103Scan7
theorem root103Scan5 : scan edge (search edge 3) 4 [577] 577 root103Tail5 winner = winner :=
  (root103Step5).trans root103Scan6
theorem root103Scan4 : scan edge (search edge 3) 4 [577] 577 root103Tail4 winner = winner :=
  (root103Step4).trans root103Scan5
theorem root103Scan3 : scan edge (search edge 3) 4 [577] 577 root103Tail3 winner = winner :=
  (root103Step3).trans root103Scan4
theorem root103Scan2 : scan edge (search edge 3) 4 [577] 577 root103Tail2 winner = winner :=
  (root103Step2).trans root103Scan3
theorem root103Scan1 : scan edge (search edge 3) 4 [577] 577 root103Tail1 winner = winner :=
  (root103Step1).trans root103Scan2
theorem root103Scan0 : scan edge (search edge 3) 4 [577] 577 root103Tail0 winner = winner :=
  (root103Step0).trans root103Scan1
set_option maxRecDepth 4096 in
theorem root103Branch : search edge 4 [577] root103Candidates 577 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [577] 577 root103Tail0 winner = winner
  exact root103Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
