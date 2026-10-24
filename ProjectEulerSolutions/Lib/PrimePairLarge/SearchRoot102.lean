import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root102Candidates : List Nat := Data.row571
set_option maxRecDepth 8192 in
theorem root102Candidates_checked : candidates edge 5 0 571 tail103 winner.bound = root102Candidates := by
  apply candidates_eq_row edge 5 0 571 tail103 root102Candidates winner.bound tail103_ordered
  · decide
  · decide
  · funext q
    rfl
def root102Tail64 : List Nat := []
def root102Tail63 : List Nat := 25303 :: root102Tail64
def root102Tail62 : List Nat := 25147 :: root102Tail63
def root102Tail61 : List Nat := 24919 :: root102Tail62
def root102Tail60 : List Nat := 24547 :: root102Tail61
def root102Tail59 : List Nat := 22129 :: root102Tail60
def root102Tail58 : List Nat := 22111 :: root102Tail59
def root102Tail57 : List Nat := 20533 :: root102Tail58
def root102Tail56 : List Nat := 20359 :: root102Tail57
def root102Tail55 : List Nat := 19801 :: root102Tail56
def root102Tail54 : List Nat := 19753 :: root102Tail55
def root102Tail53 : List Nat := 19477 :: root102Tail54
def root102Tail52 : List Nat := 17839 :: root102Tail53
def root102Tail51 : List Nat := 17509 :: root102Tail52
def root102Tail50 : List Nat := 16963 :: root102Tail51
def root102Tail49 : List Nat := 16519 :: root102Tail50
def root102Tail48 : List Nat := 16273 :: root102Tail49
def root102Tail47 : List Nat := 15907 :: root102Tail48
def root102Tail46 : List Nat := 15493 :: root102Tail47
def root102Tail45 : List Nat := 15451 :: root102Tail46
def root102Tail44 : List Nat := 15319 :: root102Tail45
def root102Tail43 : List Nat := 15277 :: root102Tail44
def root102Tail42 : List Nat := 15031 :: root102Tail43
def root102Tail41 : List Nat := 14947 :: root102Tail42
def root102Tail40 : List Nat := 14737 :: root102Tail41
def root102Tail39 : List Nat := 14503 :: root102Tail40
def root102Tail38 : List Nat := 14401 :: root102Tail39
def root102Tail37 : List Nat := 14149 :: root102Tail38
def root102Tail36 : List Nat := 13879 :: root102Tail37
def root102Tail35 : List Nat := 13627 :: root102Tail36
def root102Tail34 : List Nat := 13291 :: root102Tail35
def root102Tail33 : List Nat := 13219 :: root102Tail34
def root102Tail32 : List Nat := 12967 :: root102Tail33
def root102Tail31 : List Nat := 12109 :: root102Tail32
def root102Tail30 : List Nat := 11839 :: root102Tail31
def root102Tail29 : List Nat := 11779 :: root102Tail30
def root102Tail28 : List Nat := 11503 :: root102Tail29
def root102Tail27 : List Nat := 10597 :: root102Tail28
def root102Tail26 : List Nat := 9661 :: root102Tail27
def root102Tail25 : List Nat := 8737 :: root102Tail26
def root102Tail24 : List Nat := 8641 :: root102Tail25
def root102Tail23 : List Nat := 8263 :: root102Tail24
def root102Tail22 : List Nat := 8233 :: root102Tail23
def root102Tail21 : List Nat := 7951 :: root102Tail22
def root102Tail20 : List Nat := 7549 :: root102Tail21
def root102Tail19 : List Nat := 7297 :: root102Tail20
def root102Tail18 : List Nat := 7213 :: root102Tail19
def root102Tail17 : List Nat := 7069 :: root102Tail18
def root102Tail16 : List Nat := 6553 :: root102Tail17
def root102Tail15 : List Nat := 5839 :: root102Tail16
def root102Tail14 : List Nat := 5701 :: root102Tail15
def root102Tail13 : List Nat := 4987 :: root102Tail14
def root102Tail12 : List Nat := 4759 :: root102Tail13
def root102Tail11 : List Nat := 4483 :: root102Tail12
def root102Tail10 : List Nat := 3163 :: root102Tail11
def root102Tail9 : List Nat := 3121 :: root102Tail10
def root102Tail8 : List Nat := 3061 :: root102Tail9
def root102Tail7 : List Nat := 2953 :: root102Tail8
def root102Tail6 : List Nat := 2659 :: root102Tail7
def root102Tail5 : List Nat := 2647 :: root102Tail6
def root102Tail4 : List Nat := 2383 :: root102Tail5
def root102Tail3 : List Nat := 2341 :: root102Tail4
def root102Tail2 : List Nat := 1933 :: root102Tail3
def root102Tail1 : List Nat := 1051 :: root102Tail2
def root102Tail0 : List Nat := 853 :: root102Tail1
def root102Candidates0 : List Nat := [11839, 12967, 14947, 15277, 19477, 22129]
set_option maxRecDepth 8192 in
theorem root102Candidates0_checked : candidates edge 4 571 853 root102Tail1 winner.bound = root102Candidates0 := by
  unfold candidates
  have row : edge 853 = (SortedNatTree.contains FastRows.row853) := by
    change Data.row853.contains = (SortedNatTree.contains FastRows.row853)
    exact FastRows.row853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root102Child0 : search edge 3 [571, 853] root102Candidates0 1424 winner = winner := by decide
theorem root102Step0 : scan edge (search edge 3) 4 [571] 571 root102Tail0 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail1 winner := by
  rw [root102Tail0, scan, if_neg (by decide), root102Candidates0_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail1
    (if 3 ≤ root102Candidates0.length then search edge 3 [571, 853] root102Candidates0 1424 winner else winner) = _
  rw [if_pos (by decide), root102Child0]
def root102Candidates1 : List Nat := [5839]
set_option maxRecDepth 8192 in
theorem root102Candidates1_checked : candidates edge 4 571 1051 root102Tail2 winner.bound = root102Candidates1 := by
  unfold candidates
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row]
  decide
theorem root102Step1 : scan edge (search edge 3) 4 [571] 571 root102Tail1 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail2 winner := by
  rw [root102Tail1, scan, if_neg (by decide), root102Candidates1_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail2
    (if 3 ≤ root102Candidates1.length then search edge 3 [571, 1051] root102Candidates1 1622 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates2 : List Nat := [10597]
set_option maxRecDepth 8192 in
theorem root102Candidates2_checked : candidates edge 4 571 1933 root102Tail3 winner.bound = root102Candidates2 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root102Step2 : scan edge (search edge 3) 4 [571] 571 root102Tail2 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail3 winner := by
  rw [root102Tail2, scan, if_neg (by decide), root102Candidates2_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail3
    (if 3 ≤ root102Candidates2.length then search edge 3 [571, 1933] root102Candidates2 2504 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates3 : List Nat := [15907]
set_option maxRecDepth 8192 in
theorem root102Candidates3_checked : candidates edge 4 571 2341 root102Tail4 winner.bound = root102Candidates3 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root102Step3 : scan edge (search edge 3) 4 [571] 571 root102Tail3 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail4 winner := by
  rw [root102Tail3, scan, if_neg (by decide), root102Candidates3_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail4
    (if 3 ≤ root102Candidates3.length then search edge 3 [571, 2341] root102Candidates3 2912 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates4 : List Nat := [2659, 8263, 11779]
set_option maxRecDepth 8192 in
theorem root102Candidates4_checked : candidates edge 4 571 2383 root102Tail5 winner.bound = root102Candidates4 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root102Child4 : search edge 3 [571, 2383] root102Candidates4 2954 winner = winner := by decide
theorem root102Step4 : scan edge (search edge 3) 4 [571] 571 root102Tail4 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail5 winner := by
  rw [root102Tail4, scan, if_neg (by decide), root102Candidates4_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail5
    (if 3 ≤ root102Candidates4.length then search edge 3 [571, 2383] root102Candidates4 2954 winner else winner) = _
  rw [if_pos (by decide), root102Child4]
def root102Candidates5 : List Nat := [6553, 7951, 8737, 13219, 20359]
set_option maxRecDepth 8192 in
theorem root102Candidates5_checked : candidates edge 4 571 2647 root102Tail6 winner.bound = root102Candidates5 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root102Child5 : search edge 3 [571, 2647] root102Candidates5 3218 winner = winner := by decide
theorem root102Step5 : scan edge (search edge 3) 4 [571] 571 root102Tail5 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail6 winner := by
  rw [root102Tail5, scan, if_neg (by decide), root102Candidates5_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail6
    (if 3 ≤ root102Candidates5.length then search edge 3 [571, 2647] root102Candidates5 3218 winner else winner) = _
  rw [if_pos (by decide), root102Child5]
def root102Candidates6 : List Nat := [4759, 8641, 12109, 14503]
set_option maxRecDepth 8192 in
theorem root102Candidates6_checked : candidates edge 4 571 2659 root102Tail7 winner.bound = root102Candidates6 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root102Child6 : search edge 3 [571, 2659] root102Candidates6 3230 winner = winner := by decide
theorem root102Step6 : scan edge (search edge 3) 4 [571] 571 root102Tail6 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail7 winner := by
  rw [root102Tail6, scan, if_neg (by decide), root102Candidates6_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail7
    (if 3 ≤ root102Candidates6.length then search edge 3 [571, 2659] root102Candidates6 3230 winner else winner) = _
  rw [if_pos (by decide), root102Child6]
def root102Candidates7 : List Nat := [7213, 20359]
set_option maxRecDepth 8192 in
theorem root102Candidates7_checked : candidates edge 4 571 2953 root102Tail8 winner.bound = root102Candidates7 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
theorem root102Step7 : scan edge (search edge 3) 4 [571] 571 root102Tail7 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail8 winner := by
  rw [root102Tail7, scan, if_neg (by decide), root102Candidates7_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail8
    (if 3 ≤ root102Candidates7.length then search edge 3 [571, 2953] root102Candidates7 3524 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates8 : List Nat := [13627, 15493]
set_option maxRecDepth 8192 in
theorem root102Candidates8_checked : candidates edge 4 571 3061 root102Tail9 winner.bound = root102Candidates8 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root102Step8 : scan edge (search edge 3) 4 [571] 571 root102Tail8 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail9 winner := by
  rw [root102Tail8, scan, if_neg (by decide), root102Candidates8_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail9
    (if 3 ≤ root102Candidates8.length then search edge 3 [571, 3061] root102Candidates8 3632 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root102Candidates9_checked : candidates edge 4 571 3121 root102Tail10 winner.bound = root102Candidates9 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
theorem root102Step9 : scan edge (search edge 3) 4 [571] 571 root102Tail9 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail10 winner := by
  rw [root102Tail9, scan, if_neg (by decide), root102Candidates9_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail10
    (if 3 ≤ root102Candidates9.length then search edge 3 [571, 3121] root102Candidates9 3692 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates10 : List Nat := [7951, 12967]
set_option maxRecDepth 8192 in
theorem root102Candidates10_checked : candidates edge 4 571 3163 root102Tail11 winner.bound = root102Candidates10 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root102Step10 : scan edge (search edge 3) 4 [571] 571 root102Tail10 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail11 winner := by
  rw [root102Tail10, scan, if_neg (by decide), root102Candidates10_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail11
    (if 3 ≤ root102Candidates10.length then search edge 3 [571, 3163] root102Candidates10 3734 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root102Candidates11_checked : candidates edge 4 571 4483 root102Tail12 winner.bound = root102Candidates11 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root102Step11 : scan edge (search edge 3) 4 [571] 571 root102Tail11 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail12 winner := by
  rw [root102Tail11, scan, if_neg (by decide), root102Candidates11_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail12
    (if 3 ≤ root102Candidates11.length then search edge 3 [571, 4483] root102Candidates11 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates12 : List Nat := [5701, 9661]
set_option maxRecDepth 8192 in
theorem root102Candidates12_checked : candidates edge 4 571 4759 root102Tail13 winner.bound = root102Candidates12 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root102Step12 : scan edge (search edge 3) 4 [571] 571 root102Tail12 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail13 winner := by
  rw [root102Tail12, scan, if_neg (by decide), root102Candidates12_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail13
    (if 3 ≤ root102Candidates12.length then search edge 3 [571, 4759] root102Candidates12 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates13 : List Nat := [8641, 14149, 14947]
set_option maxRecDepth 8192 in
theorem root102Candidates13_checked : candidates edge 4 571 4987 root102Tail14 winner.bound = root102Candidates13 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root102Child13 : search edge 3 [571, 4987] root102Candidates13 5558 winner = winner := by decide
theorem root102Step13 : scan edge (search edge 3) 4 [571] 571 root102Tail13 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail14 winner := by
  rw [root102Tail13, scan, if_neg (by decide), root102Candidates13_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail14
    (if 3 ≤ root102Candidates13.length then search edge 3 [571, 4987] root102Candidates13 5558 winner else winner) = _
  rw [if_pos (by decide), root102Child13]
def root102Candidates14 : List Nat := [14947]
set_option maxRecDepth 8192 in
theorem root102Candidates14_checked : candidates edge 4 571 5701 root102Tail15 winner.bound = root102Candidates14 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root102Step14 : scan edge (search edge 3) 4 [571] 571 root102Tail14 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail15 winner := by
  rw [root102Tail14, scan, if_neg (by decide), root102Candidates14_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail15
    (if 3 ≤ root102Candidates14.length then search edge 3 [571, 5701] root102Candidates14 6272 winner else winner) = _
  rw [if_neg (by decide)]
def root102Candidates15 : List Nat := [8233, 10597]
set_option maxRecDepth 8192 in
theorem root102Candidates15_checked : candidates edge 4 571 5839 root102Tail16 winner.bound = root102Candidates15 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root102Step15 : scan edge (search edge 3) 4 [571] 571 root102Tail15 winner =
    scan edge (search edge 3) 4 [571] 571 root102Tail16 winner := by
  rw [root102Tail15, scan, if_neg (by decide), root102Candidates15_checked]
  change scan edge (search edge 3) 4 [571] 571 root102Tail16
    (if 3 ≤ root102Candidates15.length then search edge 3 [571, 5839] root102Candidates15 6410 winner else winner) = _
  rw [if_neg (by decide)]
theorem root102Scan16 : scan edge (search edge 3) 4 [571] 571 root102Tail16 winner = winner := by rw [root102Tail16, scan, if_pos (by decide)]
theorem root102Scan15 : scan edge (search edge 3) 4 [571] 571 root102Tail15 winner = winner :=
  (root102Step15).trans root102Scan16
theorem root102Scan14 : scan edge (search edge 3) 4 [571] 571 root102Tail14 winner = winner :=
  (root102Step14).trans root102Scan15
theorem root102Scan13 : scan edge (search edge 3) 4 [571] 571 root102Tail13 winner = winner :=
  (root102Step13).trans root102Scan14
theorem root102Scan12 : scan edge (search edge 3) 4 [571] 571 root102Tail12 winner = winner :=
  (root102Step12).trans root102Scan13
theorem root102Scan11 : scan edge (search edge 3) 4 [571] 571 root102Tail11 winner = winner :=
  (root102Step11).trans root102Scan12
theorem root102Scan10 : scan edge (search edge 3) 4 [571] 571 root102Tail10 winner = winner :=
  (root102Step10).trans root102Scan11
theorem root102Scan9 : scan edge (search edge 3) 4 [571] 571 root102Tail9 winner = winner :=
  (root102Step9).trans root102Scan10
theorem root102Scan8 : scan edge (search edge 3) 4 [571] 571 root102Tail8 winner = winner :=
  (root102Step8).trans root102Scan9
theorem root102Scan7 : scan edge (search edge 3) 4 [571] 571 root102Tail7 winner = winner :=
  (root102Step7).trans root102Scan8
theorem root102Scan6 : scan edge (search edge 3) 4 [571] 571 root102Tail6 winner = winner :=
  (root102Step6).trans root102Scan7
theorem root102Scan5 : scan edge (search edge 3) 4 [571] 571 root102Tail5 winner = winner :=
  (root102Step5).trans root102Scan6
theorem root102Scan4 : scan edge (search edge 3) 4 [571] 571 root102Tail4 winner = winner :=
  (root102Step4).trans root102Scan5
theorem root102Scan3 : scan edge (search edge 3) 4 [571] 571 root102Tail3 winner = winner :=
  (root102Step3).trans root102Scan4
theorem root102Scan2 : scan edge (search edge 3) 4 [571] 571 root102Tail2 winner = winner :=
  (root102Step2).trans root102Scan3
theorem root102Scan1 : scan edge (search edge 3) 4 [571] 571 root102Tail1 winner = winner :=
  (root102Step1).trans root102Scan2
theorem root102Scan0 : scan edge (search edge 3) 4 [571] 571 root102Tail0 winner = winner :=
  (root102Step0).trans root102Scan1
set_option maxRecDepth 4096 in
theorem root102Branch : search edge 4 [571] root102Candidates 571 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [571] 571 root102Tail0 winner = winner
  exact root102Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
