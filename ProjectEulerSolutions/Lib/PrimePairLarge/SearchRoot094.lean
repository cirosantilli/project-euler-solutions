import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root94Candidates : List Nat := Data.row509
set_option maxRecDepth 8192 in
theorem root94Candidates_checked : candidates edge 5 0 509 tail95 winner.bound = root94Candidates := by
  apply candidates_eq_row edge 5 0 509 tail95 root94Candidates winner.bound tail95_ordered
  · decide
  · decide
  · funext q
    rfl
def root94Tail68 : List Nat := []
def root94Tail67 : List Nat := 25463 :: root94Tail68
def root94Tail66 : List Nat := 25439 :: root94Tail67
def root94Tail65 : List Nat := 25253 :: root94Tail66
def root94Tail64 : List Nat := 25031 :: root94Tail65
def root94Tail63 : List Nat := 23099 :: root94Tail64
def root94Tail62 : List Nat := 23039 :: root94Tail63
def root94Tail61 : List Nat := 21683 :: root94Tail62
def root94Tail60 : List Nat := 21323 :: root94Tail61
def root94Tail59 : List Nat := 21107 :: root94Tail60
def root94Tail58 : List Nat := 20759 :: root94Tail59
def root94Tail57 : List Nat := 20747 :: root94Tail58
def root94Tail56 : List Nat := 20663 :: root94Tail57
def root94Tail55 : List Nat := 20357 :: root94Tail56
def root94Tail54 : List Nat := 20327 :: root94Tail55
def root94Tail53 : List Nat := 19889 :: root94Tail54
def root94Tail52 : List Nat := 19793 :: root94Tail53
def root94Tail51 : List Nat := 19751 :: root94Tail52
def root94Tail50 : List Nat := 19553 :: root94Tail51
def root94Tail49 : List Nat := 18311 :: root94Tail50
def root94Tail48 : List Nat := 17597 :: root94Tail49
def root94Tail47 : List Nat := 17333 :: root94Tail48
def root94Tail46 : List Nat := 17327 :: root94Tail47
def root94Tail45 : List Nat := 17021 :: root94Tail46
def root94Tail44 : List Nat := 16487 :: root94Tail45
def root94Tail43 : List Nat := 16193 :: root94Tail44
def root94Tail42 : List Nat := 15467 :: root94Tail43
def root94Tail41 : List Nat := 14753 :: root94Tail42
def root94Tail40 : List Nat := 13859 :: root94Tail41
def root94Tail39 : List Nat := 13463 :: root94Tail40
def root94Tail38 : List Nat := 12377 :: root94Tail39
def root94Tail37 : List Nat := 12011 :: root94Tail38
def root94Tail36 : List Nat := 11813 :: root94Tail37
def root94Tail35 : List Nat := 11801 :: root94Tail36
def root94Tail34 : List Nat := 11369 :: root94Tail35
def root94Tail33 : List Nat := 11321 :: root94Tail34
def root94Tail32 : List Nat := 11087 :: root94Tail33
def root94Tail31 : List Nat := 10949 :: root94Tail32
def root94Tail30 : List Nat := 10487 :: root94Tail31
def root94Tail29 : List Nat := 10193 :: root94Tail30
def root94Tail28 : List Nat := 10163 :: root94Tail29
def root94Tail27 : List Nat := 9257 :: root94Tail28
def root94Tail26 : List Nat := 8369 :: root94Tail27
def root94Tail25 : List Nat := 8117 :: root94Tail26
def root94Tail24 : List Nat := 7793 :: root94Tail25
def root94Tail23 : List Nat := 7583 :: root94Tail24
def root94Tail22 : List Nat := 7577 :: root94Tail23
def root94Tail21 : List Nat := 7193 :: root94Tail22
def root94Tail20 : List Nat := 6863 :: root94Tail21
def root94Tail19 : List Nat := 6563 :: root94Tail20
def root94Tail18 : List Nat := 5711 :: root94Tail19
def root94Tail17 : List Nat := 5483 :: root94Tail18
def root94Tail16 : List Nat := 5471 :: root94Tail17
def root94Tail15 : List Nat := 4673 :: root94Tail16
def root94Tail14 : List Nat := 4241 :: root94Tail15
def root94Tail13 : List Nat := 3929 :: root94Tail14
def root94Tail12 : List Nat := 3359 :: root94Tail13
def root94Tail11 : List Nat := 2111 :: root94Tail12
def root94Tail10 : List Nat := 1973 :: root94Tail11
def root94Tail9 : List Nat := 1613 :: root94Tail10
def root94Tail8 : List Nat := 1553 :: root94Tail9
def root94Tail7 : List Nat := 1451 :: root94Tail8
def root94Tail6 : List Nat := 1091 :: root94Tail7
def root94Tail5 : List Nat := 1061 :: root94Tail6
def root94Tail4 : List Nat := 1019 :: root94Tail5
def root94Tail3 : List Nat := 947 :: root94Tail4
def root94Tail2 : List Nat := 863 :: root94Tail3
def root94Tail1 : List Nat := 797 :: root94Tail2
def root94Tail0 : List Nat := 647 :: root94Tail1
def root94Candidates0 : List Nat := [1019, 4241, 5471, 21323]
set_option maxRecDepth 8192 in
theorem root94Candidates0_checked : candidates edge 4 509 647 root94Tail1 winner.bound = root94Candidates0 := by
  unfold candidates
  have row : edge 647 = (SortedNatTree.contains FastRows.row647) := by
    change Data.row647.contains = (SortedNatTree.contains FastRows.row647)
    exact FastRows.row647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child0 : search edge 3 [509, 647] root94Candidates0 1156 winner = winner := by decide
theorem root94Step0 : scan edge (search edge 3) 4 [509] 509 root94Tail0 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail1 winner := by
  rw [root94Tail0, scan, if_neg (by decide), root94Candidates0_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail1
    (if 3 ≤ root94Candidates0.length then search edge 3 [509, 647] root94Candidates0 1156 winner else winner) = _
  rw [if_pos (by decide), root94Child0]
def root94Candidates1 : List Nat := [3929, 10193, 13463, 20759, 21683]
set_option maxRecDepth 8192 in
theorem root94Candidates1_checked : candidates edge 4 509 797 root94Tail2 winner.bound = root94Candidates1 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child1 : search edge 3 [509, 797] root94Candidates1 1306 winner = winner := by decide
theorem root94Step1 : scan edge (search edge 3) 4 [509] 509 root94Tail1 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail2 winner := by
  rw [root94Tail1, scan, if_neg (by decide), root94Candidates1_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail2
    (if 3 ≤ root94Candidates1.length then search edge 3 [509, 797] root94Candidates1 1306 winner else winner) = _
  rw [if_pos (by decide), root94Child1]
def root94Candidates2 : List Nat := [10487, 11369]
set_option maxRecDepth 8192 in
theorem root94Candidates2_checked : candidates edge 4 509 863 root94Tail3 winner.bound = root94Candidates2 := by
  unfold candidates
  have row : edge 863 = (SortedNatTree.contains FastRows.row863) := by
    change Data.row863.contains = (SortedNatTree.contains FastRows.row863)
    exact FastRows.row863_checked.symm
  rw [row]
  decide
theorem root94Step2 : scan edge (search edge 3) 4 [509] 509 root94Tail2 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail3 winner := by
  rw [root94Tail2, scan, if_neg (by decide), root94Candidates2_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail3
    (if 3 ≤ root94Candidates2.length then search edge 3 [509, 863] root94Candidates2 1372 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates3 : List Nat := [12377]
set_option maxRecDepth 8192 in
theorem root94Candidates3_checked : candidates edge 4 509 947 root94Tail4 winner.bound = root94Candidates3 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
theorem root94Step3 : scan edge (search edge 3) 4 [509] 509 root94Tail3 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail4 winner := by
  rw [root94Tail3, scan, if_neg (by decide), root94Candidates3_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail4
    (if 3 ≤ root94Candidates3.length then search edge 3 [509, 947] root94Candidates3 1456 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates4 : List Nat := [3929, 5483, 7193, 17327]
set_option maxRecDepth 8192 in
theorem root94Candidates4_checked : candidates edge 4 509 1019 root94Tail5 winner.bound = root94Candidates4 := by
  unfold candidates
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child4 : search edge 3 [509, 1019] root94Candidates4 1528 winner = winner := by decide
theorem root94Step4 : scan edge (search edge 3) 4 [509] 509 root94Tail4 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail5 winner := by
  rw [root94Tail4, scan, if_neg (by decide), root94Candidates4_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail5
    (if 3 ≤ root94Candidates4.length then search edge 3 [509, 1019] root94Candidates4 1528 winner else winner) = _
  rw [if_pos (by decide), root94Child4]
def root94Candidates5 : List Nat := [8117, 15467, 17021, 19889]
set_option maxRecDepth 8192 in
theorem root94Candidates5_checked : candidates edge 4 509 1061 root94Tail6 winner.bound = root94Candidates5 := by
  unfold candidates
  have row : edge 1061 = (SortedNatTree.contains FastRows.row1061) := by
    change Data.row1061.contains = (SortedNatTree.contains FastRows.row1061)
    exact FastRows.row1061_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child5 : search edge 3 [509, 1061] root94Candidates5 1570 winner = winner := by decide
theorem root94Step5 : scan edge (search edge 3) 4 [509] 509 root94Tail5 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail6 winner := by
  rw [root94Tail5, scan, if_neg (by decide), root94Candidates5_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail6
    (if 3 ≤ root94Candidates5.length then search edge 3 [509, 1061] root94Candidates5 1570 winner else winner) = _
  rw [if_pos (by decide), root94Child5]
def root94Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root94Candidates6_checked : candidates edge 4 509 1091 root94Tail7 winner.bound = root94Candidates6 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
theorem root94Step6 : scan edge (search edge 3) 4 [509] 509 root94Tail6 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail7 winner := by
  rw [root94Tail6, scan, if_neg (by decide), root94Candidates6_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail7
    (if 3 ≤ root94Candidates6.length then search edge 3 [509, 1091] root94Candidates6 1600 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates7 : List Nat := [7193]
set_option maxRecDepth 8192 in
theorem root94Candidates7_checked : candidates edge 4 509 1451 root94Tail8 winner.bound = root94Candidates7 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
theorem root94Step7 : scan edge (search edge 3) 4 [509] 509 root94Tail7 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail8 winner := by
  rw [root94Tail7, scan, if_neg (by decide), root94Candidates7_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail8
    (if 3 ≤ root94Candidates7.length then search edge 3 [509, 1451] root94Candidates7 1960 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates8 : List Nat := [1613, 10949, 13859, 19751]
set_option maxRecDepth 8192 in
theorem root94Candidates8_checked : candidates edge 4 509 1553 root94Tail9 winner.bound = root94Candidates8 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child8 : search edge 3 [509, 1553] root94Candidates8 2062 winner = winner := by decide
theorem root94Step8 : scan edge (search edge 3) 4 [509] 509 root94Tail8 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail9 winner := by
  rw [root94Tail8, scan, if_neg (by decide), root94Candidates8_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail9
    (if 3 ≤ root94Candidates8.length then search edge 3 [509, 1553] root94Candidates8 2062 winner else winner) = _
  rw [if_pos (by decide), root94Child8]
def root94Candidates9 : List Nat := [4673, 21683]
set_option maxRecDepth 8192 in
theorem root94Candidates9_checked : candidates edge 4 509 1613 root94Tail10 winner.bound = root94Candidates9 := by
  unfold candidates
  have row : edge 1613 = (SortedNatTree.contains FastRows.row1613) := by
    change Data.row1613.contains = (SortedNatTree.contains FastRows.row1613)
    exact FastRows.row1613_checked.symm
  rw [row]
  decide
theorem root94Step9 : scan edge (search edge 3) 4 [509] 509 root94Tail9 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail10 winner := by
  rw [root94Tail9, scan, if_neg (by decide), root94Candidates9_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail10
    (if 3 ≤ root94Candidates9.length then search edge 3 [509, 1613] root94Candidates9 2122 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates10 : List Nat := [5711, 17333, 20759]
set_option maxRecDepth 8192 in
theorem root94Candidates10_checked : candidates edge 4 509 1973 root94Tail11 winner.bound = root94Candidates10 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child10 : search edge 3 [509, 1973] root94Candidates10 2482 winner = winner := by decide
theorem root94Step10 : scan edge (search edge 3) 4 [509] 509 root94Tail10 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail11 winner := by
  rw [root94Tail10, scan, if_neg (by decide), root94Candidates10_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail11
    (if 3 ≤ root94Candidates10.length then search edge 3 [509, 1973] root94Candidates10 2482 winner else winner) = _
  rw [if_pos (by decide), root94Child10]
def root94Candidates11 : List Nat := [7793, 9257, 20759, 21107]
set_option maxRecDepth 8192 in
theorem root94Candidates11_checked : candidates edge 4 509 2111 root94Tail12 winner.bound = root94Candidates11 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child11 : search edge 3 [509, 2111] root94Candidates11 2620 winner = winner := by decide
theorem root94Step11 : scan edge (search edge 3) 4 [509] 509 root94Tail11 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail12 winner := by
  rw [root94Tail11, scan, if_neg (by decide), root94Candidates11_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail12
    (if 3 ≤ root94Candidates11.length then search edge 3 [509, 2111] root94Candidates11 2620 winner else winner) = _
  rw [if_pos (by decide), root94Child11]
def root94Candidates12 : List Nat := [11087]
set_option maxRecDepth 8192 in
theorem root94Candidates12_checked : candidates edge 4 509 3359 root94Tail13 winner.bound = root94Candidates12 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root94Step12 : scan edge (search edge 3) 4 [509] 509 root94Tail12 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail13 winner := by
  rw [root94Tail12, scan, if_neg (by decide), root94Candidates12_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail13
    (if 3 ≤ root94Candidates12.length then search edge 3 [509, 3359] root94Candidates12 3868 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates13 : List Nat := [10487, 17597]
set_option maxRecDepth 8192 in
theorem root94Candidates13_checked : candidates edge 4 509 3929 root94Tail14 winner.bound = root94Candidates13 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root94Step13 : scan edge (search edge 3) 4 [509] 509 root94Tail13 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail14 winner := by
  rw [root94Tail13, scan, if_neg (by decide), root94Candidates13_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail14
    (if 3 ≤ root94Candidates13.length then search edge 3 [509, 3929] root94Candidates13 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates14 : List Nat := [10949, 20327]
set_option maxRecDepth 8192 in
theorem root94Candidates14_checked : candidates edge 4 509 4241 root94Tail15 winner.bound = root94Candidates14 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root94Step14 : scan edge (search edge 3) 4 [509] 509 root94Tail14 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail15 winner := by
  rw [root94Tail14, scan, if_neg (by decide), root94Candidates14_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail15
    (if 3 ≤ root94Candidates14.length then search edge 3 [509, 4241] root94Candidates14 4750 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root94Candidates15_checked : candidates edge 4 509 4673 root94Tail16 winner.bound = root94Candidates15 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root94Step15 : scan edge (search edge 3) 4 [509] 509 root94Tail15 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail16 winner := by
  rw [root94Tail15, scan, if_neg (by decide), root94Candidates15_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail16
    (if 3 ≤ root94Candidates15.length then search edge 3 [509, 4673] root94Candidates15 5182 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates16 : List Nat := [7583]
set_option maxRecDepth 8192 in
theorem root94Candidates16_checked : candidates edge 4 509 5471 root94Tail17 winner.bound = root94Candidates16 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root94Step16 : scan edge (search edge 3) 4 [509] 509 root94Tail16 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail17 winner := by
  rw [root94Tail16, scan, if_neg (by decide), root94Candidates16_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail17
    (if 3 ≤ root94Candidates16.length then search edge 3 [509, 5471] root94Candidates16 5980 winner else winner) = _
  rw [if_neg (by decide)]
def root94Candidates17 : List Nat := [8369, 11321, 13463]
set_option maxRecDepth 8192 in
theorem root94Candidates17_checked : candidates edge 4 509 5483 root94Tail18 winner.bound = root94Candidates17 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root94Child17 : search edge 3 [509, 5483] root94Candidates17 5992 winner = winner := by decide
theorem root94Step17 : scan edge (search edge 3) 4 [509] 509 root94Tail17 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail18 winner := by
  rw [root94Tail17, scan, if_neg (by decide), root94Candidates17_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail18
    (if 3 ≤ root94Candidates17.length then search edge 3 [509, 5483] root94Candidates17 5992 winner else winner) = _
  rw [if_pos (by decide), root94Child17]
def root94Candidates18 : List Nat := [13859]
set_option maxRecDepth 8192 in
theorem root94Candidates18_checked : candidates edge 4 509 5711 root94Tail19 winner.bound = root94Candidates18 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root94Step18 : scan edge (search edge 3) 4 [509] 509 root94Tail18 winner =
    scan edge (search edge 3) 4 [509] 509 root94Tail19 winner := by
  rw [root94Tail18, scan, if_neg (by decide), root94Candidates18_checked]
  change scan edge (search edge 3) 4 [509] 509 root94Tail19
    (if 3 ≤ root94Candidates18.length then search edge 3 [509, 5711] root94Candidates18 6220 winner else winner) = _
  rw [if_neg (by decide)]
theorem root94Scan19 : scan edge (search edge 3) 4 [509] 509 root94Tail19 winner = winner := by rw [root94Tail19, scan, if_pos (by decide)]
theorem root94Scan18 : scan edge (search edge 3) 4 [509] 509 root94Tail18 winner = winner :=
  (root94Step18).trans root94Scan19
theorem root94Scan17 : scan edge (search edge 3) 4 [509] 509 root94Tail17 winner = winner :=
  (root94Step17).trans root94Scan18
theorem root94Scan16 : scan edge (search edge 3) 4 [509] 509 root94Tail16 winner = winner :=
  (root94Step16).trans root94Scan17
theorem root94Scan15 : scan edge (search edge 3) 4 [509] 509 root94Tail15 winner = winner :=
  (root94Step15).trans root94Scan16
theorem root94Scan14 : scan edge (search edge 3) 4 [509] 509 root94Tail14 winner = winner :=
  (root94Step14).trans root94Scan15
theorem root94Scan13 : scan edge (search edge 3) 4 [509] 509 root94Tail13 winner = winner :=
  (root94Step13).trans root94Scan14
theorem root94Scan12 : scan edge (search edge 3) 4 [509] 509 root94Tail12 winner = winner :=
  (root94Step12).trans root94Scan13
theorem root94Scan11 : scan edge (search edge 3) 4 [509] 509 root94Tail11 winner = winner :=
  (root94Step11).trans root94Scan12
theorem root94Scan10 : scan edge (search edge 3) 4 [509] 509 root94Tail10 winner = winner :=
  (root94Step10).trans root94Scan11
theorem root94Scan9 : scan edge (search edge 3) 4 [509] 509 root94Tail9 winner = winner :=
  (root94Step9).trans root94Scan10
theorem root94Scan8 : scan edge (search edge 3) 4 [509] 509 root94Tail8 winner = winner :=
  (root94Step8).trans root94Scan9
theorem root94Scan7 : scan edge (search edge 3) 4 [509] 509 root94Tail7 winner = winner :=
  (root94Step7).trans root94Scan8
theorem root94Scan6 : scan edge (search edge 3) 4 [509] 509 root94Tail6 winner = winner :=
  (root94Step6).trans root94Scan7
theorem root94Scan5 : scan edge (search edge 3) 4 [509] 509 root94Tail5 winner = winner :=
  (root94Step5).trans root94Scan6
theorem root94Scan4 : scan edge (search edge 3) 4 [509] 509 root94Tail4 winner = winner :=
  (root94Step4).trans root94Scan5
theorem root94Scan3 : scan edge (search edge 3) 4 [509] 509 root94Tail3 winner = winner :=
  (root94Step3).trans root94Scan4
theorem root94Scan2 : scan edge (search edge 3) 4 [509] 509 root94Tail2 winner = winner :=
  (root94Step2).trans root94Scan3
theorem root94Scan1 : scan edge (search edge 3) 4 [509] 509 root94Tail1 winner = winner :=
  (root94Step1).trans root94Scan2
theorem root94Scan0 : scan edge (search edge 3) 4 [509] 509 root94Tail0 winner = winner :=
  (root94Step0).trans root94Scan1
set_option maxRecDepth 4096 in
theorem root94Branch : search edge 4 [509] root94Candidates 509 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [509] 509 root94Tail0 winner = winner
  exact root94Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
