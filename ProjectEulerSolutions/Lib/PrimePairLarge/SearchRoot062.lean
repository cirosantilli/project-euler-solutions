import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root62Candidates : List Nat := Data.row313
set_option maxRecDepth 8192 in
theorem root62Candidates_checked : candidates edge 5 0 313 tail63 winner.bound = root62Candidates := by
  apply candidates_eq_row edge 5 0 313 tail63 root62Candidates winner.bound tail63_ordered
  · decide
  · decide
  · funext q
    rfl
def root62Tail66 : List Nat := []
def root62Tail65 : List Nat := 24877 :: root62Tail66
def root62Tail64 : List Nat := 24133 :: root62Tail65
def root62Tail63 : List Nat := 24061 :: root62Tail64
def root62Tail62 : List Nat := 23629 :: root62Tail63
def root62Tail61 : List Nat := 23251 :: root62Tail62
def root62Tail60 : List Nat := 22717 :: root62Tail61
def root62Tail59 : List Nat := 21559 :: root62Tail60
def root62Tail58 : List Nat := 21163 :: root62Tail59
def root62Tail57 : List Nat := 20173 :: root62Tail58
def root62Tail56 : List Nat := 18127 :: root62Tail57
def root62Tail55 : List Nat := 18043 :: root62Tail56
def root62Tail54 : List Nat := 18013 :: root62Tail55
def root62Tail53 : List Nat := 17851 :: root62Tail54
def root62Tail52 : List Nat := 17683 :: root62Tail53
def root62Tail51 : List Nat := 17581 :: root62Tail52
def root62Tail50 : List Nat := 16981 :: root62Tail51
def root62Tail49 : List Nat := 16843 :: root62Tail50
def root62Tail48 : List Nat := 16699 :: root62Tail49
def root62Tail47 : List Nat := 16267 :: root62Tail48
def root62Tail46 : List Nat := 16111 :: root62Tail47
def root62Tail45 : List Nat := 15901 :: root62Tail46
def root62Tail44 : List Nat := 15541 :: root62Tail45
def root62Tail43 : List Nat := 15313 :: root62Tail44
def root62Tail42 : List Nat := 14947 :: root62Tail43
def root62Tail41 : List Nat := 14827 :: root62Tail42
def root62Tail40 : List Nat := 14683 :: root62Tail41
def root62Tail39 : List Nat := 14281 :: root62Tail40
def root62Tail38 : List Nat := 12907 :: root62Tail39
def root62Tail37 : List Nat := 12613 :: root62Tail38
def root62Tail36 : List Nat := 12433 :: root62Tail37
def root62Tail35 : List Nat := 12049 :: root62Tail36
def root62Tail34 : List Nat := 12007 :: root62Tail35
def root62Tail33 : List Nat := 11941 :: root62Tail34
def root62Tail32 : List Nat := 11593 :: root62Tail33
def root62Tail31 : List Nat := 11587 :: root62Tail32
def root62Tail30 : List Nat := 11491 :: root62Tail31
def root62Tail29 : List Nat := 11383 :: root62Tail30
def root62Tail28 : List Nat := 11251 :: root62Tail29
def root62Tail27 : List Nat := 11197 :: root62Tail28
def root62Tail26 : List Nat := 11173 :: root62Tail27
def root62Tail25 : List Nat := 10597 :: root62Tail26
def root62Tail24 : List Nat := 10369 :: root62Tail25
def root62Tail23 : List Nat := 9901 :: root62Tail24
def root62Tail22 : List Nat := 9811 :: root62Tail23
def root62Tail21 : List Nat := 9733 :: root62Tail22
def root62Tail20 : List Nat := 9601 :: root62Tail21
def root62Tail19 : List Nat := 9391 :: root62Tail20
def root62Tail18 : List Nat := 9187 :: root62Tail19
def root62Tail17 : List Nat := 8539 :: root62Tail18
def root62Tail16 : List Nat := 8329 :: root62Tail17
def root62Tail15 : List Nat := 8221 :: root62Tail16
def root62Tail14 : List Nat := 6079 :: root62Tail15
def root62Tail13 : List Nat := 6037 :: root62Tail14
def root62Tail12 : List Nat := 4591 :: root62Tail13
def root62Tail11 : List Nat := 4231 :: root62Tail12
def root62Tail10 : List Nat := 3853 :: root62Tail11
def root62Tail9 : List Nat := 3733 :: root62Tail10
def root62Tail8 : List Nat := 3583 :: root62Tail9
def root62Tail7 : List Nat := 3469 :: root62Tail8
def root62Tail6 : List Nat := 2281 :: root62Tail7
def root62Tail5 : List Nat := 1747 :: root62Tail6
def root62Tail4 : List Nat := 1669 :: root62Tail5
def root62Tail3 : List Nat := 1231 :: root62Tail4
def root62Tail2 : List Nat := 991 :: root62Tail3
def root62Tail1 : List Nat := 727 :: root62Tail2
def root62Tail0 : List Nat := 619 :: root62Tail1
def root62Candidates0 : List Nat := [1747, 9601, 14683, 15901, 16111, 24133]
set_option maxRecDepth 8192 in
theorem root62Candidates0_checked : candidates edge 4 313 619 root62Tail1 winner.bound = root62Candidates0 := by
  unfold candidates
  have row : edge 619 = (SortedNatTree.contains FastRows.row619) := by
    change Data.row619.contains = (SortedNatTree.contains FastRows.row619)
    exact FastRows.row619_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child0 : search edge 3 [313, 619] root62Candidates0 932 winner = winner := by decide
theorem root62Step0 : scan edge (search edge 3) 4 [313] 313 root62Tail0 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail1 winner := by
  rw [root62Tail0, scan, if_neg (by decide), root62Candidates0_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail1
    (if 3 ≤ root62Candidates0.length then search edge 3 [313, 619] root62Candidates0 932 winner else winner) = _
  rw [if_pos (by decide), root62Child0]
def root62Candidates1 : List Nat := [1669, 9187, 10369, 17851]
set_option maxRecDepth 8192 in
theorem root62Candidates1_checked : candidates edge 4 313 727 root62Tail2 winner.bound = root62Candidates1 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child1 : search edge 3 [313, 727] root62Candidates1 1040 winner = winner := by decide
theorem root62Step1 : scan edge (search edge 3) 4 [313] 313 root62Tail1 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail2 winner := by
  rw [root62Tail1, scan, if_neg (by decide), root62Candidates1_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail2
    (if 3 ≤ root62Candidates1.length then search edge 3 [313, 727] root62Candidates1 1040 winner else winner) = _
  rw [if_pos (by decide), root62Child1]
def root62Candidates2 : List Nat := [16267, 21163]
set_option maxRecDepth 8192 in
theorem root62Candidates2_checked : candidates edge 4 313 991 root62Tail3 winner.bound = root62Candidates2 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
theorem root62Step2 : scan edge (search edge 3) 4 [313] 313 root62Tail2 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail3 winner := by
  rw [root62Tail2, scan, if_neg (by decide), root62Candidates2_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail3
    (if 3 ≤ root62Candidates2.length then search edge 3 [313, 991] root62Candidates2 1304 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates3 : List Nat := [4231, 6037, 8221, 12907, 14683, 15901, 16111]
set_option maxRecDepth 8192 in
theorem root62Candidates3_checked : candidates edge 4 313 1231 root62Tail4 winner.bound = root62Candidates3 := by
  unfold candidates
  have row : edge 1231 = (SortedNatTree.contains FastRows.row1231) := by
    change Data.row1231.contains = (SortedNatTree.contains FastRows.row1231)
    exact FastRows.row1231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child3 : search edge 3 [313, 1231] root62Candidates3 1544 winner = winner := by decide
theorem root62Step3 : scan edge (search edge 3) 4 [313] 313 root62Tail3 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail4 winner := by
  rw [root62Tail3, scan, if_neg (by decide), root62Candidates3_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail4
    (if 3 ≤ root62Candidates3.length then search edge 3 [313, 1231] root62Candidates3 1544 winner else winner) = _
  rw [if_pos (by decide), root62Child3]
def root62Candidates4 : List Nat := [9733, 21163]
set_option maxRecDepth 8192 in
theorem root62Candidates4_checked : candidates edge 4 313 1669 root62Tail5 winner.bound = root62Candidates4 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
theorem root62Step4 : scan edge (search edge 3) 4 [313] 313 root62Tail4 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail5 winner := by
  rw [root62Tail4, scan, if_neg (by decide), root62Candidates4_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail5
    (if 3 ≤ root62Candidates4.length then search edge 3 [313, 1669] root62Candidates4 1982 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates5 : List Nat := [4231, 6037, 9811, 12007]
set_option maxRecDepth 8192 in
theorem root62Candidates5_checked : candidates edge 4 313 1747 root62Tail6 winner.bound = root62Candidates5 := by
  unfold candidates
  have row : edge 1747 = (SortedNatTree.contains FastRows.row1747) := by
    change Data.row1747.contains = (SortedNatTree.contains FastRows.row1747)
    exact FastRows.row1747_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child5 : search edge 3 [313, 1747] root62Candidates5 2060 winner = winner := by decide
theorem root62Step5 : scan edge (search edge 3) 4 [313] 313 root62Tail5 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail6 winner := by
  rw [root62Tail5, scan, if_neg (by decide), root62Candidates5_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail6
    (if 3 ≤ root62Candidates5.length then search edge 3 [313, 1747] root62Candidates5 2060 winner else winner) = _
  rw [if_pos (by decide), root62Child5]
def root62Candidates6 : List Nat := [4591, 11197, 17581, 20173]
set_option maxRecDepth 8192 in
theorem root62Candidates6_checked : candidates edge 4 313 2281 root62Tail7 winner.bound = root62Candidates6 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child6 : search edge 3 [313, 2281] root62Candidates6 2594 winner = winner := by decide
theorem root62Step6 : scan edge (search edge 3) 4 [313] 313 root62Tail6 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail7 winner := by
  rw [root62Tail6, scan, if_neg (by decide), root62Candidates6_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail7
    (if 3 ≤ root62Candidates6.length then search edge 3 [313, 2281] root62Candidates6 2594 winner else winner) = _
  rw [if_pos (by decide), root62Child6]
def root62Candidates7 : List Nat := [18043]
set_option maxRecDepth 8192 in
theorem root62Candidates7_checked : candidates edge 4 313 3469 root62Tail8 winner.bound = root62Candidates7 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root62Step7 : scan edge (search edge 3) 4 [313] 313 root62Tail7 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail8 winner := by
  rw [root62Tail7, scan, if_neg (by decide), root62Candidates7_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail8
    (if 3 ≤ root62Candidates7.length then search edge 3 [313, 3469] root62Candidates7 3782 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates8 : List Nat := [12613, 16267]
set_option maxRecDepth 8192 in
theorem root62Candidates8_checked : candidates edge 4 313 3583 root62Tail9 winner.bound = root62Candidates8 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root62Step8 : scan edge (search edge 3) 4 [313] 313 root62Tail8 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail9 winner := by
  rw [root62Tail8, scan, if_neg (by decide), root62Candidates8_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail9
    (if 3 ≤ root62Candidates8.length then search edge 3 [313, 3583] root62Candidates8 3896 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates9 : List Nat := [11941, 15541, 16981]
set_option maxRecDepth 8192 in
theorem root62Candidates9_checked : candidates edge 4 313 3733 root62Tail10 winner.bound = root62Candidates9 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child9 : search edge 3 [313, 3733] root62Candidates9 4046 winner = winner := by decide
theorem root62Step9 : scan edge (search edge 3) 4 [313] 313 root62Tail9 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail10 winner := by
  rw [root62Tail9, scan, if_neg (by decide), root62Candidates9_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail10
    (if 3 ≤ root62Candidates9.length then search edge 3 [313, 3733] root62Candidates9 4046 winner else winner) = _
  rw [if_pos (by decide), root62Child9]
def root62Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root62Candidates10_checked : candidates edge 4 313 3853 root62Tail11 winner.bound = root62Candidates10 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root62Step10 : scan edge (search edge 3) 4 [313] 313 root62Tail10 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail11 winner := by
  rw [root62Tail10, scan, if_neg (by decide), root62Candidates10_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail11
    (if 3 ≤ root62Candidates10.length then search edge 3 [313, 3853] root62Candidates10 4166 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates11 : List Nat := [6037, 14947]
set_option maxRecDepth 8192 in
theorem root62Candidates11_checked : candidates edge 4 313 4231 root62Tail12 winner.bound = root62Candidates11 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root62Step11 : scan edge (search edge 3) 4 [313] 313 root62Tail11 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail12 winner := by
  rw [root62Tail11, scan, if_neg (by decide), root62Candidates11_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail12
    (if 3 ≤ root62Candidates11.length then search edge 3 [313, 4231] root62Candidates11 4544 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates12 : List Nat := [8329, 17683, 17851]
set_option maxRecDepth 8192 in
theorem root62Candidates12_checked : candidates edge 4 313 4591 root62Tail13 winner.bound = root62Candidates12 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root62Child12 : search edge 3 [313, 4591] root62Candidates12 4904 winner = winner := by decide
theorem root62Step12 : scan edge (search edge 3) 4 [313] 313 root62Tail12 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail13 winner := by
  rw [root62Tail12, scan, if_neg (by decide), root62Candidates12_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail13
    (if 3 ≤ root62Candidates12.length then search edge 3 [313, 4591] root62Candidates12 4904 winner else winner) = _
  rw [if_pos (by decide), root62Child12]
def root62Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root62Candidates13_checked : candidates edge 4 313 6037 root62Tail14 winner.bound = root62Candidates13 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root62Step13 : scan edge (search edge 3) 4 [313] 313 root62Tail13 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail14 winner := by
  rw [root62Tail13, scan, if_neg (by decide), root62Candidates13_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail14
    (if 3 ≤ root62Candidates13.length then search edge 3 [313, 6037] root62Candidates13 6350 winner else winner) = _
  rw [if_neg (by decide)]
def root62Candidates14 : List Nat := [9733]
set_option maxRecDepth 8192 in
theorem root62Candidates14_checked : candidates edge 4 313 6079 root62Tail15 winner.bound = root62Candidates14 := by
  unfold candidates
  have row : edge 6079 = (SortedNatTree.contains FastRows.row6079) := by
    change Data.row6079.contains = (SortedNatTree.contains FastRows.row6079)
    exact FastRows.row6079_checked.symm
  rw [row]
  decide
theorem root62Step14 : scan edge (search edge 3) 4 [313] 313 root62Tail14 winner =
    scan edge (search edge 3) 4 [313] 313 root62Tail15 winner := by
  rw [root62Tail14, scan, if_neg (by decide), root62Candidates14_checked]
  change scan edge (search edge 3) 4 [313] 313 root62Tail15
    (if 3 ≤ root62Candidates14.length then search edge 3 [313, 6079] root62Candidates14 6392 winner else winner) = _
  rw [if_neg (by decide)]
theorem root62Scan15 : scan edge (search edge 3) 4 [313] 313 root62Tail15 winner = winner := by rw [root62Tail15, scan, if_pos (by decide)]
theorem root62Scan14 : scan edge (search edge 3) 4 [313] 313 root62Tail14 winner = winner :=
  (root62Step14).trans root62Scan15
theorem root62Scan13 : scan edge (search edge 3) 4 [313] 313 root62Tail13 winner = winner :=
  (root62Step13).trans root62Scan14
theorem root62Scan12 : scan edge (search edge 3) 4 [313] 313 root62Tail12 winner = winner :=
  (root62Step12).trans root62Scan13
theorem root62Scan11 : scan edge (search edge 3) 4 [313] 313 root62Tail11 winner = winner :=
  (root62Step11).trans root62Scan12
theorem root62Scan10 : scan edge (search edge 3) 4 [313] 313 root62Tail10 winner = winner :=
  (root62Step10).trans root62Scan11
theorem root62Scan9 : scan edge (search edge 3) 4 [313] 313 root62Tail9 winner = winner :=
  (root62Step9).trans root62Scan10
theorem root62Scan8 : scan edge (search edge 3) 4 [313] 313 root62Tail8 winner = winner :=
  (root62Step8).trans root62Scan9
theorem root62Scan7 : scan edge (search edge 3) 4 [313] 313 root62Tail7 winner = winner :=
  (root62Step7).trans root62Scan8
theorem root62Scan6 : scan edge (search edge 3) 4 [313] 313 root62Tail6 winner = winner :=
  (root62Step6).trans root62Scan7
theorem root62Scan5 : scan edge (search edge 3) 4 [313] 313 root62Tail5 winner = winner :=
  (root62Step5).trans root62Scan6
theorem root62Scan4 : scan edge (search edge 3) 4 [313] 313 root62Tail4 winner = winner :=
  (root62Step4).trans root62Scan5
theorem root62Scan3 : scan edge (search edge 3) 4 [313] 313 root62Tail3 winner = winner :=
  (root62Step3).trans root62Scan4
theorem root62Scan2 : scan edge (search edge 3) 4 [313] 313 root62Tail2 winner = winner :=
  (root62Step2).trans root62Scan3
theorem root62Scan1 : scan edge (search edge 3) 4 [313] 313 root62Tail1 winner = winner :=
  (root62Step1).trans root62Scan2
theorem root62Scan0 : scan edge (search edge 3) 4 [313] 313 root62Tail0 winner = winner :=
  (root62Step0).trans root62Scan1
set_option maxRecDepth 4096 in
theorem root62Branch : search edge 4 [313] root62Candidates 313 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [313] 313 root62Tail0 winner = winner
  exact root62Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
