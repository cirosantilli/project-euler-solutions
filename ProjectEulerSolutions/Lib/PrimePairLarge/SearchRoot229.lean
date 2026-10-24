import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root229Candidates : List Nat := Data.row1459
set_option maxRecDepth 8192 in
theorem root229Candidates_checked : candidates edge 5 0 1459 tail230 winner.bound = root229Candidates := by
  apply candidates_eq_row edge 5 0 1459 tail230 root229Candidates winner.bound tail230_ordered
  · decide
  · decide
  · funext q
    rfl
def root229Tail45 : List Nat := []
def root229Tail44 : List Nat := 24469 :: root229Tail45
def root229Tail43 : List Nat := 24001 :: root229Tail44
def root229Tail42 : List Nat := 22453 :: root229Tail43
def root229Tail41 : List Nat := 22369 :: root229Tail42
def root229Tail40 : List Nat := 21283 :: root229Tail41
def root229Tail39 : List Nat := 20899 :: root229Tail40
def root229Tail38 : List Nat := 20023 :: root229Tail39
def root229Tail37 : List Nat := 20011 :: root229Tail38
def root229Tail36 : List Nat := 19801 :: root229Tail37
def root229Tail35 : List Nat := 19219 :: root229Tail36
def root229Tail34 : List Nat := 18919 :: root229Tail35
def root229Tail33 : List Nat := 18211 :: root229Tail34
def root229Tail32 : List Nat := 18181 :: root229Tail33
def root229Tail31 : List Nat := 17839 :: root229Tail32
def root229Tail30 : List Nat := 17581 :: root229Tail31
def root229Tail29 : List Nat := 17509 :: root229Tail30
def root229Tail28 : List Nat := 17257 :: root229Tail29
def root229Tail27 : List Nat := 16921 :: root229Tail28
def root229Tail26 : List Nat := 16747 :: root229Tail27
def root229Tail25 : List Nat := 15817 :: root229Tail26
def root229Tail24 : List Nat := 15661 :: root229Tail25
def root229Tail23 : List Nat := 15277 :: root229Tail24
def root229Tail22 : List Nat := 14107 :: root229Tail23
def root229Tail21 : List Nat := 12301 :: root229Tail22
def root229Tail20 : List Nat := 11113 :: root229Tail21
def root229Tail19 : List Nat := 10957 :: root229Tail20
def root229Tail18 : List Nat := 10159 :: root229Tail19
def root229Tail17 : List Nat := 9001 :: root229Tail18
def root229Tail16 : List Nat := 7963 :: root229Tail17
def root229Tail15 : List Nat := 7717 :: root229Tail16
def root229Tail14 : List Nat := 7027 :: root229Tail15
def root229Tail13 : List Nat := 6991 :: root229Tail14
def root229Tail12 : List Nat := 6271 :: root229Tail13
def root229Tail11 : List Nat := 6079 :: root229Tail12
def root229Tail10 : List Nat := 5437 :: root229Tail11
def root229Tail9 : List Nat := 5233 :: root229Tail10
def root229Tail8 : List Nat := 5011 :: root229Tail9
def root229Tail7 : List Nat := 4813 :: root229Tail8
def root229Tail6 : List Nat := 3673 :: root229Tail7
def root229Tail5 : List Nat := 3511 :: root229Tail6
def root229Tail4 : List Nat := 2089 :: root229Tail5
def root229Tail3 : List Nat := 1873 :: root229Tail4
def root229Tail2 : List Nat := 1861 :: root229Tail3
def root229Tail1 : List Nat := 1657 :: root229Tail2
def root229Tail0 : List Nat := 1567 :: root229Tail1
def root229Candidates0 : List Nat := [7027]
set_option maxRecDepth 8192 in
theorem root229Candidates0_checked : candidates edge 4 1459 1567 root229Tail1 winner.bound = root229Candidates0 := by
  unfold candidates
  have row : edge 1567 = (SortedNatTree.contains FastRows.row1567) := by
    change Data.row1567.contains = (SortedNatTree.contains FastRows.row1567)
    exact FastRows.row1567_checked.symm
  rw [row]
  decide
theorem root229Step0 : scan edge (search edge 3) 4 [1459] 1459 root229Tail0 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail1 winner := by
  rw [root229Tail0, scan, if_neg (by decide), root229Candidates0_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail1
    (if 3 ≤ root229Candidates0.length then search edge 3 [1459, 1567] root229Candidates0 3026 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates1 : List Nat := [12301]
set_option maxRecDepth 8192 in
theorem root229Candidates1_checked : candidates edge 4 1459 1657 root229Tail2 winner.bound = root229Candidates1 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
theorem root229Step1 : scan edge (search edge 3) 4 [1459] 1459 root229Tail1 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail2 winner := by
  rw [root229Tail1, scan, if_neg (by decide), root229Candidates1_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail2
    (if 3 ≤ root229Candidates1.length then search edge 3 [1459, 1657] root229Candidates1 3116 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates2 : List Nat := [5011, 16747]
set_option maxRecDepth 8192 in
theorem root229Candidates2_checked : candidates edge 4 1459 1861 root229Tail3 winner.bound = root229Candidates2 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
theorem root229Step2 : scan edge (search edge 3) 4 [1459] 1459 root229Tail2 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail3 winner := by
  rw [root229Tail2, scan, if_neg (by decide), root229Candidates2_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail3
    (if 3 ≤ root229Candidates2.length then search edge 3 [1459, 1861] root229Candidates2 3320 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates3 : List Nat := [3511, 7963, 14107]
set_option maxRecDepth 8192 in
theorem root229Candidates3_checked : candidates edge 4 1459 1873 root229Tail4 winner.bound = root229Candidates3 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root229Child3 : search edge 3 [1459, 1873] root229Candidates3 3332 winner = winner := by decide
theorem root229Step3 : scan edge (search edge 3) 4 [1459] 1459 root229Tail3 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail4 winner := by
  rw [root229Tail3, scan, if_neg (by decide), root229Candidates3_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail4
    (if 3 ≤ root229Candidates3.length then search edge 3 [1459, 1873] root229Candidates3 3332 winner else winner) = _
  rw [if_pos (by decide), root229Child3]
def root229Candidates4 : List Nat := [15277, 18181, 22453]
set_option maxRecDepth 8192 in
theorem root229Candidates4_checked : candidates edge 4 1459 2089 root229Tail5 winner.bound = root229Candidates4 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root229Child4 : search edge 3 [1459, 2089] root229Candidates4 3548 winner = winner := by decide
theorem root229Step4 : scan edge (search edge 3) 4 [1459] 1459 root229Tail4 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail5 winner := by
  rw [root229Tail4, scan, if_neg (by decide), root229Candidates4_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail5
    (if 3 ≤ root229Candidates4.length then search edge 3 [1459, 2089] root229Candidates4 3548 winner else winner) = _
  rw [if_pos (by decide), root229Child4]
def root229Candidates5 : List Nat := [5233, 7963, 11113, 20899]
set_option maxRecDepth 8192 in
theorem root229Candidates5_checked : candidates edge 4 1459 3511 root229Tail6 winner.bound = root229Candidates5 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root229Child5 : search edge 3 [1459, 3511] root229Candidates5 4970 winner = winner := by decide
theorem root229Step5 : scan edge (search edge 3) 4 [1459] 1459 root229Tail5 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail6 winner := by
  rw [root229Tail5, scan, if_neg (by decide), root229Candidates5_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail6
    (if 3 ≤ root229Candidates5.length then search edge 3 [1459, 3511] root229Candidates5 4970 winner else winner) = _
  rw [if_pos (by decide), root229Child5]
def root229Candidates6 : List Nat := [7717]
set_option maxRecDepth 8192 in
theorem root229Candidates6_checked : candidates edge 4 1459 3673 root229Tail7 winner.bound = root229Candidates6 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root229Step6 : scan edge (search edge 3) 4 [1459] 1459 root229Tail6 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail7 winner := by
  rw [root229Tail6, scan, if_neg (by decide), root229Candidates6_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail7
    (if 3 ≤ root229Candidates6.length then search edge 3 [1459, 3673] root229Candidates6 5132 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates7 : List Nat := [17257, 18919]
set_option maxRecDepth 8192 in
theorem root229Candidates7_checked : candidates edge 4 1459 4813 root229Tail8 winner.bound = root229Candidates7 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root229Step7 : scan edge (search edge 3) 4 [1459] 1459 root229Tail7 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail8 winner := by
  rw [root229Tail7, scan, if_neg (by decide), root229Candidates7_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail8
    (if 3 ≤ root229Candidates7.length then search edge 3 [1459, 4813] root229Candidates7 6272 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates8 : List Nat := [6271]
set_option maxRecDepth 8192 in
theorem root229Candidates8_checked : candidates edge 4 1459 5011 root229Tail9 winner.bound = root229Candidates8 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root229Step8 : scan edge (search edge 3) 4 [1459] 1459 root229Tail8 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail9 winner := by
  rw [root229Tail8, scan, if_neg (by decide), root229Candidates8_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail9
    (if 3 ≤ root229Candidates8.length then search edge 3 [1459, 5011] root229Candidates8 6470 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root229Candidates9_checked : candidates edge 4 1459 5233 root229Tail10 winner.bound = root229Candidates9 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root229Step9 : scan edge (search edge 3) 4 [1459] 1459 root229Tail9 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail10 winner := by
  rw [root229Tail9, scan, if_neg (by decide), root229Candidates9_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail10
    (if 3 ≤ root229Candidates9.length then search edge 3 [1459, 5233] root229Candidates9 6692 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates10 : List Nat := [9001]
set_option maxRecDepth 8192 in
theorem root229Candidates10_checked : candidates edge 4 1459 5437 root229Tail11 winner.bound = root229Candidates10 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root229Step10 : scan edge (search edge 3) 4 [1459] 1459 root229Tail10 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail11 winner := by
  rw [root229Tail10, scan, if_neg (by decide), root229Candidates10_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail11
    (if 3 ≤ root229Candidates10.length then search edge 3 [1459, 5437] root229Candidates10 6896 winner else winner) = _
  rw [if_neg (by decide)]
def root229Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root229Candidates11_checked : candidates edge 4 1459 6079 root229Tail12 winner.bound = root229Candidates11 := by
  unfold candidates
  have row : edge 6079 = (SortedNatTree.contains FastRows.row6079) := by
    change Data.row6079.contains = (SortedNatTree.contains FastRows.row6079)
    exact FastRows.row6079_checked.symm
  rw [row]
  decide
theorem root229Step11 : scan edge (search edge 3) 4 [1459] 1459 root229Tail11 winner =
    scan edge (search edge 3) 4 [1459] 1459 root229Tail12 winner := by
  rw [root229Tail11, scan, if_neg (by decide), root229Candidates11_checked]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail12
    (if 3 ≤ root229Candidates11.length then search edge 3 [1459, 6079] root229Candidates11 7538 winner else winner) = _
  rw [if_neg (by decide)]
theorem root229Scan12 : scan edge (search edge 3) 4 [1459] 1459 root229Tail12 winner = winner := by rw [root229Tail12, scan, if_pos (by decide)]
theorem root229Scan11 : scan edge (search edge 3) 4 [1459] 1459 root229Tail11 winner = winner :=
  (root229Step11).trans root229Scan12
theorem root229Scan10 : scan edge (search edge 3) 4 [1459] 1459 root229Tail10 winner = winner :=
  (root229Step10).trans root229Scan11
theorem root229Scan9 : scan edge (search edge 3) 4 [1459] 1459 root229Tail9 winner = winner :=
  (root229Step9).trans root229Scan10
theorem root229Scan8 : scan edge (search edge 3) 4 [1459] 1459 root229Tail8 winner = winner :=
  (root229Step8).trans root229Scan9
theorem root229Scan7 : scan edge (search edge 3) 4 [1459] 1459 root229Tail7 winner = winner :=
  (root229Step7).trans root229Scan8
theorem root229Scan6 : scan edge (search edge 3) 4 [1459] 1459 root229Tail6 winner = winner :=
  (root229Step6).trans root229Scan7
theorem root229Scan5 : scan edge (search edge 3) 4 [1459] 1459 root229Tail5 winner = winner :=
  (root229Step5).trans root229Scan6
theorem root229Scan4 : scan edge (search edge 3) 4 [1459] 1459 root229Tail4 winner = winner :=
  (root229Step4).trans root229Scan5
theorem root229Scan3 : scan edge (search edge 3) 4 [1459] 1459 root229Tail3 winner = winner :=
  (root229Step3).trans root229Scan4
theorem root229Scan2 : scan edge (search edge 3) 4 [1459] 1459 root229Tail2 winner = winner :=
  (root229Step2).trans root229Scan3
theorem root229Scan1 : scan edge (search edge 3) 4 [1459] 1459 root229Tail1 winner = winner :=
  (root229Step1).trans root229Scan2
theorem root229Scan0 : scan edge (search edge 3) 4 [1459] 1459 root229Tail0 winner = winner :=
  (root229Step0).trans root229Scan1
set_option maxRecDepth 4096 in
theorem root229Branch : search edge 4 [1459] root229Candidates 1459 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1459] 1459 root229Tail0 winner = winner
  exact root229Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
