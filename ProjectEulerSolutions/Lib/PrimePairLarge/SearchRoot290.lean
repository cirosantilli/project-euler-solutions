import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root290Candidates : List Nat := Data.row1913
set_option maxRecDepth 8192 in
theorem root290Candidates_checked : candidates edge 5 0 1913 tail291 winner.bound = root290Candidates := by
  apply candidates_eq_row edge 5 0 1913 tail291 root290Candidates winner.bound tail291_ordered
  · decide
  · decide
  · funext q
    rfl
def root290Tail49 : List Nat := []
def root290Tail48 : List Nat := 23801 :: root290Tail49
def root290Tail47 : List Nat := 22871 :: root290Tail48
def root290Tail46 : List Nat := 22091 :: root290Tail47
def root290Tail45 : List Nat := 21617 :: root290Tail46
def root290Tail44 : List Nat := 21503 :: root290Tail45
def root290Tail43 : List Nat := 18839 :: root290Tail44
def root290Tail42 : List Nat := 18671 :: root290Tail43
def root290Tail41 : List Nat := 18593 :: root290Tail42
def root290Tail40 : List Nat := 17351 :: root290Tail41
def root290Tail39 : List Nat := 17123 :: root290Tail40
def root290Tail38 : List Nat := 16889 :: root290Tail39
def root290Tail37 : List Nat := 16361 :: root290Tail38
def root290Tail36 : List Nat := 14831 :: root290Tail37
def root290Tail35 : List Nat := 14771 :: root290Tail36
def root290Tail34 : List Nat := 14723 :: root290Tail35
def root290Tail33 : List Nat := 13781 :: root290Tail34
def root290Tail32 : List Nat := 12119 :: root290Tail33
def root290Tail31 : List Nat := 11789 :: root290Tail32
def root290Tail30 : List Nat := 10949 :: root290Tail31
def root290Tail29 : List Nat := 10739 :: root290Tail30
def root290Tail28 : List Nat := 10529 :: root290Tail29
def root290Tail27 : List Nat := 10193 :: root290Tail28
def root290Tail26 : List Nat := 10169 :: root290Tail27
def root290Tail25 : List Nat := 9851 :: root290Tail26
def root290Tail24 : List Nat := 9629 :: root290Tail25
def root290Tail23 : List Nat := 8807 :: root290Tail24
def root290Tail22 : List Nat := 8699 :: root290Tail23
def root290Tail21 : List Nat := 8297 :: root290Tail22
def root290Tail20 : List Nat := 8219 :: root290Tail21
def root290Tail19 : List Nat := 7919 :: root290Tail20
def root290Tail18 : List Nat := 7841 :: root290Tail19
def root290Tail17 : List Nat := 7583 :: root290Tail18
def root290Tail16 : List Nat := 7253 :: root290Tail17
def root290Tail15 : List Nat := 6389 :: root290Tail16
def root290Tail14 : List Nat := 6011 :: root290Tail15
def root290Tail13 : List Nat := 5861 :: root290Tail14
def root290Tail12 : List Nat := 5483 :: root290Tail13
def root290Tail11 : List Nat := 5381 :: root290Tail12
def root290Tail10 : List Nat := 5189 :: root290Tail11
def root290Tail9 : List Nat := 5087 :: root290Tail10
def root290Tail8 : List Nat := 5009 :: root290Tail9
def root290Tail7 : List Nat := 4211 :: root290Tail8
def root290Tail6 : List Nat := 3203 :: root290Tail7
def root290Tail5 : List Nat := 2843 :: root290Tail6
def root290Tail4 : List Nat := 2741 :: root290Tail5
def root290Tail3 : List Nat := 2693 :: root290Tail4
def root290Tail2 : List Nat := 2381 :: root290Tail3
def root290Tail1 : List Nat := 2273 :: root290Tail2
def root290Tail0 : List Nat := 2213 :: root290Tail1
def root290Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root290Candidates0_checked : candidates edge 4 1913 2213 root290Tail1 winner.bound = root290Candidates0 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root290Step0 : scan edge (search edge 3) 4 [1913] 1913 root290Tail0 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail1 winner := by
  rw [root290Tail0, scan, if_neg (by decide), root290Candidates0_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail1
    (if 3 ≤ root290Candidates0.length then search edge 3 [1913, 2213] root290Candidates0 4126 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates1 : List Nat := [5861, 8297, 10739]
set_option maxRecDepth 8192 in
theorem root290Candidates1_checked : candidates edge 4 1913 2273 root290Tail2 winner.bound = root290Candidates1 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root290Child1 : search edge 3 [1913, 2273] root290Candidates1 4186 winner = winner := by decide
theorem root290Step1 : scan edge (search edge 3) 4 [1913] 1913 root290Tail1 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail2 winner := by
  rw [root290Tail1, scan, if_neg (by decide), root290Candidates1_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail2
    (if 3 ≤ root290Candidates1.length then search edge 3 [1913, 2273] root290Candidates1 4186 winner else winner) = _
  rw [if_pos (by decide), root290Child1]
def root290Candidates2 : List Nat := [5483, 8807, 10193, 11789]
set_option maxRecDepth 8192 in
theorem root290Candidates2_checked : candidates edge 4 1913 2381 root290Tail3 winner.bound = root290Candidates2 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root290Child2 : search edge 3 [1913, 2381] root290Candidates2 4294 winner = winner := by decide
theorem root290Step2 : scan edge (search edge 3) 4 [1913] 1913 root290Tail2 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail3 winner := by
  rw [root290Tail2, scan, if_neg (by decide), root290Candidates2_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail3
    (if 3 ≤ root290Candidates2.length then search edge 3 [1913, 2381] root290Candidates2 4294 winner else winner) = _
  rw [if_pos (by decide), root290Child2]
def root290Candidates3 : List Nat := [6011, 18593]
set_option maxRecDepth 8192 in
theorem root290Candidates3_checked : candidates edge 4 1913 2693 root290Tail4 winner.bound = root290Candidates3 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root290Step3 : scan edge (search edge 3) 4 [1913] 1913 root290Tail3 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail4 winner := by
  rw [root290Tail3, scan, if_neg (by decide), root290Candidates3_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail4
    (if 3 ≤ root290Candidates3.length then search edge 3 [1913, 2693] root290Candidates3 4606 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates4 : List Nat := [9851]
set_option maxRecDepth 8192 in
theorem root290Candidates4_checked : candidates edge 4 1913 2741 root290Tail5 winner.bound = root290Candidates4 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root290Step4 : scan edge (search edge 3) 4 [1913] 1913 root290Tail4 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail5 winner := by
  rw [root290Tail4, scan, if_neg (by decide), root290Candidates4_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail5
    (if 3 ≤ root290Candidates4.length then search edge 3 [1913, 2741] root290Candidates4 4654 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates5 : List Nat := [3203, 7841]
set_option maxRecDepth 8192 in
theorem root290Candidates5_checked : candidates edge 4 1913 2843 root290Tail6 winner.bound = root290Candidates5 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root290Step5 : scan edge (search edge 3) 4 [1913] 1913 root290Tail5 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail6 winner := by
  rw [root290Tail5, scan, if_neg (by decide), root290Candidates5_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail6
    (if 3 ≤ root290Candidates5.length then search edge 3 [1913, 2843] root290Candidates5 4756 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates6 : List Nat := [8219, 18839]
set_option maxRecDepth 8192 in
theorem root290Candidates6_checked : candidates edge 4 1913 3203 root290Tail7 winner.bound = root290Candidates6 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
theorem root290Step6 : scan edge (search edge 3) 4 [1913] 1913 root290Tail6 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail7 winner := by
  rw [root290Tail6, scan, if_neg (by decide), root290Candidates6_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail7
    (if 3 ≤ root290Candidates6.length then search edge 3 [1913, 3203] root290Candidates6 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates7 : List Nat := [5483, 18671]
set_option maxRecDepth 8192 in
theorem root290Candidates7_checked : candidates edge 4 1913 4211 root290Tail8 winner.bound = root290Candidates7 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root290Step7 : scan edge (search edge 3) 4 [1913] 1913 root290Tail7 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail8 winner := by
  rw [root290Tail7, scan, if_neg (by decide), root290Candidates7_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail8
    (if 3 ≤ root290Candidates7.length then search edge 3 [1913, 4211] root290Candidates7 6124 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates8 : List Nat := [5861]
set_option maxRecDepth 8192 in
theorem root290Candidates8_checked : candidates edge 4 1913 5009 root290Tail9 winner.bound = root290Candidates8 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root290Step8 : scan edge (search edge 3) 4 [1913] 1913 root290Tail8 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail9 winner := by
  rw [root290Tail8, scan, if_neg (by decide), root290Candidates8_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail9
    (if 3 ≤ root290Candidates8.length then search edge 3 [1913, 5009] root290Candidates8 6922 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates9 : List Nat := [5861, 11789]
set_option maxRecDepth 8192 in
theorem root290Candidates9_checked : candidates edge 4 1913 5087 root290Tail10 winner.bound = root290Candidates9 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root290Step9 : scan edge (search edge 3) 4 [1913] 1913 root290Tail9 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail10 winner := by
  rw [root290Tail9, scan, if_neg (by decide), root290Candidates9_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail10
    (if 3 ≤ root290Candidates9.length then search edge 3 [1913, 5087] root290Candidates9 7000 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates10 : List Nat := [14723, 16889]
set_option maxRecDepth 8192 in
theorem root290Candidates10_checked : candidates edge 4 1913 5189 root290Tail11 winner.bound = root290Candidates10 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root290Step10 : scan edge (search edge 3) 4 [1913] 1913 root290Tail10 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail11 winner := by
  rw [root290Tail10, scan, if_neg (by decide), root290Candidates10_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail11
    (if 3 ≤ root290Candidates10.length then search edge 3 [1913, 5189] root290Candidates10 7102 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates11 : List Nat := [6011, 7919, 9851]
set_option maxRecDepth 8192 in
theorem root290Candidates11_checked : candidates edge 4 1913 5381 root290Tail12 winner.bound = root290Candidates11 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root290Child11 : search edge 3 [1913, 5381] root290Candidates11 7294 winner = winner := by decide
theorem root290Step11 : scan edge (search edge 3) 4 [1913] 1913 root290Tail11 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail12 winner := by
  rw [root290Tail11, scan, if_neg (by decide), root290Candidates11_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail12
    (if 3 ≤ root290Candidates11.length then search edge 3 [1913, 5381] root290Candidates11 7294 winner else winner) = _
  rw [if_pos (by decide), root290Child11]
def root290Candidates12 : List Nat := [8297, 16361]
set_option maxRecDepth 8192 in
theorem root290Candidates12_checked : candidates edge 4 1913 5483 root290Tail13 winner.bound = root290Candidates12 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root290Step12 : scan edge (search edge 3) 4 [1913] 1913 root290Tail12 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail13 winner := by
  rw [root290Tail12, scan, if_neg (by decide), root290Candidates12_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail13
    (if 3 ≤ root290Candidates12.length then search edge 3 [1913, 5483] root290Candidates12 7396 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates13 : List Nat := [10169]
set_option maxRecDepth 8192 in
theorem root290Candidates13_checked : candidates edge 4 1913 5861 root290Tail14 winner.bound = root290Candidates13 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root290Step13 : scan edge (search edge 3) 4 [1913] 1913 root290Tail13 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail14 winner := by
  rw [root290Tail13, scan, if_neg (by decide), root290Candidates13_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail14
    (if 3 ≤ root290Candidates13.length then search edge 3 [1913, 5861] root290Candidates13 7774 winner else winner) = _
  rw [if_neg (by decide)]
def root290Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root290Candidates14_checked : candidates edge 4 1913 6011 root290Tail15 winner.bound = root290Candidates14 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root290Step14 : scan edge (search edge 3) 4 [1913] 1913 root290Tail14 winner =
    scan edge (search edge 3) 4 [1913] 1913 root290Tail15 winner := by
  rw [root290Tail14, scan, if_neg (by decide), root290Candidates14_checked]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail15
    (if 3 ≤ root290Candidates14.length then search edge 3 [1913, 6011] root290Candidates14 7924 winner else winner) = _
  rw [if_neg (by decide)]
theorem root290Scan15 : scan edge (search edge 3) 4 [1913] 1913 root290Tail15 winner = winner := by rw [root290Tail15, scan, if_pos (by decide)]
theorem root290Scan14 : scan edge (search edge 3) 4 [1913] 1913 root290Tail14 winner = winner :=
  (root290Step14).trans root290Scan15
theorem root290Scan13 : scan edge (search edge 3) 4 [1913] 1913 root290Tail13 winner = winner :=
  (root290Step13).trans root290Scan14
theorem root290Scan12 : scan edge (search edge 3) 4 [1913] 1913 root290Tail12 winner = winner :=
  (root290Step12).trans root290Scan13
theorem root290Scan11 : scan edge (search edge 3) 4 [1913] 1913 root290Tail11 winner = winner :=
  (root290Step11).trans root290Scan12
theorem root290Scan10 : scan edge (search edge 3) 4 [1913] 1913 root290Tail10 winner = winner :=
  (root290Step10).trans root290Scan11
theorem root290Scan9 : scan edge (search edge 3) 4 [1913] 1913 root290Tail9 winner = winner :=
  (root290Step9).trans root290Scan10
theorem root290Scan8 : scan edge (search edge 3) 4 [1913] 1913 root290Tail8 winner = winner :=
  (root290Step8).trans root290Scan9
theorem root290Scan7 : scan edge (search edge 3) 4 [1913] 1913 root290Tail7 winner = winner :=
  (root290Step7).trans root290Scan8
theorem root290Scan6 : scan edge (search edge 3) 4 [1913] 1913 root290Tail6 winner = winner :=
  (root290Step6).trans root290Scan7
theorem root290Scan5 : scan edge (search edge 3) 4 [1913] 1913 root290Tail5 winner = winner :=
  (root290Step5).trans root290Scan6
theorem root290Scan4 : scan edge (search edge 3) 4 [1913] 1913 root290Tail4 winner = winner :=
  (root290Step4).trans root290Scan5
theorem root290Scan3 : scan edge (search edge 3) 4 [1913] 1913 root290Tail3 winner = winner :=
  (root290Step3).trans root290Scan4
theorem root290Scan2 : scan edge (search edge 3) 4 [1913] 1913 root290Tail2 winner = winner :=
  (root290Step2).trans root290Scan3
theorem root290Scan1 : scan edge (search edge 3) 4 [1913] 1913 root290Tail1 winner = winner :=
  (root290Step1).trans root290Scan2
theorem root290Scan0 : scan edge (search edge 3) 4 [1913] 1913 root290Tail0 winner = winner :=
  (root290Step0).trans root290Scan1
set_option maxRecDepth 4096 in
theorem root290Branch : search edge 4 [1913] root290Candidates 1913 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1913] 1913 root290Tail0 winner = winner
  exact root290Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
