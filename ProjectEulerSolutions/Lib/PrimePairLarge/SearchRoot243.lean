import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root243Candidates : List Nat := Data.row1559
set_option maxRecDepth 8192 in
theorem root243Candidates_checked : candidates edge 5 0 1559 tail244 winner.bound = root243Candidates := by
  apply candidates_eq_row edge 5 0 1559 tail244 root243Candidates winner.bound tail244_ordered
  · decide
  · decide
  · funext q
    rfl
def root243Tail49 : List Nat := []
def root243Tail48 : List Nat := 23687 :: root243Tail49
def root243Tail47 : List Nat := 22787 :: root243Tail48
def root243Tail46 : List Nat := 22367 :: root243Tail47
def root243Tail45 : List Nat := 20543 :: root243Tail46
def root243Tail44 : List Nat := 20411 :: root243Tail45
def root243Tail43 : List Nat := 20123 :: root243Tail44
def root243Tail42 : List Nat := 19889 :: root243Tail43
def root243Tail41 : List Nat := 19391 :: root243Tail42
def root243Tail40 : List Nat := 18797 :: root243Tail41
def root243Tail39 : List Nat := 18401 :: root243Tail40
def root243Tail38 : List Nat := 18143 :: root243Tail39
def root243Tail37 : List Nat := 18041 :: root243Tail38
def root243Tail36 : List Nat := 17393 :: root243Tail37
def root243Tail35 : List Nat := 17093 :: root243Tail36
def root243Tail34 : List Nat := 16703 :: root243Tail35
def root243Tail33 : List Nat := 16661 :: root243Tail34
def root243Tail32 : List Nat := 16529 :: root243Tail33
def root243Tail31 : List Nat := 15101 :: root243Tail32
def root243Tail30 : List Nat := 13697 :: root243Tail31
def root243Tail29 : List Nat := 13553 :: root243Tail30
def root243Tail28 : List Nat := 13421 :: root243Tail29
def root243Tail27 : List Nat := 13007 :: root243Tail28
def root243Tail26 : List Nat := 12503 :: root243Tail27
def root243Tail25 : List Nat := 12473 :: root243Tail26
def root243Tail24 : List Nat := 12227 :: root243Tail25
def root243Tail23 : List Nat := 12041 :: root243Tail24
def root243Tail22 : List Nat := 10613 :: root243Tail23
def root243Tail21 : List Nat := 9833 :: root243Tail22
def root243Tail20 : List Nat := 9743 :: root243Tail21
def root243Tail19 : List Nat := 9491 :: root243Tail20
def root243Tail18 : List Nat := 9413 :: root243Tail19
def root243Tail17 : List Nat := 9371 :: root243Tail18
def root243Tail16 : List Nat := 9137 :: root243Tail17
def root243Tail15 : List Nat := 8237 :: root243Tail16
def root243Tail14 : List Nat := 8171 :: root243Tail15
def root243Tail13 : List Nat := 7559 :: root243Tail14
def root243Tail12 : List Nat := 7079 :: root243Tail13
def root243Tail11 : List Nat := 6299 :: root243Tail12
def root243Tail10 : List Nat := 5897 :: root243Tail11
def root243Tail9 : List Nat := 5861 :: root243Tail10
def root243Tail8 : List Nat := 5507 :: root243Tail9
def root243Tail7 : List Nat := 5441 :: root243Tail8
def root243Tail6 : List Nat := 3911 :: root243Tail7
def root243Tail5 : List Nat := 3593 :: root243Tail6
def root243Tail4 : List Nat := 3191 :: root243Tail5
def root243Tail3 : List Nat := 2693 :: root243Tail4
def root243Tail2 : List Nat := 2477 :: root243Tail3
def root243Tail1 : List Nat := 1787 :: root243Tail2
def root243Tail0 : List Nat := 1721 :: root243Tail1
def root243Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root243Candidates0_checked : candidates edge 4 1559 1721 root243Tail1 winner.bound = root243Candidates0 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root243Step0 : scan edge (search edge 3) 4 [1559] 1559 root243Tail0 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail1 winner := by
  rw [root243Tail0, scan, if_neg (by decide), root243Candidates0_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail1
    (if 3 ≤ root243Candidates0.length then search edge 3 [1559, 1721] root243Candidates0 3280 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates1 : List Nat := [13553, 18401, 18797]
set_option maxRecDepth 8192 in
theorem root243Candidates1_checked : candidates edge 4 1559 1787 root243Tail2 winner.bound = root243Candidates1 := by
  unfold candidates
  have row : edge 1787 = (SortedNatTree.contains FastRows.row1787) := by
    change Data.row1787.contains = (SortedNatTree.contains FastRows.row1787)
    exact FastRows.row1787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root243Child1 : search edge 3 [1559, 1787] root243Candidates1 3346 winner = winner := by decide
theorem root243Step1 : scan edge (search edge 3) 4 [1559] 1559 root243Tail1 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail2 winner := by
  rw [root243Tail1, scan, if_neg (by decide), root243Candidates1_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail2
    (if 3 ≤ root243Candidates1.length then search edge 3 [1559, 1787] root243Candidates1 3346 winner else winner) = _
  rw [if_pos (by decide), root243Child1]
def root243Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root243Candidates2_checked : candidates edge 4 1559 2477 root243Tail3 winner.bound = root243Candidates2 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
theorem root243Step2 : scan edge (search edge 3) 4 [1559] 1559 root243Tail2 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail3 winner := by
  rw [root243Tail2, scan, if_neg (by decide), root243Candidates2_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail3
    (if 3 ≤ root243Candidates2.length then search edge 3 [1559, 2477] root243Candidates2 4036 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates3 : List Nat := [9833, 13553]
set_option maxRecDepth 8192 in
theorem root243Candidates3_checked : candidates edge 4 1559 2693 root243Tail4 winner.bound = root243Candidates3 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root243Step3 : scan edge (search edge 3) 4 [1559] 1559 root243Tail3 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail4 winner := by
  rw [root243Tail3, scan, if_neg (by decide), root243Candidates3_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail4
    (if 3 ≤ root243Candidates3.length then search edge 3 [1559, 2693] root243Candidates3 4252 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates4 : List Nat := [20411]
set_option maxRecDepth 8192 in
theorem root243Candidates4_checked : candidates edge 4 1559 3191 root243Tail5 winner.bound = root243Candidates4 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
theorem root243Step4 : scan edge (search edge 3) 4 [1559] 1559 root243Tail4 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail5 winner := by
  rw [root243Tail4, scan, if_neg (by decide), root243Candidates4_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail5
    (if 3 ≤ root243Candidates4.length then search edge 3 [1559, 3191] root243Candidates4 4750 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates5 : List Nat := [19391]
set_option maxRecDepth 8192 in
theorem root243Candidates5_checked : candidates edge 4 1559 3593 root243Tail6 winner.bound = root243Candidates5 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root243Step5 : scan edge (search edge 3) 4 [1559] 1559 root243Tail5 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail6 winner := by
  rw [root243Tail5, scan, if_neg (by decide), root243Candidates5_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail6
    (if 3 ≤ root243Candidates5.length then search edge 3 [1559, 3593] root243Candidates5 5152 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root243Candidates6_checked : candidates edge 4 1559 3911 root243Tail7 winner.bound = root243Candidates6 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
theorem root243Step6 : scan edge (search edge 3) 4 [1559] 1559 root243Tail6 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail7 winner := by
  rw [root243Tail6, scan, if_neg (by decide), root243Candidates6_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail7
    (if 3 ≤ root243Candidates6.length then search edge 3 [1559, 3911] root243Candidates6 5470 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates7 : List Nat := [17393]
set_option maxRecDepth 8192 in
theorem root243Candidates7_checked : candidates edge 4 1559 5441 root243Tail8 winner.bound = root243Candidates7 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root243Step7 : scan edge (search edge 3) 4 [1559] 1559 root243Tail7 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail8 winner := by
  rw [root243Tail7, scan, if_neg (by decide), root243Candidates7_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail8
    (if 3 ≤ root243Candidates7.length then search edge 3 [1559, 5441] root243Candidates7 7000 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root243Candidates8_checked : candidates edge 4 1559 5507 root243Tail9 winner.bound = root243Candidates8 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root243Step8 : scan edge (search edge 3) 4 [1559] 1559 root243Tail8 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail9 winner := by
  rw [root243Tail8, scan, if_neg (by decide), root243Candidates8_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail9
    (if 3 ≤ root243Candidates8.length then search edge 3 [1559, 5507] root243Candidates8 7066 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates9 : List Nat := [8171]
set_option maxRecDepth 8192 in
theorem root243Candidates9_checked : candidates edge 4 1559 5861 root243Tail10 winner.bound = root243Candidates9 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root243Step9 : scan edge (search edge 3) 4 [1559] 1559 root243Tail9 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail10 winner := by
  rw [root243Tail9, scan, if_neg (by decide), root243Candidates9_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail10
    (if 3 ≤ root243Candidates9.length then search edge 3 [1559, 5861] root243Candidates9 7420 winner else winner) = _
  rw [if_neg (by decide)]
def root243Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root243Candidates10_checked : candidates edge 4 1559 5897 root243Tail11 winner.bound = root243Candidates10 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root243Step10 : scan edge (search edge 3) 4 [1559] 1559 root243Tail10 winner =
    scan edge (search edge 3) 4 [1559] 1559 root243Tail11 winner := by
  rw [root243Tail10, scan, if_neg (by decide), root243Candidates10_checked]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail11
    (if 3 ≤ root243Candidates10.length then search edge 3 [1559, 5897] root243Candidates10 7456 winner else winner) = _
  rw [if_neg (by decide)]
theorem root243Scan11 : scan edge (search edge 3) 4 [1559] 1559 root243Tail11 winner = winner := by rw [root243Tail11, scan, if_pos (by decide)]
theorem root243Scan10 : scan edge (search edge 3) 4 [1559] 1559 root243Tail10 winner = winner :=
  (root243Step10).trans root243Scan11
theorem root243Scan9 : scan edge (search edge 3) 4 [1559] 1559 root243Tail9 winner = winner :=
  (root243Step9).trans root243Scan10
theorem root243Scan8 : scan edge (search edge 3) 4 [1559] 1559 root243Tail8 winner = winner :=
  (root243Step8).trans root243Scan9
theorem root243Scan7 : scan edge (search edge 3) 4 [1559] 1559 root243Tail7 winner = winner :=
  (root243Step7).trans root243Scan8
theorem root243Scan6 : scan edge (search edge 3) 4 [1559] 1559 root243Tail6 winner = winner :=
  (root243Step6).trans root243Scan7
theorem root243Scan5 : scan edge (search edge 3) 4 [1559] 1559 root243Tail5 winner = winner :=
  (root243Step5).trans root243Scan6
theorem root243Scan4 : scan edge (search edge 3) 4 [1559] 1559 root243Tail4 winner = winner :=
  (root243Step4).trans root243Scan5
theorem root243Scan3 : scan edge (search edge 3) 4 [1559] 1559 root243Tail3 winner = winner :=
  (root243Step3).trans root243Scan4
theorem root243Scan2 : scan edge (search edge 3) 4 [1559] 1559 root243Tail2 winner = winner :=
  (root243Step2).trans root243Scan3
theorem root243Scan1 : scan edge (search edge 3) 4 [1559] 1559 root243Tail1 winner = winner :=
  (root243Step1).trans root243Scan2
theorem root243Scan0 : scan edge (search edge 3) 4 [1559] 1559 root243Tail0 winner = winner :=
  (root243Step0).trans root243Scan1
set_option maxRecDepth 4096 in
theorem root243Branch : search edge 4 [1559] root243Candidates 1559 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1559] 1559 root243Tail0 winner = winner
  exact root243Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
