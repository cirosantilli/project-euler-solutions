import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root113Candidates : List Nat := Data.row641
set_option maxRecDepth 8192 in
theorem root113Candidates_checked : candidates edge 5 0 641 tail114 winner.bound = root113Candidates := by
  apply candidates_eq_row edge 5 0 641 tail114 root113Candidates winner.bound tail114_ordered
  · decide
  · decide
  · funext q
    rfl
def root113Tail63 : List Nat := []
def root113Tail62 : List Nat := 24407 :: root113Tail63
def root113Tail61 : List Nat := 24317 :: root113Tail62
def root113Tail60 : List Nat := 23831 :: root113Tail61
def root113Tail59 : List Nat := 23819 :: root113Tail60
def root113Tail58 : List Nat := 23663 :: root113Tail59
def root113Tail57 : List Nat := 23459 :: root113Tail58
def root113Tail56 : List Nat := 23399 :: root113Tail57
def root113Tail55 : List Nat := 23327 :: root113Tail56
def root113Tail54 : List Nat := 23291 :: root113Tail55
def root113Tail53 : List Nat := 22469 :: root113Tail54
def root113Tail52 : List Nat := 22433 :: root113Tail53
def root113Tail51 : List Nat := 22109 :: root113Tail52
def root113Tail50 : List Nat := 22031 :: root113Tail51
def root113Tail49 : List Nat := 20477 :: root113Tail50
def root113Tail48 : List Nat := 20249 :: root113Tail49
def root113Tail47 : List Nat := 20051 :: root113Tail48
def root113Tail46 : List Nat := 19463 :: root113Tail47
def root113Tail45 : List Nat := 18731 :: root113Tail46
def root113Tail44 : List Nat := 18149 :: root113Tail45
def root113Tail43 : List Nat := 17519 :: root113Tail44
def root113Tail42 : List Nat := 16901 :: root113Tail43
def root113Tail41 : List Nat := 16487 :: root113Tail42
def root113Tail40 : List Nat := 16229 :: root113Tail41
def root113Tail39 : List Nat := 15683 :: root113Tail40
def root113Tail38 : List Nat := 15647 :: root113Tail39
def root113Tail37 : List Nat := 15377 :: root113Tail38
def root113Tail36 : List Nat := 14489 :: root113Tail37
def root113Tail35 : List Nat := 14411 :: root113Tail36
def root113Tail34 : List Nat := 14387 :: root113Tail35
def root113Tail33 : List Nat := 14051 :: root113Tail34
def root113Tail32 : List Nat := 14009 :: root113Tail33
def root113Tail31 : List Nat := 13697 :: root113Tail32
def root113Tail30 : List Nat := 13523 :: root113Tail31
def root113Tail29 : List Nat := 13331 :: root113Tail30
def root113Tail28 : List Nat := 12641 :: root113Tail29
def root113Tail27 : List Nat := 12239 :: root113Tail28
def root113Tail26 : List Nat := 11783 :: root113Tail27
def root113Tail25 : List Nat := 11657 :: root113Tail26
def root113Tail24 : List Nat := 10853 :: root113Tail25
def root113Tail23 : List Nat := 10667 :: root113Tail24
def root113Tail22 : List Nat := 10601 :: root113Tail23
def root113Tail21 : List Nat := 9467 :: root113Tail22
def root113Tail20 : List Nat := 9227 :: root113Tail21
def root113Tail19 : List Nat := 9173 :: root113Tail20
def root113Tail18 : List Nat := 8543 :: root113Tail19
def root113Tail17 : List Nat := 7457 :: root113Tail18
def root113Tail16 : List Nat := 7247 :: root113Tail17
def root113Tail15 : List Nat := 6653 :: root113Tail16
def root113Tail14 : List Nat := 6203 :: root113Tail15
def root113Tail13 : List Nat := 6197 :: root113Tail14
def root113Tail12 : List Nat := 5693 :: root113Tail13
def root113Tail11 : List Nat := 5441 :: root113Tail12
def root113Tail10 : List Nat := 5237 :: root113Tail11
def root113Tail9 : List Nat := 4937 :: root113Tail10
def root113Tail8 : List Nat := 4787 :: root113Tail9
def root113Tail7 : List Nat := 4013 :: root113Tail8
def root113Tail6 : List Nat := 2957 :: root113Tail7
def root113Tail5 : List Nat := 2801 :: root113Tail6
def root113Tail4 : List Nat := 2543 :: root113Tail5
def root113Tail3 : List Nat := 929 :: root113Tail4
def root113Tail2 : List Nat := 881 :: root113Tail3
def root113Tail1 : List Nat := 863 :: root113Tail2
def root113Tail0 : List Nat := 821 :: root113Tail1
def root113Candidates0 : List Nat := [16901]
set_option maxRecDepth 8192 in
theorem root113Candidates0_checked : candidates edge 4 641 821 root113Tail1 winner.bound = root113Candidates0 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
theorem root113Step0 : scan edge (search edge 3) 4 [641] 641 root113Tail0 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail1 winner := by
  rw [root113Tail0, scan, if_neg (by decide), root113Candidates0_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail1
    (if 3 ≤ root113Candidates0.length then search edge 3 [641, 821] root113Candidates0 1462 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates1 : List Nat := [2801, 18731, 19463]
set_option maxRecDepth 8192 in
theorem root113Candidates1_checked : candidates edge 4 641 863 root113Tail2 winner.bound = root113Candidates1 := by
  unfold candidates
  have row : edge 863 = (SortedNatTree.contains FastRows.row863) := by
    change Data.row863.contains = (SortedNatTree.contains FastRows.row863)
    exact FastRows.row863_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root113Child1 : search edge 3 [641, 863] root113Candidates1 1504 winner = winner := by decide
theorem root113Step1 : scan edge (search edge 3) 4 [641] 641 root113Tail1 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail2 winner := by
  rw [root113Tail1, scan, if_neg (by decide), root113Candidates1_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail2
    (if 3 ≤ root113Candidates1.length then search edge 3 [641, 863] root113Candidates1 1504 winner else winner) = _
  rw [if_pos (by decide), root113Child1]
def root113Candidates2 : List Nat := [16487]
set_option maxRecDepth 8192 in
theorem root113Candidates2_checked : candidates edge 4 641 881 root113Tail3 winner.bound = root113Candidates2 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
theorem root113Step2 : scan edge (search edge 3) 4 [641] 641 root113Tail2 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail3 winner := by
  rw [root113Tail2, scan, if_neg (by decide), root113Candidates2_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail3
    (if 3 ≤ root113Candidates2.length then search edge 3 [641, 881] root113Candidates2 1522 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates3 : List Nat := [4013]
set_option maxRecDepth 8192 in
theorem root113Candidates3_checked : candidates edge 4 641 929 root113Tail4 winner.bound = root113Candidates3 := by
  unfold candidates
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row]
  decide
theorem root113Step3 : scan edge (search edge 3) 4 [641] 641 root113Tail3 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail4 winner := by
  rw [root113Tail3, scan, if_neg (by decide), root113Candidates3_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail4
    (if 3 ≤ root113Candidates3.length then search edge 3 [641, 929] root113Candidates3 1570 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates4 : List Nat := [5237, 5441, 17519]
set_option maxRecDepth 8192 in
theorem root113Candidates4_checked : candidates edge 4 641 2543 root113Tail5 winner.bound = root113Candidates4 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root113Child4 : search edge 3 [641, 2543] root113Candidates4 3184 winner = winner := by decide
theorem root113Step4 : scan edge (search edge 3) 4 [641] 641 root113Tail4 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail5 winner := by
  rw [root113Tail4, scan, if_neg (by decide), root113Candidates4_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail5
    (if 3 ≤ root113Candidates4.length then search edge 3 [641, 2543] root113Candidates4 3184 winner else winner) = _
  rw [if_pos (by decide), root113Child4]
def root113Candidates5 : List Nat := [5693, 10667]
set_option maxRecDepth 8192 in
theorem root113Candidates5_checked : candidates edge 4 641 2801 root113Tail6 winner.bound = root113Candidates5 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
theorem root113Step5 : scan edge (search edge 3) 4 [641] 641 root113Tail5 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail6 winner := by
  rw [root113Tail5, scan, if_neg (by decide), root113Candidates5_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail6
    (if 3 ≤ root113Candidates5.length then search edge 3 [641, 2801] root113Candidates5 3442 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates6 : List Nat := [8543, 14411]
set_option maxRecDepth 8192 in
theorem root113Candidates6_checked : candidates edge 4 641 2957 root113Tail7 winner.bound = root113Candidates6 := by
  unfold candidates
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row]
  decide
theorem root113Step6 : scan edge (search edge 3) 4 [641] 641 root113Tail6 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail7 winner := by
  rw [root113Tail6, scan, if_neg (by decide), root113Candidates6_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail7
    (if 3 ≤ root113Candidates6.length then search edge 3 [641, 2957] root113Candidates6 3598 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root113Candidates7_checked : candidates edge 4 641 4013 root113Tail8 winner.bound = root113Candidates7 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root113Step7 : scan edge (search edge 3) 4 [641] 641 root113Tail7 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail8 winner := by
  rw [root113Tail7, scan, if_neg (by decide), root113Candidates7_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail8
    (if 3 ≤ root113Candidates7.length then search edge 3 [641, 4013] root113Candidates7 4654 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates8 : List Nat := [5237]
set_option maxRecDepth 8192 in
theorem root113Candidates8_checked : candidates edge 4 641 4787 root113Tail9 winner.bound = root113Candidates8 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root113Step8 : scan edge (search edge 3) 4 [641] 641 root113Tail8 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail9 winner := by
  rw [root113Tail8, scan, if_neg (by decide), root113Candidates8_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail9
    (if 3 ≤ root113Candidates8.length then search edge 3 [641, 4787] root113Candidates8 5428 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates9 : List Nat := [7247, 13523]
set_option maxRecDepth 8192 in
theorem root113Candidates9_checked : candidates edge 4 641 4937 root113Tail10 winner.bound = root113Candidates9 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root113Step9 : scan edge (search edge 3) 4 [641] 641 root113Tail9 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail10 winner := by
  rw [root113Tail9, scan, if_neg (by decide), root113Candidates9_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail10
    (if 3 ≤ root113Candidates9.length then search edge 3 [641, 4937] root113Candidates9 5578 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates10 : List Nat := [5693]
set_option maxRecDepth 8192 in
theorem root113Candidates10_checked : candidates edge 4 641 5237 root113Tail11 winner.bound = root113Candidates10 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root113Step10 : scan edge (search edge 3) 4 [641] 641 root113Tail10 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail11 winner := by
  rw [root113Tail10, scan, if_neg (by decide), root113Candidates10_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail11
    (if 3 ≤ root113Candidates10.length then search edge 3 [641, 5237] root113Candidates10 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates11 : List Nat := [5693, 7457, 11657]
set_option maxRecDepth 8192 in
theorem root113Candidates11_checked : candidates edge 4 641 5441 root113Tail12 winner.bound = root113Candidates11 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root113Child11 : search edge 3 [641, 5441] root113Candidates11 6082 winner = winner := by decide
theorem root113Step11 : scan edge (search edge 3) 4 [641] 641 root113Tail11 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail12 winner := by
  rw [root113Tail11, scan, if_neg (by decide), root113Candidates11_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail12
    (if 3 ≤ root113Candidates11.length then search edge 3 [641, 5441] root113Candidates11 6082 winner else winner) = _
  rw [if_pos (by decide), root113Child11]
def root113Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root113Candidates12_checked : candidates edge 4 641 5693 root113Tail13 winner.bound = root113Candidates12 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root113Step12 : scan edge (search edge 3) 4 [641] 641 root113Tail12 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail13 winner := by
  rw [root113Tail12, scan, if_neg (by decide), root113Candidates12_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail13
    (if 3 ≤ root113Candidates12.length then search edge 3 [641, 5693] root113Candidates12 6334 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates13 : List Nat := [15377]
set_option maxRecDepth 8192 in
theorem root113Candidates13_checked : candidates edge 4 641 6197 root113Tail14 winner.bound = root113Candidates13 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root113Step13 : scan edge (search edge 3) 4 [641] 641 root113Tail13 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail14 winner := by
  rw [root113Tail13, scan, if_neg (by decide), root113Candidates13_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail14
    (if 3 ≤ root113Candidates13.length then search edge 3 [641, 6197] root113Candidates13 6838 winner else winner) = _
  rw [if_neg (by decide)]
def root113Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root113Candidates14_checked : candidates edge 4 641 6203 root113Tail15 winner.bound = root113Candidates14 := by
  unfold candidates
  have row : edge 6203 = (SortedNatTree.contains FastRows.row6203) := by
    change Data.row6203.contains = (SortedNatTree.contains FastRows.row6203)
    exact FastRows.row6203_checked.symm
  rw [row]
  decide
theorem root113Step14 : scan edge (search edge 3) 4 [641] 641 root113Tail14 winner =
    scan edge (search edge 3) 4 [641] 641 root113Tail15 winner := by
  rw [root113Tail14, scan, if_neg (by decide), root113Candidates14_checked]
  change scan edge (search edge 3) 4 [641] 641 root113Tail15
    (if 3 ≤ root113Candidates14.length then search edge 3 [641, 6203] root113Candidates14 6844 winner else winner) = _
  rw [if_neg (by decide)]
theorem root113Scan15 : scan edge (search edge 3) 4 [641] 641 root113Tail15 winner = winner := by rw [root113Tail15, scan, if_pos (by decide)]
theorem root113Scan14 : scan edge (search edge 3) 4 [641] 641 root113Tail14 winner = winner :=
  (root113Step14).trans root113Scan15
theorem root113Scan13 : scan edge (search edge 3) 4 [641] 641 root113Tail13 winner = winner :=
  (root113Step13).trans root113Scan14
theorem root113Scan12 : scan edge (search edge 3) 4 [641] 641 root113Tail12 winner = winner :=
  (root113Step12).trans root113Scan13
theorem root113Scan11 : scan edge (search edge 3) 4 [641] 641 root113Tail11 winner = winner :=
  (root113Step11).trans root113Scan12
theorem root113Scan10 : scan edge (search edge 3) 4 [641] 641 root113Tail10 winner = winner :=
  (root113Step10).trans root113Scan11
theorem root113Scan9 : scan edge (search edge 3) 4 [641] 641 root113Tail9 winner = winner :=
  (root113Step9).trans root113Scan10
theorem root113Scan8 : scan edge (search edge 3) 4 [641] 641 root113Tail8 winner = winner :=
  (root113Step8).trans root113Scan9
theorem root113Scan7 : scan edge (search edge 3) 4 [641] 641 root113Tail7 winner = winner :=
  (root113Step7).trans root113Scan8
theorem root113Scan6 : scan edge (search edge 3) 4 [641] 641 root113Tail6 winner = winner :=
  (root113Step6).trans root113Scan7
theorem root113Scan5 : scan edge (search edge 3) 4 [641] 641 root113Tail5 winner = winner :=
  (root113Step5).trans root113Scan6
theorem root113Scan4 : scan edge (search edge 3) 4 [641] 641 root113Tail4 winner = winner :=
  (root113Step4).trans root113Scan5
theorem root113Scan3 : scan edge (search edge 3) 4 [641] 641 root113Tail3 winner = winner :=
  (root113Step3).trans root113Scan4
theorem root113Scan2 : scan edge (search edge 3) 4 [641] 641 root113Tail2 winner = winner :=
  (root113Step2).trans root113Scan3
theorem root113Scan1 : scan edge (search edge 3) 4 [641] 641 root113Tail1 winner = winner :=
  (root113Step1).trans root113Scan2
theorem root113Scan0 : scan edge (search edge 3) 4 [641] 641 root113Tail0 winner = winner :=
  (root113Step0).trans root113Scan1
set_option maxRecDepth 4096 in
theorem root113Branch : search edge 4 [641] root113Candidates 641 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [641] 641 root113Tail0 winner = winner
  exact root113Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
