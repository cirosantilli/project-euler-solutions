import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root203Candidates : List Nat := Data.row1277
set_option maxRecDepth 8192 in
theorem root203Candidates_checked : candidates edge 5 0 1277 tail204 winner.bound = root203Candidates := by
  apply candidates_eq_row edge 5 0 1277 tail204 root203Candidates winner.bound tail204_ordered
  · decide
  · decide
  · funext q
    rfl
def root203Tail47 : List Nat := []
def root203Tail46 : List Nat := 22157 :: root203Tail47
def root203Tail45 : List Nat := 21563 :: root203Tail46
def root203Tail44 : List Nat := 20897 :: root203Tail45
def root203Tail43 : List Nat := 20807 :: root203Tail44
def root203Tail42 : List Nat := 20639 :: root203Tail43
def root203Tail41 : List Nat := 18911 :: root203Tail42
def root203Tail40 : List Nat := 18443 :: root203Tail41
def root203Tail39 : List Nat := 17729 :: root203Tail40
def root203Tail38 : List Nat := 17519 :: root203Tail39
def root203Tail37 : List Nat := 17351 :: root203Tail38
def root203Tail36 : List Nat := 17033 :: root203Tail37
def root203Tail35 : List Nat := 16097 :: root203Tail36
def root203Tail34 : List Nat := 15881 :: root203Tail35
def root203Tail33 : List Nat := 14831 :: root203Tail34
def root203Tail32 : List Nat := 14723 :: root203Tail33
def root203Tail31 : List Nat := 14207 :: root203Tail32
def root203Tail30 : List Nat := 14051 :: root203Tail31
def root203Tail29 : List Nat := 13127 :: root203Tail30
def root203Tail28 : List Nat := 12569 :: root203Tail29
def root203Tail27 : List Nat := 12161 :: root203Tail28
def root203Tail26 : List Nat := 11399 :: root203Tail27
def root203Tail25 : List Nat := 10601 :: root203Tail26
def root203Tail24 : List Nat := 9539 :: root203Tail25
def root203Tail23 : List Nat := 9209 :: root203Tail24
def root203Tail22 : List Nat := 8741 :: root203Tail23
def root203Tail21 : List Nat := 8369 :: root203Tail22
def root203Tail20 : List Nat := 8291 :: root203Tail21
def root203Tail19 : List Nat := 8147 :: root203Tail20
def root203Tail18 : List Nat := 7703 :: root203Tail19
def root203Tail17 : List Nat := 7577 :: root203Tail18
def root203Tail16 : List Nat := 7559 :: root203Tail17
def root203Tail15 : List Nat := 7529 :: root203Tail16
def root203Tail14 : List Nat := 7433 :: root203Tail15
def root203Tail13 : List Nat := 6761 :: root203Tail14
def root203Tail12 : List Nat := 6353 :: root203Tail13
def root203Tail11 : List Nat := 6173 :: root203Tail12
def root203Tail10 : List Nat := 6053 :: root203Tail11
def root203Tail9 : List Nat := 6011 :: root203Tail10
def root203Tail8 : List Nat := 5003 :: root203Tail9
def root203Tail7 : List Nat := 4967 :: root203Tail8
def root203Tail6 : List Nat := 4889 :: root203Tail7
def root203Tail5 : List Nat := 4787 :: root203Tail6
def root203Tail4 : List Nat := 4457 :: root203Tail5
def root203Tail3 : List Nat := 4049 :: root203Tail4
def root203Tail2 : List Nat := 3359 :: root203Tail3
def root203Tail1 : List Nat := 1523 :: root203Tail2
def root203Tail0 : List Nat := 1511 :: root203Tail1
def root203Candidates0 : List Nat := [4889, 20639]
set_option maxRecDepth 8192 in
theorem root203Candidates0_checked : candidates edge 4 1277 1511 root203Tail1 winner.bound = root203Candidates0 := by
  unfold candidates
  have row : edge 1511 = (SortedNatTree.contains FastRows.row1511) := by
    change Data.row1511.contains = (SortedNatTree.contains FastRows.row1511)
    exact FastRows.row1511_checked.symm
  rw [row]
  decide
theorem root203Step0 : scan edge (search edge 3) 4 [1277] 1277 root203Tail0 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail1 winner := by
  rw [root203Tail0, scan, if_neg (by decide), root203Candidates0_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail1
    (if 3 ≤ root203Candidates0.length then search edge 3 [1277, 1511] root203Candidates0 2788 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates1 : List Nat := [11399]
set_option maxRecDepth 8192 in
theorem root203Candidates1_checked : candidates edge 4 1277 1523 root203Tail2 winner.bound = root203Candidates1 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
theorem root203Step1 : scan edge (search edge 3) 4 [1277] 1277 root203Tail1 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail2 winner := by
  rw [root203Tail1, scan, if_neg (by decide), root203Candidates1_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail2
    (if 3 ≤ root203Candidates1.length then search edge 3 [1277, 1523] root203Candidates1 2800 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root203Candidates2_checked : candidates edge 4 1277 3359 root203Tail3 winner.bound = root203Candidates2 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root203Step2 : scan edge (search edge 3) 4 [1277] 1277 root203Tail2 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail3 winner := by
  rw [root203Tail2, scan, if_neg (by decide), root203Candidates2_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail3
    (if 3 ≤ root203Candidates2.length then search edge 3 [1277, 3359] root203Candidates2 4636 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates3 : List Nat := [7703, 17351]
set_option maxRecDepth 8192 in
theorem root203Candidates3_checked : candidates edge 4 1277 4049 root203Tail4 winner.bound = root203Candidates3 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root203Step3 : scan edge (search edge 3) 4 [1277] 1277 root203Tail3 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail4 winner := by
  rw [root203Tail3, scan, if_neg (by decide), root203Candidates3_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail4
    (if 3 ≤ root203Candidates3.length then search edge 3 [1277, 4049] root203Candidates3 5326 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates4 : List Nat := [7433, 17729]
set_option maxRecDepth 8192 in
theorem root203Candidates4_checked : candidates edge 4 1277 4457 root203Tail5 winner.bound = root203Candidates4 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root203Step4 : scan edge (search edge 3) 4 [1277] 1277 root203Tail4 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail5 winner := by
  rw [root203Tail4, scan, if_neg (by decide), root203Candidates4_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail5
    (if 3 ≤ root203Candidates4.length then search edge 3 [1277, 4457] root203Candidates4 5734 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates5 : List Nat := [7433]
set_option maxRecDepth 8192 in
theorem root203Candidates5_checked : candidates edge 4 1277 4787 root203Tail6 winner.bound = root203Candidates5 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root203Step5 : scan edge (search edge 3) 4 [1277] 1277 root203Tail5 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail6 winner := by
  rw [root203Tail5, scan, if_neg (by decide), root203Candidates5_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail6
    (if 3 ≤ root203Candidates5.length then search edge 3 [1277, 4787] root203Candidates5 6064 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates6 : List Nat := [6053]
set_option maxRecDepth 8192 in
theorem root203Candidates6_checked : candidates edge 4 1277 4889 root203Tail7 winner.bound = root203Candidates6 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root203Step6 : scan edge (search edge 3) 4 [1277] 1277 root203Tail6 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail7 winner := by
  rw [root203Tail6, scan, if_neg (by decide), root203Candidates6_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail7
    (if 3 ≤ root203Candidates6.length then search edge 3 [1277, 4889] root203Candidates6 6166 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root203Candidates7_checked : candidates edge 4 1277 4967 root203Tail8 winner.bound = root203Candidates7 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root203Step7 : scan edge (search edge 3) 4 [1277] 1277 root203Tail7 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail8 winner := by
  rw [root203Tail7, scan, if_neg (by decide), root203Candidates7_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail8
    (if 3 ≤ root203Candidates7.length then search edge 3 [1277, 4967] root203Candidates7 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates8 : List Nat := [7703, 8147, 14051, 14723]
set_option maxRecDepth 8192 in
theorem root203Candidates8_checked : candidates edge 4 1277 5003 root203Tail9 winner.bound = root203Candidates8 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root203Child8 : search edge 3 [1277, 5003] root203Candidates8 6280 winner = winner := by decide
theorem root203Step8 : scan edge (search edge 3) 4 [1277] 1277 root203Tail8 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail9 winner := by
  rw [root203Tail8, scan, if_neg (by decide), root203Candidates8_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail9
    (if 3 ≤ root203Candidates8.length then search edge 3 [1277, 5003] root203Candidates8 6280 winner else winner) = _
  rw [if_pos (by decide), root203Child8]
def root203Candidates9 : List Nat := [8741, 14051]
set_option maxRecDepth 8192 in
theorem root203Candidates9_checked : candidates edge 4 1277 6011 root203Tail10 winner.bound = root203Candidates9 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root203Step9 : scan edge (search edge 3) 4 [1277] 1277 root203Tail9 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail10 winner := by
  rw [root203Tail9, scan, if_neg (by decide), root203Candidates9_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail10
    (if 3 ≤ root203Candidates9.length then search edge 3 [1277, 6011] root203Candidates9 7288 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates10 : List Nat := [6173]
set_option maxRecDepth 8192 in
theorem root203Candidates10_checked : candidates edge 4 1277 6053 root203Tail11 winner.bound = root203Candidates10 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root203Step10 : scan edge (search edge 3) 4 [1277] 1277 root203Tail10 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail11 winner := by
  rw [root203Tail10, scan, if_neg (by decide), root203Candidates10_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail11
    (if 3 ≤ root203Candidates10.length then search edge 3 [1277, 6053] root203Candidates10 7330 winner else winner) = _
  rw [if_neg (by decide)]
def root203Candidates11 : List Nat := [10601]
set_option maxRecDepth 8192 in
theorem root203Candidates11_checked : candidates edge 4 1277 6173 root203Tail12 winner.bound = root203Candidates11 := by
  unfold candidates
  have row : edge 6173 = (SortedNatTree.contains FastRows.row6173) := by
    change Data.row6173.contains = (SortedNatTree.contains FastRows.row6173)
    exact FastRows.row6173_checked.symm
  rw [row]
  decide
theorem root203Step11 : scan edge (search edge 3) 4 [1277] 1277 root203Tail11 winner =
    scan edge (search edge 3) 4 [1277] 1277 root203Tail12 winner := by
  rw [root203Tail11, scan, if_neg (by decide), root203Candidates11_checked]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail12
    (if 3 ≤ root203Candidates11.length then search edge 3 [1277, 6173] root203Candidates11 7450 winner else winner) = _
  rw [if_neg (by decide)]
theorem root203Scan12 : scan edge (search edge 3) 4 [1277] 1277 root203Tail12 winner = winner := by rw [root203Tail12, scan, if_pos (by decide)]
theorem root203Scan11 : scan edge (search edge 3) 4 [1277] 1277 root203Tail11 winner = winner :=
  (root203Step11).trans root203Scan12
theorem root203Scan10 : scan edge (search edge 3) 4 [1277] 1277 root203Tail10 winner = winner :=
  (root203Step10).trans root203Scan11
theorem root203Scan9 : scan edge (search edge 3) 4 [1277] 1277 root203Tail9 winner = winner :=
  (root203Step9).trans root203Scan10
theorem root203Scan8 : scan edge (search edge 3) 4 [1277] 1277 root203Tail8 winner = winner :=
  (root203Step8).trans root203Scan9
theorem root203Scan7 : scan edge (search edge 3) 4 [1277] 1277 root203Tail7 winner = winner :=
  (root203Step7).trans root203Scan8
theorem root203Scan6 : scan edge (search edge 3) 4 [1277] 1277 root203Tail6 winner = winner :=
  (root203Step6).trans root203Scan7
theorem root203Scan5 : scan edge (search edge 3) 4 [1277] 1277 root203Tail5 winner = winner :=
  (root203Step5).trans root203Scan6
theorem root203Scan4 : scan edge (search edge 3) 4 [1277] 1277 root203Tail4 winner = winner :=
  (root203Step4).trans root203Scan5
theorem root203Scan3 : scan edge (search edge 3) 4 [1277] 1277 root203Tail3 winner = winner :=
  (root203Step3).trans root203Scan4
theorem root203Scan2 : scan edge (search edge 3) 4 [1277] 1277 root203Tail2 winner = winner :=
  (root203Step2).trans root203Scan3
theorem root203Scan1 : scan edge (search edge 3) 4 [1277] 1277 root203Tail1 winner = winner :=
  (root203Step1).trans root203Scan2
theorem root203Scan0 : scan edge (search edge 3) 4 [1277] 1277 root203Tail0 winner = winner :=
  (root203Step0).trans root203Scan1
set_option maxRecDepth 4096 in
theorem root203Branch : search edge 4 [1277] root203Candidates 1277 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1277] 1277 root203Tail0 winner = winner
  exact root203Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
