import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root337Candidates : List Nat := Data.row2287
set_option maxRecDepth 8192 in
theorem root337Candidates_checked : candidates edge 5 0 2287 tail338 winner.bound = root337Candidates := by
  apply candidates_eq_row edge 5 0 2287 tail338 root337Candidates winner.bound tail338_ordered
  · decide
  · decide
  · funext q
    rfl
def root337Tail47 : List Nat := []
def root337Tail46 : List Nat := 23371 :: root337Tail47
def root337Tail45 : List Nat := 22861 :: root337Tail46
def root337Tail44 : List Nat := 22147 :: root337Tail45
def root337Tail43 : List Nat := 22129 :: root337Tail44
def root337Tail42 : List Nat := 21841 :: root337Tail43
def root337Tail41 : List Nat := 21481 :: root337Tail42
def root337Tail40 : List Nat := 21313 :: root337Tail41
def root337Tail39 : List Nat := 20707 :: root337Tail40
def root337Tail38 : List Nat := 20641 :: root337Tail39
def root337Tail37 : List Nat := 19423 :: root337Tail38
def root337Tail36 : List Nat := 18013 :: root337Tail37
def root337Tail35 : List Nat := 16843 :: root337Tail36
def root337Tail34 : List Nat := 16231 :: root337Tail35
def root337Tail33 : List Nat := 14713 :: root337Tail34
def root337Tail32 : List Nat := 14449 :: root337Tail33
def root337Tail31 : List Nat := 13999 :: root337Tail32
def root337Tail30 : List Nat := 12553 :: root337Tail31
def root337Tail29 : List Nat := 12049 :: root337Tail30
def root337Tail28 : List Nat := 11503 :: root337Tail29
def root337Tail27 : List Nat := 11437 :: root337Tail28
def root337Tail26 : List Nat := 11173 :: root337Tail27
def root337Tail25 : List Nat := 10663 :: root337Tail26
def root337Tail24 : List Nat := 10453 :: root337Tail25
def root337Tail23 : List Nat := 9613 :: root337Tail24
def root337Tail22 : List Nat := 9049 :: root337Tail23
def root337Tail21 : List Nat := 9007 :: root337Tail22
def root337Tail20 : List Nat := 8377 :: root337Tail21
def root337Tail19 : List Nat := 8221 :: root337Tail20
def root337Tail18 : List Nat := 8101 :: root337Tail19
def root337Tail17 : List Nat := 7681 :: root337Tail18
def root337Tail16 : List Nat := 7537 :: root337Tail17
def root337Tail15 : List Nat := 6793 :: root337Tail16
def root337Tail14 : List Nat := 6277 :: root337Tail15
def root337Tail13 : List Nat := 5857 :: root337Tail14
def root337Tail12 : List Nat := 5431 :: root337Tail13
def root337Tail11 : List Nat := 5077 :: root337Tail12
def root337Tail10 : List Nat := 5059 :: root337Tail11
def root337Tail9 : List Nat := 4933 :: root337Tail10
def root337Tail8 : List Nat := 4861 :: root337Tail9
def root337Tail7 : List Nat := 4483 :: root337Tail8
def root337Tail6 : List Nat := 4339 :: root337Tail7
def root337Tail5 : List Nat := 3943 :: root337Tail6
def root337Tail4 : List Nat := 3607 :: root337Tail5
def root337Tail3 : List Nat := 3457 :: root337Tail4
def root337Tail2 : List Nat := 3187 :: root337Tail3
def root337Tail1 : List Nat := 2683 :: root337Tail2
def root337Tail0 : List Nat := 2341 :: root337Tail1
def root337Candidates0 : List Nat := [8377, 21313]
set_option maxRecDepth 8192 in
theorem root337Candidates0_checked : candidates edge 4 2287 2341 root337Tail1 winner.bound = root337Candidates0 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root337Step0 : scan edge (search edge 3) 4 [2287] 2287 root337Tail0 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail1 winner := by
  rw [root337Tail0, scan, if_neg (by decide), root337Candidates0_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail1
    (if 3 ≤ root337Candidates0.length then search edge 3 [2287, 2341] root337Candidates0 4628 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root337Candidates1_checked : candidates edge 4 2287 2683 root337Tail2 winner.bound = root337Candidates1 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root337Step1 : scan edge (search edge 3) 4 [2287] 2287 root337Tail1 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail2 winner := by
  rw [root337Tail1, scan, if_neg (by decide), root337Candidates1_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail2
    (if 3 ≤ root337Candidates1.length then search edge 3 [2287, 2683] root337Candidates1 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates2 : List Nat := [8101, 9049]
set_option maxRecDepth 8192 in
theorem root337Candidates2_checked : candidates edge 4 2287 3187 root337Tail3 winner.bound = root337Candidates2 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root337Step2 : scan edge (search edge 3) 4 [2287] 2287 root337Tail2 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail3 winner := by
  rw [root337Tail2, scan, if_neg (by decide), root337Candidates2_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail3
    (if 3 ≤ root337Candidates2.length then search edge 3 [2287, 3187] root337Candidates2 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates3 : List Nat := [8377]
set_option maxRecDepth 8192 in
theorem root337Candidates3_checked : candidates edge 4 2287 3457 root337Tail4 winner.bound = root337Candidates3 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root337Step3 : scan edge (search edge 3) 4 [2287] 2287 root337Tail3 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail4 winner := by
  rw [root337Tail3, scan, if_neg (by decide), root337Candidates3_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail4
    (if 3 ≤ root337Candidates3.length then search edge 3 [2287, 3457] root337Candidates3 5744 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates4 : List Nat := [5077, 8101]
set_option maxRecDepth 8192 in
theorem root337Candidates4_checked : candidates edge 4 2287 3607 root337Tail5 winner.bound = root337Candidates4 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root337Step4 : scan edge (search edge 3) 4 [2287] 2287 root337Tail4 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail5 winner := by
  rw [root337Tail4, scan, if_neg (by decide), root337Candidates4_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail5
    (if 3 ≤ root337Candidates4.length then search edge 3 [2287, 3607] root337Candidates4 5894 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates5 : List Nat := [12553]
set_option maxRecDepth 8192 in
theorem root337Candidates5_checked : candidates edge 4 2287 3943 root337Tail6 winner.bound = root337Candidates5 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root337Step5 : scan edge (search edge 3) 4 [2287] 2287 root337Tail5 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail6 winner := by
  rw [root337Tail5, scan, if_neg (by decide), root337Candidates5_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail6
    (if 3 ≤ root337Candidates5.length then search edge 3 [2287, 3943] root337Candidates5 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root337Candidates6_checked : candidates edge 4 2287 4339 root337Tail7 winner.bound = root337Candidates6 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root337Step6 : scan edge (search edge 3) 4 [2287] 2287 root337Tail6 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail7 winner := by
  rw [root337Tail6, scan, if_neg (by decide), root337Candidates6_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail7
    (if 3 ≤ root337Candidates6.length then search edge 3 [2287, 4339] root337Candidates6 6626 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates7 : List Nat := [6793, 7537, 9049]
set_option maxRecDepth 8192 in
theorem root337Candidates7_checked : candidates edge 4 2287 4483 root337Tail8 winner.bound = root337Candidates7 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root337Child7 : search edge 3 [2287, 4483] root337Candidates7 6770 winner = winner := by decide
theorem root337Step7 : scan edge (search edge 3) 4 [2287] 2287 root337Tail7 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail8 winner := by
  rw [root337Tail7, scan, if_neg (by decide), root337Candidates7_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail8
    (if 3 ≤ root337Candidates7.length then search edge 3 [2287, 4483] root337Candidates7 6770 winner else winner) = _
  rw [if_pos (by decide), root337Child7]
def root337Candidates8 : List Nat := [9007]
set_option maxRecDepth 8192 in
theorem root337Candidates8_checked : candidates edge 4 2287 4861 root337Tail9 winner.bound = root337Candidates8 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root337Step8 : scan edge (search edge 3) 4 [2287] 2287 root337Tail8 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail9 winner := by
  rw [root337Tail8, scan, if_neg (by decide), root337Candidates8_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail9
    (if 3 ≤ root337Candidates8.length then search edge 3 [2287, 4861] root337Candidates8 7148 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root337Candidates9_checked : candidates edge 4 2287 4933 root337Tail10 winner.bound = root337Candidates9 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root337Step9 : scan edge (search edge 3) 4 [2287] 2287 root337Tail9 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail10 winner := by
  rw [root337Tail9, scan, if_neg (by decide), root337Candidates9_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail10
    (if 3 ≤ root337Candidates9.length then search edge 3 [2287, 4933] root337Candidates9 7220 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates10 : List Nat := [9007, 9613, 11437]
set_option maxRecDepth 8192 in
theorem root337Candidates10_checked : candidates edge 4 2287 5059 root337Tail11 winner.bound = root337Candidates10 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root337Child10 : search edge 3 [2287, 5059] root337Candidates10 7346 winner = winner := by decide
theorem root337Step10 : scan edge (search edge 3) 4 [2287] 2287 root337Tail10 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail11 winner := by
  rw [root337Tail10, scan, if_neg (by decide), root337Candidates10_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail11
    (if 3 ≤ root337Candidates10.length then search edge 3 [2287, 5059] root337Candidates10 7346 winner else winner) = _
  rw [if_pos (by decide), root337Child10]
def root337Candidates11 : List Nat := [10663, 16843]
set_option maxRecDepth 8192 in
theorem root337Candidates11_checked : candidates edge 4 2287 5077 root337Tail12 winner.bound = root337Candidates11 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root337Step11 : scan edge (search edge 3) 4 [2287] 2287 root337Tail11 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail12 winner := by
  rw [root337Tail11, scan, if_neg (by decide), root337Candidates11_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail12
    (if 3 ≤ root337Candidates11.length then search edge 3 [2287, 5077] root337Candidates11 7364 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root337Candidates12_checked : candidates edge 4 2287 5431 root337Tail13 winner.bound = root337Candidates12 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root337Step12 : scan edge (search edge 3) 4 [2287] 2287 root337Tail12 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail13 winner := by
  rw [root337Tail12, scan, if_neg (by decide), root337Candidates12_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail13
    (if 3 ≤ root337Candidates12.length then search edge 3 [2287, 5431] root337Candidates12 7718 winner else winner) = _
  rw [if_neg (by decide)]
def root337Candidates13 : List Nat := [9007]
set_option maxRecDepth 8192 in
theorem root337Candidates13_checked : candidates edge 4 2287 5857 root337Tail14 winner.bound = root337Candidates13 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root337Step13 : scan edge (search edge 3) 4 [2287] 2287 root337Tail13 winner =
    scan edge (search edge 3) 4 [2287] 2287 root337Tail14 winner := by
  rw [root337Tail13, scan, if_neg (by decide), root337Candidates13_checked]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail14
    (if 3 ≤ root337Candidates13.length then search edge 3 [2287, 5857] root337Candidates13 8144 winner else winner) = _
  rw [if_neg (by decide)]
theorem root337Scan14 : scan edge (search edge 3) 4 [2287] 2287 root337Tail14 winner = winner := by rw [root337Tail14, scan, if_pos (by decide)]
theorem root337Scan13 : scan edge (search edge 3) 4 [2287] 2287 root337Tail13 winner = winner :=
  (root337Step13).trans root337Scan14
theorem root337Scan12 : scan edge (search edge 3) 4 [2287] 2287 root337Tail12 winner = winner :=
  (root337Step12).trans root337Scan13
theorem root337Scan11 : scan edge (search edge 3) 4 [2287] 2287 root337Tail11 winner = winner :=
  (root337Step11).trans root337Scan12
theorem root337Scan10 : scan edge (search edge 3) 4 [2287] 2287 root337Tail10 winner = winner :=
  (root337Step10).trans root337Scan11
theorem root337Scan9 : scan edge (search edge 3) 4 [2287] 2287 root337Tail9 winner = winner :=
  (root337Step9).trans root337Scan10
theorem root337Scan8 : scan edge (search edge 3) 4 [2287] 2287 root337Tail8 winner = winner :=
  (root337Step8).trans root337Scan9
theorem root337Scan7 : scan edge (search edge 3) 4 [2287] 2287 root337Tail7 winner = winner :=
  (root337Step7).trans root337Scan8
theorem root337Scan6 : scan edge (search edge 3) 4 [2287] 2287 root337Tail6 winner = winner :=
  (root337Step6).trans root337Scan7
theorem root337Scan5 : scan edge (search edge 3) 4 [2287] 2287 root337Tail5 winner = winner :=
  (root337Step5).trans root337Scan6
theorem root337Scan4 : scan edge (search edge 3) 4 [2287] 2287 root337Tail4 winner = winner :=
  (root337Step4).trans root337Scan5
theorem root337Scan3 : scan edge (search edge 3) 4 [2287] 2287 root337Tail3 winner = winner :=
  (root337Step3).trans root337Scan4
theorem root337Scan2 : scan edge (search edge 3) 4 [2287] 2287 root337Tail2 winner = winner :=
  (root337Step2).trans root337Scan3
theorem root337Scan1 : scan edge (search edge 3) 4 [2287] 2287 root337Tail1 winner = winner :=
  (root337Step1).trans root337Scan2
theorem root337Scan0 : scan edge (search edge 3) 4 [2287] 2287 root337Tail0 winner = winner :=
  (root337Step0).trans root337Scan1
set_option maxRecDepth 4096 in
theorem root337Branch : search edge 4 [2287] root337Candidates 2287 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2287] 2287 root337Tail0 winner = winner
  exact root337Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
