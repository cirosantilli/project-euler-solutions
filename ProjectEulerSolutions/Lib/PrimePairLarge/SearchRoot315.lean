import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root315Candidates : List Nat := Data.row2111
set_option maxRecDepth 8192 in
theorem root315Candidates_checked : candidates edge 5 0 2111 tail316 winner.bound = root315Candidates := by
  apply candidates_eq_row edge 5 0 2111 tail316 root315Candidates winner.bound tail316_ordered
  · decide
  · decide
  · funext q
    rfl
def root315Tail48 : List Nat := []
def root315Tail47 : List Nat := 23909 :: root315Tail48
def root315Tail46 : List Nat := 23879 :: root315Tail47
def root315Tail45 : List Nat := 23789 :: root315Tail46
def root315Tail44 : List Nat := 23291 :: root315Tail45
def root315Tail43 : List Nat := 22283 :: root315Tail44
def root315Tail42 : List Nat := 22277 :: root315Tail43
def root315Tail41 : List Nat := 21557 :: root315Tail42
def root315Tail40 : List Nat := 21107 :: root315Tail41
def root315Tail39 : List Nat := 20771 :: root315Tail40
def root315Tail38 : List Nat := 20759 :: root315Tail39
def root315Tail37 : List Nat := 19289 :: root315Tail38
def root315Tail36 : List Nat := 17957 :: root315Tail37
def root315Tail35 : List Nat := 17321 :: root315Tail36
def root315Tail34 : List Nat := 15881 :: root315Tail35
def root315Tail33 : List Nat := 15803 :: root315Tail34
def root315Tail32 : List Nat := 14843 :: root315Tail33
def root315Tail31 : List Nat := 14747 :: root315Tail32
def root315Tail30 : List Nat := 13577 :: root315Tail31
def root315Tail29 : List Nat := 12437 :: root315Tail30
def root315Tail28 : List Nat := 11939 :: root315Tail29
def root315Tail27 : List Nat := 11399 :: root315Tail28
def root315Tail26 : List Nat := 10733 :: root315Tail27
def root315Tail25 : List Nat := 10313 :: root315Tail26
def root315Tail24 : List Nat := 10061 :: root315Tail25
def root315Tail23 : List Nat := 9257 :: root315Tail24
def root315Tail22 : List Nat := 8849 :: root315Tail23
def root315Tail21 : List Nat := 8753 :: root315Tail22
def root315Tail20 : List Nat := 8219 :: root315Tail21
def root315Tail19 : List Nat := 7793 :: root315Tail20
def root315Tail18 : List Nat := 7487 :: root315Tail19
def root315Tail17 : List Nat := 6911 :: root315Tail18
def root315Tail16 : List Nat := 6857 :: root315Tail17
def root315Tail15 : List Nat := 5261 :: root315Tail16
def root315Tail14 : List Nat := 4931 :: root315Tail15
def root315Tail13 : List Nat := 4649 :: root315Tail14
def root315Tail12 : List Nat := 4229 :: root315Tail13
def root315Tail11 : List Nat := 4217 :: root315Tail12
def root315Tail10 : List Nat := 4127 :: root315Tail11
def root315Tail9 : List Nat := 4019 :: root315Tail10
def root315Tail8 : List Nat := 3467 :: root315Tail9
def root315Tail7 : List Nat := 3329 :: root315Tail8
def root315Tail6 : List Nat := 3251 :: root315Tail7
def root315Tail5 : List Nat := 3221 :: root315Tail6
def root315Tail4 : List Nat := 3083 :: root315Tail5
def root315Tail3 : List Nat := 3041 :: root315Tail4
def root315Tail2 : List Nat := 2999 :: root315Tail3
def root315Tail1 : List Nat := 2969 :: root315Tail2
def root315Tail0 : List Nat := 2411 :: root315Tail1
def root315Candidates0 : List Nat := [9257]
set_option maxRecDepth 8192 in
theorem root315Candidates0_checked : candidates edge 4 2111 2411 root315Tail1 winner.bound = root315Candidates0 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root315Step0 : scan edge (search edge 3) 4 [2111] 2111 root315Tail0 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail1 winner := by
  rw [root315Tail0, scan, if_neg (by decide), root315Candidates0_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail1
    (if 3 ≤ root315Candidates0.length then search edge 3 [2111, 2411] root315Candidates0 4522 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates1_checked : candidates edge 4 2111 2969 root315Tail2 winner.bound = root315Candidates1 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root315Step1 : scan edge (search edge 3) 4 [2111] 2111 root315Tail1 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail2 winner := by
  rw [root315Tail1, scan, if_neg (by decide), root315Candidates1_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail2
    (if 3 ≤ root315Candidates1.length then search edge 3 [2111, 2969] root315Candidates1 5080 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates2_checked : candidates edge 4 2111 2999 root315Tail3 winner.bound = root315Candidates2 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root315Step2 : scan edge (search edge 3) 4 [2111] 2111 root315Tail2 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail3 winner := by
  rw [root315Tail2, scan, if_neg (by decide), root315Candidates2_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail3
    (if 3 ≤ root315Candidates2.length then search edge 3 [2111, 2999] root315Candidates2 5110 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates3 : List Nat := [14843, 20759]
set_option maxRecDepth 8192 in
theorem root315Candidates3_checked : candidates edge 4 2111 3041 root315Tail4 winner.bound = root315Candidates3 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root315Step3 : scan edge (search edge 3) 4 [2111] 2111 root315Tail3 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail4 winner := by
  rw [root315Tail3, scan, if_neg (by decide), root315Candidates3_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail4
    (if 3 ≤ root315Candidates3.length then search edge 3 [2111, 3041] root315Candidates3 5152 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates4 : List Nat := [3329, 4931, 20759]
set_option maxRecDepth 8192 in
theorem root315Candidates4_checked : candidates edge 4 2111 3083 root315Tail5 winner.bound = root315Candidates4 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root315Child4 : search edge 3 [2111, 3083] root315Candidates4 5194 winner = winner := by decide
theorem root315Step4 : scan edge (search edge 3) 4 [2111] 2111 root315Tail4 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail5 winner := by
  rw [root315Tail4, scan, if_neg (by decide), root315Candidates4_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail5
    (if 3 ≤ root315Candidates4.length then search edge 3 [2111, 3083] root315Candidates4 5194 winner else winner) = _
  rw [if_pos (by decide), root315Child4]
def root315Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates5_checked : candidates edge 4 2111 3221 root315Tail6 winner.bound = root315Candidates5 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root315Step5 : scan edge (search edge 3) 4 [2111] 2111 root315Tail5 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail6 winner := by
  rw [root315Tail5, scan, if_neg (by decide), root315Candidates5_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail6
    (if 3 ≤ root315Candidates5.length then search edge 3 [2111, 3221] root315Candidates5 5332 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates6 : List Nat := [17321]
set_option maxRecDepth 8192 in
theorem root315Candidates6_checked : candidates edge 4 2111 3251 root315Tail7 winner.bound = root315Candidates6 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root315Step6 : scan edge (search edge 3) 4 [2111] 2111 root315Tail6 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail7 winner := by
  rw [root315Tail6, scan, if_neg (by decide), root315Candidates6_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail7
    (if 3 ≤ root315Candidates6.length then search edge 3 [2111, 3251] root315Candidates6 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates7 : List Nat := [4127, 4649]
set_option maxRecDepth 8192 in
theorem root315Candidates7_checked : candidates edge 4 2111 3329 root315Tail8 winner.bound = root315Candidates7 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root315Step7 : scan edge (search edge 3) 4 [2111] 2111 root315Tail7 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail8 winner := by
  rw [root315Tail7, scan, if_neg (by decide), root315Candidates7_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail8
    (if 3 ≤ root315Candidates7.length then search edge 3 [2111, 3329] root315Candidates7 5440 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates8_checked : candidates edge 4 2111 3467 root315Tail9 winner.bound = root315Candidates8 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root315Step8 : scan edge (search edge 3) 4 [2111] 2111 root315Tail8 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail9 winner := by
  rw [root315Tail8, scan, if_neg (by decide), root315Candidates8_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail9
    (if 3 ≤ root315Candidates8.length then search edge 3 [2111, 3467] root315Candidates8 5578 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates9_checked : candidates edge 4 2111 4019 root315Tail10 winner.bound = root315Candidates9 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root315Step9 : scan edge (search edge 3) 4 [2111] 2111 root315Tail9 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail10 winner := by
  rw [root315Tail9, scan, if_neg (by decide), root315Candidates9_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail10
    (if 3 ≤ root315Candidates9.length then search edge 3 [2111, 4019] root315Candidates9 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates10_checked : candidates edge 4 2111 4127 root315Tail11 winner.bound = root315Candidates10 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root315Step10 : scan edge (search edge 3) 4 [2111] 2111 root315Tail10 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail11 winner := by
  rw [root315Tail10, scan, if_neg (by decide), root315Candidates10_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail11
    (if 3 ≤ root315Candidates10.length then search edge 3 [2111, 4127] root315Candidates10 6238 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates11_checked : candidates edge 4 2111 4217 root315Tail12 winner.bound = root315Candidates11 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root315Step11 : scan edge (search edge 3) 4 [2111] 2111 root315Tail11 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail12 winner := by
  rw [root315Tail11, scan, if_neg (by decide), root315Candidates11_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail12
    (if 3 ≤ root315Candidates11.length then search edge 3 [2111, 4217] root315Candidates11 6328 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates12 : List Nat := [11399]
set_option maxRecDepth 8192 in
theorem root315Candidates12_checked : candidates edge 4 2111 4229 root315Tail13 winner.bound = root315Candidates12 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root315Step12 : scan edge (search edge 3) 4 [2111] 2111 root315Tail12 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail13 winner := by
  rw [root315Tail12, scan, if_neg (by decide), root315Candidates12_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail13
    (if 3 ≤ root315Candidates12.length then search edge 3 [2111, 4229] root315Candidates12 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates13_checked : candidates edge 4 2111 4649 root315Tail14 winner.bound = root315Candidates13 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root315Step13 : scan edge (search edge 3) 4 [2111] 2111 root315Tail13 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail14 winner := by
  rw [root315Tail13, scan, if_neg (by decide), root315Candidates13_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail14
    (if 3 ≤ root315Candidates13.length then search edge 3 [2111, 4649] root315Candidates13 6760 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates14_checked : candidates edge 4 2111 4931 root315Tail15 winner.bound = root315Candidates14 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root315Step14 : scan edge (search edge 3) 4 [2111] 2111 root315Tail14 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail15 winner := by
  rw [root315Tail14, scan, if_neg (by decide), root315Candidates14_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail15
    (if 3 ≤ root315Candidates14.length then search edge 3 [2111, 4931] root315Candidates14 7042 winner else winner) = _
  rw [if_neg (by decide)]
def root315Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root315Candidates15_checked : candidates edge 4 2111 5261 root315Tail16 winner.bound = root315Candidates15 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root315Step15 : scan edge (search edge 3) 4 [2111] 2111 root315Tail15 winner =
    scan edge (search edge 3) 4 [2111] 2111 root315Tail16 winner := by
  rw [root315Tail15, scan, if_neg (by decide), root315Candidates15_checked]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail16
    (if 3 ≤ root315Candidates15.length then search edge 3 [2111, 5261] root315Candidates15 7372 winner else winner) = _
  rw [if_neg (by decide)]
theorem root315Scan16 : scan edge (search edge 3) 4 [2111] 2111 root315Tail16 winner = winner := by rw [root315Tail16, scan, if_pos (by decide)]
theorem root315Scan15 : scan edge (search edge 3) 4 [2111] 2111 root315Tail15 winner = winner :=
  (root315Step15).trans root315Scan16
theorem root315Scan14 : scan edge (search edge 3) 4 [2111] 2111 root315Tail14 winner = winner :=
  (root315Step14).trans root315Scan15
theorem root315Scan13 : scan edge (search edge 3) 4 [2111] 2111 root315Tail13 winner = winner :=
  (root315Step13).trans root315Scan14
theorem root315Scan12 : scan edge (search edge 3) 4 [2111] 2111 root315Tail12 winner = winner :=
  (root315Step12).trans root315Scan13
theorem root315Scan11 : scan edge (search edge 3) 4 [2111] 2111 root315Tail11 winner = winner :=
  (root315Step11).trans root315Scan12
theorem root315Scan10 : scan edge (search edge 3) 4 [2111] 2111 root315Tail10 winner = winner :=
  (root315Step10).trans root315Scan11
theorem root315Scan9 : scan edge (search edge 3) 4 [2111] 2111 root315Tail9 winner = winner :=
  (root315Step9).trans root315Scan10
theorem root315Scan8 : scan edge (search edge 3) 4 [2111] 2111 root315Tail8 winner = winner :=
  (root315Step8).trans root315Scan9
theorem root315Scan7 : scan edge (search edge 3) 4 [2111] 2111 root315Tail7 winner = winner :=
  (root315Step7).trans root315Scan8
theorem root315Scan6 : scan edge (search edge 3) 4 [2111] 2111 root315Tail6 winner = winner :=
  (root315Step6).trans root315Scan7
theorem root315Scan5 : scan edge (search edge 3) 4 [2111] 2111 root315Tail5 winner = winner :=
  (root315Step5).trans root315Scan6
theorem root315Scan4 : scan edge (search edge 3) 4 [2111] 2111 root315Tail4 winner = winner :=
  (root315Step4).trans root315Scan5
theorem root315Scan3 : scan edge (search edge 3) 4 [2111] 2111 root315Tail3 winner = winner :=
  (root315Step3).trans root315Scan4
theorem root315Scan2 : scan edge (search edge 3) 4 [2111] 2111 root315Tail2 winner = winner :=
  (root315Step2).trans root315Scan3
theorem root315Scan1 : scan edge (search edge 3) 4 [2111] 2111 root315Tail1 winner = winner :=
  (root315Step1).trans root315Scan2
theorem root315Scan0 : scan edge (search edge 3) 4 [2111] 2111 root315Tail0 winner = winner :=
  (root315Step0).trans root315Scan1
set_option maxRecDepth 4096 in
theorem root315Branch : search edge 4 [2111] root315Candidates 2111 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2111] 2111 root315Tail0 winner = winner
  exact root315Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
