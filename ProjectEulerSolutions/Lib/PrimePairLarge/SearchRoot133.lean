import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root133Candidates : List Nat := Data.row769
set_option maxRecDepth 8192 in
theorem root133Candidates_checked : candidates edge 5 0 769 tail134 winner.bound = root133Candidates := by
  apply candidates_eq_row edge 5 0 769 tail134 root133Candidates winner.bound tail134_ordered
  · decide
  · decide
  · funext q
    rfl
def root133Tail64 : List Nat := []
def root133Tail63 : List Nat := 25111 :: root133Tail64
def root133Tail62 : List Nat := 24943 :: root133Tail63
def root133Tail61 : List Nat := 24697 :: root133Tail62
def root133Tail60 : List Nat := 23893 :: root133Tail61
def root133Tail59 : List Nat := 23767 :: root133Tail60
def root133Tail58 : List Nat := 23269 :: root133Tail59
def root133Tail57 : List Nat := 23131 :: root133Tail58
def root133Tail56 : List Nat := 22543 :: root133Tail57
def root133Tail55 : List Nat := 22381 :: root133Tail56
def root133Tail54 : List Nat := 21319 :: root133Tail55
def root133Tail53 : List Nat := 21193 :: root133Tail54
def root133Tail52 : List Nat := 20983 :: root133Tail53
def root133Tail51 : List Nat := 20323 :: root133Tail52
def root133Tail50 : List Nat := 20071 :: root133Tail51
def root133Tail49 : List Nat := 19471 :: root133Tail50
def root133Tail48 : List Nat := 19309 :: root133Tail49
def root133Tail47 : List Nat := 18691 :: root133Tail48
def root133Tail46 : List Nat := 18679 :: root133Tail47
def root133Tail45 : List Nat := 18313 :: root133Tail46
def root133Tail44 : List Nat := 18253 :: root133Tail45
def root133Tail43 : List Nat := 18043 :: root133Tail44
def root133Tail42 : List Nat := 17683 :: root133Tail43
def root133Tail41 : List Nat := 16417 :: root133Tail42
def root133Tail40 : List Nat := 16339 :: root133Tail41
def root133Tail39 : List Nat := 15583 :: root133Tail40
def root133Tail38 : List Nat := 15271 :: root133Tail39
def root133Tail37 : List Nat := 14281 :: root133Tail38
def root133Tail36 : List Nat := 14221 :: root133Tail37
def root133Tail35 : List Nat := 13807 :: root133Tail36
def root133Tail34 : List Nat := 13381 :: root133Tail35
def root133Tail33 : List Nat := 12967 :: root133Tail34
def root133Tail32 : List Nat := 12799 :: root133Tail33
def root133Tail31 : List Nat := 12763 :: root133Tail32
def root133Tail30 : List Nat := 12037 :: root133Tail31
def root133Tail29 : List Nat := 11971 :: root133Tail30
def root133Tail28 : List Nat := 11251 :: root133Tail29
def root133Tail27 : List Nat := 11173 :: root133Tail28
def root133Tail26 : List Nat := 10369 :: root133Tail27
def root133Tail25 : List Nat := 10243 :: root133Tail26
def root133Tail24 : List Nat := 9931 :: root133Tail25
def root133Tail23 : List Nat := 9649 :: root133Tail24
def root133Tail22 : List Nat := 9403 :: root133Tail23
def root133Tail21 : List Nat := 9157 :: root133Tail22
def root133Tail20 : List Nat := 7507 :: root133Tail21
def root133Tail19 : List Nat := 7057 :: root133Tail20
def root133Tail18 : List Nat := 6967 :: root133Tail19
def root133Tail17 : List Nat := 6763 :: root133Tail18
def root133Tail16 : List Nat := 6343 :: root133Tail17
def root133Tail15 : List Nat := 6211 :: root133Tail16
def root133Tail14 : List Nat := 6133 :: root133Tail15
def root133Tail13 : List Nat := 5791 :: root133Tail14
def root133Tail12 : List Nat := 5503 :: root133Tail13
def root133Tail11 : List Nat := 5227 :: root133Tail12
def root133Tail10 : List Nat := 5197 :: root133Tail11
def root133Tail9 : List Nat := 5113 :: root133Tail10
def root133Tail8 : List Nat := 4987 :: root133Tail9
def root133Tail7 : List Nat := 4651 :: root133Tail8
def root133Tail6 : List Nat := 4021 :: root133Tail7
def root133Tail5 : List Nat := 3313 :: root133Tail6
def root133Tail4 : List Nat := 2647 :: root133Tail5
def root133Tail3 : List Nat := 2053 :: root133Tail4
def root133Tail2 : List Nat := 1381 :: root133Tail3
def root133Tail1 : List Nat := 997 :: root133Tail2
def root133Tail0 : List Nat := 919 :: root133Tail1
def root133Candidates0 : List Nat := [9649]
set_option maxRecDepth 8192 in
theorem root133Candidates0_checked : candidates edge 4 769 919 root133Tail1 winner.bound = root133Candidates0 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
theorem root133Step0 : scan edge (search edge 3) 4 [769] 769 root133Tail0 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail1 winner := by
  rw [root133Tail0, scan, if_neg (by decide), root133Candidates0_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail1
    (if 3 ≤ root133Candidates0.length then search edge 3 [769, 919] root133Candidates0 1688 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates1 : List Nat := [6763, 7057, 14221, 14281, 19471, 21319]
set_option maxRecDepth 8192 in
theorem root133Candidates1_checked : candidates edge 4 769 997 root133Tail2 winner.bound = root133Candidates1 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root133Child1 : search edge 3 [769, 997] root133Candidates1 1766 winner = winner := by decide
theorem root133Step1 : scan edge (search edge 3) 4 [769] 769 root133Tail1 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail2 winner := by
  rw [root133Tail1, scan, if_neg (by decide), root133Candidates1_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail2
    (if 3 ≤ root133Candidates1.length then search edge 3 [769, 997] root133Candidates1 1766 winner else winner) = _
  rw [if_pos (by decide), root133Child1]
def root133Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root133Candidates2_checked : candidates edge 4 769 1381 root133Tail3 winner.bound = root133Candidates2 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root133Step2 : scan edge (search edge 3) 4 [769] 769 root133Tail2 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail3 winner := by
  rw [root133Tail2, scan, if_neg (by decide), root133Candidates2_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail3
    (if 3 ≤ root133Candidates2.length then search edge 3 [769, 1381] root133Candidates2 2150 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates3 : List Nat := [6343]
set_option maxRecDepth 8192 in
theorem root133Candidates3_checked : candidates edge 4 769 2053 root133Tail4 winner.bound = root133Candidates3 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root133Step3 : scan edge (search edge 3) 4 [769] 769 root133Tail3 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail4 winner := by
  rw [root133Tail3, scan, if_neg (by decide), root133Candidates3_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail4
    (if 3 ≤ root133Candidates3.length then search edge 3 [769, 2053] root133Candidates3 2822 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates4 : List Nat := [13807]
set_option maxRecDepth 8192 in
theorem root133Candidates4_checked : candidates edge 4 769 2647 root133Tail5 winner.bound = root133Candidates4 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root133Step4 : scan edge (search edge 3) 4 [769] 769 root133Tail4 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail5 winner := by
  rw [root133Tail4, scan, if_neg (by decide), root133Candidates4_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail5
    (if 3 ≤ root133Candidates4.length then search edge 3 [769, 2647] root133Candidates4 3416 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates5 : List Nat := [17683]
set_option maxRecDepth 8192 in
theorem root133Candidates5_checked : candidates edge 4 769 3313 root133Tail6 winner.bound = root133Candidates5 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root133Step5 : scan edge (search edge 3) 4 [769] 769 root133Tail5 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail6 winner := by
  rw [root133Tail5, scan, if_neg (by decide), root133Candidates5_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail6
    (if 3 ≤ root133Candidates5.length then search edge 3 [769, 3313] root133Candidates5 4082 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates6 : List Nat := [12037]
set_option maxRecDepth 8192 in
theorem root133Candidates6_checked : candidates edge 4 769 4021 root133Tail7 winner.bound = root133Candidates6 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root133Step6 : scan edge (search edge 3) 4 [769] 769 root133Tail6 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail7 winner := by
  rw [root133Tail6, scan, if_neg (by decide), root133Candidates6_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail7
    (if 3 ≤ root133Candidates6.length then search edge 3 [769, 4021] root133Candidates6 4790 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates7 : List Nat := [12763]
set_option maxRecDepth 8192 in
theorem root133Candidates7_checked : candidates edge 4 769 4651 root133Tail8 winner.bound = root133Candidates7 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root133Step7 : scan edge (search edge 3) 4 [769] 769 root133Tail7 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail8 winner := by
  rw [root133Tail7, scan, if_neg (by decide), root133Candidates7_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail8
    (if 3 ≤ root133Candidates7.length then search edge 3 [769, 4651] root133Candidates7 5420 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates8 : List Nat := [5197, 12763, 18043]
set_option maxRecDepth 8192 in
theorem root133Candidates8_checked : candidates edge 4 769 4987 root133Tail9 winner.bound = root133Candidates8 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root133Child8 : search edge 3 [769, 4987] root133Candidates8 5756 winner = winner := by decide
theorem root133Step8 : scan edge (search edge 3) 4 [769] 769 root133Tail8 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail9 winner := by
  rw [root133Tail8, scan, if_neg (by decide), root133Candidates8_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail9
    (if 3 ≤ root133Candidates8.length then search edge 3 [769, 4987] root133Candidates8 5756 winner else winner) = _
  rw [if_pos (by decide), root133Child8]
def root133Candidates9 : List Nat := [18679]
set_option maxRecDepth 8192 in
theorem root133Candidates9_checked : candidates edge 4 769 5113 root133Tail10 winner.bound = root133Candidates9 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root133Step9 : scan edge (search edge 3) 4 [769] 769 root133Tail9 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail10 winner := by
  rw [root133Tail9, scan, if_neg (by decide), root133Candidates9_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail10
    (if 3 ≤ root133Candidates9.length then search edge 3 [769, 5113] root133Candidates9 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root133Candidates10_checked : candidates edge 4 769 5197 root133Tail11 winner.bound = root133Candidates10 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root133Step10 : scan edge (search edge 3) 4 [769] 769 root133Tail10 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail11 winner := by
  rw [root133Tail10, scan, if_neg (by decide), root133Candidates10_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail11
    (if 3 ≤ root133Candidates10.length then search edge 3 [769, 5197] root133Candidates10 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root133Candidates11_checked : candidates edge 4 769 5227 root133Tail12 winner.bound = root133Candidates11 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root133Step11 : scan edge (search edge 3) 4 [769] 769 root133Tail11 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail12 winner := by
  rw [root133Tail11, scan, if_neg (by decide), root133Candidates11_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail12
    (if 3 ≤ root133Candidates11.length then search edge 3 [769, 5227] root133Candidates11 5996 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root133Candidates12_checked : candidates edge 4 769 5503 root133Tail13 winner.bound = root133Candidates12 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root133Step12 : scan edge (search edge 3) 4 [769] 769 root133Tail12 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail13 winner := by
  rw [root133Tail12, scan, if_neg (by decide), root133Candidates12_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail13
    (if 3 ≤ root133Candidates12.length then search edge 3 [769, 5503] root133Candidates12 6272 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates13 : List Nat := [10243]
set_option maxRecDepth 8192 in
theorem root133Candidates13_checked : candidates edge 4 769 5791 root133Tail14 winner.bound = root133Candidates13 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root133Step13 : scan edge (search edge 3) 4 [769] 769 root133Tail13 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail14 winner := by
  rw [root133Tail13, scan, if_neg (by decide), root133Candidates13_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail14
    (if 3 ≤ root133Candidates13.length then search edge 3 [769, 5791] root133Candidates13 6560 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root133Candidates14_checked : candidates edge 4 769 6133 root133Tail15 winner.bound = root133Candidates14 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root133Step14 : scan edge (search edge 3) 4 [769] 769 root133Tail14 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail15 winner := by
  rw [root133Tail14, scan, if_neg (by decide), root133Candidates14_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail15
    (if 3 ≤ root133Candidates14.length then search edge 3 [769, 6133] root133Candidates14 6902 winner else winner) = _
  rw [if_neg (by decide)]
def root133Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root133Candidates15_checked : candidates edge 4 769 6211 root133Tail16 winner.bound = root133Candidates15 := by
  unfold candidates
  have row : edge 6211 = (SortedNatTree.contains FastRows.row6211) := by
    change Data.row6211.contains = (SortedNatTree.contains FastRows.row6211)
    exact FastRows.row6211_checked.symm
  rw [row]
  decide
theorem root133Step15 : scan edge (search edge 3) 4 [769] 769 root133Tail15 winner =
    scan edge (search edge 3) 4 [769] 769 root133Tail16 winner := by
  rw [root133Tail15, scan, if_neg (by decide), root133Candidates15_checked]
  change scan edge (search edge 3) 4 [769] 769 root133Tail16
    (if 3 ≤ root133Candidates15.length then search edge 3 [769, 6211] root133Candidates15 6980 winner else winner) = _
  rw [if_neg (by decide)]
theorem root133Scan16 : scan edge (search edge 3) 4 [769] 769 root133Tail16 winner = winner := by rw [root133Tail16, scan, if_pos (by decide)]
theorem root133Scan15 : scan edge (search edge 3) 4 [769] 769 root133Tail15 winner = winner :=
  (root133Step15).trans root133Scan16
theorem root133Scan14 : scan edge (search edge 3) 4 [769] 769 root133Tail14 winner = winner :=
  (root133Step14).trans root133Scan15
theorem root133Scan13 : scan edge (search edge 3) 4 [769] 769 root133Tail13 winner = winner :=
  (root133Step13).trans root133Scan14
theorem root133Scan12 : scan edge (search edge 3) 4 [769] 769 root133Tail12 winner = winner :=
  (root133Step12).trans root133Scan13
theorem root133Scan11 : scan edge (search edge 3) 4 [769] 769 root133Tail11 winner = winner :=
  (root133Step11).trans root133Scan12
theorem root133Scan10 : scan edge (search edge 3) 4 [769] 769 root133Tail10 winner = winner :=
  (root133Step10).trans root133Scan11
theorem root133Scan9 : scan edge (search edge 3) 4 [769] 769 root133Tail9 winner = winner :=
  (root133Step9).trans root133Scan10
theorem root133Scan8 : scan edge (search edge 3) 4 [769] 769 root133Tail8 winner = winner :=
  (root133Step8).trans root133Scan9
theorem root133Scan7 : scan edge (search edge 3) 4 [769] 769 root133Tail7 winner = winner :=
  (root133Step7).trans root133Scan8
theorem root133Scan6 : scan edge (search edge 3) 4 [769] 769 root133Tail6 winner = winner :=
  (root133Step6).trans root133Scan7
theorem root133Scan5 : scan edge (search edge 3) 4 [769] 769 root133Tail5 winner = winner :=
  (root133Step5).trans root133Scan6
theorem root133Scan4 : scan edge (search edge 3) 4 [769] 769 root133Tail4 winner = winner :=
  (root133Step4).trans root133Scan5
theorem root133Scan3 : scan edge (search edge 3) 4 [769] 769 root133Tail3 winner = winner :=
  (root133Step3).trans root133Scan4
theorem root133Scan2 : scan edge (search edge 3) 4 [769] 769 root133Tail2 winner = winner :=
  (root133Step2).trans root133Scan3
theorem root133Scan1 : scan edge (search edge 3) 4 [769] 769 root133Tail1 winner = winner :=
  (root133Step1).trans root133Scan2
theorem root133Scan0 : scan edge (search edge 3) 4 [769] 769 root133Tail0 winner = winner :=
  (root133Step0).trans root133Scan1
set_option maxRecDepth 4096 in
theorem root133Branch : search edge 4 [769] root133Candidates 769 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [769] 769 root133Tail0 winner = winner
  exact root133Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
