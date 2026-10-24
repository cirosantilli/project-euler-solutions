import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root157Candidates : List Nat := Data.row941
set_option maxRecDepth 8192 in
theorem root157Candidates_checked : candidates edge 5 0 941 tail158 winner.bound = root157Candidates := by
  apply candidates_eq_row edge 5 0 941 tail158 root157Candidates winner.bound tail158_ordered
  · decide
  · decide
  · funext q
    rfl
def root157Tail65 : List Nat := []
def root157Tail64 : List Nat := 24977 :: root157Tail65
def root157Tail63 : List Nat := 24197 :: root157Tail64
def root157Tail62 : List Nat := 24179 :: root157Tail63
def root157Tail61 : List Nat := 23633 :: root157Tail62
def root157Tail60 : List Nat := 23417 :: root157Tail61
def root157Tail59 : List Nat := 23333 :: root157Tail60
def root157Tail58 : List Nat := 23279 :: root157Tail59
def root157Tail57 : List Nat := 23039 :: root157Tail58
def root157Tail56 : List Nat := 21803 :: root157Tail57
def root157Tail55 : List Nat := 21323 :: root157Tail56
def root157Tail54 : List Nat := 20693 :: root157Tail55
def root157Tail53 : List Nat := 20333 :: root157Tail54
def root157Tail52 : List Nat := 19919 :: root157Tail53
def root157Tail51 : List Nat := 19793 :: root157Tail52
def root157Tail50 : List Nat := 18593 :: root157Tail51
def root157Tail49 : List Nat := 17417 :: root157Tail50
def root157Tail48 : List Nat := 16787 :: root157Tail49
def root157Tail47 : List Nat := 16193 :: root157Tail48
def root157Tail46 : List Nat := 15971 :: root157Tail47
def root157Tail45 : List Nat := 15551 :: root157Tail46
def root157Tail44 : List Nat := 15233 :: root157Tail45
def root157Tail43 : List Nat := 14897 :: root157Tail44
def root157Tail42 : List Nat := 14879 :: root157Tail43
def root157Tail41 : List Nat := 14621 :: root157Tail42
def root157Tail40 : List Nat := 14411 :: root157Tail41
def root157Tail39 : List Nat := 14243 :: root157Tail40
def root157Tail38 : List Nat := 14177 :: root157Tail39
def root157Tail37 : List Nat := 13781 :: root157Tail38
def root157Tail36 : List Nat := 13043 :: root157Tail37
def root157Tail35 : List Nat := 12959 :: root157Tail36
def root157Tail34 : List Nat := 12503 :: root157Tail35
def root157Tail33 : List Nat := 12413 :: root157Tail34
def root157Tail32 : List Nat := 12401 :: root157Tail33
def root157Tail31 : List Nat := 12269 :: root157Tail32
def root157Tail30 : List Nat := 12107 :: root157Tail31
def root157Tail29 : List Nat := 11981 :: root157Tail30
def root157Tail28 : List Nat := 11261 :: root157Tail29
def root157Tail27 : List Nat := 10889 :: root157Tail28
def root157Tail26 : List Nat := 10631 :: root157Tail27
def root157Tail25 : List Nat := 10607 :: root157Tail26
def root157Tail24 : List Nat := 9923 :: root157Tail25
def root157Tail23 : List Nat := 9239 :: root157Tail24
def root157Tail22 : List Nat := 8693 :: root157Tail23
def root157Tail21 : List Nat := 8681 :: root157Tail22
def root157Tail20 : List Nat := 8369 :: root157Tail21
def root157Tail19 : List Nat := 7229 :: root157Tail20
def root157Tail18 : List Nat := 6971 :: root157Tail19
def root157Tail17 : List Nat := 6299 :: root157Tail18
def root157Tail16 : List Nat := 6263 :: root157Tail17
def root157Tail15 : List Nat := 6053 :: root157Tail16
def root157Tail14 : List Nat := 6047 :: root157Tail15
def root157Tail13 : List Nat := 4817 :: root157Tail14
def root157Tail12 : List Nat := 4649 :: root157Tail13
def root157Tail11 : List Nat := 4409 :: root157Tail12
def root157Tail10 : List Nat := 3527 :: root157Tail11
def root157Tail9 : List Nat := 3389 :: root157Tail10
def root157Tail8 : List Nat := 3221 :: root157Tail9
def root157Tail7 : List Nat := 3137 :: root157Tail8
def root157Tail6 : List Nat := 2099 :: root157Tail7
def root157Tail5 : List Nat := 1709 :: root157Tail6
def root157Tail4 : List Nat := 1607 :: root157Tail5
def root157Tail3 : List Nat := 1511 :: root157Tail4
def root157Tail2 : List Nat := 1229 :: root157Tail3
def root157Tail1 : List Nat := 1187 :: root157Tail2
def root157Tail0 : List Nat := 1049 :: root157Tail1
def root157Candidates0 : List Nat := [6971, 7229]
set_option maxRecDepth 8192 in
theorem root157Candidates0_checked : candidates edge 4 941 1049 root157Tail1 winner.bound = root157Candidates0 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
theorem root157Step0 : scan edge (search edge 3) 4 [941] 941 root157Tail0 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail1 winner := by
  rw [root157Tail0, scan, if_neg (by decide), root157Candidates0_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail1
    (if 3 ≤ root157Candidates0.length then search edge 3 [941, 1049] root157Candidates0 1990 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates1 : List Nat := [4649, 12413, 13781, 14243]
set_option maxRecDepth 8192 in
theorem root157Candidates1_checked : candidates edge 4 941 1187 root157Tail2 winner.bound = root157Candidates1 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root157Child1 : search edge 3 [941, 1187] root157Candidates1 2128 winner = winner := by decide
theorem root157Step1 : scan edge (search edge 3) 4 [941] 941 root157Tail1 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail2 winner := by
  rw [root157Tail1, scan, if_neg (by decide), root157Candidates1_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail2
    (if 3 ≤ root157Candidates1.length then search edge 3 [941, 1187] root157Candidates1 2128 winner else winner) = _
  rw [if_pos (by decide), root157Child1]
def root157Candidates2 : List Nat := [1607, 3389, 19919]
set_option maxRecDepth 8192 in
theorem root157Candidates2_checked : candidates edge 4 941 1229 root157Tail3 winner.bound = root157Candidates2 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root157Child2 : search edge 3 [941, 1229] root157Candidates2 2170 winner = winner := by decide
theorem root157Step2 : scan edge (search edge 3) 4 [941] 941 root157Tail2 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail3 winner := by
  rw [root157Tail2, scan, if_neg (by decide), root157Candidates2_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail3
    (if 3 ≤ root157Candidates2.length then search edge 3 [941, 1229] root157Candidates2 2170 winner else winner) = _
  rw [if_pos (by decide), root157Child2]
def root157Candidates3 : List Nat := [3389]
set_option maxRecDepth 8192 in
theorem root157Candidates3_checked : candidates edge 4 941 1511 root157Tail4 winner.bound = root157Candidates3 := by
  unfold candidates
  have row : edge 1511 = (SortedNatTree.contains FastRows.row1511) := by
    change Data.row1511.contains = (SortedNatTree.contains FastRows.row1511)
    exact FastRows.row1511_checked.symm
  rw [row]
  decide
theorem root157Step3 : scan edge (search edge 3) 4 [941] 941 root157Tail3 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail4 winner := by
  rw [root157Tail3, scan, if_neg (by decide), root157Candidates3_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail4
    (if 3 ≤ root157Candidates3.length then search edge 3 [941, 1511] root157Candidates3 2452 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates4 : List Nat := [19919]
set_option maxRecDepth 8192 in
theorem root157Candidates4_checked : candidates edge 4 941 1607 root157Tail5 winner.bound = root157Candidates4 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
theorem root157Step4 : scan edge (search edge 3) 4 [941] 941 root157Tail4 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail5 winner := by
  rw [root157Tail4, scan, if_neg (by decide), root157Candidates4_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail5
    (if 3 ≤ root157Candidates4.length then search edge 3 [941, 1607] root157Candidates4 2548 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates5 : List Nat := [3137, 3389, 9923, 15971, 21323]
set_option maxRecDepth 8192 in
theorem root157Candidates5_checked : candidates edge 4 941 1709 root157Tail6 winner.bound = root157Candidates5 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root157Child5 : search edge 3 [941, 1709] root157Candidates5 2650 winner = winner := by decide
theorem root157Step5 : scan edge (search edge 3) 4 [941] 941 root157Tail5 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail6 winner := by
  rw [root157Tail5, scan, if_neg (by decide), root157Candidates5_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail6
    (if 3 ≤ root157Candidates5.length then search edge 3 [941, 1709] root157Candidates5 2650 winner else winner) = _
  rw [if_pos (by decide), root157Child5]
def root157Candidates6 : List Nat := [3527, 14879, 19793]
set_option maxRecDepth 8192 in
theorem root157Candidates6_checked : candidates edge 4 941 2099 root157Tail7 winner.bound = root157Candidates6 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root157Child6 : search edge 3 [941, 2099] root157Candidates6 3040 winner = winner := by decide
theorem root157Step6 : scan edge (search edge 3) 4 [941] 941 root157Tail6 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail7 winner := by
  rw [root157Tail6, scan, if_neg (by decide), root157Candidates6_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail7
    (if 3 ≤ root157Candidates6.length then search edge 3 [941, 2099] root157Candidates6 3040 winner else winner) = _
  rw [if_pos (by decide), root157Child6]
def root157Candidates7 : List Nat := [9923, 10889, 12107]
set_option maxRecDepth 8192 in
theorem root157Candidates7_checked : candidates edge 4 941 3137 root157Tail8 winner.bound = root157Candidates7 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root157Child7 : search edge 3 [941, 3137] root157Candidates7 4078 winner = winner := by decide
theorem root157Step7 : scan edge (search edge 3) 4 [941] 941 root157Tail7 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail8 winner := by
  rw [root157Tail7, scan, if_neg (by decide), root157Candidates7_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail8
    (if 3 ≤ root157Candidates7.length then search edge 3 [941, 3137] root157Candidates7 4078 winner else winner) = _
  rw [if_pos (by decide), root157Child7]
def root157Candidates8 : List Nat := [4409, 4817]
set_option maxRecDepth 8192 in
theorem root157Candidates8_checked : candidates edge 4 941 3221 root157Tail9 winner.bound = root157Candidates8 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root157Step8 : scan edge (search edge 3) 4 [941] 941 root157Tail8 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail9 winner := by
  rw [root157Tail8, scan, if_neg (by decide), root157Candidates8_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail9
    (if 3 ≤ root157Candidates8.length then search edge 3 [941, 3221] root157Candidates8 4162 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates9 : List Nat := [4409]
set_option maxRecDepth 8192 in
theorem root157Candidates9_checked : candidates edge 4 941 3389 root157Tail10 winner.bound = root157Candidates9 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
theorem root157Step9 : scan edge (search edge 3) 4 [941] 941 root157Tail9 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail10 winner := by
  rw [root157Tail9, scan, if_neg (by decide), root157Candidates9_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail10
    (if 3 ≤ root157Candidates9.length then search edge 3 [941, 3389] root157Candidates9 4330 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates10 : List Nat := [4649, 9239, 14897]
set_option maxRecDepth 8192 in
theorem root157Candidates10_checked : candidates edge 4 941 3527 root157Tail11 winner.bound = root157Candidates10 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root157Child10 : search edge 3 [941, 3527] root157Candidates10 4468 winner = winner := by decide
theorem root157Step10 : scan edge (search edge 3) 4 [941] 941 root157Tail10 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail11 winner := by
  rw [root157Tail10, scan, if_neg (by decide), root157Candidates10_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail11
    (if 3 ≤ root157Candidates10.length then search edge 3 [941, 3527] root157Candidates10 4468 winner else winner) = _
  rw [if_pos (by decide), root157Child10]
def root157Candidates11 : List Nat := [9239, 19793]
set_option maxRecDepth 8192 in
theorem root157Candidates11_checked : candidates edge 4 941 4409 root157Tail12 winner.bound = root157Candidates11 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root157Step11 : scan edge (search edge 3) 4 [941] 941 root157Tail11 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail12 winner := by
  rw [root157Tail11, scan, if_neg (by decide), root157Candidates11_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail12
    (if 3 ≤ root157Candidates11.length then search edge 3 [941, 4409] root157Candidates11 5350 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root157Candidates12_checked : candidates edge 4 941 4649 root157Tail13 winner.bound = root157Candidates12 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root157Step12 : scan edge (search edge 3) 4 [941] 941 root157Tail12 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail13 winner := by
  rw [root157Tail12, scan, if_neg (by decide), root157Candidates12_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail13
    (if 3 ≤ root157Candidates12.length then search edge 3 [941, 4649] root157Candidates12 5590 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates13 : List Nat := [17417]
set_option maxRecDepth 8192 in
theorem root157Candidates13_checked : candidates edge 4 941 4817 root157Tail14 winner.bound = root157Candidates13 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root157Step13 : scan edge (search edge 3) 4 [941] 941 root157Tail13 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail14 winner := by
  rw [root157Tail13, scan, if_neg (by decide), root157Candidates13_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail14
    (if 3 ≤ root157Candidates13.length then search edge 3 [941, 4817] root157Candidates13 5758 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates14 : List Nat := [12401, 12503]
set_option maxRecDepth 8192 in
theorem root157Candidates14_checked : candidates edge 4 941 6047 root157Tail15 winner.bound = root157Candidates14 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root157Step14 : scan edge (search edge 3) 4 [941] 941 root157Tail14 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail15 winner := by
  rw [root157Tail14, scan, if_neg (by decide), root157Candidates14_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail15
    (if 3 ≤ root157Candidates14.length then search edge 3 [941, 6047] root157Candidates14 6988 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates15 : List Nat := [12503, 14243]
set_option maxRecDepth 8192 in
theorem root157Candidates15_checked : candidates edge 4 941 6053 root157Tail16 winner.bound = root157Candidates15 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root157Step15 : scan edge (search edge 3) 4 [941] 941 root157Tail15 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail16 winner := by
  rw [root157Tail15, scan, if_neg (by decide), root157Candidates15_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail16
    (if 3 ≤ root157Candidates15.length then search edge 3 [941, 6053] root157Candidates15 6994 winner else winner) = _
  rw [if_neg (by decide)]
def root157Candidates16 : List Nat := [11261]
set_option maxRecDepth 8192 in
theorem root157Candidates16_checked : candidates edge 4 941 6263 root157Tail17 winner.bound = root157Candidates16 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root157Step16 : scan edge (search edge 3) 4 [941] 941 root157Tail16 winner =
    scan edge (search edge 3) 4 [941] 941 root157Tail17 winner := by
  rw [root157Tail16, scan, if_neg (by decide), root157Candidates16_checked]
  change scan edge (search edge 3) 4 [941] 941 root157Tail17
    (if 3 ≤ root157Candidates16.length then search edge 3 [941, 6263] root157Candidates16 7204 winner else winner) = _
  rw [if_neg (by decide)]
theorem root157Scan17 : scan edge (search edge 3) 4 [941] 941 root157Tail17 winner = winner := by rw [root157Tail17, scan, if_pos (by decide)]
theorem root157Scan16 : scan edge (search edge 3) 4 [941] 941 root157Tail16 winner = winner :=
  (root157Step16).trans root157Scan17
theorem root157Scan15 : scan edge (search edge 3) 4 [941] 941 root157Tail15 winner = winner :=
  (root157Step15).trans root157Scan16
theorem root157Scan14 : scan edge (search edge 3) 4 [941] 941 root157Tail14 winner = winner :=
  (root157Step14).trans root157Scan15
theorem root157Scan13 : scan edge (search edge 3) 4 [941] 941 root157Tail13 winner = winner :=
  (root157Step13).trans root157Scan14
theorem root157Scan12 : scan edge (search edge 3) 4 [941] 941 root157Tail12 winner = winner :=
  (root157Step12).trans root157Scan13
theorem root157Scan11 : scan edge (search edge 3) 4 [941] 941 root157Tail11 winner = winner :=
  (root157Step11).trans root157Scan12
theorem root157Scan10 : scan edge (search edge 3) 4 [941] 941 root157Tail10 winner = winner :=
  (root157Step10).trans root157Scan11
theorem root157Scan9 : scan edge (search edge 3) 4 [941] 941 root157Tail9 winner = winner :=
  (root157Step9).trans root157Scan10
theorem root157Scan8 : scan edge (search edge 3) 4 [941] 941 root157Tail8 winner = winner :=
  (root157Step8).trans root157Scan9
theorem root157Scan7 : scan edge (search edge 3) 4 [941] 941 root157Tail7 winner = winner :=
  (root157Step7).trans root157Scan8
theorem root157Scan6 : scan edge (search edge 3) 4 [941] 941 root157Tail6 winner = winner :=
  (root157Step6).trans root157Scan7
theorem root157Scan5 : scan edge (search edge 3) 4 [941] 941 root157Tail5 winner = winner :=
  (root157Step5).trans root157Scan6
theorem root157Scan4 : scan edge (search edge 3) 4 [941] 941 root157Tail4 winner = winner :=
  (root157Step4).trans root157Scan5
theorem root157Scan3 : scan edge (search edge 3) 4 [941] 941 root157Tail3 winner = winner :=
  (root157Step3).trans root157Scan4
theorem root157Scan2 : scan edge (search edge 3) 4 [941] 941 root157Tail2 winner = winner :=
  (root157Step2).trans root157Scan3
theorem root157Scan1 : scan edge (search edge 3) 4 [941] 941 root157Tail1 winner = winner :=
  (root157Step1).trans root157Scan2
theorem root157Scan0 : scan edge (search edge 3) 4 [941] 941 root157Tail0 winner = winner :=
  (root157Step0).trans root157Scan1
set_option maxRecDepth 4096 in
theorem root157Branch : search edge 4 [941] root157Candidates 941 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [941] 941 root157Tail0 winner = winner
  exact root157Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
