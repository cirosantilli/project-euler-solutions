import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root136Candidates : List Nat := Data.row797
set_option maxRecDepth 8192 in
theorem root136Candidates_checked : candidates edge 5 0 797 tail137 winner.bound = root136Candidates := by
  apply candidates_eq_row edge 5 0 797 tail137 root136Candidates winner.bound tail137_ordered
  · decide
  · decide
  · funext q
    rfl
def root136Tail59 : List Nat := []
def root136Tail58 : List Nat := 24851 :: root136Tail59
def root136Tail57 : List Nat := 24473 :: root136Tail58
def root136Tail56 : List Nat := 24419 :: root136Tail57
def root136Tail55 : List Nat := 24251 :: root136Tail56
def root136Tail54 : List Nat := 23531 :: root136Tail55
def root136Tail53 : List Nat := 22109 :: root136Tail54
def root136Tail52 : List Nat := 21683 :: root136Tail53
def root136Tail51 : List Nat := 21149 :: root136Tail52
def root136Tail50 : List Nat := 20759 :: root136Tail51
def root136Tail49 : List Nat := 19841 :: root136Tail50
def root136Tail48 : List Nat := 19727 :: root136Tail49
def root136Tail47 : List Nat := 19583 :: root136Tail48
def root136Tail46 : List Nat := 18917 :: root136Tail47
def root136Tail45 : List Nat := 18911 :: root136Tail46
def root136Tail44 : List Nat := 18257 :: root136Tail45
def root136Tail43 : List Nat := 17669 :: root136Tail44
def root136Tail42 : List Nat := 17207 :: root136Tail43
def root136Tail41 : List Nat := 13967 :: root136Tail42
def root136Tail40 : List Nat := 13463 :: root136Tail41
def root136Tail39 : List Nat := 12251 :: root136Tail40
def root136Tail38 : List Nat := 12071 :: root136Tail39
def root136Tail37 : List Nat := 11903 :: root136Tail38
def root136Tail36 : List Nat := 11867 :: root136Tail37
def root136Tail35 : List Nat := 11393 :: root136Tail36
def root136Tail34 : List Nat := 11243 :: root136Tail35
def root136Tail33 : List Nat := 11069 :: root136Tail34
def root136Tail32 : List Nat := 10973 :: root136Tail33
def root136Tail31 : List Nat := 10853 :: root136Tail32
def root136Tail30 : List Nat := 10193 :: root136Tail31
def root136Tail29 : List Nat := 10181 :: root136Tail30
def root136Tail28 : List Nat := 10061 :: root136Tail29
def root136Tail27 : List Nat := 9833 :: root136Tail28
def root136Tail26 : List Nat := 9479 :: root136Tail27
def root136Tail25 : List Nat := 9173 :: root136Tail26
def root136Tail24 : List Nat := 8429 :: root136Tail25
def root136Tail23 : List Nat := 7841 :: root136Tail24
def root136Tail22 : List Nat := 7649 :: root136Tail23
def root136Tail21 : List Nat := 7457 :: root136Tail22
def root136Tail20 : List Nat := 6911 :: root136Tail21
def root136Tail19 : List Nat := 6833 :: root136Tail20
def root136Tail18 : List Nat := 6389 :: root136Tail19
def root136Tail17 : List Nat := 5531 :: root136Tail18
def root136Tail16 : List Nat := 5519 :: root136Tail17
def root136Tail15 : List Nat := 5189 :: root136Tail16
def root136Tail14 : List Nat := 5147 :: root136Tail15
def root136Tail13 : List Nat := 4937 :: root136Tail14
def root136Tail12 : List Nat := 4481 :: root136Tail13
def root136Tail11 : List Nat := 4259 :: root136Tail12
def root136Tail10 : List Nat := 3929 :: root136Tail11
def root136Tail9 : List Nat := 3659 :: root136Tail10
def root136Tail8 : List Nat := 3461 :: root136Tail9
def root136Tail7 : List Nat := 3299 :: root136Tail8
def root136Tail6 : List Nat := 3209 :: root136Tail7
def root136Tail5 : List Nat := 2969 :: root136Tail6
def root136Tail4 : List Nat := 2621 :: root136Tail5
def root136Tail3 : List Nat := 2417 :: root136Tail4
def root136Tail2 : List Nat := 2411 :: root136Tail3
def root136Tail1 : List Nat := 2339 :: root136Tail2
def root136Tail0 : List Nat := 2297 :: root136Tail1
def root136Candidates0 : List Nat := [11243, 18257]
set_option maxRecDepth 8192 in
theorem root136Candidates0_checked : candidates edge 4 797 2297 root136Tail1 winner.bound = root136Candidates0 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
theorem root136Step0 : scan edge (search edge 3) 4 [797] 797 root136Tail0 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail1 winner := by
  rw [root136Tail0, scan, if_neg (by decide), root136Candidates0_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail1
    (if 3 ≤ root136Candidates0.length then search edge 3 [797, 2297] root136Candidates0 3094 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates1 : List Nat := [7841, 18917]
set_option maxRecDepth 8192 in
theorem root136Candidates1_checked : candidates edge 4 797 2339 root136Tail2 winner.bound = root136Candidates1 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root136Step1 : scan edge (search edge 3) 4 [797] 797 root136Tail1 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail2 winner := by
  rw [root136Tail1, scan, if_neg (by decide), root136Candidates1_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail2
    (if 3 ≤ root136Candidates1.length then search edge 3 [797, 2339] root136Candidates1 3136 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates2 : List Nat := [5519, 11069]
set_option maxRecDepth 8192 in
theorem root136Candidates2_checked : candidates edge 4 797 2411 root136Tail3 winner.bound = root136Candidates2 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root136Step2 : scan edge (search edge 3) 4 [797] 797 root136Tail2 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail3 winner := by
  rw [root136Tail2, scan, if_neg (by decide), root136Candidates2_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail3
    (if 3 ≤ root136Candidates2.length then search edge 3 [797, 2411] root136Candidates2 3208 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates3 : List Nat := [6389]
set_option maxRecDepth 8192 in
theorem root136Candidates3_checked : candidates edge 4 797 2417 root136Tail4 winner.bound = root136Candidates3 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root136Step3 : scan edge (search edge 3) 4 [797] 797 root136Tail3 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail4 winner := by
  rw [root136Tail3, scan, if_neg (by decide), root136Candidates3_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail4
    (if 3 ≤ root136Candidates3.length then search edge 3 [797, 2417] root136Candidates3 3214 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root136Candidates4_checked : candidates edge 4 797 2621 root136Tail5 winner.bound = root136Candidates4 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root136Step4 : scan edge (search edge 3) 4 [797] 797 root136Tail4 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail5 winner := by
  rw [root136Tail4, scan, if_neg (by decide), root136Candidates4_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail5
    (if 3 ≤ root136Candidates4.length then search edge 3 [797, 2621] root136Candidates4 3418 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates5 : List Nat := [7841]
set_option maxRecDepth 8192 in
theorem root136Candidates5_checked : candidates edge 4 797 2969 root136Tail6 winner.bound = root136Candidates5 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root136Step5 : scan edge (search edge 3) 4 [797] 797 root136Tail5 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail6 winner := by
  rw [root136Tail5, scan, if_neg (by decide), root136Candidates5_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail6
    (if 3 ≤ root136Candidates5.length then search edge 3 [797, 2969] root136Candidates5 3766 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root136Candidates6_checked : candidates edge 4 797 3209 root136Tail7 winner.bound = root136Candidates6 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root136Step6 : scan edge (search edge 3) 4 [797] 797 root136Tail6 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail7 winner := by
  rw [root136Tail6, scan, if_neg (by decide), root136Candidates6_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail7
    (if 3 ≤ root136Candidates6.length then search edge 3 [797, 3209] root136Candidates6 4006 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates7 : List Nat := [8429, 19727]
set_option maxRecDepth 8192 in
theorem root136Candidates7_checked : candidates edge 4 797 3299 root136Tail8 winner.bound = root136Candidates7 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root136Step7 : scan edge (search edge 3) 4 [797] 797 root136Tail7 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail8 winner := by
  rw [root136Tail7, scan, if_neg (by decide), root136Candidates7_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail8
    (if 3 ≤ root136Candidates7.length then search edge 3 [797, 3299] root136Candidates7 4096 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates8 : List Nat := [3659, 11069, 13967, 18917]
set_option maxRecDepth 8192 in
theorem root136Candidates8_checked : candidates edge 4 797 3461 root136Tail9 winner.bound = root136Candidates8 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root136Child8 : search edge 3 [797, 3461] root136Candidates8 4258 winner = winner := by decide
theorem root136Step8 : scan edge (search edge 3) 4 [797] 797 root136Tail8 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail9 winner := by
  rw [root136Tail8, scan, if_neg (by decide), root136Candidates8_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail9
    (if 3 ≤ root136Candidates8.length then search edge 3 [797, 3461] root136Candidates8 4258 winner else winner) = _
  rw [if_pos (by decide), root136Child8]
def root136Candidates9 : List Nat := [11069]
set_option maxRecDepth 8192 in
theorem root136Candidates9_checked : candidates edge 4 797 3659 root136Tail10 winner.bound = root136Candidates9 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
theorem root136Step9 : scan edge (search edge 3) 4 [797] 797 root136Tail9 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail10 winner := by
  rw [root136Tail9, scan, if_neg (by decide), root136Candidates9_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail10
    (if 3 ≤ root136Candidates9.length then search edge 3 [797, 3659] root136Candidates9 4456 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates10 : List Nat := [7457]
set_option maxRecDepth 8192 in
theorem root136Candidates10_checked : candidates edge 4 797 3929 root136Tail11 winner.bound = root136Candidates10 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root136Step10 : scan edge (search edge 3) 4 [797] 797 root136Tail10 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail11 winner := by
  rw [root136Tail10, scan, if_neg (by decide), root136Candidates10_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail11
    (if 3 ≤ root136Candidates10.length then search edge 3 [797, 3929] root136Candidates10 4726 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates11 : List Nat := [10181]
set_option maxRecDepth 8192 in
theorem root136Candidates11_checked : candidates edge 4 797 4259 root136Tail12 winner.bound = root136Candidates11 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root136Step11 : scan edge (search edge 3) 4 [797] 797 root136Tail11 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail12 winner := by
  rw [root136Tail11, scan, if_neg (by decide), root136Candidates11_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail12
    (if 3 ≤ root136Candidates11.length then search edge 3 [797, 4259] root136Candidates11 5056 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates12 : List Nat := [5189, 19583]
set_option maxRecDepth 8192 in
theorem root136Candidates12_checked : candidates edge 4 797 4481 root136Tail13 winner.bound = root136Candidates12 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root136Step12 : scan edge (search edge 3) 4 [797] 797 root136Tail12 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail13 winner := by
  rw [root136Tail12, scan, if_neg (by decide), root136Candidates12_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail13
    (if 3 ≤ root136Candidates12.length then search edge 3 [797, 4481] root136Candidates12 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates13 : List Nat := [7649, 7841, 18257]
set_option maxRecDepth 8192 in
theorem root136Candidates13_checked : candidates edge 4 797 4937 root136Tail14 winner.bound = root136Candidates13 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root136Child13 : search edge 3 [797, 4937] root136Candidates13 5734 winner = winner := by decide
theorem root136Step13 : scan edge (search edge 3) 4 [797] 797 root136Tail13 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail14 winner := by
  rw [root136Tail13, scan, if_neg (by decide), root136Candidates13_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail14
    (if 3 ≤ root136Candidates13.length then search edge 3 [797, 4937] root136Candidates13 5734 winner else winner) = _
  rw [if_pos (by decide), root136Child13]
def root136Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root136Candidates14_checked : candidates edge 4 797 5147 root136Tail15 winner.bound = root136Candidates14 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root136Step14 : scan edge (search edge 3) 4 [797] 797 root136Tail14 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail15 winner := by
  rw [root136Tail14, scan, if_neg (by decide), root136Candidates14_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail15
    (if 3 ≤ root136Candidates14.length then search edge 3 [797, 5147] root136Candidates14 5944 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates15 : List Nat := [19727]
set_option maxRecDepth 8192 in
theorem root136Candidates15_checked : candidates edge 4 797 5189 root136Tail16 winner.bound = root136Candidates15 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root136Step15 : scan edge (search edge 3) 4 [797] 797 root136Tail15 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail16 winner := by
  rw [root136Tail15, scan, if_neg (by decide), root136Candidates15_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail16
    (if 3 ≤ root136Candidates15.length then search edge 3 [797, 5189] root136Candidates15 5986 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root136Candidates16_checked : candidates edge 4 797 5519 root136Tail17 winner.bound = root136Candidates16 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root136Step16 : scan edge (search edge 3) 4 [797] 797 root136Tail16 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail17 winner := by
  rw [root136Tail16, scan, if_neg (by decide), root136Candidates16_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail17
    (if 3 ≤ root136Candidates16.length then search edge 3 [797, 5519] root136Candidates16 6316 winner else winner) = _
  rw [if_neg (by decide)]
def root136Candidates17 : List Nat := [10193]
set_option maxRecDepth 8192 in
theorem root136Candidates17_checked : candidates edge 4 797 5531 root136Tail18 winner.bound = root136Candidates17 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root136Step17 : scan edge (search edge 3) 4 [797] 797 root136Tail17 winner =
    scan edge (search edge 3) 4 [797] 797 root136Tail18 winner := by
  rw [root136Tail17, scan, if_neg (by decide), root136Candidates17_checked]
  change scan edge (search edge 3) 4 [797] 797 root136Tail18
    (if 3 ≤ root136Candidates17.length then search edge 3 [797, 5531] root136Candidates17 6328 winner else winner) = _
  rw [if_neg (by decide)]
theorem root136Scan18 : scan edge (search edge 3) 4 [797] 797 root136Tail18 winner = winner := by rw [root136Tail18, scan, if_pos (by decide)]
theorem root136Scan17 : scan edge (search edge 3) 4 [797] 797 root136Tail17 winner = winner :=
  (root136Step17).trans root136Scan18
theorem root136Scan16 : scan edge (search edge 3) 4 [797] 797 root136Tail16 winner = winner :=
  (root136Step16).trans root136Scan17
theorem root136Scan15 : scan edge (search edge 3) 4 [797] 797 root136Tail15 winner = winner :=
  (root136Step15).trans root136Scan16
theorem root136Scan14 : scan edge (search edge 3) 4 [797] 797 root136Tail14 winner = winner :=
  (root136Step14).trans root136Scan15
theorem root136Scan13 : scan edge (search edge 3) 4 [797] 797 root136Tail13 winner = winner :=
  (root136Step13).trans root136Scan14
theorem root136Scan12 : scan edge (search edge 3) 4 [797] 797 root136Tail12 winner = winner :=
  (root136Step12).trans root136Scan13
theorem root136Scan11 : scan edge (search edge 3) 4 [797] 797 root136Tail11 winner = winner :=
  (root136Step11).trans root136Scan12
theorem root136Scan10 : scan edge (search edge 3) 4 [797] 797 root136Tail10 winner = winner :=
  (root136Step10).trans root136Scan11
theorem root136Scan9 : scan edge (search edge 3) 4 [797] 797 root136Tail9 winner = winner :=
  (root136Step9).trans root136Scan10
theorem root136Scan8 : scan edge (search edge 3) 4 [797] 797 root136Tail8 winner = winner :=
  (root136Step8).trans root136Scan9
theorem root136Scan7 : scan edge (search edge 3) 4 [797] 797 root136Tail7 winner = winner :=
  (root136Step7).trans root136Scan8
theorem root136Scan6 : scan edge (search edge 3) 4 [797] 797 root136Tail6 winner = winner :=
  (root136Step6).trans root136Scan7
theorem root136Scan5 : scan edge (search edge 3) 4 [797] 797 root136Tail5 winner = winner :=
  (root136Step5).trans root136Scan6
theorem root136Scan4 : scan edge (search edge 3) 4 [797] 797 root136Tail4 winner = winner :=
  (root136Step4).trans root136Scan5
theorem root136Scan3 : scan edge (search edge 3) 4 [797] 797 root136Tail3 winner = winner :=
  (root136Step3).trans root136Scan4
theorem root136Scan2 : scan edge (search edge 3) 4 [797] 797 root136Tail2 winner = winner :=
  (root136Step2).trans root136Scan3
theorem root136Scan1 : scan edge (search edge 3) 4 [797] 797 root136Tail1 winner = winner :=
  (root136Step1).trans root136Scan2
theorem root136Scan0 : scan edge (search edge 3) 4 [797] 797 root136Tail0 winner = winner :=
  (root136Step0).trans root136Scan1
set_option maxRecDepth 4096 in
theorem root136Branch : search edge 4 [797] root136Candidates 797 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [797] 797 root136Tail0 winner = winner
  exact root136Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
