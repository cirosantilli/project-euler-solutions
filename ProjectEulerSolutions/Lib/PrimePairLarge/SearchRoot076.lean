import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root76Candidates : List Nat := Data.row401
set_option maxRecDepth 8192 in
theorem root76Candidates_checked : candidates edge 5 0 401 tail77 winner.bound = root76Candidates := by
  apply candidates_eq_row edge 5 0 401 tail77 root76Candidates winner.bound tail77_ordered
  · decide
  · decide
  · funext q
    rfl
def root76Tail59 : List Nat := []
def root76Tail58 : List Nat := 25469 :: root76Tail59
def root76Tail57 : List Nat := 25229 :: root76Tail58
def root76Tail56 : List Nat := 24809 :: root76Tail57
def root76Tail55 : List Nat := 24419 :: root76Tail56
def root76Tail54 : List Nat := 23813 :: root76Tail55
def root76Tail53 : List Nat := 23399 :: root76Tail54
def root76Tail52 : List Nat := 23189 :: root76Tail53
def root76Tail51 : List Nat := 22877 :: root76Tail52
def root76Tail50 : List Nat := 22259 :: root76Tail51
def root76Tail49 : List Nat := 21839 :: root76Tail50
def root76Tail48 : List Nat := 20963 :: root76Tail49
def root76Tail47 : List Nat := 20063 :: root76Tail48
def root76Tail46 : List Nat := 19379 :: root76Tail47
def root76Tail45 : List Nat := 18257 :: root76Tail46
def root76Tail44 : List Nat := 17783 :: root76Tail45
def root76Tail43 : List Nat := 17333 :: root76Tail44
def root76Tail42 : List Nat := 16691 :: root76Tail43
def root76Tail41 : List Nat := 16253 :: root76Tail42
def root76Tail40 : List Nat := 15083 :: root76Tail41
def root76Tail39 : List Nat := 14939 :: root76Tail40
def root76Tail38 : List Nat := 14537 :: root76Tail39
def root76Tail37 : List Nat := 13499 :: root76Tail38
def root76Tail36 : List Nat := 12983 :: root76Tail37
def root76Tail35 : List Nat := 12893 :: root76Tail36
def root76Tail34 : List Nat := 12659 :: root76Tail35
def root76Tail33 : List Nat := 12263 :: root76Tail34
def root76Tail32 : List Nat := 11969 :: root76Tail33
def root76Tail31 : List Nat := 11927 :: root76Tail32
def root76Tail30 : List Nat := 11801 :: root76Tail31
def root76Tail29 : List Nat := 11171 :: root76Tail30
def root76Tail28 : List Nat := 10709 :: root76Tail29
def root76Tail27 : List Nat := 10211 :: root76Tail28
def root76Tail26 : List Nat := 9689 :: root76Tail27
def root76Tail25 : List Nat := 9623 :: root76Tail26
def root76Tail24 : List Nat := 9437 :: root76Tail25
def root76Tail23 : List Nat := 8837 :: root76Tail24
def root76Tail22 : List Nat := 8753 :: root76Tail23
def root76Tail21 : List Nat := 7691 :: root76Tail22
def root76Tail20 : List Nat := 7589 :: root76Tail21
def root76Tail19 : List Nat := 6833 :: root76Tail20
def root76Tail18 : List Nat := 6317 :: root76Tail19
def root76Tail17 : List Nat := 5801 :: root76Tail18
def root76Tail16 : List Nat := 5051 :: root76Tail17
def root76Tail15 : List Nat := 4271 :: root76Tail16
def root76Tail14 : List Nat := 4217 :: root76Tail15
def root76Tail13 : List Nat := 4211 :: root76Tail14
def root76Tail12 : List Nat := 4133 :: root76Tail13
def root76Tail11 : List Nat := 3671 :: root76Tail12
def root76Tail10 : List Nat := 3623 :: root76Tail11
def root76Tail9 : List Nat := 3407 :: root76Tail10
def root76Tail8 : List Nat := 2999 :: root76Tail9
def root76Tail7 : List Nat := 2441 :: root76Tail8
def root76Tail6 : List Nat := 2213 :: root76Tail7
def root76Tail5 : List Nat := 1487 :: root76Tail6
def root76Tail4 : List Nat := 1289 :: root76Tail5
def root76Tail3 : List Nat := 887 :: root76Tail4
def root76Tail2 : List Nat := 809 :: root76Tail3
def root76Tail1 : List Nat := 743 :: root76Tail2
def root76Tail0 : List Nat := 593 :: root76Tail1
def root76Candidates0 : List Nat := [4133, 4211, 10709, 12263]
set_option maxRecDepth 8192 in
theorem root76Candidates0_checked : candidates edge 4 401 593 root76Tail1 winner.bound = root76Candidates0 := by
  unfold candidates
  have row : edge 593 = (SortedNatTree.contains FastRows.row593) := by
    change Data.row593.contains = (SortedNatTree.contains FastRows.row593)
    exact FastRows.row593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root76Child0 : search edge 3 [401, 593] root76Candidates0 994 winner = winner := by decide
theorem root76Step0 : scan edge (search edge 3) 4 [401] 401 root76Tail0 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail1 winner := by
  rw [root76Tail0, scan, if_neg (by decide), root76Candidates0_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail1
    (if 3 ≤ root76Candidates0.length then search edge 3 [401, 593] root76Candidates0 994 winner else winner) = _
  rw [if_pos (by decide), root76Child0]
def root76Candidates1 : List Nat := [9623, 11927]
set_option maxRecDepth 8192 in
theorem root76Candidates1_checked : candidates edge 4 401 743 root76Tail2 winner.bound = root76Candidates1 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
theorem root76Step1 : scan edge (search edge 3) 4 [401] 401 root76Tail1 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail2 winner := by
  rw [root76Tail1, scan, if_neg (by decide), root76Candidates1_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail2
    (if 3 ≤ root76Candidates1.length then search edge 3 [401, 743] root76Candidates1 1144 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates2 : List Nat := [4271, 10211, 12893, 19379]
set_option maxRecDepth 8192 in
theorem root76Candidates2_checked : candidates edge 4 401 809 root76Tail3 winner.bound = root76Candidates2 := by
  unfold candidates
  have row : edge 809 = (SortedNatTree.contains FastRows.row809) := by
    change Data.row809.contains = (SortedNatTree.contains FastRows.row809)
    exact FastRows.row809_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root76Child2 : search edge 3 [401, 809] root76Candidates2 1210 winner = winner := by decide
theorem root76Step2 : scan edge (search edge 3) 4 [401] 401 root76Tail2 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail3 winner := by
  rw [root76Tail2, scan, if_neg (by decide), root76Candidates2_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail3
    (if 3 ≤ root76Candidates2.length then search edge 3 [401, 809] root76Candidates2 1210 winner else winner) = _
  rw [if_pos (by decide), root76Child2]
def root76Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates3_checked : candidates edge 4 401 887 root76Tail4 winner.bound = root76Candidates3 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
theorem root76Step3 : scan edge (search edge 3) 4 [401] 401 root76Tail3 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail4 winner := by
  rw [root76Tail3, scan, if_neg (by decide), root76Candidates3_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail4
    (if 3 ≤ root76Candidates3.length then search edge 3 [401, 887] root76Candidates3 1288 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates4_checked : candidates edge 4 401 1289 root76Tail5 winner.bound = root76Candidates4 := by
  unfold candidates
  have row : edge 1289 = (SortedNatTree.contains FastRows.row1289) := by
    change Data.row1289.contains = (SortedNatTree.contains FastRows.row1289)
    exact FastRows.row1289_checked.symm
  rw [row]
  decide
theorem root76Step4 : scan edge (search edge 3) 4 [401] 401 root76Tail4 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail5 winner := by
  rw [root76Tail4, scan, if_neg (by decide), root76Candidates4_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail5
    (if 3 ≤ root76Candidates4.length then search edge 3 [401, 1289] root76Candidates4 1690 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates5 : List Nat := [3671, 5801, 19379]
set_option maxRecDepth 8192 in
theorem root76Candidates5_checked : candidates edge 4 401 1487 root76Tail6 winner.bound = root76Candidates5 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root76Child5 : search edge 3 [401, 1487] root76Candidates5 1888 winner = winner := by decide
theorem root76Step5 : scan edge (search edge 3) 4 [401] 401 root76Tail5 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail6 winner := by
  rw [root76Tail5, scan, if_neg (by decide), root76Candidates5_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail6
    (if 3 ≤ root76Candidates5.length then search edge 3 [401, 1487] root76Candidates5 1888 winner else winner) = _
  rw [if_pos (by decide), root76Child5]
def root76Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates6_checked : candidates edge 4 401 2213 root76Tail7 winner.bound = root76Candidates6 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root76Step6 : scan edge (search edge 3) 4 [401] 401 root76Tail6 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail7 winner := by
  rw [root76Tail6, scan, if_neg (by decide), root76Candidates6_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail7
    (if 3 ≤ root76Candidates6.length then search edge 3 [401, 2213] root76Candidates6 2614 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates7 : List Nat := [3671, 7691, 15083]
set_option maxRecDepth 8192 in
theorem root76Candidates7_checked : candidates edge 4 401 2441 root76Tail8 winner.bound = root76Candidates7 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root76Child7 : search edge 3 [401, 2441] root76Candidates7 2842 winner = winner := by decide
theorem root76Step7 : scan edge (search edge 3) 4 [401] 401 root76Tail7 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail8 winner := by
  rw [root76Tail7, scan, if_neg (by decide), root76Candidates7_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail8
    (if 3 ≤ root76Candidates7.length then search edge 3 [401, 2441] root76Candidates7 2842 winner else winner) = _
  rw [if_pos (by decide), root76Child7]
def root76Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates8_checked : candidates edge 4 401 2999 root76Tail9 winner.bound = root76Candidates8 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root76Step8 : scan edge (search edge 3) 4 [401] 401 root76Tail8 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail9 winner := by
  rw [root76Tail8, scan, if_neg (by decide), root76Candidates8_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail9
    (if 3 ≤ root76Candidates8.length then search edge 3 [401, 2999] root76Candidates8 3400 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates9 : List Nat := [3671, 7691]
set_option maxRecDepth 8192 in
theorem root76Candidates9_checked : candidates edge 4 401 3407 root76Tail10 winner.bound = root76Candidates9 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root76Step9 : scan edge (search edge 3) 4 [401] 401 root76Tail9 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail10 winner := by
  rw [root76Tail9, scan, if_neg (by decide), root76Candidates9_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail10
    (if 3 ≤ root76Candidates9.length then search edge 3 [401, 3407] root76Candidates9 3808 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates10 : List Nat := [4271]
set_option maxRecDepth 8192 in
theorem root76Candidates10_checked : candidates edge 4 401 3623 root76Tail11 winner.bound = root76Candidates10 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root76Step10 : scan edge (search edge 3) 4 [401] 401 root76Tail10 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail11 winner := by
  rw [root76Tail10, scan, if_neg (by decide), root76Candidates10_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail11
    (if 3 ≤ root76Candidates10.length then search edge 3 [401, 3623] root76Candidates10 4024 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates11 : List Nat := [7691, 9623, 14537]
set_option maxRecDepth 8192 in
theorem root76Candidates11_checked : candidates edge 4 401 3671 root76Tail12 winner.bound = root76Candidates11 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root76Child11 : search edge 3 [401, 3671] root76Candidates11 4072 winner = winner := by decide
theorem root76Step11 : scan edge (search edge 3) 4 [401] 401 root76Tail11 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail12 winner := by
  rw [root76Tail11, scan, if_neg (by decide), root76Candidates11_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail12
    (if 3 ≤ root76Candidates11.length then search edge 3 [401, 3671] root76Candidates11 4072 winner else winner) = _
  rw [if_pos (by decide), root76Child11]
def root76Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates12_checked : candidates edge 4 401 4133 root76Tail13 winner.bound = root76Candidates12 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root76Step12 : scan edge (search edge 3) 4 [401] 401 root76Tail12 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail13 winner := by
  rw [root76Tail12, scan, if_neg (by decide), root76Candidates12_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail13
    (if 3 ≤ root76Candidates12.length then search edge 3 [401, 4133] root76Candidates12 4534 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates13_checked : candidates edge 4 401 4211 root76Tail14 winner.bound = root76Candidates13 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root76Step13 : scan edge (search edge 3) 4 [401] 401 root76Tail13 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail14 winner := by
  rw [root76Tail13, scan, if_neg (by decide), root76Candidates13_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail14
    (if 3 ≤ root76Candidates13.length then search edge 3 [401, 4211] root76Candidates13 4612 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates14 : List Nat := [14537]
set_option maxRecDepth 8192 in
theorem root76Candidates14_checked : candidates edge 4 401 4217 root76Tail15 winner.bound = root76Candidates14 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root76Step14 : scan edge (search edge 3) 4 [401] 401 root76Tail14 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail15 winner := by
  rw [root76Tail14, scan, if_neg (by decide), root76Candidates14_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail15
    (if 3 ≤ root76Candidates14.length then search edge 3 [401, 4217] root76Candidates14 4618 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates15 : List Nat := [12659, 16691, 20063]
set_option maxRecDepth 8192 in
theorem root76Candidates15_checked : candidates edge 4 401 4271 root76Tail16 winner.bound = root76Candidates15 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root76Child15 : search edge 3 [401, 4271] root76Candidates15 4672 winner = winner := by decide
theorem root76Step15 : scan edge (search edge 3) 4 [401] 401 root76Tail15 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail16 winner := by
  rw [root76Tail15, scan, if_neg (by decide), root76Candidates15_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail16
    (if 3 ≤ root76Candidates15.length then search edge 3 [401, 4271] root76Candidates15 4672 winner else winner) = _
  rw [if_pos (by decide), root76Child15]
def root76Candidates16 : List Nat := [6833, 17783]
set_option maxRecDepth 8192 in
theorem root76Candidates16_checked : candidates edge 4 401 5051 root76Tail17 winner.bound = root76Candidates16 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root76Step16 : scan edge (search edge 3) 4 [401] 401 root76Tail16 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail17 winner := by
  rw [root76Tail16, scan, if_neg (by decide), root76Candidates16_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail17
    (if 3 ≤ root76Candidates16.length then search edge 3 [401, 5051] root76Candidates16 5452 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates17 : List Nat := [12983]
set_option maxRecDepth 8192 in
theorem root76Candidates17_checked : candidates edge 4 401 5801 root76Tail18 winner.bound = root76Candidates17 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root76Step17 : scan edge (search edge 3) 4 [401] 401 root76Tail17 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail18 winner := by
  rw [root76Tail17, scan, if_neg (by decide), root76Candidates17_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail18
    (if 3 ≤ root76Candidates17.length then search edge 3 [401, 5801] root76Candidates17 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root76Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root76Candidates18_checked : candidates edge 4 401 6317 root76Tail19 winner.bound = root76Candidates18 := by
  unfold candidates
  have row : edge 6317 = (SortedNatTree.contains FastRows.row6317) := by
    change Data.row6317.contains = (SortedNatTree.contains FastRows.row6317)
    exact FastRows.row6317_checked.symm
  rw [row]
  decide
theorem root76Step18 : scan edge (search edge 3) 4 [401] 401 root76Tail18 winner =
    scan edge (search edge 3) 4 [401] 401 root76Tail19 winner := by
  rw [root76Tail18, scan, if_neg (by decide), root76Candidates18_checked]
  change scan edge (search edge 3) 4 [401] 401 root76Tail19
    (if 3 ≤ root76Candidates18.length then search edge 3 [401, 6317] root76Candidates18 6718 winner else winner) = _
  rw [if_neg (by decide)]
theorem root76Scan19 : scan edge (search edge 3) 4 [401] 401 root76Tail19 winner = winner := by rw [root76Tail19, scan, if_pos (by decide)]
theorem root76Scan18 : scan edge (search edge 3) 4 [401] 401 root76Tail18 winner = winner :=
  (root76Step18).trans root76Scan19
theorem root76Scan17 : scan edge (search edge 3) 4 [401] 401 root76Tail17 winner = winner :=
  (root76Step17).trans root76Scan18
theorem root76Scan16 : scan edge (search edge 3) 4 [401] 401 root76Tail16 winner = winner :=
  (root76Step16).trans root76Scan17
theorem root76Scan15 : scan edge (search edge 3) 4 [401] 401 root76Tail15 winner = winner :=
  (root76Step15).trans root76Scan16
theorem root76Scan14 : scan edge (search edge 3) 4 [401] 401 root76Tail14 winner = winner :=
  (root76Step14).trans root76Scan15
theorem root76Scan13 : scan edge (search edge 3) 4 [401] 401 root76Tail13 winner = winner :=
  (root76Step13).trans root76Scan14
theorem root76Scan12 : scan edge (search edge 3) 4 [401] 401 root76Tail12 winner = winner :=
  (root76Step12).trans root76Scan13
theorem root76Scan11 : scan edge (search edge 3) 4 [401] 401 root76Tail11 winner = winner :=
  (root76Step11).trans root76Scan12
theorem root76Scan10 : scan edge (search edge 3) 4 [401] 401 root76Tail10 winner = winner :=
  (root76Step10).trans root76Scan11
theorem root76Scan9 : scan edge (search edge 3) 4 [401] 401 root76Tail9 winner = winner :=
  (root76Step9).trans root76Scan10
theorem root76Scan8 : scan edge (search edge 3) 4 [401] 401 root76Tail8 winner = winner :=
  (root76Step8).trans root76Scan9
theorem root76Scan7 : scan edge (search edge 3) 4 [401] 401 root76Tail7 winner = winner :=
  (root76Step7).trans root76Scan8
theorem root76Scan6 : scan edge (search edge 3) 4 [401] 401 root76Tail6 winner = winner :=
  (root76Step6).trans root76Scan7
theorem root76Scan5 : scan edge (search edge 3) 4 [401] 401 root76Tail5 winner = winner :=
  (root76Step5).trans root76Scan6
theorem root76Scan4 : scan edge (search edge 3) 4 [401] 401 root76Tail4 winner = winner :=
  (root76Step4).trans root76Scan5
theorem root76Scan3 : scan edge (search edge 3) 4 [401] 401 root76Tail3 winner = winner :=
  (root76Step3).trans root76Scan4
theorem root76Scan2 : scan edge (search edge 3) 4 [401] 401 root76Tail2 winner = winner :=
  (root76Step2).trans root76Scan3
theorem root76Scan1 : scan edge (search edge 3) 4 [401] 401 root76Tail1 winner = winner :=
  (root76Step1).trans root76Scan2
theorem root76Scan0 : scan edge (search edge 3) 4 [401] 401 root76Tail0 winner = winner :=
  (root76Step0).trans root76Scan1
set_option maxRecDepth 4096 in
theorem root76Branch : search edge 4 [401] root76Candidates 401 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [401] 401 root76Tail0 winner = winner
  exact root76Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
