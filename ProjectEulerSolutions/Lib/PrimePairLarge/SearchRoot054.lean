import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root54Candidates : List Nat := Data.row269
set_option maxRecDepth 8192 in
theorem root54Candidates_checked : candidates edge 5 0 269 tail55 winner.bound = root54Candidates := by
  apply candidates_eq_row edge 5 0 269 tail55 root54Candidates winner.bound tail55_ordered
  · decide
  · decide
  · funext q
    rfl
def root54Tail62 : List Nat := []
def root54Tail61 : List Nat := 25673 :: root54Tail62
def root54Tail60 : List Nat := 25667 :: root54Tail61
def root54Tail59 : List Nat := 23909 :: root54Tail60
def root54Tail58 : List Nat := 23027 :: root54Tail59
def root54Tail57 : List Nat := 22901 :: root54Tail58
def root54Tail56 : List Nat := 22367 :: root54Tail57
def root54Tail55 : List Nat := 22283 :: root54Tail56
def root54Tail54 : List Nat := 22157 :: root54Tail55
def root54Tail53 : List Nat := 21107 :: root54Tail54
def root54Tail52 : List Nat := 19259 :: root54Tail53
def root54Tail51 : List Nat := 19121 :: root54Tail52
def root54Tail50 : List Nat := 19031 :: root54Tail51
def root54Tail49 : List Nat := 18713 :: root54Tail50
def root54Tail48 : List Nat := 18233 :: root54Tail49
def root54Tail47 : List Nat := 17939 :: root54Tail48
def root54Tail46 : List Nat := 17921 :: root54Tail47
def root54Tail45 : List Nat := 17837 :: root54Tail46
def root54Tail44 : List Nat := 17729 :: root54Tail45
def root54Tail43 : List Nat := 17609 :: root54Tail44
def root54Tail42 : List Nat := 16619 :: root54Tail43
def root54Tail41 : List Nat := 16301 :: root54Tail42
def root54Tail40 : List Nat := 16193 :: root54Tail41
def root54Tail39 : List Nat := 16139 :: root54Tail40
def root54Tail38 : List Nat := 16073 :: root54Tail39
def root54Tail37 : List Nat := 15731 :: root54Tail38
def root54Tail36 : List Nat := 15569 :: root54Tail37
def root54Tail35 : List Nat := 15017 :: root54Tail36
def root54Tail34 : List Nat := 13697 :: root54Tail35
def root54Tail33 : List Nat := 13577 :: root54Tail34
def root54Tail32 : List Nat := 12251 :: root54Tail33
def root54Tail31 : List Nat := 11411 :: root54Tail32
def root54Tail30 : List Nat := 11057 :: root54Tail31
def root54Tail29 : List Nat := 10889 :: root54Tail30
def root54Tail28 : List Nat := 10781 :: root54Tail29
def root54Tail27 : List Nat := 9461 :: root54Tail28
def root54Tail26 : List Nat := 8951 :: root54Tail27
def root54Tail25 : List Nat := 8867 :: root54Tail26
def root54Tail24 : List Nat := 8819 :: root54Tail25
def root54Tail23 : List Nat := 8807 :: root54Tail24
def root54Tail22 : List Nat := 8609 :: root54Tail23
def root54Tail21 : List Nat := 8573 :: root54Tail22
def root54Tail20 : List Nat := 7109 :: root54Tail21
def root54Tail19 : List Nat := 6563 :: root54Tail20
def root54Tail18 : List Nat := 6521 :: root54Tail19
def root54Tail17 : List Nat := 6329 :: root54Tail18
def root54Tail16 : List Nat := 6053 :: root54Tail17
def root54Tail15 : List Nat := 5471 :: root54Tail16
def root54Tail14 : List Nat := 5237 :: root54Tail15
def root54Tail13 : List Nat := 5051 :: root54Tail14
def root54Tail12 : List Nat := 4733 :: root54Tail13
def root54Tail11 : List Nat := 4457 :: root54Tail12
def root54Tail10 : List Nat := 4157 :: root54Tail11
def root54Tail9 : List Nat := 4001 :: root54Tail10
def root54Tail8 : List Nat := 2741 :: root54Tail9
def root54Tail7 : List Nat := 2003 :: root54Tail8
def root54Tail6 : List Nat := 1979 :: root54Tail7
def root54Tail5 : List Nat := 887 :: root54Tail6
def root54Tail4 : List Nat := 617 :: root54Tail5
def root54Tail3 : List Nat := 461 :: root54Tail4
def root54Tail2 : List Nat := 431 :: root54Tail3
def root54Tail1 : List Nat := 389 :: root54Tail2
def root54Tail0 : List Nat := 317 :: root54Tail1
def root54Candidates0 : List Nat := [6053]
set_option maxRecDepth 8192 in
theorem root54Candidates0_checked : candidates edge 4 269 317 root54Tail1 winner.bound = root54Candidates0 := by
  unfold candidates
  have row : edge 317 = (SortedNatTree.contains FastRows.row317) := by
    change Data.row317.contains = (SortedNatTree.contains FastRows.row317)
    exact FastRows.row317_checked.symm
  rw [row]
  decide
theorem root54Step0 : scan edge (search edge 3) 4 [269] 269 root54Tail0 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail1 winner := by
  rw [root54Tail0, scan, if_neg (by decide), root54Candidates0_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail1
    (if 3 ≤ root54Candidates0.length then search edge 3 [269, 317] root54Candidates0 586 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates1 : List Nat := [4733, 18233]
set_option maxRecDepth 8192 in
theorem root54Candidates1_checked : candidates edge 4 269 389 root54Tail2 winner.bound = root54Candidates1 := by
  unfold candidates
  have row : edge 389 = (SortedNatTree.contains FastRows.row389) := by
    change Data.row389.contains = (SortedNatTree.contains FastRows.row389)
    exact FastRows.row389_checked.symm
  rw [row]
  decide
theorem root54Step1 : scan edge (search edge 3) 4 [269] 269 root54Tail1 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail2 winner := by
  rw [root54Tail1, scan, if_neg (by decide), root54Candidates1_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail2
    (if 3 ≤ root54Candidates1.length then search edge 3 [269, 389] root54Candidates1 658 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates2 : List Nat := [6329]
set_option maxRecDepth 8192 in
theorem root54Candidates2_checked : candidates edge 4 269 431 root54Tail3 winner.bound = root54Candidates2 := by
  unfold candidates
  have row : edge 431 = (SortedNatTree.contains FastRows.row431) := by
    change Data.row431.contains = (SortedNatTree.contains FastRows.row431)
    exact FastRows.row431_checked.symm
  rw [row]
  decide
theorem root54Step2 : scan edge (search edge 3) 4 [269] 269 root54Tail2 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail3 winner := by
  rw [root54Tail2, scan, if_neg (by decide), root54Candidates2_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail3
    (if 3 ≤ root54Candidates2.length then search edge 3 [269, 431] root54Candidates2 700 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates3 : List Nat := [10781]
set_option maxRecDepth 8192 in
theorem root54Candidates3_checked : candidates edge 4 269 461 root54Tail4 winner.bound = root54Candidates3 := by
  unfold candidates
  have row : edge 461 = (SortedNatTree.contains FastRows.row461) := by
    change Data.row461.contains = (SortedNatTree.contains FastRows.row461)
    exact FastRows.row461_checked.symm
  rw [row]
  decide
theorem root54Step3 : scan edge (search edge 3) 4 [269] 269 root54Tail3 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail4 winner := by
  rw [root54Tail3, scan, if_neg (by decide), root54Candidates3_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail4
    (if 3 ≤ root54Candidates3.length then search edge 3 [269, 461] root54Candidates3 730 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates4 : List Nat := [887, 2741, 4001, 10889, 11057, 17609, 18233]
set_option maxRecDepth 8192 in
theorem root54Candidates4_checked : candidates edge 4 269 617 root54Tail5 winner.bound = root54Candidates4 := by
  unfold candidates
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root54Child4 : search edge 3 [269, 617] root54Candidates4 886 winner = winner := by decide
theorem root54Step4 : scan edge (search edge 3) 4 [269] 269 root54Tail4 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail5 winner := by
  rw [root54Tail4, scan, if_neg (by decide), root54Candidates4_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail5
    (if 3 ≤ root54Candidates4.length then search edge 3 [269, 617] root54Candidates4 886 winner else winner) = _
  rw [if_pos (by decide), root54Child4]
def root54Candidates5 : List Nat := [2741, 4157, 7109, 16139]
set_option maxRecDepth 8192 in
theorem root54Candidates5_checked : candidates edge 4 269 887 root54Tail6 winner.bound = root54Candidates5 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root54Child5 : search edge 3 [269, 887] root54Candidates5 1156 winner = winner := by decide
theorem root54Step5 : scan edge (search edge 3) 4 [269] 269 root54Tail5 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail6 winner := by
  rw [root54Tail5, scan, if_neg (by decide), root54Candidates5_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail6
    (if 3 ≤ root54Candidates5.length then search edge 3 [269, 887] root54Candidates5 1156 winner else winner) = _
  rw [if_pos (by decide), root54Child5]
def root54Candidates6 : List Nat := [22367]
set_option maxRecDepth 8192 in
theorem root54Candidates6_checked : candidates edge 4 269 1979 root54Tail7 winner.bound = root54Candidates6 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root54Step6 : scan edge (search edge 3) 4 [269] 269 root54Tail6 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail7 winner := by
  rw [root54Tail6, scan, if_neg (by decide), root54Candidates6_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail7
    (if 3 ≤ root54Candidates6.length then search edge 3 [269, 1979] root54Candidates6 2248 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates7 : List Nat := [5051, 6563, 8609]
set_option maxRecDepth 8192 in
theorem root54Candidates7_checked : candidates edge 4 269 2003 root54Tail8 winner.bound = root54Candidates7 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root54Child7 : search edge 3 [269, 2003] root54Candidates7 2272 winner = winner := by decide
theorem root54Step7 : scan edge (search edge 3) 4 [269] 269 root54Tail7 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail8 winner := by
  rw [root54Tail7, scan, if_neg (by decide), root54Candidates7_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail8
    (if 3 ≤ root54Candidates7.length then search edge 3 [269, 2003] root54Candidates7 2272 winner else winner) = _
  rw [if_pos (by decide), root54Child7]
def root54Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root54Candidates8_checked : candidates edge 4 269 2741 root54Tail9 winner.bound = root54Candidates8 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root54Step8 : scan edge (search edge 3) 4 [269] 269 root54Tail8 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail9 winner := by
  rw [root54Tail8, scan, if_neg (by decide), root54Candidates8_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail9
    (if 3 ≤ root54Candidates8.length then search edge 3 [269, 2741] root54Candidates8 3010 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates9 : List Nat := [6329, 6563, 17609]
set_option maxRecDepth 8192 in
theorem root54Candidates9_checked : candidates edge 4 269 4001 root54Tail10 winner.bound = root54Candidates9 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root54Child9 : search edge 3 [269, 4001] root54Candidates9 4270 winner = winner := by decide
theorem root54Step9 : scan edge (search edge 3) 4 [269] 269 root54Tail9 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail10 winner := by
  rw [root54Tail9, scan, if_neg (by decide), root54Candidates9_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail10
    (if 3 ≤ root54Candidates9.length then search edge 3 [269, 4001] root54Candidates9 4270 winner else winner) = _
  rw [if_pos (by decide), root54Child9]
def root54Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root54Candidates10_checked : candidates edge 4 269 4157 root54Tail11 winner.bound = root54Candidates10 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root54Step10 : scan edge (search edge 3) 4 [269] 269 root54Tail10 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail11 winner := by
  rw [root54Tail10, scan, if_neg (by decide), root54Candidates10_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail11
    (if 3 ≤ root54Candidates10.length then search edge 3 [269, 4157] root54Candidates10 4426 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates11 : List Nat := [17729]
set_option maxRecDepth 8192 in
theorem root54Candidates11_checked : candidates edge 4 269 4457 root54Tail12 winner.bound = root54Candidates11 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root54Step11 : scan edge (search edge 3) 4 [269] 269 root54Tail11 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail12 winner := by
  rw [root54Tail11, scan, if_neg (by decide), root54Candidates11_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail12
    (if 3 ≤ root54Candidates11.length then search edge 3 [269, 4457] root54Candidates11 4726 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates12 : List Nat := [5237]
set_option maxRecDepth 8192 in
theorem root54Candidates12_checked : candidates edge 4 269 4733 root54Tail13 winner.bound = root54Candidates12 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root54Step12 : scan edge (search edge 3) 4 [269] 269 root54Tail12 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail13 winner := by
  rw [root54Tail12, scan, if_neg (by decide), root54Candidates12_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail13
    (if 3 ≤ root54Candidates12.length then search edge 3 [269, 4733] root54Candidates12 5002 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates13 : List Nat := [17729, 18233]
set_option maxRecDepth 8192 in
theorem root54Candidates13_checked : candidates edge 4 269 5051 root54Tail14 winner.bound = root54Candidates13 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root54Step13 : scan edge (search edge 3) 4 [269] 269 root54Tail13 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail14 winner := by
  rw [root54Tail13, scan, if_neg (by decide), root54Candidates13_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail14
    (if 3 ≤ root54Candidates13.length then search edge 3 [269, 5051] root54Candidates13 5320 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates14 : List Nat := [5471, 6563, 17609, 17921]
set_option maxRecDepth 8192 in
theorem root54Candidates14_checked : candidates edge 4 269 5237 root54Tail15 winner.bound = root54Candidates14 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root54Child14 : search edge 3 [269, 5237] root54Candidates14 5506 winner = winner := by decide
theorem root54Step14 : scan edge (search edge 3) 4 [269] 269 root54Tail14 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail15 winner := by
  rw [root54Tail14, scan, if_neg (by decide), root54Candidates14_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail15
    (if 3 ≤ root54Candidates14.length then search edge 3 [269, 5237] root54Candidates14 5506 winner else winner) = _
  rw [if_pos (by decide), root54Child14]
def root54Candidates15 : List Nat := [15731, 18233]
set_option maxRecDepth 8192 in
theorem root54Candidates15_checked : candidates edge 4 269 5471 root54Tail16 winner.bound = root54Candidates15 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root54Step15 : scan edge (search edge 3) 4 [269] 269 root54Tail15 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail16 winner := by
  rw [root54Tail15, scan, if_neg (by decide), root54Candidates15_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail16
    (if 3 ≤ root54Candidates15.length then search edge 3 [269, 5471] root54Candidates15 5740 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root54Candidates16_checked : candidates edge 4 269 6053 root54Tail17 winner.bound = root54Candidates16 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root54Step16 : scan edge (search edge 3) 4 [269] 269 root54Tail16 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail17 winner := by
  rw [root54Tail16, scan, if_neg (by decide), root54Candidates16_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail17
    (if 3 ≤ root54Candidates16.length then search edge 3 [269, 6053] root54Candidates16 6322 winner else winner) = _
  rw [if_neg (by decide)]
def root54Candidates17 : List Nat := [19121]
set_option maxRecDepth 8192 in
theorem root54Candidates17_checked : candidates edge 4 269 6329 root54Tail18 winner.bound = root54Candidates17 := by
  unfold candidates
  have row : edge 6329 = (SortedNatTree.contains FastRows.row6329) := by
    change Data.row6329.contains = (SortedNatTree.contains FastRows.row6329)
    exact FastRows.row6329_checked.symm
  rw [row]
  decide
theorem root54Step17 : scan edge (search edge 3) 4 [269] 269 root54Tail17 winner =
    scan edge (search edge 3) 4 [269] 269 root54Tail18 winner := by
  rw [root54Tail17, scan, if_neg (by decide), root54Candidates17_checked]
  change scan edge (search edge 3) 4 [269] 269 root54Tail18
    (if 3 ≤ root54Candidates17.length then search edge 3 [269, 6329] root54Candidates17 6598 winner else winner) = _
  rw [if_neg (by decide)]
theorem root54Scan18 : scan edge (search edge 3) 4 [269] 269 root54Tail18 winner = winner := by rw [root54Tail18, scan, if_pos (by decide)]
theorem root54Scan17 : scan edge (search edge 3) 4 [269] 269 root54Tail17 winner = winner :=
  (root54Step17).trans root54Scan18
theorem root54Scan16 : scan edge (search edge 3) 4 [269] 269 root54Tail16 winner = winner :=
  (root54Step16).trans root54Scan17
theorem root54Scan15 : scan edge (search edge 3) 4 [269] 269 root54Tail15 winner = winner :=
  (root54Step15).trans root54Scan16
theorem root54Scan14 : scan edge (search edge 3) 4 [269] 269 root54Tail14 winner = winner :=
  (root54Step14).trans root54Scan15
theorem root54Scan13 : scan edge (search edge 3) 4 [269] 269 root54Tail13 winner = winner :=
  (root54Step13).trans root54Scan14
theorem root54Scan12 : scan edge (search edge 3) 4 [269] 269 root54Tail12 winner = winner :=
  (root54Step12).trans root54Scan13
theorem root54Scan11 : scan edge (search edge 3) 4 [269] 269 root54Tail11 winner = winner :=
  (root54Step11).trans root54Scan12
theorem root54Scan10 : scan edge (search edge 3) 4 [269] 269 root54Tail10 winner = winner :=
  (root54Step10).trans root54Scan11
theorem root54Scan9 : scan edge (search edge 3) 4 [269] 269 root54Tail9 winner = winner :=
  (root54Step9).trans root54Scan10
theorem root54Scan8 : scan edge (search edge 3) 4 [269] 269 root54Tail8 winner = winner :=
  (root54Step8).trans root54Scan9
theorem root54Scan7 : scan edge (search edge 3) 4 [269] 269 root54Tail7 winner = winner :=
  (root54Step7).trans root54Scan8
theorem root54Scan6 : scan edge (search edge 3) 4 [269] 269 root54Tail6 winner = winner :=
  (root54Step6).trans root54Scan7
theorem root54Scan5 : scan edge (search edge 3) 4 [269] 269 root54Tail5 winner = winner :=
  (root54Step5).trans root54Scan6
theorem root54Scan4 : scan edge (search edge 3) 4 [269] 269 root54Tail4 winner = winner :=
  (root54Step4).trans root54Scan5
theorem root54Scan3 : scan edge (search edge 3) 4 [269] 269 root54Tail3 winner = winner :=
  (root54Step3).trans root54Scan4
theorem root54Scan2 : scan edge (search edge 3) 4 [269] 269 root54Tail2 winner = winner :=
  (root54Step2).trans root54Scan3
theorem root54Scan1 : scan edge (search edge 3) 4 [269] 269 root54Tail1 winner = winner :=
  (root54Step1).trans root54Scan2
theorem root54Scan0 : scan edge (search edge 3) 4 [269] 269 root54Tail0 winner = winner :=
  (root54Step0).trans root54Scan1
set_option maxRecDepth 4096 in
theorem root54Branch : search edge 4 [269] root54Candidates 269 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [269] 269 root54Tail0 winner = winner
  exact root54Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
