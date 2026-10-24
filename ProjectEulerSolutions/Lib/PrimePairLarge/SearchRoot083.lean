import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root83Candidates : List Nat := Data.row443
set_option maxRecDepth 8192 in
theorem root83Candidates_checked : candidates edge 5 0 443 tail84 winner.bound = root83Candidates := by
  apply candidates_eq_row edge 5 0 443 tail84 root83Candidates winner.bound tail84_ordered
  · decide
  · decide
  · funext q
    rfl
def root83Tail71 : List Nat := []
def root83Tail70 : List Nat := 25121 :: root83Tail71
def root83Tail69 : List Nat := 25037 :: root83Tail70
def root83Tail68 : List Nat := 24923 :: root83Tail69
def root83Tail67 : List Nat := 24851 :: root83Tail68
def root83Tail66 : List Nat := 24419 :: root83Tail67
def root83Tail65 : List Nat := 24317 :: root83Tail66
def root83Tail64 : List Nat := 24107 :: root83Tail65
def root83Tail63 : List Nat := 23603 :: root83Tail64
def root83Tail62 : List Nat := 23567 :: root83Tail63
def root83Tail61 : List Nat := 23549 :: root83Tail62
def root83Tail60 : List Nat := 23537 :: root83Tail61
def root83Tail59 : List Nat := 22709 :: root83Tail60
def root83Tail58 : List Nat := 20807 :: root83Tail59
def root83Tail57 : List Nat := 20789 :: root83Tail58
def root83Tail56 : List Nat := 20249 :: root83Tail57
def root83Tail55 : List Nat := 20063 :: root83Tail56
def root83Tail54 : List Nat := 19661 :: root83Tail55
def root83Tail53 : List Nat := 19559 :: root83Tail54
def root83Tail52 : List Nat := 18749 :: root83Tail53
def root83Tail51 : List Nat := 17729 :: root83Tail52
def root83Tail50 : List Nat := 16931 :: root83Tail51
def root83Tail49 : List Nat := 16451 :: root83Tail50
def root83Tail48 : List Nat := 15569 :: root83Tail49
def root83Tail47 : List Nat := 15053 :: root83Tail48
def root83Tail46 : List Nat := 14591 :: root83Tail47
def root83Tail45 : List Nat := 14033 :: root83Tail46
def root83Tail44 : List Nat := 13931 :: root83Tail45
def root83Tail43 : List Nat := 13397 :: root83Tail44
def root83Tail42 : List Nat := 12491 :: root83Tail43
def root83Tail41 : List Nat := 12071 :: root83Tail42
def root83Tail40 : List Nat := 11717 :: root83Tail41
def root83Tail39 : List Nat := 11273 :: root83Tail40
def root83Tail38 : List Nat := 11261 :: root83Tail39
def root83Tail37 : List Nat := 10457 :: root83Tail38
def root83Tail36 : List Nat := 10181 :: root83Tail37
def root83Tail35 : List Nat := 10169 :: root83Tail36
def root83Tail34 : List Nat := 9857 :: root83Tail35
def root83Tail33 : List Nat := 9377 :: root83Tail34
def root83Tail32 : List Nat := 9341 :: root83Tail33
def root83Tail31 : List Nat := 8699 :: root83Tail32
def root83Tail30 : List Nat := 8573 :: root83Tail31
def root83Tail29 : List Nat := 8501 :: root83Tail30
def root83Tail28 : List Nat := 7883 :: root83Tail29
def root83Tail27 : List Nat := 7757 :: root83Tail28
def root83Tail26 : List Nat := 7547 :: root83Tail27
def root83Tail25 : List Nat := 7529 :: root83Tail26
def root83Tail24 : List Nat := 7523 :: root83Tail25
def root83Tail23 : List Nat := 7283 :: root83Tail24
def root83Tail22 : List Nat := 6863 :: root83Tail23
def root83Tail21 : List Nat := 6737 :: root83Tail22
def root83Tail20 : List Nat := 6389 :: root83Tail21
def root83Tail19 : List Nat := 5897 :: root83Tail20
def root83Tail18 : List Nat := 5741 :: root83Tail19
def root83Tail17 : List Nat := 5051 :: root83Tail18
def root83Tail16 : List Nat := 4691 :: root83Tail17
def root83Tail15 : List Nat := 4673 :: root83Tail16
def root83Tail14 : List Nat := 3881 :: root83Tail15
def root83Tail13 : List Nat := 3833 :: root83Tail14
def root83Tail12 : List Nat := 3449 :: root83Tail13
def root83Tail11 : List Nat := 3203 :: root83Tail12
def root83Tail10 : List Nat := 3167 :: root83Tail11
def root83Tail9 : List Nat := 2693 :: root83Tail10
def root83Tail8 : List Nat := 2531 :: root83Tail9
def root83Tail7 : List Nat := 2069 :: root83Tail8
def root83Tail6 : List Nat := 1811 :: root83Tail7
def root83Tail5 : List Nat := 1523 :: root83Tail6
def root83Tail4 : List Nat := 1151 :: root83Tail5
def root83Tail3 : List Nat := 983 :: root83Tail4
def root83Tail2 : List Nat := 953 :: root83Tail3
def root83Tail1 : List Nat := 761 :: root83Tail2
def root83Tail0 : List Nat := 701 :: root83Tail1
def root83Candidates0 : List Nat := [6389, 12491, 16931, 18749]
set_option maxRecDepth 8192 in
theorem root83Candidates0_checked : candidates edge 4 443 701 root83Tail1 winner.bound = root83Candidates0 := by
  unfold candidates
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child0 : search edge 3 [443, 701] root83Candidates0 1144 winner = winner := by decide
theorem root83Step0 : scan edge (search edge 3) 4 [443] 443 root83Tail0 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail1 winner := by
  rw [root83Tail0, scan, if_neg (by decide), root83Candidates0_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail1
    (if 3 ≤ root83Candidates0.length then search edge 3 [443, 701] root83Candidates0 1144 winner else winner) = _
  rw [if_pos (by decide), root83Child0]
def root83Candidates1 : List Nat := [2531, 7283, 7529, 13931, 20789]
set_option maxRecDepth 8192 in
theorem root83Candidates1_checked : candidates edge 4 443 761 root83Tail2 winner.bound = root83Candidates1 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child1 : search edge 3 [443, 761] root83Candidates1 1204 winner = winner := by decide
theorem root83Step1 : scan edge (search edge 3) 4 [443] 443 root83Tail1 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail2 winner := by
  rw [root83Tail1, scan, if_neg (by decide), root83Candidates1_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail2
    (if 3 ≤ root83Candidates1.length then search edge 3 [443, 761] root83Candidates1 1204 winner else winner) = _
  rw [if_pos (by decide), root83Child1]
def root83Candidates2 : List Nat := [1523, 10181, 14591, 23537]
set_option maxRecDepth 8192 in
theorem root83Candidates2_checked : candidates edge 4 443 953 root83Tail3 winner.bound = root83Candidates2 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child2 : search edge 3 [443, 953] root83Candidates2 1396 winner = winner := by decide
theorem root83Step2 : scan edge (search edge 3) 4 [443] 443 root83Tail2 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail3 winner := by
  rw [root83Tail2, scan, if_neg (by decide), root83Candidates2_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail3
    (if 3 ≤ root83Candidates2.length then search edge 3 [443, 953] root83Candidates2 1396 winner else winner) = _
  rw [if_pos (by decide), root83Child2]
def root83Candidates3 : List Nat := [6389, 9857]
set_option maxRecDepth 8192 in
theorem root83Candidates3_checked : candidates edge 4 443 983 root83Tail4 winner.bound = root83Candidates3 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root83Step3 : scan edge (search edge 3) 4 [443] 443 root83Tail3 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail4 winner := by
  rw [root83Tail3, scan, if_neg (by decide), root83Candidates3_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail4
    (if 3 ≤ root83Candidates3.length then search edge 3 [443, 983] root83Candidates3 1426 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates4 : List Nat := [5897, 6863, 16931]
set_option maxRecDepth 8192 in
theorem root83Candidates4_checked : candidates edge 4 443 1151 root83Tail5 winner.bound = root83Candidates4 := by
  unfold candidates
  have row : edge 1151 = (SortedNatTree.contains FastRows.row1151) := by
    change Data.row1151.contains = (SortedNatTree.contains FastRows.row1151)
    exact FastRows.row1151_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child4 : search edge 3 [443, 1151] root83Candidates4 1594 winner = winner := by decide
theorem root83Step4 : scan edge (search edge 3) 4 [443] 443 root83Tail4 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail5 winner := by
  rw [root83Tail4, scan, if_neg (by decide), root83Candidates4_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail5
    (if 3 ≤ root83Candidates4.length then search edge 3 [443, 1151] root83Candidates4 1594 winner else winner) = _
  rw [if_pos (by decide), root83Child4]
def root83Candidates5 : List Nat := [3203, 3833, 9377]
set_option maxRecDepth 8192 in
theorem root83Candidates5_checked : candidates edge 4 443 1523 root83Tail6 winner.bound = root83Candidates5 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child5 : search edge 3 [443, 1523] root83Candidates5 1966 winner = winner := by decide
theorem root83Step5 : scan edge (search edge 3) 4 [443] 443 root83Tail5 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail6 winner := by
  rw [root83Tail5, scan, if_neg (by decide), root83Candidates5_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail6
    (if 3 ≤ root83Candidates5.length then search edge 3 [443, 1523] root83Candidates5 1966 winner else winner) = _
  rw [if_pos (by decide), root83Child5]
def root83Candidates6 : List Nat := [6863, 8699, 14033, 16451]
set_option maxRecDepth 8192 in
theorem root83Candidates6_checked : candidates edge 4 443 1811 root83Tail7 winner.bound = root83Candidates6 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child6 : search edge 3 [443, 1811] root83Candidates6 2254 winner = winner := by decide
theorem root83Step6 : scan edge (search edge 3) 4 [443] 443 root83Tail6 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail7 winner := by
  rw [root83Tail6, scan, if_neg (by decide), root83Candidates6_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail7
    (if 3 ≤ root83Candidates6.length then search edge 3 [443, 1811] root83Candidates6 2254 winner else winner) = _
  rw [if_pos (by decide), root83Child6]
def root83Candidates7 : List Nat := [4691, 7283, 7757, 14033, 20807]
set_option maxRecDepth 8192 in
theorem root83Candidates7_checked : candidates edge 4 443 2069 root83Tail8 winner.bound = root83Candidates7 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child7 : search edge 3 [443, 2069] root83Candidates7 2512 winner = winner := by decide
theorem root83Step7 : scan edge (search edge 3) 4 [443] 443 root83Tail7 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail8 winner := by
  rw [root83Tail7, scan, if_neg (by decide), root83Candidates7_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail8
    (if 3 ≤ root83Candidates7.length then search edge 3 [443, 2069] root83Candidates7 2512 winner else winner) = _
  rw [if_pos (by decide), root83Child7]
def root83Candidates8 : List Nat := [7883]
set_option maxRecDepth 8192 in
theorem root83Candidates8_checked : candidates edge 4 443 2531 root83Tail9 winner.bound = root83Candidates8 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
theorem root83Step8 : scan edge (search edge 3) 4 [443] 443 root83Tail8 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail9 winner := by
  rw [root83Tail8, scan, if_neg (by decide), root83Candidates8_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail9
    (if 3 ≤ root83Candidates8.length then search edge 3 [443, 2531] root83Candidates8 2974 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates9 : List Nat := [8501]
set_option maxRecDepth 8192 in
theorem root83Candidates9_checked : candidates edge 4 443 2693 root83Tail10 winner.bound = root83Candidates9 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root83Step9 : scan edge (search edge 3) 4 [443] 443 root83Tail9 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail10 winner := by
  rw [root83Tail9, scan, if_neg (by decide), root83Candidates9_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail10
    (if 3 ≤ root83Candidates9.length then search edge 3 [443, 2693] root83Candidates9 3136 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates10 : List Nat := [9377]
set_option maxRecDepth 8192 in
theorem root83Candidates10_checked : candidates edge 4 443 3167 root83Tail11 winner.bound = root83Candidates10 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root83Step10 : scan edge (search edge 3) 4 [443] 443 root83Tail10 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail11 winner := by
  rw [root83Tail10, scan, if_neg (by decide), root83Candidates10_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail11
    (if 3 ≤ root83Candidates10.length then search edge 3 [443, 3167] root83Candidates10 3610 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates11 : List Nat := [18749]
set_option maxRecDepth 8192 in
theorem root83Candidates11_checked : candidates edge 4 443 3203 root83Tail12 winner.bound = root83Candidates11 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
theorem root83Step11 : scan edge (search edge 3) 4 [443] 443 root83Tail11 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail12 winner := by
  rw [root83Tail11, scan, if_neg (by decide), root83Candidates11_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail12
    (if 3 ≤ root83Candidates11.length then search edge 3 [443, 3203] root83Candidates11 3646 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates12 : List Nat := [20063]
set_option maxRecDepth 8192 in
theorem root83Candidates12_checked : candidates edge 4 443 3449 root83Tail13 winner.bound = root83Candidates12 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root83Step12 : scan edge (search edge 3) 4 [443] 443 root83Tail12 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail13 winner := by
  rw [root83Tail12, scan, if_neg (by decide), root83Candidates12_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail13
    (if 3 ≤ root83Candidates12.length then search edge 3 [443, 3449] root83Candidates12 3892 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates13 : List Nat := [9857]
set_option maxRecDepth 8192 in
theorem root83Candidates13_checked : candidates edge 4 443 3833 root83Tail14 winner.bound = root83Candidates13 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root83Step13 : scan edge (search edge 3) 4 [443] 443 root83Tail13 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail14 winner := by
  rw [root83Tail13, scan, if_neg (by decide), root83Candidates13_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail14
    (if 3 ≤ root83Candidates13.length then search edge 3 [443, 3833] root83Candidates13 4276 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates14 : List Nat := [14591, 15053]
set_option maxRecDepth 8192 in
theorem root83Candidates14_checked : candidates edge 4 443 3881 root83Tail15 winner.bound = root83Candidates14 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root83Step14 : scan edge (search edge 3) 4 [443] 443 root83Tail14 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail15 winner := by
  rw [root83Tail14, scan, if_neg (by decide), root83Candidates14_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail15
    (if 3 ≤ root83Candidates14.length then search edge 3 [443, 3881] root83Candidates14 4324 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates15 : List Nat := [6737]
set_option maxRecDepth 8192 in
theorem root83Candidates15_checked : candidates edge 4 443 4673 root83Tail16 winner.bound = root83Candidates15 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root83Step15 : scan edge (search edge 3) 4 [443] 443 root83Tail15 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail16 winner := by
  rw [root83Tail15, scan, if_neg (by decide), root83Candidates15_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail16
    (if 3 ≤ root83Candidates15.length then search edge 3 [443, 4673] root83Candidates15 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root83Candidates16_checked : candidates edge 4 443 4691 root83Tail17 winner.bound = root83Candidates16 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root83Step16 : scan edge (search edge 3) 4 [443] 443 root83Tail16 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail17 winner := by
  rw [root83Tail16, scan, if_neg (by decide), root83Candidates16_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail17
    (if 3 ≤ root83Candidates16.length then search edge 3 [443, 4691] root83Candidates16 5134 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates17 : List Nat := [6737, 6863, 7283, 17729]
set_option maxRecDepth 8192 in
theorem root83Candidates17_checked : candidates edge 4 443 5051 root83Tail18 winner.bound = root83Candidates17 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root83Child17 : search edge 3 [443, 5051] root83Candidates17 5494 winner = winner := by decide
theorem root83Step17 : scan edge (search edge 3) 4 [443] 443 root83Tail17 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail18 winner := by
  rw [root83Tail17, scan, if_neg (by decide), root83Candidates17_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail18
    (if 3 ≤ root83Candidates17.length then search edge 3 [443, 5051] root83Candidates17 5494 winner else winner) = _
  rw [if_pos (by decide), root83Child17]
def root83Candidates18 : List Nat := [13931]
set_option maxRecDepth 8192 in
theorem root83Candidates18_checked : candidates edge 4 443 5741 root83Tail19 winner.bound = root83Candidates18 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
theorem root83Step18 : scan edge (search edge 3) 4 [443] 443 root83Tail18 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail19 winner := by
  rw [root83Tail18, scan, if_neg (by decide), root83Candidates18_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail19
    (if 3 ≤ root83Candidates18.length then search edge 3 [443, 5741] root83Candidates18 6184 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates19 : List Nat := [11261]
set_option maxRecDepth 8192 in
theorem root83Candidates19_checked : candidates edge 4 443 5897 root83Tail20 winner.bound = root83Candidates19 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root83Step19 : scan edge (search edge 3) 4 [443] 443 root83Tail19 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail20 winner := by
  rw [root83Tail19, scan, if_neg (by decide), root83Candidates19_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail20
    (if 3 ≤ root83Candidates19.length then search edge 3 [443, 5897] root83Candidates19 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root83Candidates20 : List Nat := [7523]
set_option maxRecDepth 8192 in
theorem root83Candidates20_checked : candidates edge 4 443 6389 root83Tail21 winner.bound = root83Candidates20 := by
  unfold candidates
  have row : edge 6389 = (SortedNatTree.contains FastRows.row6389) := by
    change Data.row6389.contains = (SortedNatTree.contains FastRows.row6389)
    exact FastRows.row6389_checked.symm
  rw [row]
  decide
theorem root83Step20 : scan edge (search edge 3) 4 [443] 443 root83Tail20 winner =
    scan edge (search edge 3) 4 [443] 443 root83Tail21 winner := by
  rw [root83Tail20, scan, if_neg (by decide), root83Candidates20_checked]
  change scan edge (search edge 3) 4 [443] 443 root83Tail21
    (if 3 ≤ root83Candidates20.length then search edge 3 [443, 6389] root83Candidates20 6832 winner else winner) = _
  rw [if_neg (by decide)]
theorem root83Scan21 : scan edge (search edge 3) 4 [443] 443 root83Tail21 winner = winner := by rw [root83Tail21, scan, if_pos (by decide)]
theorem root83Scan20 : scan edge (search edge 3) 4 [443] 443 root83Tail20 winner = winner :=
  (root83Step20).trans root83Scan21
theorem root83Scan19 : scan edge (search edge 3) 4 [443] 443 root83Tail19 winner = winner :=
  (root83Step19).trans root83Scan20
theorem root83Scan18 : scan edge (search edge 3) 4 [443] 443 root83Tail18 winner = winner :=
  (root83Step18).trans root83Scan19
theorem root83Scan17 : scan edge (search edge 3) 4 [443] 443 root83Tail17 winner = winner :=
  (root83Step17).trans root83Scan18
theorem root83Scan16 : scan edge (search edge 3) 4 [443] 443 root83Tail16 winner = winner :=
  (root83Step16).trans root83Scan17
theorem root83Scan15 : scan edge (search edge 3) 4 [443] 443 root83Tail15 winner = winner :=
  (root83Step15).trans root83Scan16
theorem root83Scan14 : scan edge (search edge 3) 4 [443] 443 root83Tail14 winner = winner :=
  (root83Step14).trans root83Scan15
theorem root83Scan13 : scan edge (search edge 3) 4 [443] 443 root83Tail13 winner = winner :=
  (root83Step13).trans root83Scan14
theorem root83Scan12 : scan edge (search edge 3) 4 [443] 443 root83Tail12 winner = winner :=
  (root83Step12).trans root83Scan13
theorem root83Scan11 : scan edge (search edge 3) 4 [443] 443 root83Tail11 winner = winner :=
  (root83Step11).trans root83Scan12
theorem root83Scan10 : scan edge (search edge 3) 4 [443] 443 root83Tail10 winner = winner :=
  (root83Step10).trans root83Scan11
theorem root83Scan9 : scan edge (search edge 3) 4 [443] 443 root83Tail9 winner = winner :=
  (root83Step9).trans root83Scan10
theorem root83Scan8 : scan edge (search edge 3) 4 [443] 443 root83Tail8 winner = winner :=
  (root83Step8).trans root83Scan9
theorem root83Scan7 : scan edge (search edge 3) 4 [443] 443 root83Tail7 winner = winner :=
  (root83Step7).trans root83Scan8
theorem root83Scan6 : scan edge (search edge 3) 4 [443] 443 root83Tail6 winner = winner :=
  (root83Step6).trans root83Scan7
theorem root83Scan5 : scan edge (search edge 3) 4 [443] 443 root83Tail5 winner = winner :=
  (root83Step5).trans root83Scan6
theorem root83Scan4 : scan edge (search edge 3) 4 [443] 443 root83Tail4 winner = winner :=
  (root83Step4).trans root83Scan5
theorem root83Scan3 : scan edge (search edge 3) 4 [443] 443 root83Tail3 winner = winner :=
  (root83Step3).trans root83Scan4
theorem root83Scan2 : scan edge (search edge 3) 4 [443] 443 root83Tail2 winner = winner :=
  (root83Step2).trans root83Scan3
theorem root83Scan1 : scan edge (search edge 3) 4 [443] 443 root83Tail1 winner = winner :=
  (root83Step1).trans root83Scan2
theorem root83Scan0 : scan edge (search edge 3) 4 [443] 443 root83Tail0 winner = winner :=
  (root83Step0).trans root83Scan1
set_option maxRecDepth 4096 in
theorem root83Branch : search edge 4 [443] root83Candidates 443 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [443] 443 root83Tail0 winner = winner
  exact root83Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
