import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root42Candidates : List Nat := Data.row197
set_option maxRecDepth 8192 in
theorem root42Candidates_checked : candidates edge 5 0 197 tail43 winner.bound = root42Candidates := by
  apply candidates_eq_row edge 5 0 197 tail43 root42Candidates winner.bound tail43_ordered
  · decide
  · decide
  · funext q
    rfl
def root42Tail68 : List Nat := []
def root42Tail67 : List Nat := 25163 :: root42Tail68
def root42Tail66 : List Nat := 24623 :: root42Tail67
def root42Tail65 : List Nat := 23687 :: root42Tail66
def root42Tail64 : List Nat := 23117 :: root42Tail65
def root42Tail63 : List Nat := 22433 :: root42Tail64
def root42Tail62 : List Nat := 22259 :: root42Tail63
def root42Tail61 : List Nat := 21503 :: root42Tail62
def root42Tail60 : List Nat := 21269 :: root42Tail61
def root42Tail59 : List Nat := 20639 :: root42Tail60
def root42Tail58 : List Nat := 20543 :: root42Tail59
def root42Tail57 : List Nat := 19919 :: root42Tail58
def root42Tail56 : List Nat := 19289 :: root42Tail57
def root42Tail55 : List Nat := 18911 :: root42Tail56
def root42Tail54 : List Nat := 18443 :: root42Tail55
def root42Tail53 : List Nat := 16691 :: root42Tail54
def root42Tail52 : List Nat := 16649 :: root42Tail53
def root42Tail51 : List Nat := 16349 :: root42Tail52
def root42Tail50 : List Nat := 16253 :: root42Tail51
def root42Tail49 : List Nat := 16097 :: root42Tail50
def root42Tail48 : List Nat := 15881 :: root42Tail49
def root42Tail47 : List Nat := 15791 :: root42Tail48
def root42Tail46 : List Nat := 15671 :: root42Tail47
def root42Tail45 : List Nat := 15497 :: root42Tail46
def root42Tail44 : List Nat := 15299 :: root42Tail45
def root42Tail43 : List Nat := 14759 :: root42Tail44
def root42Tail42 : List Nat := 14423 :: root42Tail43
def root42Tail41 : List Nat := 14411 :: root42Tail42
def root42Tail40 : List Nat := 13877 :: root42Tail41
def root42Tail39 : List Nat := 13487 :: root42Tail40
def root42Tail38 : List Nat := 13451 :: root42Tail39
def root42Tail37 : List Nat := 13229 :: root42Tail38
def root42Tail36 : List Nat := 12611 :: root42Tail37
def root42Tail35 : List Nat := 12347 :: root42Tail36
def root42Tail34 : List Nat := 11903 :: root42Tail35
def root42Tail33 : List Nat := 10331 :: root42Tail34
def root42Tail32 : List Nat := 9749 :: root42Tail33
def root42Tail31 : List Nat := 9491 :: root42Tail32
def root42Tail30 : List Nat := 9281 :: root42Tail31
def root42Tail29 : List Nat := 8693 :: root42Tail30
def root42Tail28 : List Nat := 8363 :: root42Tail29
def root42Tail27 : List Nat := 8297 :: root42Tail28
def root42Tail26 : List Nat := 7817 :: root42Tail27
def root42Tail25 : List Nat := 7187 :: root42Tail26
def root42Tail24 : List Nat := 6761 :: root42Tail25
def root42Tail23 : List Nat := 6173 :: root42Tail24
def root42Tail22 : List Nat := 6113 :: root42Tail23
def root42Tail21 : List Nat := 5399 :: root42Tail22
def root42Tail20 : List Nat := 5387 :: root42Tail21
def root42Tail19 : List Nat := 5021 :: root42Tail20
def root42Tail18 : List Nat := 4967 :: root42Tail19
def root42Tail17 : List Nat := 4937 :: root42Tail18
def root42Tail16 : List Nat := 4919 :: root42Tail17
def root42Tail15 : List Nat := 4547 :: root42Tail16
def root42Tail14 : List Nat := 4493 :: root42Tail15
def root42Tail13 : List Nat := 4457 :: root42Tail14
def root42Tail12 : List Nat := 4073 :: root42Tail13
def root42Tail11 : List Nat := 2657 :: root42Tail12
def root42Tail10 : List Nat := 2207 :: root42Tail11
def root42Tail9 : List Nat := 2099 :: root42Tail10
def root42Tail8 : List Nat := 1637 :: root42Tail9
def root42Tail7 : List Nat := 1481 :: root42Tail8
def root42Tail6 : List Nat := 1181 :: root42Tail7
def root42Tail5 : List Nat := 971 :: root42Tail6
def root42Tail4 : List Nat := 947 :: root42Tail5
def root42Tail3 : List Nat := 641 :: root42Tail4
def root42Tail2 : List Nat := 569 :: root42Tail3
def root42Tail1 : List Nat := 347 :: root42Tail2
def root42Tail0 : List Nat := 311 :: root42Tail1
def root42Candidates0 : List Nat := [4967, 6761, 18911, 20543]
set_option maxRecDepth 8192 in
theorem root42Candidates0_checked : candidates edge 4 197 311 root42Tail1 winner.bound = root42Candidates0 := by
  unfold candidates
  have row : edge 311 = (SortedNatTree.contains FastRows.row311) := by
    change Data.row311.contains = (SortedNatTree.contains FastRows.row311)
    exact FastRows.row311_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child0 : search edge 3 [197, 311] root42Candidates0 508 winner = winner := by decide
theorem root42Step0 : scan edge (search edge 3) 4 [197] 197 root42Tail0 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail1 winner := by
  rw [root42Tail0, scan, if_neg (by decide), root42Candidates0_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail1
    (if 3 ≤ root42Candidates0.length then search edge 3 [197, 311] root42Candidates0 508 winner else winner) = _
  rw [if_pos (by decide), root42Child0]
def root42Candidates1 : List Nat := [6761, 11903]
set_option maxRecDepth 8192 in
theorem root42Candidates1_checked : candidates edge 4 197 347 root42Tail2 winner.bound = root42Candidates1 := by
  unfold candidates
  have row : edge 347 = (SortedNatTree.contains FastRows.row347) := by
    change Data.row347.contains = (SortedNatTree.contains FastRows.row347)
    exact FastRows.row347_checked.symm
  rw [row]
  decide
theorem root42Step1 : scan edge (search edge 3) 4 [197] 197 root42Tail1 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail2 winner := by
  rw [root42Tail1, scan, if_neg (by decide), root42Candidates1_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail2
    (if 3 ≤ root42Candidates1.length then search edge 3 [197, 347] root42Candidates1 544 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates2 : List Nat := [2657, 4919, 14411]
set_option maxRecDepth 8192 in
theorem root42Candidates2_checked : candidates edge 4 197 569 root42Tail3 winner.bound = root42Candidates2 := by
  unfold candidates
  have row : edge 569 = (SortedNatTree.contains FastRows.row569) := by
    change Data.row569.contains = (SortedNatTree.contains FastRows.row569)
    exact FastRows.row569_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child2 : search edge 3 [197, 569] root42Candidates2 766 winner = winner := by decide
theorem root42Step2 : scan edge (search edge 3) 4 [197] 197 root42Tail2 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail3 winner := by
  rw [root42Tail2, scan, if_neg (by decide), root42Candidates2_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail3
    (if 3 ≤ root42Candidates2.length then search edge 3 [197, 569] root42Candidates2 766 winner else winner) = _
  rw [if_pos (by decide), root42Child2]
def root42Candidates3 : List Nat := [4937, 14411, 22433]
set_option maxRecDepth 8192 in
theorem root42Candidates3_checked : candidates edge 4 197 641 root42Tail4 winner.bound = root42Candidates3 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child3 : search edge 3 [197, 641] root42Candidates3 838 winner = winner := by decide
theorem root42Step3 : scan edge (search edge 3) 4 [197] 197 root42Tail3 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail4 winner := by
  rw [root42Tail3, scan, if_neg (by decide), root42Candidates3_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail4
    (if 3 ≤ root42Candidates3.length then search edge 3 [197, 641] root42Candidates3 838 winner else winner) = _
  rw [if_pos (by decide), root42Child3]
def root42Candidates4 : List Nat := [2657, 20543, 21269, 21503]
set_option maxRecDepth 8192 in
theorem root42Candidates4_checked : candidates edge 4 197 947 root42Tail5 winner.bound = root42Candidates4 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child4 : search edge 3 [197, 947] root42Candidates4 1144 winner = winner := by decide
theorem root42Step4 : scan edge (search edge 3) 4 [197] 197 root42Tail4 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail5 winner := by
  rw [root42Tail4, scan, if_neg (by decide), root42Candidates4_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail5
    (if 3 ≤ root42Candidates4.length then search edge 3 [197, 947] root42Candidates4 1144 winner else winner) = _
  rw [if_pos (by decide), root42Child4]
def root42Candidates5 : List Nat := [21503]
set_option maxRecDepth 8192 in
theorem root42Candidates5_checked : candidates edge 4 197 971 root42Tail6 winner.bound = root42Candidates5 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
theorem root42Step5 : scan edge (search edge 3) 4 [197] 197 root42Tail5 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail6 winner := by
  rw [root42Tail5, scan, if_neg (by decide), root42Candidates5_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail6
    (if 3 ≤ root42Candidates5.length then search edge 3 [197, 971] root42Candidates5 1168 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root42Candidates6_checked : candidates edge 4 197 1181 root42Tail7 winner.bound = root42Candidates6 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
theorem root42Step6 : scan edge (search edge 3) 4 [197] 197 root42Tail6 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail7 winner := by
  rw [root42Tail6, scan, if_neg (by decide), root42Candidates6_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail7
    (if 3 ≤ root42Candidates6.length then search edge 3 [197, 1181] root42Candidates6 1378 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates7 : List Nat := [23117]
set_option maxRecDepth 8192 in
theorem root42Candidates7_checked : candidates edge 4 197 1481 root42Tail8 winner.bound = root42Candidates7 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
theorem root42Step7 : scan edge (search edge 3) 4 [197] 197 root42Tail7 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail8 winner := by
  rw [root42Tail7, scan, if_neg (by decide), root42Candidates7_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail8
    (if 3 ≤ root42Candidates7.length then search edge 3 [197, 1481] root42Candidates7 1678 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates8 : List Nat := [4073, 8363, 12347, 19289]
set_option maxRecDepth 8192 in
theorem root42Candidates8_checked : candidates edge 4 197 1637 root42Tail9 winner.bound = root42Candidates8 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child8 : search edge 3 [197, 1637] root42Candidates8 1834 winner = winner := by decide
theorem root42Step8 : scan edge (search edge 3) 4 [197] 197 root42Tail8 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail9 winner := by
  rw [root42Tail8, scan, if_neg (by decide), root42Candidates8_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail9
    (if 3 ≤ root42Candidates8.length then search edge 3 [197, 1637] root42Candidates8 1834 winner else winner) = _
  rw [if_pos (by decide), root42Child8]
def root42Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root42Candidates9_checked : candidates edge 4 197 2099 root42Tail10 winner.bound = root42Candidates9 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
theorem root42Step9 : scan edge (search edge 3) 4 [197] 197 root42Tail9 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail10 winner := by
  rw [root42Tail9, scan, if_neg (by decide), root42Candidates9_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail10
    (if 3 ≤ root42Candidates9.length then search edge 3 [197, 2099] root42Candidates9 2296 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates10 : List Nat := [8363, 20543]
set_option maxRecDepth 8192 in
theorem root42Candidates10_checked : candidates edge 4 197 2207 root42Tail11 winner.bound = root42Candidates10 := by
  unfold candidates
  have row : edge 2207 = (SortedNatTree.contains FastRows.row2207) := by
    change Data.row2207.contains = (SortedNatTree.contains FastRows.row2207)
    exact FastRows.row2207_checked.symm
  rw [row]
  decide
theorem root42Step10 : scan edge (search edge 3) 4 [197] 197 root42Tail10 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail11 winner := by
  rw [root42Tail10, scan, if_neg (by decide), root42Candidates10_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail11
    (if 3 ≤ root42Candidates10.length then search edge 3 [197, 2207] root42Candidates10 2404 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates11 : List Nat := [18443, 20639]
set_option maxRecDepth 8192 in
theorem root42Candidates11_checked : candidates edge 4 197 2657 root42Tail12 winner.bound = root42Candidates11 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root42Step11 : scan edge (search edge 3) 4 [197] 197 root42Tail11 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail12 winner := by
  rw [root42Tail11, scan, if_neg (by decide), root42Candidates11_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail12
    (if 3 ≤ root42Candidates11.length then search edge 3 [197, 2657] root42Candidates11 2854 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root42Candidates12_checked : candidates edge 4 197 4073 root42Tail13 winner.bound = root42Candidates12 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root42Step12 : scan edge (search edge 3) 4 [197] 197 root42Tail12 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail13 winner := by
  rw [root42Tail12, scan, if_neg (by decide), root42Candidates12_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail13
    (if 3 ≤ root42Candidates12.length then search edge 3 [197, 4073] root42Candidates12 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates13 : List Nat := [9281, 13487, 15791]
set_option maxRecDepth 8192 in
theorem root42Candidates13_checked : candidates edge 4 197 4457 root42Tail14 winner.bound = root42Candidates13 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child13 : search edge 3 [197, 4457] root42Candidates13 4654 winner = winner := by decide
theorem root42Step13 : scan edge (search edge 3) 4 [197] 197 root42Tail13 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail14 winner := by
  rw [root42Tail13, scan, if_neg (by decide), root42Candidates13_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail14
    (if 3 ≤ root42Candidates13.length then search edge 3 [197, 4457] root42Candidates13 4654 winner else winner) = _
  rw [if_pos (by decide), root42Child13]
def root42Candidates14 : List Nat := [6173, 7187, 8693, 15497]
set_option maxRecDepth 8192 in
theorem root42Candidates14_checked : candidates edge 4 197 4493 root42Tail15 winner.bound = root42Candidates14 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root42Child14 : search edge 3 [197, 4493] root42Candidates14 4690 winner = winner := by decide
theorem root42Step14 : scan edge (search edge 3) 4 [197] 197 root42Tail14 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail15 winner := by
  rw [root42Tail14, scan, if_neg (by decide), root42Candidates14_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail15
    (if 3 ≤ root42Candidates14.length then search edge 3 [197, 4493] root42Candidates14 4690 winner else winner) = _
  rw [if_pos (by decide), root42Child14]
def root42Candidates15 : List Nat := [15299]
set_option maxRecDepth 8192 in
theorem root42Candidates15_checked : candidates edge 4 197 4547 root42Tail16 winner.bound = root42Candidates15 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root42Step15 : scan edge (search edge 3) 4 [197] 197 root42Tail15 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail16 winner := by
  rw [root42Tail15, scan, if_neg (by decide), root42Candidates15_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail16
    (if 3 ≤ root42Candidates15.length then search edge 3 [197, 4547] root42Candidates15 4744 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root42Candidates16_checked : candidates edge 4 197 4919 root42Tail17 winner.bound = root42Candidates16 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root42Step16 : scan edge (search edge 3) 4 [197] 197 root42Tail16 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail17 winner := by
  rw [root42Tail16, scan, if_neg (by decide), root42Candidates16_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail17
    (if 3 ≤ root42Candidates16.length then search edge 3 [197, 4919] root42Candidates16 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates17 : List Nat := [5399, 20543]
set_option maxRecDepth 8192 in
theorem root42Candidates17_checked : candidates edge 4 197 4937 root42Tail18 winner.bound = root42Candidates17 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root42Step17 : scan edge (search edge 3) 4 [197] 197 root42Tail17 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail18 winner := by
  rw [root42Tail17, scan, if_neg (by decide), root42Candidates17_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail18
    (if 3 ≤ root42Candidates17.length then search edge 3 [197, 4937] root42Candidates17 5134 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates18 : List Nat := [5387]
set_option maxRecDepth 8192 in
theorem root42Candidates18_checked : candidates edge 4 197 4967 root42Tail19 winner.bound = root42Candidates18 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root42Step18 : scan edge (search edge 3) 4 [197] 197 root42Tail18 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail19 winner := by
  rw [root42Tail18, scan, if_neg (by decide), root42Candidates18_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail19
    (if 3 ≤ root42Candidates18.length then search edge 3 [197, 4967] root42Candidates18 5164 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root42Candidates19_checked : candidates edge 4 197 5021 root42Tail20 winner.bound = root42Candidates19 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root42Step19 : scan edge (search edge 3) 4 [197] 197 root42Tail19 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail20 winner := by
  rw [root42Tail19, scan, if_neg (by decide), root42Candidates19_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail20
    (if 3 ≤ root42Candidates19.length then search edge 3 [197, 5021] root42Candidates19 5218 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates20 : List Nat := [5399]
set_option maxRecDepth 8192 in
theorem root42Candidates20_checked : candidates edge 4 197 5387 root42Tail21 winner.bound = root42Candidates20 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root42Step20 : scan edge (search edge 3) 4 [197] 197 root42Tail20 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail21 winner := by
  rw [root42Tail20, scan, if_neg (by decide), root42Candidates20_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail21
    (if 3 ≤ root42Candidates20.length then search edge 3 [197, 5387] root42Candidates20 5584 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates21 : List Nat := [9749, 15791]
set_option maxRecDepth 8192 in
theorem root42Candidates21_checked : candidates edge 4 197 5399 root42Tail22 winner.bound = root42Candidates21 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root42Step21 : scan edge (search edge 3) 4 [197] 197 root42Tail21 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail22 winner := by
  rw [root42Tail21, scan, if_neg (by decide), root42Candidates21_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail22
    (if 3 ≤ root42Candidates21.length then search edge 3 [197, 5399] root42Candidates21 5596 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates22 : List Nat := [15791]
set_option maxRecDepth 8192 in
theorem root42Candidates22_checked : candidates edge 4 197 6113 root42Tail23 winner.bound = root42Candidates22 := by
  unfold candidates
  have row : edge 6113 = (SortedNatTree.contains FastRows.row6113) := by
    change Data.row6113.contains = (SortedNatTree.contains FastRows.row6113)
    exact FastRows.row6113_checked.symm
  rw [row]
  decide
theorem root42Step22 : scan edge (search edge 3) 4 [197] 197 root42Tail22 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail23 winner := by
  rw [root42Tail22, scan, if_neg (by decide), root42Candidates22_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail23
    (if 3 ≤ root42Candidates22.length then search edge 3 [197, 6113] root42Candidates22 6310 winner else winner) = _
  rw [if_neg (by decide)]
def root42Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root42Candidates23_checked : candidates edge 4 197 6173 root42Tail24 winner.bound = root42Candidates23 := by
  unfold candidates
  have row : edge 6173 = (SortedNatTree.contains FastRows.row6173) := by
    change Data.row6173.contains = (SortedNatTree.contains FastRows.row6173)
    exact FastRows.row6173_checked.symm
  rw [row]
  decide
theorem root42Step23 : scan edge (search edge 3) 4 [197] 197 root42Tail23 winner =
    scan edge (search edge 3) 4 [197] 197 root42Tail24 winner := by
  rw [root42Tail23, scan, if_neg (by decide), root42Candidates23_checked]
  change scan edge (search edge 3) 4 [197] 197 root42Tail24
    (if 3 ≤ root42Candidates23.length then search edge 3 [197, 6173] root42Candidates23 6370 winner else winner) = _
  rw [if_neg (by decide)]
theorem root42Scan24 : scan edge (search edge 3) 4 [197] 197 root42Tail24 winner = winner := by rw [root42Tail24, scan, if_pos (by decide)]
theorem root42Scan23 : scan edge (search edge 3) 4 [197] 197 root42Tail23 winner = winner :=
  (root42Step23).trans root42Scan24
theorem root42Scan22 : scan edge (search edge 3) 4 [197] 197 root42Tail22 winner = winner :=
  (root42Step22).trans root42Scan23
theorem root42Scan21 : scan edge (search edge 3) 4 [197] 197 root42Tail21 winner = winner :=
  (root42Step21).trans root42Scan22
theorem root42Scan20 : scan edge (search edge 3) 4 [197] 197 root42Tail20 winner = winner :=
  (root42Step20).trans root42Scan21
theorem root42Scan19 : scan edge (search edge 3) 4 [197] 197 root42Tail19 winner = winner :=
  (root42Step19).trans root42Scan20
theorem root42Scan18 : scan edge (search edge 3) 4 [197] 197 root42Tail18 winner = winner :=
  (root42Step18).trans root42Scan19
theorem root42Scan17 : scan edge (search edge 3) 4 [197] 197 root42Tail17 winner = winner :=
  (root42Step17).trans root42Scan18
theorem root42Scan16 : scan edge (search edge 3) 4 [197] 197 root42Tail16 winner = winner :=
  (root42Step16).trans root42Scan17
theorem root42Scan15 : scan edge (search edge 3) 4 [197] 197 root42Tail15 winner = winner :=
  (root42Step15).trans root42Scan16
theorem root42Scan14 : scan edge (search edge 3) 4 [197] 197 root42Tail14 winner = winner :=
  (root42Step14).trans root42Scan15
theorem root42Scan13 : scan edge (search edge 3) 4 [197] 197 root42Tail13 winner = winner :=
  (root42Step13).trans root42Scan14
theorem root42Scan12 : scan edge (search edge 3) 4 [197] 197 root42Tail12 winner = winner :=
  (root42Step12).trans root42Scan13
theorem root42Scan11 : scan edge (search edge 3) 4 [197] 197 root42Tail11 winner = winner :=
  (root42Step11).trans root42Scan12
theorem root42Scan10 : scan edge (search edge 3) 4 [197] 197 root42Tail10 winner = winner :=
  (root42Step10).trans root42Scan11
theorem root42Scan9 : scan edge (search edge 3) 4 [197] 197 root42Tail9 winner = winner :=
  (root42Step9).trans root42Scan10
theorem root42Scan8 : scan edge (search edge 3) 4 [197] 197 root42Tail8 winner = winner :=
  (root42Step8).trans root42Scan9
theorem root42Scan7 : scan edge (search edge 3) 4 [197] 197 root42Tail7 winner = winner :=
  (root42Step7).trans root42Scan8
theorem root42Scan6 : scan edge (search edge 3) 4 [197] 197 root42Tail6 winner = winner :=
  (root42Step6).trans root42Scan7
theorem root42Scan5 : scan edge (search edge 3) 4 [197] 197 root42Tail5 winner = winner :=
  (root42Step5).trans root42Scan6
theorem root42Scan4 : scan edge (search edge 3) 4 [197] 197 root42Tail4 winner = winner :=
  (root42Step4).trans root42Scan5
theorem root42Scan3 : scan edge (search edge 3) 4 [197] 197 root42Tail3 winner = winner :=
  (root42Step3).trans root42Scan4
theorem root42Scan2 : scan edge (search edge 3) 4 [197] 197 root42Tail2 winner = winner :=
  (root42Step2).trans root42Scan3
theorem root42Scan1 : scan edge (search edge 3) 4 [197] 197 root42Tail1 winner = winner :=
  (root42Step1).trans root42Scan2
theorem root42Scan0 : scan edge (search edge 3) 4 [197] 197 root42Tail0 winner = winner :=
  (root42Step0).trans root42Scan1
set_option maxRecDepth 4096 in
theorem root42Branch : search edge 4 [197] root42Candidates 197 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [197] 197 root42Tail0 winner = winner
  exact root42Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
