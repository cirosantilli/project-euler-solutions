import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root110Candidates : List Nat := Data.row617
set_option maxRecDepth 8192 in
theorem root110Candidates_checked : candidates edge 5 0 617 tail111 winner.bound = root110Candidates := by
  apply candidates_eq_row edge 5 0 617 tail111 root110Candidates winner.bound tail111_ordered
  · decide
  · decide
  · funext q
    rfl
def root110Tail64 : List Nat := []
def root110Tail63 : List Nat := 25373 :: root110Tail64
def root110Tail62 : List Nat := 24821 :: root110Tail63
def root110Tail61 : List Nat := 23537 :: root110Tail62
def root110Tail60 : List Nat := 22277 :: root110Tail61
def root110Tail59 : List Nat := 22109 :: root110Tail60
def root110Tail58 : List Nat := 20369 :: root110Tail59
def root110Tail57 : List Nat := 19919 :: root110Tail58
def root110Tail56 : List Nat := 19709 :: root110Tail57
def root110Tail55 : List Nat := 19421 :: root110Tail56
def root110Tail54 : List Nat := 19211 :: root110Tail55
def root110Tail53 : List Nat := 19157 :: root110Tail54
def root110Tail52 : List Nat := 19001 :: root110Tail53
def root110Tail51 : List Nat := 18539 :: root110Tail52
def root110Tail50 : List Nat := 18233 :: root110Tail51
def root110Tail49 : List Nat := 18077 :: root110Tail50
def root110Tail48 : List Nat := 17609 :: root110Tail49
def root110Tail47 : List Nat := 17483 :: root110Tail48
def root110Tail46 : List Nat := 17321 :: root110Tail47
def root110Tail45 : List Nat := 16649 :: root110Tail46
def root110Tail44 : List Nat := 15887 :: root110Tail45
def root110Tail43 : List Nat := 15581 :: root110Tail44
def root110Tail42 : List Nat := 15053 :: root110Tail43
def root110Tail41 : List Nat := 14699 :: root110Tail42
def root110Tail40 : List Nat := 14669 :: root110Tail41
def root110Tail39 : List Nat := 14621 :: root110Tail40
def root110Tail38 : List Nat := 14447 :: root110Tail39
def root110Tail37 : List Nat := 13997 :: root110Tail38
def root110Tail36 : List Nat := 12653 :: root110Tail37
def root110Tail35 : List Nat := 12569 :: root110Tail36
def root110Tail34 : List Nat := 12479 :: root110Tail35
def root110Tail33 : List Nat := 11633 :: root110Tail34
def root110Tail32 : List Nat := 11213 :: root110Tail33
def root110Tail31 : List Nat := 11057 :: root110Tail32
def root110Tail30 : List Nat := 10973 :: root110Tail31
def root110Tail29 : List Nat := 10937 :: root110Tail30
def root110Tail28 : List Nat := 10889 :: root110Tail29
def root110Tail27 : List Nat := 9473 :: root110Tail28
def root110Tail26 : List Nat := 9221 :: root110Tail27
def root110Tail25 : List Nat := 8969 :: root110Tail26
def root110Tail24 : List Nat := 8747 :: root110Tail25
def root110Tail23 : List Nat := 8693 :: root110Tail24
def root110Tail22 : List Nat := 8243 :: root110Tail23
def root110Tail21 : List Nat := 8087 :: root110Tail22
def root110Tail20 : List Nat := 7541 :: root110Tail21
def root110Tail19 : List Nat := 7229 :: root110Tail20
def root110Tail18 : List Nat := 6323 :: root110Tail19
def root110Tail17 : List Nat := 6011 :: root110Tail18
def root110Tail16 : List Nat := 5693 :: root110Tail17
def root110Tail15 : List Nat := 5333 :: root110Tail16
def root110Tail14 : List Nat := 5261 :: root110Tail15
def root110Tail13 : List Nat := 4391 :: root110Tail14
def root110Tail12 : List Nat := 4253 :: root110Tail13
def root110Tail11 : List Nat := 4001 :: root110Tail12
def root110Tail10 : List Nat := 3593 :: root110Tail11
def root110Tail9 : List Nat := 2909 :: root110Tail10
def root110Tail8 : List Nat := 2741 :: root110Tail9
def root110Tail7 : List Nat := 2657 :: root110Tail8
def root110Tail6 : List Nat := 2063 :: root110Tail7
def root110Tail5 : List Nat := 1427 :: root110Tail6
def root110Tail4 : List Nat := 1301 :: root110Tail5
def root110Tail3 : List Nat := 1091 :: root110Tail4
def root110Tail2 : List Nat := 983 :: root110Tail3
def root110Tail1 : List Nat := 887 :: root110Tail2
def root110Tail0 : List Nat := 647 :: root110Tail1
def root110Candidates0 : List Nat := [14669]
set_option maxRecDepth 8192 in
theorem root110Candidates0_checked : candidates edge 4 617 647 root110Tail1 winner.bound = root110Candidates0 := by
  unfold candidates
  have row : edge 647 = (SortedNatTree.contains FastRows.row647) := by
    change Data.row647.contains = (SortedNatTree.contains FastRows.row647)
    exact FastRows.row647_checked.symm
  rw [row]
  decide
theorem root110Step0 : scan edge (search edge 3) 4 [617] 617 root110Tail0 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail1 winner := by
  rw [root110Tail0, scan, if_neg (by decide), root110Candidates0_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail1
    (if 3 ≤ root110Candidates0.length then search edge 3 [617, 647] root110Candidates0 1264 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates1 : List Nat := [1091, 1427, 2741, 8087, 15581]
set_option maxRecDepth 8192 in
theorem root110Candidates1_checked : candidates edge 4 617 887 root110Tail2 winner.bound = root110Candidates1 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root110Child1 : search edge 3 [617, 887] root110Candidates1 1504 winner = winner := by decide
theorem root110Step1 : scan edge (search edge 3) 4 [617] 617 root110Tail1 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail2 winner := by
  rw [root110Tail1, scan, if_neg (by decide), root110Candidates1_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail2
    (if 3 ≤ root110Candidates1.length then search edge 3 [617, 887] root110Candidates1 1504 winner else winner) = _
  rw [if_pos (by decide), root110Child1]
def root110Candidates2 : List Nat := [1091, 8693]
set_option maxRecDepth 8192 in
theorem root110Candidates2_checked : candidates edge 4 617 983 root110Tail3 winner.bound = root110Candidates2 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root110Step2 : scan edge (search edge 3) 4 [617] 617 root110Tail2 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail3 winner := by
  rw [root110Tail2, scan, if_neg (by decide), root110Candidates2_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail3
    (if 3 ≤ root110Candidates2.length then search edge 3 [617, 983] root110Candidates2 1600 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates3 : List Nat := [15581]
set_option maxRecDepth 8192 in
theorem root110Candidates3_checked : candidates edge 4 617 1091 root110Tail4 winner.bound = root110Candidates3 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
theorem root110Step3 : scan edge (search edge 3) 4 [617] 617 root110Tail3 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail4 winner := by
  rw [root110Tail3, scan, if_neg (by decide), root110Candidates3_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail4
    (if 3 ≤ root110Candidates3.length then search edge 3 [617, 1091] root110Candidates3 1708 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates4 : List Nat := [6011, 14669]
set_option maxRecDepth 8192 in
theorem root110Candidates4_checked : candidates edge 4 617 1301 root110Tail5 winner.bound = root110Candidates4 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
theorem root110Step4 : scan edge (search edge 3) 4 [617] 617 root110Tail4 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail5 winner := by
  rw [root110Tail4, scan, if_neg (by decide), root110Candidates4_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail5
    (if 3 ≤ root110Candidates4.length then search edge 3 [617, 1301] root110Candidates4 1918 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates5 : List Nat := [11213, 17321, 19919]
set_option maxRecDepth 8192 in
theorem root110Candidates5_checked : candidates edge 4 617 1427 root110Tail6 winner.bound = root110Candidates5 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root110Child5 : search edge 3 [617, 1427] root110Candidates5 2044 winner = winner := by decide
theorem root110Step5 : scan edge (search edge 3) 4 [617] 617 root110Tail5 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail6 winner := by
  rw [root110Tail5, scan, if_neg (by decide), root110Candidates5_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail6
    (if 3 ≤ root110Candidates5.length then search edge 3 [617, 1427] root110Candidates5 2044 winner else winner) = _
  rw [if_pos (by decide), root110Child5]
def root110Candidates6 : List Nat := [14621]
set_option maxRecDepth 8192 in
theorem root110Candidates6_checked : candidates edge 4 617 2063 root110Tail7 winner.bound = root110Candidates6 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root110Step6 : scan edge (search edge 3) 4 [617] 617 root110Tail6 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail7 winner := by
  rw [root110Tail6, scan, if_neg (by decide), root110Candidates6_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail7
    (if 3 ≤ root110Candidates6.length then search edge 3 [617, 2063] root110Candidates6 2680 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates7 : List Nat := [4391, 8243, 9221, 12653, 18539]
set_option maxRecDepth 8192 in
theorem root110Candidates7_checked : candidates edge 4 617 2657 root110Tail8 winner.bound = root110Candidates7 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root110Child7 : search edge 3 [617, 2657] root110Candidates7 3274 winner = winner := by decide
theorem root110Step7 : scan edge (search edge 3) 4 [617] 617 root110Tail7 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail8 winner := by
  rw [root110Tail7, scan, if_neg (by decide), root110Candidates7_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail8
    (if 3 ≤ root110Candidates7.length then search edge 3 [617, 2657] root110Candidates7 3274 winner else winner) = _
  rw [if_pos (by decide), root110Child7]
def root110Candidates8 : List Nat := [4253, 8747]
set_option maxRecDepth 8192 in
theorem root110Candidates8_checked : candidates edge 4 617 2741 root110Tail9 winner.bound = root110Candidates8 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root110Step8 : scan edge (search edge 3) 4 [617] 617 root110Tail8 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail9 winner := by
  rw [root110Tail8, scan, if_neg (by decide), root110Candidates8_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail9
    (if 3 ≤ root110Candidates8.length then search edge 3 [617, 2741] root110Candidates8 3358 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates9 : List Nat := [4253, 6323, 10889]
set_option maxRecDepth 8192 in
theorem root110Candidates9_checked : candidates edge 4 617 2909 root110Tail10 winner.bound = root110Candidates9 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root110Child9 : search edge 3 [617, 2909] root110Candidates9 3526 winner = winner := by decide
theorem root110Step9 : scan edge (search edge 3) 4 [617] 617 root110Tail9 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail10 winner := by
  rw [root110Tail9, scan, if_neg (by decide), root110Candidates9_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail10
    (if 3 ≤ root110Candidates9.length then search edge 3 [617, 2909] root110Candidates9 3526 winner else winner) = _
  rw [if_pos (by decide), root110Child9]
def root110Candidates10 : List Nat := [4001, 5693, 9473, 14621]
set_option maxRecDepth 8192 in
theorem root110Candidates10_checked : candidates edge 4 617 3593 root110Tail11 winner.bound = root110Candidates10 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root110Child10 : search edge 3 [617, 3593] root110Candidates10 4210 winner = winner := by decide
theorem root110Step10 : scan edge (search edge 3) 4 [617] 617 root110Tail10 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail11 winner := by
  rw [root110Tail10, scan, if_neg (by decide), root110Candidates10_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail11
    (if 3 ≤ root110Candidates10.length then search edge 3 [617, 3593] root110Candidates10 4210 winner else winner) = _
  rw [if_pos (by decide), root110Child10]
def root110Candidates11 : List Nat := [4253, 4391, 6323, 12569, 17609, 19919]
set_option maxRecDepth 8192 in
theorem root110Candidates11_checked : candidates edge 4 617 4001 root110Tail12 winner.bound = root110Candidates11 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root110Child11 : search edge 3 [617, 4001] root110Candidates11 4618 winner = winner := by decide
theorem root110Step11 : scan edge (search edge 3) 4 [617] 617 root110Tail11 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail12 winner := by
  rw [root110Tail11, scan, if_neg (by decide), root110Candidates11_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail12
    (if 3 ≤ root110Candidates11.length then search edge 3 [617, 4001] root110Candidates11 4618 winner else winner) = _
  rw [if_pos (by decide), root110Child11]
def root110Candidates12 : List Nat := [13997, 19709]
set_option maxRecDepth 8192 in
theorem root110Candidates12_checked : candidates edge 4 617 4253 root110Tail13 winner.bound = root110Candidates12 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root110Step12 : scan edge (search edge 3) 4 [617] 617 root110Tail12 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail13 winner := by
  rw [root110Tail12, scan, if_neg (by decide), root110Candidates12_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail13
    (if 3 ≤ root110Candidates12.length then search edge 3 [617, 4253] root110Candidates12 4870 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root110Candidates13_checked : candidates edge 4 617 4391 root110Tail14 winner.bound = root110Candidates13 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root110Step13 : scan edge (search edge 3) 4 [617] 617 root110Tail13 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail14 winner := by
  rw [root110Tail13, scan, if_neg (by decide), root110Candidates13_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail14
    (if 3 ≤ root110Candidates13.length then search edge 3 [617, 4391] root110Candidates13 5008 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates14 : List Nat := [9473, 10973]
set_option maxRecDepth 8192 in
theorem root110Candidates14_checked : candidates edge 4 617 5261 root110Tail15 winner.bound = root110Candidates14 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root110Step14 : scan edge (search edge 3) 4 [617] 617 root110Tail14 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail15 winner := by
  rw [root110Tail14, scan, if_neg (by decide), root110Candidates14_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail15
    (if 3 ≤ root110Candidates14.length then search edge 3 [617, 5261] root110Candidates14 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates15 : List Nat := [16649]
set_option maxRecDepth 8192 in
theorem root110Candidates15_checked : candidates edge 4 617 5333 root110Tail16 winner.bound = root110Candidates15 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root110Step15 : scan edge (search edge 3) 4 [617] 617 root110Tail15 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail16 winner := by
  rw [root110Tail15, scan, if_neg (by decide), root110Candidates15_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail16
    (if 3 ≤ root110Candidates15.length then search edge 3 [617, 5333] root110Candidates15 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root110Candidates16_checked : candidates edge 4 617 5693 root110Tail17 winner.bound = root110Candidates16 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root110Step16 : scan edge (search edge 3) 4 [617] 617 root110Tail16 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail17 winner := by
  rw [root110Tail16, scan, if_neg (by decide), root110Candidates16_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail17
    (if 3 ≤ root110Candidates16.length then search edge 3 [617, 5693] root110Candidates16 6310 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root110Candidates17_checked : candidates edge 4 617 6011 root110Tail18 winner.bound = root110Candidates17 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root110Step17 : scan edge (search edge 3) 4 [617] 617 root110Tail17 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail18 winner := by
  rw [root110Tail17, scan, if_neg (by decide), root110Candidates17_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail18
    (if 3 ≤ root110Candidates17.length then search edge 3 [617, 6011] root110Candidates17 6628 winner else winner) = _
  rw [if_neg (by decide)]
def root110Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root110Candidates18_checked : candidates edge 4 617 6323 root110Tail19 winner.bound = root110Candidates18 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root110Step18 : scan edge (search edge 3) 4 [617] 617 root110Tail18 winner =
    scan edge (search edge 3) 4 [617] 617 root110Tail19 winner := by
  rw [root110Tail18, scan, if_neg (by decide), root110Candidates18_checked]
  change scan edge (search edge 3) 4 [617] 617 root110Tail19
    (if 3 ≤ root110Candidates18.length then search edge 3 [617, 6323] root110Candidates18 6940 winner else winner) = _
  rw [if_neg (by decide)]
theorem root110Scan19 : scan edge (search edge 3) 4 [617] 617 root110Tail19 winner = winner := by rw [root110Tail19, scan, if_pos (by decide)]
theorem root110Scan18 : scan edge (search edge 3) 4 [617] 617 root110Tail18 winner = winner :=
  (root110Step18).trans root110Scan19
theorem root110Scan17 : scan edge (search edge 3) 4 [617] 617 root110Tail17 winner = winner :=
  (root110Step17).trans root110Scan18
theorem root110Scan16 : scan edge (search edge 3) 4 [617] 617 root110Tail16 winner = winner :=
  (root110Step16).trans root110Scan17
theorem root110Scan15 : scan edge (search edge 3) 4 [617] 617 root110Tail15 winner = winner :=
  (root110Step15).trans root110Scan16
theorem root110Scan14 : scan edge (search edge 3) 4 [617] 617 root110Tail14 winner = winner :=
  (root110Step14).trans root110Scan15
theorem root110Scan13 : scan edge (search edge 3) 4 [617] 617 root110Tail13 winner = winner :=
  (root110Step13).trans root110Scan14
theorem root110Scan12 : scan edge (search edge 3) 4 [617] 617 root110Tail12 winner = winner :=
  (root110Step12).trans root110Scan13
theorem root110Scan11 : scan edge (search edge 3) 4 [617] 617 root110Tail11 winner = winner :=
  (root110Step11).trans root110Scan12
theorem root110Scan10 : scan edge (search edge 3) 4 [617] 617 root110Tail10 winner = winner :=
  (root110Step10).trans root110Scan11
theorem root110Scan9 : scan edge (search edge 3) 4 [617] 617 root110Tail9 winner = winner :=
  (root110Step9).trans root110Scan10
theorem root110Scan8 : scan edge (search edge 3) 4 [617] 617 root110Tail8 winner = winner :=
  (root110Step8).trans root110Scan9
theorem root110Scan7 : scan edge (search edge 3) 4 [617] 617 root110Tail7 winner = winner :=
  (root110Step7).trans root110Scan8
theorem root110Scan6 : scan edge (search edge 3) 4 [617] 617 root110Tail6 winner = winner :=
  (root110Step6).trans root110Scan7
theorem root110Scan5 : scan edge (search edge 3) 4 [617] 617 root110Tail5 winner = winner :=
  (root110Step5).trans root110Scan6
theorem root110Scan4 : scan edge (search edge 3) 4 [617] 617 root110Tail4 winner = winner :=
  (root110Step4).trans root110Scan5
theorem root110Scan3 : scan edge (search edge 3) 4 [617] 617 root110Tail3 winner = winner :=
  (root110Step3).trans root110Scan4
theorem root110Scan2 : scan edge (search edge 3) 4 [617] 617 root110Tail2 winner = winner :=
  (root110Step2).trans root110Scan3
theorem root110Scan1 : scan edge (search edge 3) 4 [617] 617 root110Tail1 winner = winner :=
  (root110Step1).trans root110Scan2
theorem root110Scan0 : scan edge (search edge 3) 4 [617] 617 root110Tail0 winner = winner :=
  (root110Step0).trans root110Scan1
set_option maxRecDepth 4096 in
theorem root110Branch : search edge 4 [617] root110Candidates 617 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [617] 617 root110Tail0 winner = winner
  exact root110Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
