import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root77Candidates : List Nat := Data.row409
set_option maxRecDepth 8192 in
theorem root77Candidates_checked : candidates edge 5 0 409 tail78 winner.bound = root77Candidates := by
  apply candidates_eq_row edge 5 0 409 tail78 root77Candidates winner.bound tail78_ordered
  · decide
  · decide
  · funext q
    rfl
def root77Tail68 : List Nat := []
def root77Tail67 : List Nat := 25537 :: root77Tail68
def root77Tail66 : List Nat := 25111 :: root77Tail67
def root77Tail65 : List Nat := 24907 :: root77Tail66
def root77Tail64 : List Nat := 24421 :: root77Tail65
def root77Tail63 : List Nat := 23977 :: root77Tail64
def root77Tail62 : List Nat := 23899 :: root77Tail63
def root77Tail61 : List Nat := 22303 :: root77Tail62
def root77Tail60 : List Nat := 22279 :: root77Tail61
def root77Tail59 : List Nat := 21841 :: root77Tail60
def root77Tail58 : List Nat := 21757 :: root77Tail59
def root77Tail57 : List Nat := 21589 :: root77Tail58
def root77Tail56 : List Nat := 19069 :: root77Tail57
def root77Tail55 : List Nat := 18013 :: root77Tail56
def root77Tail54 : List Nat := 17791 :: root77Tail55
def root77Tail53 : List Nat := 17581 :: root77Tail54
def root77Tail52 : List Nat := 17509 :: root77Tail53
def root77Tail51 : List Nat := 17047 :: root77Tail52
def root77Tail50 : List Nat := 16657 :: root77Tail51
def root77Tail49 : List Nat := 16567 :: root77Tail50
def root77Tail48 : List Nat := 16417 :: root77Tail49
def root77Tail47 : List Nat := 14851 :: root77Tail48
def root77Tail46 : List Nat := 14551 :: root77Tail47
def root77Tail45 : List Nat := 13879 :: root77Tail46
def root77Tail44 : List Nat := 13183 :: root77Tail45
def root77Tail43 : List Nat := 12583 :: root77Tail44
def root77Tail42 : List Nat := 12457 :: root77Tail43
def root77Tail41 : List Nat := 12451 :: root77Tail42
def root77Tail40 : List Nat := 11863 :: root77Tail41
def root77Tail39 : List Nat := 11551 :: root77Tail40
def root77Tail38 : List Nat := 11443 :: root77Tail39
def root77Tail37 : List Nat := 10729 :: root77Tail38
def root77Tail36 : List Nat := 10651 :: root77Tail37
def root77Tail35 : List Nat := 10459 :: root77Tail36
def root77Tail34 : List Nat := 10369 :: root77Tail35
def root77Tail33 : List Nat := 9661 :: root77Tail34
def root77Tail32 : List Nat := 9283 :: root77Tail33
def root77Tail31 : List Nat := 9133 :: root77Tail32
def root77Tail30 : List Nat := 9049 :: root77Tail31
def root77Tail29 : List Nat := 8839 :: root77Tail30
def root77Tail28 : List Nat := 8779 :: root77Tail29
def root77Tail27 : List Nat := 8467 :: root77Tail28
def root77Tail26 : List Nat := 8233 :: root77Tail27
def root77Tail25 : List Nat := 8089 :: root77Tail26
def root77Tail24 : List Nat := 7699 :: root77Tail25
def root77Tail23 : List Nat := 7573 :: root77Tail24
def root77Tail22 : List Nat := 7069 :: root77Tail23
def root77Tail21 : List Nat := 7057 :: root77Tail22
def root77Tail20 : List Nat := 6607 :: root77Tail21
def root77Tail19 : List Nat := 6199 :: root77Tail20
def root77Tail18 : List Nat := 5857 :: root77Tail19
def root77Tail17 : List Nat := 5647 :: root77Tail18
def root77Tail16 : List Nat := 4999 :: root77Tail17
def root77Tail15 : List Nat := 4561 :: root77Tail16
def root77Tail14 : List Nat := 4243 :: root77Tail15
def root77Tail13 : List Nat := 3847 :: root77Tail14
def root77Tail12 : List Nat := 3343 :: root77Tail13
def root77Tail11 : List Nat := 3301 :: root77Tail12
def root77Tail10 : List Nat := 3163 :: root77Tail11
def root77Tail9 : List Nat := 2791 :: root77Tail10
def root77Tail8 : List Nat := 2503 :: root77Tail9
def root77Tail7 : List Nat := 2287 :: root77Tail8
def root77Tail6 : List Nat := 2269 :: root77Tail7
def root77Tail5 : List Nat := 1873 :: root77Tail6
def root77Tail4 : List Nat := 1741 :: root77Tail5
def root77Tail3 : List Nat := 1471 :: root77Tail4
def root77Tail2 : List Nat := 733 :: root77Tail3
def root77Tail1 : List Nat := 709 :: root77Tail2
def root77Tail0 : List Nat := 691 :: root77Tail1
def root77Candidates0 : List Nat := [709, 6199, 17791, 21841]
set_option maxRecDepth 8192 in
theorem root77Candidates0_checked : candidates edge 4 409 691 root77Tail1 winner.bound = root77Candidates0 := by
  unfold candidates
  have row : edge 691 = (SortedNatTree.contains FastRows.row691) := by
    change Data.row691.contains = (SortedNatTree.contains FastRows.row691)
    exact FastRows.row691_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root77Child0 : search edge 3 [409, 691] root77Candidates0 1100 winner = winner := by decide
theorem root77Step0 : scan edge (search edge 3) 4 [409] 409 root77Tail0 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail1 winner := by
  rw [root77Tail0, scan, if_neg (by decide), root77Candidates0_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail1
    (if 3 ≤ root77Candidates0.length then search edge 3 [409, 691] root77Candidates0 1100 winner else winner) = _
  rw [if_pos (by decide), root77Child0]
def root77Candidates1 : List Nat := [1471, 10729, 12451]
set_option maxRecDepth 8192 in
theorem root77Candidates1_checked : candidates edge 4 409 709 root77Tail2 winner.bound = root77Candidates1 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root77Child1 : search edge 3 [409, 709] root77Candidates1 1118 winner = winner := by decide
theorem root77Step1 : scan edge (search edge 3) 4 [409] 409 root77Tail1 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail2 winner := by
  rw [root77Tail1, scan, if_neg (by decide), root77Candidates1_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail2
    (if 3 ≤ root77Candidates1.length then search edge 3 [409, 709] root77Candidates1 1118 winner else winner) = _
  rw [if_pos (by decide), root77Child1]
def root77Candidates2 : List Nat := [21757, 21841]
set_option maxRecDepth 8192 in
theorem root77Candidates2_checked : candidates edge 4 409 733 root77Tail3 winner.bound = root77Candidates2 := by
  unfold candidates
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row]
  decide
theorem root77Step2 : scan edge (search edge 3) 4 [409] 409 root77Tail2 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail3 winner := by
  rw [root77Tail2, scan, if_neg (by decide), root77Candidates2_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail3
    (if 3 ≤ root77Candidates2.length then search edge 3 [409, 733] root77Candidates2 1142 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates3 : List Nat := [4561, 7069, 8779, 21841]
set_option maxRecDepth 8192 in
theorem root77Candidates3_checked : candidates edge 4 409 1471 root77Tail4 winner.bound = root77Candidates3 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root77Child3 : search edge 3 [409, 1471] root77Candidates3 1880 winner = winner := by decide
theorem root77Step3 : scan edge (search edge 3) 4 [409] 409 root77Tail3 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail4 winner := by
  rw [root77Tail3, scan, if_neg (by decide), root77Candidates3_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail4
    (if 3 ≤ root77Candidates3.length then search edge 3 [409, 1471] root77Candidates3 1880 winner else winner) = _
  rw [if_pos (by decide), root77Child3]
def root77Candidates4 : List Nat := [2287, 7573]
set_option maxRecDepth 8192 in
theorem root77Candidates4_checked : candidates edge 4 409 1741 root77Tail5 winner.bound = root77Candidates4 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
theorem root77Step4 : scan edge (search edge 3) 4 [409] 409 root77Tail4 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail5 winner := by
  rw [root77Tail4, scan, if_neg (by decide), root77Candidates4_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail5
    (if 3 ≤ root77Candidates4.length then search edge 3 [409, 1741] root77Candidates4 2150 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates5 : List Nat := [2269, 14851]
set_option maxRecDepth 8192 in
theorem root77Candidates5_checked : candidates edge 4 409 1873 root77Tail6 winner.bound = root77Candidates5 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root77Step5 : scan edge (search edge 3) 4 [409] 409 root77Tail5 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail6 winner := by
  rw [root77Tail5, scan, if_neg (by decide), root77Candidates5_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail6
    (if 3 ≤ root77Candidates5.length then search edge 3 [409, 1873] root77Candidates5 2282 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates6 : List Nat := [8779]
set_option maxRecDepth 8192 in
theorem root77Candidates6_checked : candidates edge 4 409 2269 root77Tail7 winner.bound = root77Candidates6 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
theorem root77Step6 : scan edge (search edge 3) 4 [409] 409 root77Tail6 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail7 winner := by
  rw [root77Tail6, scan, if_neg (by decide), root77Candidates6_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail7
    (if 3 ≤ root77Candidates6.length then search edge 3 [409, 2269] root77Candidates6 2678 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates7 : List Nat := [5857, 9049, 18013, 21841]
set_option maxRecDepth 8192 in
theorem root77Candidates7_checked : candidates edge 4 409 2287 root77Tail8 winner.bound = root77Candidates7 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root77Child7 : search edge 3 [409, 2287] root77Candidates7 2696 winner = winner := by decide
theorem root77Step7 : scan edge (search edge 3) 4 [409] 409 root77Tail7 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail8 winner := by
  rw [root77Tail7, scan, if_neg (by decide), root77Candidates7_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail8
    (if 3 ≤ root77Candidates7.length then search edge 3 [409, 2287] root77Candidates7 2696 winner else winner) = _
  rw [if_pos (by decide), root77Child7]
def root77Candidates8 : List Nat := [17581]
set_option maxRecDepth 8192 in
theorem root77Candidates8_checked : candidates edge 4 409 2503 root77Tail9 winner.bound = root77Candidates8 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
theorem root77Step8 : scan edge (search edge 3) 4 [409] 409 root77Tail8 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail9 winner := by
  rw [root77Tail8, scan, if_neg (by decide), root77Candidates8_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail9
    (if 3 ≤ root77Candidates8.length then search edge 3 [409, 2503] root77Candidates8 2912 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates9 : List Nat := [6607]
set_option maxRecDepth 8192 in
theorem root77Candidates9_checked : candidates edge 4 409 2791 root77Tail10 winner.bound = root77Candidates9 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root77Step9 : scan edge (search edge 3) 4 [409] 409 root77Tail9 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail10 winner := by
  rw [root77Tail9, scan, if_neg (by decide), root77Candidates9_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail10
    (if 3 ≤ root77Candidates9.length then search edge 3 [409, 2791] root77Candidates9 3200 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates10 : List Nat := [14851]
set_option maxRecDepth 8192 in
theorem root77Candidates10_checked : candidates edge 4 409 3163 root77Tail11 winner.bound = root77Candidates10 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root77Step10 : scan edge (search edge 3) 4 [409] 409 root77Tail10 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail11 winner := by
  rw [root77Tail10, scan, if_neg (by decide), root77Candidates10_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail11
    (if 3 ≤ root77Candidates10.length then search edge 3 [409, 3163] root77Candidates10 3572 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates11 : List Nat := [13879]
set_option maxRecDepth 8192 in
theorem root77Candidates11_checked : candidates edge 4 409 3301 root77Tail12 winner.bound = root77Candidates11 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root77Step11 : scan edge (search edge 3) 4 [409] 409 root77Tail11 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail12 winner := by
  rw [root77Tail11, scan, if_neg (by decide), root77Candidates11_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail12
    (if 3 ≤ root77Candidates11.length then search edge 3 [409, 3301] root77Candidates11 3710 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates12 : List Nat := [4561, 8467, 17791]
set_option maxRecDepth 8192 in
theorem root77Candidates12_checked : candidates edge 4 409 3343 root77Tail13 winner.bound = root77Candidates12 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root77Child12 : search edge 3 [409, 3343] root77Candidates12 3752 winner = winner := by decide
theorem root77Step12 : scan edge (search edge 3) 4 [409] 409 root77Tail12 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail13 winner := by
  rw [root77Tail12, scan, if_neg (by decide), root77Candidates12_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail13
    (if 3 ≤ root77Candidates12.length then search edge 3 [409, 3343] root77Candidates12 3752 winner else winner) = _
  rw [if_pos (by decide), root77Child12]
def root77Candidates13 : List Nat := [8233, 11863, 12451, 13183, 16567]
set_option maxRecDepth 8192 in
theorem root77Candidates13_checked : candidates edge 4 409 3847 root77Tail14 winner.bound = root77Candidates13 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root77Child13 : search edge 3 [409, 3847] root77Candidates13 4256 winner = winner := by decide
theorem root77Step13 : scan edge (search edge 3) 4 [409] 409 root77Tail13 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail14 winner := by
  rw [root77Tail13, scan, if_neg (by decide), root77Candidates13_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail14
    (if 3 ≤ root77Candidates13.length then search edge 3 [409, 3847] root77Candidates13 4256 winner else winner) = _
  rw [if_pos (by decide), root77Child13]
def root77Candidates14 : List Nat := [7699, 12457]
set_option maxRecDepth 8192 in
theorem root77Candidates14_checked : candidates edge 4 409 4243 root77Tail15 winner.bound = root77Candidates14 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root77Step14 : scan edge (search edge 3) 4 [409] 409 root77Tail14 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail15 winner := by
  rw [root77Tail14, scan, if_neg (by decide), root77Candidates14_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail15
    (if 3 ≤ root77Candidates14.length then search edge 3 [409, 4243] root77Candidates14 4652 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates15 : List Nat := [8779, 9283]
set_option maxRecDepth 8192 in
theorem root77Candidates15_checked : candidates edge 4 409 4561 root77Tail16 winner.bound = root77Candidates15 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root77Step15 : scan edge (search edge 3) 4 [409] 409 root77Tail15 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail16 winner := by
  rw [root77Tail15, scan, if_neg (by decide), root77Candidates15_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail16
    (if 3 ≤ root77Candidates15.length then search edge 3 [409, 4561] root77Candidates15 4970 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates16 : List Nat := [7069, 8779]
set_option maxRecDepth 8192 in
theorem root77Candidates16_checked : candidates edge 4 409 4999 root77Tail17 winner.bound = root77Candidates16 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root77Step16 : scan edge (search edge 3) 4 [409] 409 root77Tail16 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail17 winner := by
  rw [root77Tail16, scan, if_neg (by decide), root77Candidates16_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail17
    (if 3 ≤ root77Candidates16.length then search edge 3 [409, 4999] root77Candidates16 5408 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates17 : List Nat := [17047]
set_option maxRecDepth 8192 in
theorem root77Candidates17_checked : candidates edge 4 409 5647 root77Tail18 winner.bound = root77Candidates17 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root77Step17 : scan edge (search edge 3) 4 [409] 409 root77Tail17 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail18 winner := by
  rw [root77Tail17, scan, if_neg (by decide), root77Candidates17_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail18
    (if 3 ≤ root77Candidates17.length then search edge 3 [409, 5647] root77Candidates17 6056 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root77Candidates18_checked : candidates edge 4 409 5857 root77Tail19 winner.bound = root77Candidates18 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root77Step18 : scan edge (search edge 3) 4 [409] 409 root77Tail18 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail19 winner := by
  rw [root77Tail18, scan, if_neg (by decide), root77Candidates18_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail19
    (if 3 ≤ root77Candidates18.length then search edge 3 [409, 5857] root77Candidates18 6266 winner else winner) = _
  rw [if_neg (by decide)]
def root77Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root77Candidates19_checked : candidates edge 4 409 6199 root77Tail20 winner.bound = root77Candidates19 := by
  unfold candidates
  have row : edge 6199 = (SortedNatTree.contains FastRows.row6199) := by
    change Data.row6199.contains = (SortedNatTree.contains FastRows.row6199)
    exact FastRows.row6199_checked.symm
  rw [row]
  decide
theorem root77Step19 : scan edge (search edge 3) 4 [409] 409 root77Tail19 winner =
    scan edge (search edge 3) 4 [409] 409 root77Tail20 winner := by
  rw [root77Tail19, scan, if_neg (by decide), root77Candidates19_checked]
  change scan edge (search edge 3) 4 [409] 409 root77Tail20
    (if 3 ≤ root77Candidates19.length then search edge 3 [409, 6199] root77Candidates19 6608 winner else winner) = _
  rw [if_neg (by decide)]
theorem root77Scan20 : scan edge (search edge 3) 4 [409] 409 root77Tail20 winner = winner := by rw [root77Tail20, scan, if_pos (by decide)]
theorem root77Scan19 : scan edge (search edge 3) 4 [409] 409 root77Tail19 winner = winner :=
  (root77Step19).trans root77Scan20
theorem root77Scan18 : scan edge (search edge 3) 4 [409] 409 root77Tail18 winner = winner :=
  (root77Step18).trans root77Scan19
theorem root77Scan17 : scan edge (search edge 3) 4 [409] 409 root77Tail17 winner = winner :=
  (root77Step17).trans root77Scan18
theorem root77Scan16 : scan edge (search edge 3) 4 [409] 409 root77Tail16 winner = winner :=
  (root77Step16).trans root77Scan17
theorem root77Scan15 : scan edge (search edge 3) 4 [409] 409 root77Tail15 winner = winner :=
  (root77Step15).trans root77Scan16
theorem root77Scan14 : scan edge (search edge 3) 4 [409] 409 root77Tail14 winner = winner :=
  (root77Step14).trans root77Scan15
theorem root77Scan13 : scan edge (search edge 3) 4 [409] 409 root77Tail13 winner = winner :=
  (root77Step13).trans root77Scan14
theorem root77Scan12 : scan edge (search edge 3) 4 [409] 409 root77Tail12 winner = winner :=
  (root77Step12).trans root77Scan13
theorem root77Scan11 : scan edge (search edge 3) 4 [409] 409 root77Tail11 winner = winner :=
  (root77Step11).trans root77Scan12
theorem root77Scan10 : scan edge (search edge 3) 4 [409] 409 root77Tail10 winner = winner :=
  (root77Step10).trans root77Scan11
theorem root77Scan9 : scan edge (search edge 3) 4 [409] 409 root77Tail9 winner = winner :=
  (root77Step9).trans root77Scan10
theorem root77Scan8 : scan edge (search edge 3) 4 [409] 409 root77Tail8 winner = winner :=
  (root77Step8).trans root77Scan9
theorem root77Scan7 : scan edge (search edge 3) 4 [409] 409 root77Tail7 winner = winner :=
  (root77Step7).trans root77Scan8
theorem root77Scan6 : scan edge (search edge 3) 4 [409] 409 root77Tail6 winner = winner :=
  (root77Step6).trans root77Scan7
theorem root77Scan5 : scan edge (search edge 3) 4 [409] 409 root77Tail5 winner = winner :=
  (root77Step5).trans root77Scan6
theorem root77Scan4 : scan edge (search edge 3) 4 [409] 409 root77Tail4 winner = winner :=
  (root77Step4).trans root77Scan5
theorem root77Scan3 : scan edge (search edge 3) 4 [409] 409 root77Tail3 winner = winner :=
  (root77Step3).trans root77Scan4
theorem root77Scan2 : scan edge (search edge 3) 4 [409] 409 root77Tail2 winner = winner :=
  (root77Step2).trans root77Scan3
theorem root77Scan1 : scan edge (search edge 3) 4 [409] 409 root77Tail1 winner = winner :=
  (root77Step1).trans root77Scan2
theorem root77Scan0 : scan edge (search edge 3) 4 [409] 409 root77Tail0 winner = winner :=
  (root77Step0).trans root77Scan1
set_option maxRecDepth 4096 in
theorem root77Branch : search edge 4 [409] root77Candidates 409 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [409] 409 root77Tail0 winner = winner
  exact root77Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
